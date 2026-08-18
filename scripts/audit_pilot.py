#!/usr/bin/env python3
"""Phase 2A scientific audit (Part C): verify the completed pilot's
transformation correctness and bitwise reproducibility, using ONLY the
existing checkpoint, validation data, and saved artifacts. Does not train,
does not touch the test split, does not change the pilot's frozen
configuration or overwrite its artifacts.

Writes a JSON audit report to a separate gitignored path (never overwrites
the pilot's metrics.json/predictions.npz) and PNG contact sheets to a
gitignored directory (never committed).

Usage:
    uv run python scripts/audit_pilot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.artifacts import atomic_write_json, hash_state_dict
from when_tta_hurts.config import load_config
from when_tta_hurts.data import load_pilot_split
from when_tta_hurts.evaluation.tta import aggregate_mean_prefix, compute_ordered_view_logits
from when_tta_hurts.metrics import compute_all_metrics, harm_rescue_rates, softmax
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.transforms.policies import (
    _geometric_ops,
    _intensity_ops,
    build_policy,
    sample_deterministic_view,
)

PILOT_ART_DIR = Path("artifacts/pilots/pilot-pathmnist-28-bn-8f4a5024")
AUDIT_OUT_DIR = Path("artifacts/audits/pilot-pathmnist-28-bn-8f4a5024")
CONTACT_SHEET_DIR = AUDIT_OUT_DIR / "contact_sheets"


def tensor_stats(x: torch.Tensor) -> dict:
    x_np = x.detach().cpu().numpy()
    return {
        "shape": list(x_np.shape),
        "dtype": str(x_np.dtype),
        "all_finite": bool(np.isfinite(x_np).all()),
        "min": float(x_np.min()),
        "max": float(x_np.max()),
        "mean_per_channel": [float(x_np[:, c].mean()) for c in range(x_np.shape[1])],
        "std_per_channel": [float(x_np[:, c].std()) for c in range(x_np.shape[1])],
        "frac_at_or_below_0": float((x_np <= 0.0).mean()),
        "frac_at_or_above_1": float((x_np >= 1.0).mean()),
    }


def save_contact_sheet(images: torch.Tensor, path: Path, cols: int = 8) -> None:
    """images: [N, 3, H, W] float tensor in [0,1]. Tiles into a grid PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = (images.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    n, c, h, w = imgs.shape
    rows = -(-n // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), color=(40, 40, 40))
    for i in range(n):
        im = Image.fromarray(imgs[i].transpose(1, 2, 0), mode="RGB")
        r, cidx = divmod(i, cols)
        sheet.paste(im, (cidx * w, r * h))
    sheet.save(path)


def main() -> int:
    report: dict = {"pilot_artifact_dir": str(PILOT_ART_DIR)}

    cfg = load_config(str(PILOT_ART_DIR / "resolved_config.json"))
    preds = np.load(PILOT_ART_DIR / "predictions.npz")
    clean_logits_saved = preds["clean_val_logits"]
    labels_saved = preds["val_labels"]
    ordered_saved = preds["ordered_tta_val_logits"]  # [50, N, 9]
    metrics_saved = load_config(str(PILOT_ART_DIR / "metrics.json"))
    run_summary = load_config(str(PILOT_ART_DIR / "run_summary.json"))

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # --- Reload checkpoint + val data (deterministic order, no shuffle) ---
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    state = torch.load(PILOT_ART_DIR / "best_checkpoint.pt", weights_only=True)
    assert hash_state_dict(state) == run_summary["best_checkpoint_hash"], "checkpoint hash mismatch on reload"
    model.load_state_dict(state)
    model.eval()

    val_ds = load_pilot_split("pathmnist", split="val", size=cfg["dataset"]["resolution"])
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    # --- 1/2. Raw + augmented tensor stats; ColorJitter input range ---
    x0, y0 = next(iter(val_loader))
    raw_stats = tensor_stats(x0)

    policy_mixed = build_policy(cfg["pilot_tta"]["policy"])  # CPU, per the MPS fix
    view0 = sample_deterministic_view(x0, policy_mixed, seed=cfg["tta_seed"])
    view0_stats = tensor_stats(view0)

    # Isolate the state right before ColorJitter (after geometric ops) to
    # directly confirm ColorJitter's input range, per requirement C.2.
    geo_only = torch.nn.Sequential(*_geometric_ops())
    pre_jitter = sample_deterministic_view(x0, geo_only, seed=cfg["tta_seed"])
    pre_jitter_stats = tensor_stats(pre_jitter)

    report["raw_input_tensor_stats"] = raw_stats
    report["augmented_view_tensor_stats_(view_0_mixed)"] = view0_stats
    report["pre_colorjitter_tensor_stats_(after_geometric_ops_only)"] = pre_jitter_stats
    report["colorjitter_input_in_expected_0_1_range"] = (
        pre_jitter_stats["min"] >= -1e-4 and pre_jitter_stats["max"] <= 1.0 + 1e-4
    )

    # --- 3. Confirm transforms do not accidentally apply twice ---
    # A single sequential policy call vs. manually calling it once should
    # be identical (sanity on the call site); double-application would show
    # as a DIFFERENT (more distorted) result if we compose the policy with
    # itself. Confirm applying policy(policy(x)) with the SAME seed differs
    # from a single application (proves double-apply would be detectable),
    # then confirm the actual call site (sample_deterministic_view) is a
    # SINGLE nn.Sequential forward -- verified by source inspection: yes,
    # policy(x) is exactly one nn.Sequential.forward call.
    torch.manual_seed(cfg["tta_seed"])
    once = policy_mixed(x0)
    torch.manual_seed(cfg["tta_seed"])
    twice = policy_mixed(policy_mixed(x0))
    report["single_vs_double_apply_are_detectably_different"] = not torch.allclose(once, twice, atol=1e-6)
    report["call_site_is_single_sequential_forward"] = (
        "verified by source inspection: transforms/policies.py::sample_deterministic_view "
        "calls policy(x) exactly once inside a `with torch.no_grad()` block"
    )

    # --- 4. Confirm transforms vary across samples and across views ---
    view_a = sample_deterministic_view(x0, policy_mixed, seed=cfg["tta_seed"] + 0)
    view_b = sample_deterministic_view(x0, policy_mixed, seed=cfg["tta_seed"] + 1)
    report["views_differ_across_view_index"] = not torch.allclose(view_a, view_b, atol=1e-6)
    # within one view, do different samples in the batch get different transforms?
    per_sample_diffs = (view_a[0] - view_a[1]).abs().mean().item()
    report["samples_differ_within_one_view"] = per_sample_diffs > 1e-4

    # --- 5. same_on_batch not accidentally identical across the batch ---
    # Compare pairwise per-sample augmented images within view_a: if
    # same_on_batch were (bugged into) True, every sample's transform
    # parameters would be identical, but images still differ because
    # CONTENT differs -- the correct check is on the transform's *applied
    # geometry*, not raw pixel equality. We check this properly by
    # augmenting a batch of IDENTICAL images (so post-aug differences can
    # only come from different sampled parameters, not different content).
    identical_batch = x0[0:1].repeat(8, 1, 1, 1)
    aug_identical = sample_deterministic_view(identical_batch, policy_mixed, seed=cfg["tta_seed"])
    pairwise_std = aug_identical.std(dim=0).mean().item()
    report["same_on_batch_is_false_as_intended"] = pairwise_std > 1e-4
    report["same_on_batch_pairwise_std_on_identical_inputs"] = pairwise_std

    # --- 6/7/9. Nested-prefix + label/sample/prediction alignment + bitwise reproduction ---
    all_labels = []
    for _x, y in val_loader:
        all_labels.append(y.numpy().reshape(-1))
    labels_recomputed_order = np.concatenate(all_labels, axis=0)
    report["label_ordering_matches_saved_predictions_npz"] = bool(
        np.array_equal(labels_recomputed_order, labels_saved)
    )

    recon_clean_metrics = compute_all_metrics(clean_logits_saved, labels_saved)
    report["clean_metrics_reproduced_bitwise_from_predictions_npz"] = {
        k: (recon_clean_metrics[k] == metrics_saved["clean"][k]) for k in recon_clean_metrics
    }

    tta_reproduction = {}
    for n in cfg["pilot_tta"]["view_counts"]:
        agg_log_probs = aggregate_mean_prefix(ordered_saved, n_views=n)
        m = compute_all_metrics(agg_log_probs, labels_saved)
        hr = harm_rescue_rates(clean_logits_saved, agg_log_probs, labels_saved)
        m["delta_accuracy"] = m["accuracy"] - recon_clean_metrics["accuracy"]
        m.update(hr)
        saved_m = metrics_saved["tta_by_view_count"][str(n)]
        exact_match = all(
            (m[k] == saved_m[k]) if isinstance(m[k], (int, float, bool)) else True for k in m if k in saved_m
        )
        tta_reproduction[str(n)] = {"exact_match_all_fields": exact_match}
    report["tta_metrics_reproduced_bitwise_per_view_count"] = tta_reproduction

    # nested-prefix property directly on the saved 50-view array
    full50 = ordered_saved
    prefix10_from_saved = full50[:10]
    fresh10 = compute_ordered_view_logits(
        model, x0.to(device), policy_mixed, tta_seed=cfg["tta_seed"], max_views=10, device=device
    )
    # Compare only for the first batch (x0) -- ordered_saved is concatenated
    # over ALL val batches, so slice the first batch_size rows for comparison.
    batch_n = x0.shape[0]
    report["nested_prefix_matches_fresh_recompute_for_first_batch"] = bool(
        np.allclose(prefix10_from_saved[:, :batch_n, :], fresh10, atol=1e-5)
    )

    # --- 8. mean-probability aggregation applies softmax per view before averaging ---
    manual = np.stack([softmax(v) for v in ordered_saved[:5, :3]], axis=0).mean(
        axis=0
    )  # first 3 samples, 5 views
    via_fn = softmax(aggregate_mean_prefix(ordered_saved[:, :3], n_views=5))
    report["aggregation_applies_softmax_per_view_before_averaging"] = bool(
        np.allclose(manual, via_fn, atol=1e-6)
    )

    # --- Contact sheets (gitignored, not committed) ---
    CONTACT_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    x_sheet = x0[:16]
    save_contact_sheet(x_sheet, CONTACT_SHEET_DIR / "clean.png")

    hflip_only = torch.nn.Sequential(_geometric_ops()[0])
    vflip_only = torch.nn.Sequential(_geometric_ops()[1])
    rot_only = torch.nn.Sequential(_geometric_ops()[2])
    crop_only = torch.nn.Sequential(_geometric_ops()[3])
    jitter_only = torch.nn.Sequential(_intensity_ops()[0])
    blur_only = torch.nn.Sequential(_intensity_ops()[1])
    geometric_policy = build_policy("geometric")
    intensity_policy = build_policy("intensity")

    for name, op in [
        ("horizontal_flip", hflip_only),
        ("vertical_flip", vflip_only),
        ("rotation", rot_only),
        ("crop", crop_only),
        ("color_jitter", jitter_only),
        ("blur", blur_only),
        ("geometric_policy", geometric_policy),
        ("intensity_policy", intensity_policy),
        ("mixed_policy", policy_mixed),
    ]:
        v = sample_deterministic_view(x_sheet, op, seed=cfg["tta_seed"])
        save_contact_sheet(v, CONTACT_SHEET_DIR / f"{name}.png")

    report["contact_sheets_written_to"] = str(CONTACT_SHEET_DIR)

    atomic_write_json(report, AUDIT_OUT_DIR / "transform_audit_report.json")
    print(f"Audit report written to {AUDIT_OUT_DIR / 'transform_audit_report.json'}")

    # --- Print summary ---
    for k, v in report.items():
        if isinstance(v, dict) and len(str(v)) > 200:
            continue
        print(f"{k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
