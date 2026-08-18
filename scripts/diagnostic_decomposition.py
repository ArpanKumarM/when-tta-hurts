#!/usr/bin/env python3
"""Phase 2A scientific audit (Part D): fixed diagnostic decomposition.

EXPLORATORY DIAGNOSIS ONLY -- NOT a confirmatory experiment, NOT policy
selection. Uses ONLY the existing saved pilot checkpoint, the validation
split, the existing TTA seed (271828), mean-probability aggregation, and
augmented-views-only, to evaluate single-transform and policy-subset
conditions at N={1,10,50}. Does not train, does not touch the test split,
does not modify the registered pilot's metrics.json/predictions.npz, and
does not change the frozen augmentation policy, thresholds, architecture,
or hyperparameters. Its purpose is purely diagnostic: does one transform
(or an implementation bug) explain the pilot's extreme TTA degradation?

Usage:
    uv run python scripts/diagnostic_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.artifacts import atomic_write_json, hash_state_dict
from when_tta_hurts.config import load_config
from when_tta_hurts.data import load_pilot_split
from when_tta_hurts.evaluation.tta import aggregate_mean_prefix, compute_ordered_view_logits
from when_tta_hurts.metrics import compute_all_metrics, harm_rescue_rates
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.transforms.policies import _geometric_ops, _intensity_ops, build_policy

PILOT_ART_DIR = Path("artifacts/pilots/pilot-pathmnist-28-bn-8f4a5024")
DIAGNOSTIC_OUT_PATH = Path("artifacts/audits/pilot-pathmnist-28-bn-8f4a5024/diagnostic_decomposition.json")

VIEW_COUNTS = [1, 10, 50]
MAX_VIEWS = 50


def build_conditions() -> dict[str, torch.nn.Module]:
    geo = _geometric_ops()
    inten = _intensity_ops()
    return {
        "horizontal_flip_only": torch.nn.Sequential(geo[0]),
        "vertical_flip_only": torch.nn.Sequential(geo[1]),
        "rotation_only": torch.nn.Sequential(geo[2]),
        "random_resized_crop_only": torch.nn.Sequential(geo[3]),
        "color_jitter_only": torch.nn.Sequential(inten[0]),
        "gaussian_blur_only": torch.nn.Sequential(inten[1]),
        "geometric_policy": build_policy("geometric"),
        "intensity_policy": build_policy("intensity"),
        "mixed_policy_existing": build_policy("mixed"),
    }


def main() -> int:
    cfg = load_config(str(PILOT_ART_DIR / "resolved_config.json"))
    run_summary = load_config(str(PILOT_ART_DIR / "run_summary.json"))
    preds = np.load(PILOT_ART_DIR / "predictions.npz")
    clean_logits = preds["clean_val_logits"]
    labels = preds["val_labels"]
    clean_metrics = compute_all_metrics(clean_logits, labels)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    state = torch.load(PILOT_ART_DIR / "best_checkpoint.pt", weights_only=True)
    assert hash_state_dict(state) == run_summary["best_checkpoint_hash"]
    model.load_state_dict(state)
    model.eval()

    val_ds = load_pilot_split("pathmnist", split="val", size=cfg["dataset"]["resolution"])
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    tta_seed = cfg["tta_seed"]  # 271828, unchanged -- existing TTA seed only
    conditions = build_conditions()

    results: dict[str, dict] = {}
    for cond_name, policy in conditions.items():
        print(f"--- condition: {cond_name} ---")
        all_ordered = []
        for x, _y in val_loader:
            x = x.to(device)
            view_logits = compute_ordered_view_logits(
                model, x, policy, tta_seed=tta_seed, max_views=MAX_VIEWS, device=device
            )
            all_ordered.append(view_logits)
        ordered = np.concatenate(all_ordered, axis=1)  # [MAX_VIEWS, N, C]

        cond_results = {}
        for n in VIEW_COUNTS:
            agg = aggregate_mean_prefix(ordered, n_views=n)
            m = compute_all_metrics(agg, labels)
            hr = harm_rescue_rates(clean_logits, agg, labels)
            m["delta_accuracy"] = m["accuracy"] - clean_metrics["accuracy"]
            m.update(hr)
            cond_results[str(n)] = m
            print(
                f"  N={n}: acc={m['accuracy']:.4f} delta={m['delta_accuracy']:+.4f} harm={m['harm_rate']:.4f}"
            )
        results[cond_name] = cond_results

    output = {
        "note": (
            "EXPLORATORY DIAGNOSIS ONLY -- not a confirmatory experiment, not "
            "policy selection. Uses the existing pilot checkpoint, TTA seed "
            "271828, validation split, mean-probability aggregation, "
            "augmented-views-only. Frozen pilot artifacts were NOT modified."
        ),
        "pilot_artifact_dir": str(PILOT_ART_DIR),
        "checkpoint_hash": run_summary["best_checkpoint_hash"],
        "tta_seed": tta_seed,
        "clean_metrics": clean_metrics,
        "conditions": results,
    }
    atomic_write_json(output, DIAGNOSTIC_OUT_PATH)
    print(f"\nDiagnostic decomposition written to {DIAGNOSTIC_OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
