#!/usr/bin/env python3
"""Phase 2C.2 — Per-Augmentation-Component Ablation.

Implements docs/phase2c2_component_ablation_addendum.md. Validation split
only; reuses the 12 Block A 28px confirmatory checkpoints; no training,
no test-split access. Compares geometric-only and intensity-only TTA
delta accuracy against the already-computed mixed-policy validation delta.

Usage:
    uv run python scripts/component_ablation.py            # full run (12 cells)
    uv run python scripts/component_ablation.py --smoke    # 1 cell, 3 views, _smoke dir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from when_tta_hurts.artifacts import hash_state_dict  # noqa: E402
from when_tta_hurts.devices import select_device  # noqa: E402
from when_tta_hurts.evaluation.validation_loader import load_validation_evaluation_split  # noqa: E402
from when_tta_hurts.evaluation.views import generate_single_view  # noqa: E402
from when_tta_hurts.models.small_cnn import build_small_cnn  # noqa: E402
from when_tta_hurts.statistical_analysis import effect_sizes, mcnemar_test, paired_bootstrap_ci  # noqa: E402
from when_tta_hurts.transforms.policies import build_policy  # noqa: E402

ADDENDUM = REPO / "docs" / "phase2c2_component_ablation_addendum.md"
VAL_CFG = REPO / "configs" / "validation_evaluation.yaml"
CONF_A = REPO / "artifacts" / "confirmatory" / "A"
VAL_EVAL = REPO / "artifacts" / "validation_evaluation"
OUT = REPO / "artifacts" / "component_ablation"

NUM_CLASSES = {"pathmnist": 9, "bloodmnist": 8}
POLICIES = ("geometric", "intensity")
VIEW_COUNTS = (25, 50)
MAX_VIEWS = 50
N_RESAMPLES = 10_000
FWD_BATCH = 1024

CELLS = [
    f"A-{ds}-28px-{norm}-policy-none-s{s}"
    for ds in ("pathmnist", "bloodmnist")
    for norm in ("batchnorm", "groupnorm")
    for s in (0, 1, 2)
]

# frozen sec.7 cutoffs
DOMINANCE_RATIO = 2.0
GEO_SMALL_PP = 10.0
ADDITIVE_PP = 5.0


def tta_seed() -> int:
    return int(yaml.safe_load(VAL_CFG.read_text())["confirmatory_tta_seed"])


def seed_int(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def latest_attempt_with(path: Path, *needed: str) -> Path:
    cands = sorted(
        (d for d in path.glob("attempt_*") if all((d / n).exists() for n in needed)),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not cands:
        raise SystemExit(f"no attempt under {path} with {needed}")
    return cands[-1]


def resolve_cell(run_id: str) -> dict:
    ds = run_id.split("-")[1]
    norm = "batchnorm" if "batchnorm" in run_id else "groupnorm"
    val_att = latest_attempt_with(VAL_EVAL / run_id, "predictions.npz", "metadata.json")
    val_meta = json.loads((val_att / "metadata.json").read_text())
    expected_hash = val_meta["checkpoint_hash"]
    ckpt_att = latest_attempt_with(CONF_A / run_id, "best_checkpoint.pt")
    ckpt_path = ckpt_att / "best_checkpoint.pt"
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actual = hash_state_dict(state)
    if actual != expected_hash:
        # try every attempt dir for a matching checkpoint
        match = None
        for d in sorted((CONF_A / run_id).glob("attempt_*")):
            p = d / "best_checkpoint.pt"
            if p.exists() and hash_state_dict(torch.load(p, map_location="cpu", weights_only=True)) == expected_hash:
                match = p
                break
        if match is None:
            raise SystemExit(f"{run_id}: no confirmatory checkpoint matches validation checkpoint_hash {expected_hash}")
        ckpt_path = match
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return {"run_id": run_id, "dataset": ds, "norm": norm,
            "ckpt_path": ckpt_path, "state": state, "expected_hash": expected_hash,
            "val_predictions": val_att / "predictions.npz"}


def prob_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    p = np.clip(probs, 1e-12, 1.0)
    conf = p.max(-1)
    pred = p.argmax(-1)
    correct = (pred == labels).astype(float)
    nll = float(-np.log(p[np.arange(len(labels)), labels]).mean())
    onehot = np.zeros_like(p)
    onehot[np.arange(len(labels)), labels] = 1.0
    brier = float(((p - onehot) ** 2).sum(-1).mean())
    edges = np.linspace(0, 1, 16)
    ece = 0.0
    for i in range(15):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {"ece": float(ece), "nll": nll, "brier": brier}


@torch.no_grad()
def forward_probs(model, x: torch.Tensor, device) -> np.ndarray:
    out = []
    for i in range(0, x.shape[0], FWD_BATCH):
        logits = model(x[i : i + FWD_BATCH].to(device))
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


def run(smoke: bool) -> int:
    SEED = tta_seed()
    out_dir = OUT.parent / "component_ablation_smoke" if smoke else OUT
    cells = CELLS[:1] if smoke else CELLS
    max_views = 3 if smoke else MAX_VIEWS
    view_counts = (3,) if smoke else VIEW_COUNTS
    device = select_device("auto")
    print(f"device={device}  cells={len(cells)}  views={max_views}  seed={SEED}")

    per_cell_rows: list[dict] = []
    manifest_inputs: dict[str, str] = {}
    cache_dir = out_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(run_id: str, policy: str) -> Path:
        return cache_dir / f"{run_id}__{policy}.npz"

    def load_cached(run_id: str, policy: str, n_samples: int) -> dict[int, np.ndarray] | None:
        p = cache_path(run_id, policy)
        if not p.exists():
            return None
        try:
            z = np.load(p)
            if int(z["seed"]) != SEED or int(z["max_views"]) != max_views or int(z["n_samples"]) != n_samples:
                return None
            return {int(n): z[f"mean{n}"] for n in view_counts}
        except Exception:
            return None

    for ci, run_id in enumerate(cells):
        c = resolve_cell(run_id)
        ds, norm = c["dataset"], c["norm"]
        manifest_inputs[str(c["ckpt_path"].relative_to(REPO))] = c["expected_hash"]

        split = load_validation_evaluation_split(ds, 28)
        images, labels, sidx = split.images, split.labels, split.sample_indices

        vp = np.load(c["val_predictions"])
        assert np.array_equal(vp["labels"], labels), f"{run_id}: label mismatch vs validation predictions"
        clean_probs = vp["clean_probs"]
        clean_correct = clean_probs.argmax(-1) == labels
        mixed_views = vp["view_probs"]  # (100, N, C)

        model = None
        for policy in POLICIES:
            snaps = load_cached(run_id, policy, len(labels))
            if snaps is not None:
                print(f"  [{ci+1}/{len(cells)}] {run_id} {policy}: cached", flush=True)
            else:
                if model is None:
                    model = build_small_cnn(NUM_CLASSES[ds], normalization=norm).to(device).eval()
                    model.load_state_dict(c["state"], strict=True)
                t0 = time.time()
                pol = build_policy(policy, (28, 28))
                psum = np.zeros_like(clean_probs, dtype=np.float64)
                snaps = {}
                for v in range(max_views):
                    view = generate_single_view(images, pol, SEED, ds, 28, sidx, v)
                    psum += forward_probs(model, view, device)
                    if (v + 1) in view_counts:
                        snaps[v + 1] = (psum / (v + 1)).astype(np.float32)
                    if (v + 1) % 10 == 0 or v + 1 == max_views:
                        print(f"  [{ci+1}/{len(cells)}] {run_id} {policy}: view {v+1}/{max_views}  "
                              f"({time.time()-t0:.0f}s)", flush=True)
                tmp = cache_dir / f"{run_id}__{policy}.tmp.npz"
                np.savez(tmp, seed=SEED, max_views=max_views, n_samples=len(labels),
                         **{f"mean{n}": a for n, a in snaps.items()})
                tmp.replace(cache_path(run_id, policy))

            for n, agg in snaps.items():
                cc = agg.argmax(-1) == labels
                rng = np.random.default_rng(seed_int(f"phase2c2_component|{run_id}|{policy}|n{n}"))
                boot = paired_bootstrap_ci(clean_correct, cc, n_resamples=N_RESAMPLES, rng=rng)
                mc = mcnemar_test(clean_correct, cc)
                es = effect_sizes(clean_correct, cc)
                mix_agg = mixed_views[:n].mean(0)
                mix_delta = float((mix_agg.argmax(-1) == labels).mean() - clean_correct.mean())
                pm = prob_metrics(agg, labels)
                per_cell_rows.append({
                    "run_id": run_id, "dataset": ds, "norm": norm, "policy": policy, "n_views": n,
                    "delta_accuracy": boot["delta_accuracy"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                    "ci_excludes_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
                    "mcnemar_p": mc["p_value"], "harm_rate": es["harm_rate"], "rescue_rate": es["rescue_rate"],
                    "mixed_delta_accuracy": mix_delta,
                    "ece": pm["ece"], "nll": pm["nll"], "brier": pm["brier"],
                    "n_samples": int(len(labels)),
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_cell.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_cell_rows[0].keys()))
        w.writeheader()
        w.writerows(per_cell_rows)

    # pooled + verdict
    def pool(n: int) -> dict:
        rows_n = [r for r in per_cell_rows if r["n_views"] == n]
        geo = np.array([r["delta_accuracy"] for r in rows_n if r["policy"] == "geometric"]) * 100
        inten = np.array([r["delta_accuracy"] for r in rows_n if r["policy"] == "intensity"]) * 100
        mixed = np.array([r["mixed_delta_accuracy"] for r in rows_n if r["policy"] == "geometric"]) * 100
        resid = mixed - (geo + inten)
        mg, mi = float(geo.mean()), float(inten.mean())
        if abs(mi) >= DOMINANCE_RATIO * abs(mg) and abs(mg) < GEO_SMALL_PP:
            verdict = "intensity_dominated"
        elif abs(mg) >= DOMINANCE_RATIO * abs(mi):
            verdict = "geometry_dominated"
        else:
            verdict = "both_contribute"
        add = ("approximately_additive" if abs(resid.mean()) < ADDITIVE_PP
               else "super_additive" if resid.mean() < 0 else "sub_additive")
        return {
            "n_cells": len(geo),
            "geometric_delta_pp": {"mean": mg, "min": float(geo.min()), "max": float(geo.max())},
            "intensity_delta_pp": {"mean": mi, "min": float(inten.min()), "max": float(inten.max())},
            "mixed_delta_pp": {"mean": float(mixed.mean()), "min": float(mixed.min()), "max": float(mixed.max())},
            "additivity_residual_pp": {"mean": float(resid.mean()), "min": float(resid.min()), "max": float(resid.max())},
            "verdict": verdict, "additivity": add,
        }

    summary = {
        "phase": "2C.2-component-ablation",
        "smoke": smoke,
        "addendum_sha256": hashlib.sha256(ADDENDUM.read_bytes()).hexdigest(),
        "confirmatory_tta_seed": SEED,
        "n_resamples": N_RESAMPLES,
        "frozen_cutoffs": {"dominance_ratio": DOMINANCE_RATIO, "geo_small_pp": GEO_SMALL_PP, "additive_pp": ADDITIVE_PP},
        "pooled_by_n": {str(n): pool(n) for n in view_counts},
        "per_cell_rows": per_cell_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps({
        "addendum_sha256": summary["addendum_sha256"],
        "confirmatory_tta_seed": SEED,
        "input_checkpoints_sha256": manifest_inputs,
    }, indent=2, sort_keys=True) + "\n")

    print("\n=== pooled ===")
    for n in view_counts:
        p = summary["pooled_by_n"][str(n)]
        print(f"N={n}: geometric {p['geometric_delta_pp']['mean']:+.2f}pp  "
              f"intensity {p['intensity_delta_pp']['mean']:+.2f}pp  mixed {p['mixed_delta_pp']['mean']:+.2f}pp  "
              f"resid {p['additivity_residual_pp']['mean']:+.2f}pp  -> {p['verdict']} / {p['additivity']}")
    print(f"wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    raise SystemExit(run(args.smoke))
