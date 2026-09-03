#!/usr/bin/env python3
"""Phase 2C — Secondary-Analysis Expansion (read-only extraction).

Surfaces the preregistered secondary/descriptive analyses that the sealed
final-test pipeline already computed but that paper/manuscript.md does not
report (it reports only the frozen primary endpoint: naive mean_probability
TTA at N=50). See docs/phase2c_secondary_analysis_expansion_plan.md.

Reads ONLY artifacts/final_test/ (canonical, authorized completions) and
artifacts/final_test_scientific_summary.json (integrity gate). Writes ONLY
artifacts/secondary_analysis_expansion/. Never trains, never touches the
test-set loader, never modifies a sealed artifact or a docs/ freeze.

Usage:
    uv run python scripts/expand_secondary_analysis.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from when_tta_hurts.artifacts import atomic_write_json, hash_file  # noqa: E402
from when_tta_hurts.statistical_analysis import (  # noqa: E402
    effect_sizes,
    mcnemar_test,
    paired_bootstrap_ci,
)
from when_tta_hurts.final_test_statistical_analysis import (  # noqa: E402
    resolve_final_test_canonical_evaluation_identity,
    verify_final_test_authorization,
)
from when_tta_hurts.statistical_analysis import derive_family_cells  # noqa: E402

FINAL_TEST_ROOT = REPO / "artifacts" / "final_test"
SEALED_SUMMARY = REPO / "artifacts" / "final_test_scientific_summary.json"
OUT_DIR = REPO / "artifacts" / "secondary_analysis_expansion"
PLAN_DOC = REPO / "docs" / "phase2c_secondary_analysis_expansion_plan.md"

VIEW_COUNTS = [1, 2, 5, 10, 25, 50, 100]
N_RESAMPLES = 10_000
CI_LEVEL = 0.95
FAMILIES = ("H1", "H2", "H3", "BLOCK_C")


def bootstrap_seed(analysis: str, run_id: str, n: int) -> int:
    """Deterministic uint64 seed, mirroring the project's hash-derived
    seed discipline (see cross_condition_addendum.derive_bootstrap_seed)."""
    digest = hashlib.sha256(f"phase2c|{analysis}|{run_id}|n{n}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def canonical_attempt_dir(run_id: str, auth) -> Path:
    identity = resolve_final_test_canonical_evaluation_identity(run_id, auth)
    if identity.get("evaluation_status") != "eligible":
        raise SystemExit(
            f"Cell {run_id!r} not eligible (status={identity.get('evaluation_status')!r}); aborting."
        )
    return FINAL_TEST_ROOT / run_id / f"attempt_{identity['evaluation_attempt']:03d}"


def load_predictions(attempt_dir: Path) -> dict[str, np.ndarray]:
    return dict(np.load(attempt_dir / "predictions.npz"))


def paired_delta(clean_correct: np.ndarray, cond_correct: np.ndarray, analysis: str, run_id: str, n: int):
    rng = np.random.default_rng(bootstrap_seed(analysis, run_id, n))
    boot = paired_bootstrap_ci(clean_correct, cond_correct, n_resamples=N_RESAMPLES, ci_level=CI_LEVEL, rng=rng)
    mc = mcnemar_test(clean_correct, cond_correct)
    es = effect_sizes(clean_correct, cond_correct)
    return {
        "delta_accuracy": boot["delta_accuracy"],
        "ci_low": boot["ci_low"],
        "ci_high": boot["ci_high"],
        "ci_excludes_zero": bool(boot["ci_low"] > 0 or boot["ci_high"] < 0),
        "mcnemar_p": mc["p_value"],
        "harm_rate": es["harm_rate"],
        "rescue_rate": es["rescue_rate"],
        "n_samples": int(clean_correct.size),
        "bootstrap_seed": bootstrap_seed(analysis, run_id, n),
    }


def main() -> int:
    if not SEALED_SUMMARY.exists():
        raise SystemExit(f"Missing sealed summary: {SEALED_SUMMARY}")
    sealed = json.loads(SEALED_SUMMARY.read_text())

    # sealed per-cell N=50 delta accuracy, for the integrity gate
    sealed_delta_n50: dict[str, float] = {}
    for fam in ("H1", "H2", "H3", "BLOCK_C"):
        for cell in sealed["preregistered"][fam]["cells"]:
            sealed_delta_n50.setdefault(cell["run_id"], cell["bootstrap"]["delta_accuracy"])

    auth = verify_final_test_authorization()

    # distinct canonical cells + which families/policy each belongs to
    cell_families: dict[str, set[str]] = {}
    for fam in FAMILIES:
        for fc in derive_family_cells(fam):
            cell_families.setdefault(fc.run_id, set()).add(fam)
    run_ids = sorted(cell_families)
    print(f"{len(run_ids)} distinct canonical cells")

    scaling_rows: list[dict] = []
    bn_rows: list[dict] = []
    anchored_rows: list[dict] = []
    agg_rows: list[dict] = []
    calib_rows: list[dict] = []
    input_hashes: dict[str, str] = {}
    integrity_failures: list[str] = []

    for run_id in run_ids:
        attempt_dir = canonical_attempt_dir(run_id, auth)
        preds = load_predictions(attempt_dir)
        input_hashes[str((attempt_dir / "predictions.npz").relative_to(REPO))] = hash_file(
            attempt_dir / "predictions.npz"
        )
        metrics = json.loads((attempt_dir / "metrics.json").read_text())

        labels = preds["labels"]
        clean_probs = preds["clean_probs"]
        view_probs = preds["view_probs"]  # (100, N, C)
        clean_correct = clean_probs.argmax(-1) == labels
        policy = "matched_mixed" if "matched_mixed" in run_id else "none"
        dataset = run_id.split("-")[1]

        # --- 1. scaling curve (naive mean_probability) ---
        for n in VIEW_COUNTS:
            agg = view_probs[:n].mean(axis=0)
            cond_correct = agg.argmax(-1) == labels
            rec = paired_delta(clean_correct, cond_correct, "scaling_curve", run_id, n)
            if n == 50:
                recomputed = rec["delta_accuracy"]
                expected = sealed_delta_n50.get(run_id)
                if expected is not None and abs(recomputed - expected) > 1e-12:
                    integrity_failures.append(
                        f"{run_id}: N=50 delta {recomputed!r} != sealed {expected!r}"
                    )
            scaling_rows.append(
                {"run_id": run_id, "dataset": dataset, "policy": policy,
                 "families": "|".join(sorted(cell_families[run_id])), "n_views": n, **rec}
            )

        # --- 2. BatchNorm-statistics adaptation (BatchNorm cells only) ---
        if "bn_adapted_probs" in preds:
            bn_seq = list(preds["bn_adapted_prefix_sequence"])
            for i, n in enumerate(bn_seq):
                bn_correct = preds["bn_adapted_probs"][i].argmax(-1) == labels
                naive_correct = view_probs[:n].mean(axis=0).argmax(-1) == labels
                vs_clean = paired_delta(clean_correct, bn_correct, "bn_vs_clean", run_id, int(n))
                rng = np.random.default_rng(bootstrap_seed("bn_vs_naive", run_id, int(n)))
                vs_naive = paired_bootstrap_ci(
                    naive_correct, bn_correct, n_resamples=N_RESAMPLES, ci_level=CI_LEVEL, rng=rng
                )
                bn_rows.append(
                    {"run_id": run_id, "dataset": dataset, "n_views": int(n),
                     "bn_delta_vs_clean": vs_clean["delta_accuracy"],
                     "bn_vs_clean_ci_low": vs_clean["ci_low"], "bn_vs_clean_ci_high": vs_clean["ci_high"],
                     "bn_vs_clean_excludes_zero": vs_clean["ci_excludes_zero"],
                     "bn_delta_vs_naive": vs_naive["delta_accuracy"],
                     "bn_vs_naive_ci_low": vs_naive["ci_low"], "bn_vs_naive_ci_high": vs_naive["ci_high"],
                     "bn_vs_naive_excludes_zero": bool(vs_naive["ci_low"] > 0 or vs_naive["ci_high"] < 0)}
                )

        # --- 3/4/5. point estimates from the canonical metrics.json ---
        conds = metrics["conditions"]
        clean_m = metrics["clean"]
        for n in VIEW_COUNTS:
            sn = str(n)
            if conds.get("original_anchored_tta") and sn in conds["original_anchored_tta"]:
                a = conds["original_anchored_tta"][sn]
                anchored_rows.append(
                    {"run_id": run_id, "dataset": dataset, "n_views": n,
                     "anchored_accuracy": a["accuracy"], "anchored_delta_accuracy": a["delta_accuracy"],
                     "anchored_harm_rate": a["harm_rate"], "anchored_rescue_rate": a["rescue_rate"]}
                )
            row = {"run_id": run_id, "dataset": dataset, "n_views": n}
            naive = conds.get("naive_tta") or {}
            for aggname in ("mean_probability", "majority_vote", "confidence_weighted_average"):
                if (naive.get(aggname) or {}).get(sn):
                    row[f"{aggname}_delta"] = naive[aggname][sn]["delta_accuracy"]
            agg_rows.append(row)

        def cal(d):
            return {"ece": d.get("expected_calibration_error"), "nll": d.get("negative_log_likelihood"),
                    "brier": d.get("brier_score")}

        crow = {"run_id": run_id, "dataset": dataset,
                **{f"clean_{k}": v for k, v in cal(clean_m).items()},
                **{f"naive50_{k}": v for k, v in cal(conds["naive_tta"]["mean_probability"]["50"]).items()}}
        if conds.get("original_anchored_tta") and "50" in conds["original_anchored_tta"]:
            crow.update({f"anchored50_{k}": v for k, v in cal(conds["original_anchored_tta"]["50"]).items()})
        if conds.get("bn_adapted_tta") and "50" in conds["bn_adapted_tta"]:
            crow.update({f"bn50_{k}": v for k, v in cal(conds["bn_adapted_tta"]["50"]).items()})
        calib_rows.append(crow)

    if integrity_failures:
        print("INTEGRITY GATE FAILED:")
        for f in integrity_failures:
            print("  " + f)
        return 1
    print("integrity gate passed: all N=50 deltas match the sealed summary")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)

    def write_csv(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        keys = list({k for r in rows for k in r})
        # stable column order: identifiers first
        lead = [k for k in ("run_id", "dataset", "policy", "families", "n_views") if k in keys]
        cols = lead + [k for k in keys if k not in lead]
        with (OUT_DIR / "tables" / name).open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv("scaling_curve.csv", scaling_rows)
    write_csv("bn_adaptation.csv", bn_rows)
    write_csv("anchored.csv", anchored_rows)
    write_csv("aggregation_ablation.csv", agg_rows)
    write_csv("calibration.csv", calib_rows)

    # ---- compact machine-readable roll-up ----
    def curve_summary(rows: list[dict], predicate) -> dict:
        out: dict = {}
        for n in VIEW_COUNTS:
            sub = [r for r in rows if r["n_views"] == n and predicate(r)]
            deltas = np.array([r["delta_accuracy"] for r in sub])
            out[str(n)] = {
                "n_cells": len(sub),
                "mean_delta_accuracy_pp": float(deltas.mean() * 100),
                "min_delta_accuracy_pp": float(deltas.min() * 100),
                "max_delta_accuracy_pp": float(deltas.max() * 100),
                "n_cells_delta_negative": int((deltas < 0).sum()),
                "n_cells_ci_excludes_zero_negative": int(
                    sum(1 for r in sub if r["ci_excludes_zero"] and r["delta_accuracy"] < 0)
                ),
            }
        return out

    summary = {
        "phase": "2C",
        "plan_doc_sha256": hash_file(PLAN_DOC),
        "sealed_summary_sha256": hash_file(SEALED_SUMMARY),
        "sealed_summary_reporting_fingerprint": sealed["reporting_fingerprint"],
        "n_resamples": N_RESAMPLES,
        "ci_level": CI_LEVEL,
        "view_counts": VIEW_COUNTS,
        "n_distinct_cells": len(run_ids),
        "integrity_gate": "passed",
        "scaling_curve_headline30_by_n": curve_summary(
            scaling_rows, lambda r: r["policy"] == "none" and "BLOCK_C" not in r["families"]
        ),
        "scaling_curve_block_c_by_n": curve_summary(
            scaling_rows, lambda r: "BLOCK_C" in r["families"]
        ),
        "scaling_curve_matched_policy_by_n": curve_summary(
            scaling_rows, lambda r: r["policy"] == "matched_mixed"
        ),
        "scaling_curve_rows": scaling_rows,
        "bn_adaptation_rows": bn_rows,
        "anchored_rows": anchored_rows,
        "aggregation_ablation_rows": agg_rows,
        "calibration_rows": calib_rows,
    }
    atomic_write_json(summary, OUT_DIR / "summary.json")
    atomic_write_json(
        {"input_predictions_sha256": input_hashes, "plan_doc_sha256": hash_file(PLAN_DOC),
         "sealed_summary_sha256": hash_file(SEALED_SUMMARY)},
        OUT_DIR / "manifest.json",
    )

    # ---- console digest ----
    print("\nView-count scaling curve (unmatched cells, naive mean_probability):")
    print(f"  {'N':>4}  {'mean Δacc pp':>13}  {'min':>8}  {'max':>8}  {'<0':>5}  {'CI<0':>6}")
    cs = summary["scaling_curve_headline30_by_n"]
    for n in VIEW_COUNTS:
        d = cs[str(n)]
        print(f"  {n:>4}  {d['mean_delta_accuracy_pp']:>13.2f}  {d['min_delta_accuracy_pp']:>8.2f}  "
              f"{d['max_delta_accuracy_pp']:>8.2f}  {d['n_cells_delta_negative']:>3}/{d['n_cells']:<2}  "
              f"{d['n_cells_ci_excludes_zero_negative']:>6}")
    print(f"\nwrote {OUT_DIR}/summary.json (+ 5 CSV tables, manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
