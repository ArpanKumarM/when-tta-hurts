# Phase 2C — Secondary-Analysis Expansion Plan

**Status: plan + read-only extraction. No new training, no new
test-split evaluation, no change to any sealed artifact, the frozen
manuscript claims freeze, or `configs/experiment_matrix.yaml`.**

This phase responds to `paper/reviews/` (Reviewer 1 §5, Reviewer 3 §12)
and `docs/phase2b_submission_gap_analysis.md` Tier-1 items #3 and #4,
which observe that several **preregistered secondary/descriptive
analyses** listed in `docs/phase2b_protocol.md` §3 were computed by the
sealed final-test evaluation pipeline but never surfaced in
`paper/manuscript.md` (which reports only the frozen primary endpoint:
naive `mean_probability` TTA at N=50).

## What this phase does

Read the already-sealed, already-authorized final-test artifacts under
`artifacts/final_test/` and extract / compute the following, writing
**only** to the new directory `artifacts/secondary_analysis_expansion/`:

| # | Analysis | Source | Evidentiary status |
|---|----------|--------|--------------------|
| 1 | **View-count scaling curve** — naive `mean_probability` delta accuracy at N ∈ {1,2,5,10,25,50,100}, per cell, with paired-bootstrap 95% CI + McNemar, all 39 canonical cells | recomputed from `predictions.npz::view_probs` prefixes | **Preregistered secondary/descriptive** (`phase2b_protocol.md` §3.1) |
| 2 | **BatchNorm-statistics adaptation** — `bn_adapted_tta` delta vs clean and vs naive, per N, paired-bootstrap 95% CI, BatchNorm cells only | recomputed from `predictions.npz::bn_adapted_probs` | **Preregistered secondary baseline** (`phase2b_protocol.md` §3.5; source-paper Appendix B — NOT a project contribution) |
| 3 | **Original-anchored condition** — delta vs clean per N | point estimates read from per-cell `metrics.json` | **Preregistered secondary baseline** (`phase2b_protocol.md` §3.4; source-paper Appendix B — NOT a project contribution) |
| 4 | **Aggregation ablation** — `mean_probability` vs `majority_vote` vs `confidence_weighted_average` delta accuracy, per N | point estimates read from per-cell `metrics.json` | **Preregistered secondary** (`phase2b_protocol.md` §3.3) |
| 5 | **Calibration table** — ECE / NLL / multiclass Brier for `clean`, naive N=50, anchored N=50, bn-adapted N=50 | point estimates read from per-cell `metrics.json` | **Preregistered secondary endpoint** (`experimental_protocol.md`); descriptive, non-inferential |

## What this phase explicitly does NOT do

* Does **not** run the geometric-only / intensity-only augmentation-
  strategy ablation (`phase2b_protocol.md` §3.2). Those policies were
  never evaluated on the test split; adding them requires a fresh
  evaluation run over existing checkpoints and its own authorization
  step. Tracked as a separate follow-up (Phase 2C.2).
* Does **not** modify `artifacts/final_test_scientific_summary.json`,
  `artifacts/paper_evidence/`, `paper/manuscript.md`, or any `docs/`
  freeze document.
* Does **not** introduce any new inferential test, pooled estimate,
  significance threshold, or model-population claim. Per-cell
  paired-bootstrap CIs and McNemar p-values use the **frozen**
  primitives in `src/when_tta_hurts/statistical_analysis.py`
  (`paired_bootstrap_ci`, `mcnemar_test`, `effect_sizes`), 10,000
  resamples, ci_level 0.95, with a deterministic per-(analysis, run_id,
  N) bootstrap seed.
* Does **not** re-label any preregistered/confirmatory result. The
  frozen primary endpoint (N=50) is unchanged; the scaling curve
  *contains* that point and must reproduce it bit-for-bit as an
  integrity check (the script aborts on any mismatch against the sealed
  summary).

## Integrity gate

Before writing any output, the script recomputes every cell's N=50
`mean_probability` delta accuracy and asserts exact equality with the
corresponding value in `artifacts/final_test_scientific_summary.json`.
Any mismatch is a hard failure.

## Outputs

```
artifacts/secondary_analysis_expansion/
  summary.json                     # all five analyses, machine-readable
  manifest.json                    # sha256 of every input predictions.npz + this plan
  tables/
    scaling_curve.csv
    bn_adaptation.csv
    anchored.csv
    aggregation_ablation.csv
    calibration.csv
```

Findings are written to `docs/phase2c_secondary_analysis_findings.md`
(narrative, with every number traceable to `summary.json`).
