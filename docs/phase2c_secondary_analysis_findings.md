# Phase 2C — Secondary-Analysis Expansion: Findings

**Status: descriptive report of preregistered secondary/descriptive
analyses. No confirmatory claim, no protocol change, no sealed artifact
modified.** Every number below traces to
`artifacts/secondary_analysis_expansion/summary.json` (and the CSV
tables beside it), produced by `scripts/expand_secondary_analysis.py`
reading only the sealed, authorized final-test artifacts. The N=50
`mean_probability` delta accuracy for every one of the 39 canonical
cells was recomputed and matched bit-for-bit against
`artifacts/final_test_scientific_summary.json` before any output was
written (integrity gate: **passed**).

See `docs/phase2c_secondary_analysis_expansion_plan.md` for scope and
the evidentiary-tier table.

---

## 1. View-count scaling curve (preregistered secondary, `phase2b_protocol.md` §3.1)

Naive `mean_probability` TTA, delta accuracy (TTA − clean) in
percentage points, per-cell paired-bootstrap 95% CI (10,000 resamples).

**Headline 30 unmatched cells (Block A + Block D):**

| N | mean Δ (pp) | min | max | cells with Δ<0 | CIs excluding 0 (negative) |
|---|---|---|---|---|---|
| 1 | −49.41 | −64.10 | −29.58 | 30/30 | 30/30 |
| 2 | −46.77 | −65.42 | −27.76 | 30/30 | 30/30 |
| 5 | −44.83 | −64.25 | −25.37 | 30/30 | 30/30 |
| 10 | −43.58 | −64.40 | −22.54 | 30/30 | 30/30 |
| 25 | −42.89 | −65.57 | −20.57 | 30/30 | 30/30 |
| 50 | −42.56 | −66.09 | −18.76 | 30/30 | 30/30 |
| 100 | −42.42 | −66.79 | −18.37 | 30/30 | 30/30 |

**Interpretation.** Harm is present and large at *every* view count
from 1 to 100, in all 30 cells, with every 95% CI excluding zero on the
negative side. Harm magnitude is *largest at N=1* (a single random
augmented view, −49.4 pp mean), decreases monotonically as views are
added, and has essentially flattened by N≈25 (−42.9 → −42.4 pp from
N=25 to N=100). The frozen primary condition N=50 (−42.6 pp) sits on
this asymptote. **The curve never crosses zero and never reverses
direction.** This directly answers the reviewer question (Reviewer 1
§5, Reviewer 3 §12) of whether N=50 is an unrepresentative or
cherry-picked operating point: it is not — it is near the best case for
naive TTA across the whole registered range, and the harm at N=50 is
if anything slightly *smaller* than at every N < 50.

Split by dataset (30-cell set): PathMNIST −47.95 pp (N=1) → −40.32 pp
(N=100); BloodMNIST −50.86 pp (N=1) → −44.51 pp (N=100). Same shape in
both.

**BLOCK_C (DermaMNIST / ResNet-18, 3 cells)** shows the same *shape* at
low N — −6.47 pp mean at N=1, all three CIs excluding zero — decaying
toward zero by N=50 (−0.86 pp mean, 1/3 CIs excluding zero). BLOCK_C's
near-null result at the frozen N=50 is therefore itself view-count
dependent: at N=1 even the positive-control cells are harmed.

**Matched-policy cells (Block B, 6 cells)** are mildly negative at N=1
(−1.63 pp mean, 4/6 CIs excluding zero) and turn slightly positive and
stable from N=5 onward (+1.5 pp mean at N=50/100). Training-time policy
matching approximately neutralizes the harm across the full curve, not
only at N=50.

## 2. BatchNorm-statistics adaptation (preregistered secondary baseline, §3.5; source-paper Appendix B — not a project contribution)

`bn_adapted_tta` at N=50, BatchNorm cells only (27 cells), paired-
bootstrap 95% CI:

| dataset | Δ vs clean (mean pp) | Δ vs naive-TTA (mean pp) | cells where BN-adapt beats naive (CI>0) |
|---|---|---|---|
| BloodMNIST | −10.26 (range −19.6 … +2.3) | +27.19 | 9/12 |
| PathMNIST | −15.90 (range −36.6 … −1.1) | +10.78 | 8/12 |
| DermaMNIST | +0.71 (range +0.4 … +0.9) | +1.58 | 1/3 |

**Interpretation.** Adapting BatchNorm running statistics to the
augmented-batch distribution *reduces* naive-TTA harm substantially
(BloodMNIST +27 pp, PathMNIST +11 pp relative to naive) but does **not
rescue accuracy**: net of clean, BN-adapted TTA is still −10 to −16 pp
on the two SmallCNN datasets. If BatchNorm's reliance on stale running
statistics were the *whole* mechanism, BN adaptation would close the
gap; it does not. This is consistent with the source study's Appendix B
(BN adaptation helped BloodMNIST, hurt PathMNIST) in direction on
BloodMNIST, and is a mechanism-relevant negative result: **BatchNorm
running statistics are part, but not all, of the story.**

## 3. Original-anchored condition (preregistered secondary baseline, §3.4; source-paper Appendix B — not a project contribution)

One clean view + N augmented views, equal-weight mean, N=50, 39 cells:
mean Δ vs clean **−31.43 pp** (range −64.1 … +4.1), negative in 32/39
cells (point estimates from the canonical `metrics.json`).

**Interpretation.** Clean-image anchoring provides only a small
improvement over unanchored naive TTA (≈ −38 pp pooled / −42.6 pp on
the 30-cell set). The source study reported clean anchoring cutting
SmallCNN/PathMNIST's drop from −37.0 pp to −8.6 pp; **we do not
reproduce a rescue of that magnitude.** This is a second source-paper
Appendix-B mitigation that does not transfer to our implementation
(alongside the BLOCK_C positive-control non-reproduction already in the
manuscript).

## 4. Aggregation ablation (preregistered secondary, §3.3)

Delta accuracy at N=25, 39 cells, point estimates from `metrics.json`:

| aggregator | mean Δ (pp) | range |
|---|---|---|
| mean probability | −32.86 | −65.6 … +3.8 |
| majority vote | −33.60 | −66.3 … +3.8 |
| confidence-weighted average | −33.01 | −67.1 … +3.9 |

**Interpretation.** The three aggregation rules are within ~0.7 pp of
each other on average. The harm is **not an artifact of mean-probability
pooling specifically** — it survives majority vote and confidence
weighting essentially unchanged. (This is a robustness check, not a
claim that a better aggregator cannot help; learned aggregators à la
Shanmugam et al. 2021 were not evaluated.)

## 5. Calibration (preregistered secondary endpoint; descriptive, non-inferential)

Mean over cells, N=50 conditions (point estimates from `metrics.json`):

| metric | clean | naive TTA | anchored | BN-adapted |
|---|---|---|---|---|
| ECE | 0.056 | 0.119 | 0.117 | 0.117 |
| NLL | 0.486 | 1.260 | 1.179 | 0.925 |
| multiclass Brier | 0.216 | 0.570 | 0.556 | 0.416 |

**Interpretation.** Naive TTA roughly doubles ECE and increases NLL and
Brier ~2.5×: the harm is a **calibration** degradation as well as an
accuracy degradation. BN-statistics adaptation recovers a large part of
the NLL/Brier inflation (NLL 1.26 → 0.93, Brier 0.57 → 0.42) while
leaving ECE essentially unchanged — i.e. it sharpens the probability
estimates without fixing confidence–accuracy alignment. This connects
the study to the TTA-uncertainty prior art surfaced in the novelty
review (Ayhan & Berens 2018; BayTTA).

---

## What this closes, and what it does not

**Closes (from `paper/reviews/` + `phase2b_submission_gap_analysis.md`):**

* Gap-analysis Tier-1 #3 (view-count scaling curve) — done, and the
  result *strengthens* the headline: harm is stable and CI-separated
  from zero across the entire 1–100 range, worst at N=1.
* Gap-analysis Tier-1 #4 (BatchNorm adaptation comparison) — done;
  yields a mechanism-relevant negative result.
* Adds an aggregation-invariance robustness check and a calibration
  characterisation, both requested by Reviewer 3 §12.
* Surfaces a second source-paper mitigation non-reproduction
  (clean anchoring), reported plainly.

**Does NOT close:**

* **Label-preservation audit (Reviewer 1 §4 — the single most damaging
  open question).** Still requires a small new frozen protocol +
  augmented-view annotation. Nothing here rules out "the mixed policy
  destroys diagnostic content on a meaningful fraction of views."
  Tracked as Phase 2C.2.
* **Per-augmentation-component ablation (geometric-only vs.
  intensity-only).** Those policies were never evaluated on the test
  split; needs a fresh evaluation run over existing checkpoints (no
  retraining) plus its own authorization step. Tracked as Phase 2C.2.
* Cross-architecture coverage of the ablation axes (Reviewer 1 §1) —
  out of scope for an analysis-only phase; would require new training.

## Suggested manuscript use

1. Add the scaling curve as a new figure + one Results paragraph
   ("Harm is stable across view counts"), replacing the implicit
   "N=50 only" framing Reviewer 1 §5 objected to.
2. Fold BN-adaptation, anchoring, aggregation-invariance, and
   calibration into a single new "Secondary conditions" subsection,
   each explicitly labeled preregistered-secondary/descriptive.
3. State the two additional non-reproductions (anchoring, partial
   BN-adaptation) in Discussion alongside the existing BLOCK_C
   non-reproduction — it reinforces the paper's implementation-
   sensitivity point rather than weakening it.
