# Phase 2B.5C — Pre-Test Fixed-Model Difference-in-Differences Addendum (FROZEN)

**Status: FROZEN.** This document and its machine-readable companion,
`configs/final_test_cross_condition_addendum.yaml`, together specify the
*entire* secondary cross-condition analysis authorized by the Phase 2B.5B
adjudication memo. Nothing in this addendum may be changed after the
official test split is opened without a new, explicitly flagged amendment.

## 1. What this addendum is, and is not

* **It is:** a secondary, fixed-trained-model, image-paired
  difference-in-differences (DiD) analysis, computed once the official
  test split is opened, using only the primary already-frozen TTA
  endpoint (accuracy, N=50, mean-probability aggregation).
* **It is not:** a replacement for, or reinterpretation of, the originally
  preregistered within-cell clean-vs-TTA bootstrap/McNemar analyses
  (`docs/statistical_analysis_plan.md`). Those remain the sole
  originally-preregistered primary confirmatory test-set analyses.
* **It is not:** a test of H1/H2/H3 "in general" (i.e., BatchNorm vs
  GroupNorm as methods, or resolution as a general causal lever, or
  matched-policy training as a general technique). Per the Phase 2B.5B
  memo, no design available to this project (3 seeds per condition)
  supports that inferential target — see Part C of that memo for the
  combinatorial proof (exact sign/permutation tests cannot fall below
  p=0.25 at n=3, regardless of true effect size).
* **It is not** an authorization to access the test split. This task
  (Phase 2B.5C) freezes and engineers the addendum against synthetic data
  only; execution against real test-set predictions requires separate,
  future authorization.

## 2. Provenance — validation results were already observed

This addendum is being frozen on top of commit
`8d4c07e39718dbdd1bbf0ce2008594a1a65306f5` ("feat: implement frozen Phase
2B statistical analysis runner"), itself built on top of:

* `99b20be088278f21f2ddbf72007d63a12e3062c1` — "results: close Phase 2B
  evaluator-fingerprint reconciliation" (all 39 confirmatory
  validation-stage cells closed and accepted as complete/uniform).
* `0ba6f7236f9db2e19968581085fac3fa4eafdd67` — "results: record Phase 2B
  Block D validation evaluation" (Block D's 5 remaining cells recorded).

At the time this addendum is written, **all 39 planned validation-stage
cells have completed and their descriptive validation-stage outcomes have
already been visible in this conversation and in the committed ledger**.
This addendum is therefore, and must always be described as:

> **post-validation, pre-test-specified** — not originally preregistered.

It was frozen and committed *before* any test-split access, so its
specification cannot have been tuned on test-set outcomes (the only
firewall `docs/statistical_analysis_plan.md`'s "test firewall enforcement"
section actually requires). It was, however, specified *after*
validation-stage descriptive results were seen, which is a materially
weaker guarantee than true preregistration, and every reporting surface
(paper, README, results tables) must carry this distinction explicitly —
see §7 below.

## 3. Primary endpoint (no new endpoint introduced)

Per the Phase 2B.5B decision and `configs/final_test_cross_condition_addendum.yaml`'s
`endpoint:` block, this addendum uses **only**:

* metric: accuracy
* condition: `naive_tta`
* TTA view count: **N=50**
* aggregator: **mean_probability**
* degradation definition: `delta = accuracy_TTA50 - accuracy_clean`

No other N, aggregator, metric, or harm/rescue threshold is introduced.
Calibration metrics, alternative aggregators, BN adaptation, N-curves,
harm/rescue rates, and Block C remain governed by their existing
classifications (secondary/descriptive, per
`docs/phase2b_statistical_analysis_engineering_freeze.md`).

## 4. The fixed-pair difference-in-differences estimand

For a fixed pair of trained models (condition A, condition B) evaluated on
the same aligned set of test images, define per-image correctness
indicators `C_A,i`, `T_A,i`, `C_B,i`, `T_B,i` (clean/TTA-correct for A/B on
image i). The per-image DiD contribution is:

```
d_i = (T_B,i - C_B,i) - (T_A,i - C_A,i)
```

and the fixed-pair estimand is:

```
DiD = mean_i(d_i)
     = (accuracy_TTA,B - accuracy_clean,B) - (accuracy_TTA,A - accuracy_clean,A)
```

This algebraic equivalence is exact for the mean over the same index set
and is the primary correctness property the implementation's tests must
verify.

**Alignment requirement:** all four correctness arrays must be computed
from predictions keyed to the *identical* aligned test-image index/label
set. Any label or index mismatch between the four arrays, or between the
two conditions' evaluation artifacts, is a hard failure — never silently
reconciled, subsetted, or reordered.

## 5. Pair mappings (mechanically derived, never hand-typed)

### H1 — Normalization
For each `(dataset, resolution, seed)` with `resolution` in `{28, 64}`:
A = BatchNorm cell, B = GroupNorm cell (same dataset/resolution/seed,
Block A). **Two-sided**, per the frozen wording "differs." Expected pair
count: 2 datasets × 2 resolutions × 3 seeds = **12 pairs**.

### H2 — Resolution
For each `(dataset, normalization, seed)`: A = 28px cell, B = 64px cell
(same dataset/normalization/seed, Block A). **One-sided** (positive DiD =
degradation reduced at 64px), per the frozen wording "decreases as
resolution increases." Expected pair count: 2 datasets × 2 normalizations
× 3 seeds = **12 pairs**. Block D's 128px cells are **excluded from this
inferential pairing** — they remain trend-only/descriptive exactly as
`docs/statistical_analysis_plan.md`'s H2 confirmatory scope already
states, and are not given a fixed-pair partner here (128px has no
GroupNorm arm and no independent 28px/64px comparator built at the same
seed under the addendum's design).

**Sample-index-alignment gate:** `docs/research_plan.md`'s H2 section
already establishes, from the MedMNIST+ source-image documentation, that
sample indices and dataset splits are preserved across resolution tiers
for PathMNIST, DermaMNIST, and BloodMNIST (28/64/128px are independently
resized from the same underlying source images at the same split
indices, not upsampled from the 28px files). This is a *pre-existing,
already-verified* structural fact, not a new claim invented for this
addendum. The implementation must still verify label/index alignment
*mechanically* at runtime (identical label arrays, identical sample
count) before computing any H2 pair, and must hard-fail rather than
proceed if that check does not hold for a given pair — the addendum does
not rely on the documentation claim alone.

### H3 — Policy matching
For each `(dataset, seed)` at 28px BatchNorm: A = unmatched Block A cell
(`training_policy=none`), B = matched Block B cell
(`training_policy=matched_to_approved_tta_policy`). **One-sided**
(positive DiD = matched-policy training reduces degradation), per the
frozen wording "reduces." Expected pair count: 2 datasets × 3 seeds =
**6 pairs**.

**No pooling.** Within every hypothesis, each seed-specific fixed pair
produces its own independent DiD point estimate and CI. Seeds are never
pooled into a single model-population p-value; datasets, resolutions, and
normalizations are never pooled across pairs.

## 6. Bootstrap specification

Reuses `docs/statistical_analysis_plan.md`'s frozen paired-bootstrap
parameters exactly: resample **aligned test-image indices** with
replacement, **≥10,000 resamples**, **95% CI**. For every replicate, the
*same* sampled index set is applied jointly to all four correctness
arrays (`C_A`, `T_A`, `C_B`, `T_B`) before recomputing `DiD` — independent
resampling of the four arrays is explicitly forbidden (that would break
the pairing structure the estimand depends on and inflate variance
non-physically).

The one bootstrap detail not already pinned by the SAP — the RNG seed for
a given pair's resampling — is fixed by a deterministic derivation rule
(`configs/final_test_cross_condition_addendum.yaml`'s `bootstrap.seed_derivation`):
a SHA-256 hash of the hypothesis name, pair ID, and analysis fingerprint,
truncated to a `uint64` seed. This is justified purely by a
reproducibility convention (a pure function of the analysis's own
identity, collision-resistant across pairs) and is unrelated to any
observed result.

## 7. Disclosure taxonomy (frozen)

Every statistical claim in any future report must be tagged with exactly
one of:

1. **Originally preregistered, primary confirmatory:** the within-cell
   clean-vs-TTA test-set bootstrap and McNemar analyses
   (`docs/statistical_analysis_plan.md`, frozen before any result was
   observed).
2. **Post-validation, pre-test-specified, secondary:** the fixed-model,
   image-paired DiD estimates and intervals defined by this addendum.
3. **Descriptive/exploratory:** cross-seed summaries, any general
   method-level H1/H2/H3 comparison, Block C's external comparison, Block
   D's 128px trend, calibration metrics, alternative aggregators, BN
   adaptation, N-curves, and any analysis not explicitly registered as
   primary or secondary above.

## 8. Interpretation — permitted and forbidden wording

**Forbidden**, for any DiD result produced by this addendum:
* "BatchNorm and GroupNorm differ in general."
* "Higher resolution reduces TTA harm in general."
* "Matched augmentation universally prevents TTA harm."
* "The result generalizes over retraining randomness."

**Permitted:**
* "For this fixed pair of trained models, evaluated on unseen test
  samples, the estimated difference in TTA degradation was X (95% CI:
  [lo, hi])."

## 9. Confidence intervals and multiplicity

Each pair's CI is a **marginal fixed-pair secondary interval** — not a
family-wise simultaneous-coverage interval, and no adjustment for that is
applied unless explicitly frozen (none is, in this addendum). These
intervals are never converted into a binary significant/not-significant
verdict, and no new Benjamini-Hochberg family is constructed across pairs
for this addendum (BH correction remains scoped to the original
within-cell McNemar family, per `docs/statistical_analysis_plan.md`). All
eligible pair estimates must be displayed in any report — including pairs
whose interval contains zero.

## 10. Explicitly excluded from this addendum

* Seed-level sign/permutation test across the 3 seeds (Phase 2B.5B memo:
  valid but combinatorially incapable of significance at n=3 — would read
  as a formal test while being functionally decorative).
* Mixed-effects/hierarchical random-effects model with seed as a random
  effect (Phase 2B.5B memo: 3 levels is below the range where the
  between-seed variance component is stably identifiable).
* Any pooled, cross-dataset, cross-resolution, or cross-normalization
  statistic.
* Any new training.
* Any test-split access (this document freezes a specification only).
