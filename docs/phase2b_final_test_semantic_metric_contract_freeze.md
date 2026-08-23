# Phase 2B.6J Part D — Final-Test Semantic Metric Contract Freeze

**Status: FROZEN before any engineering change.** This document records
the mechanically-proven root cause of the cell-2-attempt-1 semantic
verification failure and freezes the correction/recovery policy. No
production code is modified by this document; no test-split access
occurred at any point in this investigation (Part B/C used only static
source inspection, synthetic arrays, and cell 1's already-persisted,
already-authorized `predictions.npz`/`metrics.json`).

## 1. Root cause (mechanically proven)

`original_anchored_mean_probability(clean_logits, ordered_view_logits,
n_views)` (`src/when_tta_hurts/evaluation/aggregation.py`) internally
computes `clean_probs = softmax(clean_logits)` and averages it,
unclipped, with the augmented-view probabilities before returning
`_to_log_probs(mean)`.

Two call sites feed this function DIFFERENT representations of the same
underlying clean prediction:

* **Live path** (`compute_validation_evaluation()`,
  `validation_evaluation.py:1278-1282`): passes the raw model
  `clean_logits` directly. `softmax(clean_logits)` inside the aggregator
  is computed exactly once, unclipped.
* **Semantic-verification recompute path**
  (`_recompute_all_conditions_from_predictions()`,
  `validation_evaluation.py:1069-1071`): passes
  `clean_logits_equivalent = np.log(np.clip(clean_probs, 1e-12, 1.0))`,
  where `clean_probs` is the already-softmaxed, already-persisted array.
  This round-trips through `clip -> log -> softmax` a SECOND time inside
  the aggregator.

In exact real arithmetic, `softmax(log(clip(p, eps, 1)))` reproduces `p`
exactly whenever every entry of `p` exceeds `eps`. **When any entry of
the true clean-probability vector is smaller than `eps=1e-12`** (a real,
observed condition -- see below), the recompute path's clip pins that
entry UP to `1e-12` before re-softmaxing, changing the row's
normalizing sum and therefore every entry of the resulting probability
vector -- including entries far from the clip floor. The live path never
clips this internal probability at all. The two paths therefore
compute `original_anchored_tta` from two **numerically different**
"clean anchor" probability vectors whenever clipping is triggered for
any class of any sample.

### Mechanical reproduction

Synthetic reproduction (`np.random.default_rng(0)`, extreme-magnitude
logits producing genuine sub-`1e-12` softmax entries) at prefix `n=1`
(the exact prefix that failed in the real incident) showed a mean NLL
divergence of `1.55e-4` between the live and recompute formulas --
two orders of magnitude past the frozen `atol=rtol=1e-6` tolerance, and
the same order of magnitude as the real incident's observed
`negative_log_likelihood` divergence of `3.65e-5`. No divergence was
observed for aggregators that never pass a clean/anchor value through
this specific double-round-trip (`mean_probability`, `majority_vote`,
`confidence_weighted_average` only ever consume `view_log_probs`,
derived and reused identically in both paths) -- consistent with only
`original_anchored_tta` having failed in the real incident.

### Confirmed real-world trigger condition

Cell 1's own already-persisted, already-authorized `clean_probs` array
(`artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_003/predictions.npz`)
contains 201 entries below `1e-12` (minimum `3.39e-22`) out of
7180x9=64620 total -- confirming this clipping-boundary condition is a
real, recurring feature of this classifier's genuine over-confident
predictions, not a synthetic edge case.

### Determination (per Part B's required classification)

The mismatch is caused by **clipping/normalization asymmetry between
two probability representations of the same underlying value** --
specifically, the recompute path applies an extra, avoidable
`clip -> log -> softmax` round-trip that the live path never performs.
It is NOT caused by: different formulas (the mean-of-probabilities
definition is identical in both call sites), float32/float64 dtype
mismatch (both paths operate in the same dtype), device-dependent
reduction (both computations are CPU/NumPy, post-inference), or
aggregation-order differences (the averaging order and view selection
are identical).

## 2. Cell 1 compatibility gate (Part C) -- PASSED

Cell 1's `original_anchored_tta` metrics were independently recomputed
under the proposed corrected contract (§3 below) directly from its
persisted `clean_probs`/`view_probs`/`labels`, entirely offline, and
compared against its already-persisted `metrics.json` values. All 56
checks (7 registered prefixes x 8 metric fields: accuracy, macro_f1,
negative_log_likelihood, expected_calibration_error, brier_score,
delta_accuracy, harm_rate, rescue_rate) passed within the existing
`atol=rtol=1e-6` tolerance. No endpoint definition changed, no
classification (argmax-based harm/rescue) changed. Cell 1 requires no
amendment, supersession, rerun, or reauthorization.

## 3. Frozen correction contract

1. **Persistable probability arrays are the single numerical source of
   truth for reported metrics.** `clean_probs` and `view_probs` (the
   exact arrays written to `predictions.npz`) are the only representation
   from which any reported metric may be computed -- never a raw-logit
   or re-derived higher-precision shadow copy computed in parallel.
2. **Metrics must be computed from the exact canonical probability
   representation that will be persisted -- not from a parallel,
   differently normalized representation.** Concretely:
   `original_anchored_mean_probability()` must accept an
   already-computed clean **probability** array directly (not clean
   logits, and never internally re-derive it via its own `softmax`
   call) -- eliminating the double-softmax/round-trip divergence by
   construction, for both the live and recompute call sites, which will
   now pass the identical `clean_probs` object.
3. Aggregation formulas (equal-weight mean of clean + n augmented
   views), endpoints, clipping rules elsewhere (`_EPS = 1e-12` for the
   augmented-view log-probability conversion, unchanged), prefixes
   (`PREFIX_SEQUENCE`), and hypotheses are unchanged.
4. The existing `atol=1e-6`/`rtol=1e-6` semantic-verification tolerance
   is NOT loosened. The fix eliminates the source of divergence rather
   than tolerating it.
5. Canonicalization is explicit: dtype is whatever `clean_probs`/
   `view_probs` already are at persistence time (float32, CPU NumPy,
   row-normalized by construction of `softmax()`); operation order is
   "concatenate clean + augmented probability rows, then arithmetic
   mean along the view axis" -- unchanged from the current definition,
   only the INPUT REPRESENTATION for the clean anchor changes (from
   logits to already-computed probabilities).
6. Independent semantic verification (`_verify_metrics_semantically()`)
   remains mandatory before any attempt may reach `status="completed"`.
7. Cell 2 attempt 1 (`A-pathmnist-28px-batchnorm-policy-none-s1`) is
   permanently `failed` and consumed -- never amended, deleted, or
   retried at attempt 1.
8. Cell 2 may receive exactly one recovery authorization, at attempt 2,
   after this correction is implemented, tested, and reauthorized.
9. Cell 1 remains `completed_consumed` at attempt 3 and must never
   rerun.
10. Cells 3-39 retain their existing authorization at attempt 1.
11. **Required paper/audit disclosure:** cell 2's first real test
    execution (attempt 1) computed predictions and metrics but failed
    BEFORE persistence, due to a since-corrected floating-point
    clipping asymmetry in the `original_anchored_tta` metric's clean-
    anchor representation. Exactly one diagnostic NLL comparison
    (`recomputed=1.2021862268447876, persisted=1.2022227048873901`)
    appeared in the failure traceback as a byproduct of the integrity
    check itself. No TTA-efficacy conclusion, comparison, or scientific
    choice was made from this value at any point.
