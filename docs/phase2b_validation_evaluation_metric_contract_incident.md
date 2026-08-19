# Phase 2B.4D Metric-Contract Incident: attempt 3 recorded completed-but-noncanonical

**Recorded: 2026-08-19T17:58:59Z.** This document records, honestly and
without deletion, a scientific-correctness defect found during the
mandatory post-run audit of validation-evaluation attempt 3. No source
code is changed by this document. Attempt 3's artifacts
(`artifacts/validation_evaluation/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_003/`)
remain byte-identical and untouched. This document is committed together
with the strict, single-row ledger append that already resulted from the
production completion path, and a new append-only evaluation-amendments
row excluding attempt 3 from canonical selection -- neither modifies the
existing rows.

## 1. What happened, mechanically confirmed

Attempt 3 completed mechanically: MPS ran, the canonical checkpoint was
verified and restored, the official dataset artifact was checksum-
verified, 100 deterministic TTA views were generated and scored in
bounded 256-image batches, BN-adaptation ran via `sequential_microbatch_v1`
for all seven registered N values, and every persisted artifact passed
its schema/manifest verification before `status="completed"` was written.
**Validation metrics were observed. Test metrics were not observed** --
no test-split loading mechanism exists anywhere in this call path.

The post-run audit (below) found that `_per_prefix_metrics()`
(`src/when_tta_hurts/validation_evaluation.py`) applies an unintended
extra `softmax()` to every aggregate condition's probability distribution
before computing NLL, ECE, and Brier score. **This is a formula-contract
defect discovered during the mandatory post-run audit, found by tracing
the frozen aggregation-module's own documented contract against the
actual caller behavior and confirming the discrepancy with a
hand-calculated numeric example -- not a defect chosen or tuned in
response to the observed −28.04pp accuracy result.** The accuracy result
itself is untouched by this defect (see below) and was already known
before the code trace began.

## 2. Mechanical adjudication (full detail: see the accompanying report)

`src/when_tta_hurts/evaluation/aggregation.py`'s module docstring states
the aggregation functions deliberately return their probability output in
**log-probability space** specifically "so it composes directly with
metrics.py's softmax-based functions: `softmax(log(p)) == p` exactly" --
i.e. exactly ONE softmax, inside the metric function, recovers the
correct probabilities. Every call site in `compute_validation_evaluation()`
correctly performs this single conversion
(e.g. `softmax(mean_probability(view_log_probs, n))`, `adapted_probs =
softmax(adapted_logits)`). The resulting **genuine, already-normalized
probabilities** are then passed into `_per_prefix_metrics(clean_probs,
agg_probs, labels)`, which converts `clean_probs` to log-space before use
(correct) but passes `agg_probs` **directly, unconverted**, into
`accuracy()`/`macro_f1()`/`negative_log_likelihood()`/
`expected_calibration_error()`/`brier_score()` -- functions that all
apply `softmax()` internally, expecting a logit-equivalent input. The
result: every aggregate condition (`naive_tta` x3 aggregators,
`original_anchored_tta`, `bn_adapted_tta`) receives a **second, spurious
softmax** applied to an already-normalized probability distribution
immediately before NLL/ECE/Brier are computed.

A hand-calculated example (`p=[0.9,0.1]`, true class 0) confirms this
numerically: correct NLL = `-log(0.9) = 0.10536`; NLL after one extra
softmax = `0.37110` (3.5x too large); Brier correct = `0.02000`, after
one extra softmax = `0.19223` (9.6x too large); `softmax(p) != p` for
this nonuniform distribution, confirmed exactly
(`softmax([0.9,0.1]) = [0.6900, 0.3100]`).

## 3. Which quantities are affected

- **Unaffected (argmax-based, invariant to any number of monotonic
  softmax reapplications):** accuracy, macro-F1, delta-accuracy, harm
  rate, rescue rate. Confirmed computationally: recomputing accuracy for
  every condition/N via the correct probability-native path produced
  zero differences from the persisted values.
- **Affected (confidence/probability-value-dependent):** negative
  log-likelihood, expected calibration error, Brier score, and any
  derived confidence quantity. Quantified per-condition-per-N differences
  are reported in the accompanying engineering report (Part B) --
  relative differences range from roughly 6% to over 700% depending on
  condition and N, with inconsistent sign (not a uniform scaling), which
  is itself further evidence this is a genuine formula defect rather than
  a benign constant-offset artifact.

**Confidence-weighted aggregation was audited specifically**: its
per-view confidence weights are computed from correctly, singly-softmaxed
per-view probabilities entirely inside `aggregation.py`, before the
result ever reaches the buggy code path. **The aggregated prediction
itself (its class-probability values) is unaffected -- only the
downstream NLL/ECE/Brier computed from that already-correct aggregate is
wrong**, via the same `_per_prefix_metrics` root cause affecting every
other condition identically.

## 4. Primary confirmatory-relevant result

The −28.04-percentage-point accuracy result (clean 73.88% -> TTA@N=50
mean-probability 45.84%) is **argmax-based and numerically unaffected**
by this defect. It is not retracted or altered. The affected
quantities are the secondary calibration/confidence endpoints (NLL, ECE,
Brier) for every reported condition.

## 5. Disposition

- Attempt 3 **remains recorded exactly as it completed** -- the ledger
  row, `status.json`, `predictions.npz`, `metrics.json`, `metadata.json`,
  `view_manifest.json`, and `artifact_manifest.json` are all byte-
  identical to what the production run produced. Nothing is deleted,
  rewritten, or regenerated.
- Attempt 3 is **excluded from canonical confirmatory analysis** via a
  new, append-only evaluation-amendments ledger row (below) --
  `canonical_eligible=False`, mirroring exactly how training-side
  amendments already exclude ineligible completed training attempts
  without deleting their records.
- **All artifacts remain immutable.** No test split was accessed at any
  point, before or during this audit.
- **This correction is formula-driven**, derived mechanically from the
  frozen aggregation-module docstring and a hand-calculated numeric
  proof, **not chosen because of the observed accuracy result** -- the
  accuracy result is unaffected by the defect and was not used to decide
  whether a defect existed.

## 6. Evaluation-amendments ledger entry

New ledger: `artifacts/ledger_validation_evaluation_amendments.csv`
(append-only, mirrors `artifacts/ledger_amendments.csv`'s discipline --
idempotent on duplicate, hard-fails on conflicting duplicate, case-
insensitive fail-closed boolean parsing).

| Field | Value |
|---|---|
| evaluation_id | `75aa7e37a9fe5454bf8edf6483d676a182d6dde9ff4a3730e4ada7195e09eb9e` |
| evaluation_attempt | `3` |
| historical_status | `completed` |
| canonical_eligible | `False` |
| reason | `probability_metric_double_softmax` |
| validation_metrics_observed | `True` |
| test_metrics_observed | `False` |
| artifacts_preserved | `True` |
| rerun_required | `True` |
| predictions_sha256 | `c9930c594f974f6d4019475cbcb51d4896a1bf27d497628ef42457038d77823a` |
| source_commit | `b826338322d75f56894b6f50cfb3fbbd957ae4f3` |
| recorded_at | `2026-08-19T17:58:59Z` |

## 7. Next steps (implemented separately, after this record is committed)

The metric-input contract is frozen and the fix implemented in
subsequent, separate commits (documentation/config freeze, then
engineering implementation + tests). Attempts 1 (aborted), 2 (failed),
and 3 (completed, canonical-ineligible) collectively make the next real
evaluation attempt for this training run resolve to **attempt 4**.
