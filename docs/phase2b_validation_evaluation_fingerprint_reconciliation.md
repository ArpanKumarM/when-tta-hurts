# Phase 2B Evaluator-Fingerprint Reconciliation: PathMNIST 28px BatchNorm cohort

**Recorded: 2026-08-21.** This document records the decision and mechanical
proof behind superseding three historical validation-evaluation
completions for evaluator-fingerprint uniformity. It does not modify any
historical artifact or ledger row; it only records a new amendment for
each affected completion, per the existing eligibility-overlay mechanism
already used for the double-softmax exclusion.

## 1. Affected cells

| Run ID | Evaluation attempt | Evaluation ID | Evaluator fingerprint |
|---|---|---|---|
| `A-pathmnist-28px-batchnorm-policy-none-s0` | 4 | `e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4` | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` |
| `A-pathmnist-28px-batchnorm-policy-none-s1` | 1 | `d453bc9c9e13aac9d413c5827407ddfff87985796896fd70adf7401a78997f3c` | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` |
| `A-pathmnist-28px-batchnorm-policy-none-s2` | 1 | `add32ac4b38553726ad79cc207cfbeeeef6f52fda563d83f243235e91373e00a` | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` |

**Old evaluator fingerprint**: `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7`.
**Current evaluator fingerprint**: `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`.

All three are `A-pathmnist-28px-batchnorm-policy-none-*` cells (BatchNorm,
`policy=none`, 28px). Every other canonical validation-evaluation
completion in the project (21 remaining Block A cells, all of Block B/C/D)
already uses the current fingerprint.

## 2. Manifested-file diff

The evaluator fingerprint is a content hash over `EVALUATOR_FINGERPRINT_MANIFEST`
(21 files). Comparing the old cohort's persisted per-file manifest
(`metadata.json["evaluator_fingerprint_manifest"]`) against the current
manifest computed by `compute_evaluator_fingerprint()`, **exactly two
files differ**:

- `src/when_tta_hurts/evaluation_result_artifacts.py`
- `src/when_tta_hurts/validation_evaluation.py`

All other 19 manifested files (including `metrics.py`,
`evaluation/aggregation.py`, `evaluation/bn_adaptation.py`,
`evaluation/views.py`, `evaluation/validation_loader.py`,
`evaluation/latency.py`, `data.py`, `dataset_verification.py`,
`configs/validation_evaluation.yaml`, `matrix.py`, `orchestrator.py`,
`models/small_cnn.py`, `models/resnet.py`, `transforms/policies.py`,
`devices.py`, `config.py`, `artifacts.py`, `pyproject.toml`, `uv.lock`)
are byte-identical between the old cohort and current HEAD.

The old cohort's persisted per-file hashes for these two files were
independently confirmed to match the git blob at the persisted
`source_commit` (`34528ee36c529829b42ca8c2ab670479516f4b5a` for seed 0;
`13be8539787b24a3b5712dbfe6c4232a05e81e1e` for seeds 1/2) exactly
(`git show <commit>:<path> | shasum -a 256`), confirming the comparison is
against the true historical source, not an approximation.

### Exact diff, `evaluation_result_artifacts.py`

- Added `"bn_adaptation_applicable"` to `_BATCHING_REQUIRED_KEYS`.
- `_validate_batching_schema()`: added a bool-type check for the new key;
  extended one error message's text (no change to the underlying
  condition, which remains `n_micro < 0`).
- Added new function `_validate_bn_adaptation_applicability_consistency()`,
  called once from `persist_and_verify_evaluation_completion()`
  immediately after `_validate_metrics_schema()`. This function only
  **reads** already-computed `metadata`/`predictions`/`metrics` and either
  passes silently or raises `EvaluationSchemaValidationError`. It never
  mutates any value.

### Exact diff, `validation_evaluation.py`

- In `compute_validation_evaluation()`'s batching-metadata dict
  construction: added a new key,
  `"bn_adaptation_applicable": conditions["bn_adapted_tta"] is not None`,
  and changed `bn_adaptation_microbatch_counts.get(PRIMARY_N)` to
  `bn_adaptation_microbatch_counts.get(PRIMARY_N, 0)`.

## 3. Reachability analysis for these three BatchNorm cells

- `_validate_bn_adaptation_applicability_consistency()` is called only
  inside `persist_and_verify_evaluation_completion()`, strictly **after**
  `compute_validation_evaluation()` has already fully computed
  `predictions` and `metrics`. It is a post-hoc persistence-time gate,
  not part of probability generation, aggregation, or metric computation.
- For a BatchNorm cell, BN-adaptation always runs, so
  `conditions["bn_adapted_tta"] is not None` is always `True` and
  `bn_adaptation_microbatch_counts` always already contains a real,
  positive value at the `PRIMARY_N` key (populated during the BN-adaptation
  forward pass, which is unchanged by this diff). Consequently:
  - `.get(PRIMARY_N, 0)` returns byte-for-byte the same value as
    `.get(PRIMARY_N)` for these cells -- the `0` fallback is unreachable
    for BatchNorm; it is reachable only for GroupNorm cells (which never
    populate this dict at all, since BN-adaptation does not run for
    GroupNorm).
  - The new consistency check's `applicable=True` branch requires a
    positive microbatch count, `bn_adapted_tta` present, and
    `bn_adapted_probs`/`bn_adapted_prefix_sequence` present -- all of
    which were already true for these three historical completions (as
    independently confirmed in Section 4 below), so the check is a
    no-op pass for them.
- **No changed line in either file touches**: probability generation
  (`evaluation/views.py`, unchanged), aggregation
  (`evaluation/aggregation.py`, unchanged), metric computation
  (`metrics.py`, unchanged), BN-adaptation's own forward-pass/statistics
  logic (`evaluation/bn_adaptation.py`, unchanged), batching constants or
  execution (`INFERENCE_BATCH_SIZE`/`BN_ADAPTATION_BATCH_SIZE`, both
  unchanged as constants and as call sites), the frozen TTA view seed
  (`evaluation/validation_loader.py`, unchanged), latency measurement
  (`evaluation/latency.py`, unchanged), or dataset loading/checksum
  verification (`data.py`, `dataset_verification.py`, unchanged).
- **The change was limited to GroupNorm persistence/applicability
  handling and its associated validation** -- exactly the correction
  frozen in `docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md`
  and implemented per
  `docs/phase2b_validation_evaluation_groupnorm_persistence_incident.md`.
  It has zero reachable effect on any BatchNorm cell's scientific
  computation.

## 4. Semantic-verification results

For each of the three affected cells, independently re-derived from
persisted `predictions.npz`:

| Run ID (attempt) | Manifest verify | Semantic recomputation mismatches |
|---|---|---|
| `A-pathmnist-28px-batchnorm-policy-none-s0` (4) | OK | 0 |
| `A-pathmnist-28px-batchnorm-policy-none-s1` (1) | OK | 0 |
| `A-pathmnist-28px-batchnorm-policy-none-s2` (1) | OK | 0 |

Recomputation covered clean metrics (`compute_metrics_from_probabilities()`)
and all 7 registered prefixes x 3 `naive_tta` aggregators
(`_recompute_all_conditions_from_predictions()`), compared against each
cell's persisted `metrics.json` within the frozen `1e-6` tolerance --
**zero mismatches** in all three cells.

## 5. Why the old results remain scientifically valid

The manifested-file diff is confined to two files, and every changed
branch is either (a) unreachable for BatchNorm cells, or (b) a read-only
consistency check that these three cells already satisfy. Combined with
the zero-mismatch independent semantic recomputation in Section 4, this
establishes that **the persisted predictions and metrics for these three
completions are scientifically identical to what the current evaluator
implementation would produce for the same checkpoint, dataset, and
frozen TTA seed.** Nothing about the correctness or validity of these
three historical results is in question.

## 6. Why controlled reruns are nevertheless preferred

Despite the equivalence established above, three considerations favor a
controlled rerun over a permanent split-fingerprint cohort in the final
published record:

- **Publication-quality uniformity**: a single evaluator fingerprint
  across every canonical completion removes any need for a reader (or a
  future maintainer) to reason about whether a persisted-manifest
  equivalence proof still applies after any future code change.
- **Avoiding a permanent bypass**: the alternative -- teaching the
  skip/conflict logic to treat two different fingerprints as
  interchangeable -- was explicitly rejected in the original
  fingerprint-drift addendum and remains rejected here. A rerun under the
  current fingerprint is the only mechanism that does not require any
  such compatibility logic.
- **Zero marginal cost of correctness**: because the change is proven
  reachability-inert for these cells, a rerun is expected (not merely
  hoped) to reproduce byte-identical scientific outputs, which Part E of
  the accompanying task verifies directly. The rerun is a provenance
  action, not a scientific correction.

## 7. Non-result-driven guarantee

**These reruns are not result-driven.** No hypothesis, threshold, policy,
or analysis will change based on their outcome. If a rerun produces a
prediction or metric that differs from its historical counterpart beyond
the frozen tolerance, execution stops immediately and the discrepancy is
reported -- it is not silently accepted, and no protocol change is made
in response. The expected outcome, established above, is exact
scientific equivalence with only identity/provenance metadata (evaluation
ID, evaluator fingerprint, timestamps, and the new
`bn_adaptation_applicable` field) differing.

## 8. Historical preservation guarantee

The three historical attempt directories
(`artifacts/validation_evaluation/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_004`,
`.../s1/attempt_001`, `.../s2/attempt_001`) and their ledger rows in
`artifacts/ledger_validation_evaluation.csv` **remain permanently
preserved, byte-identical, and untouched**. This reconciliation appends
new amendment rows to the eligibility-overlay ledger
(`artifacts/ledger_validation_evaluation_amendments.csv`) -- it does not
delete, rewrite, or modify any original row or artifact.

## 9. Not a scientific-fraud or computational-error finding

**The old attempts are being superseded for implementation-uniformity,
not because they are scientifically fraudulent or computationally
incorrect.** Section 3-5 above establish the opposite: they are correct,
valid, and semantically equivalent to what a current-fingerprint rerun
is expected to produce. The amendment `reason` field records this
explicitly as `superseded_for_current_evaluator_fingerprint_uniformity`,
distinct in kind from the pre-existing `probability_metric_double_softmax`
amendment (a genuine computational defect in an unrelated attempt).

## 10. Amendment ledger schema note

`EVALUATION_AMENDMENTS_LEDGER_FIELDNAMES` does not include a dedicated
"scientific validity" boolean field. The closest fields are
`historical_status` (preserved as `completed`, per Section 8) and
`rerun_required` (set to `True` for these three rows, since a rerun is
in fact being performed as part of this reconciliation -- distinct from
the existing double-softmax row, where `rerun_required=True` reflects a
genuine defect requiring a corrected rerun for a different reason). No
existing field is repurposed or overloaded to claim scientific invalidity
that Section 5 above disproves.
