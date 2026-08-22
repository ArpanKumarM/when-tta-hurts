# Phase 2B Evaluator-Fingerprint Reconciliation: Final Audit

**Recorded: 2026-08-22.** This document closes the Phase 2B
evaluator-fingerprint reconciliation: three historical PathMNIST-28px
BatchNorm validation-evaluation completions, previously computed under
an older evaluator fingerprint, have been superseded by controlled
reruns under the current fingerprint. All numbers below are generated
mechanically from persisted artifacts and ledgers -- none are
hand-transcribed. **No test split was accessed at any point.**

## 1. Reconciliation purpose

Three scientifically valid BatchNorm evaluations
(`A-pathmnist-28px-batchnorm-policy-none-s0/-s1/-s2`) were computed
under an older evaluator fingerprint
(`f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7`).
A later, unrelated correction to GroupNorm persistence/applicability
handling (Phase 2B.4F, frozen in
`docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md`)
changed two fingerprint-manifested source files, producing a new
evaluator fingerprint
(`7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`).
The GroupNorm correction had no reachable effect on these three
BatchNorm cells (see Section 3), so their persisted results remained
scientifically valid but fell into a separate, older-fingerprint
cohort. For publication-quality uniformity -- not because of any
scientific defect -- all three were superseded via the eligibility-
overlay amendment mechanism and re-evaluated under the current
fingerprint. This document is the closing record of that
reconciliation, following
`docs/phase2b_validation_evaluation_fingerprint_reconciliation.md`
(the decision/proof document written before the reruns) and the three
reruns' own independent verification (already reported in-session; this
document is the persisted mechanical closure).

## 2. Historical mapping

| Cell | Superseded eval ID (attempt) | Replacement eval ID (attempt) | Old fingerprint | Current fingerprint | Checkpoint hash | Amendment reason | Canonical selection |
|---|---|---|---|---|---|---|---|
| `A-pathmnist-28px-batchnorm-policy-none-s0` | `e59debe937108abf...` (attempt 4) | `fd7094a0ffc95a99...` (attempt 5) | `f6435f98c133a4bf...` | `7fdce1db496ffb14...` | `30bc1ca6ef364e2a...` | `superseded_for_current_evaluator_fingerprint_uniformity` | new attempt is sole eligible completion |
| `A-pathmnist-28px-batchnorm-policy-none-s1` | `d453bc9c9e13aac9...` (attempt 1) | `e1996b8a9526337d...` (attempt 2) | `f6435f98c133a4bf...` | `7fdce1db496ffb14...` | `f3be88438078ce36...` | `superseded_for_current_evaluator_fingerprint_uniformity` | new attempt is sole eligible completion |
| `A-pathmnist-28px-batchnorm-policy-none-s2` | `add32ac4b3855372...` (attempt 1) | `b482ffbf1384fe7b...` (attempt 2) | `f6435f98c133a4bf...` | `7fdce1db496ffb14...` | `b8b971407b6b149d...` | `superseded_for_current_evaluator_fingerprint_uniformity` | new attempt is sole eligible completion |

### Artifact hashes

| Cell | Old predictions.npz SHA-256 | New predictions.npz SHA-256 | Bitwise identical |
|---|---|---|---|
| `A-pathmnist-28px-batchnorm-policy-none-s0` | `48b6ff9cf6900853043426ed3381537a84dba29b944670302229008ee1e3ba07` | `48b6ff9cf6900853043426ed3381537a84dba29b944670302229008ee1e3ba07` | True |
| `A-pathmnist-28px-batchnorm-policy-none-s1` | `964a79f1b38485e5843d53313d006c2589af1f1aa8b1aaae4fddf215d22588f2` | `964a79f1b38485e5843d53313d006c2589af1f1aa8b1aaae4fddf215d22588f2` | True |
| `A-pathmnist-28px-batchnorm-policy-none-s2` | `aa63c533109ca62e3b75be06eabe8278c6357007cdb37e19f4f773cc9286eef6` | `aa63c533109ca62e3b75be06eabe8278c6357007cdb37e19f4f773cc9286eef6` | True |

## 3. Reachability proof (summary)

The evaluator-fingerprint manifest diff between the old and current
cohorts is confined to exactly two files:
`src/when_tta_hurts/evaluation_result_artifacts.py` and
`src/when_tta_hurts/validation_evaluation.py`. Independent verification
(`git show <old_source_commit>:<path> | shasum -a 256`) confirmed the
old cohort's persisted per-file hashes match the git blob at their
recorded `source_commit` exactly, establishing the comparison is
against the true historical source.

The diff adds a new `bn_adaptation_applicable` metadata field, a new
post-hoc consistency validator
(`_validate_bn_adaptation_applicability_consistency()`, called only
inside `persist_and_verify_evaluation_completion()`, strictly **after**
`compute_validation_evaluation()` has already computed all predictions
and metrics -- it only reads and either passes or raises, never
mutates), and changes a dict-lookup default from
`bn_adaptation_microbatch_counts.get(PRIMARY_N)` to
`.get(PRIMARY_N, 0)`.

For a BatchNorm cell, `bn_adaptation_microbatch_counts` always already
contains a real, positive value at the `PRIMARY_N` key (BN-adaptation
always runs for BatchNorm), so the `0` fallback is unreachable for
these cells -- it is reachable only for GroupNorm cells, which never
populate this dict at all. The new consistency check's `applicable=True`
branch requires exactly the state these three historical completions
already had (positive microbatch count, `bn_adapted_tta` present,
`bn_adapted_probs`/`bn_adapted_prefix_sequence` present), so it is a
no-op pass for them. **No line in either changed file touches**
probability generation, aggregation, metric computation, BN-adaptation's
own forward-pass logic, batching execution, the frozen TTA view seed,
latency measurement, or dataset loading/checksum verification -- all of
which live in files confirmed byte-identical between the two cohorts.
**The change was limited to GroupNorm persistence/applicability
handling and its associated validation**, with zero reachable effect on
any BatchNorm cell's scientific computation. Full detail, including the
complete diff text, is in
`docs/phase2b_validation_evaluation_fingerprint_reconciliation.md`
Sections 2-3.

## 4. Prediction equivalence

### `A-pathmnist-28px-batchnorm-policy-none-s0`

| Array | Shape | Bitwise identical |
|---|---|---|
| `labels` | (10004,) | True |
| `sample_indices` | (10004,) | True |
| `clean_probs` | (10004, 9) | True |
| `view_probs` | (100, 10004, 9) | True |
| `bn_adapted_probs` | (7, 10004, 9) | True |
| `bn_adapted_prefix_sequence` | (7,) | True |

**Seed manifest (`view_manifest.json`, encodes the frozen deterministic view-generation seed/order) bitwise identical**: True

### `A-pathmnist-28px-batchnorm-policy-none-s1`

| Array | Shape | Bitwise identical |
|---|---|---|
| `labels` | (10004,) | True |
| `sample_indices` | (10004,) | True |
| `clean_probs` | (10004, 9) | True |
| `view_probs` | (100, 10004, 9) | True |
| `bn_adapted_probs` | (7, 10004, 9) | True |
| `bn_adapted_prefix_sequence` | (7,) | True |

**Seed manifest (`view_manifest.json`, encodes the frozen deterministic view-generation seed/order) bitwise identical**: True

### `A-pathmnist-28px-batchnorm-policy-none-s2`

| Array | Shape | Bitwise identical |
|---|---|---|
| `labels` | (10004,) | True |
| `sample_indices` | (10004,) | True |
| `clean_probs` | (10004, 9) | True |
| `view_probs` | (100, 10004, 9) | True |
| `bn_adapted_probs` | (7, 10004, 9) | True |
| `bn_adapted_prefix_sequence` | (7,) | True |

**Seed manifest (`view_manifest.json`, encodes the frozen deterministic view-generation seed/order) bitwise identical**: True

**`predictions.npz` was bitwise identical for all three pairs** (0 mismatches found across all cells x arrays).

## 5. Metric equivalence

For each cell, every scientific metric was mechanically compared
between the superseded and replacement `metrics.json` files: clean
inference (accuracy, macro-F1, NLL, ECE, Brier), and every registered
N (1/2/5/10/25/50/100) under mean-probability, majority-vote,
confidence-weighted, original-anchored, and BN-adapted conditions
(accuracy, macro-F1, NLL, ECE, Brier, harm rate, rescue rate, delta
accuracy).

- `A-pathmnist-28px-batchnorm-policy-none-s0`: 23 total differing leaves; **0 scientific mismatches**
- `A-pathmnist-28px-batchnorm-policy-none-s1`: 23 total differing leaves; **0 scientific mismatches**
- `A-pathmnist-28px-batchnorm-policy-none-s2`: 23 total differing leaves; **0 scientific mismatches**

**Zero scientific mismatches across all three cells** (0 total). Every
accuracy, macro-F1, NLL, ECE, Brier, harm-rate, rescue-rate, and
delta-accuracy value matched exactly between each superseded
completion and its replacement.

## 6. Expected differences

The only fields that differed between each superseded/replacement pair
were:

- **`evaluation_config_hash` / evaluation identity** -- expected, since
  the evaluator fingerprint (an input to the evaluation-ID hash)
  legitimately changed.
- **Descriptive latency measurements** (`metrics.json` `latency.*`
  fields: `clean_latency_seconds`, `tta_latency_seconds`,
  `compute_multiplier`, `per_sample_latency_seconds` at every N) --
  these are wall-clock hardware timing measurements taken at
  evaluation time, not deterministic scientific computation. They are
  **not expected to be bitwise reproducible** across separate process
  invocations on real hardware, and their small differences (typically
  well under 5%) are consistent in magnitude with ordinary run-to-run
  hardware timing variance observed throughout this project.

## 7. Historical preservation

- **No original ledger row was rewritten.** The three superseded rows
  (`A-pathmnist-28px-batchnorm-policy-none-s0` attempt 4,
  `-s1` attempt 1, `-s2` attempt 1) remain exactly as originally
  appended, byte-identical, in `artifacts/ledger_validation_evaluation.csv`.
- **No historical artifact was modified.** All three superseded attempt
  directories' `artifact_manifest.json` entries were independently
  re-verified against the on-disk files in this closure and passed
  with zero exceptions (Part A, checks 8-9).
- **All historical attempts remain auditable** -- every file
  (`predictions.npz`, `metrics.json`, `metadata.json`,
  `view_manifest.json`, `status.json`, `artifact_manifest.json`) is
  still present and readable at its original path.
- **`canonical_eligible=False` means superseded for final selection,
  not fabricated or deleted.** The amendments ledger is a strictly
  additive eligibility overlay
  (`artifacts/ledger_validation_evaluation_amendments.csv`) -- it never
  modifies, hides, or removes the underlying evaluation-ledger row or
  attempt directory it refers to.
- **The old computations remain scientifically valid under their
  original identities.** Sections 3-5 above establish that the
  superseded results are bit-for-bit and metric-for-metric equivalent
  to their replacements; nothing about their correctness is in
  question. They are superseded solely for current-fingerprint
  provenance uniformity.

## 8. Final canonical state

- **39/39 canonical validation evaluations** -- Block A 24, Block B 6, Block C 3, Block D 6 = 39.
- **All 39 current-fingerprint-compatible**: 39/39 confirmed matching
  `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`.
- **Ledger totals**: 46 data rows -- 43 completed, 2 failed, 1 aborted.
- **Amendment rows**: 4 total (1 pre-existing double-softmax exclusion + 3 fingerprint-uniformity supersessions).
- **No test-split access** at any point -- every ledger row has
  `split=validation`; no test-split code path exists in
  `validation_evaluation.py`.
- **No final-test authorization exists.** Phase 2B.5 (the one-time
  test-split unlock) has not been started or approved.

