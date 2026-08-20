# Phase 2B.4F GroupNorm Persistence-Schema Freeze

This document is a documentation-only operationalization of an
already-frozen scientific rule -- BN adaptation is inapplicable to
GroupNorm models (see `configs/experiment_matrix.yaml`'s
`normalization_variants: [batchnorm, groupnorm]` and
`src/when_tta_hurts/evaluation/bn_adaptation.py`'s
`BNAdaptationNotApplicableError`, both pre-existing and unchanged). It
does **not** change TTA views, aggregation, model inference, metric
formulas, seeds, or batching -- it defines the exact persistence/
validation contract for the one field
(`batching.bn_adaptation_microbatches_at_primary_n`) and one new field
(`batching.bn_adaptation_applicable`) that must correctly encode
"BN-adaptation did not run" for a GroupNorm cell, closing the schema gap
that failed
`A-pathmnist-28px-groupnorm-policy-none-s0` attempt 1 (see
`docs/phase2b_validation_evaluation_groupnorm_persistence_incident.md`).
No implementation change is made by this document; it is committed
before the engineering correction that implements it.

## Frozen contract

**`normalization == "batchnorm"`** (persisted `metadata.normalization`
is the single authoritative signal this contract binds to):

- `metadata.batching.bn_adaptation_applicable = true`
- `metadata.batching.bn_adaptation_microbatches_at_primary_n` must be a
  **positive integer** (`> 0`)
- `metrics.conditions.bn_adapted_tta` must be present and non-null, with
  an entry for every registered prefix in `prefix_sequence`
- `predictions.npz` must contain `bn_adapted_probs` (shape
  `[len(prefix_sequence), N, C]`) and `bn_adapted_prefix_sequence` --
  every reported `bn_adapted_tta[n]` metric must remain independently
  recomputable from persisted evidence (already-frozen Part F
  requirement, unchanged)

**`normalization == "groupnorm"`**:

- `metadata.batching.bn_adaptation_applicable = false`
- `metadata.batching.bn_adaptation_microbatches_at_primary_n = 0`
  (exactly zero -- not `null`/`None`, not omitted, not any other
  sentinel)
- `metrics.conditions.bn_adapted_tta` must be `null`/absent -- **no
  BN-adapted metric of any kind may be reported**
- `predictions.npz` must **not** contain `bn_adapted_probs` or
  `bn_adapted_prefix_sequence` -- **no BN-adapted probability array or
  prefix sequence may be persisted**

**In both cases:**

- `None` is never an accepted value for
  `bn_adaptation_microbatches_at_primary_n` -- it is not a valid
  substitute for either "not applicable" (which is `0`) or a real count
  (which is a positive integer). A `None` here indicates a construction
  bug, not a legitimate not-applicable state, and must fail closed.
- The validator must **cross-check** `bn_adaptation_applicable` against
  the persisted `metadata.normalization` field itself and fail closed on
  any contradiction (e.g. `normalization=batchnorm` paired with
  `bn_adaptation_applicable=false`, or vice versa) -- applicability is
  never taken on faith from a single field in isolation.
- This binds three independent signals (the applicability flag, the
  microbatch count, and the actual presence/absence of BN-adapted
  metrics and probability arrays) to each other and to the authoritative
  normalization value, so that a partial/inconsistent write (e.g. a
  future code change that sets the flag correctly but still leaks a
  stray `bn_adapted_probs` array) is caught mechanically rather than
  silently persisted.

## What does not change

TTA view generation, the frozen aggregation formulas (mean-probability,
majority-vote, confidence-weighted, original-anchored), the
`probability_native_v1` metric-input contract, the frozen TTA seed,
`inference_batch_size`/`bn_adaptation_batch_size`/
`bn_adaptation_algorithm`/`bn_adaptation_enumeration_order`, the primary
N=50 endpoint, and the test firewall are all unaffected. This freeze
touches only how BN-adaptation's already-correct
runtime behavior (skip entirely for GroupNorm, via
`BNAdaptationNotApplicableError`, unchanged) gets reflected in persisted
metadata and validated before `status="completed"` can be written.

## Effect on already-completed cells

The two Block A cells that completed successfully before this incident
(`A-pathmnist-28px-batchnorm-policy-none-s1`,
`A-pathmnist-28px-batchnorm-policy-none-s2`) are both **BatchNorm**
cells -- their `bn_adaptation_microbatches_at_primary_n` was already a
correctly-populated positive integer (the code path this freeze changes,
the GroupNorm `.get(PRIMARY_N, 0)` default, was never reached for them).
They are **not** rerun, amended, or excluded by this freeze or its
implementation.
