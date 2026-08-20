# Phase 2B.4F Evaluator-Fingerprint Drift Addendum

**Recorded: 2026-08-20.** This is a separate, documentation-only
addendum recording a mechanically-discovered finding about evaluator-
fingerprint drift on three already-completed Block A BatchNorm cells,
found while pre-verifying the remaining 20 Block A cells for execution.
It does **not** amend, rewrite, or supersede
`docs/phase2b_validation_evaluation_groupnorm_canary_audit.md`, which
remains committed exactly as recorded. No source code, amendment ledger
row, skip-logic, or compatibility override is added by this document.
No cell is rerun by this document.

## 1. What was found

`src/when_tta_hurts/validation_evaluation.py` and
`src/when_tta_hurts/evaluation_result_artifacts.py` are both listed in
`EVALUATOR_FINGERPRINT_MANIFEST`. The Phase 2B.4F GroupNorm persistence-
schema correction (commit `d5227d0`) modified both files, which changed
`compute_evaluator_fingerprint()`'s output -- exactly as it should,
since the evaluator's implementation genuinely changed. This is the
**second** such fingerprint change in this phase (the first was the
Phase 2B.4D metric-contract correction, which produced fingerprint
`f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` and
is what `A-pathmnist-28px-batchnorm-policy-none-s0` attempt 4,
`-s1` attempt 1, and `-s2` attempt 1 all ran under).

Recomputing each cell's evaluation identity **today**, under the
current evaluator fingerprint
(`7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`),
produces:

| Run ID | Persisted attempt | Persisted `evaluator_fingerprint` | Persisted `evaluation_id` | Recomputed `evaluation_id` (current fingerprint) | Match? |
|---|---|---|---|---|---|
| `A-pathmnist-28px-batchnorm-policy-none-s0` | 4 | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` | `e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4` | `fd7094a0ffc95a998e0ae03110c8991f6819a5ca1524036d873658a18c6bdc29` | **No** |
| `A-pathmnist-28px-batchnorm-policy-none-s1` | 1 | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` | `d453bc9c9e13aac9d413c5827407ddfff87985796896fd70adf7401a78997f3c` | `e1996b8a9526337d6cdf8dad1e7226064c216ecec42a8023a30826faf4665137` | **No** |
| `A-pathmnist-28px-batchnorm-policy-none-s2` | 1 | `f6435f98c133a4bfba5d122caf5046d32e09b38d61d67e9c9d54fb8ad47affa7` | `add32ac4b38553726ad79cc207cfbeeeef6f52fda563d83f243235e91373e00a` | `b482ffbf1384fe7bc0d153a903746c3233a0b7e660aa52c4a44ea5f10331091c` | **No** |
| `A-pathmnist-28px-groupnorm-policy-none-s0` | 2 | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | `db274d0aba7d32dc65ee9a6406d0842137602ad15ad9b9657115ff67485520ef` | `db274d0aba7d32dc65ee9a6406d0842137602ad15ad9b9657115ff67485520ef` | **Yes** -- ran after the correction, already current |

## 2. Proof the GroupNorm correction is unreachable from the BatchNorm computation branch

The correction (`git diff e9f061b d5227d0 -- src/`) is exactly 79
lines, 2 files:

- `src/when_tta_hurts/validation_evaluation.py` (+9/-2): changes exactly
  one line,
  `bn_adaptation_microbatch_counts.get(PRIMARY_N)` ->
  `bn_adaptation_microbatch_counts.get(PRIMARY_N, 0)`, plus adds one new
  dict key `"bn_adaptation_applicable": conditions["bn_adapted_tta"] is not None`.
  For a **BatchNorm** cell, `bn_adaptation_microbatch_counts` is
  populated for every prefix in `PREFIX_SEQUENCE` (including
  `PRIMARY_N`) in the unchanged BN-adaptation loop above this line --
  the key always exists, so `.get(PRIMARY_N)` and `.get(PRIMARY_N, 0)`
  return **identical** values. The default `0` only ever fires when the
  key is absent, which happens only for GroupNorm (empty dict, via the
  unchanged `BNAdaptationNotApplicableError` path). The new
  `bn_adaptation_applicable` key is purely additive -- it does not
  change any previously-computed value.
- `src/when_tta_hurts/evaluation_result_artifacts.py` (+72/-0, purely
  additive): adds `_validate_bn_adaptation_applicability_consistency()`,
  a **validation-only** function that raises on inconsistent
  combinations but never computes, transforms, or persists any new
  value. For an already-correctly-formed BatchNorm write (applicable
  `True`, positive microbatch count, `bn_adapted_tta` present,
  `bn_adapted_probs`/`bn_adapted_prefix_sequence` present -- exactly
  what `-s0`/`-s1`/`-s2` already had), this function is a no-op: it
  passes without raising.

**Neither change alters any value a BatchNorm cell computes or
persists.** The GroupNorm correction is only reachable through the
code path GroupNorm cells (and only GroupNorm cells) execute.

## 3. Artifact integrity confirmed unchanged

Independently reverified this session, after the correction was
committed:

| Run ID | `predictions.npz` SHA-256 (unchanged) | `artifact_manifest.json` |
|---|---|---|
| `-s0` attempt 4 | `48b6ff9cf6900853043426ed3381537a84dba29b944670302229008ee1e3ba07` | verified OK |
| `-s1` attempt 1 | `964a79f1b38485e5843d53313d006c2589af1f1aa8b1aaae4fddf215d22588f2` | verified OK |
| `-s2` attempt 1 | `aa63c533109ca62e3b75be06eabe8278c6357007cdb37e19f4f773cc9286eef6` | verified OK |

All three match their originally-recorded hashes exactly. No file was
touched, rewritten, or regenerated.

## 4. Precise state classification

For `A-pathmnist-28px-batchnorm-policy-none-s0` (attempt 4), `-s1`
(attempt 1), and `-s2` (attempt 1):

- **Scientifically valid historical completion**: Yes -- unaffected by
  the GroupNorm correction; the correction's only reachable branch for
  these cells is a byte-for-byte no-op.
- **Canonical-eligible according to the preserved ledger**: Yes -- no
  amendment excludes them; `status=completed`; artifacts intact.
- **Compatible with the current evaluator fingerprint**: **No** --
  their persisted `evaluator_fingerprint`/`evaluation_id` reflect the
  prior (metric-contract-only) correction, not the current
  (GroupNorm-corrected) implementation.
- **Sole completion selected under its original persisted identity**:
  Yes -- `check_evaluation_skip(run_id, <original persisted
  evaluation_id>)` still resolves each to its own completed attempt
  cleanly.
- **Sole completion selected under a freshly-recomputed, current-
  fingerprint identity**: **No** -- confirmed by direct invocation:
  `check_evaluation_skip(run_id, <freshly recomputed evaluation_id>)`
  raises `ConflictingEvaluationImplementationError` for all three,
  reported verbatim:
  ```
  A-pathmnist-28px-batchnorm-policy-none-s0 already has a COMPLETED
  evaluation (attempt(s) [4], evaluation_config_hash(es)
  ['e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4'])
  that does NOT match the current request's evaluation_config_hash=
  fd7094a0ffc95a998e0ae03110c8991f6819a5ca1524036d873658a18c6bdc29.
  ```
  (identical in kind for `-s1`/`-s2` with their own hashes).

For `A-pathmnist-28px-groupnorm-policy-none-s0` (attempt 2): all five
of the above are **Yes** -- it ran after the correction and its
persisted identity already matches the current fingerprint exactly.

## 5. Effect on the remaining 20 Block A cells

None. All 20 have zero prior attempts, so no conflict is possible --
`check_evaluation_skip()` returns `None` for each, cleanly permitting
attempt 1. This drift only concerns re-querying the three already-
completed BatchNorm cells' identity against the current fingerprint; it
does not block, alter, or slow the 20-cell execution in any way.

## 6. Deferred reconciliation

No amendment row, compatibility override, or rerun is performed by this
document. The evaluator implementation is not yet final -- it may
continue to change across the remaining datasets/resolutions/
normalization branches (64px, BloodMNIST, DermaMNIST/ResNet-18, 128px)
still to be evaluated. Reconciling `-s0`/`-s1`/`-s2` now, before that
surface is fully exercised, risks needing to reconcile again after each
subsequent fix -- as already happened once (metric-contract fix, then
GroupNorm fix). The preferred resolution, once the evaluator
implementation is confirmed final across all remaining branches, is to
**rerun these three cells once under the final frozen fingerprint**
for the strongest confirmatory record -- not to invent a compatibility
bypass that would let a completion under an old implementation stand in
for validation under a new one. That decision is explicitly deferred
until the remaining Block A branches have executed successfully.
