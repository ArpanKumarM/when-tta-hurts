# Phase 2B.7B — Sealed Final-Test Analysis Verification: Frozen Design

**Status: FROZEN before any code changes.** This document contains no
scientific values. It governs the minimum missing wiring identified in
Phase 2B.7B Part A: the Phase 2B.7A real-analysis paths
(`compute_final_test_family_analysis`, `compute_final_test_hypothesis_did`)
validate schema and artifact-manifest integrity before persistence, but
never independently recompute the statistics they are about to persist
-- unlike the final-test EVALUATION path's `_verify_metrics_semantically`
(Phase 2B.6J). The preregistered path's bootstrap also uses an unseeded
`np.random.default_rng()` by default, which is incompatible with
independent recomputability. Both gaps are closed here.

## 1. Production entry points (unchanged from Phase 2B.7A)

* Preregistered final-test statistical analysis: plan mode
  `plan_final_test_statistical_analysis()`; real mode
  `compute_final_test_family_analysis()`.
* Secondary final-test cross-condition addendum: plan mode
  `plan_final_test_cross_condition_addendum()`; real mode
  `compute_final_test_hypothesis_did()` / `compute_final_test_pair_did()`.

No new entry point is added. No existing entry point's public contract
(parameter names, default paths) changes.

## 2. Authorization-before-prediction-load ordering (unchanged, reaffirmed)

Both real-mode functions resolve `authorization` (via
`verify_final_test_authorization()` or the test-only `_authorization`
seam) as their FIRST action, before any cell/pair identity resolution,
before any `np.load` call. Per-cell/per-pair identity is re-resolved
immediately before its own `np.load` call
(`_load_final_test_cell_correctness`), so a single ineligible cell can
never be masked by an earlier eligible one. This ordering is verified by
a dedicated test that monkeypatches `numpy.load` to raise if reached
before authorization/identity resolution completes for that cell.

## 3. Input resolution and final-test-only requirements (unchanged, reaffirmed)

Every real/plan function reads exclusively from `artifacts/final_test/`
and `artifacts/ledger_final_test.csv` (via
`resolve_final_test_canonical_evaluation_identity`). No function in
`final_test_statistical_analysis.py` accepts a validation-root override
parameter (enforced by an existing test,
`test_no_public_function_accepts_a_validation_root_or_split_override`).
Validation-evaluation artifacts and `validation_metric_reconciliation.py`
remain reachable ONLY as historical provenance narration (never as
computation input) -- no function added by this phase changes that.

## 4. Atomic attempt allocation, failure recording, persistence, manifest, completion-ledger ordering

Unchanged from Phase 2B.7A, reaffirmed and now covered by an explicit
failure-preservation test:

1. Resolve authorization and every required cell/pair's identity
   (cheap, metadata-only). Any ineligible input raises
   `FinalTestAnalysisInputError` BEFORE any attempt directory is
   created, before any ledger row is written, before any prediction is
   loaded for ANY cell in the family/hypothesis (fail-closed on the
   whole unit, never a partial family).
2. Compute the analysis_id from resolved identities. Check
   `existing_completed_attempt()` (see sec.8) -- short-circuits without
   recomputation if a prior completion exists.
3. Load predictions and compute statistics for every cell/pair.
4. **NEW (this phase): independently re-verify every computed statistic
   from the persisted input bindings (sec.6/7) BEFORE persistence.**
5. `attempt_dir = <ROOT>/<family_or_hypothesis>/attempt_{N:03d}`, where N
   is `next_final_test_analysis_attempt_number(analysis_id)` -- an
   append-only, ledger-derived allocation, never a caller-supplied
   number.
6. Persist atomically via the EXISTING, unmodified
   `persist_and_verify_analysis_completion()` /
   `persist_and_verify_cross_condition_completion()` (schema + manifest
   verification).
7. Append exactly one ledger row via `append_final_test_analysis_entry()`
   only after step 6 succeeds. A conflicting duplicate (same
   analysis_id + attempt, different content) is a hard failure
   (`FinalTestAnalysisLedgerConflictError`), never a silent overwrite.
8. Any failure at any step leaves no ledger row and no directory
   falsely claiming `status=completed` -- verified by
   `test_failure_never_produces_a_completed_analysis_result` (already
   existing) and a new interrupted-attempt test (sec.9 below covers the
   new cases this phase adds).

## 5. Sealed-output contract

No function in `final_test_statistical_analysis.py`, and no CLI surface
built on top of it, ever prints or returns from a **plan**, **failure**,
or **idempotent-skip** code path any of: effect estimate, confidence
interval, p-value, corrected p-value, significance indicator, metric
value, ranking, or conclusion. Plan-mode, authorization-verification,
and failure paths may emit only: lifecycle status tokens (`approved`,
`not_approved`, `eligible`, `not_completed_consumed`,
`unauthorized_binding`, `unknown_run_id`, `completed`, `failed`),
counts, opaque IDs (`run_id`, `pair_id`, `evaluation_id`, `analysis_id`),
hashes, and file paths.

A **real-analysis** call's return value and persisted
`analysis_result.json` / `cross_condition_result.json` DO legitimately
contain the real statistics once computed -- that is the artifact's
purpose. This phase does not change that. What this phase adds is: no
engineering-task test, script, or log in this repository ever prints,
inspects, ranks, summarizes, or otherwise unseals those values (enforced
by `test_plan_report_json_never_contains_forbidden_scientific_keys`,
extended in this phase to also cover verify-mode and idempotent-skip
outputs), and no real-analysis function is invoked against real
repository data anywhere in Phase 2B.7B. Persisted scientific results
remain sealed pending a later, explicitly separate unsealing task.

## 6. Semantic verification before `status=completed` -- preregistered family analysis

Before calling `persist_and_verify_analysis_completion()`,
`compute_final_test_family_analysis()` independently recomputes, per
cell, from a FRESH, independent reload of that cell's persisted
`predictions.npz` (via `_load_final_test_cell_correctness`, the exact
same function used for the original computation -- reused, not forked,
since the goal is to catch a stale-variable/mutation bug between compute
and persist, not to re-derive the aggregation formula a second way):

* the paired bootstrap CI, using the EXACT deterministic seed the
  original computation used (see below) -- exact floating-point
  equality is required (both calls use the identical seeded
  `np.random.default_rng`, so results must match bit-for-bit, not just
  within tolerance);
* the McNemar result (deterministic, no RNG -- exact equality);
* the effect sizes (deterministic -- exact equality);
* the Benjamini-Hochberg correction across the family's raw p-values
  (deterministic -- exact equality).

**Deterministic bootstrap seeding (new requirement):** the primary
bootstrap for a real final-test family analysis MUST use a derived,
reproducible seed, never an unseeded `np.random.default_rng()`. The
seed is `derive_final_test_bootstrap_seed(family, run_id,
final_test_analysis_fingerprint)` -- a pure SHA-256-derived uint64,
exactly mirroring `cross_condition_addendum.derive_bootstrap_seed()`'s
existing discipline. The derived seed is stored in the persisted
per-cell result (`bootstrap.bootstrap_seed`) so a THIRD PARTY, given
only the persisted artifact and the persisted input bindings, can
independently reproduce the exact bootstrap CI without any access to
this repository's runtime state.

A recomputation mismatch raises `FinalTestAnalysisSemanticVerificationError`
before any write -- the attempt is never persisted, no ledger row is
appended, exactly like a schema or manifest failure.

## 7. Semantic verification before `status=completed` -- cross-condition addendum

`compute_final_test_hypothesis_did()` already uses a deterministic
`derive_bootstrap_seed(hypothesis, pair_id, analysis_fingerprint)` for
every pair (inherited unchanged from `cross_condition_addendum.py`).
Before calling `persist_and_verify_cross_condition_completion()`, it
independently recomputes, per pair, from a fresh reload of both
conditions' `predictions.npz`:

* the label/sample-index equality check (already fail-closed at compute
  time -- reaffirmed, not reimplemented);
* the DiD point estimate and bootstrap CI, using the joint four-array
  resampling procedure (`did_bootstrap_ci`) with the exact same derived
  seed -- exact floating-point equality required.

A mismatch raises `FinalTestAnalysisSemanticVerificationError` before
any write, identically to sec.6.

## 8. Idempotent-skip and conflicting-completion behavior

`existing_completed_attempt(analysis_id)` must return the attempt number
of the SOLE completed row for that `analysis_id`, or `None` if none
exists. **New requirement (closes a real gap found in Phase 2B.7B Part
A):** if MORE THAN ONE completed row exists for the same `analysis_id`
(which should be structurally impossible under normal operation, since
the idempotent short-circuit prevents a second real computation once one
exists, but must never be silently tolerated if it occurs through ledger
corruption, manual editing, or a future bug), this is `ambiguous` and
must raise `FinalTestAnalysisLedgerConflictError` rather than silently
returning an arbitrary one of the conflicting attempts. This mirrors
every other "ambiguous -> fail closed, never guess" rule already
established for validation/final-test evaluation identity resolution
(`_resolve_canonical_evaluation_identity`, `_classify_final_test_cell`).

A genuinely identical re-append (same analysis_id, same attempt, byte-
identical row) remains `"duplicate_ignored"` at the ledger-append layer,
unchanged from Phase 2B.7A.

## 9. Fingerprint and reauthorization consequences

`final_test_statistical_analysis.py` and any new sibling module this
phase adds are already members of `FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST`
(or must be added to it if new files are created), so ANY code change
made under this phase changes `final_test_analysis_fingerprint`. This is
expected and is not itself a defect -- Phase 2B.7A's authorization
(`68db8c5`, artifact SHA-256
`6b94796df62d3778c2a7f566c6e53f2b98d6483442fa299cdfe358219b585db5`) binds
the PRE-2B.7B `final_test_analysis_fingerprint`
(`d720d0dbf147804b9cb11e7c1e79bb5b9166fe69fb0f20373418198bb73dad0a`), so
if this phase's engineering changes that fingerprint, the existing
authorization becomes stale and MUST be superseded by a new generation
(never silently reused, never silently ignored) before any real analysis
may proceed. `ledger.py` remains untouched throughout this phase (per
the Phase 2B.7A lesson: it is a member of
`CROSS_CONDITION_ADDENDUM_MANIFEST`/`FINAL_TEST_RUNNER_MANIFEST`, so
editing it would cascade into the evaluator/cross-condition/runner
fingerprints and invalidate the generation-5 FINAL-TEST authorization
itself, a strictly larger blast radius that this phase must not risk).
The evaluator, validation-statistical-analysis, cross-condition,
reconciliation, and final-test-runner fingerprints are expected to
remain UNCHANGED by this phase -- verified mechanically immediately
after every commit in Part D.
