# Phase 2B.7C-Engineering — Sealed Final-Test Analysis CLI: Frozen Design

**Status: FROZEN before any code changes.** This document contains no
scientific values. It governs `scripts/run_final_test_statistical_analysis.py`,
the missing production dispatch identified in Phase 2B.7C Part 1 (no CLI
route existed to `final_test_statistical_analysis.py`'s already-built,
already-tested library functions).

## 1. New production entry point

`scripts/run_final_test_statistical_analysis.py` -- a new file, disjoint
from `scripts/run_statistical_analysis.py` (which remains the
validation-only CLI, byte-for-byte unchanged).

## 2. Exact subcommands

* `plan` -- side-effect-free, metadata-only.
* `analyze-preregistered` -- real analysis, preregistered H1/H2/H3/BLOCK_C
  families only.
* `analyze-cross-condition` -- real analysis, secondary H1/H2/H3 fixed-pair
  cross-condition addendum only.

No other subcommand exists.

## 3. Atomic dispatch, no cross-routing

* `analyze-preregistered` calls `final_test_statistical_analysis.compute_final_test_family_analysis()`
  for every family in `KNOWN_FAMILIES` (H1, H2, H3, BLOCK_C) and nothing
  else. It never imports or calls `compute_final_test_hypothesis_did()`.
* `analyze-cross-condition` calls `compute_final_test_hypothesis_did()`
  for every hypothesis in `KNOWN_HYPOTHESES` (H1, H2, H3) and nothing
  else. It never imports or calls `compute_final_test_family_analysis()`.
* Neither subcommand imports `statistical_analysis.compute_family_analysis`,
  `statistical_analysis.plan_statistical_analysis`,
  `cross_condition_addendum.compute_hypothesis_did`, or
  `cross_condition_addendum.plan_cross_condition_addendum` (all
  validation-stage). This is enforced by an AST-level source-import test
  (Part D.6/16), not merely a docstring claim.
* `plan` reports on BOTH the preregistered and cross-condition sides
  (calling `plan_final_test_statistical_analysis()` and
  `plan_final_test_cross_condition_addendum()`), since it is metadata-only
  and non-destructive -- this is the only subcommand that touches both.

## 4. No scientific configuration surface

The argument parser accepts exactly: the subcommand (`plan` /
`analyze-preregistered` / `analyze-cross-condition`) and nothing else --
no `--run-id`, `--family`, `--hypothesis`, `-n`, `--aggregator`,
`--endpoint`, `--bootstrap-resamples`, `--ci-level`, `--seed`,
`--authorization-path`, `--final-test-root`, `--ledger-path`, `--split`,
`--force`, `--retry`, `--bypass`, `--unseal`, `--print-results`,
`--debug-results`, or any environment-variable read. `KNOWN_FAMILIES` and
`KNOWN_HYPOTHESES` are iterated internally, in full, every time -- a
partial family/hypothesis selection is not expressible from the CLI.
Enforced by an argparse test that asserts every one of these flags is
rejected (`SystemExit` from argparse) and a source-scan test that greps
for `os.environ`/`os.getenv` (must find none).

## 5. `plan`: metadata-only, zero side effects

`plan` calls `plan_final_test_statistical_analysis()` and
`plan_final_test_cross_condition_addendum()` (both already side-effect-
free and already verified in Phase 2B.7A/B to make zero `np.load`
calls) and prints their JSON reports (already sealed -- no scientific
value reachable from either report's schema). No new file, directory, or
ledger row is created by `plan`.

## 6. Both analyze commands: authorization-first, sealed lifecycle

Each analyze subcommand:

1. Resolves the hardcoded final-test-analysis authorization artifact
   path (`artifacts/final_test_analysis_authorization.json`, no
   override) and verifies it BEFORE any prediction load or attempt
   allocation. (Verification wiring for this artifact is added in this
   phase -- see sec.6a below; it did not previously exist as a
   standalone checkable function.)
2. Dispatches through the existing sealed lifecycle functions
   (`compute_final_test_family_analysis()` /
   `compute_final_test_hypothesis_did()`), which already implement:
   atomic attempt allocation (`next_final_test_analysis_attempt_number`),
   failure recording (exceptions propagate before any ledger write),
   semantic verification before `status=completed`
   (`FinalTestAnalysisSemanticVerificationError`, Phase 2B.7B), and
   idempotent short-circuiting (`existing_completed_attempt`).
3. Returns ONLY an allowlisted sealed receipt (sec.7) -- never the full
   internal result dictionary.

### 6a. New: `verify_final_test_analysis_authorization()`

A new, minimal function added to `final_test_statistical_analysis.py`
(the CLI's only source of authorization truth): reads
`artifacts/final_test_analysis_authorization.json`, requires it exists,
is valid JSON, has `status == "approved"`, and its recorded
`final_test_analysis_fingerprint` matches the CURRENT
`compute_final_test_analysis_fingerprint()`. Raises
`FinalTestAnalysisAuthorizationError` (new, in the same module) on any
mismatch. This function performs no git subprocess calls (unlike
`verify_final_test_authorization()`, which is a heavier gate one layer
below) -- it is a fast, CLI-facing content/fingerprint check layered on
top of the fact that `compute_final_test_family_analysis()` /
`compute_final_test_hypothesis_did()` already call
`verify_final_test_authorization()` internally (the FINAL-TEST-EVALUATION
gate) as their own first action. The CLI calls
`verify_final_test_analysis_authorization()` BEFORE calling either real-
analysis function, so an invalid/stale ANALYSIS authorization is rejected
even earlier than the pre-existing evaluation-authorization check inside
the library functions.

## 7. Allowed CLI output fields (allowlist, both directions)

Success receipt (JSON): `command`, `mode`, `status` (`"completed"` or
`"skipped_idempotent"`), `analysis_id` (or per-family/per-hypothesis
`analysis_ids` for a multi-unit run), `attempt`, `runtime_seconds`,
`n_inputs_required`, `n_inputs_resolved`, artifact `filenames`/`sizes`/
`sha256` hashes, `manifest_verification` (`"PASS"`/`"FAIL"`),
`semantic_verification` (`"PASS"`/`"FAIL"`). Nothing else.

## 8. Forbidden CLI output (never printed, never returned to stdout/stderr)

Accuracy/any metric value, effect estimates, CI bounds, raw/adjusted
p-values, significance decisions, harm/rescue rates, per-cell outcomes,
rankings, scientific comparisons/conclusions, or the complete internal
result dictionary (`per_cell_statistics`, `per_pair_results`,
`multiplicity`, `bootstrap`, `mcnemar`, `effect_sizes` keys and
everything nested under them are never serialized to stdout/stderr by
this CLI).

## 9. Error sealing

On any exception from the dispatched library call, the CLI catches
exactly the known, allowlisted exception types --
`FinalTestAnalysisAuthorizationError` (new, sec.6a),
`FinalTestAuthorizationError` (from `final_test_authorization.py`),
`FinalTestAnalysisInputError`, `FinalTestAnalysisSemanticVerificationError`,
`FinalTestAnalysisLedgerConflictError` (from `final_test_analysis_ledger.py`)
-- and prints only: the exception's CLASS NAME (an allowlisted lifecycle-
stage token, never the raw `str(exception)`, since several of these
messages are explicitly designed to be human-debuggable and could in
principle echo an input identity or a stale-fingerprint value, though
never a metric), plus (where available) the opaque `run_id`/`pair_id`/
`analysis_id` already known to the CLI from its own dispatch loop before
the failure. Any OTHER, unexpected exception type is re-raised after
printing only its class name and exit code `1` -- never its message body
-- failing loudly but without risking a leaked numeric value. A failed
attempt is never marked `completed`; this is enforced by the underlying
library functions already (Phase 2B.7A/B), not reimplemented here.

## 10. Fingerprint manifest change

`scripts/run_final_test_statistical_analysis.py` is added to
`FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST` in
`final_test_statistical_analysis.py`. This file's content can change
which analysis is dispatched, how authorization is checked, and what
output is sealed -- all analysis-identity/lifecycle-relevant concerns --
so it belongs in the SAME manifest as `final_test_statistical_analysis.py`
and `final_test_analysis_ledger.py`, and in NO other manifest.
Specifically:

* It is NOT added to `ANALYSIS_FINGERPRINT_MANIFEST` (validation-mode) --
  the CLI never touches validation-stage code paths.
* It is NOT added to `CROSS_CONDITION_ADDENDUM_MANIFEST` or
  `FINAL_TEST_RUNNER_MANIFEST` directly -- those are transitively
  included INTO `FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST` already (see
  `final_test_statistical_analysis.py`'s existing
  `FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST = FINAL_TEST_RUNNER_MANIFEST + (...)`
  definition); adding the CLI file directly to either of those lower-
  level manifests would cascade into the evaluator/cross-condition/
  runner fingerprints and invalidate the generation-5 FINAL-TEST
  authorization itself -- exactly the mistake Phase 2B.7A's `ledger.py`
  lesson exists to prevent. Adding it only to
  `FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST` changes ONLY
  `final_test_analysis_fingerprint`, leaving evaluator/validation-
  analysis/cross-condition/runner/reconciliation fingerprints
  untouched -- verified mechanically in Part E before any reauthorization
  decision is made.

## 11. What this phase does not do

This phase does not execute `analyze-preregistered` or
`analyze-cross-condition` against real data. It does not create
`artifacts/final_test_analysis/`, `artifacts/final_test_cross_condition/`,
or `artifacts/ledger_final_test_analysis.csv` in the real repository. Real
execution remains a further, separately-authorized step (Phase 2B.7C
Part 1, resumed only after this CLI exists and is reauthorized).
