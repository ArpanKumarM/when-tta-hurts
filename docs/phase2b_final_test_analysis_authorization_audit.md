# Phase 2B.7A — Final-Test Analysis Authorization Audit

**Status: authorizes the ENGINEERING (plan/analyze-mode functions) built
in Phase 2B.7A to be pointed at real final-test data by a future, still-
separately-gated task. It does NOT itself run, unseal, or produce any
real analysis result.** This document contains no scientific values.

## 1. What this authorizes and what it does not

`artifacts/final_test_analysis_authorization.json` is the tracked gate
artifact for `final_test_statistical_analysis.py`'s real-analysis
functions (`compute_final_test_family_analysis`,
`compute_final_test_hypothesis_did`), mirroring the discipline of the
final-test-evaluation gate (`artifacts/final_test_authorization.json`)
one layer up the stack. Its existence and `status: "approved"` are a
necessary precondition for a later, separately-authorized real-analysis
run -- they are not sufficient on their own, and this task does not
invoke either real-analysis function against real repository data.
Creating this artifact is itself the action authorized by the current
task's instructions; running real analysis against it remains a further,
separately-authorized step.

## 2. Bindings

| Field | Value |
|---|---|
| `final_test_closure_commit` | `581143e6d1c080c3bfaad941514a356089313926` (`results: close Phase 2B final-test matrix`) |
| `generation5_final_test_authorization_sha256` | `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32` |
| `generation5_final_test_authorization_commit` | `6d68da1dc34c276374cdab133a03b171b8b45381` |
| `evaluator_fingerprint` | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` |
| `final_test_runner_fingerprint` | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` |
| `final_test_analysis_fingerprint` | `d720d0dbf147804b9cb11e7c1e79bb5b9166fe69fb0f20373418198bb73dad0a` |
| `final_test_cross_condition_analysis_fingerprint` | `d720d0dbf147804b9cb11e7c1e79bb5b9166fe69fb0f20373418198bb73dad0a` (identical to the preregistered-analysis fingerprint -- both plan/analyze pairs share exactly `FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST`, since the addendum math and the family math live in the same new module) |
| `validation_reconciliation_implementation_fingerprint` | `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3` |
| `frozen_statistical_analysis_plan_sha256` (`docs/statistical_analysis_plan.md`) | `566840a15e11d3fafe4aa781e705e2c8ac005dd21c5c79c93da03bcb74b69fca` |
| `frozen_cross_condition_addendum_config_sha256` (`configs/final_test_cross_condition_addendum.yaml`) | `bf2f1a260c2906d974659f65434babb8a3196fa5ca70948b84219edca36abfc5` |

`frozen_endpoint`: `metric=accuracy`, `condition=naive_tta`,
`aggregator=mean_probability`, `tta_view_count=50` -- the single frozen
primary endpoint, unchanged from the validation-mode contract.

The artifact's own content hash is
`6b94796df62d3778c2a7f566c6e53f2b98d6483442fa299cdfe358219b585db5`
(SHA-256 of `artifacts/final_test_analysis_authorization.json` as
committed).

## 3. Exact 39-cell binding

`authorized_cells` in the artifact lists all 39 final-test run_ids, each
with its authorized `evaluation_attempt`, `final_test_evaluation_id`, and
`primary_artifact_hash`, read directly from the closed
`artifacts/ledger_final_test.csv` (42 total rows: 39 completed / 2 failed
/ 1 aborted, per `docs/phase2b_final_test_matrix_closure_audit.md`).
Cell 1 (`A-pathmnist-28px-batchnorm-policy-none-s0`) is bound at attempt
3; cell 2 (`A-pathmnist-28px-batchnorm-policy-none-s1`) at attempt 2;
every other cell at attempt 1 -- matching the closure audit's mapping
exactly, byte-for-byte pulled from the ledger rather than retyped.

## 4. Production plan-mode verification (re-run fresh for this audit)

`plan_final_test_statistical_analysis()` and
`plan_final_test_cross_condition_addendum()` were re-invoked against the
real, current repository state as part of this audit (read-only,
metadata-only, zero prediction loads -- verified by the module's own
test suite's `numpy.load`-tracking tests):

* `authorization_status`: `"approved"`.
* `n_cells_total`: `39`; `n_completed_consumed`: `39`; `n_pending`: `0`;
  `n_invalid`: `0`.
* Preregistered families: H1 requires 24 cells (complete), H2 requires
  30 cells (complete), H3 requires 12 cells (complete), BLOCK_C requires
  3 cells (complete). All four `complete: true`.
* Cross-condition addendum pairs: H1 requires 12 pairs (complete), H2
  requires 12 pairs (complete), H3 requires 6 pairs (complete). All
  three `complete: true`.
* Zero result artifacts exist anywhere under
  `artifacts/final_test_analysis/` or
  `artifacts/final_test_cross_condition/`, and
  `artifacts/ledger_final_test_analysis.csv` does not exist -- confirmed
  by direct filesystem check before and after this audit's plan-mode
  calls.

## 5. What remains gated

Real analysis (`compute_final_test_family_analysis`,
`compute_final_test_hypothesis_did`) is fully implemented and tested
against synthetic fixtures only (Phase 2B.7A engineering commit
`c21f9cc`). No production verifier yet checks this new authorization
artifact before allowing a real-analysis call to proceed -- building
that check, and then separately authorizing and running real analysis
against the 39-cell final-test matrix, are both future, explicitly
separate steps. This task creates the tracked gate artifact and its
audit trail only.
