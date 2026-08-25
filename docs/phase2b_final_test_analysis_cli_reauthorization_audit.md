# Phase 2B.7C-Engineering Part E — Final-Test Analysis Reauthorization After CLI Wiring

**Status: reauthorizes the sealed final-test analysis engineering after
the CLI-wiring commit changed its fingerprint. Does not authorize or run
real analysis against real data.** This document contains no scientific
values.

## 1. Why reauthorization was required

Phase 2B.7C-Engineering Part C (commit
`bb5112c4f7ef6fda3e2f9cf10c770728bfc8d606`, `feat: add sealed final-test
analysis CLI`) added `scripts/run_final_test_statistical_analysis.py` to
`FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST` and added
`verify_final_test_analysis_authorization()`/
`FinalTestAnalysisAuthorizationError` to
`final_test_statistical_analysis.py`. This changed
`final_test_analysis_fingerprint`:

| Fingerprint | Before (generation 2, bound in `9568d4d`) | After (generation 3, this document) |
|---|---|---|
| `final_test_analysis_fingerprint` | `bbe529eca523b44ecc149df086b9330950827cba94719ec80c6ad23eea4b453f` | `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc` |

This makes the generation-2 analysis authorization (committed at
`9568d4d30db642a37fbcfc87ac01c81f9f68b11c`, SHA-256
`51152a79a044a44cec781bac10abd51b9ea64de6e0b4c623392293c60bedccaf`) stale.
It is superseded here, never silently reused -- confirmed empirically:
attempting `analyze-preregistered` against the stale generation-2
authorization (before this reauthorization) correctly raised
`FinalTestAnalysisAuthorizationError` and created zero artifacts.

## 2. All other fingerprints confirmed unchanged

Recomputed fresh immediately after the Part C commit:

| Fingerprint | Value | Changed? |
|---|---|---|
| Evaluator | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` | No |
| Validation statistical analysis | `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01` | No |
| Cross-condition addendum | `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51` | No |
| Final-test runner | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` | No |
| Validation reconciliation | `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3` | No |

Because these are unchanged, the **generation-5 final-test authorization**
(`1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32`,
commit `6d68da1dc34c276374cdab133a03b171b8b45381`) remains valid --
re-verified: `status=approved`, `39/39 completed_consumed`.

## 3. Supersession chain

Generation 1 (`68db8c5f18fb5de414b28a4a8bcd5255f6e542b4`, SHA-256
`6b94796df62d3778c2a7f566c6e53f2b98d6483442fa299cdfe358219b585db5`) and
generation 2 (`9568d4d30db642a37fbcfc87ac01c81f9f68b11c`, SHA-256
`51152a79a044a44cec781bac10abd51b9ea64de6e0b4c623392293c60bedccaf`) are
both preserved unmodified in git history. Generation 3 (this
authorization, schema `phase2b.7c-v3`) records
`supersedes_authorization_sha256`/`supersedes_authorization_commit`
pointing at generation 2's exact historical content, independently
re-verified via `git show 9568d4d30db642a37fbcfc87ac01c81f9f68b11c:artifacts/final_test_analysis_authorization.json
| sha256sum`, which reproduces `51152a79...` exactly.

## 4. Generation-3 bindings

| Field | Value |
|---|---|
| `final_test_closure_commit` | `581143e6d1c080c3bfaad941514a356089313926` (unchanged) |
| `generation5_final_test_authorization_sha256` / `_commit` | `1e217e7e.../6d68da1...` (unchanged) |
| `evaluator_fingerprint` | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` (unchanged) |
| `final_test_runner_fingerprint` | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` (unchanged) |
| `final_test_analysis_fingerprint` / `final_test_cross_condition_analysis_fingerprint` | `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc` (**new**) |
| `validation_reconciliation_implementation_fingerprint` | `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3` (unchanged) |
| `frozen_statistical_analysis_plan_sha256` | `566840a15e11d3fafe4aa781e705e2c8ac005dd21c5c79c93da03bcb74b69fca` (unchanged) |
| `frozen_cross_condition_addendum_config_sha256` | `bf2f1a260c2906d974659f65434babb8a3196fa5ca70948b84219edca36abfc5` (unchanged) |

`authorized_analyses`: exactly `["analyze-preregistered",
"analyze-cross-condition"]` -- the two CLI subcommand names, nothing
else, no bypass or alternative analysis path. `authorized_cells`: the
same 39 run_ids/attempts/evaluation IDs/artifact hashes as generations 1
and 2, re-read fresh from `artifacts/ledger_final_test.csv` (unchanged
by this phase).

Generation-3 artifact SHA-256 (`artifacts/final_test_analysis_authorization.json`
as committed):

```
50d272dd02c1cbb907400fd763e7cc0bd7a07f52670ddba0bd48c660b269f418
```

## 5. Production plan-mode re-verification

`verify_final_test_analysis_authorization()`: `status=approved`.
`plan_final_test_statistical_analysis()`: `authorization_status=approved`,
`n_completed_consumed=39`, all four families (H1/H2/H3/BLOCK_C)
`complete=true`. `plan_final_test_cross_condition_addendum()`:
`authorization_status=approved`, all three hypotheses (H1/H2/H3)
`complete=true`. Both calls made zero `numpy.load` calls and created zero
new files/directories (verified by direct filesystem comparison before
and after).

## 6. What remains gated

This authorization is a necessary but not sufficient precondition for a
future, separately-authorized real analysis run via `analyze-preregistered`
or `analyze-cross-condition`. This task creates only the reauthorized
gate artifact and its audit trail -- no real analysis was executed.
