# Phase 2B.7B Part D — Final-Test Analysis Reauthorization Audit

**Status: reauthorizes the sealed final-test analysis engineering after
a fingerprint change. Does not authorize or run real analysis against
real data.** This document contains no scientific values.

## 1. Why reauthorization was required

Phase 2B.7B Part C's engineering (commit `b4cdf5b7c90cf7060ac8590621feed118b5c338c`,
`feat: verify and seal final-test statistical analysis`) modified
`src/when_tta_hurts/final_test_statistical_analysis.py` and
`src/when_tta_hurts/final_test_analysis_ledger.py` -- both members of
`FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST`. This changed
`final_test_analysis_fingerprint`:

| Fingerprint | Before (generation 1, bound in `68db8c5`) | After (generation 2, this document) |
|---|---|---|
| `final_test_analysis_fingerprint` | `d720d0dbf147804b9cb11e7c1e79bb5b9166fe69fb0f20373418198bb73dad0a` | `bbe529eca523b44ecc149df086b9330950827cba94719ec80c6ad23eea4b453f` |

Per Phase 2B.7A's own frozen design and Phase 2B.7B Part B sec.9, this
makes the generation-1 analysis authorization
(`artifacts/final_test_analysis_authorization.json` as committed at
`68db8c5f18fb5de414b28a4a8bcd5255f6e542b4`, SHA-256
`6b94796df62d3778c2a7f566c6e53f2b98d6483442fa299cdfe358219b585db5`)
stale. It is superseded here, never silently reused.

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
commit `6d68da1dc34c276374cdab133a03b171b8b45381`) remains valid and was
re-verified successfully (`verify_final_test_authorization()` returns
`status=approved`, `39/39 completed_consumed`) -- no cascade into the
final-test-evaluation layer occurred, exactly as the Phase 2B.7A lesson
(keeping `ledger.py` untouched) was designed to prevent.

## 3. Supersession chain

Generation 1 (`68db8c5f18fb5de414b28a4a8bcd5255f6e542b4`, SHA-256
`6b94796df62d3778c2a7f566c6e53f2b98d6483442fa299cdfe358219b585db5`) is
preserved unmodified in git history -- never rewritten, never deleted.
Generation 2 (this authorization, schema `phase2b.7b-v2`) explicitly
records `supersedes_authorization_sha256` and
`supersedes_authorization_commit` pointing at generation 1's exact
historical content, independently re-verified via `git show
68db8c5f18fb5de414b28a4a8bcd5255f6e542b4:artifacts/final_test_analysis_authorization.json
| sha256sum`, which reproduces `6b94796d...` exactly.

## 4. Generation-2 bindings

| Field | Value |
|---|---|
| `final_test_closure_commit` | `581143e6d1c080c3bfaad941514a356089313926` (unchanged from generation 1) |
| `generation5_final_test_authorization_sha256` / `_commit` | `1e217e7e.../6d68da1...` (unchanged) |
| `evaluator_fingerprint` | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` (unchanged) |
| `final_test_runner_fingerprint` | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` (unchanged) |
| `final_test_analysis_fingerprint` / `final_test_cross_condition_analysis_fingerprint` | `bbe529eca523b44ecc149df086b9330950827cba94719ec80c6ad23eea4b453f` (**new**) |
| `validation_reconciliation_implementation_fingerprint` | `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3` (unchanged) |
| `frozen_statistical_analysis_plan_sha256` | `566840a15e11d3fafe4aa781e705e2c8ac005dd21c5c79c93da03bcb74b69fca` (unchanged) |
| `frozen_cross_condition_addendum_config_sha256` | `bf2f1a260c2906d974659f65434babb8a3196fa5ca70948b84219edca36abfc5` (unchanged) |

`authorized_analyses`: exactly
`["final_test_preregistered_statistical_analysis", "final_test_cross_condition_addendum"]`
-- nothing else. `authorized_cells`: the same 39 run_ids/attempts/
evaluation IDs/artifact hashes as generation 1, re-read fresh from
`artifacts/ledger_final_test.csv` (unchanged by this phase).

Generation-2 artifact SHA-256 (`artifacts/final_test_analysis_authorization.json`
as committed):

```
51152a79a044a44cec781bac10abd51b9ea64de6e0b4c623392293c60bedccaf
```

## 5. What remains gated

Identical to generation 1: this authorization is a necessary but not
sufficient precondition for a future, separately-authorized real
analysis run. No production verifier yet checks this authorization
artifact before allowing `compute_final_test_family_analysis()` or
`compute_final_test_hypothesis_did()` to proceed against real data --
building that check, and separately authorizing and running real
analysis, remain future, explicitly separate steps. This task only
creates the reauthorized gate artifact and its audit trail.