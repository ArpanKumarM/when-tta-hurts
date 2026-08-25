# Phase 2B.8A Part H — Controlled Final-Test Unsealing Authorization Audit

**Status: authorizes exactly one future `unseal` operation. Does not
itself unseal, parse, or report any scientific value.** This document
contains no scientific values.

## 1. Construction discipline

`artifacts/final_test_unsealing_authorization.json` was built by hashing
the seven already-committed sealed result files as opaque bytes
(`sha256` of each `analysis_result.json`/`cross_condition_result.json`
and its manifest file) via `resolve_seven_sealed_inputs()` -- the same
metadata-only function `plan` mode uses. At no point during construction
was any result file's JSON content parsed, printed, or examined.

## 2. Bindings

| Field | Value |
|---|---|
| `status` | `approved` |
| `unsealing_freeze_commit` | `486028c3b2e40014a38b3bfc818e113a95d39f9c` |
| `preregistered_results_commit` | `4426bf55476d1761afe15ded4d56d48ded0fee51` |
| `cross_condition_results_commit` | `29a3bfe6c7c7bec4eb131d48a63d8dce51e32412` |
| `generation5_final_test_authorization_sha256` | `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32` |
| `generation3_final_test_analysis_authorization_sha256` | `50d272dd02c1cbb907400fd763e7cc0bd7a07f52670ddba0bd48c660b269f418` |
| `evaluator_fingerprint` | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` |
| `validation_statistical_analysis_fingerprint` | `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01` |
| `cross_condition_addendum_fingerprint` | `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51` |
| `final_test_runner_fingerprint` | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` |
| `final_test_analysis_fingerprint` | `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc` |
| `final_test_reporting_fingerprint` | `baf02763f9b1f3dc5163dbb52069af842aef47de94acfcbca945e2be5d3736be` |

All seven `(kind, identifier, analysis_id, attempt, result_sha256,
manifest_sha256)` bindings are recorded in `sealed_inputs`, mechanically
generated, matching the values already recorded in
`docs/phase2b_final_test_preregistered_analysis_audit.md` and
`docs/phase2b_final_test_cross_condition_analysis_audit.md` exactly.

## 3. Output paths and contract

Exactly three output paths are authorized:
`artifacts/final_test_scientific_summary.json`,
`docs/phase2b_final_test_scientific_results.md`,
`docs/phase2b_final_test_scientific_interpretation.md`. The
`complete_reporting_contract` block records the frozen no-selective-
omission rule, the BLOCK_C-regardless-of-direction rule, and the no-H4/
no-pooled-inference rules from
`docs/phase2b_final_test_unsealing_freeze.md`, verbatim. `authorized_operations`
is exactly `["unseal"]`, `max_unseal_operations` is `1`, and
`no_alternate_input_output_or_configuration_route` is `true` -- matching
the CLI's own no-flags contract.

## 4. Artifact identity

`artifacts/final_test_unsealing_authorization.json` SHA-256:

```
2fa1686fb5443dab58c6ceb7bf7f1e385b20f85ee45f3ec1203d19d363f9dc92
```

## 5. Production verification

`verify_unsealing_authorization()`: `status=approved`. Production `plan`
mode: `unsealing_authorization_status=approved`, `inputs_ready=true`,
`n_inputs=7`. Both re-verified fresh immediately before this document
was written.

## 6. What remains gated

This authorization makes `unseal` runnable, but `unseal` has not been
invoked in this task. Running it -- the actual real generation of the
three scientific output files -- is a separate, explicitly authorized
future step.
