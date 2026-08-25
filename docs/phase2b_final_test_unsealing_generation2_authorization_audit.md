# Phase 2B.8C Part D — Final-Test Unsealing Generation-2 Authorization Audit

**Status: authorizes exactly one generation-2 `unseal` operation.** This
document contains no scientific values.

## 1. Why reauthorization was required

Phase 2B.8C Part C corrected exactly one sentence in
`render_interpretation_markdown()` (commit
`990e5b5e464fb08f5b95a1a3dffc452613e3d925`). Since this file is a member
of `FINAL_TEST_REPORTING_MANIFEST`, this changed
`final_test_reporting_fingerprint`:

| Fingerprint | Before (generation 1, bound in `3804bac`) | After (generation 2, this document) |
|---|---|---|
| `final_test_reporting_fingerprint` | `baf02763f9b1f3dc5163dbb52069af842aef47de94acfcbca945e2be5d3736be` | `2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026` |

This makes the generation-1 unsealing authorization (committed at
`3804bac0b10e6e4c9209ae1dabf944051ef67cbf`, SHA-256
`2fa1686fb5443dab58c6ceb7bf7f1e385b20f85ee45f3ec1203d19d363f9dc92`) stale.
It is superseded here, never silently reused.

## 2. All other fingerprints and authorizations confirmed unchanged

| Fingerprint / authorization | Value | Changed? |
|---|---|---|
| Evaluator | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` | No |
| Validation statistical analysis | `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01` | No |
| Cross-condition addendum | `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51` | No |
| Final-test runner | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` | No |
| Validation reconciliation | `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3` | No |
| Final-test analysis | `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc` | No |
| Generation-5 final-test authorization | `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32` | No |
| Generation-3 final-test-analysis authorization | `50d272dd02c1cbb907400fd763e7cc0bd7a07f52670ddba0bd48c660b269f418` | No |

Confirmed by full quality-suite run (1136 tests passed) and fresh
fingerprint computation immediately after the correction commit.

## 3. Supersession chain

Generation 1 (`3804bac0b10e6e4c9209ae1dabf944051ef67cbf`, SHA-256
`2fa1686fb5443dab58c6ceb7bf7f1e385b20f85ee45f3ec1203d19d363f9dc92`) is
preserved unmodified in git history. Generation 2 (this authorization,
schema `phase2b.8c-v2`) records `supersedes_authorization_sha256` /
`supersedes_authorization_commit` pointing at generation 1's exact
historical content, independently re-verified via `git show
3804bac0b10e6e4c9209ae1dabf944051ef67cbf:artifacts/final_test_unsealing_authorization.json
| sha256sum`, which reproduces `2fa1686f...` exactly.

## 4. Bindings

Binds, unchanged from generation 1: `unsealing_freeze_commit` (`486028c`),
`preregistered_results_commit` (`4426bf5`),
`cross_condition_results_commit` (`29a3bfe`), the generation-5 and
generation-3 authorizations, and the seven `sealed_inputs` (identical
analysis IDs/attempts/result hashes/manifest hashes to generation 1).

New in generation 2: `wording_correction_freeze_commit` (`3d8ba37`),
`generation1_preservation_commit` (`447dfe0`),
`correction_implementation_commit` (`990e5b5`), the new
`final_test_reporting_fingerprint`
(`2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026`), and
`generation1_output_hashes` (the three original generation-1 hashes,
recorded so a future comparison can verify the allowed-difference set
mechanically).

`authorized_operations` is exactly `["unseal"]`, `max_unseal_operations`
is `1`.

## 5. Permitted output-content difference

The only permitted difference between generation-1 and generation-2
output content is: (a) reporting provenance caused by the new
authorization/fingerprint being embedded in the summary JSON, and (b) the
one frozen sentence corrected in
`docs/phase2b_final_test_reporting_wording_correction_freeze.md`. No
scientific value, table, count, effect, interval, p-value, adjusted
p-value, classification, limitation, or conclusion may differ. This is
verified mechanically in Phase 2B.8C Part G before generation 2 may be
committed.

## 6. Artifact identity

`artifacts/final_test_unsealing_authorization.json` SHA-256 (as
committed):

```
cdeb4227178d0afa8264f9385e35c5a3bb94c8a83b23245f9a26071806d86977
```

## 7. Production verification

`verify_unsealing_authorization()`: `status=approved`, confirmed fresh
immediately before this document was written.

## 8. What remains gated

This authorization makes exactly one generation-2 `unseal` invocation
runnable. It has not been invoked in this task.
