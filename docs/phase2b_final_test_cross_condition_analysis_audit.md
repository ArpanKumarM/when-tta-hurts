# Phase 2B.7E Part B — Sealed Final-Test Cross-Condition Analysis Audit

**Status: records the completed, sealed secondary cross-condition
final-test analysis as a tracked research record.** This document
contains no scientific values -- no difference-in-differences estimate,
confidence-interval bound, raw or adjusted p-value, significance
language, metric value, pair-level outcome, ranking, comparison, or
conclusion. The persisted results remain sealed pending the controlled
unsealing procedure frozen separately in this phase.

## 1. Execution

Command:

```
uv run python3 scripts/run_final_test_statistical_analysis.py analyze-cross-condition
```

Exit code: `0`. Runtime: `32.764s` (`29.54s` user, `2.58s` system).
Single real invocation (Phase 2B.7D Part D).

## 2. Opaque analysis IDs and attempts

| Hypothesis | Analysis ID | Attempt | Pairs |
|---|---|---:|---:|
| H1 | `d9e335e0f28b578631975e83442480e59055edab2d2ba340a71a70f44bc93b7f` | 1 | 12 |
| H2 | `5645a6cd96f414c173937a27fcc133c3bc5547eec454fae1787aa14bc6edd09c` | 1 | 12 |
| H3 | `5bc1e82cc6a07932b0b736c6f8d37396a47ba0be94e59632a1d36d764ddcd6a7` | 1 | 6 |

Pair counts (12/12/6) mechanically re-verified against
`derive_fixed_pairs()`'s current matrix-derived membership.

## 3. Authorization and fingerprint bindings

* Generation-5 final-test authorization: `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32`
  (commit `6d68da1dc34c276374cdab133a03b171b8b45381`).
* Generation-3 final-test-analysis authorization: `50d272dd02c1cbb907400fd763e7cc0bd7a07f52670ddba0bd48c660b269f418`
  (commit `26c9db04134b3314492c1f04f527c7c02dc1ebc8`).
* Evaluator fingerprint: `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2`.
* Validation statistical-analysis fingerprint: `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01`.
* Cross-condition addendum fingerprint: `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51`.
* Final-test runner fingerprint: `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834`.
* Final-test-analysis fingerprint: `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc`.

All confirmed unchanged before and after this commit.

## 4. Endpoint/resampling provenance

Frozen fixed-model secondary-addendum contract
(`configs/final_test_cross_condition_addendum.yaml`, re-validated at
analysis time by `load_addendum_spec()`, which fails closed on any
drift): `metric=accuracy`, `condition=naive_tta`,
`aggregator=mean_probability`, `tta_view_count=50`; bootstrap
`n_resamples=10000`, `ci_level=0.95`, joint four-array resampling
(`independent_resampling_of_four_arrays: forbidden`), deterministic
per-pair seed (`derive_bootstrap_seed(hypothesis, pair_id,
final_test_analysis_fingerprint)`); Block D 128px excluded from H2
inference (`block_d_128px_included_in_inference: false`). Every
persisted per-pair `bootstrap_seed` independently re-derived and matched
for all 30 pairs (12+12+6) -- confirmed programmatically, no seed or CI
value printed.

## 5. Artifact inventory: filenames, sizes, SHA-256

| Path | Size (bytes) | SHA-256 |
|---|---:|---|
| `artifacts/final_test_cross_condition/H1/attempt_001/cross_condition_result.json` | 10181 | `6624abeff9112b416bed92a4124dc1d38504548a0e66fa3db8b999bfa1b6cbe2` |
| `artifacts/final_test_cross_condition/H1/attempt_001/cross_condition_artifact_manifest.json` | 191 | `91141627513dd358ea5516db855849e22e828d67247e3137e2387af87a379438` |
| `artifacts/final_test_cross_condition/H2/attempt_001/cross_condition_result.json` | 10376 | `2350de61c3ff83362eb9a0ab2cd88ff736c5f4c2ebd9f90eb97e46700ab11732` |
| `artifacts/final_test_cross_condition/H2/attempt_001/cross_condition_artifact_manifest.json` | 191 | `a3c4a8186defccb94bb6850bff77076cb66f8ce0f7e4eab17bf897206cd2d647` |
| `artifacts/final_test_cross_condition/H3/attempt_001/cross_condition_result.json` | 5282 | `f7922b2984aba16c8a5a0833d9f0d500e8bc58a15fedd5529011dc70375f9bf8` |
| `artifacts/final_test_cross_condition/H3/attempt_001/cross_condition_artifact_manifest.json` | 190 | `ffa9882ba14bfb839b8bede4ab6cac96d9e6a3c687151d9e6581b4f2b5acbda7` |

`artifacts/ledger_final_test_analysis.csv`: 7 data rows total (4
preregistered `kind=family`, 3 secondary `kind=cross_condition`), all
`status=completed`, attempt 1, no failed/aborted/duplicate/ambiguous
rows.

## 6. Verification status

* Schema validation (`validate_cross_condition_result_schema`): **PASS**
  for H1/H2/H3.
* Artifact-manifest verification (`verify_analysis_artifact_manifest`):
  **PASS** for H1/H2/H3.
* Semantic recomputation: performed automatically inside
  `compute_final_test_hypothesis_did()` before persistence (Phase
  2B.7B), bit-for-bit joint-resampling re-derivation per pair -- a
  mismatch would have raised `FinalTestAnalysisSemanticVerificationError`
  before any write, which did not occur.
* Label/sample-index equality checked per pair before any DiD
  computation (`compute_final_test_pair_did`'s fail-closed check) --
  satisfied by construction for all 30 pairs, since the run completed.
* `test_split_accessed=False`, `classification=post_validation_pre_test_secondary`
  on all three results.

## 7. Preregistered artifacts unchanged

All 8 files committed in `4426bf5` (H1/H2/H3/BLOCK_C preregistered
results) were re-hashed immediately before this commit and matched
their previously-recorded SHA-256 values exactly -- confirmed
unmodified by the cross-condition run.

## 8. Sealed-result confirmation

The six cross-condition files and the updated ledger are now tracked
(via the narrow `.gitignore` exception already scoped to
`artifacts/final_test_cross_condition/`, added in the preceding
commit) but their scientific contents remain sealed: no number from any
of these files is reproduced in this document, in any commit message, or
in any prior conversational report. Controlled unsealing is governed by
`docs/phase2b_final_test_unsealing_freeze.md`, frozen separately and not
yet executed.
