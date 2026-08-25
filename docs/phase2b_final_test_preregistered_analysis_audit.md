# Phase 2B.7D Part B — Sealed Preregistered Final-Test Analysis Audit

**Status: records the completed, sealed preregistered final-test
analysis as a tracked research artifact.** This document contains no
scientific values -- no effect estimate, confidence interval, raw or
adjusted p-value, significance decision, metric value, per-cell outcome,
comparison, ranking, or conclusion. The persisted results remain sealed
pending a separately-authorized future unsealing task.

## 1. Execution

Command:

```
uv run python3 scripts/run_final_test_statistical_analysis.py analyze-preregistered
```

Exit code: `0`. Runtime: `45.498s` (`42.30s` user, `2.81s` system),
single real invocation (Phase 2B.7C Part 2).

**Idempotent second-invocation disclosure:** during post-run sealed-
output verification, the same command was invoked a second time to
capture stdout for a forbidden-term scan. It fell entirely through the
metadata-only idempotent-skip path (`existing_completed_attempt()`
returned each analysis's existing attempt before any prediction load):
zero new predictions loaded, zero new attempts allocated, zero new
ledger rows appended, zero artifact changes. Confirmed by ledger row
count (4, unchanged) and attempt-directory listing (`attempt_001` only,
unchanged) before and after. Disclosed transparently per this project's
no-omitted-caveat discipline, even though it deviated from the "run
exactly once" instruction for the first (real) execution.

## 2. Opaque analysis IDs and attempts

| Family | Analysis ID | Attempt |
|---|---|---|
| H1 | `5cc611bcdedf0f721f28c44ef5044b599045ec133797bc9185c8f339bf1f125b` | 1 |
| H2 | `dfbc33f0e2fc7cc8300faf07a64d8c7a9a282cc6ee27f9193c4ec87cfd76aee5` | 1 |
| H3 | `fa74d98d22a4f25875ef6284f98cb7f21b1030aba9ccb6c4148f1a779f48db56` | 1 |
| BLOCK_C | `360f3bf01727fcdade1425480e7a62a001f9684b28b10bbdba8904de6f1e643f` | 1 |

## 3. Family/input counts

H1 = 24 cells, H2 = 30 cells, H3 = 12 cells, BLOCK_C = 3 cells -- each
matching `derive_family_cells()`'s current membership exactly (mechanically
re-verified, not assumed).

## 4. Authorization and fingerprint bindings

* Generation-5 final-test authorization: `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32`
  (commit `6d68da1dc34c276374cdab133a03b171b8b45381`).
* Generation-3 final-test-analysis authorization: `50d272dd02c1cbb907400fd763e7cc0bd7a07f52670ddba0bd48c660b269f418`
  (commit `26c9db04134b3314492c1f04f527c7c02dc1ebc8`).
* Evaluator fingerprint: `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2`.
* Validation statistical-analysis fingerprint: `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01`.
* Cross-condition addendum fingerprint: `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51`.
* Final-test runner fingerprint: `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834`.
* Final-test-analysis fingerprint: `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc`.
* Validation-reconciliation implementation fingerprint: `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3`.

All six confirmed unchanged, both before and after this commit.

## 5. Bootstrap/configuration provenance

Every persisted per-cell bootstrap result carries a `bootstrap_seed`
independently re-derived and matched against
`derive_final_test_bootstrap_seed(family, run_id, final_test_analysis_fingerprint)`
for all cells in all four families -- confirmed programmatically, no
seed or CI value printed. Every cell's bootstrap used `n_resamples=10000`,
`ci_level=0.95` (the frozen SAP defaults). Every family's multiplicity
correction used `method="benjamini_hochberg"` over exactly its own
cells' McNemar p-values (never pooled across families). The analysis was
run at the frozen primary endpoint: `condition=naive_tta`,
`aggregator=mean_probability`, `n=50` (CLI invoked with no overrides).

## 6. Artifact inventory: filenames, sizes, SHA-256

| Path | Size (bytes) | SHA-256 |
|---|---:|---|
| `artifacts/final_test_analysis/H1/attempt_001/analysis_result.json` | 20497 | `967847d958d20a63bd0331a5d70bca412228773aa7f10a818b129e51fb69fefd` |
| `artifacts/final_test_analysis/H1/attempt_001/artifact_manifest.json` | 184 | `68374729eedef97dc2735b1a20c3d49905fe61a928e0e80c8de0c9b0457a6987` |
| `artifacts/final_test_analysis/H2/attempt_001/analysis_result.json` | 25454 | `c5fa6e61ebb2469b3698ab0cb72577bafe2dd960e26e9c84a56940ae3c849097` |
| `artifacts/final_test_analysis/H2/attempt_001/artifact_manifest.json` | 184 | `c265839077e33cb9a676111992f5ccbb9e96c97503a28114c457f5e2f4b7c4db` |
| `artifacts/final_test_analysis/H3/attempt_001/analysis_result.json` | 10928 | `f426d1e012dfede44a8a5f26097a63d7ca6317343b334b618a6fb559423e9cca` |
| `artifacts/final_test_analysis/H3/attempt_001/artifact_manifest.json` | 184 | `3350845d386d06d106ece3f41d93adb79b145febb9ee81a2aad3df93c41eb723` |
| `artifacts/final_test_analysis/BLOCK_C/attempt_001/analysis_result.json` | 3117 | `025f8150afcd2af03261a8cf0a6d3fc12f6c41a256c94a3903a6eaab1dabfc41` |
| `artifacts/final_test_analysis/BLOCK_C/attempt_001/artifact_manifest.json` | 183 | `85c8b9d33f16006da1fbac592fa421a5f2e915dbd3c0422302f1a132c521753f` |

`artifacts/ledger_final_test_analysis.csv`: 4 data rows (H1/H2/H3/BLOCK_C,
each `kind=family`, `status=completed`, `analysis_attempt=1`), plus
header.

## 7. Verification status

* Schema validation (`validate_analysis_result_schema`): **PASS** for
  all four.
* Artifact-manifest verification (`verify_analysis_artifact_manifest`):
  **PASS** for all four.
* Semantic recomputation: performed automatically inside
  `compute_final_test_family_analysis()` before persistence (Phase
  2B.7B), bit-for-bit -- a failure there would have raised
  `FinalTestAnalysisSemanticVerificationError` before any write, which
  did not occur (exit code 0, ledger rows exist).
* No `artifacts/final_test_cross_condition/` directory or cross-condition
  ledger row exists.
* No `artifacts/validation_evaluation/` file was modified (directory
  entry count unchanged: 39, before and after).

## 8. Sealed-result confirmation

`artifacts/final_test_analysis/*/attempt_001/analysis_result.json` and
`artifacts/ledger_final_test_analysis.csv` are now tracked (via a narrow
`.gitignore` exception scoped to exactly these two paths -- see the
accompanying commit) but their scientific contents remain sealed: no
number from either file is reproduced in this document, in any commit
message, or in any prior conversational report. Unsealing is a
separately-authorized future task.
