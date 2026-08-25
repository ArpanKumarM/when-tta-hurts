# Phase 2B.8C Part H — Final-Test Unsealing Generation-2 Audit (Canonical)

**Status: records the corrected, canonical final-test scientific
unsealing output set.** This document contains no scientific values.

## 1. Execution

Command:

```
uv run python3 scripts/generate_final_test_scientific_report.py unseal
```

Exit code: `0`. Runtime: `2.900s` (`2.07s` user, `0.29s` system). Single
real invocation.

## 2. Supersession chain

| Generation | Authorization commit | Authorization SHA-256 | Reporting fingerprint |
|---|---|---|---|
| 1 | `3804bac0b10e6e4c9209ae1dabf944051ef67cbf` | `2fa1686fb5443dab58c6ceb7bf7f1e385b20f85ee45f3ec1203d19d363f9dc92` | `baf02763f9b1f3dc5163dbb52069af842aef47de94acfcbca945e2be5d3736be` |
| 2 (canonical) | `cb8b65fc7cf803090658204c09bc1524df395fc5` | `cdeb4227178d0afa8264f9385e35c5a3bb94c8a83b23245f9a26071806d86977` | `2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026` |

Generation 1 remains preserved, byte-identical, both as its original
commit (`447dfe0`) and as archived files
(`artifacts/final_test_scientific_unsealing/generation_001/final_test_scientific_summary.json`,
`docs/phase2b_final_test_scientific_results_generation_001.md`,
`docs/phase2b_final_test_scientific_interpretation_generation_001.md`).

## 3. Output hashes

| File | Generation 1 SHA-256 | Generation 2 SHA-256 | Identical? |
|---|---|---|---|
| Summary JSON | `3f726f92c4659a807a5c7b1e51ddccb12c543ff439e8d0e8f6ff1721dc3f8a76` | `23cc083741e4e16fae232c9fce7d2c4095bc4e40d5897fd7956c550a78e24fce` | No (embeds new `reporting_fingerprint`) |
| Results Markdown | `d435171ebf4c141ed7a9edad87f7c6ee52e7e316893b4261b142bcf6befbee8a` | `d435171ebf4c141ed7a9edad87f7c6ee52e7e316893b4261b142bcf6befbee8a` | **Yes, byte-identical** |
| Interpretation Markdown | `dcb80c27a902a9c3a6dec7619d25da68031170bb297657877694c1c422827adc` | `92c8861c15b2a0d1254bb319c0aa68921dc85cb9610b26383a75e28ea6a798b8` | No (one sentence corrected) |

## 4. Exact allowed differences (mechanically verified)

* **Summary JSON**: every top-level key's value is identical between
  generations except `reporting_fingerprint` itself. Confirmed
  programmatically: zero differing keys among `descriptive_summaries`,
  `inputs`, `preregistered`, `provenance`, `schema_version`,
  `secondary_cross_condition`.
* **Results Markdown**: `diff` reports **zero lines differing** --
  byte-for-byte identical to generation 1, as expected (this file never
  renders the corrected sentence).
* **Interpretation Markdown**: `diff` reports exactly one line differing
  -- the frozen sentence, replacing "No scientific result from any of
  the seven sealed artifacts was examined by any person or process
  before this controlled-unsealing phase." with "No final-test
  scientific result was inspected by a human before this controlled
  unsealing." No other line changed.

**No unauthorized difference was found anywhere.**

## 5. Zero scientific-value differences

Every family/hypothesis cell/pair count, every delta-accuracy, bootstrap
CI, McNemar statistic/p-value, BH-adjusted p-value, DiD estimate, and
descriptive summary is identical between generation 1 and generation 2
(proven by the JSON top-level-key equality check in sec.4, which covers
`preregistered` and `secondary_cross_condition` in full).

## 6. Corrected observation wording

Generation 2's interpretation document states: "No final-test scientific
result was inspected by a human before this controlled unsealing" --
precise, since automated computation and semantic-verification
necessarily processed these values programmatically before this
unsealing phase; only human inspection was novel at unsealing time.

## 7. Corrected unique-cell accounting (for future reporting; not itself
part of the generated files, which never used the flawed "54" figure)

Mechanically re-verified against generation 2's JSON: **39 unique cells**
span H1∪H2∪H3∪BLOCK_C; **30 distinct unmatched-policy base cells**
(H1's 24 ∪ H2's 6 Block-D-only cells ∪ H3's 6-cell unmatched arm,
deduplicated) are **all 30 negative**. The naive, double-counted
`len(H1)+len(H2)=54` figure must never be described as "54 distinct
cells."

## 8. Corrected evidentiary classification (for future reporting)

* Within-cell preregistered tests (H1/H2/H3/BLOCK_C) establish
  clean-vs-TTA harm *within the specific evaluated models*, never a
  cross-condition "differs" verdict.
* Matched-policy mitigation is supported by the **secondary** fixed-model
  DiD analysis (6/6 pairs favor matched training) and is *descriptively
  corroborated* by the separate within-cell facts (unmatched strongly
  negative; matched near-zero/mixed) -- it is not a preregistered
  cross-condition test.
* Normalization and resolution comparisons are **secondary only**; both
  normalization types experience harm within-cell, while their relative
  difference (and its dataset-dependent reversal) is secondary-only
  evidence.
* Secondary CI-excludes-zero is never labeled "significant" anywhere in
  either generated document.
* BLOCK_C's frozen target is the source paper's reported **positive**
  ~+1.6pp TTA improvement, never harm -- generation 2's interpretation
  document (like generation 1's) correctly avoids asserting any
  "expected" direction, stating only that BLOCK_C is "reported
  regardless of direction."
* No H4 claim appears anywhere in either generation.

## 9. Verification summary

| Check | Result |
|---|---|
| Exactly three canonical outputs exist | Pass |
| Schema/manifest/authorization/fingerprint verification | Pass |
| Summary JSON payload identical except `reporting_fingerprint` | Pass |
| Results Markdown byte-identical | Pass |
| Interpretation Markdown differs only in the one frozen sentence | Pass |
| 39 unique cells / 30 distinct unmatched cells, all negative | Pass |
| No H4, no pooled/model-population inference, no invented significance | Pass |
| Post-generation `plan`: ready, outputs recognized, zero further writes | Pass |
| Full quality suite (1136 tests) | Pass |
| All pre-existing fingerprints unchanged except reporting | Pass |

Generation 2 is hereby the canonical `artifacts/final_test_scientific_summary.json`,
`docs/phase2b_final_test_scientific_results.md`, and
`docs/phase2b_final_test_scientific_interpretation.md`.
