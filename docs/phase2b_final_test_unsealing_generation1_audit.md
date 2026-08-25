# Phase 2B.8C Part A — Final-Test Unsealing Generation-1 Preservation Audit

**Status: preserves the first controlled unsealing output set as an
immutable historical record.** This document contains no scientific
values.

## 1. First controlled unsealing

Command:

```
uv run python3 scripts/generate_final_test_scientific_report.py unseal
```

Exit code: `0`. Runtime: `2.380s` (`2.09s` user, `0.26s` system), single
real invocation (Phase 2B.8B).

## 2. Generation-1 output hashes

| Path | SHA-256 |
|---|---|
| `artifacts/final_test_scientific_summary.json` | `3f726f92c4659a807a5c7b1e51ddccb12c543ff439e8d0e8f6ff1721dc3f8a76` |
| `docs/phase2b_final_test_scientific_results.md` | `d435171ebf4c141ed7a9edad87f7c6ee52e7e316893b4261b142bcf6befbee8a` |
| `docs/phase2b_final_test_scientific_interpretation.md` | `dcb80c27a902a9c3a6dec7619d25da68031170bb297657877694c1c422827adc` |

## 3. Mechanical checks (Phase 2B.8B Part C, all passed)

Exactly the three authorized outputs, at the exact authorized paths;
valid UTF-8/JSON; cardinalities exact (H1=24, H2=30, H3=12, BLOCK_C=3,
cross-H1=12, cross-H2=12, cross-H3=6); every analysis ID/attempt/hash
matched the generation-1 authorization exactly; no H4 anywhere; no
invented pooled p-value/alpha/significance field; every cell/pair unique
and appeared once; BLOCK_C present; descriptive summaries all labeled
`descriptive_non_inferential`; all required incident/limitation
disclosures present; seven sealed inputs and both ledgers unchanged;
only the three output files appeared in `git status`; a post-run `plan`
recognized the complete output set without rewriting anything.

## 4. All numerical/statistical content is valid

An independent post-unsealing audit (Phase 2B.8C Part 1) re-verified
every numeric value programmatically: every JSON field traces correctly
to the seven sealed inputs; every value in
`phase2b_final_test_scientific_results.md`'s tables matches the JSON
exactly; schema, manifest, and semantic verification all passed.
**No numerical value in any of the three generation-1 files is
incorrect, omitted, or requires correction.**

## 5. Generation-1 file-by-file disposition

* **`artifacts/final_test_scientific_summary.json`**: valid as-is. No
  defect found anywhere in this file.
* **`docs/phase2b_final_test_scientific_results.md`**: valid as-is. Pure
  factual tables, no interpretive language, no significance claims, no
  defect found anywhere in this file.
* **`docs/phase2b_final_test_scientific_interpretation.md`**: contains
  exactly one non-scientific wording defect. Its final incident-
  disclosure sentence states final-test scientific results were "not
  examined by any person or **process**" before controlled unsealing.
  This is too broad: automated computation (bootstrap resampling,
  McNemar) and semantic-verification processes necessarily processed
  every value programmatically, multiple times, before this unsealing
  phase — only **human** inspection is actually novel at unsealing time.
  This wording defect exists in the committed generator source
  (`src/when_tta_hurts/final_test_scientific_reporting.py`, the sentence
  rendered by `render_interpretation_markdown()`), not merely in this
  generated file. **No other sentence, value, table, count, effect,
  interval, p-value, adjusted p-value, classification, limitation, or
  conclusion in this file requires correction.**

## 6. Chat-report overstatements (do not exist in the generated files)

The independent post-unsealing audit (Phase 2B.8C Part 1) additionally
found several claim-overstatement and factual-inversion defects **in
the prior chat-turn summary only**:

* A double-counted "54 of 54 relevant preregistered cells" tally
  (`len(H1)+len(H2)`, ignoring the 24-cell H1/H2 overlap). Mechanically
  verified: 39 unique cells span H1∪H2∪H3∪BLOCK_C; 30 distinct
  unmatched-policy cells exist across H1/H2/H3-unmatched, all 30
  negative.
* Two overstated confirmatory-evidence claims (matched-policy
  mitigation; normalization changing harm magnitude) that the frozen
  SAP (`docs/statistical_analysis_plan.md` §3 item 1) and the frozen
  cross-condition addendum (`docs/phase2b_final_test_cross_condition_addendum.md`
  §1) explicitly classify as unauthorized cross-condition "differs"
  verdicts.
* A factual inversion of BLOCK_C's frozen target direction (calling
  harm the "expected" signal, when the frozen positive-control
  definition targets a **positive** ~+1.6pp improvement, per
  `docs/phase2b_validation_evaluation_block_c_audit.md`).

**None of these three defects exist in any of the three generated
files** (`final_test_scientific_summary.json`,
`phase2b_final_test_scientific_results.md`,
`phase2b_final_test_scientific_interpretation.md`) — `results.md` is
pure data with no interpretive prose, and `interpretation.md`'s actual
committed text is already appropriately cautious (it correctly attributes
cross-condition differences to the secondary analysis and never asserts
a directional "expected" outcome for BLOCK_C). They are chat-only prose
defects, corrected in this project's conversational record separately
from this commit.

## 7. Preservation rationale

Generation 1 is preserved here, byte-identical, as a permanent historical
record. It is superseded **only for wording precision** (sec.5 above) --
never because any scientific value, table, count, effect, interval,
p-value, adjusted p-value, classification, limitation, or conclusion was
wrong. A future generation-2 unsealing (separately authorized) will
correct exactly the one sentence identified in sec.5 and nothing else.
