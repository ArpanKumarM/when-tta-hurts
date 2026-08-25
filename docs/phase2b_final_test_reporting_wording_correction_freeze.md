# Phase 2B.8C Part B — Final-Test Reporting Wording Correction Freeze

**Status: FROZEN before any code change.** This document contains no
scientific values. It authorizes exactly one sentence-level wording
correction to the deterministic report generator, per
`docs/phase2b_final_test_unsealing_generation1_audit.md` sec.5.

## 1. The one permitted change

In `render_interpretation_markdown()`
(`src/when_tta_hurts/final_test_scientific_reporting.py`), replace the
sentence containing:

> "...examined by any person or process before this controlled-unsealing..."

with exactly:

> "No final-test scientific result was inspected by a human before this controlled unsealing."

No other string literal in `render_interpretation_markdown()`,
`render_results_markdown()`, or `build_scientific_summary()` may change.

## 2. No other interpretation sentence may change

Every other sentence currently rendered by `render_interpretation_markdown()`
-- the scientific-classification block, the required-limitations block,
and every other incident-disclosure bullet -- remains byte-identical to
generation 1.

## 3. No scientific content may change

No value, table, count, effect, interval, p-value, adjusted p-value,
classification, limitation, or conclusion may change anywhere in any of
the three outputs. This correction is a pure wording-precision fix, not
a scientific revision.

## 4. Machine-readable payload identity

`artifacts/final_test_scientific_summary.json`'s scientific payload
(`inputs`, `preregistered`, `secondary_cross_condition`,
`descriptive_summaries`, `provenance`) must remain byte-for-byte
identical between generation 1 and generation 2. Only the top-level
`reporting_fingerprint` field is expected to change (since the
generator's own source file changes, per sec.9 of the manifest
discipline already established in Phase 2B.7A-C).

## 5. Results-Markdown identity

`docs/phase2b_final_test_scientific_results.md`'s scientific tables
(every family/hypothesis section, every row, every column) must remain
byte-for-byte identical between generation 1 and generation 2 -- this
file never renders the sentence being corrected, so it must be
completely unchanged.

## 6. Generation-1 preservation

Generation 1's committed outputs (`447dfe0`) and their archived copies
(Part C of this phase) both remain permanently in git history,
unmodified, byte-identical to their originally recorded hashes,
regardless of what generation 2 produces.

## 7. Generation-2 canonicalization gate

Generation 2 becomes canonical (i.e., is committed as the active
`artifacts/final_test_scientific_summary.json` /
`docs/phase2b_final_test_scientific_results.md` /
`docs/phase2b_final_test_scientific_interpretation.md`) only if every
allowed-difference check in Phase 2B.8C Part G passes: the scientific
JSON payload identical except versioned reporting-provenance fields, the
results-Markdown tables byte-identical, and the interpretation-Markdown
differing only in the one frozen sentence (plus any explicitly
authorized provenance line).

## 8. Fail-closed on any unauthorized difference

If any difference beyond reporting provenance and the one frozen
sentence is detected when generation 2 is compared against archived
generation 1, generation 2 must NOT be committed -- the discrepancy must
be reported and adjudicated before any further action.
