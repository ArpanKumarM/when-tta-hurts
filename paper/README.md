# `paper/` — Manuscript, References, and Verification (Phase 2B.9B)

This directory contains the first, venue-neutral manuscript draft for
"When Test-Time Augmentation Hurts: A Controlled Study in Medical Image
Classification," produced under
`docs/phase2b_manuscript_claims_and_structure_freeze.md`.

## Contents

* `manuscript.md` -- the manuscript itself (Markdown). Anonymous
  authorship placeholders; no venue-specific formatting or template
  compliance claim.
* `references.bib` -- BibTeX entries for every citation used in
  `manuscript.md`. Every entry traces to a primary source actually
  fetched and read in this phase; see `citation_audit.md`.
* `citation_audit.md` -- the fetch-by-fetch verification record for
  every BibTeX entry: which primary-source URL was opened, what was
  confirmed, and any gap (one entry, Wu & He's ECCV 2018 venue/DOI, is
  disclosed as corroborated-but-not-directly-opened rather than fully
  verified, because the publisher pages returned HTTP 403).
* `verify_manuscript_claims.py` -- a read-only verification script (see
  below) that checks the manuscript's numeric claims and required
  wording against the canonical evidence package. It never reads raw
  predictions, datasets, checkpoints, or sealed per-family analysis
  results -- only `artifacts/final_test_scientific_summary.json` and
  the committed `artifacts/paper_evidence/` tables/manifest.

## What this manuscript is and is not

This is a first draft, not a submission-ready document. It does not
claim compliance with any specific conference or journal's formatting
requirements, and no claim about publication venue, acceptance
likelihood, or novelty tier beyond what is stated in its own "Related
Work" section is made anywhere in this directory. Converting this draft
to a venue-specific template, assessing its fit for any particular
venue, or preparing submission materials is explicitly out of scope for
this phase.

## Every number is mechanically traceable

Per `docs/phase2b_manuscript_claims_and_structure_freeze.md` sec.9,
every numerical statement in `manuscript.md`'s Results section must
trace to the canonical scientific summary
(`artifacts/final_test_scientific_summary.json`), the committed
paper-evidence tables (`artifacts/paper_evidence/tables/`), or the
paper-evidence manifest
(`artifacts/paper_evidence/paper_evidence_manifest.json`). No number was
hand-transcribed without the automated check in
`verify_manuscript_claims.py` (and its test suite,
`tests/test_verify_manuscript_claims.py`) passing.

## Running the verification script

```
uv run python3 paper/verify_manuscript_claims.py
```

Exits 0 and prints a JSON readiness report if every check passes; exits
1 and prints the specific failing check(s) otherwise. It performs no
writes and accesses no scientific artifact other than the canonical
summary and the already-committed paper-evidence outputs.

## Evidentiary tiers used throughout

1. **Preregistered within-cell** (confirmatory): H1/H2/H3 unmatched arm
   + BLOCK_C.
2. **Secondary, post-validation/pre-test-specified fixed-model
   comparisons** (non-confirmatory): cross-condition H1/H2/H3.
3. **Descriptive summaries** (non-inferential): seed-level groupings.
4. **External BLOCK_C reference** (descriptive-only comparator, not an
   acceptance threshold).

See `docs/phase2b_manuscript_claims_and_structure_freeze.md` for the
full, binding definition of each tier and the exact list of forbidden
claims.

## History

This directory was empty as of the Phase 0 preregistration (commit
`a22db01`), which set four ground rules for whenever writing started:
every claim must trace to a specific artifact and script, no novelty
claim without a current literature review, no clinical-use language,
and confirmatory/exploratory results must be labeled as such. All four
are carried forward and made mechanically enforced in this phase: claim
traceability is now checked by `verify_manuscript_claims.py` rather than
asserted in prose; the literature review was re-verified against live
primary sources for this draft (`citation_audit.md`); no clinical-use
language appears anywhere in `manuscript.md` (checked by the same
script); and the preregistered/secondary/descriptive distinction is the
manuscript's central organizing structure, not an afterthought.
