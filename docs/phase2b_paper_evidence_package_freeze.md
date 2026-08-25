# Phase 2B.9A Part B — Paper Evidence Package Freeze

**Status: FROZEN before implementation.** This document contains no
scientific values. It freezes the exact figure/table set generated from
the canonical generation-2 scientific summary
(`artifacts/final_test_scientific_summary.json`,
`reporting_fingerprint=2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026`).
Every figure and table is mechanically derived; none introduces a new
statistic, a pooled estimate, or a significance label beyond what the
summary already contains.

## 0. Field-to-figure/table map (verified against the real summary)

| Summary field | Used by |
|---|---|
| `preregistered.H1.cells[*]` (24) | Figure 1 (subset), Table 2 (subset) |
| `preregistered.H2.cells[*]` (30) | Figure 1 (subset), Table 2 (subset) |
| `preregistered.H3.cells[*]` (12: 6 unmatched + 6 matched) | Figure 1 (unmatched subset), Table 2 (unmatched subset), Figure 2 (matched within-cell), Table 3 |
| `preregistered.BLOCK_C.cells[*]` (3) | Figure 5, Table 6 |
| `secondary_cross_condition.H1.pairs[*]` (12) | Figure 3, Table 4 |
| `secondary_cross_condition.H2.pairs[*]` (12) | Figure 4, Table 5 |
| `secondary_cross_condition.H3.pairs[*]` (6) | Figure 2 (DiD half), Table 3 |
| `descriptive_summaries.preregistered_seed_level[*]` | Table 1 (context only, not re-plotted) |
| `provenance`, `reporting_fingerprint`, `inputs` | Manifest binding only, never plotted |

Mechanically re-verified counts against the real canonical summary
before freezing this design: 30 distinct unmatched-policy cells (H1's 24
∪ H2's 6 Block-D-only ∪ H3's 6-cell unmatched arm, deduplicated by
`run_id`), 6 matched within-cell rows, 12/12/6 cross-condition pairs, 3
BLOCK_C cells. All 39 real run_ids parse cleanly under the frozen naming
convention (`<block>-<dataset>-<resolution>px-[<model>-]<normalization>-policy-<policy>-s<seed>`).
No required field is absent from the canonical summary; no raw
prediction access is needed for any figure or table below.

## 1. Figures

### Figure 1 — Unmatched-policy TTA effects

* All 30 distinct unmatched-policy base cells, exactly once (deduplicated
  by `run_id` across H1/H2/H3's unmatched arm -- never double-plotted).
* Horizontal forest plot: point = `bootstrap.delta_accuracy` (N=50,
  percentage-point axis), whisker = `[ci_low, ci_high]` (frozen 95%
  paired-bootstrap interval).
* Rows grouped by dataset, then resolution, then normalization, then
  seed (the frozen deterministic ordering already used by
  `docs/statistical_analysis_plan.md` sec.8).
* A solid vertical zero-reference line.
* Caption states explicitly: "Preregistered within-cell clean-versus-TTA
  evidence. Each row is one trained model (dataset x resolution x
  normalization x seed); no cross-condition comparison is made or
  implied here."
* No cell is a member of more than one row.

### Figure 2 — Matched-policy mitigation

* All 6 matched-versus-unmatched secondary DiD pairs
  (`secondary_cross_condition.H3.pairs`), one row per pair, both
  BloodMNIST and PathMNIST seeds shown (3 + 3).
* Point = `bootstrap.did`, whisker = `[ci_low, ci_high]`, zero-reference
  line.
* Caption states explicitly: "Secondary, fixed-model, post-validation/
  pre-test-specified difference-in-differences comparison -- not a
  preregistered cross-condition inference. No significance decision is
  made for these estimates."
* No "significant"/"significance"/"passed"/"confirmed" wording anywhere
  near this figure.

### Figure 3 — Normalization heterogeneity

* All 12 BatchNorm-vs-GroupNorm secondary DiD pairs
  (`secondary_cross_condition.H1.pairs`).
* Faceted/grouped by dataset (BloodMNIST panel, PathMNIST panel) so the
  sign reversal between datasets is visually apparent.
* All resolutions (28px, 64px) and all 3 seeds per dataset/resolution
  shown.
* Caption states explicitly: "The direction of this secondary estimate
  is dataset-dependent (see panels) and must not be read as a general
  BatchNorm-vs-GroupNorm verdict."

### Figure 4 — Resolution comparison

* All 12 secondary resolution DiD pairs
  (`secondary_cross_condition.H2.pairs`), grouped by dataset and
  normalization.
* All seeds shown.
* A dashed reference line marks the frozen hypothesized positive
  direction ("higher resolution reduces degradation") purely as a visual
  reference -- never as a confirmation marker, never annotated with a
  check mark or verdict symbol.
* Caption states explicitly: "BloodMNIST pairs trend contrary to the
  hypothesized direction; PathMNIST pairs are mixed/near-null. Neither
  pattern is a preregistered or confirmatory test of H2."

### Figure 5 — BLOCK_C positive control

* All 3 DermaMNIST/ResNet-18 seed effects and intervals
  (`preregistered.BLOCK_C.cells`).
* Zero-reference line.
* An external reference line at **+1.6 percentage points**, sourced from
  the already-frozen project documentation
  (`docs/phase2b_validation_evaluation_block_c_audit.md` sec.7: "the
  source paper's own reported '+1.6pp at N=50 views' condition"),
  labeled explicitly "external reference (source paper), descriptive
  only -- not an acceptance threshold."
* Caption states explicitly: "The expected positive TTA improvement
  (~+1.6pp) was not reproduced in this project's frozen operationalization."

## 2. Tables

1. **Experimental-design and evidence-classification table** -- static,
   mechanically rendered from this freeze document's own frozen
   classification rules (which evidence tier each family/pair-set
   belongs to): preregistered within-cell (H1/H2/H3/BLOCK_C), secondary
   fixed-model (cross-condition H1/H2/H3), descriptive-only
   (seed-level summaries). No scientific value in this table -- it is a
   map of evidence tiers, not a results table.
2. **Complete 30-cell unmatched-policy table** -- one row per
   `extract_unmatched_cells()` entry: dataset, resolution, normalization,
   seed, delta_accuracy, 95% CI, raw McNemar p-value, BH-adjusted p-value
   (the latter two pulled from whichever family first reports that cell,
   since the value is identical across families for the same underlying
   computation).
3. **Complete matched-policy table** -- 6 within-cell rows
   (`extract_matched_within_cell()`) plus the 6 corresponding DiD pairs
   (`secondary_cross_condition.H3.pairs`), shown together so the
   within-cell/secondary distinction (sec.6 of
   `docs/phase2b_final_test_unsealing_freeze.md`) is visible in one
   table.
4. **Complete 12-pair normalization table** -- `secondary_cross_condition.H1.pairs`,
   full DiD/CI columns.
5. **Complete 12-pair resolution table** -- `secondary_cross_condition.H2.pairs`,
   full DiD/CI columns.
6. **Complete three-seed BLOCK_C table** -- `preregistered.BLOCK_C.cells`,
   full delta/CI/McNemar/BH columns, plus the external +1.6pp reference
   as a labeled descriptive footnote (never a table column implying a
   comparison test).
7. **Claim-adjudication table** -- exactly the four-tier classification
   already adjudicated in Phase 2B.8C: (a) preregistered within-cell
   evidence, (b) secondary fixed-model comparisons, (c) descriptive
   summaries, (d) explicitly unsupported/generalized claims (H4, pooled/
   model-population verdicts, "significant" secondary language) --
   listed as never-permitted, not as a result.

Every planned cell (39 total, non-overlapping-union basis) and every
planned pair (30 total across the three hypotheses) appears in at least
one complete table or figure above. No selective omission by magnitude,
sign, CI, or p-value occurs anywhere.

## 3. Rendering discipline (binding on Part C's implementation)

* Colorblind-safe palette: Okabe-Ito (`#000000, #E69F00, #56B4E9,
  #009E73, #F0E442, #0072B2, #D55E00, #CC79A7`) used consistently across
  all five figures for the same semantic role (e.g., BatchNorm always
  the same color, GroupNorm always the same color).
* Readable fonts: default matplotlib DejaVu Sans at a minimum 9pt for
  axis/tick labels, 10pt for titles, no font smaller than 7pt anywhere.
* Percentage-point axes labeled explicitly ("Δ accuracy (pp)" or "DiD
  (pp)"), consistent sign convention (TTA − clean) across all figures.
  Values in the summary are already fractional (e.g. −0.301); rendering
  code performs a pure `×100` unit conversion for display only (not a
  new statistic).
* Visible zero-reference line on every figure.
* Complete captions embedded as figure text (not only in the manifest),
  each stating the evidence tier explicitly per sec.1 above.
* No truncated/non-zero-origin axis that would visually exaggerate an
  effect; axis limits are computed from the actual data range with a
  fixed symmetric padding, never manually tuned per figure.
* No decorative elements (no 3D, no drop shadows, no unnecessary
  gridlines beyond a light horizontal reference grid for row alignment
  in forest plots).
* Every figure rendered in both PDF (vector) and PNG (raster, >=150 DPI)
  from the isolated `tools/paper_evidence` matplotlib environment.

## 4. What this phase does not do

No new inferential statistic is computed. No raw prediction, dataset, or
checkpoint file is read. No manuscript prose, novelty claim, or
publication/venue assessment is produced. No existing scientific
fingerprint, authorization, or sealed artifact is modified.
