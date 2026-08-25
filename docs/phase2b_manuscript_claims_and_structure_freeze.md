# Phase 2B.9B Part B — Manuscript Claims and Structure Freeze

**Status: FROZEN before drafting.** This document contains no new
scientific values -- every count below (39, 30, 6, 12, 12, 3) is a
structural cardinality already established and verified in
`docs/phase2b_paper_evidence_package_freeze.md` and the committed
canonical generation-2 summary (`reporting_fingerprint=
2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026`).
This document governs Part C's manuscript draft: any claim not listed
in sec.3 below, or any claim listed in sec.4, may not appear anywhere
in `paper/manuscript.md`.

## 0. Provisional title

> When Test-Time Augmentation Hurts: A Controlled Study in Medical
> Image Classification

Provisional and venue-neutral. No claim of acceptance, novelty tier, or
venue fit is attached to this title.

## 1. Evidence hierarchy (binding on every claim in the manuscript)

1. **Preregistered within-cell clean-versus-TTA evidence** (H1/H2/H3
   unmatched arm + BLOCK_C, `preregistered.*` in the canonical
   summary). The only tier that may use confirmatory language ("harmed",
   "reduced accuracy"). Established *within the specific evaluated
   model*, never as a cross-condition comparison.
2. **Secondary, post-validation/pre-test-specified fixed-model DiD
   comparisons** (`secondary_cross_condition.*`). May describe a
   pattern, a direction, a magnitude, and whether a confidence interval
   excludes zero -- may never use "significant"/"significance" or any
   synonym implying an inferential threshold was passed.
3. **Descriptive three-seed / seed-level summaries**
   (`descriptive_summaries.preregistered_seed_level`). Purely
   descriptive; carry no p-value or CI of their own; may never be
   described as evidence for or against a hypothesis.
4. **External BLOCK_C reference** (source paper's own reported ~+1.6pp
   TTA improvement at N=50 views, `docs/phase2b_validation_evaluation_block_c_audit.md`
   sec.7). Descriptive-only comparator; never an acceptance threshold;
   never re-derived as a new statistic in this project.
5. **Unsupported population-level generalizations.** Not evidence at
   any tier -- explicitly out of scope for every claim in this
   manuscript (sec.4 below enumerates the specific forbidden forms).

A claim may cite evidence from at most the tier(s) it is actually drawn
from, stated in the language that tier permits. No claim may borrow the
confidence of tier 1 while citing tier 2 or 3 evidence.

## 2. Structural cardinalities (already established, reused verbatim)

* 39 total preregistered matrix cells (H1 ∪ H2 ∪ H3 ∪ BLOCK_C,
  non-overlapping-union basis).
* 30 distinct unmatched-policy (policy=none) base cells (H1's 24 ∪ H2's
  6 Block-D-only ∪ H3's 6-cell unmatched arm, deduplicated by
  `run_id`) -- **all 30 negative**.
* 6 matched-policy within-cell rows (H3's matched arm) + 6 H3 secondary
  DiD pairs.
* 12 H1 secondary normalization DiD pairs.
* 12 H2 secondary resolution DiD pairs.
* 3 BLOCK_C preregistered seeds.
* Overlapping family membership (H1 ⊂ H2; H3's unmatched arm ⊂ H1 ∩ H2)
  is bookkeeping, never additional experiments -- the manuscript must
  never describe the union of family sizes (e.g. 24+30+12+3=69, or
  24+30=54) as a cell count.

## 3. Defensible central claims (the only claims permitted in Results/Discussion)

1. Naive mixed-policy TTA reduced final-test accuracy in all 30
   distinct unmatched-policy base cells evaluated (tier 1).
2. This establishes harm for these specific trained models, datasets,
   resolutions, normalization choices, and seeds -- not for medical
   imaging models generally (tier 1, with its own scope boundary
   stated explicitly every time the finding is stated).
3. Matched-policy training mitigation is supported by the six secondary
   fixed-model DiD comparisons (tier 2) and descriptively corroborated
   by the separate within-cell pattern (unmatched strongly negative,
   matched near-zero/mixed; tier 1, stated as corroboration, never as a
   second confirmatory test of the same claim).
4. Both normalization types experienced harm (tier 1); their relative
   difference reversed by dataset in the secondary analysis (tier 2).
5. Higher resolution did not consistently reduce harm (tier 2:
   contrary trend for BloodMNIST, mixed/near-null for PathMNIST).
6. BLOCK_C did not reproduce the expected positive TTA improvement
   (tier 1 result compared against the tier-4 external reference).

No other claim may appear as a central/headline claim.

## 4. Forbidden claims (must never appear anywhere in the manuscript)

* "54 distinct cells" (or any other double-counted union-of-family-sizes
  figure) in place of the correct 30 distinct unmatched-policy cells or
  39 total cells.
* Any population-level model inference ("TTA harms models" without the
  specific-models qualifier).
* Any broad medical-imaging generalization beyond PathMNIST, BloodMNIST,
  and DermaMNIST at the tested resolutions/architectures.
* Any causal-mechanism claim not explicitly labeled a hypothesis for
  future work (per requirement 16 below).
* Any framing of a secondary (tier 2) comparison as a preregistered
  cross-condition test.
* "Significant"/"significance" (or synonyms implying a passed
  inferential threshold) applied to any secondary/tier-2 result.
* Any claim that BLOCK_C reproduced, replicated, or confirmed the
  external +1.6pp reference.
* Any H4 (Validation-Gated TTA) claim -- H4 was never implemented, has
  no derivable family, and must not appear as a result of any kind.
* Any state-of-the-art performance claim.
* Any clinical utility, diagnostic, or safety claim (binding also per
  `CLAUDE.md` rule 7 -- MedMNIST is explicitly not validated for
  clinical use).
* Any CVPR-worthiness, venue-fit, or acceptance-likelihood claim.

## 5. Novelty audit

Per the already-verified primary-source literature review
(`docs/literature_review.md`, re-verified in this phase against the
live arXiv/proceedings/publisher pages -- see `paper/citation_audit.md`
for the fetch-by-fetch record), the following gap analysis holds:

* **Systematic evidence of TTA harm in general**: prior art exists.
  Shanmugam et al. (ICCV 2021) already documents that naive TTA
  averaging can flip correct predictions to incorrect even when net
  accuracy improves. The target preprint (Medeiros, arXiv:2604.09697)
  already reports harm in 11/12 model-dataset combinations under mixed-
  policy TTA. **Not a novel finding of this project by itself** -- this
  project's contribution is an independent, preregistered, audited
  reproduction-and-extension with a frozen statistical protocol and a
  cross-condition secondary layer, not the first report that TTA can
  harm.
* **Augmentation-policy mismatch (train-time vs. test-time policy) as a
  driver of harm**: the target preprint's own Appendix B already
  evaluates clean-image anchoring and basic BatchNorm adaptation as
  mitigations, and this project's matched-policy arm is conceptually
  adjacent to that existing mitigation family. **This project's specific
  angle -- comparing matched-vs-unmatched training policy via a
  secondary fixed-model DiD design, rather than modifying the test-time
  aggregation itself -- was not found described in this form in any
  source reviewed**, but the reviewed set is not exhaustive (see
  `docs/literature_review.md` sec.10, "known gap": no systematic
  citation-graph search was performed). This must be stated as a
  narrow, non-exhaustively-verified gap, not a confirmed first.
* **Medical-image-classification context specifically**: BayTTA
  (Sherkatghanad et al.) already evaluates a TTA aggregation method
  (Bayesian model averaging) specifically in medical imaging (skin
  cancer, breast cancer, chest X-ray). MedMNIST-C (Di Salvo et al.)
  builds a corruption-robustness benchmark on MedMNIST+ but addresses
  training-time augmentation robustness, not test-time aggregation
  harm. **Medical-imaging TTA evaluation is an active area with direct
  prior art (BayTTA)** -- this project does not claim to be the first
  medical-imaging TTA study.
* **Normalization/resolution heterogeneity as a specific ablation axis**:
  no source reviewed performs a controlled BatchNorm-vs-GroupNorm or
  resolution ablation of TTA harm on MedMNIST-family data. This is the
  narrowest defensible novel angle identified, consistent with
  `docs/literature_review.md`'s own conclusion, but is reported here
  only as "not found in the sources reviewed," never as an exhaustively
  verified absence.
* **Distinction between within-model (preregistered) harm and
  cross-condition (secondary) mitigation evidence as a reporting
  discipline**: this two-tier evidentiary separation (with the harder
  preregistered/secondary distinction enforced end-to-end by the
  computational pipeline itself, not just prose) was not found
  articulated as a methodological contribution in any source reviewed.
  This is a methodological/reporting-discipline observation, not a new
  empirical finding, and must be described as such.

**No claim of novelty beyond the above four bullet points may appear in
the manuscript's Introduction, Related Work, or Discussion.** Every
novelty statement must carry the same non-exhaustive-search caveat
stated in `docs/literature_review.md` sec.10.

## 6. Manuscript structure (exact section order)

1. Abstract
2. Introduction
3. Related Work
4. Methods
5. Experimental Design
6. Statistical Analysis
7. Results
8. Discussion
9. Limitations
10. Reproducibility and Audit Trail
11. Conclusion
12. References
13. Supplementary Material (outline only, not drafted as full content
    in this phase)

## 7. Figure/table placement (main text vs. supplementary)

**Main text** (all from the committed evidence package,
`artifacts/paper_evidence/`, manifest-bound):

* Figure 1 (unmatched-policy forest plot, 30 cells) -- Results, tier-1
  headline finding.
* Figure 5 (BLOCK_C positive control) -- Results, positive-control
  finding.
* Table 1 (design/evidence-classification) -- Methods or Statistical
  Analysis, to orient the reader to the evidence hierarchy before any
  result is presented.
* Table 7 (claim adjudication) -- Discussion, as a compact summary of
  what is and is not supported.

**Main text, condensed reference; full table in supplementary:**

* Figure 2 (matched-policy mitigation), Figure 3 (normalization
  heterogeneity), Figure 4 (resolution comparison) -- Results, tier-2
  secondary findings, shown in full (all three are already compact:
  6, 12, 12 rows respectively fit on one page each).

**Supplementary material only** (too large for main-text flow):

* Table 2 (complete 30-cell unmatched-policy table) -- referenced from
  Results by summary statistics and Figure 1; full table deferred.
* Table 3 (matched-policy table: 6 within-cell + 6 DiD pairs) --
  referenced from Results; full table deferred.
* Table 4 (12-pair normalization table) and Table 5 (12-pair
  resolution table) -- referenced from Results by Figures 3/4; full
  tables deferred.
* Table 6 (BLOCK_C table) -- referenced from Results by Figure 5; full
  table deferred (only 3 rows, but placed with the other complete
  tables for consistency).

Every one of the 5 figures and 7 tables in the committed evidence
package is referenced from the main text at least once; none is
introduced for the first time in supplementary material.

## 8. Target length

Approximately 5,000-6,500 words, excluding References and the
Supplementary Material outline, per the following indicative
allocation (not binding to the word, but directional):

| Section | Target words |
|---|---|
| Abstract | 200-250 |
| Introduction | 500-700 |
| Related Work | 600-900 |
| Methods | 700-1000 |
| Experimental Design | 500-700 |
| Statistical Analysis | 500-700 |
| Results | 900-1300 |
| Discussion | 700-1000 |
| Limitations | 400-600 |
| Reproducibility and Audit Trail | 300-500 |
| Conclusion | 150-250 |

## 9. Binding requirements carried into Part C

* Anonymous authorship placeholders only.
* Venue-neutral formatting; no claim of compliance with a specific
  conference template.
* Every numerical statement mechanically traceable to the canonical
  scientific summary, the committed paper-evidence tables, or the
  paper-evidence manifest -- verified by the Part D tooling before the
  manuscript is considered complete, never hand-transcribed without
  that check.
* Preregistered / secondary / descriptive labeling maintained
  everywhere a numeric result is stated.
* No secondary result ever called statistically significant.
* CI exclusion described factually ("the 95% interval excludes zero")
  where relevant, never elevated to "significant."
* 39 total cells / 30 distinct unmatched-policy cells / all 30 negative
  / overlapping families are not additional experiments -- stated
  exactly this way at first mention in Results.
* Matched-policy mitigation presented cautiously, as secondary
  cross-condition evidence descriptively corroborated by within-cell
  patterns -- never as a confirmed causal fix.
* Normalization reversal and resolution behavior presented as
  heterogeneous secondary findings, not as a general rule.
* BLOCK_C's failure to reproduce the external positive reference stated
  plainly, without inventing an explanation beyond labeled hypotheses.
* All required limitations and incidents disclosed (Phase 2B.6*
  metadata inaccuracy, the analysis-runner re-invocation deviation, the
  paper-evidence rendering-defect deviation and its three real-data
  generation runs, the pre-existing stale-test-guard gap fixed in Part
  A) without letting engineering history dominate the scientific
  narrative -- confined to the Limitations and Reproducibility sections.
* Human result inspection explicitly distinguished from automated
  computation/verification, consistent with the corrected generation-2
  wording already frozen in
  `docs/phase2b_final_test_reporting_wording_correction_freeze.md`.
* No invented causal mechanism; any proposed explanation labeled
  "hypothesis for future work."
* No external validity claim beyond the tested scope (three MedMNIST
  datasets, the specific architectures/resolutions/seeds actually run).
* No omission of null, contrary, or inconsistent findings (in
  particular: BLOCK_C's non-reproduction, the resolution reversal for
  BloodMNIST, the mixed/near-null resolution pattern for PathMNIST).
* Every factual related-work claim cited to a primary source verified
  in `paper/citation_audit.md`.
* No fabricated DOI, author list, venue, year, or BibTeX field --
  every `paper/references.bib` entry traces to a primary source
  actually fetched and read in this phase (see `paper/citation_audit.md`
  for the fetch record), or is marked and excluded if it could not be
  verified.
