# Phase 2B.7E Part D — Controlled Final-Test Unsealing Freeze

**Status: FROZEN before any scientific value is read.** This document
contains no scientific values. It governs how, and only how, the seven
sealed final-test analysis results committed in this repository
(`4426bf5`, `29a3bfe`) may eventually be unsealed, extracted, and
reported. No unsealing happens in this task; this document is the
contract a future, separately-authorized task must follow.

## 1. Immutable input set

Exactly seven committed result artifacts, none of which this document
or any future unsealing task may recompute, rerun, or alter:

| # | Kind | Identifier | Analysis ID | Attempt | Commit |
|---|---|---|---|---:|---|
| 1 | Preregistered | H1 | `5cc611bcdedf0f721f28c44ef5044b599045ec133797bc9185c8f339bf1f125b` | 1 | `4426bf5` |
| 2 | Preregistered | H2 | `dfbc33f0e2fc7cc8300faf07a64d8c7a9a282cc6ee27f9193c4ec87cfd76aee5` | 1 | `4426bf5` |
| 3 | Preregistered | H3 | `fa74d98d22a4f25875ef6284f98cb7f21b1030aba9ccb6c4148f1a779f48db56` | 1 | `4426bf5` |
| 4 | Preregistered | BLOCK_C | `360f3bf01727fcdade1425480e7a62a001f9684b28b10bbdba8904de6f1e643f` | 1 | `4426bf5` |
| 5 | Cross-condition | H1 | `d9e335e0f28b578631975e83442480e59055edab2d2ba340a71a70f44bc93b7f` | 1 | `29a3bfe` |
| 6 | Cross-condition | H2 | `5645a6cd96f414c173937a27fcc133c3bc5547eec454fae1787aa14bc6edd09c` | 1 | `29a3bfe` |
| 7 | Cross-condition | H3 | `5bc1e82cc6a07932b0b736c6f8d37396a47ba0be94e59632a1d36d764ddcd6a7` | 1 | `29a3bfe` |

Authorization/fingerprint bindings applicable to all seven: generation-5
final-test authorization `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32`
(commit `6d68da1dc34c276374cdab133a03b171b8b45381`); generation-3
final-test-analysis authorization `50d272dd02c1cbb907400fd763e7cc0bd7a07f52670ddba0bd48c660b269f418`
(commit `26c9db04134b3314492c1f04f527c7c02dc1ebc8`); evaluator fingerprint
`e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2`;
final-test-analysis fingerprint
`91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc`.
Artifact SHA-256 hashes are recorded in
`docs/phase2b_final_test_preregistered_analysis_audit.md` and
`docs/phase2b_final_test_cross_condition_analysis_audit.md`. A future
unsealing task must re-verify every one of these hashes matches before
reading any file, and must halt without reporting anything if any
mismatch is found.

## 2. Scientific classification

* The **preregistered final-test analyses** (H1/H2/H3/BLOCK_C) are the
  confirmatory within-cell analyses specified by the frozen SAP
  (`docs/statistical_analysis_plan.md`). Each is a paired clean-vs-TTA
  test within one trained model/cell.
* These within-cell tests do **not**, by themselves, establish the
  cross-condition differences implied by H1 ("normalization matters"),
  H2 ("resolution matters"), or H3 ("policy matching matters") --
  those are comparisons ACROSS cells, which the preregistered analysis
  never computes.
* The **cross-condition difference-in-differences analyses** (H1/H2/H3
  fixed-pair addendum) are explicitly secondary: post-validation/
  pre-test specified (frozen after all 39 validation cells completed,
  before the test split was opened), fixed-model-only (each pair
  compares two ALREADY-TRAINED models, never a resampled population of
  retrainings), and never originally preregistered
  (`classification: post_validation_pre_test_secondary` is a schema-
  enforced field on every persisted result).
* **BLOCK_C** is a positive-control reproduction and must be reported
  regardless of direction or magnitude -- it is not filtered by outcome.
* **No H4 claim is permitted.** H4 (Validation-Gated TTA) is draft, not
  approved, and has no derivable family in this codebase
  (`derive_family_cells()` raises for any family not in `KNOWN_FAMILIES
  = (H1, H2, H3, BLOCK_C)`); no future report may introduce an H4
  finding from these seven artifacts.
* **No population-level or model-population inference may be invented.**
  Neither the preregistered nor the secondary schema contains a pooled,
  model-population, or seed-level significance field
  (`statistical_analysis_artifacts.py`'s
  `_FORBIDDEN_CROSS_CONDITION_KEYS` structurally forbids
  `pooled_p_value`/`model_population_p_value`/`family_wise_p_value`/
  `seed_level_p_value`/`alpha`/`significant` in any cross-condition
  result); a future report may not add one.

## 3. Complete-reporting rule

Every planned family (H1, H2, H3, BLOCK_C), every cell within each
family, every cross-condition pair (all 12+12+6), every seed, must be
reported -- negative, positive, null, and unexpected results alike. No
selective omission based on magnitude, direction, confidence interval,
or p-value. The generator (sec.13) enforces this mechanically: it must
assert the extracted row/pair count for each family/hypothesis exactly
equals the frozen count (H1=24, H2=30, H3=12, BLOCK_C=3 cells;
H1=12, H2=12, H3=6 pairs) before writing any output, and must fail
closed (produce no output file) if any count mismatches.

## 4. Frozen endpoints (reaffirmed, not re-decided here)

N=50 TTA views; mean-probability aggregation; `naive_tta` condition
against the frozen clean comparator; frozen paired-bootstrap parameters
(>=10,000 resamples, 95% CI, resampled from the SAME persisted
correctness arrays used at compute time); frozen McNemar method
selection rule (exact binomial when discordant total <25, continuity-
corrected chi-square otherwise); frozen Benjamini-Hochberg family
membership (each of H1/H2/H3/BLOCK_C is its own correction family, never
pooled across families); frozen cross-condition joint four-array
resampling (a single resampled index set applied to all four correctness
arrays in a pair, never independent per-array resampling). None of these
may be changed, re-derived, or substituted during unsealing.

## 5. Preregistered reporting tables (frozen structure)

The generator (sec.13) must mechanically produce, from the four
preregistered artifacts:

* One complete cell-level table for H1 (24 rows).
* One complete cell-level table for H2 (30 rows).
* One complete cell-level table for H3 (12 rows).
* One complete BLOCK_C table (3 rows).
* A family-level accounting table: per family, count of raw McNemar
  p-values and their BH-adjusted counterparts (counts only in this
  frozen document; the actual values appear only in the future
  `phase2b_final_test_scientific_results.md` deliverable, sec.11).

Each cell-level row's frozen column set (extracted verbatim from the
already-persisted `per_cell_statistics[run_id]` structure, never
recomputed): dataset, resolution, normalization, training_policy (block
role), seed -- all derivable from the `run_id` string and/or the frozen
matrix, never hand-typed; clean endpoint value; TTA endpoint value;
paired effect estimate (`bootstrap.delta_accuracy`); bootstrap interval
(`bootstrap.ci_low`, `bootstrap.ci_high`, `bootstrap.ci_level`); McNemar
raw p-value; BH-adjusted p-value. A "decision" field is included ONLY IF
it already exists in the approved persisted schema -- it does not
currently exist in `_REQUIRED_TOP_LEVEL_KEYS`/`per_cell_statistics`'s
schema, so no future report may add one; the absence of a decision field
is itself frozen, not an oversight.

## 6. Secondary reporting tables (frozen structure)

The generator must mechanically produce, from the three cross-condition
artifacts:

* All 12 H1 pairs, all 12 H2 pairs, all 6 H3 pairs -- every pair, never
  a filtered subset.
* Exact condition identities (`condition_a`/`condition_b` run_id,
  evaluation_id) and seed matching (pairs are matched by dataset/
  resolution/seed per the frozen addendum spec; the generator must
  display, not re-derive, this matching).
* DiD point estimates and bootstrap intervals (`bootstrap.did`,
  `bootstrap.ci_low`, `bootstrap.ci_high`, `bootstrap.ci_level`,
  `bootstrap.bootstrap_seed`, `bootstrap.n_resamples`).
* Every other schema-approved secondary field already present in
  `per_pair_results` -- no new field is invented.

The generator must NOT invent, compute, or display: a pooled p-value, a
model-population p-value, an alpha threshold, a significance flag, or
any new hypothesis-level decision beyond what the frozen addendum spec's
`reporting` section already permits (which explicitly forbids all of
these -- sec.2 above).

## 7. Descriptive summaries

Seed-level means, sample standard deviations, minima, maxima, and
individual seed values MAY be reported, but only as descriptive
summaries of the seed values already present in the preregistered/
secondary tables -- computed by simple, auditable arithmetic over
already-extracted, already-reported numbers (never a new statistical
test, never a new resampling procedure). They must never be presented,
labeled, or discussed as an additional confirmatory test, and must never
carry a p-value, confidence interval, or significance claim of their
own.

## 8. Interpretation rules (binding on the future interpretation document)

* Statistical evidence (the McNemar/BH result, the bootstrap interval)
  must be discussed separately from effect magnitude (the raw delta) --
  never conflated into a single "significant and large" or "significant
  and small" claim without stating both dimensions explicitly.
* "Absence of evidence" (a wide CI, a non-significant McNemar result, an
  underpowered cell) must be distinguished in prose from "evidence of no
  effect" (a tight CI centered near zero) -- the two are never
  interchangeable.
* A result may be called "replicated across datasets" only if the
  COMPLETE frozen evidence (all relevant cells/pairs, not a subset)
  supports that wording -- a partial pattern across 2 of 3 datasets is
  reported as a partial pattern, not a replication.
* No generalization beyond: MedMNIST (the specific sub-datasets actually
  used), the tested architectures (ResNet variants actually trained),
  the fixed augmentation policy actually applied, the three training
  seeds actually used, the resolutions actually tested (28/64px primary,
  128px trend-only per Block D), and the N=50 TTA-view endpoint actually
  evaluated.
* Any seed showing behavior inconsistent with the other two seeds in the
  same cell/pair must be explicitly disclosed, never silently averaged
  away.
* Any clean-performance tradeoff already present in the frozen result
  (e.g., TTA improving robustness metrics while degrading clean
  accuracy, if the persisted fields show this) must be disclosed, not
  omitted for a cleaner narrative.
* Secondary cross-condition findings must be visually and textually
  separated from preregistered findings in every future report -- never
  merged into one combined table or one combined verdict.
* No hypothesis, threshold, family, exclusion, or multiplicity rule may
  be modified after this unsealing -- H1-H4's pre-registered text in
  `docs/research_plan.md`, the frozen SAP, and the frozen addendum spec
  remain exactly as committed; any post-hoc change is an exploratory
  finding, explicitly labeled as such, per `CLAUDE.md` rule 4, never a
  silent substitution for the confirmatory result.

## 9. Required limitations and incident disclosure

The eventual `phase2b_final_test_scientific_interpretation.md` MUST
disclose, at minimum:

* Only three training seeds per cell -- sample-level paired tests within
  a cell do not substitute for a model-seed population replication
  study.
* Limited dataset/architecture coverage (the specific MedMNIST subsets
  and ResNet variants actually used, not "medical imaging" broadly).
* A fixed augmentation policy and a fixed TTA view budget (N=50) --
  results do not generalize to other budgets or policies without
  further evidence.
* The cross-condition addendum was frozen AFTER all validation-stage
  results were already observed, but BEFORE the official test split was
  opened (per `configs/final_test_cross_condition_addendum.yaml`'s own
  provenance block) -- it is a post-validation, pre-test-specified
  secondary analysis, never described as originally preregistered.
* The accidental final-test access incident for cell 1 (attempt 1,
  aborted; see `docs/phase2b_final_test_accidental_access_incident.md`).
* The two failed final-test engineering attempts (cell 1 attempt 2:
  pre-access authorization-verification defect, zero test-split access;
  cell 2 attempt 1: semantic-verification failure caught before
  persistence) and their non-persisted outcomes (see
  `docs/phase2b_final_test_attempt2_preaccess_failure.md`,
  `docs/phase2b_final_test_semantic_verification_incident.md`).
* The shared-aggregation-contract correction (Phase 2B.6J/K) and the
  validation-metric-reconciliation mechanism it required (see
  `docs/phase2b_final_test_semantic_metric_contract_freeze.md`,
  `docs/phase2b_validation_metric_reconciliation_audit.md`).
* The fact that ALL 39 canonical final-test results (and, downstream,
  all seven analysis artifacts) were produced under the FINAL, corrected
  evaluator/aggregation pipeline -- cell 1 alone completed under its
  historical generation-3 authorization/pre-fix fingerprints, with
  compatibility independently established via 56/56 recomputation
  checks (Phase 2B.6J/K), never assumed.
* That no scientific result from any of the seven sealed artifacts was
  examined by any person or process before this controlled-unsealing
  phase actually runs.

## 10. Mechanical reporting

Every scientific value in the future `phase2b_final_test_scientific_results.md`
and `artifacts/final_test_scientific_summary.json` must be extracted
programmatically, directly from the seven committed result artifacts'
JSON fields, by the generator implemented per sec.13. No hand
transcription, no manual retyping of a number, no paraphrase of a value
by a human or an LLM without a traceable extraction step.

## 11. Planned future outputs (frozen names, not yet created)

* `artifacts/final_test_scientific_summary.json`
* `docs/phase2b_final_test_scientific_results.md`
* `docs/phase2b_final_test_scientific_interpretation.md`

None of these three files exists yet. Creating them is explicitly out of
scope for this task and requires separate authorization.

## 12. Separation of generated fact and interpretation

* `final_test_scientific_summary.json`: machine-readable facts only --
  every extracted number, every ID, every hash, structured for
  programmatic reuse (e.g., by a later plotting or manuscript-drafting
  task). No prose.
* `phase2b_final_test_scientific_results.md`: the complete generated
  tables (sec.5/sec.6) and factual statistical reporting in prose form
  -- states what the numbers ARE, never what they MEAN beyond the
  literal frozen-endpoint definition.
* `phase2b_final_test_scientific_interpretation.md`: cautious
  interpretation, the sec.8 rules applied, the sec.9 limitations
  disclosed, and claim adjudication (which claims the evidence actually
  supports, at what strength, with what caveats).
* No manuscript prose, abstract, or conference-positioning claim may
  appear in any of these three files during this phase -- that is a
  further, separately-authorized step beyond even controlled unsealing.

## 13. Pre-unsealing implementation requirement

Before the generator is ever pointed at the seven real sealed artifacts,
a deterministic report-generator module and its synthetic/temporary-
repository test suite must be implemented and committed. Requirements:

* The generator's core extraction functions must be pure (same input
  bytes -> same output structure, no hidden state, no RNG).
* Before extracting anything, the generator must independently re-verify
  every artifact's manifest (`verify_analysis_artifact_manifest`) and
  schema (`validate_analysis_result_schema`/
  `validate_cross_condition_result_schema`), and re-check both
  authorization artifacts' hashes (sec.1) against the values frozen in
  this document -- any mismatch halts with no output produced.
* The generator must assert the sec.3 complete-reporting counts before
  writing any output.
* The generator's own tests must run ONLY against synthetic fixtures
  (fabricated JSON matching the real schema, in `tmp_path`) -- never
  against the real seven artifacts -- so that the generator's own CI/
  test suite never itself becomes an unsealing event.
* Only after this generator and its tests are implemented, reviewed, and
  committed does the actual real-data extraction run become a separate,
  explicitly authorized step.
