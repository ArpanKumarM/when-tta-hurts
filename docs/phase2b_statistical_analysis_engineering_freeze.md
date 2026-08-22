# Phase 2B Statistical-Analysis Engineering Freeze

**Recorded: 2026-08-22.** This document is a faithful mechanical translation
of the pre-existing frozen scientific record into an implementation
mapping. It does not introduce any new hypothesis, test, threshold,
pooling strategy, or interpretation. Every item traces to a specific
controlling document; every unresolved gap is listed explicitly rather
than filled in.

## 0. Verification of the frozen record (Part A)

| Check | Result |
|---|---|
| HEAD | `99b20be088278f21f2ddbf72007d63a12e3062c1` |
| Working tree | clean |
| Training canonical | 39/39 |
| Validation-evaluation canonical | 39/39 (A=24, B=6, C=3, D=6) |
| Every canonical evaluation uses current evaluator fingerprint | yes, `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` |
| Test split observed | never (`split=validation` on all 46 ledger rows; no test-split code path exists in `validation_evaluation.py`) |

### Controlling-document provenance

| Document | Last-touch commit | SHA-256 | Predates first real results commit (`ea31d82`)? | Scientific meaning changed after? |
|---|---|---|---|---|
| `docs/statistical_analysis_plan.md` | `ce4c962` (2026-08-17) | `566840a15e11d3fafe4aa781e705e2c8ac005dd21c5c79c93da03bcb74b69fca` | yes (`ce4c962` is an ancestor of `ea31d82`) | no — zero edits since `ce4c962` |
| `docs/phase2b_protocol.md` | `ce4c962` (2026-08-17) | `8e9e2abf69610da214b07c89b75eef6bb16f6dbd906b8d803e87b0475482c7b6` | yes | no — zero edits since `ce4c962` |
| `docs/experimental_protocol.md` | `ce4c962` (2026-08-17) | `9bdb3f7659758a1d693bd65ad240b853b0b4719825c3c0940a4c7c2fea1b70cb` | yes | no — its own test-firewall section explicitly SUPERSEDED by `phase2b_protocol.md` sec.5, not silently changed |
| `docs/research_plan.md` | `ce4c962` (2026-08-17) | `7936264cbb7d04af68b4ea702db9f283efabfda04bc10636fdb7fa059bfd949d` | yes | no |
| `configs/experiment_matrix.yaml` | `ce4c962` (2026-08-17) | `3a02eb7695757b2a1d3dadc6fd6c61f3ec588c6c5def3b21e9edad4a97e7ad93` | yes | no |
| `configs/validation_evaluation.yaml` | `27fe462` (2026-08-19) | `5037bc2f0ebd2723e984947485d6e332be842b69a4d929266eb29882affcc586` | **no** — `27fe462` is NOT an ancestor of `ea31d82` (the runner-implementation commit predates this freeze) | see below |

**Disclosure on `configs/validation_evaluation.yaml`:** this file has its
own three-commit freeze history (`124bd58` TTA seed → `163f967` batching →
`27fe462` probability-native metric contract). The final freeze (`27fe462`)
postdates the runner's first implementation commit. This is **not** a case
of a confirmatory statistical requirement being changed after observing
results: `27fe462` corrected a genuine evaluator defect (the
double-softmax metric-contract bug, documented in
`docs/phase2b_validation_evaluation_metric_contract_incident.md`) that
affected exactly one non-canonical attempt
(`A-pathmnist-28px-batchnorm-policy-none-s0` attempt 3), which was amended
`canonical_eligible=False` and never used to inform any decision. The
`docs/statistical_analysis_plan.md` file itself — the document that
actually defines hypotheses, tests, alpha, and multiplicity handling — was
never touched after its `ce4c962` freeze. **No confirmatory statistical
requirement was added or materially changed after observing results.**

## 1. Central structural finding

**`docs/statistical_analysis_plan.md` defines "confirmatory" as: cells in**
**the frozen matrix, evaluated once per seed, on the official test set.**
**`docs/phase2b_protocol.md` sec.3 defines the "primary confirmatory**
**endpoint" as clean *test* accuracy versus TTA, and sec.5's frozen**
**9-step order places the single test-split unlock strictly after**
**Validation-Gated TTA (H4) is developed and frozen (steps 5-6), before**
**which "no official test predictions or metrics may be generated, for**
**any block."**

The validation-stage results currently available (Blocks A/B/C/D, all 39
cells) are **not** the confirmatory analysis population under the frozen
plan's own definition. They exist to (a) train and validate checkpoints
under validation-based early stopping, (b) develop and freeze
Validation-Gated TTA using validation data only (Phase 2B.4, not yet
started), and (c) support pre-test descriptive characterization. **No
confirmatory H1-H4 verdict can be produced from validation-stage data,
regardless of any other design choice.** This is not an ambiguity to
resolve — it is an explicit, frozen scoping decision, and this document
does not challenge or work around it.

## 2. Complete requirement map

One row per planned analysis family. "Confirmatory-eligible" means
eligible under the frozen plan **once test-set data exists** — never
under current validation-stage data.

### H1 — Normalization (BatchNorm vs GroupNorm)

| Field | Value |
|---|---|
| Identifier | H1 |
| Status | Confirmatory-eligible (test-set only); currently no test data exists |
| Scientific question | Does TTA degradation differ between BatchNorm and GroupNorm under controlled architecture/training? |
| Eligible blocks/cells | Block A only: `small_cnn`, `{pathmnist, bloodmnist}` x `{28, 64}` x `{batchnorm, groupnorm}` x seeds `[0,1,2]` (Block D 128px strengthens H2, not H1, as a trend only) |
| Dataset population | pathmnist, bloodmnist (Block A) |
| Unit of analysis | Per confirmatory cell (dataset x resolution x normalization), each a paired clean-vs-TTA comparison on the *same* test samples from the *same* trained model, per seed |
| Pairing structure | Paired: clean prediction and TTA prediction on the identical sample, identical model |
| Independent experimental unit | The trained model (dataset x resolution x normalization x seed) — **not** individual test images. Three independent units (seeds) per cell |
| Endpoint | Delta accuracy (TTA − clean), in percentage points |
| Clean/TTA condition | `clean` vs `naive_tta` (mixed policy) |
| Registered N | 50 (primary); full curve 1/2/5/10/25/50/100 is secondary/descriptive |
| Registered aggregator | mean_probability (primary) |
| Normalization applicability | Both BatchNorm and GroupNorm (H1's comparison IS the normalization axis) |
| Directionality | Not specified as one-sided; SAP gives no directional pre-commitment — treat as requiring a two-sided/undirected specification unless the user supplies one |
| Null/alternative hypotheses | **Not formally stated as H0/H1 test statistics anywhere in the frozen record.** `docs/research_plan.md` states H1 in prose ("degradation differs") but the SAP's precisely specified test (paired bootstrap CI, McNemar) tests **within-cell** clean-vs-TTA, not a **between-condition** (BatchNorm-vs-GroupNorm) comparison of two deltas. See Section 3, item 1. |
| Statistical test | Within-cell: paired bootstrap (≥10,000 resamples) for 95% CI on delta accuracy; McNemar's test (exact or continuity-corrected depending on cell counts) on the clean/TTA 2x2 contingency table — both **fully specified**. Between-condition (BatchNorm vs GroupNorm): **not specified anywhere** — see Section 3 |
| Exact or asymptotic | McNemar: exact or continuity-corrected depending on cell counts (frozen rule, not a free choice at analysis time — determined mechanically by observed discordant-pair counts); bootstrap: nonparametric resample, not asymptotic |
| Alpha | Not stated numerically anywhere in the frozen record. Conventionally 0.05 is implied by "95% CI" language, but this is not an explicit alpha declaration for a hypothesis test — see Section 3, item 2 |
| Multiplicity family | "Within each hypothesis's confirmatory test set" — i.e., H1's family is all Block-A cells tested under H1 |
| Correction method | Benjamini-Hochberg FDR, within the H1 family; both corrected and uncorrected p-values reported |
| Effect-size definition | Raw delta-accuracy magnitude; harm rate; rescue rate; standardized differences for NLL/ECE/Brier "where meaningful" (not further defined) |
| CI method and level | Paired bootstrap, 95%, ≥10,000 resamples |
| Ties/zero differences | Not specified — McNemar's handling of zero-count cells (exact test degenerates when no discordant pairs exist) is not addressed in the frozen record — see Section 3, item 3 |
| Missing/failed/ineligible-attempt policy | Amendment-excluded and failed/aborted attempts are never canonical inputs (enforced by `check_evaluation_skip`/`is_evaluation_canonical_ineligible`, already production code) |
| Exclusion rules | Pilot seed 314159 permanently excluded; pilot TTA seed 271828 never reused |
| Seed aggregation rule | Per-seed CI/test computed individually, then summarized across the 3 seeds descriptively (median + range, or mean +/- SD) — **explicitly not** further significance-tested across seeds |
| Pooled or dataset-specific | Not specified whether H1 pools pathmnist+bloodmnist or reports them separately — see Section 3, item 4 |
| Validation-stage or final-test-stage | Final-test-stage only (confirmatory); validation-stage Block A results may only be reported descriptively |
| Frozen interpretation rule | A BatchNorm-vs-GroupNorm difference is evidence *consistent with* a running-statistics explanation, never *proof* of it (explicit critical note in `research_plan.md`) |

### H2 — Resolution

| Field | Value |
|---|---|
| Identifier | H2 |
| Status | Confirmatory-eligible (test-set only) for 28px-vs-64px; 128px (Block D) strengthens a trend claim only, never stands alone as confirmatory even with test data, per its own frozen scope |
| Scientific question | Does TTA degradation decrease as resolution increases 28->64(->128)? |
| Eligible blocks/cells | Block A (28px, 64px) primary; Block D (128px) trend-only if all 3 seeds complete (already true: 6/6 complete) |
| Dataset population | pathmnist, bloodmnist |
| Unit of analysis | Per cell (dataset x resolution x normalization), same paired structure as H1 |
| Pairing structure | Paired, same model/samples |
| Independent experimental unit | Trained model (seed) |
| Endpoint | Delta accuracy |
| Clean/TTA condition | `clean` vs `naive_tta` |
| Registered N | 50 primary |
| Registered aggregator | mean_probability |
| Normalization applicability | Both, but H2's confirmatory comparison is across resolution, not normalization |
| Directionality | Implied one-directional in the hypothesis wording ("decreases as resolution increases") but no formal one-sided test is specified in the SAP — see Section 3, item 5 |
| Null/alternative hypotheses | Same gap as H1: the cross-resolution comparison test is not specified, only the within-cell clean-vs-TTA test |
| Statistical test | Same within-cell machinery as H1 (fully specified); cross-resolution comparison unspecified |
| Alpha / multiplicity / correction | Same as H1, own BH-FDR family for H2's cells |
| Effect-size / CI | Same as H1 |
| Ties/zero differences | Same gap as H1 |
| Missing/exclusion/seed-aggregation | Same as H1 |
| Pooled or dataset-specific | Not specified — see Section 3, item 4 |
| Validation-stage or final-test-stage | Final-test-stage confirmatory; 28px/64px/128px validation-stage comparisons already reported are explicitly labeled descriptive-only in `docs/phase2b_validation_evaluation_block_d_audit.md` |
| Frozen interpretation rule | 128px alone (without test-set 3-seed completion, which does not apply here since it's validation-stage) never stands as confirmatory evidence for H2; resolution effects cannot be cleanly separated from checkpoint/seed effects with only 3 seeds (explicit disclosure already made in the Block D audit) |

### H3 — Policy matching

| Field | Value |
|---|---|
| Identifier | H3 |
| Status | Confirmatory-eligible (test-set only) |
| Scientific question | Does training-time/test-time augmentation policy matching reduce TTA degradation vs. unmatched? |
| Eligible blocks/cells | Block B (matched, 28px, BatchNorm, pathmnist+bloodmnist) vs Block A's matching unmatched comparator (`policy=none`, same dataset/resolution/normalization/seed) |
| Dataset population | pathmnist, bloodmnist |
| Unit of analysis | Per cell, paired clean-vs-TTA within each of matched and unmatched arms |
| Pairing structure | Paired within each arm (clean vs TTA, same model/samples); the matched-vs-unmatched comparison itself compares two *different* trained models (same dataset/resolution/normalization/seed, different training policy) — this is **not** a paired comparison in the SAP's sense, since the underlying model differs |
| Independent experimental unit | Trained model (seed) x training policy |
| Endpoint | Delta accuracy |
| Clean/TTA condition | `clean` vs `matched_tta`/`naive_tta` |
| Registered N | 50 primary |
| Registered aggregator | mean_probability |
| Normalization applicability | BatchNorm only (Block B's frozen scope) |
| Directionality | Not formally one-sided in the SAP, though H3's prose framing anticipates a directional effect ("close to a truism") |
| Null/alternative hypotheses | Same structural gap: the SAP specifies within-arm clean-vs-TTA tests, not a formal matched-vs-unmatched comparison test |
| Statistical test | Within-arm: bootstrap CI + McNemar (fully specified). Matched-vs-unmatched delta-of-deltas: **not specified** — see Section 3, item 1 |
| Alpha / multiplicity / correction | Own BH-FDR family for H3's cells |
| Effect-size / CI | Same as H1; already reported descriptively for the validation-stage canary/closure (e.g. harm-change = delta_B - delta_A) but never significance-tested |
| Ties/zero differences | Same gap as H1 |
| Missing/exclusion/seed-aggregation | Block A's checkpoints are reused as the unmatched comparator per the frozen design (`configs/experiment_matrix.yaml` `reuses_checkpoints_from`) |
| Pooled or dataset-specific | Not specified |
| Validation-stage or final-test-stage | Final-test-stage confirmatory; the validation-stage Block A-vs-B comparison already reported (canary + closure docs) is explicitly descriptive |
| Frozen interpretation rule | H3's scientific value is in quantifying effect size relative to H1/H2, not testing mere existence of the matching effect (explicit critical note) |

### H4 — Validation-Gated TTA

| Field | Value |
|---|---|
| Identifier | H4 |
| Status | **Draft, not approved.** Algorithm not yet developed or frozen (Phase 2B.4 has not started) |
| Scientific question | Does a validation-selected, per-sample-rejecting, clean-fallback TTA policy reduce clean-correct-to-TTA-wrong harm rate vs. naive mean aggregation? |
| Eligible blocks/cells | Not yet defined — depends on the not-yet-designed gating algorithm |
| Required comparison set | Clean; naive TTA (mean); original-anchored TTA; BN-adapted TTA (BatchNorm only); validation-gated TTA — all five required, per `research_plan.md`'s explicit "required comparison set (corrected)" |
| Endpoint | Harm rate (clean-correct -> TTA-wrong), not raw accuracy — explicit framing requirement |
| Everything else | Cannot be specified until the gating algorithm and its thresholds are frozen (Phase 2B.4/2B.5 step 6) |

### Block C — Positive-control reproduction

| Field | Value |
|---|---|
| Identifier | Block C (not H1-H4) |
| Status | Confirmatory-status cell per the matrix (`C_positive_control_reproduction`), but **not a formal H1-H4-style hypothesis test with a frozen null/alternative** |
| Scientific question | Does this project's pipeline reproduce the source paper's sole reported positive TTA result (ResNet-18/DermaMNIST, ~+1.6pp at N=50)? |
| Eligible cells | dermamnist, resnet18, 28px, batchnorm, seeds [0,1,2] |
| Comparison target | An **external** point estimate (+1.6pp) from the source paper — no access to that paper's own resampling/seed variance |
| Statistical test | **None specified.** The SAP's paired-bootstrap/McNemar machinery applies to within-cell clean-vs-TTA (which is directly computable), but comparing our result to the *external* +1.6pp figure has no frozen inferential procedure — this is inherently a descriptive comparison, since the external estimate carries no usable variance/sampling information from this project's perspective |
| Frozen interpretation rule | Already established in `docs/phase2b_validation_evaluation_block_c_audit.md`: "not reproduced under our frozen operationalization" language, explicit refusal to claim the source paper is wrong, mechanical listing of protocol differences |
| Validation-stage or final-test-stage | Confirmatory-status per the matrix implies test-set evaluation; the already-completed validation-stage 3-seed result is explicitly descriptive-only per its own audit document |

### Block D — Conditional 128px

| Field | Value |
|---|---|
| Identifier | Block D (supports H2 only) |
| Status | Approved-conditional; gate passed; all 6 cells trained and evaluated |
| Role | Strengthens H2 as a **trend claim only** — "does not stand alone as confirmatory evidence unless completed with all 3 seeds" (which it is), but even then it strengthens, does not independently confirm |
| Registered inferential family | Part of H2's family if/when H2 is confirmatorily tested on test data; not a standalone inferential family |

### Secondary/descriptive analyses (BN adaptation, alternative aggregators, N-curves)

Per `docs/phase2b_protocol.md` sec.3, **explicitly and unambiguously
labeled "Secondary/descriptive analyses (all preregistered, none
confirmatory-primary)"**:

1. Scaling curve (all 7 registered N, mean-probability, mixed policy)
2. Augmentation-strategy ablation (geometric/intensity/mixed, N=25)
3. Aggregation ablation (mean/majority/confidence-weighted, N=25)
4. Original-anchored condition (source paper's Appendix B baseline)
5. BN-adapted condition (BatchNorm only; source paper's Appendix B baseline)

**None of these carry a confirmatory verdict, ever, regardless of split.**
They are preregistered secondary/descriptive analyses. Calibration metrics
(NLL, ECE, Brier) are explicitly framed by the SAP as **effect-size /
descriptive outputs** ("standardized differences for calibration metrics
... where meaningful"), not independent endpoints with their own
confirmatory hypothesis tests.

### Validation set's designated statistical role

Per `docs/phase2b_protocol.md` sec.5's frozen 9-step order, the validation
set is used for: (a) early-stopping during training, (b) developing and
freezing Validation-Gated TTA (H4), and (c) any validation-only
engineering checks. **It is never described as a confirmatory-inference
population, nor as a formal "method selection via statistical test"
population** — method development for H4 is empirical/engineering, not
itself a hypothesis-tested selection procedure in the frozen record.

### Reserved exclusively for the final test set

Per the frozen 9-step order: **the entire primary confirmatory endpoint**
(delta accuracy under mean-probability TTA at N=50, the paired
bootstrap/McNemar tests, the BH-FDR-corrected H1-H4 verdicts) is reserved
exclusively for the single final test-set pass (step 8), performed only
after Validation-Gated TTA is frozen (step 6). Nothing computed before
that point may be labeled confirmatory.

## 3. Ambiguity gate (Part C)

| # | Item | Classification | Treatment |
|---|---|---|---|
| 1 | Cross-condition comparison test for H1 (BatchNorm vs GroupNorm), H2 (resolution), H3 (matched vs unmatched) "differs" claims | **Underspecified** | The SAP specifies only the within-cell clean-vs-TTA paired test. No frozen procedure exists for comparing one cell's delta-accuracy against another's. **Not implemented.** Reported only as side-by-side per-cell point estimates/CIs, explicitly labeled descriptive, never as a formal "differs" verdict |
| 2 | Numeric alpha for hypothesis tests | **Underspecified** | "95% CI" implies an informal 0.05, but no explicit alpha is declared for a significance-test decision rule. **Not assumed.** Any implementation exposes the CI/p-value; it does not render an accept/reject verdict |
| 3 | McNemar behavior under zero discordant pairs / ties | **Underspecified** | Frozen text says "exact or continuity-corrected... depending on cell counts" but does not give the exact decision rule (e.g., exact binomial when the smaller discordant count is below some threshold). **Not invented.** Implementation exposes discordant-pair counts and both statistics where computable; documents when McNemar is undefined (zero discordant pairs) rather than substituting a default |
| 4 | Whether H1/H2/H3 pool pathmnist+bloodmnist or report dataset-specific | **Absent** | Never stated. **Not resolved by this document.** Both datasets' per-cell results are computed and reported side-by-side; no pooled statistic is invented |
| 5 | One-sided vs two-sided test direction for H2's "decreases as resolution increases" | **Underspecified** | Hypothesis wording implies direction; SAP's specified tests (bootstrap CI, McNemar) are not inherently directional. **Two-sided/undirected retained** as the only choice that requires no invention; a one-sided test is explicitly forbidden by the task's own constraints |
| 6 | Bootstrap resampling unit | **Fully specified** | "Resample test samples with replacement" — the unit is the individual (paired) test sample within a fixed cell/seed. This is unambiguous and does not risk pseudoreplication because the *inference target* remains a single trained model's clean-vs-TTA comparison, not a claim about images as independent experimental replications across models |
| 7 | Threshold for "meaningful harm" | **Absent** | No frozen numeric threshold anywhere. **Not invented.** Harm rate and rescue rate are reported as continuous descriptive quantities; no severity threshold or categorical harm/no-harm cutoff is applied |
| 8 | Preferred N or aggregator beyond N=50/mean-probability primary | **Fully specified** | N=50, mean-probability is unambiguously the primary/confirmatory pair; all others are explicitly secondary/descriptive (Section 2) |
| 9 | Post-hoc subgroup analyses | **Absent, and forbidden** | None preregistered; none will be added |
| 10 | Confirmatory eligibility of the validation-stage data itself | **Fully specified (by exclusion)** | Explicitly test-set-only per Section 1; validation-stage data is descriptive-only by the frozen plan's own terms, not an open question |

**Materiality assessment:** item 1 (the cross-condition comparison test)
would materially affect any confirmatory H1/H2/H3 "differs" verdict.
**This blocks producing any such verdict, now or on future test data,
until the user supplies the missing test specification.** It does **not**
block building or running the fully-specified within-cell machinery
(bootstrap CI, McNemar, effect sizes on clean-vs-TTA within a single
cell), which requires no invention and is exactly what Sections 4-5 below
implement. Since no test-set data exists at all yet, no confirmatory
verdict of any kind is currently produced regardless.

## 4. Analysis populations and canonical-attempt selection

- Every analysis input is resolved through the same production selection
  logic already used by the evaluation pipeline:
  `resolve_canonical_training_completion()` for checkpoints,
  `check_evaluation_skip()` / `is_evaluation_canonical_ineligible()` for
  evaluation-ledger rows, requiring `status=completed`,
  `confirmatory=True`, current evaluator fingerprint, and amendment
  eligibility.
- No pilot run (seed 314159), no amendment-excluded attempt, no
  failed/aborted attempt is ever an analysis input.
- Analysis inputs are read exclusively from
  `artifacts/validation_evaluation/<run_id>/attempt_NNN/` directories
  (`predictions.npz`, `metrics.json`, `metadata.json`,
  `artifact_manifest.json`) — never from any test-split artifact or path.

## 5. Exact mathematical formulas (within-cell, fully specified)

- **Delta accuracy** (per seed, per cell): `accuracy(TTA_condition, N) - accuracy(clean)`.
- **Paired bootstrap CI**: resample the paired (clean-correct, TTA-correct)
  indicator vectors jointly (same resampled index set applied to both), with
  replacement, >=10,000 resamples; report the 2.5th/97.5th percentiles of
  the resampled delta-accuracy distribution as the 95% CI.
- **McNemar's test**: build the 2x2 table over paired
  (clean-correct, TTA-correct) outcomes -- cells are (correct,correct),
  (correct,wrong), (wrong,correct), (wrong,wrong). Statistic uses only the
  two discordant cells (correct,wrong) and (wrong,correct). Use the exact
  binomial form when the discordant total is small; continuity-corrected
  chi-square otherwise (exact numeric threshold for "small" is the
  underspecified item #3 above -- both raw discordant counts are always
  reported so the reader can judge).
- **Effect sizes**: raw delta-accuracy magnitude; harm rate = P(clean
  correct, TTA wrong); rescue rate = P(clean wrong, TTA correct); both
  already implemented and independently validated in
  `src/when_tta_hurts/metrics.py` (Phase 2A audit,
  `tests/test_metrics_independent_validation.py`). Standardized
  differences for NLL/ECE/Brier: not further specified ("where
  meaningful") -- reported as raw deltas only, no standardization formula
  invented.
- **Multiclass Brier score**: `mean over samples of sum over classes of
  (predicted_probability - one_hot_label)^2` (frozen in
  `docs/phase2b_protocol.md` sec.3, already implemented).

## 6. Multiplicity families and correction

- One family per hypothesis (H1, H2, H3; H4 not yet defined) -- **not** a
  single family across all hypotheses and cells.
- Benjamini-Hochberg FDR correction within each family.
- Both corrected and uncorrected p-values always reported together.
- Exploratory-only tests (anything outside the frozen matrix) are excluded
  from every corrected family and separately labeled.

## 7. Missing-data handling

- A cell with no canonical current-fingerprint completion is reported as
  **missing**, never silently imputed or substituted with a stale/older-
  fingerprint completion.
- A family requiring N cells with fewer than N eligible completions is
  reported as **incomplete for confirmatory purposes** -- descriptive
  reporting of whatever subset exists remains possible but is explicitly
  labeled partial.

## 8. Deterministic output ordering

Row order follows `configs/experiment_matrix.yaml`'s frozen deterministic
execution order (block A -> B -> C -> D; within each block, dataset-major,
then resolution, then normalization, then seed) -- the same ordering rule
already frozen in `docs/phase2b_protocol.md` sec.7 for execution, reused
for reporting.

## 9. Allowed and forbidden interpretations

**Allowed:**
- Reporting per-cell, per-seed point estimates, CIs, and McNemar results
  once test data exists.
- Reporting three-seed descriptive summaries (individual values, mean,
  sample SD, min, max) for any registered endpoint, at any stage.
- Reporting validation-stage results as descriptive characterization.
- Reporting Block C's comparison to the external +1.6pp figure as a
  directional descriptive statement ("not reproduced," "consistent,"
  "seed-sensitive/inconclusive") per the already-frozen language rules.

**Forbidden:**
- Any confirmatory H1-H4 verdict from validation-stage data.
- Any cross-condition "differs" significance verdict for H1/H2/H3 at any
  stage, until the missing test specification (Section 3, item 1) is
  resolved by the user.
- Any significance test on a three-seed descriptive summary.
- Any pooled-dataset statistic not explicitly specified.
- Any post-hoc subgroup, one-sided test, alternate alpha, or invented
  threshold.
- Any statistical computation on test-split data before the single
  authorized final pass.

## 10. Explicit separation: validation analysis vs. final-test analysis

| | Validation-stage (now) | Final-test-stage (future, gated) |
|---|---|---|
| Data source | `artifacts/validation_evaluation/` only | Official test split, accessed exactly once, after H4 is frozen |
| Confirmatory verdicts | Never | The only stage where H1-H4 confirmatory verdicts may be produced, and only for the fully-specified within-cell machinery (Section 5); the cross-condition "differs" verdict remains blocked (Section 3 item 1) even then, absent further specification |
| Statistical machinery invoked | Descriptive summaries only (mean, sample SD, min, max, individual values) | Paired bootstrap CI, McNemar, BH-FDR correction, effect sizes |
| Engineering scope of this freeze | Build read-only infrastructure now; run in descriptive/plan mode only | Real-analysis mode exists but is never invoked in this task |

## 11. Items that remain descriptive because they were not preregistered precisely enough

- H1/H2/H3's literal "differs between conditions" claims (Section 3, item 1).
- Any pooled pathmnist+bloodmnist statistic (Section 3, item 4).
- Any categorical "meaningful harm" threshold (Section 3, item 7).
- Standardized-difference formulas for calibration metrics beyond raw deltas.
- Block C's comparison to the external source-paper figure (no shared
  variance information available; inherently descriptive, not a formal
  test, by nature of comparing to a fixed external point estimate).
- All Section 2 "secondary/descriptive analyses" (scaling curve,
  augmentation ablation, aggregation ablation, original-anchored,
  BN-adapted) -- these are permanently descriptive/secondary by the
  frozen protocol's own explicit labeling, not a temporary gap.
