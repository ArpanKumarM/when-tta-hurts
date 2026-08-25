# Phase 2B.10A Part F — Submission Gap Analysis

**Status: analysis only. Nothing below is authorized or executed.** No
new experiment, protocol, or code change is made by this document. All
compute/time estimates are rough extrapolations from the already-
measured per-run wall-clock figures in `docs/pilot_audit.md` and
`docs/compute_budget.md` (Apple M3 Pro, 18GB unified memory, PyTorch
MPS backend: **~3.3 min/run at 28px, ~13.9-15.0 min/run at 64px, 30
epochs, batch 256, all measured**; the frozen 128px kill criterion is
**<=90 min/run** per `docs/phase2b_protocol.md` sec.6, not yet measured
since Block D's own gate has not required it beyond the already-
executed 6 cells). Every estimate below states its arithmetic so it can
be checked, not just asserted.

This document evaluates hypothetical additions per
`paper/reviews/meta_review.md`'s finding that the manuscript is not
currently competitive for CVPR main. It does not recommend actually
targeting CVPR main -- see the meta-review for the venue-fit
discussion -- it answers Part F's literal question ("what would a
credible CVPR-main attempt require") as a gap analysis, independent of
whether that is the recommended path.

## How to read the tables below

For every candidate addition: the exact scientific question it would
answer, its expected publication value, its risk of post-hoc
overfitting (i.e., whether choosing to run it *because* of what the
already-unsealed results show would itself be a violation of this
project's no-test-set-tuning discipline), whether it requires a new
frozen protocol before any test-split access, a rough compute/storage
estimate, and a priority tier.

---

## Tier 1 — Must-have for a credible CVPR-main attempt

### 1. Label-preservation audit

* **Exact question:** For the fixed mixed augmentation policy (rotation
  +/-15deg, random resized crop 0.8-1.0x, brightness/contrast jitter
  +/-0.3, Gaussian blur) at the view counts actually used, what
  fraction of augmented views still contain the labeled diagnostic
  structure, as judged by a human annotator or a simple automated
  proxy (e.g. lesion-region overlap for DermaMNIST)?
* **Expected publication value:** Very high. This directly answers the
  single most damaging open question raised by Reviewer 1 (sec.4) and
  Reviewer 2 (sec.4): whether the observed harm is a property of TTA
  itself or an artifact of an overly severe augmentation policy for
  these modalities. Without it, the paper's causal framing is
  incomplete regardless of venue.
* **Risk of post-hoc overfitting:** Low, if the audit protocol
  (sample size, annotation criteria, pass/fail threshold) is frozen
  *before* looking at which cells showed the most/least harm. High, if
  the sample of augmented views to audit is chosen after seeing which
  cells were most harmed (which would let the choice of what to audit
  be steered toward a convenient narrative).
* **Requires a new frozen protocol:** Yes -- a small one (annotation
  criteria, sample size, dataset/policy coverage), but it does not
  require re-opening the test split for training or re-running any
  confirmatory training.
* **Rough compute/storage estimate:** Minimal compute (a few hundred
  augmented-view renders, no training). The bottleneck is annotation
  time, not compute; not estimable from the project's existing
  wall-clock measurements since no comparable prior task exists in
  this project's history.
* **Priority: 1 (highest).**

### 2. Per-augmentation-component ablation (geometric-only vs.
intensity-only vs. mixed)

* **Exact question:** Does harm come predominantly from the geometric
  transforms (rotation, crop), the intensity transforms
  (brightness/contrast/blur), or their combination?
* **Expected publication value:** High. Directly complements the
  label-preservation audit with a quantitative decomposition, and
  `configs/experiment_matrix.yaml` already defines `geometric` and
  `intensity` as named policies distinct from `mixed`, so this is
  evaluating an already-specified condition, not inventing a new one.
* **Risk of post-hoc overfitting:** Low -- these policies are already
  named and frozen in the existing matrix configuration; running TTA
  evaluation under them is not a new statistical test invented after
  seeing results, provided the same preregistered-style analysis (paired
  bootstrap + McNemar) is applied uniformly.
* **Requires a new frozen protocol:** No new training protocol --
  reuses existing Block A/B/C/D checkpoints. A short analysis-plan
  addendum (which existing hypothesis family these evaluations belong
  to, confirmatory or exploratory) should still be frozen before
  execution, consistent with this project's confirmatory/exploratory
  discipline.
* **Rough compute/storage estimate:** No new training. Additional TTA
  evaluation only, reusing existing checkpoints; per
  `docs/compute_budget.md`'s logit-cache accounting, each additional
  policy costs roughly the same cache footprint as one existing policy
  (~1.1 GiB before Block D, ~1.5 GiB including Block D, per policy) --
  two additional policies (geometric-only, intensity-only) is
  approximately **+2-3 GiB** of cache storage and evaluation-only
  compute (no multi-minute training runs), i.e. a small fraction of the
  original training budget.
* **Priority: 1.**

### 3. Report the view-count (N) scaling curve

* **Exact question:** Does naive-TTA harm increase, saturate, or
  reverse as N grows from 1 to 100 views, and is N=50 representative of
  the general pattern or an unrepresentative single point?
* **Expected publication value:** High, and unusually cheap relative to
  its value: `docs/phase2b_protocol.md` sec.3 already lists this as a
  preregistered secondary/descriptive analysis over the same registered
  100-view sequence already used for the N=50 confirmatory condition.
* **Risk of post-hoc overfitting:** Low -- the full view-count sweep
  was preregistered before any test-split access, so reporting it now
  is disclosure, not new analysis.
* **Requires a new frozen protocol:** No -- already frozen.
* **Rough compute/storage estimate:** Likely near-zero additional
  compute. Per `docs/compute_budget.md`'s cache design, all 7
  registered view-count prefixes (1/2/5/10/25/50/100) are derived from
  the *same* cached 100-view logit tensor already computed for the
  N=50 confirmatory condition -- if that cache was populated during the
  original run (as the frozen cache design implies it would have been),
  this is a reporting/extraction task, not a new computation.
* **Priority: 1.**

### 4. BatchNorm-statistics adaptation comparison

* **Exact question:** Does adapting BatchNorm's running statistics
  using the augmented-batch distribution (the source study's own
  Appendix B condition, and Schneider et al. 2020's general mechanism,
  and TENT's more sophisticated variant) reduce the harm observed in
  the unmatched-policy cells, and does it interact with the
  normalization ablation's dataset-dependent reversal?
* **Expected publication value:** High. This is the most direct way to
  convert the paper's unmechanized normalization ablation (Reviewer 1
  sec.9, Reviewer 2 sec.4) into a mechanism-isolating result, and it
  reuses a condition (`bn_adapted_tta`) already named in
  `configs/experiment_matrix.yaml` as a required Appendix-B baseline
  that does not currently appear in the manuscript's reported results.
* **Risk of post-hoc overfitting:** Low if applied uniformly to all
  BatchNorm cells per the already-specified condition; moderate if only
  applied selectively to the cells that would make the strongest
  narrative point.
* **Requires a new frozen protocol:** No new *training* protocol
  (reuses existing checkpoints); the exact BN-adaptation procedure is
  already specified in `docs/phase2b_protocol.md` sec.4 referenced
  material and should be frozen/confirmed unchanged before execution.
* **Rough compute/storage estimate:** No new training; extra forward
  passes per BatchNorm checkpoint to recompute activation statistics
  from augmented batches, then re-run inference. Not directly
  measured in this project, but bounded above by the existing
  ~1.55-second-per-batch TTA inference benchmark (`docs/pilot_audit.md`)
  times a small constant factor for the statistics-recomputation pass;
  expected to be minutes, not hours, in aggregate across all BatchNorm
  checkpoints (18 of the 30 unmatched cells are BatchNorm).
* **Priority: 1.**

---

## Tier 2 — High-value but not mandatory

### 5. Increase seeds from 3 to 5-10

* **Exact question:** Does the within-cell harm finding and the
  secondary DiD patterns hold with more independent seed draws, and
  does the normalization/resolution sign reversal persist or wash out
  with more seeds?
* **Expected publication value:** Moderate-high -- strengthens
  statistical power and directly answers Reviewer 1 sec.3/sec.7's
  concern about the small independent-replicate count, but does not by
  itself resolve the novelty or confound issues.
* **Risk of post-hoc overfitting:** Low if seeds are drawn from a
  predetermined, disjoint pool (as this project's existing seed
  discipline already requires) and not selected after seeing partial
  results.
* **Requires a new frozen protocol:** Yes -- new seed values and an
  amended confirmatory matrix, requiring the same authorization
  discipline as the original matrix.
* **Rough compute/storage estimate:** Going from 3 to 10 seeds
  (+7 seeds) roughly triples the training-run count for every block
  that currently uses 3 seeds. Applied to the pre-Block-D matrix (33
  runs) plus Block D (6 runs): approximately **+91 additional training
  runs** (39 -> 130 total). At measured per-run times (~3.3 min at
  28px, ~13.9-15.0 min at 64px, <=90 min at 128px under the frozen
  128px ceiling), a rough blended estimate is on the order of
  **20-30 additional hours of training wall-clock** on the same single
  M3 Pro machine (exact split depends on how many of the 91 new runs
  land at each resolution), plus a proportional increase in the
  evaluation-cache storage estimated in `docs/compute_budget.md`
  (roughly **3x the existing ~3.3-4 GiB logit-cache estimate**, i.e.
  approximately 10-12 GiB).
* **Priority: 2.**

### 6. Another architecture family

* **Exact question:** Does the normalization/resolution/policy-matching
  pattern observed for SmallCNN (and separately for ResNet-18 in
  BLOCK_C) hold for a third architecture family, breaking the
  architecture/ablation-axis confound Reviewer 1 sec.1 identifies?
* **Expected publication value:** High for addressing the confound;
  moderate for novelty (adding an architecture does not by itself
  answer Reviewer 2's novelty concerns).
* **Risk of post-hoc overfitting:** Low, if the architecture and its
  hyperparameters are chosen and frozen before any evaluation.
* **Requires a new frozen protocol:** Yes -- new architecture
  specification, parameter budget, and training hyperparameters.
* **Rough compute/storage estimate:** Comparable to adding a
  SmallCNN-scale architecture across the existing Block A/B design
  (24 + 6 = 30 additional runs at 28px/64px), i.e. roughly
  **30 x ~3.3-14.5 min ~ 3-7 additional hours** of training wall-clock,
  plus proportional cache storage (~1.1-1.5 GiB more per policy).
* **Priority: 2.**

### 7. Transformation-severity sweep

* **Exact question:** Does harm scale monotonically with augmentation
  severity (e.g. rotation angle, crop range, jitter magnitude), which
  would support a distribution-shift-magnitude explanation, or is it a
  step-function/threshold effect?
* **Expected publication value:** Moderate-high, and complements the
  label-preservation audit (item 1) with a quantitative dose-response
  curve rather than a binary label-preserving/not judgment.
* **Risk of post-hoc overfitting:** Moderate -- choosing which severity
  levels to test after seeing the existing harm magnitudes could bias
  the sweep toward confirming a preferred narrative; must be frozen
  with an a priori severity grid.
* **Requires a new frozen protocol:** Yes -- a new augmentation-severity
  parameter grid, which does not exist in the current frozen policy
  definitions.
* **Rough compute/storage estimate:** No new training if applied only
  at TTA-evaluation time on existing checkpoints (severity varies the
  test-time transform only); each additional severity level costs
  roughly one additional policy's worth of cache storage (~1.1-1.5
  GiB) and evaluation-only compute, similar in scale to item 2.
* **Priority: 2.**

### 8. Calibration analysis (ECE, NLL, Brier)

* **Exact question:** Does naive TTA degrade calibration in the same
  cells where it degrades accuracy, and does matched-policy training
  improve calibration as well as accuracy?
* **Expected publication value:** Moderate-high given the
  newly-surfaced Ayhan & Berens (2018) and Kimura (2024) prior art
  (`paper/reviews/reviewer_2_novelty.md`) frame TTA partly in
  uncertainty/calibration terms this manuscript currently does not
  engage with at all.
* **Risk of post-hoc overfitting:** Low -- calibration metrics can be
  computed from the same already-cached logits/probabilities without
  any new test-split access, provided the metric set is fixed before
  computing rather than chosen from several candidates after seeing
  which one looks best.
* **Requires a new frozen protocol:** A short metric-definition
  addendum (which calibration metrics, computed how) should be frozen
  before computation even though no new data collection is needed.
* **Rough compute/storage estimate:** Negligible new compute (metrics
  computed from already-cached per-view probabilities); no new
  training or evaluation passes required.
* **Priority: 2.**

### 9. Alternative aggregators (majority vote, confidence-weighted
average)

* **Exact question:** Is the observed harm specific to mean-probability
  aggregation, or does it persist under majority-vote and
  confidence-weighted aggregation, both already named in
  `configs/experiment_matrix.yaml`?
* **Expected publication value:** Moderate -- addresses a natural
  reviewer question ("did you only test the aggregator most likely to
  show harm?") and directly engages BayTTA/Shanmugam-et-al.-style prior
  art on aggregation choice.
* **Risk of post-hoc overfitting:** Low if all three already-named
  aggregators are evaluated uniformly.
* **Requires a new frozen protocol:** No -- already named in the
  existing matrix configuration.
* **Rough compute/storage estimate:** No new training; aggregation is
  computed from the same cached per-view logits (per
  `docs/compute_budget.md`'s cache design, aggregates are derived, not
  separately cached), so this is a near-negligible additional compute
  cost.
* **Priority: 2.**

---

## Tier 3 — Nice-to-have

### 10. Per-class analysis

* **Exact question:** Is harm concentrated in specific classes (e.g.
  rare classes, visually similar class pairs) rather than spread
  uniformly?
* **Expected publication value:** Moderate -- a natural, reviewer-
  requestable analysis (Reviewer 3 sec.12), but unlikely to change the
  paper's overall verdict either way.
* **Risk of post-hoc overfitting:** Moderate-high -- per-class
  breakdowns performed after seeing aggregate results are a classic
  path to an unplanned, narrative-driven "we found X class is
  responsible" story; would need a frozen analysis plan to avoid this.
* **Requires a new frozen protocol:** Yes, for the reason above.
* **Rough compute/storage estimate:** Negligible new compute (reuses
  already-collected predictions, per-class breakdown is a
  post-processing step over existing per-sample results).
* **Priority: 3.**

### 11. Additional medical datasets

* **Exact question:** Does the harm pattern generalize beyond
  PathMNIST/BloodMNIST/DermaMNIST to other MedMNIST subsets (or other
  medical-imaging datasets entirely)?
* **Expected publication value:** Moderate-high for a domain-venue
  submission (directly strengthens the generalization claim the title
  implies); lower marginal value for a CVPR-main attempt specifically,
  since it does not address the confound or novelty gaps that are
  CVPR-main's actual blockers.
* **Risk of post-hoc overfitting:** Low if dataset selection is frozen
  before any evaluation (e.g. selecting the next N MedMNIST subsets by
  a predetermined rule, not by which ones might show the most dramatic
  effect).
* **Requires a new frozen protocol:** Yes -- new dataset(s), new
  training matrix, new authorization chain, at full project-lifecycle
  cost (comparable in kind, if not scale, to the original Phase 2B.1-2B.9
  effort).
* **Rough compute/storage estimate:** Comparable to one additional
  Block-A-scale sub-matrix per new dataset (roughly 24 training runs at
  28px/64px per dataset added, i.e. **~3-7 hours per additional
  dataset** by the same per-run arithmetic as item 6), plus new
  download/licensing/checksum verification overhead not captured in
  wall-clock estimates.
* **Priority: 3.**

### 12. Mechanism-oriented representation/feature analysis

* **Exact question:** Do internal representations (e.g. BatchNorm
  activation statistics, feature-space distances between clean and
  augmented inputs) shift more under unmatched-policy training than
  matched-policy training, in a way that predicts the magnitude of
  observed harm?
* **Expected publication value:** High if it worked and if it produced
  a clean, generalizable mechanism story; but this is exploratory,
  open-ended research, not a well-scoped ablation, and could easily
  produce an inconclusive or uninterpretable result given the
  normalization/resolution heterogeneity already observed.
* **Risk of post-hoc overfitting:** High -- representation analysis is
  easy to steer toward a preferred narrative after the fact (choosing
  which layers, which distance metric, which visualization to report)
  unless very tightly pre-specified.
* **Requires a new frozen protocol:** Yes, and a more elaborate one
  than any other item here (exact layers, exact metric, exact
  comparison structure, frozen before inspection).
* **Rough compute/storage estimate:** Requires re-running forward
  passes with intermediate-activation capture on existing checkpoints;
  compute cost similar in order of magnitude to item 4 (BN adaptation),
  but with substantially higher engineering/design effort and a much
  larger risk of an inconclusive result relative to the effort invested.
* **Priority: 3.**

---

## Tier 4 — Low-value work that should not be done

### 13. "External" reproduction of BLOCK_C performed inside this same
project/repository

* **Exact question (as commonly framed):** Does an independent party
  reproduce the BLOCK_C non-reproduction?
* **Why this is low-value as a self-executed addition:** A second
  BLOCK_C run performed by the same team, in the same repository, with
  the same codebase and the same (or adjacent) hyperparameter choices,
  is not an external reproduction in any sense a reviewer would credit
  -- it would at best be a second internal replicate (already partially
  achieved by the existing 3 seeds) and at worst could be mistaken for,
  or presented as, independent confirmation it is not. A genuine
  external reproduction requires a different team, ideally a different
  codebase, which is outside this project's ability to authorize or
  execute.
* **Recommendation:** Do not attempt to manufacture an "external"
  reproduction internally. If external reproduction matters for a
  target venue, the correct path is releasing the audit trail and
  evidence package (already done) and inviting genuine third-party
  replication after publication, not simulating one internally.
* **Priority: not recommended.**

### 14. Superficial comparison against modern test-time-adaptation
baselines (e.g. running TENT once, without a frozen protocol, chosen
after seeing which comparison looks most favorable)

* **Exact question (as a shortcut, not a proper study):** "Does TENT do
  better than naive TTA on our cells?"
* **Why this is low-value if done superficially:** A single, informal
  run of a baseline like TENT -- without a frozen hyperparameter
  search space, without matching TENT's own recommended tuning
  procedure, and run only after seeing the existing results -- would
  produce a number that looks like a rigorous comparison but is not
  one, and would be a much weaker addition than doing nothing, because
  it invites exactly the kind of post-hoc, cherry-picked-comparison
  criticism this project has otherwise been careful to avoid
  throughout its entire audit history. A properly-scoped version of
  this comparison (frozen protocol, matched tuning budget, applied
  uniformly) is a legitimate **Tier 2 candidate**, not listed
  separately above only to avoid duplicating item 4's discussion of
  BN-style test-time adaptation, which TENT extends.
* **Recommendation:** Do not run an unscoped, single-shot baseline
  comparison. If pursued, it must be planned with the same frozen-
  protocol discipline as every other addition in this document.
* **Priority: not recommended in its superficial form.**

---

## Summary: top five prioritized additions (Tier 1 + highest Tier 2)

1. Label-preservation audit (Tier 1).
2. Per-augmentation-component ablation (Tier 1).
3. View-count (N) scaling-curve reporting (Tier 1, likely near-zero
   additional compute).
4. BatchNorm-statistics adaptation comparison (Tier 1).
5. Increase seeds from 3 to 5-10, or add a third architecture family to
   break the ablation-axis confound (Tier 2; these two are
   near-equally valuable and the choice between them should depend on
   which weakness -- statistical power vs. architecture generalization
   -- the authors judge more pressing).

None of these five is authorized or executed by this document. Each
would require its own frozen protocol (where noted above) and explicit
user approval before any new test-split access, per `CLAUDE.md`.
