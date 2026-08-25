# When Test-Time Augmentation Hurts: A Controlled Study in Medical Image Classification

**Author(s):** Anonymous
**Affiliation(s):** Anonymous
**Contact:** Anonymous

*This is a venue-neutral first draft. It does not claim compliance with
any specific conference or journal template, and no claim of
publication venue, acceptance likelihood, or novelty tier beyond what
is stated in sec. "Related Work" is made anywhere in this document.*

---

## Abstract

Test-time augmentation (TTA) is widely treated as a low-risk way to
improve model accuracy at inference time, yet a growing body of
evidence shows it can also hurt. We conduct a preregistered,
statistically audited study of naive mixed-policy TTA on three MedMNIST
datasets (PathMNIST, BloodMNIST, DermaMNIST) across two normalization
schemes (BatchNorm, GroupNorm), two resolutions (28px, 64px), and three
training seeds, plus a fixed 128px extension and a DermaMNIST/ResNet-18
positive-control replication. Within 30 distinct trained-model cells
evaluated under a fixed mixed augmentation policy at N=50 views, naive
TTA reduced test accuracy relative to a single clean forward pass in
**all 30 cells**, with per-cell drops ranging from 18.76 to 66.09
percentage points. A separate, non-preregistered secondary analysis
comparing models trained with a training-time augmentation policy
matched to the test-time TTA policy against models trained without that
match found six of six fixed-model paired comparisons favoring the
matched-policy models, with every 95% confidence interval excluding
zero. Normalization and resolution showed heterogeneous, dataset-
dependent secondary patterns rather than a single consistent
mitigation. A positive-control replication (DermaMNIST, ResNet-18) did
not reproduce the positive TTA effect reported by the source study we
build on. We report these findings using an explicit two-tier
evidentiary framework -- preregistered within-cell evidence versus
secondary fixed-model comparisons -- to avoid overstating what a
non-preregistered cross-condition comparison can support, and we
release a fully audited, hash-verified evidence and reproducibility
trail alongside this manuscript.

---

## Introduction

Test-time augmentation applies a set of input transformations at
inference time and aggregates the resulting predictions, most commonly
by averaging predicted class probabilities. It is often presented as a
nearly-free accuracy improvement, since it requires no retraining and
only additional forward passes. Medeiros (2026) recently challenged this
assumption directly, reporting that mixed-policy TTA with mean
aggregation at 50 views degraded accuracy in 11 of 12 model-dataset
combinations tested on three MedMNIST benchmarks, with drops as large as
31.6 percentage points and a single improving case (ResNet-18 on
DermaMNIST, +1.6 percentage points) [@medeiros2026tta]. This result runs
against the common practitioner intuition that TTA is a safe default,
and -- because it is a single-author preprint with no peer-reviewed
venue found at the time of our review -- warrants independent,
methodologically strengthened verification before it is treated as
established.

This project undertakes that verification. We do not treat the source
preprint's numbers as settled fact; we design an independent,
preregistered confirmatory matrix over the same three MedMNIST
datasets, add a normalization ablation (BatchNorm vs. GroupNorm) and a
resolution ablation (28px vs. 64px, with a 128px extension), add a
training-policy-matching arm, and replicate the source study's single
reported positive result (DermaMNIST, ResNet-18) as an internal
positive control. Every hypothesis, endpoint, and statistical procedure
was frozen before any test-split evaluation occurred, and the frozen
statistical analysis was executed by a sealed, cryptographically
fingerprinted pipeline rather than run ad hoc after seeing results (see
"Reproducibility and Audit Trail").

Our central empirical finding replicates and generalizes the direction
of the source preprint's headline result: naive TTA harmed every one of
30 distinct trained-model cells we evaluated under the fixed mixed-
policy/N=50 condition. Beyond replication, we ask whether this harm is
mitigated by training-time exposure to the same augmentation
distribution used at test time, and whether normalization or resolution
choices modulate the harm. We report the answers to these questions
using an explicit two-tier evidentiary framework, because the
normalization, resolution, and policy-matching questions were only
formally specified as fixed-model comparisons *after* our validation-
stage results were already observed and *before* the test split was
opened -- a design choice we disclose plainly rather than presenting
these comparisons with the same confirmatory status as our preregistered
within-cell tests. We also report, without qualification softening the
result, that our internal positive-control replication (DermaMNIST,
ResNet-18) did not reproduce the source preprint's reported positive
TTA effect.

This manuscript makes no claim of state-of-the-art performance, no
clinical-utility or diagnostic claim (MedMNIST is explicitly not
validated for clinical use), and no claim that our findings generalize
to medical imaging models broadly. Its contribution is a controlled,
statistically disciplined, fully audited account of when and how far
naive TTA harm extends within the specific models, datasets,
resolutions, and normalization choices we actually evaluated -- and an
explicit demonstration of how much confidence a non-preregistered
secondary comparison can and cannot support.

The motivation for this level of methodological care is practical, not
rhetorical. A practitioner deciding whether to enable TTA in a deployed
image-classification pipeline typically has access to exactly the kind
of evidence this study distinguishes carefully: a small number of
validation-set observations, and a temptation to compare conditions
that were never designed as a controlled experiment. If a −60
percentage-point accuracy collapse under naive TTA is possible on
ordinary, non-adversarial MedMNIST inputs -- as our within-cell results
below show it is, in this matrix -- then treating TTA as a low-risk
default is unsafe without validating it on the specific model and
augmentation policy in use. Conversely, if a secondary, non-preregistered
comparison is reported with the same confidence as a preregistered one,
a practitioner may draw a stronger conclusion (e.g., "GroupNorm always
mitigates TTA harm") than the evidence supports, since our own
normalization result reverses sign by dataset. We therefore treat the
preregistered-versus-secondary distinction not as a statistical
formality but as the paper's main methodological contribution alongside
its empirical findings: every numeric claim below is labeled with the
tier of evidence it is drawn from, and Table 1 in the Experimental
Design section makes that labeling explicit before any result is
presented.

## Related Work

**TTA can flip correct predictions to incorrect even when net accuracy
improves.** Shanmugam et al. (2021) show that naive averaging across
augmented views in TTA can change many individual correct predictions
to incorrect ones even in settings where the net accuracy effect is
positive, and propose a learned aggregation-weighting scheme as a fix
[@shanmugam2021better]. This "harm rate" framing -- that aggregate
accuracy can hide sample-level harm -- motivates our own descriptive
seed-level reporting discipline, though we do not adopt their learned
aggregation method; we cite this work as the origin of the harm-framing
concept rather than presenting it as our own.

**Validation-driven and per-sample learned TTA policy selection already
exist.** Lyzhov et al. (2020) introduce Greedy Policy Search (GPS), a
non-learned-network method that greedily selects a test-time
augmentation policy to maximize validation performance
[@lyzhov2020greedy]. Kim et al. (2020) go further, learning an auxiliary
network that predicts per-transform loss for each test input and
filters transforms per sample before aggregation, i.e. instance-level
selective TTA [@kim2020learning]. Neither of these compares a
matched-versus-unmatched training-time augmentation policy as a fixed-
model secondary analysis, which is the specific angle of our own
policy-matching arm; we note this as the narrowest gap we identified,
not as an exhaustively verified absence, since we did not perform a
systematic citation-graph search.

**TTA aggregation has been evaluated specifically in medical imaging.**
Sherkatghanad et al. (2024) propose BayTTA, a Bayesian model averaging
scheme for combining TTA-view predictions, evaluated on skin cancer,
breast cancer, and chest X-ray classification tasks
[@sherkatghanad2024baytta]. This is direct prior art for TTA aggregation
methods evaluated in a medical-imaging context; our study does not
claim to be the first to study TTA in medical imaging, and does not
evaluate Bayesian aggregation. Di Salvo et al. (2024) build MedMNIST-C,
a corruption-robustness benchmark on top of MedMNIST+, and show that
their corruption transforms improve robustness when used as
*training-time* data augmentation [@disalvo2024medmnistc] -- a different
use of "augmentation" than the test-time aggregation harm we study, but
directly relevant background on the MedMNIST+ resolution family we also
use.

**BatchNorm's running statistics are a documented distribution-shift
liability.** Schneider et al. (2020) show that recomputing or adapting
BatchNorm activation statistics at test time using unlabeled
test-distribution statistics, instead of relying on training-time
running statistics, produces large corruption-robustness gains on
ImageNet-C [@schneider2020improving]. This supports the plausibility of
BatchNorm's running statistics as one mechanism by which augmented test
inputs could degrade accuracy, and is background for our normalization
ablation; we do not adapt BatchNorm statistics ourselves and make no
claim about the specific mechanism underlying our observed harm beyond
what is stated as a hypothesis in the Discussion. Group Normalization
(Wu & He, 2018) is the batch-independent alternative we compare against
BatchNorm: it normalizes within per-sample channel groups rather than
relying on batch or running statistics [@wu2018group]. We note explicitly
that GroupNorm differs from BatchNorm in more properties than the mere
presence of running statistics, so any normalization-conditioned
difference we observe is evidence *consistent with*, not *proof of*, a
running-statistics mechanism.

**MedMNIST is the dataset family underlying this study.** MedMNIST v2
(Yang et al., 2023, published in *Scientific Data*) provides
standardized 28px classification benchmarks across biomedical imaging
modalities, explicitly stating the datasets are **not intended for
clinical use** [@yang2023medmnistv2]. The earlier MedMNIST Classification
Decathlon (Yang et al., 2021, ISBI) introduced the original 28px
benchmark suite [@yang2021medmnist]. MedMNIST+ extends select subsets to
64px, 128px, and 224px, constructed from independently higher-resolution
source images rather than upsampled from the 28px files, preserving
sample indices and splits across resolutions -- a fact we rely on
directly for our resolution ablation's validity (see Methods).

**Corruption-robustness benchmarking on MedMNIST is a separate, active
line of work.** MedMNIST-C's corruption taxonomy (blur, noise, digital,
and weather-style corruptions adapted to the 28px-family MedMNIST
images) targets robustness to naturally-occurring image corruption, and
its reported improvements come from *training* models on corrupted
data, not from aggregating predictions over augmented views at test
time [@disalvo2024medmnistc]. We cite it here as evidence that the
MedMNIST+ resolution family we use is already an active substrate for
augmentation-related robustness research, and as a candidate source of
a validated corruption/augmentation taxonomy for future extensions of
this work, not as directly overlapping prior art for the TTA-harm
question itself.

**Novelty scope.** Taken together, the reviewed literature establishes
that TTA harm, augmentation-policy mismatch as a plausible mitigation
lever, and medical-imaging-specific TTA evaluation are all active,
documented areas, and this study does not claim to originate any of
them. The specific combination we did not find directly addressed in
the sources reviewed -- comparing matched-versus-unmatched training
policy as a fixed-model secondary difference-in-differences analysis,
combined with a controlled normalization/resolution ablation, on the
MedMNIST family, under a preregistered/secondary two-tier reporting
discipline -- is offered as a narrow, non-exhaustively-verified gap,
consistent with the novelty audit in
`docs/phase2b_manuscript_claims_and_structure_freeze.md`.

## Methods

**Datasets.** We use three MedMNIST v2 / MedMNIST+ subsets
[@yang2023medmnistv2; @yang2021medmnist]: PathMNIST (train/val/test =
89,996 / 10,004 / 7,180; CC BY 4.0), BloodMNIST (11,959 / 1,712 / 3,421;
CC BY 4.0), and DermaMNIST (7,007 / 1,003 / 2,005; CC BY-NC 4.0, hence
non-commercial). Higher-resolution variants (64px, 128px) are
constructed by the MedMNIST+ project from independently-sourced
higher-resolution originals -- PathMNIST and DermaMNIST by direct
resizing (from 224x224 and 600x450 sources respectively, no cropping),
BloodMNIST by center-cropping a 360x363 source to 200x200 before
resizing -- with sample indices and train/val/test splits preserved
identically across resolutions. This means our resolution ablation (H2)
measures the effect of retained source-image information at a given
standardized resolution, not an artifact of upsampling the 28px images.

**Models.** A small paper-constrained CNN ("SmallCNN," 94,857
parameters, three convolutional layers with global average pooling so
parameter count is resolution-invariant, evaluated with both BatchNorm
and GroupNorm normalization), and a ResNet-18 variant adapted for
28x28-family inputs (approximately 11M parameters, BatchNorm only, no
initial pooling layer), following the source study's architecture
choices where they are specified there.

**Training.** All confirmatory models are trained with Adam (learning
rate 0.001, no weight decay), cross-entropy loss, up to 30 epochs with
cosine-annealed learning rate, early stopping on validation accuracy
(patience 5 epochs, no minimum-improvement threshold), restoring the
best-validation-accuracy checkpoint, batch size 256, float32 precision,
no label smoothing or class weighting, and inputs scaled from uint8 to
[0,1] with no additional channel standardization. Training-time
augmentation is absent everywhere except the policy-matching arm (see
below). These operational choices (exact epoch cap, patience,
weight-decay, and minimum-improvement values) are preregistered choices
of this study, not values extracted from the source preprint, which
reports only "25-30 epochs" and does not specify the others; this
remains a paper-constrained reproduction, not an exact replication.

**TTA procedure.** Test-time augmentation applies a fixed *mixed* policy
(geometric: horizontal/vertical flips, +/-15 degree rotation, random
resized crop 0.8-1.0x; intensity: brightness/contrast jitter +/-0.3,
Gaussian blur; mixed: both families combined) over a deterministic,
per-sample registered 100-view sequence, of which the primary
confirmatory condition uses the first 50 views (N=50), aggregated by
mean predicted probability -- reproducing the source study's headline
condition.

**Policy-matching arm.** A subset of models is additionally trained
with the same mixed augmentation policy applied once per training
sample per training step, so that the training-time and test-time
augmentation distributions match. This is compared against otherwise-
identical models trained without that augmentation (the naive-TTA
condition above) to test whether policy matching mitigates TTA harm.

**Data licensing and preprocessing.** PathMNIST and BloodMNIST are
distributed under CC BY 4.0; DermaMNIST is distributed under CC BY-NC
4.0, and we accordingly treat every DermaMNIST-derived number in this
manuscript as non-commercial-use-only. No dataset-specific channel mean
or standard deviation is applied; inputs are scaled from their native
uint8 range to [0, 1] and otherwise used as distributed. We do not
resize, crop, or otherwise modify the MedMNIST+-provided resolution
variants ourselves -- the 28px, 64px, and 128px inputs are used exactly
as constructed by the MedMNIST+ project, so any resolution effect we
observe is attributable to the MedMNIST+ construction process
described above, not to a resizing choice of our own.

**Compute environment.** All training and evaluation was executed on a
single Apple Silicon machine (Apple M3 Pro, 18GB unified memory) using
the PyTorch MPS backend, float32 precision throughout, with no CUDA
hardware involved. This is disclosed because it constrains the
practical scale of the confirmatory matrix (see Experimental Design and
Limitations) relative to what a larger GPU cluster would allow, and
because MPS-backend numerics can differ in minor ways from CUDA
reference implementations; we did not cross-validate our numbers
against a CUDA run.

## Experimental Design

The confirmatory matrix comprises four blocks. **Block A** (24 runs)
crosses PathMNIST/BloodMNIST, 28px/64px, SmallCNN with BatchNorm/
GroupNorm, and three seeds, isolating the normalization (H1) and
resolution (H2) comparisons under a fixed unmatched TTA policy. **Block
B** (6 runs) trains SmallCNN/BatchNorm at 28px on PathMNIST and
BloodMNIST with the training-time augmentation policy matched to the
test-time TTA policy, across three seeds, for the policy-matching
comparison (H3). **Block C** (3 runs) trains ResNet-18/BatchNorm on
DermaMNIST at 28px across three seeds as an internal positive-control
replication of the source study's sole reported improving case. **Block
D** (6 runs, conditional on a compute-budget gate) extends the
normalization/resolution matrix to a fixed 128px resolution for
PathMNIST and BloodMNIST with BatchNorm. Every training seed (0, 1, 2)
is drawn from three fixed confirmatory seeds; the exploratory pilot seed
used in earlier, non-confirmatory pipeline validation work is
permanently excluded from every confirmatory run.

Every model is evaluated under a clean (single, unaugmented forward
pass) condition and the naive-TTA (N=50, mixed policy, mean
probability) condition described above. This produces, across all
executed blocks, **39 total preregistered matrix cells**. Of these, 30
are distinct trained models evaluated under the unmatched (naive-TTA)
policy: Block A's 24 cells, Block D's 6 cells (Block D is
resolution/normalization-only and therefore contributes only to the
unmatched arm), and Block B's 6 cells contribute their own 6 unmatched-
arm comparisons but those 6 run_ids coincide with 6 of Block A's
run_ids at the shared dataset/resolution/normalization/seed identity, so
they are **not** additional distinct cells -- overlapping hypothesis-
family membership (a given trained model can be part of more than one
hypothesis's evaluated cell set) never inflates the distinct-cell count.
The 30-cell figure used throughout this manuscript is the deduplicated
count.

**Block D's conditional status.** Because 128px training and evaluation
is substantially more compute-intensive than 28px/64px on our single-
machine setup, Block D was preregistered as *approved-conditional*: it
executes only if a native 128px runtime/memory benchmark (measured
before any 128px training) confirms the block fits the project's
compute budget. This conditional-execution rule was frozen before any
128px result existed, so Block D's inclusion in the final 39-cell
matrix is not a post-hoc convenience -- it was always a possible outcome
of the frozen protocol, and its 6 cells are evaluated identically to
Block A's cells once activated.

**Evidentiary tiers.** We distinguish two categories of comparison
throughout. **Preregistered within-cell comparisons** (clean vs. naive-
TTA accuracy on the *same* trained model) were specified before any
test-split evaluation and constitute our confirmatory evidence. **The
normalization (H1), resolution (H2), and policy-matching (H3)
cross-condition comparisons** -- i.e., comparing *different* trained
models' TTA effects against each other via a fixed-model
difference-in-differences (DiD) estimate -- were specified as a fixed
statistical procedure after our validation-stage results were already
observed, but before the test split was opened. This ordering means
these three comparisons cannot carry the same preregistered-confirmatory
status as the within-cell tests; we report them as secondary,
post-validation/pre-test-specified evidence throughout, and we do not
apply any pooled or population-level significance procedure across
either tier.

**Table 1** (reproduced from the committed evidence package,
`artifacts/paper_evidence/tables/table_1_design_classification.md`)
makes this classification explicit before any result is presented:
preregistered within-cell evidence (H1/H2/H3/BLOCK_C, 39 unique cells)
is confirmatory; secondary fixed-model comparisons (cross-condition
H1/H2/H3, 30 pairs across the three families) are non-confirmatory;
descriptive seed-level summaries carry no inferential value of their
own; and a fourth row lists what is unsupported/forbidden at every tier
-- H4, any pooled or model-population verdict, and any secondary
significance language. We refer back to this table's tier labels
throughout the Results and Discussion sections rather than restating
the tier-boundary reasoning at every mention.

## Statistical Analysis

For every preregistered within-cell comparison, we compute a paired
bootstrap 95% confidence interval on the delta accuracy (TTA minus
clean, on the same test samples from the same trained model), using
10,000 resamples with replacement over the test set, together with an
exact or continuity-corrected McNemar test on the paired
correct/incorrect 2x2 contingency table -- the appropriate paired
binary-outcome test for this design. Within each hypothesis family (H1,
H2, H3, and the BLOCK_C positive control), Benjamini-Hochberg
false-discovery-rate correction is applied across that family's cells;
both raw and BH-adjusted p-values are reported. A cell that is a member
of more than one hypothesis family (see Experimental Design) therefore
has one raw McNemar p-value -- the same underlying paired computation --
but a *different* BH-adjusted p-value in each family's own correction
set, since each family's FDR correction is computed independently over
its own cell membership; we report this per-family rather than
collapsing it to a single implied value.

For the three secondary cross-condition comparisons (H1 normalization,
H2 resolution, H3 policy-matching), we compute a paired-bootstrap 95%
confidence interval on the difference-in-differences estimate between
two fixed, already-trained models' TTA effects, again using 10,000
resamples. We report the point estimate and interval and describe,
factually, whether the interval excludes zero. We do not compute, and
do not report, a p-value, an alpha threshold, or a significance
decision for any secondary comparison, and we never use the word
"significant" to describe one, since these comparisons were not
preregistered and are not corrected as part of any confirmatory family.

Descriptive seed-level summaries (mean, sample standard deviation, min,
max of the per-seed delta-accuracy point estimate within a dataset x
resolution x normalization group) are reported separately, carry no
p-value or confidence interval of their own, and are labeled
non-inferential throughout; they contextualize seed-to-seed variability
and are never used as evidence for or against any hypothesis.

Alongside delta accuracy and its confidence interval, effect sizes are
reported for each preregistered cell: the raw delta-accuracy magnitude
itself, and harm/rescue rates (the fraction of samples correctly
classified clean but incorrectly classified under TTA, and vice versa)
as effect-size-like quantities in the spirit of Shanmugam et al. (2021)'s
observation that net accuracy can mask sample-level flips
[@shanmugam2021better]. These are descriptive characterizations of a
single preregistered comparison, not additional hypothesis tests, and
do not receive their own confidence interval or p-value.

Pairing is preserved throughout: because clean and TTA predictions are
always evaluated on the same test samples from the same trained model,
every within-cell comparison is a paired design, and every
cross-condition secondary comparison compares two *already independently
paired* delta-accuracy distributions from two different fixed models --
never an independent-groups test applied to what is actually paired
data. The choice of Benjamini-Hochberg FDR correction per hypothesis
family, rather than a single Bonferroni correction pooled across all
four families, follows the frozen statistical analysis plan's reasoning
that the four hypotheses ask distinct scientific questions and a single
pooled correction would be needlessly conservative without a
corresponding scientific justification; this choice was fixed before any
test-split result existed, not selected after seeing which correction
produced a more favorable outcome.

No new statistic beyond the above was computed for this manuscript.
Every number reported in Results below is either copied directly from
the canonical, hash-verified evidence package described in
"Reproducibility and Audit Trail," or is a minimum/maximum taken
directly across an already-computed set of per-cell point estimates
for narrative purposes (e.g., the range of the 30 within-cell deltas);
no new inferential test, pooled estimate, or aggregate hypothesis test
was introduced beyond what the frozen statistical analysis plan already
specifies.

## Results

**All 30 distinct unmatched-policy cells showed harm.** Under the fixed
mixed-policy, N=50, mean-aggregation naive-TTA condition, every one of
the 30 distinct trained-model cells evaluated showed a negative delta
accuracy (TTA minus clean), ranging from -18.76 percentage points
(PathMNIST, 28px, BatchNorm, seed 0) to -66.09 percentage points
(BloodMNIST, 64px, BatchNorm, seed 1) (Figure 1; full per-cell values in
Supplementary Table 2). Split by dataset, the 15 BloodMNIST cells ranged
from -20.70 to -66.09 percentage points and the 15 PathMNIST cells
ranged from -18.76 to -51.92 percentage points -- both datasets show
substantial harm across their full cell sets, with no cell in either
dataset showing a non-negative delta. This is a preregistered,
within-cell result: it establishes that naive TTA harmed these specific
trained models under this specific policy and view budget, not a claim
about medical imaging models in general. Sample sizes underlying each
cell's paired bootstrap and McNemar test equal the corresponding
dataset's official test-split size (PathMNIST 7,180; BloodMNIST 3,421),
so the harm reported here is measured against the full official test
set of each dataset, not a subsample.

**Matched-policy mitigation is supported by secondary evidence and
descriptively corroborated within cells.** The six matched-policy
within-cell deltas (Block B, PathMNIST and BloodMNIST, 28px, BatchNorm)
were small and mixed in sign (-1.21 to +4.09 percentage points), in
sharp contrast to the strongly negative unmatched deltas for the
corresponding dataset/resolution/normalization cells -- a descriptive
pattern, not a second confirmatory test of the same claim. Separately,
the six secondary fixed-model DiD comparisons (matched-policy model
minus unmatched-policy model's TTA effect, same dataset/seed) were all
positive, ranging from +17.55 to +49.58 percentage points, and **all six
95% confidence intervals excluded zero** (Figure 2; Supplementary Table
3). We describe this as secondary evidence supporting mitigation,
descriptively corroborated by the separate within-cell pattern -- not as
a preregistered cross-condition test.

**Normalization shows a dataset-dependent secondary pattern.** Across
the 12 secondary H1 (GroupNorm minus BatchNorm) DiD comparisons, 12 of
12 95% confidence intervals excluded zero, but the direction reversed by
dataset: 5 of 6 BloodMNIST pairs were positive (GroupNorm showing less
harm than BatchNorm), with 1 of 6 negative; all 6 PathMNIST pairs were
negative (BatchNorm showing less harm than GroupNorm) (Figure 3;
Supplementary Table 4). Both normalization types experienced harm at
every within-cell comparison (Results, first paragraph); their relative
difference is dataset-dependent secondary evidence only, and must not be
read as a general BatchNorm-vs-GroupNorm verdict.

**Resolution did not consistently reduce harm.** Across the 12 secondary
H2 (64px minus 28px) DiD comparisons, 9 of 12 95% confidence intervals
excluded zero: the BloodMNIST group trended predominantly negative (4 of
6 pairs negative and excluding zero, indicating *more* harm at 64px than
28px, contrary to the hypothesized mitigating direction), while the
PathMNIST group was mixed and closer to null (1 of 6 pairs positive and
excluding zero, 1 negative and excluding zero, 4 including zero) (Figure
4; Supplementary Table 5). Neither pattern is a preregistered or
confirmatory test of a resolution effect, and higher resolution did not
consistently reduce TTA harm in either dataset.

**BLOCK_C did not reproduce the expected positive TTA effect.** The
source study's own reported positive TTA result was a +1.6 percentage
point improvement for ResNet-18 on DermaMNIST at N=50
[@medeiros2026tta]. Across our three positive-control seeds, delta
accuracy was +0.25, -2.89, and +0.05 percentage points; only the seed-1
interval (-4.04, -1.80) excluded zero, and it excluded zero on the
*negative* side (Figure 5; Supplementary Table 6). This preregistered
within-cell positive-control result did not reproduce the source
study's expected positive effect in any of the three seeds, and we
report this plainly as a null/contrary finding rather than omitting or
softening it.

**Summary of claim status.** Table 7 (claim adjudication) summarizes
which of the claims above are preregistered/confirmatory versus
secondary/descriptive, and explicitly lists claims this study does not
make. No H4 claim is made anywhere in this manuscript: Validation-Gated
TTA (H4) was never implemented and has no derivable evidence family in
the canonical evidence package. No population-level or model-population
inference is made anywhere in this manuscript, no secondary significance
claim is made, and no claim is made that BLOCK_C reproduced the external
reference.

## Discussion

Our central, preregistered finding -- that naive mixed-policy TTA
harmed accuracy in all 30 distinct trained-model cells we evaluated --
replicates the direction of Medeiros (2026)'s headline result on an
independent, statistically audited implementation, and extends it with
a normalization and resolution ablation the source study did not
report. That every cell we tested showed harm, across two architectures,
two normalization schemes, two resolutions, and three MedMNIST datasets,
is a stronger and more systematic demonstration of TTA harm's breadth
within this specific matrix than a single-condition result would be --
but it remains a within-model claim, not a claim about TTA or medical
imaging models generally, and it says nothing about model-dataset
combinations, architectures, augmentation policies, or view budgets we
did not test.

The secondary policy-matching result is, in our view, the most
practically suggestive finding, precisely because it is also the one we
are most careful to qualify. All six fixed-model comparisons favored
models whose training-time augmentation distribution matched the
test-time TTA policy, and this pattern is descriptively consistent with
the separate within-cell observation that matched-policy models showed
near-zero or mixed TTA effects while unmatched-policy models showed
large, uniformly negative effects. One plausible explanation -- offered
here explicitly as a **hypothesis for future work, not an established
mechanism** -- is that naive TTA harm is substantially a distribution-
shift effect: augmented test views resemble a distribution the
unmatched model never saw during training, while a matched-policy model
has already adapted to that distribution. This is consistent with, but
not proven by, our data, and is also consistent with the BatchNorm-
running-statistics mechanism Schneider et al. (2020) study in a
different context [@schneider2020improving]; we did not run the
adaptation or ablation experiments (e.g., recomputing BatchNorm
statistics at test time, or a `no_running_stats` decomposition arm) that
would be needed to isolate which mechanism, or combination of
mechanisms, is responsible.

The normalization and resolution results are, by contrast, genuinely
heterogeneous rather than a story we can simplify. If BatchNorm's
running-statistics dependence were the whole story, we would expect
GroupNorm to consistently show less harm than BatchNorm; instead the
direction reversed by dataset. If higher resolution reliably preserved
more discriminative signal against augmentation-induced distortion, we
would expect 64px to consistently show less harm than 28px; instead
BloodMNIST trended in the opposite direction and PathMNIST was largely
null. We report both patterns as dataset-dependent secondary findings
precisely because they resist a single mechanistic narrative, and we
consider this heterogeneity itself an informative (if not confirmatory)
result: it is evidence against any simple, dataset-independent
normalization or resolution rule for predicting or mitigating TTA harm
within this matrix.

Our positive-control replication failure (BLOCK_C) is a genuine null
result relative to the source study's own reported case, not a
methodological artifact we can explain away; we report it as such. It
does not, on its own, invalidate the source study's finding (a single
non-reproduction across three seeds, on a positive effect the source
study itself reported as small (+1.6 percentage points) relative to its
other results, is limited evidence), but it does mean our own dataset
does not provide independent confirmation of that specific positive
case, and we do not claim otherwise.

Finally, we note that the magnitude of harm we observe in our
preregistered cells (up to -66.09 percentage points) exceeds the
largest drop reported by the source study (-31.6 percentage points for
ResNet-18 on PathMNIST) [@medeiros2026tta]. We do not interpret this as
evidence that our implementation is more "correct" or that the source
study understated the effect; the two studies differ in exact
architecture instantiation, training hyperparameters (several of which
the source study does not fully specify, as noted in Methods), TTA view
sequencing, and dataset/resolution combinations tested, any of which
could account for the difference. We report the comparison only to
situate our numbers relative to the literature, not to adjudicate whose
measurement is more accurate -- a claim neither this manuscript nor a
single independent replication attempt is positioned to make.

## Limitations

**Design and coverage limitations.** Only three training seeds per
cell; sample-level paired tests within a cell do not substitute for a
model-seed population replication study, and our findings should not be
read as bounding the variance across a hypothetical larger seed
population. Coverage is limited to the specific MedMNIST subsets,
architectures, and normalization/resolution combinations actually
tested; a fixed augmentation policy (mixed: geometric + intensity) and a
fixed TTA view budget (N=50) were used throughout, and we make no claim
about other policies or view counts. The three secondary cross-condition
comparisons (normalization, resolution, policy-matching) were specified
as a fixed statistical procedure after validation-stage results were
already observed, but before the official test split was opened; we
disclose this ordering explicitly rather than presenting these
comparisons with confirmatory status.

**Process incidents disclosed for completeness.** During execution, one
final-test cell experienced an accidental final-test access incident on
its first attempt (aborted, no scientific value persisted) and two
further failed engineering attempts before a successful, audited
completion; a shared-aggregation-contract correction was made mid-project
after a defect was found, requiring a validation-metric-reconciliation
step; all 39
canonical final-test results reported in this manuscript were produced
under the final, corrected evaluator/aggregation pipeline, with the one
cell computed under an earlier pipeline generation independently
re-verified compatible via full recomputation rather than assumed
compatible. No final-test scientific result was inspected by a human
before the controlled, authorization-gated unsealing step that produced
the canonical evidence package this manuscript reports from. A
pre-existing test-suite gap (31 tests whose guard fixture incorrectly
assumed the canonical outputs did not yet exist) was identified and
fixed in the same project phase that produced this manuscript, as a
test-only change with no effect on any scientific result. These
incidents are recorded here, and in full detail in the project's audit
trail, precisely so that engineering history does not have to be
inferred or taken on faith; none of them altered which test-split data
contributed to any reported number, and none required re-deriving any
statistic.

**Scope of the novelty claim.** Our literature review, detailed in
`docs/literature_review.md` and re-verified for this manuscript in
`paper/citation_audit.md`, was not an exhaustive systematic search (e.g.
no citation-graph traversal was performed); the novelty gap we identify
(sec. "Related Work") is stated with that caveat and should not be read
as a confirmed first.

## Reproducibility and Audit Trail

This study's confirmatory statistical analysis, cross-condition
secondary analysis, and scientific-result unsealing were each executed
by a sealed, deterministic pipeline whose source code, environment, and
inputs are bound together by cryptographic content fingerprints,
verified unchanged before and after every execution step. The
statistical analysis, cross-condition addendum, and unsealing steps
each required an explicit, human-reviewed authorization artifact,
verified before any result was computed or read; no scientific result
in this manuscript was read from a partially-authorized or unverified
pipeline state.

All numbers in the Results section above trace mechanically to the
canonical scientific summary
(`artifacts/final_test_scientific_summary.json`) and the committed
paper-evidence figures and tables (`artifacts/paper_evidence/`), both
of which are content-hash-bound in a machine-readable manifest
(`artifacts/paper_evidence/paper_evidence_manifest.json`). A read-only
verification script (`paper/verify_manuscript_claims.py`, described in
`paper/README.md`) cross-checks every numeric claim in this manuscript
against that evidence package and rejects known-forbidden phrasings
(e.g. "54 distinct cells," any secondary-significance language, any H4
claim); this manuscript was not considered complete until that script
passed. No script or test described here accesses raw predictions, the
underlying image datasets, or model checkpoints -- only the already-
sealed, already-verified summary and evidence-package artifacts.

## Conclusion

Within a preregistered, statistically audited matrix of 30 distinct
trained-model cells across three MedMNIST datasets, two normalization
schemes, and two resolutions, naive mixed-policy test-time augmentation
reduced test accuracy in every cell evaluated. A secondary, explicitly
non-preregistered analysis suggests that matching the training-time
augmentation policy to the test-time TTA policy mitigates this harm,
while normalization and resolution show heterogeneous, dataset-dependent
secondary patterns rather than a single consistent mitigation, and an
internal positive-control replication did not reproduce the one
previously reported case of TTA helping in this setting. We report all
of this using an explicit evidentiary hierarchy that separates
confirmatory within-cell results from secondary cross-condition
comparisons, and we release the complete, hash-verified evidence and
audit trail alongside this manuscript so that every reported number can
be independently mechanically re-checked.

---

## References

Kim, I., Kim, Y., & Kim, S. (2020). Learning Loss for Test-Time
Augmentation. *Advances in Neural Information Processing Systems
(NeurIPS)*, 33. arXiv:2010.11422.

Lyzhov, A., Molchanova, Y., Ashukha, A., Molchanov, D., & Vetrov, D.
(2020). Greedy Policy Search: A Simple Baseline for Learnable Test-Time
Augmentation. *Proceedings of the 36th Conference on Uncertainty in
Artificial Intelligence (UAI)*, PMLR 124:1308-1317. arXiv:2002.09103.

Medeiros, D. N. (2026). I Can't Believe TTA Is Not Better: When
Test-Time Augmentation Hurts Medical Image Classification. *arXiv
preprint arXiv:2604.09697*.

Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W., & Bethge,
M. (2020). Improving Robustness against Common Corruptions by Covariate
Shift Adaptation. *Advances in Neural Information Processing Systems
(NeurIPS)*, 33. arXiv:2006.16971.

Shanmugam, D., Blalock, D., Balakrishnan, G., & Guttag, J. (2021).
Better Aggregation in Test-Time Augmentation. *Proceedings of the
IEEE/CVF International Conference on Computer Vision (ICCV)*.
arXiv:2011.11156.

Sherkatghanad, Z., Abdar, M., Bakhtyari, M., Plawiak, P., & Makarenkov,
V. (2024). BayTTA: Uncertainty-aware Medical Image Classification with
Optimized Test-Time Augmentation using Bayesian Model Averaging. *arXiv
preprint arXiv:2406.17640*.

Di Salvo, F., Doerrich, S., & Ledig, C. (2024). MedMNIST-C:
Comprehensive Benchmark and Improved Classifier Robustness by Simulating
Realistic Image Corruptions. *arXiv preprint arXiv:2406.17536*.

Wu, Y., & He, K. (2018). Group Normalization. *Proceedings of the
European Conference on Computer Vision (ECCV)*. Also available as
arXiv:1803.08494.

Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., & Ni,
B. (2023). MedMNIST v2 -- A Large-Scale Lightweight Benchmark for 2D and
3D Biomedical Image Classification. *Scientific Data*, 10(1).
https://doi.org/10.1038/s41597-022-01721-8. Also available as
arXiv:2110.14795.

Yang, J., Shi, R., & Ni, B. (2021). MedMNIST Classification Decathlon: A
Lightweight AutoML Benchmark for Medical Image Analysis. *IEEE 18th
International Symposium on Biomedical Imaging (ISBI)*. arXiv:2010.14925.

---

## Supplementary Material Outline

*(Outline only in this phase, per
`docs/phase2b_manuscript_claims_and_structure_freeze.md` sec.6; full
supplementary content is future work.)*

- **Supplementary Table 1** (main-text Table 1 in full): experimental-
  design and evidence-classification table.
- **Supplementary Table 2**: complete 30-cell unmatched-policy table
  (dataset, resolution, normalization, seed, delta accuracy, 95% CI,
  raw McNemar p, BH-adjusted p per member hypothesis family) --
  `artifacts/paper_evidence/tables/table_2_unmatched_policy.md`.
- **Supplementary Table 3**: complete matched-policy table (6
  within-cell rows + 6 secondary DiD pairs) --
  `artifacts/paper_evidence/tables/table_3_matched_policy.md`.
- **Supplementary Table 4**: complete 12-pair normalization DiD table --
  `artifacts/paper_evidence/tables/table_4_normalization.md`.
- **Supplementary Table 5**: complete 12-pair resolution DiD table --
  `artifacts/paper_evidence/tables/table_5_resolution.md`.
- **Supplementary Table 6**: complete 3-seed BLOCK_C table with the
  external +1.6pp descriptive reference footnote --
  `artifacts/paper_evidence/tables/table_6_block_c.md`.
- **Supplementary Table 7** (main-text Table 7 in full): claim
  adjudication table -- `artifacts/paper_evidence/tables/table_7_claim_adjudication.md`.
- **Supplementary Note A**: full frozen training/evaluation
  hyperparameter table (`docs/phase2b_protocol.md`).
- **Supplementary Note B**: full statistical analysis plan
  (`docs/statistical_analysis_plan.md`).
- **Supplementary Note C**: complete engineering and audit trail
  (fingerprint values, authorization chain, generation history) --
  `docs/phase2b_paper_evidence_package.md` and the referenced Phase
  2B.6-2B.9 documents.
