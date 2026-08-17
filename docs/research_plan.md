# Research Plan

**Status: Phase 0 — planning / pre-registration. No experiments have been run.**

## Phase definitions

- **Phase 0 (current):** scientific plan, literature review, repo skeleton,
  claims table, statistical plan. No training, no dataset downloads, no
  results, no commits.
- **Phase 1 (future, requires approval):** environment setup, dataset
  download + checksum, smoke tests, one-seed pilots.
- **Phase 2 (future, requires approval):** confirmatory 3-seed runs per the
  frozen `configs/experiment_matrix.yaml`, test-set evaluation under the
  test firewall.
- **Phase 3 (future, requires approval):** analysis, figures, draft report.

Moving between phases requires explicit user approval — see `CLAUDE.md`.

## Motivating question

The target preprint, *"I Can't Believe TTA Is Not Better: When Test-Time
Augmentation Hurts Medical Image Classification"* (arXiv:2604.09697,
preprint, April 2026, single author, no peer-review venue found — see
`docs/literature_review.md`), reports that **mixed-policy TTA with mean
aggregation at 50 views degraded accuracy in 11 of the 12 model-dataset
combinations tested**, with the sole improvement being ResNet-18 on
DermaMNIST (+1.6pp). The two largest drops were ResNet-18 on PathMNIST
(−31.6pp) and SmallCNN on PathMNIST (−28.6pp); SmallCNN on BloodMNIST
dropped 14.8pp. These figures are read directly from the paper's Table 2
(full text verified — see `docs/literature_review.md`), not estimated. This
is a striking, counter-intuitive result given the general folk wisdom that
TTA helps.

**Important:** the source paper's own Appendix B already evaluates (1)
including the original clean image as an anchor alongside augmented views,
and (2) basic BatchNorm-statistics adaptation — both with reported effect
sizes (e.g. clean anchoring reduced SmallCNN/PathMNIST's drop from −37.0%
to −8.6%; BN adaptation helped on BloodMNIST but *hurt* on PathMNIST). This
project must not present clean anchoring or basic BN adaptation as a new
contribution — they are required baselines to reproduce and compare
against, not novel mechanisms. See `docs/claims_and_risks.md`.

**Primary research question:** Are the reported TTA failures actually
attributable to (a) BatchNorm's dependence on running statistics that
augmented views violate, (b) low input resolution amplifying augmentation
distortion, and (c) a mismatch between training-time and test-time
augmentation distributions — and, separately, can a simple
validation-gated fallback mechanism reduce the harm without requiring a
learned aggregation model?

This project does not assume the source paper's numbers are correct. Part
of Phase 1 is an exact reproduction check before extending the work.

## Pre-registered hypotheses

These are proposed for pre-registration. Each is stated as written by the
user, followed by a critical note on wording.

### H1 — Normalization

**Corrected wording (was a stronger causal claim; revised per correction
pass):**
> Under controlled architecture and training conditions, TTA degradation
> differs between BatchNorm and GroupNorm.

*Critical note:* This is deliberately weaker than the original "BatchNorm
produces a larger TTA accuracy degradation... [because of running
statistics]" wording. BatchNorm-vs-GroupNorm changes multiple properties at
once (running statistics, effective batch-size sensitivity, train/eval-mode
behavior) — a difference in TTA degradation between the two **cannot alone
prove that running statistics are the causal mechanism**. The revised
wording only commits to "differs," not to a specific causal story. A
`no_running_stats` decomposition arm (BatchNorm with running stats disabled
or recomputed per batch) was originally proposed to help isolate the
mechanism further, but has been **removed from the confirmatory matrix** to
fit the corrected compute budget (see `docs/compute_budget.md`); it may
return only as an explicitly-labeled exploratory arm. Any report on H1 must
state plainly that a BatchNorm-vs-GroupNorm difference is evidence
consistent with a running-statistics explanation, not proof of it.

### H2 — Resolution
> TTA degradation decreases as input resolution increases from 28×28 to
> 64×64 to 128×128.

*Critical note (corrected):* The earlier concern that MedMNIST+'s higher
resolutions might be upsampled/interpolated from the 28×28 files has been
**checked and is false** for our three datasets: per
`github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md` (verified
directly), PathMNIST's 64/128px images are resized from independently
sourced 224×224 originals, DermaMNIST's from 600×450 originals, and
BloodMNIST's from 360×363 originals center-cropped to 200×200 before
resizing — sample indices and splits are preserved across resolutions. **H2
therefore evaluates retained source-image information at different
standardized resolutions, not interpolation artifact level.** This is a
materially stronger experimental design than previously assumed. See
`docs/data_and_licensing.md` for full sourcing.

**Confirmatory scope (corrected):** the primary confirmatory comparison is
**28px vs. 64px** (block A in `configs/experiment_matrix.yaml`), both with
full 3-seed coverage. The **128px tier is conditional** on a measured
compute-budget gate (`docs/compute_budget.md`) and, if run, strengthens a
trend claim only — it does not stand alone as confirmatory evidence unless
completed with all 3 seeds.

### H3 — Policy matching
> Training with an augmentation distribution matched to the test-time
> augmentation policy reduces TTA degradation compared with applying
> previously unseen augmentations only at inference.

*Critical note:* This is close to a truism (train/test distribution
matching helping is expected under basically any account of generalization)
— its scientific value here is in *quantifying* the effect size relative to
H1/H2, not in testing whether the effect exists at all. Framing in the
report should emphasize the magnitude/interaction with normalization and
resolution, not present "matching helps" as a novel finding.

**Confirmatory scope (corrected):** evaluated at **28px only**, on
PathMNIST and BloodMNIST, with BatchNorm normalization (block B in
`configs/experiment_matrix.yaml`) — 3 seeds each. The unmatched comparison
arm reuses block A's checkpoints rather than requiring separate training
runs.

### H4 — Validation-gated TTA
> A validation-selected TTA policy with per-sample rejection and
> clean-prediction fallback reduces the clean-correct-to-TTA-wrong harm
> rate compared with naive mean aggregation.

*Critical note:* As required by the user's constraints, this is phrased as
an empirically evaluated safety mechanism, not a guarantee — good. Note
this idea (selective/instance-level TTA view rejection, validation-tuned
aggregation) has substantial prior art (Shanmugam et al. 2021, Lyzhov et
al. 2020, learned-loss TTA, BayTTA — see `docs/literature_review.md`), and
the source paper's *own* appendix already evaluates two adjacent
mitigations (clean-image anchoring, basic BN adaptation). The contribution
here, if any, is the *specific combination* of (1) doing this in the causal
context of BatchNorm/resolution/policy-mismatch failures identified in
H1–H3, and (2) a deliberately simple, non-learned gating rule evaluated for
harm-rate reduction specifically, rather than raw accuracy. This must be
framed as an engineering/empirical contribution, not a novel algorithm —
see `docs/claims_and_risks.md`.

**Required comparison set (corrected):** Validation-Gated TTA must be
evaluated against all of the following, not just naive mean TTA:
1. Clean single-pass inference
2. Naive augmentation-only TTA (mean aggregation)
3. Original-image-anchored TTA (source paper's Appendix B condition)
4. Basic BN-statistics adaptation, where applicable (source paper's
   Appendix B condition; BatchNorm cells only)
5. Validation-selected/gated TTA (this project's condition)

Do not claim that transform selection, clean anchoring, adaptive
aggregation, or BN adaptation is new — all four are prior art (either from
the cited literature or from the source paper's own appendix). See
`docs/claims_and_risks.md`.

## Relationship to prior work (summary — full detail in literature_review.md)

Selective/learned TTA aggregation is an established subfield. This project
does not claim to invent adaptive or selective TTA. Its potential
contribution is a controlled causal isolation of *why* TTA fails in this
specific MedMNIST setting, plus an empirical (not theoretical) evaluation of
whether a simple validation-gated fallback recovers most of the harm
reduction without the complexity of a learned aggregator. Whether this is a
big enough contribution for a workshop paper is an open question to be
revisited after the literature review is complete — see
`docs/claims_and_risks.md` for the honest current answer.

## Datasets, models — see dedicated docs

- Dataset details, licensing, splits: `docs/data_and_licensing.md`
- Model/architecture rationale: `docs/experimental_protocol.md`
- Compute budget and run counts: `docs/compute_budget.md`
- Statistics: `docs/statistical_analysis_plan.md`
- Literature: `docs/literature_review.md`
- Claims discipline: `docs/claims_and_risks.md`
