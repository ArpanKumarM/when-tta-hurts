# Reviewer 2 — Novelty and Related Work

**Manuscript:** "When Test-Time Augmentation Hurts: A Controlled Study
in Medical Image Classification" (`paper/manuscript.md`, HEAD `4c70ddb`)
**Reviewer role:** hostile, reviewer-style novelty assessment for a
top-tier computer-vision main track. This review expands the
manuscript's own literature review (`docs/literature_review.md`,
`paper/citation_audit.md`) with three additional primary sources
verified in this phase, and reassesses novelty accordingly.

## Expanded literature review methodology

All new sources below were fetched and read directly (arXiv abstract
pages, an author-maintained GitHub repository, and DBLP's curated
bibliography record) -- never accepted from a search-result snippet
alone. This expands, but does not replace, the manuscript's existing
nine-source review; it specifically targets the topics Phase 2B.10A
asks this review to cover: test-time adaptation and BatchNorm
adaptation beyond Schneider et al. (2020), medical-image TTA beyond
BayTTA, and uncertainty/calibration effects of TTA, none of which the
manuscript's current Related Work section covers with a dedicated
citation.

**Important process note:** per this phase's explicit instruction not
to modify `paper/references.bib` or `paper/citation_audit.md`, the
three new sources below are cited by their verified bibliographic
identity *within this review document only*. They are not yet added to
the manuscript's reference list; doing so is future work for a
manuscript-revision phase, tracked in
`docs/phase2b_submission_gap_analysis.md`.

### New source 1 — Test-time adaptation / BatchNorm adaptation

**Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021).
Tent: Fully Test-Time Adaptation by Entropy Minimization. *ICLR 2021
(Spotlight)*. arXiv:2006.10726.**

Verified directly: `https://arxiv.org/abs/2006.10726` (title, all five
authors, version history v1-v3, ICLR 2021 Spotlight venue confirmation
explicit on the page, arXiv-issued DOI). TENT adapts BatchNorm's
channel-wise affine parameters online, at test time, by minimizing
prediction entropy over each test batch -- a substantially more
sophisticated test-time-adaptation mechanism than the "recompute
BatchNorm statistics from the augmented batch" adaptation the target
preprint's Appendix B evaluates (and which this manuscript's Methods
references only in passing, via Schneider et al.). TENT is the
best-known representative of a broader "fully test-time adaptation"
literature this manuscript does not engage with at all.

### New source 2 — Medical-image TTA and uncertainty/calibration

**Ayhan, M. S., & Berens, P. (2018). Test-time Data Augmentation for
Estimation of Heteroscedastic Aleatoric Uncertainty in Deep Neural
Networks. *1st Conference on Medical Imaging with Deep Learning
(MIDL 2018)*.**

Verified via the authors' own repository,
`https://github.com/berenslab/ttaug-midl2018` (title, both authors,
venue, and the authors' own self-provided BibTeX entry), after the
OpenReview proceedings page itself
(`https://openreview.net/forum?id=rJZz-knjz`) returned only a bot-check
page and could not be opened directly -- disclosed here exactly as for
the Wu & He case in the main citation audit. This is directly on-topic
prior art the manuscript does not cite: TTA used specifically in a
medical-imaging context, specifically to estimate *uncertainty* (via
prediction variability across augmented views) rather than to improve
point-accuracy -- a use of TTA the manuscript's Statistical Analysis
section comes close to (via harm/rescue rates) but never frames in
calibration/uncertainty terms.

### New source 3 — Theoretical account of when TTA helps or hurts

**Kimura, M. (2024). Understanding Test-Time Augmentation. *arXiv
preprint arXiv:2402.06892*.**

Verified directly: `https://arxiv.org/abs/2402.06892` and
`https://arxiv.org/html/2402.06892` (title, sole author, submission
date, CC BY 4.0 license, abstract confirming the theoretical claims
below). A `WebSearch` result set suggested a possible identical-titled
ICONIP 2021 Springer chapter by the same author area, but the arXiv
abstract page's own metadata contains no journal-reference or prior-
venue field confirming this is the same paper under an earlier venue;
this review therefore cites **only the fully-verified arXiv preprint**
and does not assert an ICONIP 2021 venue. Kimura shows, under an
i.i.d.-style model of augmented-view errors, that TTA's expected error
is bounded by the average single-view error, with the gap governed by
an "ambiguity term" that rewards *diverse, low-correlation* augmented
predictions and penalizes correlated or redundant ones. This is
directly relevant theoretical background this manuscript's Discussion
does not cite when speculating about a distribution-shift mechanism:
Kimura's framework predicts TTA harm specifically when augmented views
are *not* low-correlation-diverse-but-individually-accurate, which is
a testable, alternative (or complementary) theoretical lens to the
manuscript's own BatchNorm/distribution-shift hypothesis.

## Question-by-question assessment

### 1. Is "TTA can hurt" already well established?

**Yes, unambiguously.** The manuscript's own primary source, Medeiros
(2026), already reports the phenomenon in the same setting (MedMNIST,
mixed policy, mean aggregation) with a headline number (11/12
combinations harmed) that this manuscript's own Abstract explicitly
sets out to verify. Shanmugam et al. (2021) already show, more
generally, that naive TTA averaging degrades individual predictions
even when net accuracy improves. "TTA can hurt" as a bare empirical
fact is not this paper's novel contribution, and the manuscript does
not claim it is -- but a reviewer scoring novelty must weigh that the
paper's most quotable, headline-friendly finding ("all 30 cells
harmed") is, at the level of the underlying phenomenon, a replication,
not a discovery.

### 2. Is the medical-imaging setting novel enough on its own?

**No.** BayTTA (Sherkatghanad et al., 2024) already evaluates TTA
aggregation specifically in medical imaging (skin cancer, breast
cancer, chest X-ray); Ayhan & Berens (2018, new source above) already
study TTA specifically in a medical-imaging, uncertainty-estimation
context; and the target preprint this manuscript reproduces is already
set entirely in MedMNIST. "TTA in medical imaging" as a topic area is
active and has direct prior art; being *set* in medical imaging does
not by itself clear a novelty bar.

### 3. Is matched-policy mitigation already known or expected?

**Expected in spirit, not previously demonstrated in this specific
form (within the papers reviewed).** The general intuition that
train/test distribution mismatch causes problems, and that closing the
gap helps, is a decades-old idea in the broader domain-adaptation and
data-augmentation literature, and the target preprint's own Appendix B
already tests two different mitigations (clean-image anchoring, basic
BatchNorm adaptation) for the same harm this manuscript studies. Greedy
Policy Search (Lyzhov et al., 2020) and Kim et al. (2020) both address
*test-time* augmentation-policy selection, not *training-time* policy
matching. A hostile reviewer's likely framing: "matching train and test
augmentation distributions helps" is an unsurprising, close-to-expected
result dressed in careful secondary-DiD statistical clothing; the
paper's contribution here is methodological rigor in demonstrating an
expected effect, not the discovery of an unexpected one.

### 4. Is the dataset-dependent normalization reversal novel, or
unexplained noise?

**Closer to unexplained noise than a finding, as currently presented.**
Six paired comparisons per dataset, with no mechanism-isolating
follow-up (no BatchNorm-statistics-adaptation ablation despite citing
Schneider et al. and now, per this review, TENT as directly relevant
prior art with ready-made adaptation mechanisms that could have been
applied; no `no_running_stats` decomposition arm, despite this having
been part of the project's own original hypothesis design before being
cut for compute-budget reasons per `docs/research_plan.md`), is not
enough evidence to distinguish "there is a real, dataset-dependent
interaction between normalization and augmentation type" from "with 6
samples per group, sign reversals are an unsurprising consequence of
sampling variability across just 3 seeds x 2 resolutions." The
manuscript's own honest framing ("heterogeneous," "must not be read as
a general verdict") is the scientifically correct response to this
uncertainty, but it also means this result cannot be counted as a
novelty contribution -- an un-mechanized, unreplicated reversal is not,
by itself, new scientific knowledge.

### 5. Does the paper offer a method, mechanism, benchmark, or only an
empirical observation?

**Only an empirical observation, plus a reporting-discipline
framework.** No new TTA method or aggregation scheme is proposed (cf.
Shanmugam et al., Kim et al., BayTTA, all of which propose methods).
No mechanism is isolated or newly demonstrated (cf. Schneider et al.,
TENT, both of which give concrete adaptation mechanisms this paper
could have but did not test). No new benchmark, dataset, or task is
introduced (the paper uses existing MedMNIST subsets exactly as
distributed). The paper's most defensible non-empirical contribution is
its two-tier preregistered/secondary evidentiary reporting discipline
applied end-to-end through a sealed computational pipeline -- a
methodological-rigor contribution, not a scientific-method or
mechanism contribution, and not the kind of contribution a CVPR main
track typically rewards on its own.

### 6-7. Closest five-to-ten papers, and exactly how this manuscript
differs from each

| # | Closest work | How this manuscript differs |
|---|---|---|
| 1 | Medeiros (2026), arXiv:2604.09697 -- the direct source study | This manuscript is an independent, preregistered reproduction with (a) a normalization ablation, (b) a resolution ablation, (c) a training-policy-matching arm, and (d) an explicit preregistered/secondary evidentiary separation the source preprint does not use. It does not propose a new TTA method, matching the source's own scope. |
| 2 | Shanmugam et al. (2021), ICCV -- harm-rate framing, learned aggregation fix | This manuscript adopts the harm-framing concept but not the learned-aggregation method; it studies *why*/*when* harm occurs (normalization, resolution, policy match) rather than proposing a better aggregator. |
| 3 | Lyzhov et al. (2020), UAI -- Greedy Policy Search (validation-driven TTA policy selection) | GPS selects *which test-time augmentations to apply*; this manuscript's policy-matching arm instead asks whether *training-time* augmentation exposure mitigates harm from a *fixed* test-time policy -- a different lever, not a competing method for the same lever. |
| 4 | Kim et al. (2020), NeurIPS -- learned per-sample transform-loss prediction | Kim et al. is instance-level, learned, and operates at test time; this manuscript's mitigation is model-level (a training-time choice) and non-learned. Different granularity and different mechanism. |
| 5 | Sherkatghanad et al. (2024), BayTTA -- Bayesian aggregation for medical TTA | BayTTA proposes a new aggregation method; this manuscript uses plain mean-probability aggregation throughout and studies harm/mitigation rather than proposing an aggregator. |
| 6 | Di Salvo et al. (2024), MedMNIST-C -- corruption-robustness benchmark via training-time augmentation | Different lever entirely (training-time robustness augmentation vs. test-time aggregation harm); shared substrate (MedMNIST+) only. |
| 7 | Schneider et al. (2020), NeurIPS -- BatchNorm statistics adaptation under corruption shift | Schneider et al. demonstrate a concrete adaptation mechanism and evaluate it; this manuscript cites the mechanism as background but never implements or tests BN adaptation itself, despite BN adaptation being directly available and directly relevant to its own H1 normalization question. |
| 8 | Wang et al. (2021), TENT (new source) -- entropy-minimization test-time adaptation of BN affine parameters | Directly relevant, uncited-by-the-manuscript prior art for *actually adapting* BatchNorm at test time (rather than only ablating BatchNorm vs. GroupNorm as a static architectural choice); this manuscript does not attempt any form of test-time adaptation. |
| 9 | Ayhan & Berens (2018), MIDL (new source) -- TTA for uncertainty estimation, medical imaging | Different objective (uncertainty estimation vs. point-accuracy harm) in the same broad setting (medical imaging, TTA); this manuscript's harm/rescue-rate reporting is adjacent to, but not framed as, an uncertainty-calibration analysis. |
| 10 | Kimura (2024), arXiv (new source) -- theoretical bound on TTA error via an "ambiguity term" | Provides a theoretical lens (prediction diversity/correlation) this manuscript's Discussion could have used instead of, or alongside, its BatchNorm-distribution-shift hypothesis, but does not cite or engage with. |

### 8. Is the current novelty claim defensible?

**As narrowly worded in
`docs/phase2b_manuscript_claims_and_structure_freeze.md` sec.5 and the
manuscript's own "Novelty scope" paragraph, yes -- technically.** The
claim is hedged to a specific combination (matched-vs-unmatched
training policy as a fixed-model secondary DiD, plus a controlled
normalization/resolution ablation, on MedMNIST, under a two-tier
reporting discipline) and explicitly disclaimed as based on a
non-exhaustive search. Nothing in this expanded review contradicts that
narrow claim. **However, defensible-as-worded is not the same as
sufficient for a top-tier venue.** A CVPR main-track reviewer does not
grade novelty claims on whether they are technically true; they grade
on whether the specific new combination is *interesting and impactful*
enough to warrant publication, and on that harder standard, the
combination identified here -- a secondary, non-preregistered,
6-pair-per-arm ablation of an intuitively-expected mitigation, on a
narrow architecture/dataset matrix, with heterogeneous, unmechanized
secondary results on the other two ablation axes -- is thin. Three
directly relevant families of prior art (TENT-style test-time
adaptation, medical-image TTA uncertainty estimation, and theoretical
TTA-error analysis) are not engaged with anywhere in the current
manuscript, which further narrows how much genuinely new ground the
paper can claim once they are accounted for.

## Novelty score: **2 / 5**

The paper replicates and modestly extends an already-published
(preprint) empirical finding, in a setting (medical-imaging TTA) that
already has direct prior art on both the aggregation-method side
(BayTTA) and the uncertainty-estimation side (Ayhan & Berens), using a
mitigation lever (training-time policy matching) that is an intuitively
expected instantiation of a well-established general principle rather
than a surprising discovery, and does not engage with the closest
mechanism-oriented prior art (Schneider et al., TENT) that could have
turned an unmechanized ablation into a mechanism-isolating one.

## Recommendation for a top-tier computer-vision main track: **Reject**

The paper's genuine contribution -- rigorous, auditable, preregistered
evidentiary discipline applied to a reproduction-and-extension study --
is real but is a methodological-process contribution, not a scientific
discovery, and CVPR main-track novelty expectations are not met by
process rigor alone when the underlying empirical claims are largely
confirmatory of existing prior art. This does not mean the work lacks
value (see `paper/reviews/meta_review.md` for venue-fit discussion
distinct from this novelty verdict).
