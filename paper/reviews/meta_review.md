# Meta-Review — Venue Assessment

**Manuscript:** "When Test-Time Augmentation Hurts: A Controlled Study
in Medical Image Classification" (`paper/manuscript.md`, HEAD `4c70ddb`)
**Synthesizes:** `reviewer_1_scientific_soundness.md` (score 3/5,
reject for CVPR main), `reviewer_2_novelty.md` (score 2/5, reject for
CVPR main), `reviewer_3_presentation_and_impact.md` (clarity 4/5,
significance 2/5, reproducibility 5/5).

This meta-review does not soften any reviewer's verdict. Its job is to
synthesize, not to average toward a more comfortable middle ground.

## 1. Strongest contribution

The end-to-end, mechanically enforced separation of preregistered
within-cell evidence from secondary cross-condition evidence, backed by
a fully hash-verified, independently re-checkable evidence chain
(Reviewer 3, reproducibility 5/5). This is executed with a rigor this
reviewer has not seen in a comparable empirical-ML manuscript, and it
is what makes the paper's one strong empirical claim (naive TTA harmed
all 30 evaluated cells) trustworthy on its own terms, independent of
whether it is novel.

## 2. Most serious scientific weakness

The manuscript cannot distinguish its headline effect from a much less
interesting alternative explanation: that the fixed mixed augmentation
policy (rotation, aggressive crop, +/-0.3 brightness/contrast jitter,
blur) is simply too severe for these specific medical-image modalities
and destroys label-relevant content in a meaningful fraction of
augmented views (Reviewer 1, sec.4). No label-preservation audit and no
per-transform-component ablation exists anywhere in the evidence
package to rule this out. This is not a fatal flaw in the sense of
invalidating the reported numbers -- the numbers are real and correctly
computed -- but it is a fatal flaw *for the paper's implicit causal
framing* ("TTA hurts medical image classification") until it is
addressed.

## 3. Most serious novelty weakness

Both of the manuscript's two potential novelty anchors are already
directly established in prior art the manuscript itself cites or that
this review's expansion surfaces: "TTA can hurt" is already reported by
the manuscript's own source study (Medeiros, 2026) and by Shanmugam et
al. (2021); "TTA in medical imaging" already has direct prior art on
both the aggregation-method side (BayTTA) and the uncertainty-
estimation side (Ayhan & Berens, 2018, newly surfaced in Reviewer 2).
What remains as this paper's actual new ground -- a secondary,
6-pair-per-arm ablation of an intuitively expected mitigation
(train/test augmentation-policy matching), plus two heterogeneous,
unmechanized secondary ablations -- is a narrow, largely-expected
empirical footnote to already-published work, not a new finding a
top-tier venue would consider a discovery (Reviewer 2, sec.3-5, 8).

## 4. Most likely reviewer rejection reason (if submitted to CVPR main
as-is)

A combined novelty-and-significance rejection: "This is a careful,
well-audited reproduction and modest extension of an existing preprint
finding, using an intuitively expected mitigation and inconclusive
secondary ablations, with no new method, mechanism, or benchmark, and
without ruling out an obvious confound (augmentation severity /
label preservation). The engineering rigor is commendable but is not,
by itself, a contribution CVPR main track rewards." This is a
"marginally below threshold on both novelty and significance, with an
open soundness question," not a "the results are wrong" rejection --
an important distinction for how the authors should read this verdict.

## 5. Venue fit

* **Competitive for CVPR main: No.** Novelty score 2/5 and
  significance 2/5 (Reviewer 2, Reviewer 3) are both below what a
  CVPR main track has historically required, independent of the
  paper's genuine methodological rigor.
* **Borderline for CVPR main: No.** This is not a close call. Both the
  soundness reviewer and the novelty reviewer independently reached
  reject, for different, non-overlapping reasons (an open confound, and
  a thin novelty case), which is a stronger signal than either
  objection alone.
* **Better suited to MIDL/MICCAI or another domain venue: Yes, and
  this is the most defensible next step.** A medical-imaging-focused
  venue's reviewer pool is more likely to value (a) a large-magnitude,
  practically actionable warning about a commonly-used inference-time
  technique in exactly their domain, (b) the paper's unusually strong
  reproducibility/audit engineering, which some medical-imaging venues
  explicitly reward, and (c) a careful, non-overclaiming treatment of a
  mostly-null/heterogeneous secondary-results set, which domain venues
  tend to tolerate better than a CV main track chasing a single crisp
  headline result.
* **Strong workshop material: Yes, independent of the domain-venue
  path.** A workshop on trustworthy/reproducible ML, or a TTA- or
  robustness-focused workshop at a CV venue, would likely value this
  paper's core empirical warning and its audit-trail methodology as a
  case study, even though workshop acceptance carries lower stakes and
  does not resolve the underlying novelty/confound issues.
* **Not yet publishable: No.** The writing, statistical methodology,
  and evidentiary discipline are already at or above publishable
  quality for an appropriately-scoped venue; "not yet publishable"
  would understate what has actually been achieved here.

## 6. Critically honest recommendation

Do not submit this manuscript to CVPR main in its current form. The
paper's actual strength -- rigorous, auditable, non-overclaiming
reporting of a mostly-confirmatory, partly-inconclusive empirical study
-- is a genuine asset, but it is not the kind of asset that overcomes a
thin novelty case and an open, first-order confound at a venue that
weighs novelty and significance heavily. The lowest-risk, highest-value
path is either (a) a domain venue (MIDL/MICCAI-class) submission after
addressing the label-preservation/augmentation-severity question, which
plays to this paper's actual strengths, or (b) a workshop submission
now, with a domain-venue submission as a follow-up once the Part F gap
analysis's highest-priority items are addressed.

## 7. Qualitative acceptance likelihood (no invented percentage)

**CVPR main track, as currently written: low likelihood.** This
judgment rests on two independent, non-overlapping reviewer objections
(soundness confound, thin novelty), not a marginal score on one axis,
which this reviewer considers a stronger-than-usual signal against
acceptance. No numeric probability is asserted because no calibrated
basis for one exists in this review process; asserting a specific
percentage here would itself be an unjustified, invented statistic.

**MIDL/MICCAI-class domain venue, after addressing the label-
preservation confound and adding the view-count scaling curve (see
`docs/phase2b_submission_gap_analysis.md` Priority 1 items): moderate-
to-good likelihood**, contingent on those additions and on how the
venue's reviewer pool weighs the paper's already-strong reproducibility
engineering, which this review cannot fully anticipate without knowing
a specific venue's reviewer composition.

**Workshop submission, as currently written: good likelihood**, since
workshop bars are explicitly lower on novelty/significance and this
paper's rigor and honesty are, if anything, above the typical workshop
submission's standard.

## 8. Paper quality versus venue fit (kept explicitly separate)

**Paper quality (writing, statistical rigor, reproducibility,
intellectual honesty): high.** Nothing in any of the three reviews
identifies a false claim, a hidden result, or a methodological error in
how the collected data was analyzed. The preregistered/secondary
evidentiary discipline is executed better than in most published
empirical ML papers this reviewer is aware of.

**Venue fit for CVPR main: poor**, for reasons entirely independent of
the paper-quality assessment above -- narrow scope, no new method or
mechanism, an unresolved confound, and a novelty case that is
technically defensible but thin relative to directly-cited-or-citable
prior art. A high-quality paper and a CVPR-main-competitive paper are
not the same thing, and this manuscript is unambiguously the former
without currently being the latter.
