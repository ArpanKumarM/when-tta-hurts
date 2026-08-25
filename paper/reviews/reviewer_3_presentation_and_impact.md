# Reviewer 3 — Presentation, Impact, and Reproducibility

**Manuscript:** "When Test-Time Augmentation Hurts: A Controlled Study
in Medical Image Classification" (`paper/manuscript.md`, HEAD `4c70ddb`)
**Reviewer role:** hostile, reviewer-style read for a top-tier
computer-vision main track, focused on presentation quality,
practitioner impact, and reproducibility rather than the underlying
science (see Reviewer 1) or novelty (see Reviewer 2).

## 1. Abstract strength

Clear, well-scoped, and unusually honest for an abstract (it states the
BLOCK_C non-reproduction plainly rather than burying it). Its main
weakness is rhetorical, not factual: it leads with the two
strongest-sounding numbers ("all 30 cells," "all six 95% confidence
intervals excluding zero") before any tier-of-evidence context is
established, so a reader who stops at the abstract receives a more
confident impression of the secondary policy-matching result than the
paper's own body text intends. A reviewer skimming only the abstract at
a program-committee meeting would likely remember "matched-policy
training fixes TTA" more strongly than the paper's careful hedging
supports.

## 2. Introduction and motivation

Solid. The practitioner-facing motivation paragraph (why the
preregistered/secondary distinction matters operationally, not just
statistically) is a genuine strength and is more concrete than most
empirical-ML introductions. The Introduction correctly front-loads the
scope disclaimers (no clinical claim, no state-of-the-art claim, no
broad generalization) rather than saving them for Limitations, which
reduces the risk of a reviewer feeling misled later.

## 3. Is the contribution list concrete?

Partially. The Introduction states the contribution as "a controlled,
statistically disciplined, fully audited account... and an explicit
demonstration of how much confidence a non-preregistered secondary
comparison can and cannot support," which is accurate but is a
methodological/epistemic contribution stated in the abstract of a
methods paper, not a bulleted, concrete contribution list of the kind
CVPR reviewers expect (e.g., "(1) we show X on Y datasets; (2) we
introduce Z; (3) we release W"). There is no explicit numbered
contribution list anywhere in the manuscript. This is a presentation
gap independent of whether the underlying contributions are sound: a
reviewer skimming for "what exactly is new here" in under 30 seconds
will have to reconstruct the answer from prose rather than read it
directly.

## 4. Figure readability and narrative order

The five figures (per the committed evidence package,
`artifacts/paper_evidence/figures/`) are clean, colorblind-safe
(Okabe-Ito palette), and individually well-labeled with explicit
evidence-tier captions -- a real strength, since misleading axis
truncation and unlabeled evidence tiers are common empirical-ML
presentation failures this paper avoids. The narrative order (Figure 1:
headline harm -> Figure 2: mitigation -> Figure 3: normalization ->
Figure 4: resolution -> Figure 5: positive control) matches the
Results section's paragraph order and is logical. One presentation
weakness: Figures 3 and 4 (the two heterogeneous, sign-reversing
secondary results) are visually the most complex (faceted panels,
dashed reference lines) but carry the least conclusive narrative
payload, which risks reader fatigue exactly where the paper has the
least to show -- a reviewer would likely suggest compressing or
appendix-deferring one of them if space is tight, which cuts against
this phase's own placement freeze
(`docs/phase2b_manuscript_claims_and_structure_freeze.md` sec.7, which
keeps both in the main text).

## 5. Table usefulness

Table 1 (design/evidence classification) and Table 7 (claim
adjudication) are genuinely useful orientation devices, unusual and
valuable for a paper this careful about evidentiary tiers. The five
remaining tables are complete and correctly deferred to supplementary
material per the freeze document; the main text's reliance on
figures plus narrative point-estimates (rather than inline tables) for
Results is a reasonable and common CVPR-style choice.

## 6. Does the paper clearly distinguish preregistered and secondary
evidence?

**Yes -- this is the paper's strongest presentation asset.** Every
Results paragraph, every figure caption, and the Discussion
consistently and correctly label which tier each claim belongs to.
This is executed more rigorously than in the median empirical ML paper
this reviewer has seen, and the automated verification tooling
(`paper/verify_manuscript_claims.py`) backing this discipline is an
unusual and creditable level of process rigor, though it is a process
detail a CVPR reviewer is unlikely to weight heavily on its own (see
Reviewer 2 on novelty of process-rigor-as-contribution).

## 7. Is the audit history transparent without overwhelming the paper?

Mostly successful, with one exception. The Limitations section's
"Process incidents disclosed for completeness" paragraph is
information-dense (an access incident, two failed attempts, a
pipeline-contract correction, a reconciliation step, a recomputation
check, and a stale-test-guard fix, all in one paragraph) and reads more
like an engineering changelog than a scientific limitation. A reviewer
would likely ask for this to be compressed to its scientific
consequence ("none of these incidents changed any reported number; full
detail is in the project's public audit trail") with the itemized
detail moved to a supplementary note, which the Supplementary Material
Outline already gestures toward (Note C) but does not fully realize by
moving the paragraph itself out of the main Limitations text.

## 8. Does the main finding matter to practitioners?

Yes, and this is the paper's clearest source of practical impact: a
concrete, large-magnitude (up to -66pp) demonstration that a
commonly-recommended, "nearly free" inference-time technique can
catastrophically fail on ordinary (non-adversarial) medical-imaging
inputs is actionable information for any practitioner currently
running or considering naive TTA in a similar pipeline. This impact is
somewhat narrowed by Reviewer 1's point that the paper does not rule
out "the augmentation policy itself was too aggressive" as the driver,
which affects *how* practitioners should act on the finding (validate
policy severity, not just "avoid naive TTA") more than *whether* they
should be warned at all.

## 9. Do results generalize enough to justify the title?

The title, "When Test-Time Augmentation Hurts: A Controlled Study in
Medical Image Classification," is honest and appropriately modest --
"a controlled study," not "TTA hurts medical image classification" as
a bare claim -- and the paper's own scope disclaimers reinforce this
throughout. A reviewer would not flag the title itself as
overclaiming. The title's implicit promise ("medical image
classification" broadly) is, however, answered by a study covering
three datasets and effectively one architecture family per ablation
axis (Reviewer 1 sec.1-2), which is a narrower evidentiary base than
"medical image classification" as a category suggests, even though the
body text is careful never to claim more than it shows.

## 10. Does the paper tell a coherent story despite the failed
positive control and heterogeneous secondary results?

Yes, and this is a genuine achievement of the writing, not just the
statistics. Rather than suppressing or minimizing BLOCK_C's failure or
the normalization/resolution reversal, the paper builds its narrative
arc explicitly around the distinction between "what we can say
confidently" (the 30-cell harm) and "what we cannot yet say
confidently" (everything else), and uses the null/heterogeneous results
as evidence *for* that framework rather than as embarrassments to
explain away. This is a mature way to present a study with one strong
result and several inconclusive ones, though it does mean the paper's
narrative energy is concentrated almost entirely in its first Results
paragraph, with the remaining four largely building a case for caution
rather than additional discovery.

## 11. Reproducibility quality

**Excellent, and unusual for this class of paper.** Every reported
number traces mechanically to a hash-verified, manifest-bound evidence
package; a dedicated, tested, read-only verification script exists;
the underlying computational pipeline is cryptographically fingerprint-
bound with an authorization chain; and the paper explicitly
distinguishes automated computation from human inspection at the
generation of its own evidence. This is meaningfully above the
reproducibility bar typically seen even in accepted CVPR papers, where
"reproducibility" usually means "code will be released" rather than
"every number is independently, mechanically re-checkable against a
sealed evidence artifact." If the venue gave explicit credit for
reproducibility engineering (some workshops and journals do; CVPR main
track's reviewer form does not have a dedicated high-weight
reproducibility criterion), this would be a standout strength.

## 12. Missing ablations or analyses reviewers would likely request

* A view-count scaling curve (N=1..100), which the project's own frozen
  protocol already specifies as a secondary analysis but which does not
  appear in this manuscript at all (also flagged by Reviewer 1).
* A per-augmentation-component ablation (geometric-only vs.
  intensity-only vs. mixed) to localize which part of the policy drives
  harm -- directly relevant to the label-preservation concern in
  Reviewer 1 sec.4.
* At least one BatchNorm-adaptation or TENT-style test-time-adaptation
  comparison point, given both are cited as background (Schneider et
  al.) or newly identified as closely relevant (TENT, per Reviewer 2)
  but never empirically engaged.
* A qualitative or quantitative label-preservation check for a sample
  of augmented views (even a small human-annotated spot check would
  substantially strengthen the causal story).
* Calibration/uncertainty metrics (ECE, NLL) alongside accuracy, given
  the harm/rescue-rate framing already gestures toward
  distributional effects that calibration metrics would characterize
  more directly, and given the newly-identified Ayhan & Berens /
  BayTTA prior art frames TTA partly in exactly these terms.

## Scores

* **Clarity: 4 / 5.** Well-organized, consistently labeled, honest
  about limitations; loses one point for the missing concrete
  contribution list and the abstract's rhetorical framing outpacing its
  own hedges.
* **Significance: 2 / 5.** Real, actionable practitioner-relevant
  finding, but narrow scope, an unresolved confound (augmentation
  severity), and heterogeneous secondary results limit how much the
  field should update on this paper alone; see Reviewer 2 for the
  separate novelty dimension of significance.
* **Reproducibility: 5 / 5.** Fully mechanically verifiable, hash-bound,
  independently re-checkable evidence chain with automated tooling;
  the strongest dimension of this submission by a clear margin.
