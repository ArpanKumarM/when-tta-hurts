# Reviewer 1 — Scientific Soundness

**Manuscript:** "When Test-Time Augmentation Hurts: A Controlled Study
in Medical Image Classification" (`paper/manuscript.md`, HEAD `4c70ddb`)
**Reviewer role:** hostile, reviewer-style read for a top-tier
computer-vision main track. This review evaluates the manuscript as
submitted; it does not evaluate the underlying engineering/audit
process beyond what the manuscript itself discloses.

## 1. Experimental design

The design is a reasonably clean factorial matrix (dataset x resolution
x normalization x seed, plus a policy-matching arm and a positive
control), and its evidentiary-tier discipline (preregistered
within-cell vs. secondary cross-condition) is unusually careful for a
first draft. But the design has a structural asymmetry the manuscript
does not foreground enough: the normalization/resolution ablation
(Blocks A/D) uses **only SmallCNN**, the policy-matching ablation
(Block B) uses **only SmallCNN/BatchNorm/28px**, and the positive
control (Block C) uses **only ResNet-18**. No architecture appears in
more than one ablation axis. This means every cross-axis claim in the
paper (e.g., "does policy matching help ResNet-18 too? Does
normalization choice matter for ResNet-18?") is simply unanswerable
from this design, and the paper does not always say so as plainly as it
should outside the Limitations section.

## 2. Dataset and architecture coverage

Three datasets, one CNN family, one ResNet variant, all at 28-128px.
This is narrow even for a "controlled study" framing, and it is narrow
in a way that specifically limits the paper's own headline claim: three
of three datasets tested happen to be MedMNIST's lower-visual-complexity
subsets (blood cell, colon histopathology patches, dermatoscopy), none
with the higher inter-class visual overlap of some other MedMNIST
subsets (e.g. chest X-ray, OCT) that might behave differently under
aggressive geometric/intensity perturbation. The paper is honest that
this limits generalization (Limitations, Discussion), but a reviewer
would still ask why the two "primary_high_sensitivity" datasets and one
"positive control" dataset were selected specifically to match the
source preprint's set, rather than to maximize coverage of MedMNIST's
diversity.

## 3. Three-seed limitation

Correctly disclosed, but under-weighted in how the headline claim is
phrased. Three, not thirty, is the number of genuinely independent
draws from "seed space" per architecture/dataset/resolution/
normalization cell. See sec.7 below for why this matters more than the
Limitations section's brief mention suggests.

## 4. Is the augmentation policy plausibly too aggressive or
label-changing?

**This is the single most important unaddressed question in the
paper.** The mixed policy combines +/-15 degree rotation, random
resized crop to 0.8-1.0x, brightness/contrast jitter of +/-0.3, and
Gaussian blur. For DermaMNIST (dermatoscopy) and PathMNIST
(histopathology), a random resized crop can plausibly remove the
lesion or the diagnostically relevant tissue region entirely, and a
+/-0.3 contrast/brightness jitter is large relative to the narrow,
diagnostically meaningful intensity range of stained histopathology
images. If a meaningful fraction of augmented views are **not
label-preserving** -- i.e., a view no longer actually depicts the
labeled class in any recoverable sense -- then "naive TTA hurts" is a
much less surprising and much less generalizable finding: it would
mean "averaging in predictions on inputs that no longer show the
labeled structure hurts accuracy," which is closer to a sanity check
than a novel empirical result. The manuscript never performs or even
proposes a label-preservation audit (a spot-check of whether augmented
views still contain the diagnostic content), and Methods/Experimental
Design do not report per-transform ablations that would let a reader
gauge how much of the harm comes from geometric versus intensity
components. This is a genuine, fixable gap (see
`docs/phase2b_submission_gap_analysis.md`), but as submitted, it is a
serious threat to the causal interpretation the paper otherwise argues
for so carefully in every other respect.

## 5. Is N=50 alone sufficient for the paper's primary framing?

No, and this is a self-inflicted narrowing. The project's own frozen
protocol (`docs/phase2b_protocol.md` sec.3) lists a preregistered
"scaling curve" secondary analysis across all seven registered view
counts (1, 2, 5, 10, 25, 50, 100) -- but the manuscript reports **only
N=50** anywhere in Results. A reviewer will immediately ask: does harm
monotonically worsen with N, saturate, or peak somewhere and recede?
Without the curve, N=50 reads as a single, possibly cherry-picked
point chosen only because it matches the source paper's headline
condition, even though the matching is stated as intentional
replication rather than cherry-picking. Given the curve was already
part of the frozen protocol, omitting it from this manuscript (even if
it exists elsewhere in the project) is a presentation gap that
materially weakens the paper's primary framing, independent of whether
the underlying data exists.

## 6. Is the clean-versus-TTA comparison fair?

Within the stated scope -- naive, unweighted mean-probability TTA
versus a single clean forward pass -- yes, the comparison is fair and
correctly paired (same model, same test samples). The paper is
explicit that it studies *naive* TTA, not TTA as a class of techniques,
which is a legitimate and clearly stated scope choice, not an unfair
comparison.

## 7. Is sample-level McNemar/bootstrap evidence being confused with
model-level replication?

The paper does not literally confuse these -- it is careful never to
claim population-level or cross-model generalization from a within-cell
test -- but the *rhetorical framing* risks it. "All 30 distinct cells
showed harm" is a striking, headline-friendly number, but the 30 cells
are not 30 independent replications of "does TTA hurt": they decompose
into roughly 2 datasets x 2 resolutions x 2 normalizations x 3 seeds
(plus 6 Block-D cells at a third resolution), i.e., on the order of
**8-10 independent experimental configurations, each replicated 3
times**, not 30 independent trials. Each within-cell McNemar/bootstrap
test's statistical power comes from thousands of paired test samples,
which correctly establishes that the effect is real and large *within
that one trained model* -- but it says nothing about how consistently
the effect would recur across a hypothetical population of
independently retrained models beyond the 3 seeds actually drawn. The
manuscript's own Limitations section states this ("sample-level paired
tests within a cell do not substitute for a model-seed population
replication study"), which is the correct caveat -- but the Abstract
and Discussion's repeated, bolded "all 30 cells" framing does more
rhetorical work than the underlying independent-replicate count
supports, and a skeptical reader who only reads the Abstract could come
away with an inflated sense of how many independent times "TTA hurts"
was actually shown, versus how many times a large, consistent, but
*within-cell* effect was measured under 8-10 underlying configurations.

## 8. Is matched-policy mitigation framed correctly as secondary?

Mostly yes, and this is a genuine strength: the manuscript repeatedly
and explicitly labels the six DiD comparisons as secondary,
non-preregistered, and never uses "significant." However, the phrase
"all six 95% confidence intervals excluded zero" appears in the
Abstract itself, in bold, immediately followed by "supported by
secondary evidence" -- for a reader skimming only the Abstract, "6 of 6
CIs exclude zero" functions rhetorically almost exactly like a
significance claim, even though the word is avoided. The tier-labeling
discipline is real, but its protection against overclaiming is only as
strong as a reader's willingness to actually track the tier label
through to the Abstract, and the Abstract itself leads with the
strongest-sounding version of the secondary claim rather than its
weakest defensible form.

## 9. Do normalization and resolution results support any general
claim?

No, and the manuscript says so directly and repeatedly ("heterogeneous,"
"dataset-dependent," "must not be read as a general ... verdict"). This
is correctly and conservatively reported. The cost is that two of the
paper's four ablation axes (H1, H2) produce genuinely inconclusive,
sign-reversing results -- which is scientifically honest but weakens
the paper's overall narrative momentum (a Reviewer-3-relevant point,
noted here only because it also bears on whether the Discussion's
"informative (if not confirmatory)" framing of this heterogeneity is
oversold; "informative" is doing some work to make a null/mixed result
sound like a contribution rather than an absence of one).

## 10. Does the failed/inconsistent BLOCK_C undermine the paper?

It does not invalidate the paper's own within-cell findings (BLOCK_C is
an independent 3-cell arm, unrelated to the mechanics of the 30-cell
headline claim), but it raises a question the paper does not fully
resolve: since several training hyperparameters differ from the source
study by necessity (the source paper under-specifies them), is the
BLOCK_C non-reproduction evidence that TTA's effect is
implementation-sensitive (undermining generalizability of *either*
paper's numbers), or is it a true independent disconfirmation of the
source paper's one positive result? The Discussion acknowledges this
ambiguity but does not push it far enough: if implementation details
this sensitive can flip a result from +1.6pp to -2.89pp, that same
sensitivity calls into question how much weight the paper's own
headline "-66.09pp" and "exceeds the source study's largest drop"
comparisons should carry, since those are also implementation-sensitive
cross-paper comparisons.

## 11. Do incidents, corrections, and retries threaten validity?

Not evidently, but the manuscript's own disclosure is too compressed to
let a reviewer fully evaluate this. "A shared-aggregation-contract
correction was made mid-project after a defect was found" describes
*that* something was fixed but not *what* the defect actually changed
numerically (did any preregistered cell's sign flip? Did any CI
materially widen or narrow?) before-and-after the correction. The paper
asserts the fix was verified via "full recomputation" but does not
report what was recomputed or by how much any number moved. This is a
transparency gap: the disclosure satisfies "don't hide it" but not
"give the reviewer enough detail to independently judge whether it
mattered."

## 12. Do conclusions follow from the evidence?

For the primary claim (naive TTA harmed all 30 cells, within-model), 
yes, cleanly. For the secondary claims (policy-matching mitigation,
normalization/resolution heterogeneity), the conclusions are
appropriately hedged and match the evidentiary tier. The weakest link
in the overall argument is not any single false claim but the
unaddressed alternative explanation in sec.4 above: without a
label-preservation audit or a per-transform-component ablation, the
paper cannot distinguish "TTA harms medical image classification" from
"an aggressive augmentation policy that sometimes destroys the
diagnostic signal harms medical image classification when
naively averaged in" -- and these have very different implications for
practitioners.

## Strengths

* Genuinely careful, consistently enforced preregistered/secondary
  evidentiary separation -- rare in an empirical ML paper, and the
  paper's most distinctive methodological asset.
* Full mechanical traceability of every reported number to a
  hash-verified evidence package, independently checkable.
* Honest reporting of a null/contrary result (BLOCK_C) rather than
  omitting it.
* Correct, paired statistical methodology (bootstrap + McNemar) for the
  within-cell design, with per-family FDR correction reported
  transparently alongside raw p-values.
* Clear, explicit scope disclaimers (no clinical claim, no
  population-level claim, no H4 claim).

## Major weaknesses

1. No label-preservation audit or per-transform-component ablation to
   rule out "aggressive augmentation destroys diagnostic content" as
   the dominant driver of the observed harm (sec.4).
2. The preregistered scaling-curve analysis (view counts 1-100) exists
   in the frozen protocol but is entirely absent from this manuscript,
   leaving the N=50-only framing looking narrower than the project's
   own design intended (sec.5).
3. Architecture/dataset/ablation-axis confounding: no ablation axis is
   tested on more than one architecture, so cross-axis interaction
   claims are structurally unanswerable (sec.1).
4. The Abstract's rhetorical framing ("all 30 cells," "all six 95% CIs
   excluded zero") outpaces the actual independent-replicate count
   (~8-10 configurations x 3 seeds) in a way the body text corrects but
   the Abstract does not (sec.7-8).
5. The engineering-incident disclosure is present but too compressed to
   let a reviewer independently judge whether the shared-aggregation-
   contract defect materially moved any reported number (sec.11).

## Minor weaknesses

* No cross-validation of MPS-backend numerics against a CUDA reference
  run (disclosed, not fixed).
* BLOCK_C's single-architecture, single-dataset design cannot
  distinguish "source result not reproduced" from "this project's
  independent hyperparameter choices produced a different but equally
  valid training outcome" (sec.10).
* The Discussion's framing of heterogeneous normalization/resolution
  results as "informative (if not confirmatory)" is a soft rhetorical
  upgrade of what is, plainly, an inconclusive/null secondary result.

## Questions for the authors

1. Was the scaling curve (N in {1,2,5,10,25,50,100}) actually computed?
   If so, why is it omitted from this manuscript, and does it change
   the picture at smaller or larger N?
2. Can you report, even qualitatively, what fraction of augmented views
   under the mixed policy at N=50 would a human annotator judge to
   still contain the labeled diagnostic structure, for at least one
   dataset?
3. What exactly did the shared-aggregation-contract defect change
   before correction -- did any preregistered cell's sign or
   BH-adjusted significance status differ pre- versus post-correction?
4. Did you consider testing the policy-matching arm (H3) on ResNet-18,
   or the positive control (BLOCK_C) on SmallCNN, to break the
   architecture/ablation-axis confound noted in sec.1? If not
   attempted, why not, given the compute budget appears to have
   permitted a comparable number of additional runs?

## Fatal vs. fixable issues

**Fatal (would require a new frozen protocol and new data, not just
rewriting):** none. Every weakness identified above is addressable
either by additional, clearly-scoped experiments (label-preservation
audit, per-component ablation, scaling curve, cross-architecture
coverage) or by more careful prose (Abstract framing, incident detail).
The core preregistered finding (30-cell within-cell harm) is internally
valid as stated.

**Fixable (in a revision, with or without new data):** all items in
Major and Minor weaknesses above.

## Soundness score: **3 / 5**

The statistical methodology applied to the data actually collected is
sound and unusually disciplined about evidentiary tiers. The score is
capped at 3, not higher, because the paper does not rule out the most
obvious alternative explanation for its headline effect (augmentation
aggressiveness / label preservation), omits a preregistered analysis
(the scaling curve) that bears directly on how the headline condition
should be interpreted, and structurally cannot answer several
cross-axis questions its own Discussion raises.

## Recommendation for a top-tier computer-vision main track: **Reject**

Not because any claim is false, but because the paper does not yet
close the loop on the single most important soundness question a CVPR
main-track reviewer would raise first (is this augmentation-severity
confound, not TTA-in-general harm?), and because a preregistered
analysis directly relevant to that question (the view-count scaling
curve) was apparently run but not reported here. Both are fixable
without new frozen protocols in the harder cases (label-preservation
audit needs new, small-scope work; the scaling curve may already exist
and only needs reporting) -- see
`docs/phase2b_submission_gap_analysis.md` for prioritization.
