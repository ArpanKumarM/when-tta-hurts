# Phase 2C.2 — Label-Preservation Audit Protocol — FROZEN

**Status: pre-registered protocol. Frozen on commit of this file. No
audit sample has been drawn, no augmented view has been rendered for
this audit, and no result exists. The sampling seeds, sample sizes,
proxy formulas, decision rules, and interpretation thresholds below are
fixed *before* any audit output is produced, and before the per-cell
harm results are consulted for this purpose.**

This protocol governs one small, self-contained study. It does **not**
modify `configs/experiment_matrix.yaml`, the frozen augmentation table
in `docs/experimental_protocol.md`, the confirmatory endpoints, the
sealed final-test artifacts, `paper/manuscript.md`, or any other
`docs/` freeze. It requires no training, no checkpoint access, and **no
test-split access**.

---

## 1. Motivation

`paper/reviews/reviewer_1_scientific_soundness.md` §4 (echoed by
Reviewer 2 §4 and the meta-review §2) identifies the single most
damaging unaddressed question in the manuscript: the observed
naive-TTA harm cannot currently be distinguished from a less
interesting alternative — that the fixed **mixed** augmentation policy
(rotation ±15°, random resized crop 0.8–1.0×, brightness/contrast
jitter ±0.3, Gaussian blur) is simply too severe for these modalities
and removes the labeled diagnostic content from a meaningful fraction
of augmented views. Averaging predictions over views that no longer
depict the labeled class would degrade accuracy almost by definition,
which is closer to a sanity check than an empirical finding about TTA.

This audit measures, for the frozen mixed policy, **what fraction of
augmented views still contain the labeled structure**, via a
reproducible automated structural proxy (primary quantitative
evidence) and a small blinded human content-presence check (primary
semantic evidence). Its output feeds a **pre-registered** rule (sec. 7)
that determines how the manuscript's causal framing must be qualified.

## 2. Firewall and ordering guarantees

* **Split.** The audit uses the **validation** split of each dataset,
  never the test split. Label preservation is a property of
  (augmentation policy × image content × view seed); it does not depend
  on any trained model or on which split an image belongs to. Using
  validation keeps the test firewall (`docs/experimental_protocol.md`)
  absolutely intact. The mixed policy and the per-view seed function
  (`src/when_tta_hurts/evaluation/views.py::stable_view_seed`) are
  identical across splits, so the measured rates transfer to the
  test-split views used in the confirmatory runs in expectation.
* **No harm-result steering.** The sampling is a deterministic function
  of a frozen seed only (sec. 3). The set of images and views audited
  is fixed by this document and does **not** depend on which cells
  showed the most or least harm. The audit script must be committed and
  run to completion before its numbers are read against
  `artifacts/secondary_analysis_expansion/` or
  `artifacts/final_test_scientific_summary.json`.
* **Single run, reproducible.** The automated proxy is a pure function
  of the rendered views and the kornia transform parameters; re-running
  the audit script must reproduce byte-identical proxy outputs.

## 3. Sampling frame — FROZEN

Let `TTA_SEED` be the frozen `confirmatory_tta_seed`
(`configs/validation_evaluation.yaml`, value **1306178015**) already
used for the confirmatory view sequence — read from that file by the
audit script, not re-chosen here. Let
`AUDIT_SALT = "phase2c2_label_preservation"`.

For each dataset `d ∈ {pathmnist, bloodmnist, dermamnist}` at the
**28 px** resolution (the resolution common to all three and to the
source study's headline condition):

1. **Image sample.** Draw `N_IMG = 200` distinct sample indices from
   `d`'s validation split, uniformly without replacement, using
   `numpy.random.default_rng(seed_d)` where
   `seed_d = sha256(f"{AUDIT_SALT}|images|{d}|{TTA_SEED}") mod 2**63`.
2. **View sample.** For each sampled image, generate the first
   `N_VIEW = 16` augmented views of the **mixed** policy using exactly
   `stable_view_seed(TTA_SEED, d, 28, sample_index, view_index)` for
   `view_index ∈ [0, 16)` — i.e. a literal prefix of the same
   registered view sequence the confirmatory evaluation used.

This yields **3,200 augmented views per dataset (9,600 total)**, plus
the 200 clean originals per dataset as references. `N_IMG = 200` and
`N_VIEW = 16` are fixed here; 16 is chosen because the per-view
preservation rate is, by construction, independent of the view index
(each view draws policy parameters i.i.d.), so 16 views/image is ample
to estimate a per-dataset rate with a <±2 pp standard error at
plausible rates, and the full N=50 confirmatory budget adds no
information to this specific measurement.

## 4. Automated structural proxy — FROZEN

For each augmented view the audit script recovers the **actual** sampled
kornia parameters (not the nominal ranges) from each op's `._params`
after `sample_deterministic_view`. These keys are confirmed present in
the pinned kornia 0.8.3: `RandomHorizontalFlip`/`RandomVerticalFlip`
`_params['batch_prob']` (flip applied?), `RandomRotation`
`_params['degrees']` (signed angle), `RandomResizedCrop`
`_params['src']` (source-box quad corners, original-pixel units →
`(x0, y0, w, h)`), `ColorJitter` `_params['brightness_factor']` /
`['contrast_factor']`, `RandomGaussianBlur` `_params['sigma']` with
`['batch_prob']` (blur's `p=0.5` fired?). From these it computes:

| Quantity | Definition |
|---|---|
| `crop_area_retained` | `(w · h) / (H · W)` of the source crop box against the original image |
| `center_retained` | fraction of the original image's **central 60 % square** that lies inside the source crop box (after applying the same flip/rotation mapping), i.e. how much of the middle of the image survives the crop |
| `foreground_retained` | fraction of the clean image's **foreground mask** pixels that lie inside the source crop box. Foreground mask = Otsu threshold on the clean image's saturation channel (HSV), taking the higher-saturation side; **computed for `bloodmnist` and `dermamnist` only** (see below) |
| `intensity_shift` | `|1 − brightness_factor| + |1 − contrast_factor|` |
| `blur_sigma` | sampled Gaussian-blur sigma, or `0` if not applied |

**Per-dataset "label-plausibly-preserved" decision rule (frozen):**

* **bloodmnist** (single stained leukocyte, figure-ground structure):
  preserved ⇔ `foreground_retained ≥ 0.60` **and**
  `crop_area_retained ≥ 0.50` **and** `intensity_shift ≤ 0.45`.
* **dermamnist** (lesion usually central, automatic lesion
  segmentation unreliable at 28 px): preserved ⇔
  `center_retained ≥ 0.70` **and** `crop_area_retained ≥ 0.50`
  **and** `intensity_shift ≤ 0.45`. `foreground_retained` is recorded
  but not used in the rule.
* **pathmnist** (colorectal tissue texture, approximately stationary
  within a 28 px patch — no discrete labeled object): preserved ⇔
  `crop_area_retained ≥ 0.50` **and** `intensity_shift ≤ 0.45`.
  `foreground_retained` is not defined for stationary texture and is
  not computed.

The `0.60 / 0.70 / 0.50 / 0.45` thresholds are heuristic and are fixed
here **before** any view is rendered. They are deliberately lenient
(a view is called "preserved" unless a clear majority of the relevant
content is lost or the intensity shift is near the policy maximum), so
that the audit is biased toward *under*-reporting label loss rather
than manufacturing a confound. The rule is a geometric/photometric
stress test, **not** a semantic oracle — its role is to give a
reproducible, annotator-free scalable estimate that the human check
(sec. 5) then validates.

## 5. Human content-presence spot-check — FROZEN

**Sample.** For each dataset, select `N_HUMAN = 50` augmented views for
annotation, drawn from the 3,200 by a frozen stratified rule: 25 views
the automated rule calls *preserved* and 25 it calls *not preserved*
(if fewer than 25 not-preserved exist, take all of them and top up from
preserved), selected by
`numpy.random.default_rng(sha256(f"{AUDIT_SALT}|human|{d}|{TTA_SEED}"))`.
This guarantees the proxy–human agreement in sec. 6 is measurable
across both proxy verdicts. 150 augmented views total.

**Presentation.** Each item shows the **clean original** with its class
name, beside **one augmented view**, in randomized order, with the
automated proxy score and all harm results **hidden** from the
annotator. The task is a **content-presence** judgment, explicitly
**not a diagnosis**:

> *Score 2* — the structure the label refers to is clearly still
> present and identifiable in the augmented view (for blood: the white
> blood cell is still substantially in frame; for pathology: the tissue
> texture is still clearly visible and representative; for derma: the
> lesion is still substantially in frame).
> *Score 1* — partially present: degraded, cropped, or distorted, but a
> viewer could still plausibly tell it is the same class.
> *Score 0* — the labeled structure is largely gone: cropped out of
> frame, blurred/color-shifted beyond recognition, or otherwise not
> identifiable as the class.

**Annotators.** This audit is executed by a **single annotator**
(decided at freeze time). The manuscript discloses single-rater
annotation as a stated limitation; no inter-annotator κ is computed.
Each item's score is that annotator's score. To reduce within-rater
drift the 150 items are rated in a frozen shuffled order (seed
`sha256(f"{AUDIT_SALT}|order|{TTA_SEED}")`) with dataset identity not
grouped.

## 6. Primary reported quantities — FROZEN

Computed once, per dataset, and reported in full regardless of value:

1. `a_not_preserved(d)` — fraction of the 3,200 augmented views the
   automated rule calls **not preserved**.
2. Distributions of `crop_area_retained`, `center_retained`,
   `foreground_retained` (where defined), `intensity_shift`,
   `blur_sigma` — mean, 5th/50th/95th percentile.
3. `p_gone(d)` — fraction of the 50 human-checked views with median
   score **0**; also the score-1 and score-2 fractions.
4. Proxy–human agreement: raw agreement and Cohen's κ between
   "human score ≥ 1" and "automated rule = preserved" on the 50
   items; plus the human score distribution split by automated verdict.
5. Inter-annotator κ: N/A (single annotator, disclosed).

## 7. Pre-registered interpretation rule — FROZEN

Applied per dataset after the quantities in sec. 6 exist:

* **Rule A — confound is material for dataset `d`.** If
  `p_gone(d) ≥ 0.15` **or** `a_not_preserved(d) ≥ 0.25`, the manuscript
  will: (i) report both rates; (ii) state explicitly that augmentation
  severity / label non-preservation cannot be excluded as a substantial
  contributor to `d`'s observed harm; (iii) soften causal language for
  `d` from "TTA hurts" toward "naive TTA under this policy hurts, and
  part of that may be policy-induced label loss"; (iv) list this as a
  primary limitation.
* **Rule B — confound unlikely to dominate for dataset `d`.** If
  `p_gone(d) < 0.15` **and** `a_not_preserved(d) < 0.25`, the
  manuscript will report both rates and state that label
  non-preservation affects only a minority of views and, given
  mean-probability aggregation over 50 views, is unlikely to be the
  dominant driver of `d`'s ≥ 40 pp harm — while still disclosing the
  measured rates and the residual uncertainty.

**Rationale for the 0.15 / 0.25 cutoffs (fixed now).** Under
mean-probability aggregation over 50 views, if only a minority of views
lose the label and the majority are intact, the aggregate is dominated
by the intact majority and cannot by itself produce a 40–66 pp drop;
a not-preserved fraction of ≥ 25 % is a generous lower bound on the
"could plausibly matter a lot" regime. The thresholds decide only the
*wording* of the confound discussion; the measured rates are always
reported verbatim.

## 8. Outputs and reproducibility

```
artifacts/label_preservation_audit/
  summary.json          # all sec.6 quantities + the sec.7 verdict per dataset
  per_view.csv          # 9,600 rows: dataset, sample_index, view_index,
                        # every recovered kornia param, every proxy quantity,
                        # automated preserved/not verdict
  human_sheet.csv       # 150 rows for annotation (blinded columns only)
  human_scores.csv      # filled-in scores (input by annotator(s))
  manifest.json         # sha256 of this protocol, the rendered-view tensor,
                        # dataset checksums, TTA_SEED, all rng seeds
  views/                # PNG renders: clean + sampled augmented views
```

The audit script (`scripts/label_preservation_audit.py`, to be added)
performs steps 3–4 and 6(1–2) deterministically and writes everything
except `human_scores.csv`. Steps 5 and 6(3–5) require the filled
`human_scores.csv` and are computed by a second read-only pass.

## 9. Non-goals and disclosed limitations

* Not a clinical or diagnostic judgment; the human task is content
  presence only.
* 28 px only. The 64/128 px views could lose slightly less relative
  content to a fixed-scale crop; this audit does not measure that and
  the manuscript will not extrapolate the rate across resolutions.
* The automated rule is a geometric/photometric proxy with lenient,
  pre-committed thresholds, validated against — not replaced by — a
  small human sample. Its per-dataset validity is weakest for
  `dermamnist` (no reliable lesion segmentation at 28 px) and for
  `pathmnist` (no discrete labeled object); this is stated wherever the
  rate is reported.
* Single-annotator execution, if that is what happens, is a stated
  limitation, not a silent one.
* This audit informs how the existing harm result is *framed*; it does
  not re-open, re-run, or re-analyze any confirmatory cell.

## 10. Sibling item

The per-augmentation-component ablation (geometric-only vs.
intensity-only vs. mixed, `docs/phase2b_protocol.md` §3.2) is the other
half of the Reviewer 1 §4 response. It requires a fresh **evaluation**
pass over existing checkpoints (no training, no new test images beyond
those already used) and will be specified in a separate frozen
addendum, `docs/phase2c2_component_ablation_addendum.md`, with its own
authorization step because it does touch the test-split evaluation
path.
