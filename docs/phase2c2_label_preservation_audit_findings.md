# Phase 2C.2 — Label-Preservation Audit: Findings

**Status: descriptive report of a pre-registered audit executed exactly
per `docs/phase2c2_label_preservation_audit_protocol.md` (protocol
sha256 `59bf3248…`).** All quantities trace to
`artifacts/label_preservation_audit/summary.json` and the CSVs beside
it, produced by `scripts/label_preservation_audit.py`. Validation split
only; no training, no checkpoint access, no test-split access. Single
annotator (frozen at protocol time); no inter-annotator κ.

---

## 1. Headline

For the frozen **mixed** augmentation policy at 28 px, the labeled
diagnostic structure survives in the large majority of augmented views
on all three datasets. Applying the protocol's pre-registered
interpretation rule (sec. 7):

| dataset | human "content gone" (`p_gone`) | human "clearly present" | automated not-preserved | **verdict** |
|---|---|---|---|---|
| BloodMNIST | **0.0 %** (0/50) | 98 % | 12.2 % | **Rule B** — confound unlikely to dominate |
| DermaMNIST | **2.0 %** (1/50) | 96 % | 13.4 % | **Rule B** |
| PathMNIST | **4.0 %** (2/50) | 72 % (+ 24 % "degraded but recognisable") | 12.5 % | **Rule B** |

Both pre-registered "material confound" triggers — human `p_gone ≥
0.15` or automated not-preserved `≥ 0.25` — are missed on every
dataset. A −40 to −66 pp accuracy drop (Phase 2B / 2C) cannot be
explained by 0–4 % of augmented views losing the label under
mean-probability aggregation over 50 views.

## 2. What actually changes in an augmented view

From the 9,600-view automated parameter recovery
(`per_view.csv`), pooled per dataset:

| quantity | PathMNIST | BloodMNIST | DermaMNIST |
|---|---|---|---|
| `crop_area_retained` (mean) | 0.78 | 0.78 | 0.78 |
| `center_retained` (mean / p05) | 1.00 / 0.99 | 1.00 / 0.99 | 1.00 / 0.99 |
| `foreground_retained` (mean) | n/a (stationary texture) | 1.00 | 0.95 |
| `intensity_shift` (mean / p95) | 0.30 / 0.50 | 0.30 / 0.50 | 0.30 / 0.51 |
| `blur_sigma` (p50 / p95) | 0.0 / 1.82 | 0.12 / 1.82 | 0.14 / 1.80 |

**The geometric transforms essentially never remove the labeled
content.** The random-resized-crop keeps ~78 % of image area on
average and never approaches the 0.50 floor; `center_retained` and
`foreground_retained` are ≈ 1.0 almost everywhere. This directly
contradicts the specific mechanism proposed in
`paper/reviews/reviewer_1_scientific_soundness.md` §4 ("a random
resized crop can plausibly remove the lesion … entirely") — with
`scale=(0.8, 1.0)`, it cannot.

The only proxy quantity that ever trips the "not preserved" rule is
`intensity_shift` (strong brightness + contrast jitter), on ~12–13 %
of views.

## 3. Automated proxy vs. human judgement — they disagree

| dataset | raw agreement | Cohen's κ | human score on views the proxy called NOT-preserved |
|---|---|---|---|
| PathMNIST | 54 % | 0.08 | 72 % `2`, 20 % `1`, 8 % `0` |
| BloodMNIST | 50 % | 0.00 | 100 % `2` |
| DermaMNIST | 48 % | −0.04 | 100 % `2` |

The automated proxy's "not preserved" calls — driven almost entirely by
`intensity_shift > 0.45` — do **not** correspond to human-perceived
content loss. A large brightness/contrast shift usually leaves the
cell, tissue, or lesion fully recognisable, so the intensity threshold
**over-flags** as a semantic proxy.

Two consequences for reporting:

1. The **human `p_gone` rate is the primary quantity**; the automated
   not-preserved rate is reported as a *conservative upper bound that
   the human check does not corroborate*, not as a validated measure.
2. κ is near-degenerate here because 96–100 % of human labels fall in a
   single category (`2`), leaving almost no variance for κ to act on —
   the crosstab and raw agreement are the readable statistics, and both
   say the same thing: the proxy is pessimistic, the true content-loss
   rate is lower.

This disagreement does not weaken the audit's conclusion. The proxy was
the pessimistic signal (12–13 %), and even it over-counts relative to
human judgement, so the confound is *less* of a concern than the
automated number alone would suggest.

## 4. Per-dataset notes

* **BloodMNIST** — cleanest case. 49/50 augmented views "clearly
  present", 1 "degraded", 0 "gone". The single leukocyte is centred and
  compact; crop/rotation cannot displace it and colour shifts don't
  hide it.
* **DermaMNIST** — 48/50 "clearly present", 1 "degraded", 1 "gone".
  The one `0` was an already-low-contrast basal-cell-carcinoma image
  that the brightness jitter washed nearly to white. `foreground_retained`
  p05 = 0.67 indicates the lesion is occasionally partly clipped, but
  this rarely reached human-judged loss.
* **PathMNIST** — most affected: 24 % of views scored `1` and 4 %
  scored `0`. Consistent with the label being a *texture* and Gaussian
  blur specifically attacking texture. Still, only 2/50 views were
  judged to have actually lost the tissue structure.

## 5. Limitations (disclosed)

* Single annotator (pre-registered choice). No inter-annotator κ; the
  score distribution is one person's judgement.
* 28 px only. Higher-resolution views lose proportionally less content
  to a fixed-scale crop; not measured, not extrapolated.
* The content-presence task is not a clinical judgement.
* The automated proxy's intensity threshold is shown here to over-flag;
  it is retained only as a documented, pre-registered upper bound.

## 6. Manuscript actions (per protocol sec. 7, Rule B ×3)

For each dataset: report both rates; state that label non-preservation
affects only a minority of augmented views (0–4 % judged "gone") and,
given mean-probability aggregation over 50 views, is unlikely to be the
dominant driver of the observed ≥ 40 pp harm; disclose the residual
uncertainty and the single-annotator limitation.

Concretely, add to the manuscript:

1. A short **"Label-preservation audit"** paragraph in Results or a
   dedicated subsection: the human `p_gone` rates, the geometric-
   retention numbers from §2, and the Rule-B verdict.
2. A sentence in **Discussion** explicitly closing Reviewer 1 §4: the
   observed harm is **not** attributable to the crop removing labeled
   content (it keeps ≥ ~78 % of area, ≈ 100 % of the image centre), and
   only a small minority of views suffer human-judged label loss —
   so "aggressive augmentation destroys the label" is not a sufficient
   explanation for the effect's magnitude.
3. A **Limitations** line: single-annotator audit, 28 px, automated
   proxy over-flags on intensity and is reported as an upper bound only.
4. Keep the causal framing as-is (no softening required under Rule B),
   but cite this audit wherever the harm is described as a property of
   TTA rather than of the policy.

## 7. Still open

The per-augmentation-component ablation (geometric-only vs.
intensity-only vs. mixed) remains the other half of the Reviewer 1 §4
response and is the natural next step — it would show directly whether
the harm is carried by the geometric or the intensity family, which
this audit only characterises indirectly (geometry preserves content;
intensity shifts are large but human-survivable). Specified separately
in `docs/phase2c2_component_ablation_addendum.md` (to be written); it
touches the test-split evaluation path and needs its own authorization.
