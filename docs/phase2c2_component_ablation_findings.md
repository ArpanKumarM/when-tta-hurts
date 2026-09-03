# Phase 2C.2 — Per-Augmentation-Component Ablation: Findings

**Status: descriptive report of a pre-registered analysis executed
exactly per `docs/phase2c2_component_ablation_addendum.md` (addendum
sha256 in `artifacts/component_ablation/summary.json`).** Validation
split only; 12 Block A 28 px cells; existing confirmatory checkpoints,
hash-verified against each cell's canonical validation `checkpoint_hash`;
no training, no test-split access. Executed after the final-test
unsealing → reported as **secondary / exploratory**, per the Phase 2C
framing. All numbers trace to
`artifacts/component_ablation/{summary.json, per_cell.csv}`, produced by
`scripts/component_ablation.py`.

---

## 1. Headline

Splitting the frozen **mixed** TTA policy into its **geometric** (flips,
rotation ±15°, resized crop 0.8–1.0×) and **intensity**
(brightness/contrast jitter ±0.3, Gaussian blur) families and running
each alone:

| | geometric-only | intensity-only | mixed | additivity residual |
|---|---|---|---|---|
| **N = 25** | −26.30 pp | −25.10 pp | −43.83 pp | +7.58 pp |
| **N = 50** | −26.20 pp | −23.34 pp | −43.48 pp | +6.06 pp |

(12-cell means; per-cell paired-bootstrap 95 % CIs.)

**Pre-registered verdict (addendum sec. 7): `both_contribute` /
`sub_additive`, at both N.** Neither family dominates
(`|geo| ≈ |int|`, ratio ≈ 1.1, far below the 2× dominance cutoff), and
**every one of the 12 cells shows a negative delta whose 95 % CI
excludes zero for *both* families** at both N (12/12 and 12/12). The
mixed harm (−43 pp) is *less* than the sum of the single-family harms
(−49 pp), i.e. sub-additive (residual +6 to +8 pp): once predictions
are badly degraded by one family, adding the other does not degrade
them proportionally further.

## 2. Per-cell breakdown (N = 50)

| dataset | norm | geometric | intensity | mixed |
|---|---|---|---|---|
| PathMNIST | BatchNorm | −32.19 | −25.60 | −48.42 |
| PathMNIST | GroupNorm | −45.60 | −34.51 | −63.88 |
| BloodMNIST | BatchNorm | −10.44 | −24.61 | −38.49 |
| BloodMNIST | GroupNorm | −16.57 | −8.63 | −23.11 |

(mean over 3 seeds each, pp.)

The relative weight of the two families is **dataset-dependent**:
geometric transforms carry more of the harm on PathMNIST, intensity
transforms carry more on BloodMNIST/BatchNorm, and both are smaller
(but still CI-separated from zero) on BloodMNIST/GroupNorm. In no cell
is either family negligible — the smallest single-family effects
(BloodMNIST/GroupNorm intensity −8.6 pp, BloodMNIST/BatchNorm geometric
−10.4 pp) still have CIs excluding zero.

## 3. Interpretation — closes Reviewer 1 §4

Reviewer 1 §4's alternative explanation for the headline harm was:
*the mixed policy is so severe that a meaningful fraction of augmented
views no longer depict the labeled class, so averaging predictions on
them hurts almost by definition.* Two Phase 2C.2 results together
refute this as the dominant mechanism:

1. **Label-preservation audit** (`…_label_preservation_audit_findings.md`):
   the labeled structure is human-judged "gone" in only 0 % / 2 % / 4 %
   of augmented views (Blood / Derma / Path), and the geometric
   transforms essentially never remove it (crop keeps ≈ 78 % of area,
   ≈ 100 % of the image centre).
2. **This ablation:** the **geometric-only** family — the one the audit
   shows is content-preserving — *by itself* causes −26 pp mean harm,
   CI-separated from zero in all 12 cells.

A content-preserving perturbation that still collapses accuracy by
~26 pp is, by construction, **not a label-destruction effect**. It is
an **input-distribution-shift / model-robustness effect**: these models
were trained with no augmentation, so any augmented view — rotated,
cropped, or merely re-exposed — is off the training distribution, and
naive averaging over such views degrades accuracy regardless of whether
a human can still read the image. The intensity family produces harm of
the same order by the same mechanism.

This reframes the paper's contribution slightly and *more* defensibly:
the finding is not "TTA destroys medical images", it is "**these
models have essentially no robustness to augmentation-induced
distribution shift, geometric or photometric, and naive TTA therefore
amplifies rather than averages out that fragility**."

## 4. Limitations (disclosed)

* Validation split, not test — a mechanism decomposition, deliberately
  not a headline-number replacement. Validation-split single-family
  deltas differ slightly from the test-split mixed-policy headline
  population.
* Block A only: SmallCNN, 28 px, PathMNIST + BloodMNIST, 3 seeds. No
  ResNet-18, no DermaMNIST, no 64/128 px, no matched-policy. The 64 px
  follow-up (addendum sec. 4) was **not** run — the 28 px decomposition
  is unambiguous under sec. 7, so the gating condition for 64 px was
  not met.
* Secondary / exploratory status (post-unsealing execution), even
  though the geometric/intensity/mixed comparison itself is
  pre-registered in `docs/phase2b_protocol.md` §3.2.
* Within-family decomposition (rotation-only vs. crop-only, jitter-only
  vs. blur-only) was not performed and remains available as future
  work.

## 5. Manuscript actions

1. Add a **"Component decomposition"** paragraph to the new secondary-
   analysis subsection: the Table in sec. 1 here, the `both_contribute`
   / `sub_additive` verdict, and the 12/12 CI-exclusion statement.
2. In **Discussion**, use §3 above to close Reviewer 1 §4 explicitly:
   cite the label-preservation audit and this ablation together, and
   adopt the "no robustness to augmentation-induced distribution shift"
   framing in place of any wording that implies the augmentation
   destroys diagnostic content.
3. In **Limitations**, add the validation-split / Block-A-only /
   exploratory caveats from §4.
4. Keep the primary within-cell harm claim unchanged; this ablation
   explains *why* it happens, it does not alter *that* it happens.

## 6. Status of the Reviewer 1 §4 response

**Complete.** Both halves — label-preservation audit and
per-augmentation-component ablation — are done, both point the same way,
and the combined result converts the paper's implicit causal framing
from a vulnerable one ("TTA hurts medical image classification") to a
defensible mechanistic one ("naive TTA amplifies these models'
lack of augmentation robustness"). No further experiment is required to
answer Reviewer 1 §4; remaining reviewer items (cross-architecture
coverage, novelty) are separate and are addressed by scoping/framing,
not new runs.
