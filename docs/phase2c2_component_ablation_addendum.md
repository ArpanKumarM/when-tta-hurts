# Phase 2C.2 — Per-Augmentation-Component Ablation — FROZEN ADDENDUM

**Status: pre-registered analysis addendum. Frozen on commit of this
file. No component-ablation view has been generated and no result
exists.** This is the second half of the response to
`paper/reviews/reviewer_1_scientific_soundness.md` §4 (the first half is
`docs/phase2c2_label_preservation_audit_findings.md`).

It does **not** modify `configs/experiment_matrix.yaml`, the frozen
augmentation table in `docs/experimental_protocol.md`, the confirmatory
endpoints, the sealed final-test artifacts, or `paper/manuscript.md`.
It requires **no training** and **no test-split access**.

---

## 1. Question

Of the naive-TTA harm produced by the frozen **mixed** policy, how much
is carried by the **geometric** transform family (flips, rotation ±15°,
random resized crop 0.8–1.0×) versus the **intensity** family
(brightness/contrast jitter ±0.3, Gaussian blur)?

The label-preservation audit (Phase 2C.2) established that the geometric
transforms almost never remove the labeled content (crop keeps ~78 % of
area, ≈100 % of the image centre) while the intensity shifts are large
but human-survivable. This ablation measures which family the *accuracy*
harm actually tracks.

## 2. Evidentiary status

`docs/phase2b_protocol.md` §3.2 pre-registers exactly this comparison —
"**Augmentation-strategy ablation** — geometric, intensity, mixed; mean
probability; N=25. Preregistered secondary analysis." The `geometric`
and `intensity` policies are already named and frozen in
`configs/experiment_matrix.yaml` (§ augmentation policies) and
implemented in `src/when_tta_hurts/transforms/policies.py::build_policy`.
Nothing new is invented here.

Because this addendum is executed **after** the final-test unsealing, it
is reported as **secondary / exploratory** (matching the Phase 2C
framing), not as preregistered-confirmatory evidence, even though the
comparison itself was preregistered. All cross-family statements are
descriptive.

## 3. Split — validation only

The ablation runs on the **validation** split, never the test split.
Rationale: (a) it is a mechanism decomposition, not a headline number;
(b) the mixed-policy validation harm is already computed for every cell
below (`artifacts/validation_evaluation/<run_id>/…/predictions.npz`
`view_probs`), so geometric-only and intensity-only rows drop in as
directly comparable; (c) it keeps the test firewall
(`docs/experimental_protocol.md`) intact and needs no final-test
authorization. The tradeoff — validation-split deltas differ slightly
from the test-split headline population — is disclosed wherever the
result is reported.

## 4. Cells — FROZEN

The **12 Block A 28 px** confirmatory cells (primary), using the
already-trained checkpoints under
`artifacts/confirmatory/A/<run_id>/attempt_*/best_checkpoint.pt`
(canonical attempt = the checkpoint whose `hash_state_dict` matches the
`checkpoint_hash` recorded in that cell's canonical validation-
evaluation `metadata.json`):

* datasets: PathMNIST, BloodMNIST
* resolution: 28 px
* normalization: BatchNorm, GroupNorm
* seeds: 0, 1, 2

The 12 Block A **64 px** cells are an **optional follow-up** (same
procedure, same script) if the 28 px decomposition is inconclusive
under sec. 7; running them is not required for the addendum's primary
finding and is gated on the 28 px result.

Block B (matched-policy), Block C (ResNet-18 / DermaMNIST), and Block D
(128 px) are **out of scope** — the single-architecture and
resolution-coverage limitations already stated in the manuscript are
unchanged by this addendum.

## 5. Procedure — FROZEN

For each of the 12 cells, for each policy `p ∈ {geometric, intensity}`:

1. Load the canonical Block A checkpoint; put the model in eval mode.
2. Generate a deterministic 50-view sequence over the **full
   validation split** using
   `stable_view_seed(confirmatory_tta_seed, dataset, resolution,
   sample_index, view_index)` — the identical seed function and
   `confirmatory_tta_seed = 1306178015` used for the mixed-policy
   confirmatory views — with `build_policy(p, (res, res))`.
3. Aggregate by **mean predicted probability** over the first **N=25**
   views (the §3.2 pre-registered count); also record **N=50** for
   comparability with the manuscript's headline condition.
4. Compute, per cell and per N:
   * `delta_accuracy` = policy-TTA accuracy − clean accuracy (clean is
     read from the existing validation predictions; identity-checked),
   * paired-bootstrap 95 % CI on the delta (10,000 resamples, frozen
     `paired_bootstrap_ci`, deterministic per-(cell, policy, N) seed),
   * McNemar p (frozen `mcnemar_test`), harm/rescue rates
     (`effect_sizes`),
   * ECE / NLL / multiclass-Brier of the aggregated probabilities.

The mixed-policy delta for the same cell/N is taken from the existing
validation predictions (no recomputation).

## 6. Reported quantities — FROZEN

Per cell and per N: `geometric_delta`, `intensity_delta`,
`mixed_delta`, each with 95 % CI; plus the **additivity residual**
`r = mixed_delta − (geometric_delta + intensity_delta)` (sign and
magnitude only — descriptive, no CI).

Pooled descriptive summaries (mean, min, max across the 12 cells, and
split by dataset / resolution / normalization): the three deltas and
the residual.

## 7. Pre-registered interpretation rule — FROZEN

Applied to the 12-cell means, per N:

* **Intensity-dominated.** If `|mean intensity_delta| ≥ 2 ×
  |mean geometric_delta|` **and** `|mean geometric_delta| < 10 pp`:
  report that the harm is carried predominantly by the intensity
  family. Combined with the label-preservation finding (intensity
  shifts are large but content-preserving), this supports framing the
  harm as a **model-robustness / input-distribution effect**, not a
  label-destruction effect.
* **Geometry-dominated.** If `|mean geometric_delta| ≥ 2 ×
  |mean intensity_delta|`: report that the harm is carried
  predominantly by geometric transforms. Because the label audit shows
  geometry preserves content, this would be framed as
  geometry-induced input-distribution shift without label loss.
* **Both contribute.** Otherwise: report both families as material
  contributors, with the additivity residual `r` characterising
  whether the mixed harm is approximately additive (`|mean r| < 5 pp`),
  super-additive (`mean r` more negative), or sub-additive.

All 12-cell deltas and the residual are reported verbatim regardless of
which branch fires. The 2×, 10 pp, and 5 pp cutoffs are fixed here
before any component-ablation view is generated.

## 8. Outputs

```
artifacts/component_ablation/
  summary.json     # per-cell + pooled deltas, residuals, sec.7 verdict
  per_cell.csv     # 12 cells x {geometric, intensity} x {N=25, N=50}
  manifest.json    # sha256 of this addendum, each checkpoint, each
                   # validation predictions.npz consumed, confirmatory_tta_seed
```

Produced by `scripts/component_ablation.py`, which imports the frozen
`build_policy`, `stable_view_seed`, aggregation, `metrics`, and
`statistical_analysis` primitives and writes only to the new directory.
It does not modify `src/when_tta_hurts/validation_evaluation.py` or any
sealed artifact.

**Resumable.** After each (cell, policy) pair finishes, its aggregated
probability snapshots are written to
`artifacts/component_ablation/_cache/<run_id>__<policy>.npz` (tagged
with `confirmatory_tta_seed`, `max_views`, and sample count; a mismatch
invalidates the entry). Re-running the script skips any pair whose
cache is present and valid, so an interruption costs at most one
in-flight pair (~10–15 min for a PathMNIST cell, ~3 min for
BloodMNIST). The cheap statistics / CSV / summary are always recomputed
from the caches at the end. Deleting `_cache/` forces a full recompute.

## 9. Compute

Evaluation only, no training. 12 checkpoints × 2 policies × 50-view
generation over the 28 px validation split (10,004 PathMNIST / 1,712
BloodMNIST samples). Per-sample CPU view synthesis
(`evaluation/views.py::generate_single_view`, the frozen call site) is
the bottleneck; rough estimate **~2–4 h** on the Apple M3 Pro. Datasets
are already cached in `data/raw/`. The optional 64 px follow-up is a
similar magnitude again.

## 10. Non-goals / disclosed limitations

* Validation split, not test — a mechanism decomposition, not a
  headline replacement.
* Block A only (SmallCNN, 28/64 px, PathMNIST/BloodMNIST). No
  ResNet-18, no DermaMNIST, no 128 px, no matched-policy.
* Secondary / exploratory status (post-unsealing execution).
* `geometric` and `intensity` are the frozen policy definitions; this
  addendum does not further decompose within a family (e.g.
  rotation-only vs. crop-only) — that remains available as future work.
