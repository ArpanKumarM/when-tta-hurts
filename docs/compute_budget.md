# Compute Budget

**Status: draft, Phase 0. Runtime numbers are unmeasured placeholders — no
runs have been executed on the target machine yet. Matrix corrected against
verified source-paper facts — see `docs/literature_review.md` and
`configs/experiment_matrix.yaml`.**

## Hardware

- Apple Silicon M4, 16 GB unified memory.
- PyTorch with the MPS backend. No NVIDIA GPU, no CUDA.
- No cloud training in this phase; not approved for use without a separate
  request.

## Target run count

Approximately **30–40 total final (confirmatory, 3-seed) training runs**,
counting only unique trained checkpoints. TTA/evaluation jobs are a
separate, larger count — see "Evaluation job accounting" below. This
distinction (training runs vs. evaluation jobs) was not made explicit in
the prior draft and is corrected here.

## Corrected training matrix (see `configs/experiment_matrix.yaml`)

| Block | Description | Training runs |
|---|---|---|
| A — core normalization × resolution | PathMNIST + BloodMNIST × {28,64}px × {BatchNorm, GroupNorm} × 3 seeds | 24 |
| B — policy matching | PathMNIST + BloodMNIST, 28px, BatchNorm, matched-policy training × 3 seeds (unmatched arm reuses A's checkpoints) | 6 |
| C — positive-control reproduction | DermaMNIST, ResNet-18, 28px, BatchNorm × 3 seeds (reproduces the source paper's sole positive case: ResNet-18/DermaMNIST +1.6pp) | 3 |
| D — conditional 128px tier | PathMNIST + BloodMNIST, 128px, BatchNorm × 3 seeds | 6 (conditional) |
| **Total** | | **33 before D, 39 with D** |

`no_running_stats` has been removed from the confirmatory matrix (it may
return later only as an explicitly-labeled exploratory arm, per
`docs/statistical_analysis_plan.md`'s confirmatory/exploratory split).

## Evaluation job accounting (separate from training runs)

Every trained checkpoint from A/B/C(/D) is evaluated under multiple TTA
conditions, each swept over the source paper's 7 view counts
(1/2/5/10/25/50/100) and, where relevant, 3 aggregation methods
(mean / majority vote / confidence-weighted). This is evaluation-time
compute reusing existing checkpoints, not additional training — but it is
not negligible and must be tracked separately so it isn't hidden inside the
"30-40 runs" framing:

- **clean**: 1 job per checkpoint (no view sweep).
- **naive_tta**: up to 7 view counts × up to 3 aggregation methods per
  checkpoint — primary confirmatory number is fixed at N=50, mean
  aggregation (reproducing the source paper's headline condition); the
  full view/aggregation sweep is exploratory unless promoted.
- **original_anchored_tta**, **bn_adapted_tta**: reproductions of the
  source paper's Appendix B conditions, evaluated at the same primary view
  count (N=50) as required baselines for H4 — not swept over all 7 view
  counts by default, to control cost.
- **matched_tta**: evaluated on B's checkpoints only, at N=50 primary.
- **validation_gated_tta**: evaluated on all A/B/C checkpoints; its
  thresholds are calibrated on validation data (Phase 1/2), then a single
  frozen configuration is run once against test data per the firewall in
  `docs/experimental_protocol.md`.

Exact evaluation-job totals will be computed and logged in Phase 1 once the
smoke test confirms per-job runtime; they are not estimated here to avoid
presenting an unmeasured number as a commitment.

## Staging

1. **Smoke test** (Phase 1, first step): tiny synthetic or single-batch run
   to confirm the pipeline executes end-to-end on MPS without downloading
   full datasets or committing to a runtime estimate.
2. **Pilot measurement:** run a small subset of block A to measure actual
   wall-clock/memory usage per training run and per evaluation job on this
   machine, used only to set the kill criteria below — not a separate
   reduced-scope confirmatory matrix (the corrected matrix above is already
   pre-registered in full, unlike the prior draft's pilot-then-promote
   design).
3. **Confirmatory runs:** blocks A, B, C (and D if its gate passes), per
   the test firewall in `docs/experimental_protocol.md`.

## Resolution scope

- Confirmatory tiers: **28×28 and 64×64** (block A/B/C).
- **128×128 (block D) is conditional**: only executed if pilot-measured
  64×64 runtime for PathMNIST/BloodMNIST projects to a 128×128 run
  finishing within the per-run kill criterion (measured, not assumed).
  MedMNIST+ 64/128px images are resized from independently
  higher-resolution source images, not upsampled from 28px files (see
  `docs/data_and_licensing.md`), so this tier evaluates a genuine
  resolution/information effect, not an interpolation artifact.
- **224×224 is out of scope** for this initial study regardless of measured
  runtime, per the hard constraint on scope.

## Kill criteria (to be filled with measured numbers in Phase 1)

- Max wall-clock minutes per individual training run: **TBD**, to be set
  after the smoke test and pilot measurement, then written here and into
  `configs/experiment_matrix.yaml`.
- Max memory: 16 GB hard ceiling (physical machine limit); a practical
  working ceiling below that (e.g. leaving headroom for the OS) will be set
  after Phase 1 measurement.
- If MPS operations are unsupported or numerically unstable for a given op
  (a known occasional issue with newer PyTorch ops on MPS), fall back to CPU
  for the affected operation only, and document which ops required the
  fallback and any observed runtime cost.

## Fallback scope if the target budget can't be met

If pilot measurements show the corrected matrix (33-39 runs) cannot fit
within the 6-7 day budget on this hardware:

1. First cut: drop block D (128px tier) entirely — it is already
   conditional, so this is the lowest-cost cut.
2. Second cut: reduce block B (policy matching) to one dataset instead of
   two, noting the reduced H3 scope explicitly.
3. Last resort: reduce confirmatory seeds from 3 to 2 for block A only,
   clearly noting the reduced statistical power this implies.

Any of these cuts must be applied to the matrix in
`configs/experiment_matrix.yaml` and explicitly logged before it is marked
`status: approved` — not decided silently mid-run.

## Status

The corrected matrix (33 runs before the conditional tier, 39 with it) fits
within the 30-40 run target as scoped. This resolves the "Open problem"
flagged in the prior version of this document. Remaining unknowns are
runtime/memory measurements themselves, which require Phase 1 (not yet
approved) to obtain.
