# Compute Budget

**Status: baseline matrix (blocks A/B/C) APPROVED as of Phase 2B.1 — see
`docs/phase2b_protocol.md`. 28px/64px runtime numbers are no longer
placeholders (see "Actual hardware" and `docs/pilot_audit.md`'s corrected
canonical benchmark); 128px numbers remain unmeasured pending Block D's own
gate (`docs/phase2b_protocol.md` sec.6). Matrix corrected against verified
source-paper facts — see `docs/literature_review.md` and
`configs/experiment_matrix.yaml`.**

## Hardware

**Primary hardware (corrected — see "Actual hardware" below for the full
disclosure): Apple M3 Pro, 18 GB unified memory.** The original Phase 0
hard constraint ("Apple Silicon M4, 16 GB unified memory") is preserved
below as a documented historical planning assumption, not deleted — it was
superseded once the actual machine was measured in Phase 1, and the M3 Pro
has been the real target for all Phase 1/2A work and all Phase 2B planning
since.

- **Actual/primary: Apple M3 Pro, 18 GB unified memory** (measured via
  `sysctl`/`sw_vers`; see "Actual hardware" section below).
- *Historical planning assumption (Phase 0, superseded): Apple Silicon M4,
  16 GB unified memory.*
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

Exact evaluation-job *counts* (how many inference passes) will be computed
and logged once real runs exist; they are not estimated here. Their
**storage footprint**, however, is estimated below since it must be
budgeted before any caching begins.

## Cache storage estimate (corrected)

Formula: `samples x views x classes x 4 bytes` (float32 logits). Only
logits are cached; probabilities/aggregates (mean, majority vote,
confidence-weighted) and every view-count prefix (1/2/5/10/25/50/100) are
derived from the same cached 100-view logit tensor, never cached
separately — see `src/when_tta_hurts/evaluation/cache.py`.

For validation+test at 100 views, per checkpoint/policy:

| Dataset | (val+test) samples | Formula | Estimate |
|---|---|---|---|
| PathMNIST | 10,004 + 7,180 = 17,184 | 17,184 × 100 × 9 × 4 bytes | ≈ 59 MiB |
| BloodMNIST | 1,712 + 3,421 = 5,133 | 5,133 × 100 × 8 × 4 bytes | ≈ 15.7 MiB |
| DermaMNIST | 1,003 + 2,005 = 3,008 | 3,008 × 100 × 7 × 4 bytes | ≈ 8 MiB |

Before conditional block D, the matrix contains **15 PathMNIST checkpoints,
15 BloodMNIST checkpoints, and 3 DermaMNIST checkpoints** (blocks A+B: 12
PathMNIST + 3 matched = 15; 12 BloodMNIST + 3 matched = 15; block C: 3
DermaMNIST) requiring a cache:

- **One policy:** 15×59 + 15×15.7 + 3×8 MiB ≈ **1.1 GiB**
- **Three policies** (geometric/intensity/mixed): ≈ **3.3 GiB**

With conditional block D (6 more PathMNIST+BloodMNIST checkpoints at
128px — logit storage doesn't depend on input resolution, only on
sample/view/class counts): ≈ **4 GiB for three policies**.

This is logits only, **before**: checkpoints themselves, labels/metadata,
temporary files, BN-adapted inference (not cacheable this way — it mutates
model state, must be tracked as separate inference per
`evaluation/cache.py`), and additional validation artifacts.

**Practical working-storage allowance: approximately 5-8 GB**, not the
~1-4GB logit-only figure alone. This must be checked against available
disk space before any real caching begins in Phase 2+.

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

## Kill criteria — CORRECTED, now frozen for 128px (blocks A/B/C need no kill criterion: measured, see below)

**The "TBD" that previously appeared here was never filled in during Phase
1/2A, even after real timing data existed — this was a documentation gap
identified and closed in the Phase 2B preflight audit.** It is now
resolved as follows:

- **28px/64px (blocks A/B/C):** no separate kill criterion is needed —
  these are now **measured**, not estimated: `docs/pilot_audit.md`'s
  corrected canonical benchmark shows ~3.3min/run (28px) and ~13.9-15.0min/run
  (64px) at 30 epochs, batch 256, well within any reasonable ceiling.
- **128px (block D):** the numeric trigger is now frozen in
  `docs/phase2b_protocol.md` sec.6 — **≤90 minutes per 128px training run,
  ≤120 minutes per 128px cell end-to-end, pessimistic A+B+C+D total <24
  hours** — checked before any 128px training occurs, using a native
  (not resized-proxy) 128px benchmark.
- Max memory: 18 GB hard ceiling (actual machine, corrected from the
  16 GB historical planning assumption — see "Hardware" above); a
  practical working ceiling below that (leaving headroom for the OS) is
  informed by the measured memory fractions in `docs/pilot_audit.md`
  (all confirmed ≤10.4% of `recommended_max_memory` at 28px/64px,
  batch 256).
- If MPS operations are unsupported or numerically unstable for a given op
  (a known occasional issue with newer PyTorch ops on MPS — and confirmed
  in practice for kornia's `RandomResizedCrop`/`RandomGaussianBlur`, whose
  MPS performance pathology was found and fixed during the Phase 2A audit,
  see `docs/pilot_audit.md`), fall back to CPU for the affected operation
  only, and document which ops required the fallback and any observed
  runtime cost — do not fall back silently.

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
flagged in the prior version of this document. 28px/64px runtime/memory are
now **measured** (Phase 1/2A complete — see "Actual hardware" below and
`docs/pilot_audit.md`); 128px remains unmeasured pending Block D's own gate
(`docs/phase2b_protocol.md` sec.6), which has not yet been evaluated (no
128px artifact has been downloaded for any dataset).

## Actual hardware (measured in Phase 1) — deviation from the stated constraint

The project's hard constraint specifies "Apple Silicon M4, 16 GB unified
memory." The machine this Phase 1 work actually ran on, per
`sysctl`/`sw_vers`, is an **Apple M3 Pro with 18 GB unified memory**
(macOS 15.7.7). This is recorded here rather than silently substituted:
the M3 Pro is a different (though architecturally similar) chip with 2GB
more memory than specified. All device/memory reasoning in this document
should be read as applying to the actual machine, and the environment
manifest (`results/runs/<run_id>/env_manifest.json`, captured by
`when_tta_hurts.devices.capture_environment`) records the real chip/memory
for every run so this is auditable, not assumed. If the discrepancy
matters (e.g. the user intends to later run confirmatory experiments on an
actual M4/16GB machine), runtime/memory kill criteria measured here may not
transfer exactly and should be re-measured on the target hardware.

## Software stack and version rationale

- **Python 3.12** (`.python-version`, `pyproject.toml`): chosen as the
  `uv init` default and because it is a mature, broadly-supported version
  for the current stable PyTorch release with full MPS wheel availability,
  while still being new enough to have good typing/performance
  improvements over 3.10/3.11. Not 3.13 because ecosystem package wheel
  coverage (e.g. scikit-image, kornia) is more consistently available for
  3.12 at time of writing.
- **PyTorch 2.13.0 / torchvision 0.28.0** (resolved and locked by `uv lock`
  against the environment above): the latest stable release available at
  resolution time with confirmed MPS support (`torch.backends.mps.is_built()
  == True`, `is_available() == True` on this machine — see Phase 1
  completion report). Exact versions are pinned in `uv.lock`, not just
  `pyproject.toml`'s lower bounds, so the resolved environment is
  reproducible.
- All other dependency versions are recorded in `uv.lock` (committed
  nowhere yet, per Phase 1 scope — see CLAUDE.md) and reported in the Phase
  1 completion report.
