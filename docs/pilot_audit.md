# Phase 2A Scientific Audit: PathMNIST Validation Pilot

**Audit date: 2026-08-17.** This document is a correction/audit pass over
the completed pilot; it does not retrain, does not touch the test split,
and does not change the frozen pilot's configuration, thresholds,
architecture, or hyperparameters. All conclusions below are exploratory,
one-seed, validation-only evidence.

## Frozen pilot identity (unchanged, not touched by this audit)

- Dataset: PathMNIST, split: validation
- Model: SmallCNN, BatchNorm
- Seed: 314159, TTA seed: 271828
- Policy: existing mixed policy, aggregation: mean probability
- Artifact: `artifacts/pilots/pilot-pathmnist-28-bn-8f4a5024/`
- Observed clean accuracy: **0.9747**
- Observed TTA@50 accuracy: **0.4491**
- Observed delta: **−52.56 percentage points**

None of these artifacts were overwritten. This audit's own outputs live
under `artifacts/audits/pilot-pathmnist-28-bn-8f4a5024/` (separate, gitignored).

---

## A. The aborted attempt, recorded honestly

Before the completed pilot above, an earlier attempt with the **identical
frozen configuration and seeds** was run and aborted. It is recorded as an
append-only entry in a **separate** ledger, `artifacts/ledger_incidents.csv`
(not mixed into `artifacts/ledger.csv`, whose existing completed row's
column order/count must not be disturbed by a differently-shaped row — see
`src/when_tta_hurts/ledger.py::INCIDENTS_LEDGER_PATH` docstring for why).

**Incident ledger entry** (`run_id=pilot-pathmnist-28-bn-6d4e33b6`):

| Field | Value |
|---|---|
| phase / confirmatory / split | pilot / False / validation |
| seed / tta_seed | 314159 / 271828 (identical to the completed run) |
| status | **aborted** |
| training_completed | **True** — a checkpoint was written ~5 minutes into the run |
| tta_metrics_observed | **False** — killed before any TTA metric was computed |
| checkpoint_status | unavailable/deleted — the orphaned checkpoint from the killed run was deleted during cleanup before the corrected rerun and was not preserved |
| approx_runtime_seconds | ~2739s (~45.6 min) before SIGKILL |
| reason | MPS/kornia augmentation performance failure |

**What happened, and why this is an engineering failure, not a hidden
unfavorable result:** training completed normally and produced a valid
checkpoint. The process then stalled during 50-view TTA computation on the
validation split — CPU time barely advanced relative to wall-clock time
(consistent with the process being blocked, not computing). It was killed
after ~45 minutes with no completion in sight and **no TTA metric of any
kind had been observed at that point** — clean accuracy, TTA accuracy, and
harm/rescue rates were all unknown when the kill decision was made. Root
cause was diagnosed with isolated per-op timing tests immediately
afterward: kornia's `RandomResizedCrop` (~186ms/call) and
`RandomGaussianBlur` (~91ms/call) at batch=256/28px measured **~15x slower
on MPS than CPU** for this exact workload (full mixed-policy call:
~350ms/call on MPS vs. ~22ms/call on CPU). The fix — moving TTA
augmentation to CPU unconditionally, keeping only the model forward pass on
MPS — is implemented in `src/when_tta_hurts/evaluation/tta.py` and reduced
a 50-view/batch-256 TTA computation from an extrapolated tens-of-minutes to
**1.55 seconds** (measured). The corrected rerun, using the identical
frozen config and seeds, is the completed pilot recorded in
`artifacts/ledger.csv`.

---

## B. Exact augmentation manifest

**This is a paper-constrained operationalization, not an exact
reproduction** — the source paper (arXiv:2604.09697) specifies transform
families and headline ranges but not every implementation detail, and its
own code is unavailable (dead link, see `docs/data_and_licensing.md`).

| Transform | Order | p | same_on_batch | Interpolation | Fill/border | Range | Source-specified vs. our choice |
|---|---|---|---|---|---|---|---|
| Horizontal flip | 1st | 0.5 | False | n/a | n/a | — | Transform: paper. Probability: our choice. |
| Vertical flip | 2nd | 0.5 | False | n/a | n/a | — | Transform: paper. Probability: our choice. |
| Rotation | 3rd | 1.0 | False | Bilinear (`resample="BILINEAR"`) | kornia library default; no separate fill/border parameter exposed by kornia 0.8.3's `RandomRotation` | ±15° | Range: paper. Interpolation/fill: our choice, constrained by library. |
| Random resized crop | 4th | 1.0 | False | Bilinear | n/a (crop, not padding) | scale 0.8–1.0, **aspect ratio 3/4–4/3** | Scale: paper. **Aspect-ratio range NOT paper-specified** — kornia's conventional default, frozen explicitly. |
| Color jitter (brightness+contrast) | 5th (mixed/intensity only) | 1.0 | False | n/a | n/a | ±0.3 each | Range: paper. Probability: our choice. Internal brightness-vs-contrast application order is **randomized by kornia per call**, not independently configurable in kornia 0.8.3. |
| Gaussian blur | 6th (mixed/intensity only) | 0.5 | False | n/a | `border_type="reflect"` (kornia default) | kernel 3×3, sigma 0.1–2.0 | **Entirely our choice** — paper names "Gaussian blur" with no kernel/sigma/probability. |

**Composition:** all operations for a given policy are composed
**sequentially** in one `nn.Sequential`, called **once** per view via
`sample_deterministic_view()` (verified in Part C, not just asserted).

**Expected input/output range:** input to the policy is `[0,1]`
(`torchvision.transforms.ToTensor()` output, uint8→float32 scaled).
ColorJitter's actual input (post-geometric-ops) was measured at
min=0.0, max=0.974 — safely within `[0,1]` (see Part C). Output after the
full mixed policy showed min=0.0, max≈1.0000002 (float rounding at the
clamp boundary) with 1.9% of pixels at/above 1.0 and 0.3% at/below 0.0 —
consistent with expected saturation from brightness/contrast jitter and
rotation/crop fill, not a bug.

**Where augmentation runs vs. where inference runs:** augmentation runs on
**CPU unconditionally** (per the Part A fix); only the resulting augmented
batch is transferred to **MPS** for the model forward pass. This is model
*inference* placement, not a change to the augmentation policy's kind,
values, or outcome.

**Classification:**
1. **Source-paper-specified:** flip existence, rotation ±15°, crop scale
   0.8–1.0, brightness/contrast ±0.3, "Gaussian blur" as a named transform,
   geometric+intensity=mixed composition.
2. **Implementation choices forced by underspecification:** flip/rotation/
   crop/jitter/blur probabilities, interpolation mode, crop aspect-ratio
   range, blur kernel/sigma, execution order, `same_on_batch=False`.
3. **Deviations from the source paper:** none identified — every
   implementation choice fills a gap the paper leaves open, none
   contradicts a paper-stated value.

---

## C. Transformation-correctness audit

Script: `scripts/audit_pilot.py`. Uses only the existing checkpoint and
validation data. Full results: `artifacts/audits/pilot-pathmnist-28-bn-8f4a5024/transform_audit_report.json`.

| Check | Result |
|---|---|
| Raw input tensor: shape/dtype/finite | `[256,3,28,28]`, float32, all finite, range `[0.035, 0.980]` |
| Augmented (mixed) tensor: shape/dtype/finite | `[256,3,28,28]`, float32, all finite, range `[0.0, 1.0000002]` |
| Clipping frequency (mixed view) | 1.92% pixels ≥1.0, 0.33% pixels ≤0.0 |
| ColorJitter receives `[0,1]`-range input | **True** (measured pre-jitter min=0.0, max=0.974) |
| Transforms do not accidentally double-apply | **True** — `policy(policy(x))` is detectably different from `policy(x)` (proves double-apply would be caught), and the actual call site is a single `nn.Sequential.forward()` (verified by source inspection) |
| Transforms vary across views | **True** — different seeds produce detectably different views |
| Transforms vary across samples within a view | **True** |
| `same_on_batch` is correctly `False` (not accidentally identical per-batch) | **True** — augmenting 8 copies of an identical image under one view yields per-sample std 0.121 (nonzero → each sample got an independently sampled transform) |
| 50 stored views ordered correctly; N={1,2,5,10,25,50} are nested prefixes | **True** — a fresh 10-view recompute matches the first 10 of the saved 50-view sequence exactly (`allclose`, atol=1e-5) |
| Label/sample/prediction/view alignment | **True** — recomputed label order matches `predictions.npz` exactly |
| Mean-probability aggregation applies softmax per view before averaging | **True** — matches a manual per-view-softmax-then-mean computation exactly |
| **Frozen mixed-policy metrics reproduce bitwise from `predictions.npz` without rerunning inference** | **True at every view count (1,2,5,10,25,50)** — every metric field (`accuracy`, `macro_f1`, `negative_log_likelihood`, `expected_calibration_error`, `delta_accuracy`, `harm_rate`, `rescue_rate`) matches exactly |

**Contact sheets** (gitignored, not committed): `artifacts/audits/pilot-pathmnist-28-bn-8f4a5024/contact_sheets/{clean,horizontal_flip,vertical_flip,rotation,crop,color_jitter,blur,geometric_policy,intensity_policy,mixed_policy}.png` — 16 representative validation examples per category. Visual inspection confirms each named transform visibly and correctly does what it should (flips are mirrored, rotated images are rotated within ±15°, crops show a zoomed/shifted field of view, color-jittered images show visible brightness/contrast shifts, blurred images are visibly softened, and the mixed policy shows a visibly compounded combination) — no double-application, no degenerate/identical output artifacts, no obviously corrupted images.

**Conclusion of Part C: no bug found.** Every mechanical correctness check
passed, and the metrics reproduce bitwise from the saved artifact.

---

## D. Fixed diagnostic decomposition (post-hoc, exploratory only)

**Status: this section and the Part C contact-sheet inspection are POST-HOC
VALIDATION-ONLY AUDIT DIAGNOSTICS.** They were performed to investigate
correctness and mechanism *after* the frozen mixed-policy pilot result was
already observed (−52.56pp at N=50). They are **not preregistered
confirmatory experiments** — no hypothesis about which single transform
would dominate was registered before the pilot ran. They **must not be
used to tune, select, or replace the frozen pilot policy**
(`configs/pilot_pathmnist_28_bn.yaml` and
`docs/experimental_protocol.md`'s frozen augmentation table remain
unchanged by this audit) or to silently inform the confirmatory matrix
without a separate, properly pre-registered follow-up. **"Color jitter is
the dominant source of harm" is an exploratory diagnostic finding from a
single seed, not a confirmatory claim** — it has not been tested across
seeds, datasets, or resolutions, and carries none of the statistical
guarantees `docs/statistical_analysis_plan.md` requires for a confirmatory
result.

Script: `scripts/diagnostic_decomposition.py`. Uses only the existing
checkpoint, TTA seed 271828, validation split, mean-probability
aggregation, augmented-views-only. **Not a confirmatory experiment, not
policy selection** — purpose is to check whether one transform or a bug
explains the extreme degradation. Full results:
`artifacts/audits/pilot-pathmnist-28-bn-8f4a5024/diagnostic_decomposition.json`.

| Condition | N=1 acc (Δ) | N=10 acc (Δ) | N=50 acc (Δ) | N=50 harm | N=50 rescue |
|---|---|---|---|---|---|
| Clean | 0.9747 | — | — | — | — |
| Horizontal flip only | 0.9738 (−0.0009) | 0.9777 (+0.0030) | 0.9779 (+0.0032) | 0.0047 | 0.3083 |
| Vertical flip only | 0.9710 (−0.0037) | 0.9750 (+0.0003) | 0.9758 (+0.0011) | 0.0065 | 0.2925 |
| Rotation only | 0.6878 (−0.2869) | 0.7054 (−0.2693) | 0.7010 (−0.2737) | 0.2887 | 0.3043 |
| Random resized crop only | 0.7671 (−0.2076) | 0.7753 (−0.1994) | 0.7742 (−0.2005) | 0.2132 | 0.2885 |
| Color jitter only | 0.4900 (−0.4847) | 0.5920 (−0.3827) | 0.6171 (−0.3577) | 0.3720 | 0.1937 |
| Gaussian blur only | 0.7829 (−0.1918) | 0.9201 (−0.0546) | 0.9600 (−0.0147) | 0.0184 | 0.1265 |
| Geometric policy (flips+rot+crop) | 0.5245 (−0.4502) | 0.5479 (−0.4268) | 0.5491 (−0.4256) | 0.4426 | 0.2292 |
| Intensity policy (jitter+blur) | 0.4192 (−0.5555) | 0.5532 (−0.4215) | 0.5957 (−0.3790) | 0.3939 | 0.1937 |
| **Mixed policy (existing, registered)** | **0.3060 (−0.6687)** | **0.4065 (−0.5682)** | **0.4491 (−0.5256)** | **0.5432** | **0.1542** |

**Interpretation (exploratory, one seed, not confirmatory):**
- **Flips are essentially harmless** (near-zero delta, occasionally
  slightly positive) — plausible for H&E-stained histology patches, which
  have no canonical up/down or left/right orientation.
- **Color jitter is the single largest individual contributor** to harm
  (−36 to −48pp alone, depending on N), exceeding rotation (−27 to −29pp)
  and crop (−20pp) individually.
- **Gaussian blur is initially harmful (N=1: −19pp) but recovers almost
  fully by N=50 (−1.5pp)** — consistent with variance reduction from
  averaging many blur-sigma realizations.
- **Composition compounds harm roughly consistent with combining
  individual effects** (geometric ≈ rotation+crop interaction; intensity ≈
  dominated by color jitter; mixed ≈ worst of all, consistent with
  compounding covariate shift under BatchNorm).
- This pattern is **smooth, monotonic in view count for most conditions,
  and mechanistically plausible** — it does not show the signature of a
  bug (e.g., no condition collapses to exactly-random accuracy, no
  condition is inexplicably worse than a strict superset of itself, no
  NaN/degenerate values).
- **This directly corroborates the pre-registered dataset-validity risk**
  in `docs/claims_and_risks.md` (risk #6): color jitter driving the largest
  single harm is consistent with Ignatov & Malivenko (2024, arXiv:2409.11546)'s
  finding that NCT-CRC-HE-100K (PathMNIST's source) classification depends
  substantially on low-level color statistics — disrupting those statistics
  via color jitter would be expected to cause exactly this pattern.

---

## E. Independent metric validation

New tests in `tests/test_metrics_independent_validation.py` (7 tests, all
passing):

| Test | Validates against |
|---|---|
| `test_accuracy_matches_sklearn` | `sklearn.metrics.accuracy_score` |
| `test_macro_f1_matches_sklearn` | `sklearn.metrics.f1_score(average="macro")` |
| `test_nll_matches_direct_pytorch_cross_entropy` | `torch.nn.functional.cross_entropy` |
| `test_ece_manually_constructed_two_bin_example` | hand-calculated 2-bin example (expected ECE=0.1, exact match) |
| `test_harm_rescue_manually_constructed` | hand-constructed 6-sample harm/rescue scenario |
| `test_prefix_aggregation_hand_calculated` | 2-view, 2-class hand-calculated mean-probability aggregation |
| `test_checkpoint_reload_and_prediction_ordering` | regression guard for the exact bug class this audit checks for |

All 7 pass exactly, independently corroborating the self-consistency checks
already in `tests/test_metrics.py` and `tests/test_tta_evaluation.py`.

---

## F. Corrected runtime benchmark

### Chronological disclosure (do not skip)

1. **Round 1 (original Phase 2A implementation):** the preregistered
   batch-size gate for 64px used a `torch.rand(...)` **synthetic** tensor
   benchmark in `scripts/benchmark_runtime.py`. The official 64px artifact
   (`pathmnist_64.npz`) was downloaded and its checksum verified, but its
   pixel content was never fed through the model for timing.
2. **Audit round 1:** this defect was discovered during the scientific
   audit. `scripts/benchmark_runtime_real.py` was added, but its "64px"
   condition used real **28px** images resized up — still not the official
   64px artifact's actual content.
3. **Audit round 2 (this section):** `scripts/benchmark_runtime.py` was
   rewritten to be the canonical, protocol-compliant benchmark — it now
   loads `load_pilot_split("pathmnist", split="train", size=28|64)`
   directly, independently re-verifies each artifact's MD5 against
   `medmnist.INFO` before timing (failing closed on any mismatch), and
   times real forward/backward/optimizer steps on that real data.
   `scripts/benchmark_runtime_real.py`'s redundant resized-64px condition
   was removed (128px resized-proxy retained, clearly labeled). The
   synthetic-tensor benchmarking capability was **removed entirely**
   rather than retained-and-relabeled — `scripts/benchmark_runtime.py` no
   longer has a synthetic mode, so there is no standalone
   "compute-only-synthetic" script left in this project to mislabel.

**This protocol deviation affects only runtime-estimate provenance. It
does not alter, retroactively edit, or invalidate the frozen pilot's
metrics** (clean accuracy 0.9747, TTA@50 0.4491, delta −52.56pp) — the
pilot's training and TTA evaluation never depended on this benchmark's
numbers beyond the batch-size choice (256), which is unchanged by the
correction (see below). `docs/pilot_protocol.md`'s preregistration is not
edited retrospectively; this document records the correction separately.

### Canonical benchmark: complete measured table (native, official data)

Run on: **Apple M3 Pro, macOS 15.7.7, Python 3.12.2, PyTorch 2.13.0,
torchvision 0.28.0, kornia 0.8.3, medmnist 3.0.2**, device=mps (no CPU
fallback occurred; MPS was available and used throughout). 10 warmup + 30
measured steps per condition, fresh disposable model/optimizer per
condition, no checkpoint saved.

**Artifact verification (independently computed, matches `medmnist.INFO`
exactly):**

| Resolution | Artifact filename | Expected MD5 | Actual MD5 | Verified | Resized |
|---|---|---|---|---|---|
| 28px | `pathmnist.npz` | `a8b06965200029087d5bd730944a56c1` | `a8b06965200029087d5bd730944a56c1` | ✅ | **false** |
| 64px | `pathmnist_64.npz` | `55aa9c1e0525abe5a6b9d8343a507616` | `55aa9c1e0525abe5a6b9d8343a507616` | ✅ | **false** |

`pathmnist_64.npz`'s `train_images` array shape is natively `(89996, 64,
64, 3)` — confirmed not derived from the 28px array at load time.

**Timed results (all conditions: status=ok, non_finite_loss=False,
oom_or_memory_pressure=False):**

| Resolution | Batch | Samples/s | Mean step | Median step | Mem. fraction | Est. epoch | Est. 30-epoch |
|---|---|---|---|---|---|---|---|
| 28px (native) | 64 | 11,433.6 | 5.60ms | 5.58ms | 0.82% | 7.9s | 3.9min |
| 28px (native) | 128 | 12,830.1 | 9.98ms | 9.96ms | 1.20% | 7.0s | 3.5min |
| 28px (native) | 256 | 13,567.5 | 18.87ms | 18.74ms | 9.23% | 6.6s | 3.3min |
| 64px (native) | 64 | 3,005.8 | 21.29ms | 21.10ms | 9.10% | 30.0s | 15.0min |
| 64px (native) | 128 | 3,181.6 | 40.23ms | 40.18ms | 9.62% | 28.3s | 14.2min |
| 64px (native) | 256 | 3,241.2 | 78.98ms | 78.74ms | 10.40% | 27.8s | 13.9min |

**Corrected batch-size-gate outcome: batch 256 chosen for both
resolutions** (unchanged from the round-1 gate) — both pass cleanly with
large memory headroom (≤10.4% of `recommended_max_memory`).

### Superseded round-1 estimates (preserved, not deleted)

The round-1 table below used a synthetic 64px number
(3,256 samples/s, virtually identical to the now-measured native 3,241.2
samples/s — see "Corroboration" below) and unverified ×1.7/×5 multipliers
for augmentation/ResNet-18. **Marked SUPERSEDED, kept for audit trail:**

| Multiplier | SUPERSEDED value | Source |
|---|---|---|
| Matched-training augmentation overhead | ×1.7 (unverified guess, round 1) | Superseded by ×2.16 (measured, `benchmark_runtime_real.py`, unchanged by this round) |
| ResNet-18 vs. SmallCNN | ×5 (unverified guess, round 1) | Superseded by ×16.0 (measured, unchanged by this round) |
| 64px vs. 28px | synthetic-benchmark ratio (round 1) | Superseded by ×4.19 (measured, **native official data**, this round) |
| A+B+C+D total | ~7.16h (round 1, synthetic-based) → ~9.14h (audit round 1, real-28px-based) | Superseded by recomputed figure below |

### Corroboration, not correction, of the resolution-scaling ratio

The native 64px measurement (3,241.2 samples/s) is only **0.5% different**
from the earlier synthetic estimate (3,256 samples/s) and the earlier
resized-28px-proxy estimate (3,259 samples/s). **The 64px scaling ratio was
never actually wrong** — the defect was that it had never been verified
against real, official 64px content until now. This is now independently
confirmed with real data: **×4.19 (28px→64px slowdown, native, measured)**,
consistent with ×4.22–4.23 from the earlier (uncorrected-provenance)
estimates.

### Recomputed block estimates (native-data-based)

Using the pilot's actual measured 28px rate (304.147s / 30 epochs =
10.14s/epoch, ground truth, unchanged) and the multipliers above:

| Block | Runs | Recomputed estimate | vs. previous (superseded) |
|---|---|---|---|
| A (24 runs) | | ~2.98h | ~3.00h (−0.02h, noise-level change) |
| B (6 runs, matched-policy) | | ~0.62h | unchanged (×2.16 multiplier unaffected by this round) |
| C (3 runs, ResNet-18) | | ~0.32h | unchanged (×16.0 multiplier unaffected by this round) |
| D (6 runs, 128px, **proxy/extrapolation — no official 128px artifact exists or was measured**) | | ~5.20h | unchanged (still resized-proxy-based, not natively measured) |
| **A+B+C** | | **~3.92h** | ~3.94h |
| **A+B+C+D** | | **~9.12h** | ~9.14h |

**Runtime gate still PASSES**: ~9.12h for the full matrix (including the
proxy-based block D) is comfortably under the ~24h target. The change from
round 1's audit estimate is negligible (~0.02h) — the correction's value is
in *provenance* (now genuinely measured from official native data, not a
synthetic/resized substitute), not in materially changing the numbers.

**128px remains proxy/extrapolation-only** — no official 128px artifact
has been downloaded, checksum-verified, or measured anywhere in this
project. Block D's estimate should be treated as lower-confidence than
blocks A/B/C until a native 128px benchmark is performed.

### Unresolved short-benchmark/full-pilot gap — recomputed with corrected data

The gap is **still unresolved, and now confirmed with fully corrected,
native-data provenance on both resolutions**: the pilot's actual per-epoch
time (10.14s) is **1.527x slower** than what ANY short benchmark predicts
(native 28px: 352 steps × 18.87ms = 6.64s/epoch-equivalent). Since this
gap now persists even with genuine native 64px data (not just 28px), it is
conclusively **not** an artifact of synthetic-vs-real or resized-vs-native
data — something about the full 30-epoch sustained run (candidate causes:
per-epoch validation-pass cost inside the training loop, thermal/memory
effects over a longer run) remains unexplained. Applying this 1.527x factor
on top of the corrected A+B+C+D estimate: **~9.12h × 1.527 ≈ 13.93h — still
under the 24h gate**, but this remains an open item, not resolved here.

---

## G. Scientific interpretation

- **This is one-seed exploratory validation evidence.** Seed 314159 is
  permanently excluded from the confirmatory matrix (per
  `docs/pilot_protocol.md`).
- **It is not a direct numerical reproduction of the source paper.** This
  is a paper-constrained reimplementation (Part B) with different
  architecture details, no train-time augmentation exposure in the base
  pilot, and a self-composed 6-op mixed policy; the −52.56pp result is
  substantially larger than the source paper's reported −28.6pp for the
  nominal SmallCNN/PathMNIST cell, and no attempt was made to close that
  gap.
- **The official test split remains untouched.** No test-split loading
  mechanism exists in any script used in this audit (`load_pilot_split`
  only accepts `train`/`val`; the audit and diagnostic scripts reuse it
  exclusively).
- **The −52.56-point result is not confirmatory.** It is one seed,
  validation-only, and the diagnostic decomposition (Part D) shows it is
  mechanistically explainable (color jitter as the dominant driver,
  compounded by geometric distortion) rather than evidence of a
  measurement bug (Part C found none).
- **No post-result configuration changes were made.** The frozen pilot
  config (`configs/pilot_pathmnist_28_bn.yaml`), the augmentation policy,
  and the model architecture are unchanged from before this audit began.

## Unresolved limitations

1. The ~1.54x gap between the actual 30-epoch pilot runtime and ANY
   short-benchmark estimate (synthetic or real-data, both ≈13,700
   samples/s) is not explained by this audit. Candidate causes not yet
   isolated: per-epoch validation-pass cost inside the training loop,
   longer-run thermal/memory effects, or Python-level overhead that only
   compounds over many more steps than a 40-step microbenchmark exercises.
   Should be resolved with a full-epoch (not 40-step) real-data benchmark
   before Phase 2B schedules are finalized.
2. Patient/lesion-level leakage questions from `docs/data_and_licensing.md`
   remain open and are unrelated to, and unresolved by, this audit.
3. The diagnostic decomposition (Part D) is exploratory and single-seed;
   it should not be used to select or tune the confirmatory augmentation
   policy without a properly pre-registered follow-up.
