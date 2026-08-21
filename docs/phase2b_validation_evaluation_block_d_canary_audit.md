# Phase 2B Block D Canary: PathMNIST 128px BatchNorm, seed 0

**Recorded: 2026-08-21.** This document records the first Block D
(native 128px resolution) validation-evaluation completion:
`D-pathmnist-128px-batchnorm-policy-none-s0`. This is a **single-seed validation
canary**, not a confirmed result. No protocol, threshold, policy, seed,
prefix, aggregation, batching, evaluator, or metric code was changed as
a result of this observation. **No test split was accessed at any
point.**

## 1. Exact identity and provenance

| Field | Value |
|---|---|
| Training run | `D-pathmnist-128px-batchnorm-policy-none-s0` |
| Canonical training attempt / checkpoint hash | `1` / `276169a842c64261764c2499d520a87b72014746f9d8289dd1a3878b01d25966` |
| Evaluation attempt | `1` |
| Evaluation ID / evaluation-config hash | `b8dab819ccf0ca0f42ad24d01422dd801c21c05e7e405ed4ece51eb73f2360de` |
| Evaluator fingerprint | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` (matches current: True) |
| Dataset / resolution / split | `pathmnist` / 128 / `validation`, n_validation_samples=10004 |
| Normalization | `batchnorm` |
| Frozen TTA seed | `1306178015` |
| Metric-input contract | `probability_native_v1` |
| Source commit | `7391cb0c8b69f7489f93698c9abd2c8b380ada56` |
| Protocol commit | `ce4c962` |

### Artifact manifest

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `predictions.npz` | 39057252 | `aee175ee3bd3421d3de1521d49d4c5a1b5da58ff0a5c19be5476992429728908` |
| `metrics.json` | 16306 | `896d2c6dc346c90b04d4d2e67c416e964f9221be2c4036ef1327367e522279d6` |
| `metadata.json` | 4701 | `25d5616815dd6d95fa323327ead2dbf9293d070be2fc37116cc5fec8623ce2e2` |
| `view_manifest.json` | 99518 | `dfc5ec86f81191343fba993fc6f38bad4913a2768156af89ff552d8c4e0978fc` |

## 2. Runtime and latency

- Evaluation runtime: `10717.256802797318` s (~2.98h)
  -- landed between the two nonbinding pre-run estimates (linear-trend
  ~2.2h; pixel-count-scaling upper bound ~8.2h), closer to the
  optimistic end.
- Clean latency: 2.326366 s

| N | Compute multiplier |
|---|---|
| 1 | 0.9937 |
| 2 | 2.0142 |
| 5 | 4.9534 |
| 10 | 9.9594 |
| 25 | 24.7802 |
| 50 | 50.4771 |
| 100 | 104.1123 |

**No peak-memory field is persisted in `metadata.json` or `metrics.json`**
for this evaluation pipeline -- none is reported here, and none is
estimated or fabricated.

## 3. Checksum and bounded-batching proof

| Field | Value |
|---|---|
| Expected checksum (MD5) | `ac42d08fb904d92c244187169d1fd1d9` |
| Actual checksum (MD5) | `ac42d08fb904d92c244187169d1fd1d9` |
| Match | True |
| `resized` | `False` |
| Artifact path | `data/raw/pathmnist_128.npz` |
| Verification method | `dataset_verification.verify_official_dataset_artifact` |

| Batching field | Value |
|---|---|
| `inference_batch_size` | `256` |
| `bn_adaptation_batch_size` | `256` |
| `bn_adaptation_algorithm` | `sequential_microbatch_v1` |
| `bn_adaptation_applicable` | `True` |
| `bn_adaptation_microbatches_at_primary_n` | `2000` |

These batch-size fields are identical to every other resolution
evaluated in this project (28px/64px) -- confirming bounded, resolution-
independent batching, as designed (the batching mechanism bounds peak
memory to one microbatch of images regardless of native resolution;
only the per-image tensor size grows with resolution, not the batch
count).

## 4. Independent verification evidence (all passed)

- **Manifest**: `verify_evaluation_artifact_manifest()` -- OK.
- **Full semantic recomputation**: clean metrics + all 7 registered
  prefixes x 3 `naive_tta` aggregators, recomputed independently via
  `compute_metrics_from_probabilities()`/`_recompute_all_conditions_from_predictions()`
  -- **0 mismatches** against `metrics.json`, within the frozen `1e-6` tolerance.
- **Probability validity**:
  - `clean_probs`: finite=True, in [0,1]=True, row-normalized=True, n_classes=9
  - `view_probs`: finite=True, in [0,1]=True, row-normalized=True, n_classes=9
  - `bn_adapted_probs`: finite=True, in [0,1]=True, row-normalized=True, n_classes=9
- **Sample/label alignment**: `sample_indices` unique=True, contiguous (0..10003)=True, length matches labels=True.
- **Checkpoint binding**: `metadata.checkpoint_hash` matches
  `resolve_canonical_training_completion()`'s canonical training result exactly (True).
- **Effective training configuration**: training attempt `1`, checkpoint
  `276169a842c64261764c2499d520a87b72014746f9d8289dd1a3878b01d25966`, confirmed unambiguous and canonical via the production
  `resolve_canonical_training_completion()` selection logic.
- **Evaluator fingerprint**: matches current fingerprint exactly (True).
- **Sole canonical-compatible completion**: exactly 1 ledger row for this
  `training_run_id`, `status=completed`, `evaluation_attempt=1`.
- **`test_metrics_observed=False`** on the ledger row.
- **`confirmatory=True`**, `split=validation` on the ledger row.
- **No Block A/B/C interference**: Block A remains 24, Block B remains 6,
  Block C remains 3, confirmed mechanically.
- **Test split untouched**: no test-split access occurred at any point
  during this evaluation.

## 5. Clean metrics

| Metric | Value |
|---|---|
| accuracy | 0.809476 |
| macro_f1 | 0.791317 |
| negative_log_likelihood | 0.544360 |
| expected_calibration_error | 0.026443 |
| brier_score | 0.269084 |

## 6. Complete N=1,2,5,10,25,50,100 curves -- all conditions

### Naive mean-probability

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.335166 | -47.43pp | 0.320214 | 4.409486 | 0.446277 | 1.057157 | 0.623117 | 0.157922 |
| 2 | 0.353159 | -45.63pp | 0.326046 | 2.683034 | 0.237589 | 0.861190 | 0.596691 | 0.140084 |
| 5 | 0.377449 | -43.20pp | 0.339704 | 1.717073 | 0.120005 | 0.743405 | 0.567177 | 0.142183 |
| 10 | 0.401040 | -40.84pp | 0.352750 | 1.513733 | 0.075445 | 0.702727 | 0.536182 | 0.134313 |
| 25 | 0.416333 | -39.31pp | 0.354213 | 1.437935 | 0.057280 | 0.680425 | 0.512225 | 0.112802 |
| 50 | 0.429128 | -38.03pp | 0.358252 | 1.415067 | 0.070879 | 0.671673 | 0.496048 | 0.111228 |
| 100 | 0.437425 | -37.21pp | 0.359274 | 1.404171 | 0.089260 | 0.667002 | 0.486540 | 0.114376 |

### Majority-vote

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.335166 | -47.43pp | 18.370044 |
| 2 | 0.329768 | -47.97pp | 13.950218 |
| 5 | 0.367753 | -44.17pp | 7.968361 |
| 10 | 0.389144 | -42.03pp | 4.791819 |
| 25 | 0.407537 | -40.19pp | 2.780026 |
| 50 | 0.422131 | -38.73pp | 2.181699 |
| 100 | 0.425430 | -38.40pp | 1.919582 |

### Confidence-weighted

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.335166 | -47.43pp | 4.409486 |
| 2 | 0.352259 | -45.72pp | 2.724136 |
| 5 | 0.377749 | -43.17pp | 1.759667 |
| 10 | 0.396441 | -41.30pp | 1.548549 |
| 25 | 0.409636 | -39.98pp | 1.464912 |
| 50 | 0.421731 | -38.77pp | 1.439500 |
| 100 | 0.430628 | -37.88pp | 1.427607 |

### Original-anchored

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.661935 | -14.75pp | 0.899266 |
| 2 | 0.606657 | -20.28pp | 1.047019 |
| 5 | 0.518892 | -29.06pp | 1.214201 |
| 10 | 0.487005 | -32.25pp | 1.291832 |
| 25 | 0.458517 | -35.10pp | 1.351211 |
| 50 | 0.451519 | -35.80pp | 1.371616 |
| 100 | 0.447821 | -36.17pp | 1.382368 |

### BN-adapted

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.843263 | +3.38pp | 0.507697 | 0.083759 | 0.240471 | 0.076933 | 0.504197 |
| 2 | 0.769792 | -3.97pp | 0.676834 | 0.049604 | 0.333532 | 0.149667 | 0.427597 |
| 5 | 0.806477 | -0.30pp | 0.599897 | 0.071743 | 0.291232 | 0.110645 | 0.454355 |
| 10 | 0.809876 | +0.04pp | 0.588739 | 0.070588 | 0.285817 | 0.105582 | 0.450682 |
| 25 | 0.827469 | +1.80pp | 0.549261 | 0.079122 | 0.263408 | 0.091504 | 0.483211 |
| 50 | 0.768493 | -4.10pp | 0.675594 | 0.050293 | 0.335798 | 0.150161 | 0.422875 |
| 100 | 0.816873 | +0.74pp | 0.576782 | 0.076631 | 0.279958 | 0.100148 | 0.464323 |

BN-adaptation reduces harm substantially relative to naive TTA at every
N, but behaves **non-monotonically** across N (oscillating between
mildly positive and mildly negative deltas) rather than monotonically
improving with more views -- unlike the naive condition's monotonic
(if incomplete) recovery.

## 7. Same-seed (seed 0) descriptive resolution comparison: PathMNIST, BatchNorm, policy=none

| Resolution | Run ID (attempt) | Clean accuracy | N=50 naive delta | Runtime (s) | Fingerprint-compatible |
|---|---|---|---|---|---|
| 28px | `A-pathmnist-28px-batchnorm-policy-none-s0` (attempt 4) | 0.738804 | -28.04pp | 6785.219862937927 | False |
| 64px | `A-pathmnist-64px-batchnorm-policy-none-s0` (attempt 1) | 0.910036 | -43.26pp | 7345.785368919373 | True |
| 128px | `D-pathmnist-128px-batchnorm-policy-none-s0` (attempt 1) | 0.809476 | -38.03pp | 10717.256802797318 | True |

**Older-fingerprint disclosure**: the 28px comparator
(`A-pathmnist-28px-batchnorm-policy-none-s0`, attempt 4) belongs to the
**older evaluator-fingerprint cohort** documented in
`docs/phase2b_validation_evaluation_evaluator_fingerprint_drift_addendum.md`
-- it is scientifically valid but not current-fingerprint-compatible.
The 64px comparator is current-fingerprint-compatible.

**This is a single seed at each of the three resolutions and must not
be interpreted as establishing a resolution trend.** Clean accuracy and
N=50 naive TTA delta both move non-monotonically across
28px->64px->128px -- with only one seed per resolution, this pattern
could reflect genuine resolution effects, seed-specific noise, or both.
No hypothesis has been modified based on this comparison, and none is
implied by presenting it.

## 8. Scope and status

**This is a single-seed validation canary for one dataset/resolution/
architecture cell.** It does not establish a general claim about TTA
behavior at 128px resolution, was not used to select any threshold, and
did not trigger any change to the frozen protocol, seed, prefixes,
aggregation, batching, evaluator, or metric code. **No test split was
accessed at any point.** Confirmation requires completing the remaining
5 Block D cells (2 more PathMNIST seeds, 3 BloodMNIST seeds) per the
preregistered Block D design. No significance test has been run and
none is implied by this document.

