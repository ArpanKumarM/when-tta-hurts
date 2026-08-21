# Phase 2B Block C Canary: DermaMNIST 28px ResNet-18 BatchNorm, seed 0

**Recorded: 2026-08-21.** This document records the first Block C
(positive-control) validation-evaluation completion:
`C-dermamnist-28px-resnet18-batchnorm-policy-none-s0`. This is a **single-seed validation
canary**, not a confirmed result. No protocol, threshold, policy, seed,
prefix, aggregation, batching, or metric code was changed as a result of
this observation. No test split was accessed at any point.

## 1. Exact identity

| Field | Value |
|---|---|
| Training run | `C-dermamnist-28px-resnet18-batchnorm-policy-none-s0` |
| Canonical training attempt / checkpoint hash | `1` / `bd529f57be5f06042f0b0f29e62444e6e38e6b0aa3ab9b40df144943a0bf7d48` |
| Evaluation attempt | `1` |
| Evaluation ID / evaluation-config hash | `e4410d838c8c843bf45e95c9f6bb174ca359dd62e5f64a85f73affaf6ecfac18` |
| Evaluator fingerprint | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` (matches current: True) |
| Dataset / resolution / split | `dermamnist` / 28 / `validation`, n_validation_samples=1003 |
| Dataset checksum (expected == actual) | checksum_verified=True, resized=False |
| Model architecture | `resnet18` (`build_resnet18_small_input`), 11,172,423 total parameters, 20 BatchNorm2d layers, 7 output classes |
| Normalization | `batchnorm` |
| BatchNorm applicability | `bn_adaptation_applicable=True`, `bn_adaptation_microbatches_at_primary_n=200` |
| Frozen TTA seed | `1306178015` |
| Metric-input contract | `probability_native_v1` |
| Runtime | `823.4756288528442` s (~13.7 min) |

### Artifact manifest

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `predictions.npz` | 3050756 | `6b0ce122c6dcc5eac49cadee34315958b35177d9cb6aad7c4e9e6be8ac92acec` |
| `metrics.json` | 16376 | `b0e79a03d15b29984f87a10893ead7417e49b2e69830ef1c3ce2c1d141615f6d` |
| `metadata.json` | 4704 | `2e2fc5e7e737ce253bc524d4cdeac692c83af71e1ef88d0fb93f0fafe217c14b` |
| `view_manifest.json` | 9504 | `3094176a3f16389b8b13eaab028461162f844380e1f33c4b43e90c0d6d71dc7b` |

## 2. Independent verification (all passed)

- **Manifest**: `verify_evaluation_artifact_manifest()` -- OK (raised no exception).
- **Full semantic recomputation**: clean metrics + all 7 registered
  prefixes x 3 `naive_tta` aggregators (mean-probability, majority-vote,
  confidence-weighted), recomputed independently from persisted
  `predictions.npz` via `compute_metrics_from_probabilities()`/
  `_recompute_all_conditions_from_predictions()` (never calls softmax on
  probability-native input) -- **0 mismatches** against
  `metrics.json`, within the frozen `1e-6` tolerance.
- **Probability validity**:
  - `clean_probs`: finite=True, in [0,1]=True, row-normalized=True, n_classes=7
  - `view_probs`: finite=True, in [0,1]=True, row-normalized=True, n_classes=7
  - `bn_adapted_probs`: finite=True, in [0,1]=True, row-normalized=True, n_classes=7
- **Sample/label alignment**: `sample_indices` unique=True, contiguous (0..1002)=True, length matches labels=True.
- **Dataset checksum**: verified=True, resized=False.
- **Checkpoint binding**: `metadata.checkpoint_hash` matches
  `resolve_canonical_training_completion()`'s canonical training result exactly (True).
- **BatchNorm contract**: `bn_adaptation_applicable=True`,
  `bn_adaptation_microbatches_at_primary_n=200` (positive),
  `bn_adapted_probs` shape `(7, 1003, 7)`,
  `bn_adapted_prefix_sequence=[1, 2, 5, 10, 25, 50, 100]` -- exact match to
  the frozen `PREFIX_SEQUENCE`.
- **Frozen provenance**: `prefix_sequence=[1, 2, 5, 10, 25, 50, 100]`,
  `tta_seed=1306178015`, batching fields (`inference_batch_size=256`,
  `bn_adaptation_batch_size=256`, `bn_adaptation_algorithm=sequential_microbatch_v1`)
  all match frozen values in `configs/validation_evaluation.yaml`.
- **Sole canonical-compatible completion**: exactly 1 ledger row for this
  `training_run_id`, `status=completed`, `evaluation_attempt=1`.
- **`test_metrics_observed=False`** on the ledger row.
- **`confirmatory=True`** on the ledger row.
- **No Block A/B/D interference**: Block C is the only block affected by
  this run; Block D has zero ledger rows and zero directories.

## 3. Clean metrics

| Metric | Value |
|---|---|
| accuracy | 0.748754 |
| macro_f1 | 0.444122 |
| negative_log_likelihood | 0.736957 |
| expected_calibration_error | 0.037316 |
| brier_score | 0.363018 |

## 4. Full N=1,2,5,10,25,50,100 curves -- all conditions

### Naive mean-probability aggregation

| N | Accuracy | Delta accuracy (pp) | Macro-F1 | NLL | ECE | Brier | Harm rate | Rescue rate |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.633101 | -11.57pp | 0.287884 | 1.112323 | 0.072305 | 0.507608 | 0.206391 | 0.154762 |
| 2 | 0.687936 | -6.08pp | 0.324574 | 0.947679 | 0.025433 | 0.451745 | 0.142477 | 0.182540 |
| 5 | 0.702891 | -4.59pp | 0.331428 | 0.845379 | 0.078232 | 0.413993 | 0.110519 | 0.146825 |
| 10 | 0.720837 | -2.79pp | 0.335867 | 0.832572 | 0.091019 | 0.404819 | 0.083888 | 0.138889 |
| 25 | 0.726820 | -2.19pp | 0.348976 | 0.817230 | 0.113119 | 0.396517 | 0.078562 | 0.146825 |
| 50 | 0.721834 | -2.69pp | 0.347846 | 0.811782 | 0.112820 | 0.394470 | 0.079893 | 0.130952 |
| 100 | 0.733799 | -1.50pp | 0.363845 | 0.808742 | 0.112354 | 0.393425 | 0.065246 | 0.134921 |

### Majority-vote aggregation

| N | Accuracy | Delta accuracy (pp) | NLL |
|---|---|---|---|
| 1 | 0.633101 | -11.57pp | 10.137802 |
| 2 | 0.602193 | -14.66pp | 7.273244 |
| 5 | 0.690927 | -5.78pp | 4.534187 |
| 10 | 0.702891 | -4.59pp | 3.531235 |
| 25 | 0.723829 | -2.49pp | 2.787645 |
| 50 | 0.722832 | -2.59pp | 2.258491 |
| 100 | 0.732802 | -1.60pp | 2.179836 |

### Confidence-weighted aggregation

| N | Accuracy | Delta accuracy (pp) | NLL |
|---|---|---|---|
| 1 | 0.633101 | -11.57pp | 1.112323 |
| 2 | 0.684945 | -6.38pp | 0.940106 |
| 5 | 0.705882 | -4.29pp | 0.830524 |
| 10 | 0.720837 | -2.79pp | 0.814037 |
| 25 | 0.725823 | -2.29pp | 0.794877 |
| 50 | 0.721834 | -2.69pp | 0.788142 |
| 100 | 0.728814 | -1.99pp | 0.784530 |

### Original-anchored aggregation

| N | Accuracy | Delta accuracy (pp) | NLL |
|---|---|---|---|
| 1 | 0.729811 | -1.89pp | 0.802587 |
| 2 | 0.724826 | -2.39pp | 0.810059 |
| 5 | 0.719840 | -2.89pp | 0.804743 |
| 10 | 0.728814 | -1.99pp | 0.812308 |
| 25 | 0.728814 | -1.99pp | 0.810048 |
| 50 | 0.723829 | -2.49pp | 0.808204 |
| 100 | 0.735793 | -1.30pp | 0.806961 |

### BN-adapted TTA (separate condition -- see Section 6 for the explicit naive-vs-BN-adapted distinction)

| N | Accuracy | Delta accuracy (pp) | NLL | ECE | Brier | Harm rate | Rescue rate |
|---|---|---|---|---|---|---|---|
| 1 | 0.756730 | +0.80pp | 0.697334 | 0.030954 | 0.346399 | 0.017310 | 0.083333 |
| 2 | 0.758724 | +1.00pp | 0.683550 | 0.036662 | 0.341314 | 0.029294 | 0.126984 |
| 5 | 0.759721 | +1.10pp | 0.676706 | 0.040751 | 0.339840 | 0.042610 | 0.170635 |
| 10 | 0.753739 | +0.50pp | 0.679088 | 0.047821 | 0.341525 | 0.049268 | 0.166667 |
| 25 | 0.756730 | +0.80pp | 0.677939 | 0.044125 | 0.341111 | 0.046605 | 0.170635 |
| 50 | 0.755733 | +0.70pp | 0.678237 | 0.047909 | 0.341065 | 0.047936 | 0.170635 |
| 100 | 0.752742 | +0.40pp | 0.678152 | 0.042970 | 0.341264 | 0.050599 | 0.166667 |

## 5. Runtime and latency

- Evaluation runtime: 823.4756288528442 s (~13.7 min)
- Clean latency: 0.304979 s

| N | Compute multiplier |
|---|---|
| 1 | 1.0493 |
| 2 | 2.1109 |
| 5 | 5.3025 |
| 10 | 10.6099 |
| 25 | 26.5723 |
| 50 | 53.1059 |
| 100 | 106.6296 |

## 6. Naive TTA versus BN-adapted TTA -- explicit distinction

At N=50 (the primary registered prefix):

| Condition | Accuracy | Delta accuracy |
|---|---|---|
| Naive mean-probability | 0.721834 | -2.69pp |
| Original-anchored | 0.723829 | -2.49pp |
| BN-adapted | 0.755733 | +0.70pp |

**Naive mean-probability, majority-vote, confidence-weighted, and
original-anchored TTA are all mildly harmful at N=50 in this cell**
(deltas from -2.69pp to
-2.49pp). **BN-adapted TTA is
the only condition that is net-positive** (+0.70pp), which
uses the frozen `sequential_microbatch_v1` batch-normalization-statistics
adaptation procedure (see Section 1's BatchNorm-applicability fields) --
this is a materially different procedure from the four naive/anchored
aggregation conditions above, which never touch the model's internal
normalization statistics.

## 7. Comparison with the source paper's external +1.6pp reference

The preregistered positive-control reference is the source paper's
reported ~+1.6pp naive TTA improvement for ResNet-18/DermaMNIST. This is
used strictly as an **external descriptive reference, not an acceptance
threshold** -- no protocol, threshold, or code decision depends on
whether this cell matches it.

| | This run (naive mean-probability, N=50) | This run (original-anchored, N=50) | This run (BN-adapted, N=50) | External reference |
|---|---|---|---|---|
| Delta accuracy | -2.69pp | -2.49pp | +0.70pp | ~+1.6pp (reported) |

Under this single-seed cell, naive mean-probability TTA (`-2.69pp`) and
original-anchored TTA (`-2.49pp`) do **not** match the direction of the
reported external result. The BN-adapted condition (`+0.70pp`) is
directionally positive but is **not the same procedure** the source
paper's reported figure is presumed to reflect (a naive TTA aggregation,
absent evidence otherwise) -- **this document does not describe
`+0.70pp` as reproducing the source paper's reported result.** It is
reported only as a directionally positive secondary condition under a
materially different aggregation procedure (normalization-statistics
adaptation, not naive/anchored aggregation). One seed cannot establish
reproduction or non-reproduction of any external claim; this is a
single data point pending the remaining 2 seeds.

## 8. Scope and status

**This is a single-seed validation canary for one dataset/resolution/
architecture cell.** It does not establish a general claim about
TTA behavior on DermaMNIST/ResNet-18, was not used to select any
threshold, and did not trigger any change to the frozen protocol, seed,
prefixes, aggregation, batching, evaluator, or metric code. No test
split was accessed. Confirmation requires completing the remaining 2
Block C cells (seeds 1 and 2) and reporting the full 3-seed descriptive
summary per the preregistered Block C design. No significance test has
been run and none is implied by this document.

