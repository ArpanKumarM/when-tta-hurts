# Phase 2B Block D Validation Evaluation: Full Audit (6/6 cells)

**Recorded: 2026-08-21.** This document records the closure of Phase 2B
Block D validation evaluation: all 6 native-128px cells (PathMNIST and
BloodMNIST, seeds 0/1/2, both BatchNorm, `policy=none`), completed and
independently verified. It supersedes nothing in the Block D canary
audit (`docs/phase2b_validation_evaluation_block_d_canary_audit.md`,
which remains the authoritative record of the seed-0 PathMNIST cell's
detailed report); this document adds the remaining 5 cells and the
complete 6-cell scientific record. All numbers below are generated
mechanically from persisted `metrics.json`/`metadata.json`/`status.json`/
ledger artifacts -- none are hand-transcribed. **This is a
validation-only result.** No test split was accessed at any point. No
protocol, threshold, seed, prefix, aggregation, batching, evaluator, or
metric code was changed as a result of any observation in this
document.

## 1. Scope and frozen protocol

- **Six native-128px cells**: `pathmnist` and `bloodmnist`, seeds 0/1/2
  each, architecture `small_cnn`, normalization `batchnorm`, training
  policy `none`, resolution 128 (native, never resized/interpolated).
- **Block D gate decision**: `artifacts/block_d_gate_decision.json`,
  `final_decision="INCLUDED"`, confirmed byte-identical to the
  committed version (no working-tree diff) throughout this task.
  Provenance commits (`source_commit`, `protocol_commit`,
  `spec_commit`) all confirmed ancestors of HEAD.
- **Frozen TTA seed**: `1306178015`. **Prefix sequence**:
  `[1, 2, 5, 10, 25, 50, 100]`. **Aggregators**: `mean_probability`, `majority_vote`,
  `confidence_weighted_average`, plus `original_anchored_tta` and
  `bn_adapted_tta` as separate registered conditions.
- **Bounded batching**: `inference_batch_size=256`,
  `bn_adaptation_batch_size=256`, `bn_adaptation_algorithm=sequential_microbatch_v1`
  -- identical, resolution-independent constants confirmed in every
  cell's persisted `metadata.json` regardless of native image size.
- **Validation-only scope**: every ledger row has `split=validation`;
  no test-split code path exists anywhere in `validation_evaluation.py`;
  no test-split artifact or access record exists anywhere in this
  repository as of this commit.

## 2. Canonical cell mapping

| Run ID | Training attempt | Evaluation attempt | Checkpoint hash | Evaluation ID | Evaluator fingerprint | Runtime (s) | Canonical |
|---|---|---|---|---|---|---|---|
| `D-pathmnist-128px-batchnorm-policy-none-s0` | 1 | 1 | `276169a842c64261...` | `b8dab819ccf0ca0f...` | `7fdce1db496ffb14...` | 10717.256802797318 | True |
| `D-pathmnist-128px-batchnorm-policy-none-s1` | 1 | 1 | `7fc1cf8f52c42029...` | `c3d2b60588b95984...` | `7fdce1db496ffb14...` | 10059.791746139526 | True |
| `D-pathmnist-128px-batchnorm-policy-none-s2` | 1 | 1 | `2c302516e9cc7a6d...` | `7ccbb84b249234f1...` | `7fdce1db496ffb14...` | 10135.607163906097 | True |
| `D-bloodmnist-128px-batchnorm-policy-none-s0` | 1 | 1 | `ec97e1b8b34af028...` | `cb480b562d6a647d...` | `7fdce1db496ffb14...` | 1744.6513159275055 | True |
| `D-bloodmnist-128px-batchnorm-policy-none-s1` | 1 | 1 | `f3980a44fb10c1bf...` | `e19a7c2537220970...` | `7fdce1db496ffb14...` | 1888.7003779411316 | True |
| `D-bloodmnist-128px-batchnorm-policy-none-s2` | 1 | 1 | `f0ddb47ab6d78491...` | `29c7c68f541b2bf0...` | `7fdce1db496ffb14...` | 1844.0896799564362 | True |

Current evaluator fingerprint (full): `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`. All 6 cells match
this fingerprint exactly.

## 3. Integrity verification (all 6 cells)

- **Artifact manifest verification**: `verify_evaluation_artifact_manifest()` -- OK for all 6 cells.
- **Checkpoint binding**: `metadata.checkpoint_hash` matches
  `resolve_canonical_training_completion()`'s canonical training result
  in all 6 cells (True).
- **Dataset checksum**: expected == actual MD5 in all 6 cells
  (True).
- **`resized=False`** in all 6 cells
  (True).
- **Semantic metric recomputation**: clean metrics + all 7 registered
  prefixes x 3 `naive_tta` aggregators, recomputed independently via
  `compute_metrics_from_probabilities()`/`_recompute_all_conditions_from_predictions()`
  -- **0 mismatches** across all 6 cells combined, within the frozen `1e-6` tolerance.
- **Probability validity**: `clean_probs`, `view_probs`, `bn_adapted_probs`
  all finite, in `[0,1]`, row-normalized in all 6 cells
  (True).
- **Sample-index alignment**: `sample_indices` unique, contiguous, same
  length as `labels` in all 6 cells
  (True).
- **BN-applicability consistency**: `bn_adaptation_applicable=True` in
  all 6 cells, with positive `bn_adaptation_microbatches_at_primary_n`
  (2000 for PathMNIST cells, 350 for BloodMNIST cells), and
  `bn_adapted_probs`/`bn_adapted_prefix_sequence` present and matching
  the frozen `PREFIX_SEQUENCE` in every cell.
- **`test_metrics_observed=False`** on all 6 ledger rows.
- **Evaluator fingerprint**: matches current fingerprint exactly in all 6 cells.
- **Sole canonical-compatible completion**: each of the 6 cells has
  exactly one ledger row for its `training_run_id`, `status=completed`,
  `evaluation_attempt=1`, `confirmatory=True`, `split=validation`.

## 4. Full scientific results, per cell

### `D-pathmnist-128px-batchnorm-policy-none-s0`

**Clean performance**

| Metric | Value |
|---|---|
| accuracy | 0.809476 |
| macro_f1 | 0.791317 |
| negative_log_likelihood | 0.544360 |
| expected_calibration_error | 0.026443 |
| brier_score | 0.269084 |

**Naive mean-probability, all registered N**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.335166 | -47.43pp | 0.320214 | 4.409486 | 0.446277 | 1.057157 | 0.623117 | 0.157922 |
| 2 | 0.353159 | -45.63pp | 0.326046 | 2.683034 | 0.237589 | 0.861190 | 0.596691 | 0.140084 |
| 5 | 0.377449 | -43.20pp | 0.339704 | 1.717073 | 0.120005 | 0.743405 | 0.567177 | 0.142183 |
| 10 | 0.401040 | -40.84pp | 0.352750 | 1.513733 | 0.075445 | 0.702727 | 0.536182 | 0.134313 |
| 25 | 0.416333 | -39.31pp | 0.354213 | 1.437935 | 0.057280 | 0.680425 | 0.512225 | 0.112802 |
| 50 | 0.429128 | -38.03pp | 0.358252 | 1.415067 | 0.070879 | 0.671673 | 0.496048 | 0.111228 |
| 100 | 0.437425 | -37.21pp | 0.359274 | 1.404171 | 0.089260 | 0.667002 | 0.486540 | 0.114376 |

**Majority-vote, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.335166 | -47.43pp | 18.370044 |
| 2 | 0.329768 | -47.97pp | 13.950218 |
| 5 | 0.367753 | -44.17pp | 7.968361 |
| 10 | 0.389144 | -42.03pp | 4.791819 |
| 25 | 0.407537 | -40.19pp | 2.780026 |
| 50 | 0.422131 | -38.73pp | 2.181699 |
| 100 | 0.425430 | -38.40pp | 1.919582 |

**Confidence-weighted, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.335166 | -47.43pp | 4.409486 |
| 2 | 0.352259 | -45.72pp | 2.724136 |
| 5 | 0.377749 | -43.17pp | 1.759667 |
| 10 | 0.396441 | -41.30pp | 1.548549 |
| 25 | 0.409636 | -39.98pp | 1.464912 |
| 50 | 0.421731 | -38.77pp | 1.439500 |
| 100 | 0.430628 | -37.88pp | 1.427607 |

**Original-anchored, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.661935 | -14.75pp | 0.899266 |
| 2 | 0.606657 | -20.28pp | 1.047019 |
| 5 | 0.518892 | -29.06pp | 1.214201 |
| 10 | 0.487005 | -32.25pp | 1.291832 |
| 25 | 0.458517 | -35.10pp | 1.351211 |
| 50 | 0.451519 | -35.80pp | 1.371616 |
| 100 | 0.447821 | -36.17pp | 1.382368 |

**BN-adapted, all registered N**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.843263 | +3.38pp | 0.507697 | 0.083759 | 0.240471 | 0.076933 | 0.504197 |
| 2 | 0.769792 | -3.97pp | 0.676834 | 0.049604 | 0.333532 | 0.149667 | 0.427597 |
| 5 | 0.806477 | -0.30pp | 0.599897 | 0.071743 | 0.291232 | 0.110645 | 0.454355 |
| 10 | 0.809876 | +0.04pp | 0.588739 | 0.070588 | 0.285817 | 0.105582 | 0.450682 |
| 25 | 0.827469 | +1.80pp | 0.549261 | 0.079122 | 0.263408 | 0.091504 | 0.483211 |
| 50 | 0.768493 | -4.10pp | 0.675594 | 0.050293 | 0.335798 | 0.150161 | 0.422875 |
| 100 | 0.816873 | +0.74pp | 0.576782 | 0.076631 | 0.279958 | 0.100148 | 0.464323 |

**Runtime and latency**

- Runtime: `10717.256802797318` s (~2.98h)
- Clean latency: 2.326366 s
- N=50 compute multiplier: 50.4771
- N=100 compute multiplier: 104.1123

### `D-pathmnist-128px-batchnorm-policy-none-s1`

**Clean performance**

| Metric | Value |
|---|---|
| accuracy | 0.921232 |
| macro_f1 | 0.916634 |
| negative_log_likelihood | 0.231897 |
| expected_calibration_error | 0.010971 |
| brier_score | 0.115916 |

**Naive mean-probability, all registered N**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.326569 | -59.47pp | 0.301523 | 5.458388 | 0.508627 | 1.124914 | 0.654405 | 0.104061 |
| 2 | 0.364954 | -55.63pp | 0.331555 | 3.278441 | 0.260753 | 0.902244 | 0.612088 | 0.096447 |
| 5 | 0.371551 | -54.97pp | 0.330494 | 1.979461 | 0.157330 | 0.777546 | 0.603841 | 0.083756 |
| 10 | 0.390244 | -53.10pp | 0.340535 | 1.635758 | 0.099093 | 0.731349 | 0.583876 | 0.087563 |
| 25 | 0.413235 | -50.80pp | 0.355521 | 1.499030 | 0.056864 | 0.704929 | 0.557834 | 0.074873 |
| 50 | 0.417533 | -50.37pp | 0.352747 | 1.461483 | 0.069292 | 0.695620 | 0.553711 | 0.081218 |
| 100 | 0.428229 | -49.30pp | 0.360490 | 1.446547 | 0.076311 | 0.691002 | 0.542426 | 0.085025 |

**Majority-vote, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.326569 | -59.47pp | 18.607576 |
| 2 | 0.296581 | -62.47pp | 14.221281 |
| 5 | 0.332467 | -58.88pp | 8.509977 |
| 10 | 0.358457 | -56.28pp | 5.191609 |
| 25 | 0.383447 | -53.78pp | 3.069261 |
| 50 | 0.392743 | -52.85pp | 2.311589 |
| 100 | 0.400840 | -52.04pp | 1.984983 |

**Confidence-weighted, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.326569 | -59.47pp | 5.458388 |
| 2 | 0.365854 | -55.54pp | 3.295920 |
| 5 | 0.377049 | -54.42pp | 2.002446 |
| 10 | 0.396341 | -52.49pp | 1.651630 |
| 25 | 0.414534 | -50.67pp | 1.506792 |
| 50 | 0.420132 | -50.11pp | 1.465883 |
| 100 | 0.426030 | -49.52pp | 1.448947 |

**Original-anchored, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.782787 | -13.84pp | 0.661059 |
| 2 | 0.653439 | -26.78pp | 0.858905 |
| 5 | 0.532187 | -38.90pp | 1.115194 |
| 10 | 0.482307 | -43.89pp | 1.248498 |
| 25 | 0.456018 | -46.52pp | 1.353157 |
| 50 | 0.441923 | -47.93pp | 1.390124 |
| 100 | 0.438725 | -48.25pp | 1.410969 |

**BN-adapted, all registered N**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.742603 | -17.86pp | 0.999650 | 0.132129 | 0.396905 | 0.237088 | 0.505076 |
| 2 | 0.706018 | -21.52pp | 1.141392 | 0.154684 | 0.451543 | 0.269748 | 0.422589 |
| 5 | 0.722411 | -19.88pp | 1.069704 | 0.143101 | 0.422839 | 0.254123 | 0.447970 |
| 10 | 0.726409 | -19.48pp | 1.035495 | 0.139083 | 0.417842 | 0.249132 | 0.440355 |
| 25 | 0.731807 | -18.94pp | 1.053749 | 0.138896 | 0.413347 | 0.246202 | 0.474619 |
| 50 | 0.712515 | -20.87pp | 1.053347 | 0.145038 | 0.437674 | 0.261393 | 0.407360 |
| 100 | 0.729608 | -19.16pp | 1.029321 | 0.138560 | 0.412829 | 0.246202 | 0.446701 |

**Runtime and latency**

- Runtime: `10059.791746139526` s (~2.79h)
- Clean latency: 2.321754 s
- N=50 compute multiplier: 49.7083
- N=100 compute multiplier: 99.4625

### `D-pathmnist-128px-batchnorm-policy-none-s2`

**Clean performance**

| Metric | Value |
|---|---|
| accuracy | 0.948121 |
| macro_f1 | 0.947918 |
| negative_log_likelihood | 0.150108 |
| expected_calibration_error | 0.010884 |
| brier_score | 0.077911 |

**Naive mean-probability, all registered N**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.341463 | -60.67pp | 0.339875 | 5.786018 | 0.525682 | 1.141117 | 0.646916 | 0.129094 |
| 2 | 0.377949 | -57.02pp | 0.371122 | 3.598499 | 0.259264 | 0.896819 | 0.607802 | 0.117534 |
| 5 | 0.406138 | -54.20pp | 0.392689 | 2.082751 | 0.132809 | 0.758752 | 0.576595 | 0.090559 |
| 10 | 0.431228 | -51.69pp | 0.408078 | 1.652937 | 0.067384 | 0.708506 | 0.550026 | 0.088632 |
| 25 | 0.459316 | -48.88pp | 0.428883 | 1.469902 | 0.055185 | 0.680617 | 0.520506 | 0.090559 |
| 50 | 0.471311 | -47.68pp | 0.436603 | 1.425414 | 0.078732 | 0.670305 | 0.507011 | 0.075145 |
| 100 | 0.482207 | -46.59pp | 0.442937 | 1.406031 | 0.100313 | 0.664649 | 0.495836 | 0.080925 |

**Majority-vote, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.341463 | -60.67pp | 18.196038 |
| 2 | 0.299680 | -64.84pp | 13.706391 |
| 5 | 0.367153 | -58.10pp | 8.018400 |
| 10 | 0.401739 | -54.64pp | 4.888396 |
| 25 | 0.435826 | -51.23pp | 2.561338 |
| 50 | 0.456517 | -49.16pp | 1.825772 |
| 100 | 0.469612 | -47.85pp | 1.594110 |

**Confidence-weighted, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.341463 | -60.67pp | 5.786018 |
| 2 | 0.376849 | -57.13pp | 3.622777 |
| 5 | 0.409536 | -53.86pp | 2.113446 |
| 10 | 0.433327 | -51.48pp | 1.679781 |
| 25 | 0.457517 | -49.06pp | 1.490427 |
| 50 | 0.468313 | -47.98pp | 1.441306 |
| 100 | 0.476010 | -47.21pp | 1.420133 |

**Original-anchored, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.805578 | -14.25pp | 0.579034 |
| 2 | 0.684726 | -26.34pp | 0.779560 |
| 5 | 0.564374 | -38.37pp | 1.041588 |
| 10 | 0.525190 | -42.29pp | 1.183029 |
| 25 | 0.494102 | -45.40pp | 1.299919 |
| 50 | 0.491603 | -45.65pp | 1.342521 |
| 100 | 0.492503 | -45.56pp | 1.364551 |

**BN-adapted, all registered N**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.833067 | -11.51pp | 0.602410 | 0.079159 | 0.261921 | 0.153400 | 0.585742 |
| 2 | 0.816773 | -13.13pp | 0.687396 | 0.091926 | 0.288615 | 0.170480 | 0.583815 |
| 5 | 0.821571 | -12.65pp | 0.657767 | 0.086919 | 0.280155 | 0.165314 | 0.581888 |
| 10 | 0.823571 | -12.46pp | 0.647451 | 0.086676 | 0.277156 | 0.161834 | 0.556840 |
| 25 | 0.823770 | -12.44pp | 0.653400 | 0.086082 | 0.278918 | 0.163100 | 0.583815 |
| 50 | 0.828369 | -11.98pp | 0.619056 | 0.079270 | 0.268052 | 0.159199 | 0.601156 |
| 100 | 0.831367 | -11.68pp | 0.613284 | 0.079431 | 0.265039 | 0.155614 | 0.593449 |

**Runtime and latency**

- Runtime: `10135.607163906097` s (~2.82h)
- Clean latency: 2.291503 s
- N=50 compute multiplier: 50.9896
- N=100 compute multiplier: 100.4027

### `D-bloodmnist-128px-batchnorm-policy-none-s0`

**Clean performance**

| Metric | Value |
|---|---|
| accuracy | 0.941589 |
| macro_f1 | 0.935956 |
| negative_log_likelihood | 0.297171 |
| expected_calibration_error | 0.134402 |
| brier_score | 0.131010 |

**Naive mean-probability, all registered N**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.337033 | -60.46pp | 0.332529 | 4.451952 | 0.455992 | 1.091847 | 0.650124 | 0.130000 |
| 2 | 0.320093 | -62.15pp | 0.307488 | 2.859439 | 0.313434 | 0.925810 | 0.667494 | 0.120000 |
| 5 | 0.317757 | -62.38pp | 0.278860 | 1.766793 | 0.226783 | 0.811160 | 0.671216 | 0.140000 |
| 10 | 0.340537 | -60.11pp | 0.279082 | 1.523143 | 0.170341 | 0.768236 | 0.647022 | 0.140000 |
| 25 | 0.336449 | -60.51pp | 0.249118 | 1.434728 | 0.157152 | 0.743701 | 0.653226 | 0.170000 |
| 50 | 0.345794 | -59.58pp | 0.249329 | 1.411247 | 0.154904 | 0.733952 | 0.642680 | 0.160000 |
| 100 | 0.343458 | -59.81pp | 0.242477 | 1.393281 | 0.168976 | 0.728529 | 0.647643 | 0.200000 |

**Majority-vote, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.337033 | -60.46pp | 18.318463 |
| 2 | 0.221963 | -71.96pp | 13.601319 |
| 5 | 0.289720 | -65.19pp | 6.897257 |
| 10 | 0.328855 | -61.27pp | 3.613817 |
| 25 | 0.344042 | -59.75pp | 1.859521 |
| 50 | 0.358645 | -58.29pp | 1.521363 |
| 100 | 0.358061 | -58.35pp | 1.391808 |

**Confidence-weighted, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.337033 | -60.46pp | 4.451952 |
| 2 | 0.321262 | -62.03pp | 2.932619 |
| 5 | 0.314252 | -62.73pp | 1.860319 |
| 10 | 0.330607 | -61.10pp | 1.610167 |
| 25 | 0.328271 | -61.33pp | 1.510214 |
| 50 | 0.331776 | -60.98pp | 1.480786 |
| 100 | 0.327687 | -61.39pp | 1.459622 |

**Original-anchored, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.629089 | -31.25pp | 0.709445 |
| 2 | 0.561332 | -38.03pp | 0.905618 |
| 5 | 0.467874 | -47.37pp | 1.129557 |
| 10 | 0.409463 | -53.21pp | 1.244244 |
| 25 | 0.361565 | -58.00pp | 1.330215 |
| 50 | 0.350467 | -59.11pp | 1.359738 |
| 100 | 0.346379 | -59.52pp | 1.368240 |

**BN-adapted, all registered N**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.875000 | -6.66pp | 0.500272 | 0.157590 | 0.232249 | 0.084988 | 0.230000 |
| 2 | 0.852804 | -8.88pp | 0.603029 | 0.183694 | 0.282651 | 0.112903 | 0.300000 |
| 5 | 0.827687 | -11.39pp | 0.684210 | 0.197767 | 0.327817 | 0.140819 | 0.320000 |
| 10 | 0.823598 | -11.80pp | 0.694446 | 0.196262 | 0.332301 | 0.144541 | 0.310000 |
| 25 | 0.823598 | -11.80pp | 0.694905 | 0.198422 | 0.333288 | 0.144541 | 0.310000 |
| 50 | 0.814836 | -12.68pp | 0.700936 | 0.195337 | 0.339707 | 0.153226 | 0.300000 |
| 100 | 0.823598 | -11.80pp | 0.703334 | 0.201338 | 0.337354 | 0.145782 | 0.330000 |

**Runtime and latency**

- Runtime: `1744.6513159275055` s (~0.48h)
- Clean latency: 0.391936 s
- N=50 compute multiplier: 54.6700
- N=100 compute multiplier: 108.5587

### `D-bloodmnist-128px-batchnorm-policy-none-s1`

**Clean performance**

| Metric | Value |
|---|---|
| accuracy | 0.844042 |
| macro_f1 | 0.835549 |
| negative_log_likelihood | 0.462448 |
| expected_calibration_error | 0.043012 |
| brier_score | 0.221498 |

**Naive mean-probability, all registered N**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.318925 | -52.51pp | 0.329288 | 5.406728 | 0.509020 | 1.153852 | 0.659516 | 0.202247 |
| 2 | 0.300818 | -54.32pp | 0.312748 | 3.325816 | 0.335221 | 0.959634 | 0.674048 | 0.164794 |
| 5 | 0.315421 | -52.86pp | 0.307098 | 1.941048 | 0.222373 | 0.834177 | 0.654671 | 0.153558 |
| 10 | 0.325935 | -51.81pp | 0.300520 | 1.651465 | 0.170879 | 0.787496 | 0.640830 | 0.146067 |
| 25 | 0.334112 | -50.99pp | 0.285334 | 1.544245 | 0.147166 | 0.763202 | 0.631834 | 0.149813 |
| 50 | 0.341706 | -50.23pp | 0.288380 | 1.504987 | 0.146831 | 0.752433 | 0.620761 | 0.138577 |
| 100 | 0.338785 | -50.53pp | 0.271863 | 1.490512 | 0.147976 | 0.746935 | 0.623529 | 0.134831 |

**Majority-vote, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.318925 | -52.51pp | 18.818791 |
| 2 | 0.216706 | -62.73pp | 14.519197 |
| 5 | 0.292640 | -55.14pp | 7.968717 |
| 10 | 0.308995 | -53.50pp | 4.463695 |
| 25 | 0.331776 | -51.23pp | 2.436183 |
| 50 | 0.348131 | -49.59pp | 1.854648 |
| 100 | 0.347547 | -49.65pp | 1.675170 |

**Confidence-weighted, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.318925 | -52.51pp | 5.406728 |
| 2 | 0.300818 | -54.32pp | 3.397472 |
| 5 | 0.299650 | -54.44pp | 2.046948 |
| 10 | 0.308995 | -53.50pp | 1.746519 |
| 25 | 0.314252 | -52.98pp | 1.628311 |
| 50 | 0.318341 | -52.57pp | 1.584663 |
| 100 | 0.320093 | -52.39pp | 1.566573 |

**Original-anchored, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.591706 | -25.23pp | 0.850420 |
| 2 | 0.535631 | -30.84pp | 1.035804 |
| 5 | 0.435748 | -40.83pp | 1.247350 |
| 10 | 0.400117 | -44.39pp | 1.352054 |
| 25 | 0.356308 | -48.77pp | 1.433345 |
| 50 | 0.356308 | -48.77pp | 1.452182 |
| 100 | 0.345210 | -49.88pp | 1.464602 |

**BN-adapted, all registered N**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.856308 | +1.23pp | 0.492658 | 0.120888 | 0.235027 | 0.081661 | 0.520599 |
| 2 | 0.842290 | -0.18pp | 0.549113 | 0.140636 | 0.263630 | 0.121799 | 0.647940 |
| 5 | 0.776869 | -6.72pp | 0.653714 | 0.098246 | 0.327985 | 0.209689 | 0.704120 |
| 10 | 0.773949 | -7.01pp | 0.660731 | 0.097546 | 0.331742 | 0.212457 | 0.700375 |
| 25 | 0.772780 | -7.13pp | 0.664747 | 0.097935 | 0.334383 | 0.215225 | 0.707865 |
| 50 | 0.759346 | -8.47pp | 0.685357 | 0.087540 | 0.347502 | 0.233218 | 0.719101 |
| 100 | 0.767523 | -7.65pp | 0.674273 | 0.093281 | 0.338650 | 0.220761 | 0.704120 |

**Runtime and latency**

- Runtime: `1888.7003779411316` s (~0.52h)
- Clean latency: 0.400355 s
- N=50 compute multiplier: 57.6550
- N=100 compute multiplier: 115.1900

### `D-bloodmnist-128px-batchnorm-policy-none-s2`

**Clean performance**

| Metric | Value |
|---|---|
| accuracy | 0.924065 |
| macro_f1 | 0.915590 |
| negative_log_likelihood | 0.332627 |
| expected_calibration_error | 0.129394 |
| brier_score | 0.147962 |

**Naive mean-probability, all registered N**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.317173 | -60.69pp | 0.299102 | 4.752458 | 0.481465 | 1.130216 | 0.664981 | 0.100000 |
| 2 | 0.290888 | -63.32pp | 0.258830 | 3.083913 | 0.345743 | 0.957713 | 0.695322 | 0.123077 |
| 5 | 0.312500 | -61.16pp | 0.256916 | 1.969785 | 0.237872 | 0.846877 | 0.673198 | 0.138462 |
| 10 | 0.307827 | -61.62pp | 0.225682 | 1.693460 | 0.213163 | 0.803592 | 0.678255 | 0.138462 |
| 25 | 0.301402 | -62.27pp | 0.203313 | 1.603310 | 0.228424 | 0.782027 | 0.683312 | 0.115385 |
| 50 | 0.297897 | -62.62pp | 0.198450 | 1.570057 | 0.246392 | 0.773194 | 0.687737 | 0.123077 |
| 100 | 0.294393 | -62.97pp | 0.196355 | 1.552848 | 0.262410 | 0.768705 | 0.692162 | 0.130769 |

**Majority-vote, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.317173 | -60.69pp | 18.867210 |
| 2 | 0.247664 | -67.64pp | 14.931484 |
| 5 | 0.301402 | -62.27pp | 8.846848 |
| 10 | 0.316589 | -60.75pp | 5.106988 |
| 25 | 0.312500 | -61.16pp | 2.493759 |
| 50 | 0.306659 | -61.74pp | 1.872220 |
| 100 | 0.298481 | -62.56pp | 1.681724 |

**Confidence-weighted, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.317173 | -60.69pp | 4.752458 |
| 2 | 0.287967 | -63.61pp | 3.180787 |
| 5 | 0.300234 | -62.38pp | 2.115273 |
| 10 | 0.301986 | -62.21pp | 1.838657 |
| 25 | 0.294977 | -62.91pp | 1.734634 |
| 50 | 0.290304 | -63.38pp | 1.694956 |
| 100 | 0.287967 | -63.61pp | 1.675811 |

**Original-anchored, all registered N**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.586449 | -33.76pp | 0.756084 |
| 2 | 0.523949 | -40.01pp | 0.962941 |
| 5 | 0.419977 | -50.41pp | 1.220771 |
| 10 | 0.360397 | -56.37pp | 1.357858 |
| 25 | 0.312500 | -61.16pp | 1.471368 |
| 50 | 0.301986 | -62.21pp | 1.505482 |
| 100 | 0.296145 | -62.79pp | 1.521075 |

**BN-adapted, all registered N**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.814836 | -10.92pp | 0.585778 | 0.119509 | 0.282069 | 0.145386 | 0.330769 |
| 2 | 0.797897 | -12.62pp | 0.680094 | 0.148279 | 0.328314 | 0.163085 | 0.323077 |
| 5 | 0.753505 | -17.06pp | 0.789122 | 0.144699 | 0.389803 | 0.209861 | 0.307692 |
| 10 | 0.756425 | -16.76pp | 0.789434 | 0.150087 | 0.390359 | 0.205436 | 0.292308 |
| 25 | 0.752336 | -17.17pp | 0.794394 | 0.147948 | 0.393543 | 0.209861 | 0.292308 |
| 50 | 0.733061 | -19.10pp | 0.820810 | 0.136576 | 0.410397 | 0.230088 | 0.284615 |
| 100 | 0.750584 | -17.35pp | 0.806968 | 0.148332 | 0.399559 | 0.213021 | 0.307692 |

**Runtime and latency**

- Runtime: `1844.0896799564362` s (~0.51h)
- Clean latency: 0.392655 s
- N=50 compute multiplier: 53.7039
- N=100 compute multiplier: 107.3183

## 5. Three-seed descriptive summaries

### pathmnist

- **Clean accuracy**: individual seeds [0.809476, 0.921232, 0.948121]; mean=0.8929; sample stdev=0.0735; min=0.8095; max=0.9481
- **N=50 naive mean-probability delta accuracy**: individual seeds [-0.380348, -0.503699, -0.476809] (['-38.03pp', '-50.37pp', '-47.68pp']); mean=-45.36pp; sample stdev=+6.49pp; min=-50.37pp; max=-38.03pp
- **N=50 BN-adapted delta accuracy**: individual seeds [-0.040984, -0.208717, -0.119752] (['-4.10pp', '-20.87pp', '-11.98pp']); mean=-12.32pp; sample stdev=+8.39pp; min=-20.87pp; max=-4.10pp
- **N=50 naive harm rate**: individual seeds [0.496048, 0.553711, 0.507011]; mean=0.5189; sample stdev=0.0306; min=0.4960; max=0.5537
- **N=50 naive rescue rate**: individual seeds [0.111228, 0.081218, 0.075145]; mean=0.0892; sample stdev=0.0193; min=0.0751; max=0.1112
- **Runtime (s)**: individual seeds [10717.256803, 10059.791746, 10135.607164]; mean=10304.2186; sample stdev=359.7046; min=10059.7917; max=10717.2568
- **Clean latency (s)**: individual seeds [2.326366, 2.321754, 2.291503]; mean=2.3132; sample stdev=0.0189; min=2.2915; max=2.3264

### bloodmnist

- **Clean accuracy**: individual seeds [0.941589, 0.844042, 0.924065]; mean=0.9032; sample stdev=0.0520; min=0.8440; max=0.9416
- **N=50 naive mean-probability delta accuracy**: individual seeds [-0.595794, -0.502336, -0.626168] (['-59.58pp', '-50.23pp', '-62.62pp']); mean=-57.48pp; sample stdev=+6.45pp; min=-62.62pp; max=-50.23pp
- **N=50 BN-adapted delta accuracy**: individual seeds [-0.126752, -0.084696, -0.191005] (['-12.68pp', '-8.47pp', '-19.10pp']); mean=-13.42pp; sample stdev=+5.35pp; min=-19.10pp; max=-8.47pp
- **N=50 naive harm rate**: individual seeds [0.64268, 0.620761, 0.687737]; mean=0.6504; sample stdev=0.0341; min=0.6208; max=0.6877
- **N=50 naive rescue rate**: individual seeds [0.16, 0.138577, 0.123077]; mean=0.1406; sample stdev=0.0185; min=0.1231; max=0.1600
- **Runtime (s)**: individual seeds [1744.651316, 1888.700378, 1844.08968]; mean=1825.8138; sample stdev=73.7431; min=1744.6513; max=1888.7004
- **Clean latency (s)**: individual seeds [0.391936, 0.400355, 0.392655]; mean=0.3950; sample stdev=0.0047; min=0.3919; max=0.4004

**All summaries above are purely descriptive.** No significance test has
been run and none is implied; this is a 3-seed, 2-dataset,
single-architecture (`small_cnn`, BatchNorm), single-resolution (128px)
result.

## 6. Full N curves: three-seed descriptive view

### pathmnist: naive mean-probability delta accuracy by N, across seeds

| N | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |
|---|---|---|---|---|---|
| 1 | -47.43pp | -59.47pp | -60.67pp | -55.85pp | +7.32pp |
| 2 | -45.63pp | -55.63pp | -57.02pp | -52.76pp | +6.21pp |
| 5 | -43.20pp | -54.97pp | -54.20pp | -50.79pp | +6.58pp |
| 10 | -40.84pp | -53.10pp | -51.69pp | -48.54pp | +6.71pp |
| 25 | -39.31pp | -50.80pp | -48.88pp | -46.33pp | +6.15pp |
| 50 | -38.03pp | -50.37pp | -47.68pp | -45.36pp | +6.49pp |
| 100 | -37.21pp | -49.30pp | -46.59pp | -44.37pp | +6.35pp |

### pathmnist: BN-adapted delta accuracy by N, across seeds

| N | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | # positive |
|---|---|---|---|---|---|---|
| 1 | +3.38pp | -17.86pp | -11.51pp | -8.66pp | +10.90pp | 1/3 |
| 2 | -3.97pp | -21.52pp | -13.13pp | -12.87pp | +8.78pp | 0/3 |
| 5 | -0.30pp | -19.88pp | -12.65pp | -10.95pp | +9.90pp | 0/3 |
| 10 | +0.04pp | -19.48pp | -12.46pp | -10.63pp | +9.89pp | 1/3 |
| 25 | +1.80pp | -18.94pp | -12.44pp | -9.86pp | +10.61pp | 1/3 |
| 50 | -4.10pp | -20.87pp | -11.98pp | -12.32pp | +8.39pp | 0/3 |
| 100 | +0.74pp | -19.16pp | -11.68pp | -10.03pp | +10.05pp | 1/3 |

### bloodmnist: naive mean-probability delta accuracy by N, across seeds

| N | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |
|---|---|---|---|---|---|
| 1 | -60.46pp | -52.51pp | -60.69pp | -57.89pp | +4.66pp |
| 2 | -62.15pp | -54.32pp | -63.32pp | -59.93pp | +4.89pp |
| 5 | -62.38pp | -52.86pp | -61.16pp | -58.80pp | +5.18pp |
| 10 | -60.11pp | -51.81pp | -61.62pp | -57.85pp | +5.28pp |
| 25 | -60.51pp | -50.99pp | -62.27pp | -57.92pp | +6.07pp |
| 50 | -59.58pp | -50.23pp | -62.62pp | -57.48pp | +6.45pp |
| 100 | -59.81pp | -50.53pp | -62.97pp | -57.77pp | +6.47pp |

### bloodmnist: BN-adapted delta accuracy by N, across seeds

| N | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | # positive |
|---|---|---|---|---|---|---|
| 1 | -6.66pp | +1.23pp | -10.92pp | -5.45pp | +6.16pp | 1/3 |
| 2 | -8.88pp | -0.18pp | -12.62pp | -7.22pp | +6.38pp | 0/3 |
| 5 | -11.39pp | -6.72pp | -17.06pp | -11.72pp | +5.18pp | 0/3 |
| 10 | -11.80pp | -7.01pp | -16.76pp | -11.86pp | +4.88pp | 0/3 |
| 25 | -11.80pp | -7.13pp | -17.17pp | -12.03pp | +5.03pp | 0/3 |
| 50 | -12.68pp | -8.47pp | -19.10pp | -13.42pp | +5.35pp | 0/3 |
| 100 | -11.80pp | -7.65pp | -17.35pp | -12.27pp | +4.86pp | 0/3 |

**Accurate statement of pattern**: naive TTA remains substantially
harmful throughout the full registered N range for every one of the 6
Block D cells -- deltas never approach zero, let alone become positive,
at any N from 1 to 100. BN adaptation is less harmful than naive TTA in
every cell at every N, but **does not consistently restore clean
performance at 128px**: only `D-pathmnist-128px-batchnorm-policy-none-s0`
crosses to a positive delta at some N (e.g. +3.38pp at N=1, +1.80pp at
N=25), while the other 5 cells remain negative under BN-adaptation at
every registered N. **Increasing N does not guarantee monotonic
improvement** in either condition -- several cells show deltas that
worsen slightly at higher N (e.g. bloodmnist-s2 naive: -60.69pp at N=1
but -62.97pp at N=100; pathmnist-s0 BN-adapted oscillates between
+3.38pp and -4.10pp across the curve rather than monotonically
improving).

## 7. Cross-resolution comparison (descriptive only)

Matched same-seed comparisons across 28px/64px/128px, wherever the
matrix supports them (BatchNorm, `policy=none`, `small_cnn`):

### pathmnist

| Seed | 28px clean / N=50 delta (fp cohort) | 64px clean / N=50 delta (fp cohort) | 128px clean / N=50 delta (fp cohort) |
|---|---|---|---|
| 0 | 0.738804 / -28.04pp (older) | 0.910036 / -43.26pp (current) | 0.809476 / -38.03pp (current) |
| 1 | 0.785286 / -56.33pp (older) | 0.910136 / -41.68pp (current) | 0.921232 / -50.37pp (current) |
| 2 | 0.870152 / -60.90pp (older) | 0.936925 / -35.58pp (current) | 0.948121 / -47.68pp (current) |

### bloodmnist

| Seed | 28px clean / N=50 delta (fp cohort) | 64px clean / N=50 delta (fp cohort) | 128px clean / N=50 delta (fp cohort) |
|---|---|---|---|
| 0 | 0.892523 / -30.84pp (current) | 0.907710 / -53.27pp (current) | 0.941589 / -59.58pp (current) |
| 1 | 0.918224 / -34.64pp (current) | 0.922897 / -67.17pp (current) | 0.844042 / -50.23pp (current) |
| 2 | 0.956776 / -50.00pp (current) | 0.923481 / -50.29pp (current) | 0.924065 / -62.62pp (current) |

**Explicit limitations of this comparison:**

- Only **3 seeds** are available at each resolution -- far too few to
  characterize a distribution, let alone a trend.
- **Resolution, trained-checkpoint, and seed effects cannot be cleanly
  separated into a causal resolution claim.** Each resolution uses an
  independently trained checkpoint (never the same weights evaluated at
  different resolutions), so any observed difference conflates the
  effect of resolution with the effect of that specific training run's
  random initialization, data order, and convergence point.
- **The three PathMNIST-28px BatchNorm comparators**
  (`A-pathmnist-28px-batchnorm-policy-none-s0/-s1/-s2`) **belong to the
  older evaluator-fingerprint cohort**, documented in
  `docs/phase2b_validation_evaluation_evaluator_fingerprint_drift_addendum.md`.
  **These artifacts remain scientifically valid under their persisted
  implementation but are not current-fingerprint-compatible.** All 64px
  and 128px comparators (both datasets, all seeds) are
  current-fingerprint-compatible.
- **No hypothesis or protocol was changed after viewing these results.**
  This comparison is reported exactly as computed, with no selection,
  no retrospective threshold, and no modification to any frozen
  document as a consequence of what it shows.

## 8. Runtime and resource accounting

| Run ID | Runtime (s) | Runtime (h) |
|---|---|---|
| `D-pathmnist-128px-batchnorm-policy-none-s0` | 10717.257 | 2.977 |
| `D-pathmnist-128px-batchnorm-policy-none-s1` | 10059.792 | 2.794 |
| `D-pathmnist-128px-batchnorm-policy-none-s2` | 10135.607 | 2.815 |
| `D-bloodmnist-128px-batchnorm-policy-none-s0` | 1744.651 | 0.485 |
| `D-bloodmnist-128px-batchnorm-policy-none-s1` | 1888.700 | 0.525 |
| `D-bloodmnist-128px-batchnorm-policy-none-s2` | 1844.090 | 0.512 |

- **PathMNIST-128px total**: 30912.656s (~8.59h); mean=10304.219s, sample SD=359.705s
- **BloodMNIST-128px total**: 5477.441s (~1.52h); mean=1825.814s, sample SD=73.743s
- **Total Block D evaluation runtime (all 6 cells)**: 36390.097s (~10.11h)
- **Observed PathMNIST/BloodMNIST runtime ratio** (mean/mean): 5.644x -- consistent in direction with the
  ratio observed at 28px in Block A (BloodMNIST substantially faster than
  PathMNIST at the same resolution), now confirmed to also hold at 128px.

### Latency and compute multipliers

| Run ID | Clean latency (s) | N=50 multiplier | N=100 multiplier |
|---|---|---|---|
| `D-pathmnist-128px-batchnorm-policy-none-s0` | 2.326366 | 50.4771 | 104.1123 |
| `D-pathmnist-128px-batchnorm-policy-none-s1` | 2.321754 | 49.7083 | 99.4625 |
| `D-pathmnist-128px-batchnorm-policy-none-s2` | 2.291503 | 50.9896 | 100.4027 |
| `D-bloodmnist-128px-batchnorm-policy-none-s0` | 0.391936 | 54.6700 | 108.5587 |
| `D-bloodmnist-128px-batchnorm-policy-none-s1` | 0.400355 | 57.6550 | 115.1900 |
| `D-bloodmnist-128px-batchnorm-policy-none-s2` | 0.392655 | 53.7039 | 107.3183 |

**Peak evaluation memory was not persisted by this evaluation pipeline**
(no such field exists in any cell's `metadata.json` or `metrics.json`
`latency` section) **and is not estimated or fabricated here.**

## 9. Limitations

- **Validation-stage results only.** No test-set results exist anywhere
  in this document or this project as of this commit; the test split
  has never been accessed.
- **Three seeds per condition.** All descriptive summaries in Sections
  5-7 are based on exactly 3 seeds per dataset/resolution cell -- too
  few for any distributional or inferential claim.
- **Evaluator-fingerprint cohort difference for three historical Block A
  cells** (`A-pathmnist-28px-batchnorm-policy-none-s0/-s1/-s2`):
  scientifically valid under their persisted implementation, but not
  current-fingerprint-compatible; reconciliation remains deferred and is
  explicitly out of scope for this document.
- **No registered inferential analysis has been performed yet.** Every
  number in this document is descriptive: individual values, means,
  sample standard deviations, minimums, and maximums -- never a
  significance test, confidence interval, or hypothesis test.
- **No claim that every augmentation policy or medical-imaging dataset
  behaves identically** to what is observed here. This document reports
  only PathMNIST and BloodMNIST, `policy=none`, BatchNorm, `small_cnn`,
  at 28px/64px/128px -- it does not generalize to other datasets,
  architectures, normalization schemes, or augmentation policies not
  evaluated in this project.
- **Block C did not reproduce the external positive-control result**
  (the source paper's reported ~+1.6pp naive TTA effect for
  ResNet-18/DermaMNIST) **under this project's frozen operationalization**
  -- see `docs/phase2b_validation_evaluation_block_c_audit.md` Section 7
  for the full comparison and the mechanically-listed protocol
  differences that may explain the disagreement. This does not
  establish that the source paper is incorrect.
- **Matched-policy training results (Block B) apply only to the
  preregistered Block B comparison** (`policy=matched_to_approved_tta_policy`
  vs. `policy=none`, BatchNorm, PathMNIST/BloodMNIST at 28px) -- they
  say nothing about whether matched-policy training would similarly
  reduce TTA harm at 64px, 128px, for GroupNorm cells, or for the
  ResNet-18/DermaMNIST architecture evaluated in Block C/D.

