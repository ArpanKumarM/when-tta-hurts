# Phase 2B Block B Validation Evaluation: Full Audit (6/6 cells)

**Recorded: 2026-08-21.** This document records the closure of Phase 2B
Block B validation evaluation: all 6 matched-augmentation-training cells
(PathMNIST and BloodMNIST, seeds 0/1/2, both BatchNorm), completed and
independently verified. It supersedes nothing in the Block B canary audit
(`docs/phase2b_validation_evaluation_block_b_canary_audit.md`, which
remains the authoritative record of the seed-0 PathMNIST cell's own
thermal-throttling episode); this document adds the remaining 5 cells and
the complete 6-cell paired comparison against Block A. All numbers below
are generated mechanically from persisted `metrics.json`/`metadata.json`/
`status.json`/ledger artifacts -- none are hand-transcribed. **This is a
validation-only result.** No test split was accessed at any point across
Block A or Block B. No protocol, threshold, seed, prefix, aggregation,
batching, or metric code was changed as a result of any observation in
this document.

## 1. Six-cell identity, checkpoint, and evaluation mapping

| Dataset | Seed | Training run (B) | Training attempt | Checkpoint hash (B) | Evaluation attempt (B) | Evaluation ID (B) | Evaluator fingerprint (B) match current |
|---|---|---|---|---|---|---|---|
| pathmnist | 0 | `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` | 1 | `f9d06b302a5a0a73...` | 1 | `77f63ffdb92258f6...` | True |
| pathmnist | 1 | `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1` | 1 | `d8e1b91d421fd3c7...` | 1 | `f6da4027407a5935...` | True |
| pathmnist | 2 | `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2` | 1 | `44263c3e3db4a82d...` | 1 | `066af2461d526413...` | True |
| bloodmnist | 0 | `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0` | 1 | `fe0bc5ac9371bca5...` | 1 | `eb61fbb94bc62467...` | True |
| bloodmnist | 1 | `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1` | 1 | `233311a29208b194...` | 1 | `cb9357b9f3c2b924...` | True |
| bloodmnist | 2 | `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2` | 1 | `3d867a7872cb2a7f...` | 1 | `b525787ba6c284a1...` | True |

| Dataset | Seed | Comparator run (A) | Evaluation attempt (A) | Evaluator fingerprint (A) match current |
|---|---|---|---|---|
| pathmnist | 0 | `A-pathmnist-28px-batchnorm-policy-none-s0` | 4 | False |
| pathmnist | 1 | `A-pathmnist-28px-batchnorm-policy-none-s1` | 1 | False |
| pathmnist | 2 | `A-pathmnist-28px-batchnorm-policy-none-s2` | 1 | False |
| bloodmnist | 0 | `A-bloodmnist-28px-batchnorm-policy-none-s0` | 1 | True |
| bloodmnist | 1 | `A-bloodmnist-28px-batchnorm-policy-none-s1` | 1 | True |
| bloodmnist | 2 | `A-bloodmnist-28px-batchnorm-policy-none-s2` | 1 | True |

All 6 Block B cells: evaluation attempt 1, current-fingerprint-compatible
(`7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`). Three of the six Block A comparators
(`A-pathmnist-28px-batchnorm-policy-none-s0/-s1/-s2`) use the older
evaluator fingerprint -- see Section 9 for full disclosure. The remaining
three Block A comparators (`A-bloodmnist-28px-batchnorm-policy-none-*`)
are current-fingerprint-compatible.

## 2. Artifact manifest (all 6 Block B cells)

| Run ID | Artifact | Size (bytes) | SHA-256 |
|---|---|---|---|
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` | `predictions.npz` | 39057252 | `18cc754f1e4898db086fc825...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` | `metrics.json` | 16352 | `fffd5ae250b95bfdf677e9cc...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` | `metadata.json` | 4729 | `c6c239dba3812141a5eb57b2...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` | `view_manifest.json` | 99517 | `764f531cb816fa47cced92a7...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1` | `predictions.npz` | 39057252 | `25879f147edc03793b7b1c07...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1` | `metrics.json` | 16353 | `a760f76e7ace86df4390bf45...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1` | `metadata.json` | 4729 | `49afdadf9eaae9d7e4a1cda0...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1` | `view_manifest.json` | 99517 | `764f531cb816fa47cced92a7...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2` | `predictions.npz` | 39057252 | `8a3e592e85807d4a558ce1dc...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2` | `metrics.json` | 16374 | `8ed704701855c1a3a4ce7521...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2` | `metadata.json` | 4729 | `c86fad083d92af2c2f5b1bd7...` |
| `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2` | `view_manifest.json` | 99517 | `764f531cb816fa47cced92a7...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0` | `predictions.npz` | 5945700 | `24513080ae787ed024751834...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0` | `metrics.json` | 16374 | `8183f55060a4424259ab32c1...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0` | `metadata.json` | 4731 | `d5c299980c185f5fa75335d7...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0` | `view_manifest.json` | 16594 | `e125a1a5f1c5848cdbf730d4...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1` | `predictions.npz` | 5945700 | `1f418db1defa748022b5feb0...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1` | `metrics.json` | 16403 | `4a383d72894eb2b56c012066...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1` | `metadata.json` | 4731 | `70fb307126bd9f660e111c78...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1` | `view_manifest.json` | 16594 | `e125a1a5f1c5848cdbf730d4...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2` | `predictions.npz` | 5945700 | `ee049c389096f022423e349c...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2` | `metrics.json` | 16311 | `0e27824bf7e3c269b934e5df...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2` | `metadata.json` | 4731 | `457f4a0631aab87a1271171f...` |
| `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2` | `view_manifest.json` | 16594 | `e125a1a5f1c5848cdbf730d4...` |

All 6 manifests verified via `verify_evaluation_artifact_manifest()` --
zero mismatches (size and SHA-256 both match on-disk artifacts exactly).

## 3. Independent verification evidence (all 6 cells)

For every one of the 6 Block B cells, the following were independently
re-derived from persisted `predictions.npz` and cross-checked against
`metrics.json`, with **zero mismatches** in all 6 cells:

- **Manifest integrity**: `verify_evaluation_artifact_manifest()` -- OK for all 6.
- **Full semantic recomputation**: clean metrics (via
  `compute_metrics_from_probabilities()`, never calls softmax) + all 7
  registered prefixes x 3 `naive_tta` aggregators (mean-probability,
  majority-vote, confidence-weighted) via `_recompute_all_conditions_from_predictions()`
  -- compared against `metrics.json` within the frozen `1e-6` tolerance,
  zero mismatches across all 6 cells.
- **Probability validity**: `clean_probs`, `view_probs`, `bn_adapted_probs`
  all finite, within `[0,1]`, row-normalized (`sum(axis=-1) ~= 1.0`) in all 6 cells.
- **Dataset checksum**: `checksum_verified=True`, `resized=False` in all 6 cells.
- **Checkpoint binding**: `metadata.checkpoint_hash` matches
  `resolve_canonical_training_completion()`'s canonical training result
  exactly, in all 6 cells.
- **BatchNorm contract**: `bn_adaptation_applicable=True` in all 6 cells,
  with positive `bn_adaptation_microbatches_at_primary_n`
  (2000 for both PathMNIST cells, 350 for all three BloodMNIST cells),
  `bn_adapted_probs`/`bn_adapted_prefix_sequence` present and matching the
  frozen `PREFIX_SEQUENCE=[1,2,5,10,25,50,100]`.
- **Frozen provenance**: `prefix_sequence`, `tta_seed=1306178015`, and all
  batching fields (`inference_batch_size=256`,
  `bn_adaptation_batch_size=256`, `bn_adaptation_algorithm=sequential_microbatch_v1`)
  match `configs/validation_evaluation.yaml` exactly in all 6 cells.
- **Sole canonical-compatible completion**: each of the 6 cells is the
  sole ledger row for its `training_run_id`, `status=completed`,
  `test_metrics_observed=False`, `confirmatory=True`, `evaluation_attempt=1`.
- **No Block A/C/D interference**: Block A predictions.npz hashes
  unaffected; Block C and Block D have zero ledger rows and zero
  directories, confirmed mechanically (see Section 8).

## 4. Per-cell clean and N=50 metrics

**Correction to earlier prose** (this corrects a verbal report given during
live monitoring, not any persisted artifact): the per-prefix
mean-probability accuracy differences between A and B reported below are on
the order of approximately **22-70 percentage points** (`diff x 100`, e.g.
`+22.05pp` to `+70.21pp` across all 42 cell x prefix combinations), not
"0.22-0.70 percentage points" as an earlier interim status update
mis-stated. The underlying persisted numbers were never wrong -- only that
one verbal unit label was. All tables below carry the corrected `pp` unit
explicitly.

### pathmnist seed 0

A = `A-pathmnist-28px-batchnorm-policy-none-s0` (attempt 4); B = `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` (attempt 1).

**Clean performance**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.738804 | 0.699220 | -0.039584 |
| macro_f1 | 0.724357 | 0.679952 | -0.044405 |
| negative_log_likelihood | 0.914932 | 0.841913 | -0.073019 |
| expected_calibration_error | 0.090000 | 0.033868 | -0.056132 |
| brier_score | 0.381817 | 0.409923 | 0.028106 |

**Within-model TTA delta (N=50, mean-probability)**

- A: clean=0.738804, TTA@50=0.458417, delta=-0.280388 (-28.04pp)
- B: clean=0.699220, TTA@50=0.687425, delta=-0.011795 (-1.18pp)
- delta_B − delta_A = 0.268593 (+26.86pp), harm reduced in B

**Absolute TTA performance at N=50 mean-probability**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.458417 | 0.687425 | 0.229008 |
| macro_f1 | 0.416744 | 0.678216 | 0.261472 |
| negative_log_likelihood | 1.725584 | 0.874867 | -0.850717 |
| expected_calibration_error | 0.063112 | 0.064916 | 0.001804 |
| brier_score | 0.698864 | 0.431339 | -0.267525 |
| harm_rate | 0.429577 | 0.163688 | -0.265888 |
| rescue_rate | 0.141600 | 0.341309 | 0.199710 |

**Other aggregators / conditions at N=50 (accuracy / delta_accuracy)**

| Condition | A accuracy | B accuracy | A delta | B delta |
|---|---|---|---|---|
| Majority-vote | 0.434826 | 0.685126 | -0.303978 | -0.014094 |
| Confidence-weighted | 0.461915 | 0.690824 | -0.276889 | -0.008397 |
| Original-anchored | 0.479008 | 0.690724 | -0.259796 | -0.008497 |
| BN-adapted | 0.578069 | 0.666333 | -0.160736 | -0.032887 |

**Full N=1,2,5,10,25,50,100 mean-probability curve**

| N | A accuracy | B accuracy | diff (pp) | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.306977 | 0.642343 | +33.54pp | 5.078989 | 0.983663 | -4.095325 |
| 2 | 0.350360 | 0.659336 | +30.90pp | 3.272747 | 0.933380 | -2.339367 |
| 5 | 0.380448 | 0.678429 | +29.80pp | 2.165464 | 0.894778 | -1.270687 |
| 10 | 0.409136 | 0.682827 | +27.37pp | 1.885124 | 0.884370 | -1.000755 |
| 25 | 0.441224 | 0.687225 | +24.60pp | 1.761105 | 0.877242 | -0.883863 |
| 50 | 0.458417 | 0.687425 | +22.90pp | 1.725584 | 0.874867 | -0.850717 |
| 100 | 0.469412 | 0.689924 | +22.05pp | 1.706975 | 0.873326 | -0.833649 |

**Runtime and latency**

- Evaluation runtime: A=6785.220s, B=14166.387s
- Clean latency (s): A=0.181777, B=0.180592

| N | A compute multiplier | B compute multiplier |
|---|---|---|
| 1 | 1.1041 | 1.0974 |
| 2 | 2.2032 | 2.2135 |
| 5 | 5.7978 | 6.3515 |
| 10 | 11.0368 | 11.2142 |
| 25 | 27.6952 | 28.8745 |
| 50 | 55.2723 | 55.5014 |
| 100 | 110.7855 | 112.0600 |

### pathmnist seed 1

A = `A-pathmnist-28px-batchnorm-policy-none-s1` (attempt 1); B = `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1` (attempt 1).

**Clean performance**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.785286 | 0.856058 | 0.070772 |
| macro_f1 | 0.784397 | 0.855053 | 0.070656 |
| negative_log_likelihood | 0.636928 | 0.408918 | -0.228010 |
| expected_calibration_error | 0.050372 | 0.026085 | -0.024287 |
| brier_score | 0.308310 | 0.207283 | -0.101027 |

**Within-model TTA delta (N=50, mean-probability)**

- A: clean=0.785286, TTA@50=0.222011, delta=-0.563275 (-56.33pp)
- B: clean=0.856058, TTA@50=0.924030, delta=0.067973 (+6.80pp)
- delta_B − delta_A = 0.631248 (+63.12pp), harm reduced in B

**Absolute TTA performance at N=50 mean-probability**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.222011 | 0.924030 | 0.702019 |
| macro_f1 | 0.171661 | 0.925222 | 0.753561 |
| negative_log_likelihood | 1.939851 | 0.247082 | -1.692769 |
| expected_calibration_error | 0.264110 | 0.065379 | -0.198731 |
| brier_score | 0.861296 | 0.120568 | -0.740728 |
| harm_rate | 0.747581 | 0.040168 | -0.707413 |
| rescue_rate | 0.110801 | 0.711111 | 0.600310 |

**Other aggregators / conditions at N=50 (accuracy / delta_accuracy)**

| Condition | A accuracy | B accuracy | A delta | B delta |
|---|---|---|---|---|
| Majority-vote | 0.221311 | 0.922831 | -0.563974 | 0.066773 |
| Confidence-weighted | 0.219612 | 0.924430 | -0.565674 | 0.068373 |
| Original-anchored | 0.233507 | 0.923930 | -0.551779 | 0.067873 |
| BN-adapted | 0.465914 | 0.831967 | -0.319372 | -0.024090 |

**Full N=1,2,5,10,25,50,100 mean-probability curve**

| N | A accuracy | B accuracy | diff (pp) | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.253798 | 0.893443 | +63.96pp | 5.113330 | 0.305999 | -4.807331 |
| 2 | 0.253199 | 0.908737 | +65.55pp | 3.575924 | 0.275066 | -3.300858 |
| 5 | 0.241603 | 0.919732 | +67.81pp | 2.420792 | 0.255575 | -2.165218 |
| 10 | 0.229608 | 0.923531 | +69.39pp | 2.117874 | 0.250558 | -1.867315 |
| 25 | 0.224210 | 0.924330 | +70.01pp | 1.977247 | 0.247319 | -1.729929 |
| 50 | 0.222011 | 0.924030 | +70.20pp | 1.939851 | 0.247082 | -1.692769 |
| 100 | 0.223011 | 0.925130 | +70.21pp | 1.919338 | 0.246533 | -1.672805 |

**Runtime and latency**

- Evaluation runtime: A=6914.231s, B=6828.126s
- Clean latency (s): A=0.179636, B=0.189688

| N | A compute multiplier | B compute multiplier |
|---|---|---|
| 1 | 1.1032 | 1.0960 |
| 2 | 2.2170 | 2.2478 |
| 5 | 5.5728 | 5.6039 |
| 10 | 11.2042 | 11.1316 |
| 25 | 28.1592 | 27.7190 |
| 50 | 55.8822 | 55.7705 |
| 100 | 115.1052 | 111.0236 |

### pathmnist seed 2

A = `A-pathmnist-28px-batchnorm-policy-none-s2` (attempt 1); B = `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2` (attempt 1).

**Clean performance**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.870152 | 0.804678 | -0.065474 |
| macro_f1 | 0.870362 | 0.806595 | -0.063767 |
| negative_log_likelihood | 0.399272 | 0.545409 | 0.146137 |
| expected_calibration_error | 0.034341 | 0.032684 | -0.001657 |
| brier_score | 0.190049 | 0.272643 | 0.082594 |

**Within-model TTA delta (N=50, mean-probability)**

- A: clean=0.870152, TTA@50=0.261196, delta=-0.608956 (-60.90pp)
- B: clean=0.804678, TTA@50=0.826969, delta=0.022291 (+2.23pp)
- delta_B − delta_A = 0.631248 (+63.12pp), harm reduced in B

**Absolute TTA performance at N=50 mean-probability**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.261196 | 0.826969 | 0.565774 |
| macro_f1 | 0.179904 | 0.828007 | 0.648103 |
| negative_log_likelihood | 1.911149 | 0.508616 | -1.402533 |
| expected_calibration_error | 0.228525 | 0.067704 | -0.160821 |
| brier_score | 0.809174 | 0.256695 | -0.552478 |
| harm_rate | 0.704767 | 0.065466 | -0.639302 |
| rescue_rate | 0.033102 | 0.383828 | 0.350726 |

**Other aggregators / conditions at N=50 (accuracy / delta_accuracy)**

| Condition | A accuracy | B accuracy | A delta | B delta |
|---|---|---|---|---|
| Majority-vote | 0.260296 | 0.824170 | -0.609856 | 0.019492 |
| Confidence-weighted | 0.255398 | 0.828069 | -0.614754 | 0.023391 |
| Original-anchored | 0.272591 | 0.827369 | -0.597561 | 0.022691 |
| BN-adapted | 0.609256 | 0.830068 | -0.260896 | 0.025390 |

**Full N=1,2,5,10,25,50,100 mean-probability curve**

| N | A accuracy | B accuracy | diff (pp) | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.261495 | 0.783786 | +52.23pp | 6.771387 | 0.607072 | -6.164315 |
| 2 | 0.266393 | 0.802279 | +53.59pp | 4.435321 | 0.550938 | -3.884383 |
| 5 | 0.271092 | 0.818473 | +54.74pp | 2.710473 | 0.523776 | -2.186698 |
| 10 | 0.271991 | 0.823970 | +55.20pp | 2.211609 | 0.515629 | -1.695980 |
| 25 | 0.264294 | 0.826569 | +56.23pp | 1.975672 | 0.510490 | -1.465181 |
| 50 | 0.261196 | 0.826969 | +56.58pp | 1.911149 | 0.508616 | -1.402533 |
| 100 | 0.254198 | 0.825970 | +57.18pp | 1.883315 | 0.507847 | -1.375468 |

**Runtime and latency**

- Evaluation runtime: A=6963.303s, B=6639.001s
- Clean latency (s): A=0.185151, B=0.188176

| N | A compute multiplier | B compute multiplier |
|---|---|---|
| 1 | 1.0930 | 1.1280 |
| 2 | 2.1983 | 2.2191 |
| 5 | 5.5318 | 5.6010 |
| 10 | 11.0084 | 11.4748 |
| 25 | 27.8161 | 28.0284 |
| 50 | 56.3825 | 55.9606 |
| 100 | 117.0852 | 112.4826 |

### bloodmnist seed 0

A = `A-bloodmnist-28px-batchnorm-policy-none-s0` (attempt 1); B = `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0` (attempt 1).

**Clean performance**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.892523 | 0.855724 | -0.036799 |
| macro_f1 | 0.887763 | 0.823483 | -0.064280 |
| negative_log_likelihood | 0.330177 | 0.387361 | 0.057183 |
| expected_calibration_error | 0.022967 | 0.023205 | 0.000238 |
| brier_score | 0.166526 | 0.203058 | 0.036531 |

**Within-model TTA delta (N=50, mean-probability)**

- A: clean=0.892523, TTA@50=0.584112, delta=-0.308411 (-30.84pp)
- B: clean=0.855724, TTA@50=0.868575, delta=0.012850 (+1.29pp)
- delta_B − delta_A = 0.321262 (+32.13pp), harm reduced in B

**Absolute TTA performance at N=50 mean-probability**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.584112 | 0.868575 | 0.284463 |
| macro_f1 | 0.479058 | 0.843631 | 0.364573 |
| negative_log_likelihood | 1.033449 | 0.383152 | -0.650298 |
| expected_calibration_error | 0.107315 | 0.061419 | -0.045896 |
| brier_score | 0.539842 | 0.195394 | -0.344448 |
| harm_rate | 0.380236 | 0.049147 | -0.331089 |
| rescue_rate | 0.288043 | 0.380567 | 0.092523 |

**Other aggregators / conditions at N=50 (accuracy / delta_accuracy)**

| Condition | A accuracy | B accuracy | A delta | B delta |
|---|---|---|---|---|
| Majority-vote | 0.586449 | 0.868575 | -0.306075 | 0.012850 |
| Confidence-weighted | 0.547313 | 0.870911 | -0.345210 | 0.015187 |
| Original-anchored | 0.599883 | 0.871495 | -0.292640 | 0.015771 |
| BN-adapted | 0.785631 | 0.875000 | -0.106893 | 0.019276 |

**Full N=1,2,5,10,25,50,100 mean-probability curve**

| N | A accuracy | B accuracy | diff (pp) | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.437500 | 0.848715 | +41.12pp | 3.831318 | 0.411400 | -3.419918 |
| 2 | 0.451519 | 0.854556 | +40.30pp | 2.209141 | 0.395055 | -1.814086 |
| 5 | 0.498832 | 0.859813 | +36.10pp | 1.302308 | 0.388300 | -0.914008 |
| 10 | 0.539136 | 0.867407 | +32.83pp | 1.117786 | 0.384195 | -0.733591 |
| 25 | 0.573598 | 0.870327 | +29.67pp | 1.044426 | 0.381609 | -0.662817 |
| 50 | 0.584112 | 0.868575 | +28.45pp | 1.033449 | 0.383152 | -0.650298 |
| 100 | 0.594626 | 0.870911 | +27.63pp | 1.025607 | 0.382457 | -0.643150 |

**Runtime and latency**

- Evaluation runtime: A=1156.792s, B=1158.374s
- Clean latency (s): A=0.031008, B=0.031739

| N | A compute multiplier | B compute multiplier |
|---|---|---|
| 1 | 1.6733 | 1.8673 |
| 2 | 3.2435 | 3.7863 |
| 5 | 8.3168 | 8.5636 |
| 10 | 16.4892 | 17.0167 |
| 25 | 41.6237 | 42.5258 |
| 50 | 86.6309 | 85.7590 |
| 100 | 166.2684 | 170.3897 |

### bloodmnist seed 1

A = `A-bloodmnist-28px-batchnorm-policy-none-s1` (attempt 1); B = `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1` (attempt 1).

**Clean performance**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.918224 | 0.893107 | -0.025117 |
| macro_f1 | 0.907287 | 0.878540 | -0.028747 |
| negative_log_likelihood | 0.242306 | 0.286793 | 0.044487 |
| expected_calibration_error | 0.017754 | 0.025642 | 0.007888 |
| brier_score | 0.123539 | 0.149194 | 0.025656 |

**Within-model TTA delta (N=50, mean-probability)**

- A: clean=0.918224, TTA@50=0.571846, delta=-0.346379 (-34.64pp)
- B: clean=0.893107, TTA@50=0.930491, delta=0.037383 (+3.74pp)
- delta_B − delta_A = 0.383762 (+38.38pp), harm reduced in B

**Absolute TTA performance at N=50 mean-probability**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.571846 | 0.930491 | 0.358645 |
| macro_f1 | 0.467532 | 0.919526 | 0.451994 |
| negative_log_likelihood | 1.170985 | 0.232107 | -0.938878 |
| expected_calibration_error | 0.099176 | 0.058391 | -0.040785 |
| brier_score | 0.556272 | 0.112676 | -0.443596 |
| harm_rate | 0.405852 | 0.017659 | -0.388194 |
| rescue_rate | 0.321429 | 0.497268 | 0.175839 |

**Other aggregators / conditions at N=50 (accuracy / delta_accuracy)**

| Condition | A accuracy | B accuracy | A delta | B delta |
|---|---|---|---|---|
| Majority-vote | 0.566005 | 0.929907 | -0.352220 | 0.036799 |
| Confidence-weighted | 0.571262 | 0.930491 | -0.346963 | 0.037383 |
| Original-anchored | 0.595210 | 0.930491 | -0.323014 | 0.037383 |
| BN-adapted | 0.734813 | 0.882009 | -0.183411 | -0.011098 |

**Full N=1,2,5,10,25,50,100 mean-probability curve**

| N | A accuracy | B accuracy | diff (pp) | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.419393 | 0.911799 | +49.24pp | 3.276831 | 0.257508 | -3.019322 |
| 2 | 0.464953 | 0.924650 | +45.97pp | 2.125943 | 0.245782 | -1.880162 |
| 5 | 0.509346 | 0.925818 | +41.65pp | 1.450883 | 0.237932 | -1.212950 |
| 10 | 0.543808 | 0.932243 | +38.84pp | 1.267495 | 0.235151 | -1.032344 |
| 25 | 0.558995 | 0.929907 | +37.09pp | 1.187930 | 0.232178 | -0.955751 |
| 50 | 0.571846 | 0.930491 | +35.86pp | 1.170985 | 0.232107 | -0.938878 |
| 100 | 0.574182 | 0.931075 | +35.69pp | 1.158774 | 0.231841 | -0.926933 |

**Runtime and latency**

- Evaluation runtime: A=1168.915s, B=1146.429s
- Clean latency (s): A=0.031122, B=0.032115

| N | A compute multiplier | B compute multiplier |
|---|---|---|
| 1 | 1.5858 | 1.6851 |
| 2 | 3.1549 | 3.3638 |
| 5 | 7.9517 | 8.2435 |
| 10 | 16.1502 | 17.0206 |
| 25 | 40.9874 | 42.7312 |
| 50 | 81.9039 | 85.3834 |
| 100 | 162.0879 | 169.1138 |

### bloodmnist seed 2

A = `A-bloodmnist-28px-batchnorm-policy-none-s2` (attempt 1); B = `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2` (attempt 1).

**Clean performance**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.956776 | 0.915304 | -0.041472 |
| macro_f1 | 0.952301 | 0.901551 | -0.050750 |
| negative_log_likelihood | 0.130131 | 0.267403 | 0.137272 |
| expected_calibration_error | 0.008684 | 0.030144 | 0.021460 |
| brier_score | 0.066272 | 0.134577 | 0.068305 |

**Within-model TTA delta (N=50, mean-probability)**

- A: clean=0.956776, TTA@50=0.456776, delta=-0.500000 (-50.00pp)
- B: clean=0.915304, TTA@50=0.915304, delta=0.000000 (+0.00pp)
- delta_B − delta_A = 0.500000 (+50.00pp), harm reduced in B

**Absolute TTA performance at N=50 mean-probability**

| Metric | A | B | B − A |
|---|---|---|---|
| accuracy | 0.456776 | 0.915304 | 0.458528 |
| macro_f1 | 0.349863 | 0.897324 | 0.547461 |
| negative_log_likelihood | 1.365915 | 0.261244 | -1.104671 |
| expected_calibration_error | 0.129137 | 0.048266 | -0.080871 |
| brier_score | 0.650423 | 0.128626 | -0.521797 |
| harm_rate | 0.546398 | 0.038290 | -0.508108 |
| rescue_rate | 0.527027 | 0.413793 | -0.113234 |

**Other aggregators / conditions at N=50 (accuracy / delta_accuracy)**

| Condition | A accuracy | B accuracy | A delta | B delta |
|---|---|---|---|---|
| Majority-vote | 0.463785 | 0.913551 | -0.492991 | -0.001752 |
| Confidence-weighted | 0.463785 | 0.914720 | -0.492991 | -0.000584 |
| Original-anchored | 0.471379 | 0.915888 | -0.485397 | 0.000584 |
| BN-adapted | 0.812500 | 0.902453 | -0.144276 | -0.012850 |

**Full N=1,2,5,10,25,50,100 mean-probability curve**

| N | A accuracy | B accuracy | diff (pp) | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.403621 | 0.903621 | +50.00pp | 5.037395 | 0.290222 | -4.747173 |
| 2 | 0.436332 | 0.908294 | +47.20pp | 3.057908 | 0.275545 | -2.782363 |
| 5 | 0.453271 | 0.915888 | +46.26pp | 1.908499 | 0.265167 | -1.643332 |
| 10 | 0.467290 | 0.914136 | +44.68pp | 1.564060 | 0.262916 | -1.301144 |
| 25 | 0.458528 | 0.912967 | +45.44pp | 1.405997 | 0.261257 | -1.144741 |
| 50 | 0.456776 | 0.915304 | +45.85pp | 1.365915 | 0.261244 | -1.104671 |
| 100 | 0.452687 | 0.916472 | +46.38pp | 1.352153 | 0.260683 | -1.091469 |

**Runtime and latency**

- Evaluation runtime: A=1150.687s, B=1157.317s
- Clean latency (s): A=0.031049, B=0.032035

| N | A compute multiplier | B compute multiplier |
|---|---|---|
| 1 | 1.6011 | 1.8872 |
| 2 | 3.5308 | 3.2733 |
| 5 | 8.3207 | 8.2597 |
| 10 | 16.2630 | 16.9542 |
| 25 | 40.6275 | 41.8900 |
| 50 | 80.5703 | 85.8885 |
| 100 | 164.6722 | 170.1705 |

## 5. Per-dataset three-seed descriptive summary

### pathmnist

- **Within-model TTA delta, A (ΔA)**: individual seeds [-0.280388, -0.563275, -0.608956]; mean=-0.484206 (-48.42pp); sample stdev=0.177984 (+17.80pp); min=-0.608956 (-60.90pp); max=-0.280388 (-28.04pp)
- **Within-model TTA delta, B (ΔB)**: individual seeds [-0.011795, 0.067973, 0.022291]; mean=0.026156 (+2.62pp); sample stdev=0.040024 (+4.00pp); min=-0.011795 (-1.18pp); max=0.067973 (+6.80pp)
- **Harm improvement (ΔB − ΔA)**: individual seeds [0.268593, 0.631248, 0.631248]; mean=0.510363 (+51.04pp); sample stdev=0.209379 (+20.94pp); min=0.268593 (+26.86pp); max=0.631248 (+63.12pp)
- **Clean-performance cost (clean_B − clean_A)**: individual seeds [-0.039584, 0.070772, -0.065474]; mean=-0.011429 (-1.14pp); sample stdev=0.072355 (+7.24pp); min=-0.065474 (-6.55pp); max=0.070772 (+7.08pp)

### bloodmnist

- **Within-model TTA delta, A (ΔA)**: individual seeds [-0.308411, -0.346379, -0.5]; mean=-0.384930 (-38.49pp); sample stdev=0.101446 (+10.14pp); min=-0.500000 (-50.00pp); max=-0.308411 (-30.84pp)
- **Within-model TTA delta, B (ΔB)**: individual seeds [0.01285, 0.037383, 0.0]; mean=0.016745 (+1.67pp); sample stdev=0.018993 (+1.90pp); min=0.000000 (+0.00pp); max=0.037383 (+3.74pp)
- **Harm improvement (ΔB − ΔA)**: individual seeds [0.321262, 0.383762, 0.5]; mean=0.401674 (+40.17pp); sample stdev=0.090706 (+9.07pp); min=0.321262 (+32.13pp); max=0.500000 (+50.00pp)
- **Clean-performance cost (clean_B − clean_A)**: individual seeds [-0.036799, -0.025117, -0.041472]; mean=-0.034463 (-3.45pp); sample stdev=0.008424 (+0.84pp); min=-0.041472 (-4.15pp); max=-0.025117 (-2.51pp)

## 6. Descriptive headline

**PathMNIST**:
- mean ΔA = -48.42pp
- mean ΔB = +2.62pp
- mean harm improvement = +51.04pp
- mean clean cost = -1.14pp

**BloodMNIST**:
- mean ΔA = -38.49pp
- mean ΔB = +1.67pp
- mean harm improvement = +40.17pp
- mean clean cost = -3.45pp

- Harm improvement (ΔB − ΔA > 0) is positive in 6/6 paired cells.
- Matched-policy TTA is non-harmful (ΔB >= 0) at N=50 in 5/6 cells; the remaining cell is -1.18pp.

**No claim of statistical significance, generalization, or test-set
confirmation is made or implied by this document.** This is a 3-seed,
2-dataset, single-architecture (small_cnn, BatchNorm), 28px-resolution
descriptive result. No significance test was run and none is implied.
No threshold was selected based on these numbers, and no protocol,
policy, seed, prefix, aggregation, batching, or metric code was changed
as a result of them.

## 7. Seed-0 PathMNIST thermal-throttling episode (runtime-only anomaly)

As documented in full in
`docs/phase2b_validation_evaluation_block_b_canary_audit.md` Section 3,
the seed-0 PathMNIST Block B cell
(`B-pathmnist-28px-batchnorm-policy-matched_mixed-s0`) experienced a
mid-run thermal-throttling episode (CPU-time-to-wall-clock ratio dropped
to ~4% for roughly 49 minutes before recovering to ~100%), stretching its
runtime from an expected ~1.9h to ~3.93h
(`14166.387` s). The attempt completed
normally with `status="completed"` and no `failure_reason`; CPU time never
stopped growing at any monitored interval, confirming continuous forward
progress rather than a hang. **This is recorded as a runtime observation
only -- no amendment or incident row was created**, since the attempt
completed successfully with no scientific-validity concern. All 5
subsequently executed Block B cells (seeds 1/2 PathMNIST, seeds 0/1/2
BloodMNIST) ran at consistently ~100% CPU with no throttling observed.

## 8. Full ledger and directory state

- Total ledger data rows: **34**
- Status tally: completed=31, failed=2, aborted=1
- Distinct Block A completed run IDs: **24**
- Distinct Block B completed run IDs: **6**
- Block C ledger rows: **0**; Block C directories: **0**
- Block D ledger rows: **0**; Block D directories: **0**
- 30 distinct scientifically valid A+B completions (24 Block A + 6 Block B).

Block B completed run IDs:

- `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0`
- `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1`
- `B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2`
- `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0`
- `B-pathmnist-28px-batchnorm-policy-matched_mixed-s1`
- `B-pathmnist-28px-batchnorm-policy-matched_mixed-s2`

## 9. Evaluator-fingerprint cohort disclosure

Per `docs/phase2b_validation_evaluation_evaluator_fingerprint_drift_addendum.md`,
three of the 24 Block A completions
(`A-pathmnist-28px-batchnorm-policy-none-s0/-s1/-s2`) were computed under
an older evaluator fingerprint than the current one
(`7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`),
because the GroupNorm persistence-schema correction
(`docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md`)
changed fingerprint-manifested source files
(`src/when_tta_hurts/validation_evaluation.py`,
`src/when_tta_hurts/evaluation_result_artifacts.py`). That correction's
only reachable code path for BatchNorm cells is a byte-for-byte no-op
(the changed logic only affects GroupNorm cells, whose
`bn_adaptation_microbatches_at_primary_n` was previously `None`); the
three PathMNIST BatchNorm comparators' persisted metrics, predictions,
and metadata are therefore scientifically unaffected by the fix. This is
used in the present document as the historically canonical comparator
for those three cells (as established in
`docs/phase2b_validation_evaluation_block_a_audit.md`). **Strict
current-fingerprint reconciliation of these three cells (rerunning them
under the current fingerprint) remains deferred**, per the explicit
deferral decision recorded in the fingerprint-drift addendum. It is not
performed as part of this document, and no compatibility bypass or
override has been introduced.

The three BloodMNIST BatchNorm Block A comparators
(`A-bloodmnist-28px-batchnorm-policy-none-s0/-s1/-s2`) are
current-fingerprint-compatible, as are all 6 Block B cells.

## 10. Scope and status

**This is a 3-seed, 2-dataset validation-only descriptive comparison,
not a confirmed general finding.** No test split was accessed at any
point across Block A or Block B. No significance test has been run. No
threshold was selected based on these results. No protocol, policy,
seed, prefix, aggregation, batching, or metric code was changed as a
result of any observation documented here. The three PathMNIST Block A
comparators remain in the older-fingerprint cohort pending a deferred
reconciliation. Blocks C and D remain entirely unstarted (zero ledger
rows, zero directories).

