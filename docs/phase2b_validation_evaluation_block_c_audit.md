# Phase 2B Block C Validation Evaluation: Full Audit (3/3 cells)

**Recorded: 2026-08-21.** This document records the closure of Phase 2B
Block C validation evaluation: all 3 positive-control cells
(DermaMNIST 28px, ResNet-18, BatchNorm, seeds 0/1/2), completed and
independently verified. It supersedes nothing in the Block C canary
audit (`docs/phase2b_validation_evaluation_block_c_canary_audit.md`,
which remains the authoritative record of seed 0's own detailed report);
this document adds seeds 1 and 2 and the complete 3-seed descriptive
summary. All numbers below are generated mechanically from persisted
`metrics.json`/`metadata.json`/`status.json`/ledger artifacts -- none
are hand-transcribed. **This is a validation-only result.** No test
split was accessed at any point. No protocol, threshold, seed, prefix,
aggregation, batching, or metric code was changed as a result of any
observation in this document.

## 1. Three-cell identity, checkpoint, and evaluation mapping

| Seed | Training run | Training attempt | Checkpoint hash | Evaluation attempt | Evaluation ID | Fingerprint match current |
|---|---|---|---|---|---|---|
| 0 | `C-dermamnist-28px-resnet18-batchnorm-policy-none-s0` | 1 | `bd529f57be5f0604...` | 1 | `e4410d838c8c843b...` | True |
| 1 | `C-dermamnist-28px-resnet18-batchnorm-policy-none-s1` | 1 | `ab087ce6cf7dfc7a...` | 1 | `f158726e0e638ac4...` | True |
| 2 | `C-dermamnist-28px-resnet18-batchnorm-policy-none-s2` | 1 | `44881dca162455ee...` | 1 | `25053eb2b011adb4...` | True |

All 3 cells: evaluation attempt 1, current-fingerprint-compatible
(`7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`).

## 2. Architecture, class-count, and BatchNorm verification

`build_resnet18_small_input(num_classes=7)`: 11,172,423 total parameters,
20 `BatchNorm2d` layers, 7 output classes -- identical across all
3 seeds (confirmed via independent model construction, matching the
architecture audit in `docs/phase2b_block_c_audit.md` Section 2, which
independently confirmed the same parameter count via checkpoint reload
for all 3 training attempts). Stem `conv1` replaced with a 3x3 stride-1
convolution and `maxpool` replaced with `nn.Identity()` ("no initial
pool", per the source paper's spec).

| Seed | bn_adaptation_applicable | bn_adaptation_microbatches_at_primary_n |
|---|---|---|
| 0 | True | 200 |
| 1 | True | 200 |
| 2 | True | 200 |

## 3. Independent verification evidence (all 3 cells)

For every one of the 3 Block C cells, the following were independently
re-derived from persisted `predictions.npz` and cross-checked against
`metrics.json`, with **zero mismatches** in all 3 cells:

- **Manifest integrity**: `verify_evaluation_artifact_manifest()` -- OK for all 3.
- **Full semantic recomputation**: clean metrics + all 7 registered
  prefixes x 3 `naive_tta` aggregators, recomputed independently via
  `compute_metrics_from_probabilities()`/`_recompute_all_conditions_from_predictions()`
  -- **0 mismatches** across all 3 cells combined, within the frozen `1e-6` tolerance.
- **Probability validity**: `clean_probs`, `view_probs`, `bn_adapted_probs`
  all finite, within `[0,1]`, row-normalized in all 3 cells.
- **Dataset checksum**: `checksum_verified=True`, `resized=False` in all 3 cells.
- **Checkpoint binding**: `metadata.checkpoint_hash` matches
  `resolve_canonical_training_completion()`'s canonical training result
  exactly in all 3 cells (True).
- **Sample/label alignment**: `sample_indices` unique, contiguous,
  same length as `labels`, in all 3 cells (True).
- **BatchNorm contract**: `bn_adaptation_applicable=True` in all 3 cells,
  with `bn_adaptation_microbatches_at_primary_n=200`,
  `bn_adapted_probs`/`bn_adapted_prefix_sequence` present and matching the
  frozen `PREFIX_SEQUENCE=[1,2,5,10,25,50,100]` in all 3 cells.
- **Frozen provenance**: `prefix_sequence`, `tta_seed=1306178015`, and all
  batching fields match `configs/validation_evaluation.yaml` exactly in all 3 cells.
- **Sole canonical-compatible completion**: each of the 3 cells is the
  sole ledger row for its `training_run_id`, `status=completed`,
  `test_metrics_observed=False` (True), `confirmatory=True`, `evaluation_attempt=1`.
- **No Block A/B/D interference**: Block A remains 24, Block B remains 6,
  Block D has zero ledger rows and zero directories, confirmed mechanically.
- **Test split untouched**: no test-split access occurred at any point
  across Block C evaluation.

## 4. Per-seed clean metrics

| Seed | Accuracy | Macro-F1 | NLL | ECE | Brier |
|---|---|---|---|---|---|
| 0 | 0.748754 | 0.444122 | 0.736957 | 0.037316 | 0.363018 |
| 1 | 0.733799 | 0.383391 | 0.781769 | 0.092943 | 0.381270 |
| 2 | 0.756730 | 0.478087 | 0.693988 | 0.058167 | 0.353343 |

## 5. Complete N=1,2,5,10,25,50,100 curves -- all conditions, all seeds

### Seed 0

**Naive mean-probability**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.633101 | -11.57pp | 0.287884 | 1.112323 | 0.072305 | 0.507608 | 0.206391 | 0.154762 |
| 2 | 0.687936 | -6.08pp | 0.324574 | 0.947679 | 0.025433 | 0.451745 | 0.142477 | 0.182540 |
| 5 | 0.702891 | -4.59pp | 0.331428 | 0.845379 | 0.078232 | 0.413993 | 0.110519 | 0.146825 |
| 10 | 0.720837 | -2.79pp | 0.335867 | 0.832572 | 0.091019 | 0.404819 | 0.083888 | 0.138889 |
| 25 | 0.726820 | -2.19pp | 0.348976 | 0.817230 | 0.113119 | 0.396517 | 0.078562 | 0.146825 |
| 50 | 0.721834 | -2.69pp | 0.347846 | 0.811782 | 0.112820 | 0.394470 | 0.079893 | 0.130952 |
| 100 | 0.733799 | -1.50pp | 0.363845 | 0.808742 | 0.112354 | 0.393425 | 0.065246 | 0.134921 |

**Majority-vote**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.633101 | -11.57pp | 10.137802 |
| 2 | 0.602193 | -14.66pp | 7.273244 |
| 5 | 0.690927 | -5.78pp | 4.534187 |
| 10 | 0.702891 | -4.59pp | 3.531235 |
| 25 | 0.723829 | -2.49pp | 2.787645 |
| 50 | 0.722832 | -2.59pp | 2.258491 |
| 100 | 0.732802 | -1.60pp | 2.179836 |

**Confidence-weighted**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.633101 | -11.57pp | 1.112323 |
| 2 | 0.684945 | -6.38pp | 0.940106 |
| 5 | 0.705882 | -4.29pp | 0.830524 |
| 10 | 0.720837 | -2.79pp | 0.814037 |
| 25 | 0.725823 | -2.29pp | 0.794877 |
| 50 | 0.721834 | -2.69pp | 0.788142 |
| 100 | 0.728814 | -1.99pp | 0.784530 |

**Original-anchored**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.729811 | -1.89pp | 0.802587 |
| 2 | 0.724826 | -2.39pp | 0.810059 |
| 5 | 0.719840 | -2.89pp | 0.804743 |
| 10 | 0.728814 | -1.99pp | 0.812308 |
| 25 | 0.728814 | -1.99pp | 0.810048 |
| 50 | 0.723829 | -2.49pp | 0.808204 |
| 100 | 0.735793 | -1.30pp | 0.806961 |

**BN-adapted**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.756730 | +0.80pp | 0.697334 | 0.030954 | 0.346399 | 0.017310 | 0.083333 |
| 2 | 0.758724 | +1.00pp | 0.683550 | 0.036662 | 0.341314 | 0.029294 | 0.126984 |
| 5 | 0.759721 | +1.10pp | 0.676706 | 0.040751 | 0.339840 | 0.042610 | 0.170635 |
| 10 | 0.753739 | +0.50pp | 0.679088 | 0.047821 | 0.341525 | 0.049268 | 0.166667 |
| 25 | 0.756730 | +0.80pp | 0.677939 | 0.044125 | 0.341111 | 0.046605 | 0.170635 |
| 50 | 0.755733 | +0.70pp | 0.678237 | 0.047909 | 0.341065 | 0.047936 | 0.170635 |
| 100 | 0.752742 | +0.40pp | 0.678152 | 0.042970 | 0.341264 | 0.050599 | 0.166667 |

**Runtime and latency**

- Runtime: 823.4756288528442 s (~13.7 min)
- Clean latency: 0.304979 s
- N=50 compute multiplier: 53.1059

### Seed 1

**Naive mean-probability**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.670987 | -6.28pp | 0.241198 | 1.080486 | 0.124448 | 0.470964 | 0.108696 | 0.063670 |
| 2 | 0.692921 | -4.09pp | 0.247263 | 0.942379 | 0.082193 | 0.439512 | 0.078804 | 0.063670 |
| 5 | 0.696909 | -3.69pp | 0.248516 | 0.880772 | 0.070975 | 0.426112 | 0.069293 | 0.052434 |
| 10 | 0.701894 | -3.19pp | 0.276531 | 0.865392 | 0.064961 | 0.419665 | 0.061141 | 0.048689 |
| 25 | 0.694915 | -3.89pp | 0.247025 | 0.854080 | 0.068271 | 0.414554 | 0.070652 | 0.048689 |
| 50 | 0.691924 | -4.19pp | 0.241118 | 0.848555 | 0.069334 | 0.413769 | 0.077446 | 0.056180 |
| 100 | 0.692921 | -4.09pp | 0.244232 | 0.845519 | 0.069122 | 0.412517 | 0.072011 | 0.044944 |

**Majority-vote**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.670987 | -6.28pp | 9.090964 |
| 2 | 0.676969 | -5.68pp | 7.570559 |
| 5 | 0.703888 | -2.99pp | 6.517132 |
| 10 | 0.700897 | -3.29pp | 5.764575 |
| 25 | 0.698903 | -3.49pp | 4.531789 |
| 50 | 0.701894 | -3.19pp | 4.109605 |
| 100 | 0.704885 | -2.89pp | 3.878490 |

**Confidence-weighted**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.670987 | -6.28pp | 1.080486 |
| 2 | 0.693918 | -3.99pp | 0.946721 |
| 5 | 0.698903 | -3.49pp | 0.891673 |
| 10 | 0.697906 | -3.59pp | 0.879459 |
| 25 | 0.693918 | -3.99pp | 0.869609 |
| 50 | 0.689930 | -4.39pp | 0.864601 |
| 100 | 0.692921 | -4.09pp | 0.860899 |

**Original-anchored**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.711864 | -2.19pp | 0.841803 |
| 2 | 0.709870 | -2.39pp | 0.838058 |
| 5 | 0.703888 | -2.99pp | 0.844955 |
| 10 | 0.701894 | -3.19pp | 0.847106 |
| 25 | 0.697906 | -3.59pp | 0.847233 |
| 50 | 0.691924 | -4.19pp | 0.845188 |
| 100 | 0.692921 | -4.09pp | 0.843871 |

**BN-adapted**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.732802 | -0.10pp | 0.740914 | 0.084489 | 0.365832 | 0.013587 | 0.033708 |
| 2 | 0.734796 | +0.10pp | 0.723693 | 0.077463 | 0.359173 | 0.020380 | 0.059925 |
| 5 | 0.734796 | +0.10pp | 0.708960 | 0.065965 | 0.353723 | 0.035326 | 0.101124 |
| 10 | 0.736790 | +0.30pp | 0.704193 | 0.064832 | 0.351883 | 0.040761 | 0.123596 |
| 25 | 0.742772 | +0.90pp | 0.701303 | 0.055953 | 0.350437 | 0.039402 | 0.142322 |
| 50 | 0.743769 | +1.00pp | 0.701615 | 0.057401 | 0.350363 | 0.036685 | 0.138577 |
| 100 | 0.739781 | +0.60pp | 0.702667 | 0.062234 | 0.351317 | 0.043478 | 0.142322 |

**Runtime and latency**

- Runtime: 821.6558239459991 s (~13.7 min)
- Clean latency: 0.305218 s
- N=50 compute multiplier: 53.0280

### Seed 2

**Naive mean-probability**

| N | Accuracy | Delta (pp) | Macro-F1 | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.649053 | -10.77pp | 0.303460 | 1.051596 | 0.115917 | 0.481276 | 0.198946 | 0.176230 |
| 2 | 0.689930 | -6.68pp | 0.308616 | 0.900017 | 0.050344 | 0.428244 | 0.151515 | 0.196721 |
| 5 | 0.709870 | -4.69pp | 0.325978 | 0.798024 | 0.053345 | 0.391850 | 0.119895 | 0.180328 |
| 10 | 0.722832 | -3.39pp | 0.350086 | 0.786292 | 0.056284 | 0.383583 | 0.108037 | 0.196721 |
| 25 | 0.737787 | -1.89pp | 0.375497 | 0.761292 | 0.063438 | 0.370661 | 0.096179 | 0.221311 |
| 50 | 0.729811 | -2.69pp | 0.358757 | 0.756081 | 0.070889 | 0.369207 | 0.102767 | 0.209016 |
| 100 | 0.734796 | -2.19pp | 0.371789 | 0.751617 | 0.068484 | 0.366490 | 0.097497 | 0.213115 |

**Majority-vote**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.649053 | -10.77pp | 9.697028 |
| 2 | 0.602193 | -15.45pp | 7.001216 |
| 5 | 0.694915 | -6.18pp | 4.546951 |
| 10 | 0.714855 | -4.19pp | 3.185534 |
| 25 | 0.731805 | -2.49pp | 2.298909 |
| 50 | 0.730808 | -2.59pp | 1.966025 |
| 100 | 0.737787 | -1.89pp | 1.819475 |

**Confidence-weighted**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.649053 | -10.77pp | 1.051596 |
| 2 | 0.691924 | -6.48pp | 0.893727 |
| 5 | 0.714855 | -4.19pp | 0.789586 |
| 10 | 0.722832 | -3.39pp | 0.775607 |
| 25 | 0.739781 | -1.69pp | 0.749109 |
| 50 | 0.729811 | -2.69pp | 0.743521 |
| 100 | 0.733799 | -2.29pp | 0.739636 |

**Original-anchored**

| N | Accuracy | Delta (pp) | NLL |
|---|---|---|---|
| 1 | 0.736790 | -1.99pp | 0.748655 |
| 2 | 0.734796 | -2.19pp | 0.758937 |
| 5 | 0.725823 | -3.09pp | 0.754627 |
| 10 | 0.727817 | -2.89pp | 0.763110 |
| 25 | 0.742772 | -1.40pp | 0.752863 |
| 50 | 0.731805 | -2.49pp | 0.751884 |
| 100 | 0.735793 | -2.09pp | 0.749565 |

**BN-adapted**

| N | Accuracy | Delta (pp) | NLL | ECE | Brier | Harm | Rescue |
|---|---|---|---|---|---|---|---|
| 1 | 0.746760 | -1.00pp | 0.689653 | 0.061251 | 0.349909 | 0.023715 | 0.032787 |
| 2 | 0.748754 | -0.80pp | 0.695039 | 0.066264 | 0.350594 | 0.032938 | 0.069672 |
| 5 | 0.743769 | -1.30pp | 0.712088 | 0.071821 | 0.355560 | 0.050066 | 0.102459 |
| 10 | 0.742772 | -1.40pp | 0.717803 | 0.078720 | 0.357552 | 0.057971 | 0.122951 |
| 25 | 0.741775 | -1.50pp | 0.718425 | 0.079296 | 0.357678 | 0.059289 | 0.122951 |
| 50 | 0.741775 | -1.50pp | 0.717685 | 0.080823 | 0.357186 | 0.056653 | 0.114754 |
| 100 | 0.740778 | -1.60pp | 0.720812 | 0.076447 | 0.358966 | 0.061924 | 0.127049 |

**Runtime and latency**

- Runtime: 823.6800479888916 s (~13.7 min)
- Clean latency: 0.305007 s
- N=50 compute multiplier: 52.9670

## 6. Three-seed descriptive summary

| Metric | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | Min | Max |
|---|---|---|---|---|---|---|---|
| Clean accuracy | 0.749 | 0.734 | 0.757 | 0.746 | 0.012 | 0.734 | 0.757 |
| Clean macro-F1 | 0.444 | 0.383 | 0.478 | 0.435 | 0.048 | 0.383 | 0.478 |
| Naive mean-probability delta @50 | -2.69pp | -4.19pp | -2.69pp | -3.19pp | +0.86pp | -4.19pp | -2.69pp |
| Majority-vote delta @50 | -2.59pp | -3.19pp | -2.59pp | -2.79pp | +0.35pp | -3.19pp | -2.59pp |
| Confidence-weighted delta @50 | -2.69pp | -4.39pp | -2.69pp | -3.26pp | +0.98pp | -4.39pp | -2.69pp |
| Original-anchored delta @50 | -2.49pp | -4.19pp | -2.49pp | -3.06pp | +0.98pp | -4.19pp | -2.49pp |
| BN-adapted delta @50 | +0.70pp | +1.00pp | -1.50pp | +0.07pp | +1.36pp | -1.50pp | +1.00pp |
| Runtime (s) | 823.476 | 821.656 | 823.680 | 822.937 | 1.114 | 821.656 | 823.680 |

### Naive mean-probability delta by N, across seeds

| N | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD |
|---|---|---|---|---|---|
| 1 | -11.57pp | -6.28pp | -10.77pp | -9.54pp | +2.85pp |
| 2 | -6.08pp | -4.09pp | -6.68pp | -5.62pp | +1.36pp |
| 5 | -4.59pp | -3.69pp | -4.69pp | -4.32pp | +0.55pp |
| 10 | -2.79pp | -3.19pp | -3.39pp | -3.12pp | +0.30pp |
| 25 | -2.19pp | -3.89pp | -1.89pp | -2.66pp | +1.08pp |
| 50 | -2.69pp | -4.19pp | -2.69pp | -3.19pp | +0.86pp |
| 100 | -1.50pp | -4.09pp | -2.19pp | -2.59pp | +1.34pp |

### BN-adapted delta by N, across seeds

| N | Seed 0 | Seed 1 | Seed 2 | Mean | Sample SD | # positive |
|---|---|---|---|---|---|---|
| 1 | +0.80pp | -0.10pp | -1.00pp | -0.10pp | +0.90pp | 1/3 |
| 2 | +1.00pp | +0.10pp | -0.80pp | +0.10pp | +0.90pp | 2/3 |
| 5 | +1.10pp | +0.10pp | -1.30pp | -0.03pp | +1.20pp | 2/3 |
| 10 | +0.50pp | +0.30pp | -1.40pp | -0.20pp | +1.04pp | 2/3 |
| 25 | +0.80pp | +0.90pp | -1.50pp | +0.07pp | +1.35pp | 2/3 |
| 50 | +0.70pp | +1.00pp | -1.50pp | +0.07pp | +1.36pp | 2/3 |
| 100 | +0.40pp | +0.60pp | -1.60pp | -0.20pp | +1.21pp | 2/3 |

## 7. Comparison with the source paper's external ~+1.6pp reference

**Naive mean-probability TTA @ N=50** (the frozen primary prefix, matching
the source paper's own reported "+1.6pp at N=50 views" condition per
`docs/phase2b_block_c_audit.md`) is negative in **all three seeds**:
-2.69pp, -4.19pp, and -2.69pp; mean -3.19pp,
sample SD +0.86pp. **The source paper's reported positive
~+1.6pp effect was not reproduced under our frozen operationalization.**
**This does not establish that the source paper is incorrect** -- it is
a statement about non-reproduction under this specific operationalization,
not a claim about the ground truth of their reported result.

**Material protocol/sample differences (drawn only from this project's
frozen documentation, not invented):**

- This evaluation runs exclusively on the **validation split**
  (`docs/experimental_protocol.md`'s test firewall: the test set is
  touched only once, at the very end, under a separate frozen-config
  check -- never during Block C). If the source paper's reported +1.6pp
  was measured on its test set, that is a different sample than ours.
- `docs/phase2b_block_c_audit.md` Section 2 confirms our ResNet-18 uses
  `weights=None` (no pretrained initialization), a 3x3 stride-1 stem
  `conv1`, and `maxpool` replaced with `nn.Identity()` ("no initial
  pool," per the source paper's own spec) -- a small-input ResNet-18
  variant, not the standard ImageNet-stem ResNet-18.
- Naive TTA here uses the frozen deterministic 100-view bank with
  `PREFIX_SEQUENCE=[1,2,5,10,25,50,100]` and mean-probability aggregation
  as the primary condition; the exact augmentation-view-generation
  policy, view count, and aggregation formula used to produce the source
  paper's reported figure are not independently confirmed here beyond
  the shared "N=50" reference point.
- Original-anchoring and BN-adaptation are, per `docs/research_plan.md`,
  **the source paper's own Appendix B conditions** -- the +1.6pp
  headline figure most plausibly corresponds to the paper's main
  naive-TTA condition, not to either of these two secondary conditions;
  comparing our naive result to their headline number is the intended
  comparison, but comparing our BN-adapted/anchored results to that same
  +1.6pp figure would not be apples-to-apples even in principle.

**BN-adaptation is mixed across seeds**:
+0.70pp, +1.00pp, and -1.50pp; mean +0.07pp, sample
SD +1.36pp (larger than the mean itself). **BN adaptation is
therefore seed-sensitive and inconclusive on this 3-seed sample -- not a
reliable rescue, and not a reproduction of naive TTA** (BN-adaptation is
a materially different procedure that adapts the model's internal
normalization statistics, never touching the naive/anchored aggregation
conditions above).

## 8. Scope and status

**All conclusions in this document remain validation-stage and
descriptive.** This is a 3-seed, single-architecture (`resnet18`,
BatchNorm), single-dataset (`dermamnist`, 28px) result. No significance
test has been run and none is implied. No threshold was selected based
on these numbers. No protocol, policy, seed, prefix, aggregation,
batching, evaluator, or metric code was changed as a result of any
observation documented here. No test split was accessed at any point.
Block D remains entirely unstarted (zero ledger rows, zero directories).

