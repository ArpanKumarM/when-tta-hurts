# Phase 2B Block B Canary: PathMNIST 28px BatchNorm, matched-augmentation training

**Recorded: 2026-08-20.** This document records the first Block B
(policy-matching) validation-evaluation completion: whether training
with the same augmentation policy used by TTA reduces the severe
TTA-induced accuracy collapse observed in normally-trained models. This
is a **single-seed validation canary**, not a confirmed result. No
protocol, threshold, policy, seed, prefix, aggregation, batching, or
metric code was changed as a result of this observation. No test split
was accessed at any point.

## 1. Exact identity

| Field | Value |
|---|---|
| Training run | `B-pathmnist-28px-batchnorm-policy-matched_mixed-s0` |
| Training policy | `matched_to_approved_tta_policy` |
| Canonical training attempt / checkpoint hash | `1` / `f9d06b302a5a0a737e0476a01fa88cb1c309243f862de0f8f1a45e37ec88c47f` |
| Evaluation attempt | `1` |
| Evaluation ID / evaluation-config hash | `77f63ffdb92258f6ec24eb54785aa5dbf5b9290b117c444f1190a32f562258bd` |
| Evaluator fingerprint | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` (current) |
| Dataset / resolution / split | `pathmnist` / 28 / `validation`, `n_validation_samples=10004` |
| Dataset checksum (expected == actual) | `a8b06965200029087d5bd730944a56c1`, `resized=False` |
| Frozen TTA seed | `1306178015` |
| Metric-input contract | `probability_native_v1` |
| Batching | `inference_batch_size=256`, `bn_adaptation_batch_size=256`, `bn_adaptation_algorithm=sequential_microbatch_v1`, `bn_adaptation_applicable=True`, `bn_adaptation_microbatches_at_primary_n=2000` |
| Source commit | `057424e9e9941e8cffad351e3054375e8d2b5180` |
| Runtime | `14166.386969089508` s (~3.93h; see Section 5 -- includes a mid-run thermal-throttling episode) |

**Matched Block A comparator**: `A-pathmnist-28px-batchnorm-policy-none-s0`
(attempt 4, `evaluation_id=e59debe937108abf956f9340621f306e5af190ae445dd189bb2572361fa0a2f4`).
Identical dataset, resolution, architecture (`small_cnn`), normalization
(`batchnorm`), and seed (`0`) -- differs **only** in `training_policy`
(`none` vs `matched_to_approved_tta_policy`).

### Artifact manifest

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `predictions.npz` | 39,057,252 | `18cc754f1e4898db086fc8257fafe88384fef4a191548e2a64e4aa89c822691b` |
| `metrics.json` | 16,352 | `fffd5ae250b95bfdf677e9ccff2f8bdf72100870b63f6c1985cf752c077af8ee` |
| `metadata.json` | 4,729 | `c6c239dba3812141a5eb57b264b992a58337d788427ecf779868f3da895abebf` |
| `view_manifest.json` | 99,517 | `764f531cb816fa47cced92a7faa40090489bf46143bf39a92a31a0d8a7385b1e` |

## 2. Independent verification (all passed)

- **Manifest**: `verify_evaluation_artifact_manifest()` -- OK.
- **Full semantic recomputation**: clean metrics + all 7 registered
  prefixes x 4 aggregators (mean-probability, majority-vote,
  confidence-weighted, original-anchored) x all metrics + all 7
  BN-adapted prefixes, recomputed independently from persisted
  `predictions.npz` via `compute_metrics_from_probabilities()` (never
  calls softmax) -- **zero mismatches** against `metrics.json`, within
  the frozen `1e-6` tolerance.
- **Probability validity**: `clean_probs`, `view_probs`,
  `bn_adapted_probs` all finite, within `[0,1]`, row-normalized.
- **Dataset checksum**: verified, `resized=False`.
- **Checkpoint binding**: `metadata.checkpoint_hash` matches
  `resolve_canonical_training_completion()`'s canonical training
  result exactly.
- **Sample/label alignment**: `sample_indices` unique, contiguous
  (`0..10003`), same length as `labels`.
- **BatchNorm contract**: `bn_adaptation_applicable=True`,
  `bn_adaptation_microbatches_at_primary_n=2000` (positive),
  `bn_adapted_probs` shape `(7, 10004, 9)`,
  `bn_adapted_prefix_sequence=[1,2,5,10,25,50,100]` -- exact match to
  the frozen `PREFIX_SEQUENCE`.
- **Frozen provenance**: `prefix_sequence=[1,2,5,10,25,50,100]`,
  `tta_seed=1306178015`, batching fields all match frozen values in
  `configs/validation_evaluation.yaml`.
- **Sole canonical-compatible completion**: `check_evaluation_skip()`
  resolves this attempt as the sole match under the current evaluator
  fingerprint -- no ambiguity, no conflict.
- **`test_metrics_observed=False`** on the ledger row.
- **No Block A/C/D interference**: two spot-checked Block A
  `predictions.npz` hashes confirmed byte-identical to their originally
  recorded values; zero Block C/D ledger rows or directories exist.

## 3. Runtime: thermal-throttling episode (runtime-only anomaly, no incident)

The attempt completed normally on its first and only try, with
`status="completed"` and no `failure_reason`. Mid-run, the process's
CPU-time-to-wall-clock ratio dropped substantially (observed live: CPU
time grew from ~36 minutes to ~38 minutes over roughly 49 minutes of
wall-clock time, i.e. ~4% marginal CPU utilization during that window)
before recovering to full utilization (~100% CPU) for the remainder of
the run. This machine had been running near-continuous MPS-bound
compute for approximately 20 hours by this point in the research
session (all of Block A's 24 cells plus this canary), making thermal
throttling the most plausible explanation. Live monitoring confirmed:

- The process's CPU time **never stopped growing** across any
  monitoring interval -- confirming continuous forward progress, not a
  hang or deadlock.
- No error, exception, or non-zero exit occurred at any point.
- The process was **not** killed, interrupted, or restarted.
- The final artifacts passed every integrity check in Section 2 with
  zero anomalies.

Because the attempt completed successfully with no failure and no
scientific-validity concern, **this is recorded as a runtime
observation only -- no amendment or incident row is created.** The
elevated runtime (~3.93h vs. the ~1.9h matched Block A comparator) is
attributable entirely to hardware thermal state, not to any change in
the amount of computation performed (identical prefix sequence, seed,
batching, and aggregation formulas as every other cell).

## 4. Complete controlled A-versus-B comparison

### (a) Effect on clean performance

| Metric | A (`policy=none`) | B (`policy=matched_mixed`) | B − A |
|---|---|---|---|
| Accuracy | 0.7388044782087165 | 0.6992203118752499 | **−0.0395842** |
| Macro-F1 | 0.7243569247933442 | 0.6799518354744584 | −0.0444051 |
| NLL | 0.91493159532547 | 0.8419130444526672 | −0.0730186 |
| ECE | 0.08999954740770431 | 0.033867885979389195 | −0.0561316 |
| Brier | 0.381816953275236 | 0.409923082169106 | +0.0281061 |

**Matched-augmentation training costs ~3.96pp clean accuracy**, but
substantially improves calibration (ECE −5.61pp, NLL −0.073).

### (d) Within-model TTA delta (own clean-to-TTA drop, N=50 mean-probability)

| | Clean accuracy | TTA@50 accuracy | Delta (TTA − clean) |
|---|---|---|---|
| A (`policy=none`) | 0.7388044782087165 | 0.45841663334666133 | **−0.2803878** |
| B (`policy=matched_mixed`) | 0.6992203118752499 | 0.6874251497005988 | **−0.0117952** |

### (b) Effect on TTA robustness -- `delta_B − delta_A`

`−0.0117952 − (−0.2803878) = +0.2685926` → **TTA harm is ~26.86pp less
severe in B.** This matches the magnitude reported in the acceptance
message.

### (c) Absolute TTA performance at N=50 mean-probability

| Metric | A | B | B − A |
|---|---|---|---|
| Accuracy | 0.4584166 | 0.6874251 | +0.2290085 |
| Macro-F1 | 0.4167439 | 0.6782163 | +0.2614724 |
| NLL | 1.7255840 | 0.8748672 | −0.8507168 |
| ECE | 0.0631116 | 0.0649162 | +0.0018046 |
| Brier | 0.6988639 | 0.4313392 | −0.2675247 |
| Harm rate | 0.4295765 | 0.1636876 | −0.2658889 |
| Rescue rate | 0.1416000 | 0.3413094 | +0.1997094 |

### Other aggregators at N=50 (accuracy / delta_accuracy)

| Aggregator | A accuracy | B accuracy | A delta_accuracy | B delta_accuracy |
|---|---|---|---|---|
| Majority-vote | 0.4348260 | 0.6851259 | −0.3039784 | −0.0140944 |
| Confidence-weighted | 0.4619152 | 0.6908238 | −0.2768892 | −0.0083965 |
| Original-anchored | 0.4790084 | 0.6907237 | −0.2597961 | −0.0084966 |
| BN-adapted | 0.5780688 | 0.6663327 | −0.1607357 | −0.0328876 |

Every aggregator shows the same pattern: B's TTA delta is far closer to
zero than A's. BN-adaptation shows the smallest relative gain, since it
already substantially mitigates harm in model A on its own, leaving
less residual harm for matched-policy training to additionally
remove.

### Full N=1,2,5,10,25,50,100 mean-probability curve

| N | A accuracy | B accuracy | diff | A NLL | B NLL | diff |
|---|---|---|---|---|---|---|
| 1 | 0.3069772 | 0.6423431 | +0.3353659 | 5.0789886 | 0.9836635 | −4.0953251 |
| 2 | 0.3503599 | 0.6593363 | +0.3089764 | 3.2727470 | 0.9333799 | −2.3393671 |
| 5 | 0.3804478 | 0.6784286 | +0.2979808 | 2.1654644 | 0.8947780 | −1.2706864 |
| 10 | 0.4091363 | 0.6828269 | +0.2736906 | 1.8851244 | 0.8843699 | −1.0007545 |
| 25 | 0.4412235 | 0.6872251 | +0.2460016 | 1.7611046 | 0.8772416 | −0.8838630 |
| **50** | **0.4584166** | **0.6874251** | **+0.2290085** | **1.7255840** | **0.8748672** | **−0.8507168** |
| 100 | 0.4694122 | 0.6899240 | +0.2205118 | 1.7069746 | 0.8733258 | −0.8336488 |

B's accuracy exceeds A's at **every single prefix**, with the gap
narrowing at higher N as A's own TTA partially recovers with more
views while B is already close to saturated near its clean accuracy.

### Latency and compute multiplier

| | A | B |
|---|---|---|
| Clean latency (s) | 0.181776501 | 0.180592 |
| N=1 multiplier | 1.1041 | 1.0974 |
| N=2 multiplier | 2.2032 | 2.2135 |
| N=5 multiplier | 5.7978 | 6.3515 |
| N=10 multiplier | 11.0368 | 11.2142 |
| N=25 multiplier | 27.6952 | 28.8745 |
| N=50 multiplier | 55.2723 | 55.5014 |
| N=100 multiplier | 110.7855 | 112.0600 |

Latency and compute cost are essentially unaffected by the training
policy (within ~1-2% at every N) -- the augmentation-policy change
affects model weights, not inference cost.

## 5. Scope and status

**This is a single-seed validation canary for one dataset/resolution/
architecture cell.** It does not establish a general claim about
matched-augmentation training, was not used to select any threshold,
and did not trigger any change to the frozen protocol, policy, seed,
prefixes, aggregation, batching, or metric code. Confirmation requires
completing the remaining 5 Block B cells (2 more PathMNIST seeds, 3
BloodMNIST seeds) and comparing against their matched Block A
counterparts, per the preregistered Block B comparison design. No
significance test has been run and none is implied by this document.
