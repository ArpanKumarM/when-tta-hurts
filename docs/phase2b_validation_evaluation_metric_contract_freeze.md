# Phase 2B.4D Metric-Contract Freeze: probability-native metric input contract

This document is Part A/D of the Phase 2B.4D Metric-Contract Correction:
the mechanical adjudication of the frozen mathematical contract (Part A),
and the corrected contract this project freezes going forward (Part D).
No implementation change is made by this document. It is committed
together with the `configs/validation_evaluation.yaml` amendment that
adds `metric_input_contract: probability_native_v1` as a frozen,
identity-participating field. No real evaluation has run under this
document. `evaluation_id 75aa7e37a9fe5454bf8edf6483d676a182d6dde9ff4a3730e4ada7195e09eb9e`
(attempt 3) remains recorded canonical-ineligible, unchanged by this
document (`docs/phase2b_validation_evaluation_metric_contract_incident.md`).

## Part A1: frozen formulas, cited exactly

| Metric/aggregation | Frozen input type | Citation |
|---|---|---|
| Mean-probability aggregation | Per-view **softmax probabilities**, arithmetic mean | `docs/phase2b_protocol.md` line ~123: "**Mean probability:** arithmetic mean of per-view softmax probabilities." |
| Majority vote | Per-view argmax; ties by "highest mean probability among tied classes," then lowest class index | `docs/phase2b_protocol.md` lines ~124-126 |
| Confidence-weighted aggregation | Each view weighted by its "maximum softmax probability"; weights normalized to sum to one; probability vectors averaged | `docs/phase2b_protocol.md` lines ~130-132 |
| Original anchoring | "one clean view plus N augmented views, equal-weight mean probability over N+1 views" | `docs/phase2b_protocol.md` lines ~110-113 (secondary-analyses list) |
| BN-adapted prediction | "mixed policy; mean-probability aggregation" over the BN-adapted model's output | `docs/phase2b_protocol.md` lines ~115-118 |
| Brier score | `mean over samples of [sum over classes of (predicted_probability - one_hot_label)^2]` -- operates directly on `predicted_probability` | `docs/phase2b_protocol.md` lines ~135-140 |
| Accuracy, macro-F1, NLL, ECE, harm rate, rescue rate | "definitions unchanged from `src/when_tta_hurts/metrics.py`" | `docs/phase2b_protocol.md` lines ~146-149 |
| Secondary endpoints (NLL/ECE/Brier/harm/rescue) | "computed from saved per-sample predictions" | `docs/experimental_protocol.md` "Endpoints" section |
| Aggregation return convention | Aggregate functions "return a [N, C] array... in LOG-PROBABILITY space... so it composes directly with metrics.py's softmax-based functions: `softmax(log(p)) == p` exactly" | `src/when_tta_hurts/evaluation/aggregation.py` module docstring (lines 1-12) |
| `metrics.py`'s logits-native functions | `accuracy`/`macro_f1`/`negative_log_likelihood`/`expected_calibration_error`/`brier_score` all take a parameter literally named `logits` and call `softmax(logits)` internally exactly once | `src/when_tta_hurts/metrics.py` (all five function definitions) |
| `docs/statistical_analysis_plan.md` | References "NLL, ECE, Brier" as calibration metrics for standardized-difference reporting; does not specify or permit any additional softmax transform | `docs/statistical_analysis_plan.md` line ~30-31 |
| `configs/validation_evaluation.yaml` | Declares `primary_aggregation: mean_probability` -- names the aggregate a **probability**, not a logit | `configs/validation_evaluation.yaml` |
| Independent metric-reference tests | `tests/test_metrics_independent_validation.py` validates `metrics.py`'s logits-native functions against sklearn/direct PyTorch cross-entropy using genuine logits as input -- never validates the "probabilities-passed-as-logits" call pattern | `tests/test_metrics_independent_validation.py` |

**No frozen document anywhere specifies, permits, or even mentions
applying softmax to an already-aggregated probability distribution.**
Every frozen definition either operates directly on "probability" (Brier,
mean-probability, confidence weights) or is silent on the exact call
convention while explicitly naming its input a probability quantity.

## Part A2: condition-by-condition data-type trace

| Condition | Raw model output | First conversion | Aggregation input | Aggregation output | Input passed to metric function (as persisted) | Softmax count, model logits -> final metric |
|---|---|---|---|---|---|---|
| Clean | logits | `softmax()` once (`clean_probs`) | n/a | n/a | `clean_probs` -> `log()` -> `clean_logp` -> metric fn's internal `softmax()` | **1** (correct: `softmax(log(clean_probs)) == clean_probs`) |
| Mean probability | per-view logits | `softmax()` once per view (inside `aggregation.py`) | per-view probabilities | log-probabilities (`_to_log_probs`) | caller does `softmax(mean_probability(...))` -> genuine probabilities -> passed DIRECTLY (no log-conversion) into metric fn's internal `softmax()` | **2** (defect: one extra) |
| Majority vote | per-view logits | `softmax()` once per view | per-view probabilities (for vote + tie-break) | log(vote-fraction) | same double-softmax pattern as mean probability | **2** (defect) |
| Confidence-weighted | per-view logits | `softmax()` once per view | per-view probabilities + confidence weights (computed entirely in true probability space) | log-probabilities | same double-softmax pattern | **2** (defect, in the metric stage only -- see Part B) |
| Original anchored | clean + per-view logits | `softmax()` once each | clean + per-view probabilities | log-probabilities | same double-softmax pattern | **2** (defect) |
| BN-adapted | adapted-model logits | `softmax()` once (`adapted_probs`) | n/a | n/a | `adapted_probs` passed DIRECTLY (no log-conversion) into metric fn's internal `softmax()` | **2** (defect) |

Verified directly (not inferred from variable names): every intermediate
array at the "aggregation output" stage sums to 1 per row (confirmed via
`predictions.npz`'s persisted arrays, all of which pass
`validate_predictions_arrays()`'s row-sum check); `softmax()` is called
exactly where cited above (grep-verified call sites in
`src/when_tta_hurts/validation_evaluation.py` and
`src/when_tta_hurts/evaluation/aggregation.py`); the root cause is
localized to exactly one function,
`validation_evaluation.py::_per_prefix_metrics()`, which correctly
log-converts `clean_probs` before use but does not apply the same
conversion to `agg_probs`.

## Part A3: hand-calculated proof

`p = [0.9, 0.1]`, true class = 0. Computed directly (not estimated):

| Quantity | Value |
|---|---|
| Correct probability-native NLL (`-log(0.9)`) | `0.10536051565782628` |
| `softmax(p)` | `[0.68997448, 0.31002552]` -- confirms `softmax(p) != p` |
| NLL via `negative_log_likelihood(p, true_class)` (one extra softmax) | `0.37110066594777763` |
| `softmax(softmax(p))` | `[0.59386079, 0.40613921]` |
| NLL via `negative_log_likelihood(softmax(p), true_class)` (two extra softmaxes) | `0.5211103419809864` |
| Correct multiclass Brier from `p` directly | `0.019999999999999997` |
| Brier via `brier_score(p, true_class)` (one extra softmax) | `0.19223164470418624` |
| Brier via `brier_score(softmax(p), true_class)` (two extra) | `0.3298981106137178` |
| `log(p)` | `[-0.10536052, -2.30258509]` |
| `softmax(log(p))` | `[0.9, 0.1]` -- exactly recovers `p` |
| NLL via `negative_log_likelihood(log(p), true_class)` (CORRECT) | `0.1053605156578264` |
| Brier via `brier_score(log(p), true_class)` (CORRECT) | `0.020000000000000025` |
| Confidence `max(p)` | `0.9` |
| Confidence `max(softmax(p))` | `0.6899744811276125` |
| Confidence `max(softmax(softmax(p)))` | `0.5938607931917199` |
| ECE via `log(p)` input (CORRECT) | `0.10000000000000009` |
| ECE via `p` input (one extra softmax) | `0.3100255188723875` |

**Explicit conclusions:**
- `softmax(p) != p` for this nonuniform distribution -- confirmed
  numerically, not asserted.
- If a metric API requires logits but the desired probabilities are `p`,
  an equivalent logit representation is `log(p)` (up to an additive
  constant) -- confirmed: `softmax(log(p)) == p` exactly. **`p` itself is
  not a valid logit-equivalent input.**
- Every condition traced in Part A2 applies softmax **uniformly** in the
  same (wrong) way. Uniform application across all code paths establishes
  internal consistency, not agreement with the frozen formula --
  confirmed by this proof showing the uniformly-applied convention is
  numerically wrong at every point tested.

**Multi-view proof that mean-probability aggregation produces normalized
probabilities before metric calculation:** for a 2-view, 2-class toy
example with view probabilities `[0.8,0.2]` and `[0.6,0.4]`, the mean is
`[0.7,0.3]`, which sums to `1.0` exactly, confirming
`mean_probability()`'s output (after the single, correct
`softmax(log(p))` recovery already performed at every real call site) is
already a genuine, normalized probability distribution requiring no
further softmax.

## Part A4: decision rule applied

Per the frozen decision rule: **the frozen documents (`docs/phase2b_protocol.md`,
`docs/experimental_protocol.md`) define every affected metric directly on
aggregated probabilities** (mean-probability aggregation is defined as an
"arithmetic mean of per-view softmax probabilities"; Brier is defined
directly on "predicted_probability"), **and production applies another
softmax to that already-aggregated probability before computing NLL/ECE/
Brier. The defect is confirmed** -- mechanically, from the formula
contract and a hand-calculated numeric proof, not from observing or
reacting to the accuracy result (which is unaffected and was already
known). No frozen document anywhere explicitly requires or authorizes
this extra softmax as part of the scientific method; there is no
ambiguity to report. A generic statement that `metrics.py`'s functions
"accept logits" does not resolve the issue, because the caller
(`_per_prefix_metrics`) supplies genuine probabilities, not logits or a
valid logit-equivalent (`log(p)`), to those functions for every aggregate
condition.

## Part D: the frozen, corrected contract

```yaml
metric_input_contract: probability_native_v1
```

**Model output:**
```
logits = model(x)
p = softmax(logits)
```
Softmax is applied exactly once, to raw model logits, and never again to
the result.

**Mean probability:**
```
p_mean = mean(p_view, axis=view)
```
`p_mean` is not softmaxed.

**Original anchoring:** uses the already-frozen weighted/mean formula on
probability distributions (clean + N augmented views, equal weight); its
probability output is not softmaxed.

**Confidence weighting:** uses the already-frozen confidence
(`max` per-view softmax probability) and normalization (weights sum to
one) formulas, entirely in true probability space; its final probability
distribution is not softmaxed (the frozen formula does not require it).

**Majority vote:** preserves the frozen voting and tie-break rule exactly
(highest vote count; ties broken by highest mean probability among tied
classes; remaining ties broken by lowest class index). Its output
representation is the vote-fraction distribution (`votes / n_views` per
class), a genuine probability distribution; no undocumented softmax is
added.

**BN adaptation:** the adapted model returns logits; softmax is applied
exactly once, then probability-native metrics are computed from the
result -- symmetric with the clean and mean-probability paths.

**Metrics**, for a probability distribution `p`:
```
accuracy = mean(argmax(p) == y)
macro_f1 = macro_f1(argmax(p), y)
nll = -mean(log(clamp(p[true_class], eps, 1.0)))
brier = mean(sum((p - one_hot(y))^2, axis=class))
ece = frozen 15-bin binning formula, confidence = max(p)
```
The already-frozen epsilon (`1e-12`), ECE bin count (`15`), and
bin-boundary convention (`(lo, hi]`, with the first bin's lower edge
inclusive) are preserved exactly -- unchanged from
`src/when_tta_hurts/metrics.py`'s existing implementation.

**What this freeze does NOT change:** TTA seed (`1306178015`), policy
(`mixed`), registered prefixes (`[1,2,5,10,25,50,100]`), primary N=50
endpoint, batching (`inference_batch_size=256`,
`bn_adaptation_batch_size=256`), BN algorithm
(`sequential_microbatch_v1`), aggregation formulas (unchanged --
`aggregation.py` itself is not modified by this correction; only the
downstream metric-calling convention is), or the test firewall.

Adding `metric_input_contract` to `configs/validation_evaluation.yaml`
changes the file's content hash (`tta_seed_config_sha256`), which is
already a hashed field in `ValidationEvaluationConfig` -- this alone is
sufficient to give the next real evaluation a new `evaluation_id`,
through the existing, unchanged identity mechanism.
