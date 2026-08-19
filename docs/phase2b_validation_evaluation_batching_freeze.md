# Phase 2B.4D Batching Freeze: bounded-memory evaluation operationalization

This document is Parts 2-4 of the Phase 2B.4D OOM correction: a memory
audit of the complete evaluation path (Part 2), a rigorous analysis of
whether bounded-memory BN adaptation is mathematically equivalent to the
original single-batch implementation (Part 3 -- it is not, and this is
stated honestly rather than asserted away), and the frozen bounded-memory
operationalization this project adopts (Part 4). No implementation change
is made by this document. It is committed together with the
`configs/validation_evaluation.yaml` amendment that freezes the batching
fields as scientific/operational inputs, before any source code changes.

No real evaluation result has been observed. The frozen seed, policy,
prefixes, N=50 primary endpoint, aggregations, and metric formulas are
unchanged.

## Part 2: Memory audit of the complete evaluation path

Every path materializing data proportional to
`validation_samples x number_of_views x image_size` was inspected
directly in `src/when_tta_hurts/validation_evaluation.py`'s
`compute_validation_evaluation()` and `compute_evaluation_latency_report()`
(the only two functions in the production path that touch image tensors
at scale -- `evaluation/aggregation.py`, `metrics.py`, and the
persistence layer operate exclusively on already-reduced probability
arrays, never on raw images).

Reference sizes used below: a single float32 RGB image tensor is
`3 x R x R x 4` bytes (28px = 9,408 B; 64px = 49,152 B; 128px =
196,608 B). The largest registered validation split is PathMNIST's,
10,004 samples (BloodMNIST: 1,712; DermaMNIST: 1,003, 28px only).
"Full split as one batch" sizes below use PathMNIST's 10,004 to represent
the worst case; BloodMNIST/DermaMNIST scale proportionally smaller.

| # | Path | Largest tensor/list materialized (current code) | Theoretical shape | Approx. bytes @28px / @64px / @128px | CPU or MPS | Grows with N? | Can exceed validated 256-batch? |
|---|---|---|---|---|---|---|---|
| 1 | Clean inference | `x = split.images.to(device)`, one forward call | `(n_samples, 3, R, R)` | 89.8 MB / 469 MB / 1.83 GB | MPS (input); CPU load then `.to(device)` | No | **Yes -- entire split as one batch, unconditionally** |
| 2 | 100-view TTA prediction generation | one `view_batch` per iteration of `iter_deterministic_views()` (generator -- NOT all 100 held at once) | `(n_samples, 3, R, R)` per view | 89.8 MB / 469 MB / 1.83 GB (per view, peak = 1 view at a time) | MPS (input); generator yields on CPU, `.to(device)` per view | No (generator already bounds peak to 1 view) | **Yes -- each view is the entire split as one batch** |
| 3 | Prefix aggregation (`mean_probability`, nested slices) | `view_log_probs` (all 100 views' probabilities, already reduced) | `(100, n_samples, n_classes)` | ~36 MB (pathmnist, all resolutions -- probabilities only, independent of image resolution) | CPU (numpy) | Fixed at MAX_VIEWS=100 regardless of which prefix N is queried | No -- never image-sized |
| 4 | Majority-vote / confidence-weighted aggregation | same `view_log_probs` | same | same | CPU (numpy) | No | No |
| 5 | Original-anchored evaluation | `clean_logits` + `view_log_probs` | probabilities only | small | CPU (numpy) | No | No |
| 6 | **BN-adaptation input generation** | `adaptation_views = [view_batch for ... in iter_deterministic_views(..., n)]` -- a **Python list holding ALL n views simultaneously**, fully materialized BEFORE concatenation | `n x (n_samples, 3, R, R)` (list of n separate tensors) | **at n=100: 8.77 GB / 45.8 GB / 183 GB (28/64/128px)** | CPU (built), then `.to(device)` after `torch.cat` | **Yes -- linearly in n, up to MAX N=100** | **Yes -- this is the ROOT CAUSE of the observed failure** |
| 7 | **BN-adaptation forward pass** | `adaptation_inputs = torch.cat(adaptation_views, dim=0).to(device)` | `(n x n_samples, 3, R, R)` | at n=100, pathmnist: **~8.77 GB input tensor** (the observed crash, "Invalid buffer size: 9.35 GiB", is the intermediate convolutional activation buffer for this batch, which is larger than the input tensor itself and consistent with this order of magnitude) | MPS | **Yes -- linearly in n** | **Yes -- confirmed root cause, reproduced the real failure exactly** |
| 8 | Prediction with the adapted model | `adapted_model(x)` -- full split, same as clean inference | `(n_samples, 3, R, R)` | 89.8 MB / 469 MB / 1.83 GB | MPS | No | **Yes -- same as path 1** |
| 9 | Latency measurement at every N | `measure_clean_latency`: full-split `x`. `measure_tta_latency`: **`views = [view_batch.to(device) for ... in n]` -- the SAME list-materialization pattern as path 6**, built fresh for each N before timing | `(n_samples, 3, R, R)` (clean); `n x (n_samples, 3, R, R)` (list, TTA at N) | same as paths 1 and 6-7, respectively -- **at N=100: same ~8.77-183 GB list** | MPS (clean `x`); CPU-built list, `.to(device)` per view before the list is assembled | **Yes -- identical defect to path 6, at every registered N, worst case N=100** | **Yes -- this function never actually ran in the failed attempt (BN-adaptation crashed first, earlier in execution order), but has the IDENTICAL unbounded-list defect and would fail the same way if reached** |
| 10 | Artifact persistence | `predictions` dict: `clean_probs`, `view_probs`, `bn_adapted_probs` -- probabilities only | `(100, n_samples, n_classes)` etc. | ~36 MB, resolution-independent | CPU (numpy), written via `np.savez` | No | No -- never persists images, confirmed unaffected |

**Explicit findings requested:**

- **Clean/TTA inference forwards the complete validation split as one
  batch, unconditionally** (paths 1, 2, 8). This did not cause the
  observed failure (it succeeded at 28px in the ~26.6 minutes before the
  crash), but at 128px (`(10004, 3, 128, 128)`, Block D cells) the input
  tensor alone is 1.83 GB and the resulting intermediate activations will
  be substantially larger -- this is very likely to fail for the same
  structural reason once a Block D or 64px cell is evaluated, even though
  the 28px canary got this far. This IS treated as part of this
  correction (Part 4/5 below fixes clean/TTA inference batching, not only
  BN adaptation).
- **Latency measurement materializes a complete list of up to 100 full
  view batches simultaneously**, identically to the BN-adaptation defect
  that actually crashed. It never executed in the failed run (BN
  adaptation is called first in `compute_validation_evaluation()`,
  before `compute_evaluation_latency_report()` is ever invoked in
  `run_validation_evaluation()`), but it carries the exact same root
  cause and must be fixed identically.

## Part 3: BatchNorm semantics -- single-batch vs. sequential microbatch is NOT claimed equivalent

### 3.1 What differs, mechanically

In PyTorch, `BatchNorm2d.train()` mode updates `running_mean`/
`running_var` via an exponential-moving-average (EMA) rule on **every**
forward call:

```
running_stat <- (1 - momentum) * running_stat + momentum * batch_stat
```

with default `momentum=0.1`. This means:

- **Microbatch boundaries and order matter.** Each forward call updates
  the running statistic using only that call's batch statistics, blended
  with whatever the running statistic already was -- a sequence of N
  microbatch forward calls does **not** compute the same thing as one
  forward call over the pooled population, and does not even compute the
  same thing as the *reverse* order of the same microbatches.
- **Momentum determines how much any single call's statistics
  contribute**, and is applied identically whether that call sees the
  full population or one microbatch -- a genuinely different rule from
  "compute the exact mean/variance of the full population in closed
  form."
- **The final (possibly partial) microbatch's size and content still
  matter** the same way as any other call -- no special-casing exists or
  is needed, since the EMA rule already treats every call uniformly, but
  a smaller last batch has proportionally noisier batch statistics
  feeding into that update.
- **Statistics do not reset between N conditions** unless the
  implementation explicitly starts from a fresh checkpoint copy each
  time (already frozen protocol requirement 7, unaffected by this
  correction).

### 3.2 Deterministic synthetic demonstration

8 synthetic samples, `torch.manual_seed(0)`, a single `BatchNorm2d(2)`
layer, `momentum=0.1` (PyTorch default), `train()` mode, no-gradient
forward calls -- computed directly, not estimated:

**Single-batch** (all 8 samples in one forward call):
```
running_mean = [-0.0022622456308454275, 0.007410737220197916]
running_var  = [0.9794076085090637, 0.9978140592575073]
num_batches_tracked = 1
```

**Sequential microbatch** (samples 0-3, then samples 4-7, two forward
calls, same momentum):
```
running_mean = [-0.004185875877737999, 0.013434035703539848]
running_var  = [0.9600415825843811, 0.9945629239082336]
num_batches_tracked = 2
```

**True pooled population statistics** over all 8 samples (closed-form,
for reference -- NOT what either BN implementation computes, since
`momentum=0.1` blends 10% of any single call's batch statistic into a
zero-initialized running statistic rather than setting it equal to the
batch statistic):
```
mean = [-0.022622456774115562, 0.07410737127065659]
var (unbiased) = [0.7940764427185059, 0.9781407117843628]
```

**Reversed microbatch order** (samples 4-7, then samples 0-3):
```
running_mean = [-0.004410657566040754, 0.01472676545381546]
```

**Conclusions, computed exactly, not asserted:**
- `single-batch running_mean == true pooled mean`? **False.**
- `sequential running_mean == true pooled mean`? **False.**
- `single-batch running_mean == sequential running_mean`? **False.**
- `forward-order sequential == reversed-order sequential`? **False** --
  order-sensitive.

This confirms, with a real (not hypothetical) computation: **sequential
microbatch BN adaptation is a genuinely different algorithm from
single-batch BN adaptation**, not a memory-saving refactor of an
identical computation. It is also confirmed that the *original*
single-batch implementation's running statistics were themselves already
a momentum-scaled blend, not the raw pooled population statistic --
this is inherent to PyTorch's BatchNorm design and is unaffected by
this correction either way.

### 3.3 Protocol requirement check

`docs/phase2b_protocol.md` sec.4 ("Frozen BN-adaptation semantics"),
step 3: **"Perform one deterministic, no-gradient pass over the relevant
augmented inputs."** Step 4: **"Update BatchNorm running mean/variance
only."**

Neither step specifies a physical batch size, a single-tensor
requirement, or a closed-form pooled-statistic formula. "One... pass"
is standard ML terminology for one sweep visiting every sample exactly
once -- naturally satisfied by mini-batched iteration (this is
literally how "one epoch" is defined in ordinary training), not only by
one giant tensor. Step 9 of the same section already records that this
whole procedure is a **"paper-constrained operationalization... because
the source paper's own implementation is unavailable"** -- i.e. the
protocol itself is already filling a gap the source paper left open, at
the level of "what BN-adaptation procedure to use" in the first place.
Physical batching within that already-operationalized procedure is a
further, narrower gap of the same kind, not addressed by
`docs/phase2b_protocol.md` or `docs/experimental_protocol.md` at all.

**Conclusion: physical batching was underspecified by the frozen
protocol.** No conflict exists between this correction and the frozen
protocol's text. Per Part 3's explicit instruction, this authorizes
proceeding with the bounded-memory operationalization below --
**explicitly classified as a necessary, disclosed, non-equivalent
operationalization change, never as a mathematically identical
refactor of the original giant-batch implementation.**

## Part 4: Frozen bounded-memory operationalization

### 4.1 General inference (clean, every TTA view, adapted-model
prediction, latency measurement)

- Fixed inference batch size: **256** (matches the batch size already
  validated throughout this project's training/benchmark history --
  `configs/experiment_matrix.yaml`'s `FROZEN_TRAINING_SETTINGS.batch_size_28_64px`
  and the Block D 128px benchmark's selected batch sizes are both
  bounded at or below this).
- Applies uniformly to: clean inference, every TTA view's forward pass,
  BN-adapted-model prediction, and every latency measurement.
- No dynamic batch-size selection, no OOM-triggered fallback, no
  automatic reduction -- a fixed constant, frozen here.
- The final partial batch (`n_samples mod 256 != 0`) is allowed and
  processed as-is (a smaller final batch, never padded, never dropped).
- Output probabilities are concatenated back into the original
  validation-sample order (`sample_indices` order is authoritative and
  unchanged by batching).
- Evaluation-mode (`model.eval()`) inference does not update any model
  state -- unaffected by batching (unchanged from the existing
  implementation).
- TTA view generation remains independent of batch boundaries: each
  view's pixel content is already determined solely by
  `stable_view_seed(tta_seed, dataset, resolution, sample_index,
  view_index)` (per-sample, per-view, hashlib-based -- see
  `evaluation/views.py`), which takes no batch-size or batch-boundary
  input of any kind. Batching only changes how many already-fully-
  determined view images are forwarded through the model per call, never
  which pixels a view contains.

### 4.2 BN adaptation

- Algorithm identifier: **`sequential_microbatch_v1`**.
- Fixed adaptation batch size: **256** (same constant as general
  inference).
- Enumeration order, exactly: **(1) view index ascending; (2) within
  each view, validation sample index ascending.** I.e. all
  `(view=0, sample=0..n_samples-1)` pairs are forwarded (in one or more
  256-sized microbatches), then all `(view=1, sample=0..n_samples-1)`
  pairs, and so on up to `view = N-1`.
- Every `(sample, view)` pair for the current N is included **exactly
  once**.
- No `torch.cat()` across the full adaptation population -- only within
  a single microbatch (at most 256 images).
- No Python list ever holds more than one microbatch's (or, transiently,
  one view's worth of not-yet-chunked) images at a time -- never a list
  of all N full-resolution views.
- The existing nine-step BN procedure
  (`docs/phase2b_protocol.md` sec.4) is preserved in every respect
  **except** step 3's physical realization: "one deterministic,
  no-gradient pass" is now realized as a deterministic sequence of
  ordered microbatch forward calls instead of one physically-impossible
  giant forward call. Steps 1, 2, 4-8 are unchanged verbatim.
- Existing BN momentum (PyTorch default, `momentum=0.1`, unchanged --
  not itself part of this correction) and reset behavior (frozen
  protocol step 7: fresh checkpoint copy per N) are preserved exactly.
- Every N condition starts from a fresh copy of the original canonical
  checkpoint (unchanged).
- Adapted state is never carried from one N to another (unchanged).
- Learned parameters (conv/linear weights, BN affine `weight`/`bias`)
  remain bit-identical -- enforced by the existing runtime self-check in
  `bn_adapt()` (unchanged, still active).
- Only BatchNorm running-statistics buffers (`running_mean`,
  `running_var`, `num_batches_tracked`) may change.
- Persisted metadata records, per completed attempt: the adaptation
  batch size, the enumeration-order rule, the number of microbatches
  used for the primary N=50 condition (as a representative, auditable
  count), and the algorithm identifier `sequential_microbatch_v1`.
- This is recorded honestly, per Part 3, as a **necessary memory-safe
  operationalization -- not mathematically identical to the original
  giant-batch implementation.**

### 4.3 Latency

- Measures the actual bounded-batch production inference path (i.e. the
  SAME 256-image-chunked forward calls clean/TTA/BN-adapted inference
  now use) -- not a separate, differently-shaped measurement.
- Preserves the existing definitions, all seven registered N values,
  units (seconds), and multiplier formula (`TTA total / clean total`)
  unchanged.
- Preserves the required `torch.mps.synchronize()` calls around each
  timed region (`evaluation/latency.py`, unchanged, still the single
  source of timing/sync logic).
- **View-generation boundary, stated precisely and left unchanged in
  position**: view generation (the CPU-side deterministic augmentation
  producing each 256-image microbatch) remains OUTSIDE the timed region,
  exactly as in the current, unchanged `measure_tta_latency()` /
  `measure_clean_latency()` implementations in `evaluation/latency.py` --
  those functions time only the model forward call(s) they are given,
  never the construction of their inputs. This correction does not move
  that boundary; it only changes how many microbatches worth of
  already-generated views are fed into the (unchanged) timed forward
  loop, and ensures the population fed to it is built and consumed
  incrementally (see 4.1) rather than fully materialized first.
- Does not materialize all 193 full-validation-view batches
  simultaneously at any point (the pre-existing per-N discard pattern
  from `docs/phase2b_validation_evaluation_engineering_freeze.md` sec.1.2
  is preserved and extended with the same 256-batch chunking as every
  other inference path).

### 4.4 Frozen configuration fields

Added to `configs/validation_evaluation.yaml` (see the accompanying
config diff), as scientific/operational inputs that participate in
evaluation identity via the existing evaluator-fingerprint mechanism
(the config file is already in `EVALUATOR_FINGERPRINT_MANIFEST`, so no
separate fingerprint change is needed for the fields themselves; the
implementation files that consume them, changed in Part 5, are already
manifested):

```yaml
inference_batch_size: 256
bn_adaptation_batch_size: 256
bn_adaptation_algorithm: sequential_microbatch_v1
bn_adaptation_enumeration_order: view_major_then_sample_major
```

The seed (`1306178015`), policy (`mixed`), prefix sequence
(`[1, 2, 5, 10, 25, 50, 100]`), primary N=50 endpoint, aggregation
methods, and metric formulas are **unchanged** by this amendment.
