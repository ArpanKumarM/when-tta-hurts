# Phase 2B.4D-Engineering: freezing latency persistence and stable evaluation identity

This document freezes two **implementation-completeness** corrections to
the already-implemented, already-frozen validation-evaluation runner
(`src/when_tta_hurts/validation_evaluation.py`). Neither correction
changes the frozen scientific protocol
(`docs/phase2b_protocol.md`, `configs/validation_evaluation.yaml`): the
TTA policy, seed, prefix sequence, N=50 primary endpoint, aggregations,
metrics, and BN-adaptation procedure are untouched. This document is not
itself a scientific-protocol file and does not require the same
sign-off/freeze machinery as `docs/phase2b_protocol.md`
(`configs/experiment_matrix.yaml`-style `status: approved` gating) --
it governs engineering wiring only.

No real evaluation has been run under this document. It is written and
committed *before* any code change, per Phase 2B.4D-Engineering's Part 1
requirement.

## 1. Latency persistence

### 1.1 What is reused, unchanged

`src/when_tta_hurts/evaluation/latency.py` is reused **completely
unchanged**. No demonstrable implementation bug was found in it during
this audit. Specifically, unchanged and reused as-is:

- `_sync(device)` -- calls `torch.mps.synchronize()` iff `device.type ==
  "mps"`, matching `docs/phase2b_protocol.md` sec.3's synchronization
  requirement exactly.
- `measure_clean_latency(model, x, device)` -- `model.eval()`, sync,
  `time.perf_counter()` start, one forward pass over the **entire**
  validation population `x` in a single call, sync, `time.perf_counter()`
  stop. Returns wall-clock seconds for that single call. No repetitions,
  no warmup pass -- this measures literally the same forward call the
  scientific clean-probability computation already performs
  (`compute_validation_evaluation()`'s `clean_logits = model(x)`), just
  timed.
- `measure_tta_latency(model, views, device)` -- `model.eval()`, sync,
  `time.perf_counter()` start, **one forward pass per view in `views`**
  (a Python loop, `for v in views: model(v)`), sync, stop. Returns total
  wall-clock seconds for the whole sequence of `len(views)` forward
  calls. This is a genuine "run all N views through the model" timing,
  not divided by N here -- per-sample division happens in the caller.
- `LatencyReport` dataclass fields and their exact meaning:
  `clean_latency_seconds` (float), `tta_latency_seconds_by_n` (`{N:
  total_seconds}`), `per_sample_latency_seconds_by_n` (`{N:
  total_seconds / n_samples}`), `compute_multiplier_by_n` (`{N:
  total_seconds / clean_latency_seconds}`, or `inf` if
  `clean_latency_seconds <= 0`), `n_samples` (int, `x.shape[0]`).

`build_latency_report(model, x, ordered_views_by_n, device)` itself
(the convenience wrapper that takes a **pre-materialized** `{N: [view_1,
..., view_N]}` dict and loops over it) is **not called directly by the
production path** -- see 1.2 for why, and see 1.3 for the mechanical
equivalence proof this document requires before that substitution is
accepted.

### 1.2 Production wiring and the memory-shape decision

`build_latency_report()`'s signature requires every registered N's full
view list to already exist simultaneously in `ordered_views_by_n` before
the call. Because prefixes are nested but **not shared** in that
dict -- N=100's list and N=50's list are two independent Python lists,
each holding its own view tensors -- passing all seven registered N
values (`1, 2, 5, 10, 25, 50, 100`; sum = 193) at once would require
holding 193 view-batches in memory simultaneously, each the same size as
one full augmented copy of the validation population. This is
**materially larger** than the scientific pass's peak (which processes
one view-batch at a time via `iter_deterministic_views()`'s generator and
discards it immediately after the forward pass, per
`compute_validation_evaluation()`).

The production wiring therefore calls `measure_clean_latency()` once and
`measure_tta_latency()` once per registered N, generating and discarding
each N's view list immediately after that N's timing call, and manually
assembles a `LatencyReport` with the **identical field-by-field formulas**
`build_latency_report()` would have produced. This is not a change to any
frozen function -- both `measure_clean_latency()` and
`measure_tta_latency()` are the same frozen, unchanged functions
`build_latency_report()` itself calls internally; only the loop that
holds intermediate results is inlined, so peak memory is bounded by the
single largest registered N (100) rather than their sum (193).

### 1.3 Mechanical equivalence requirement

Part 2's regression tests must prove, on synthetic tensors, that the
production wiring's manually-assembled `LatencyReport` is field-for-field
identical (same `n_samples`, same dict keys, same formula relationships)
to what `build_latency_report()` would produce for the same
model/views/device, modulo the two independently-measured wall-clock
values themselves (which are not required to be bit-identical between two
separate timing calls, only each internally self-consistent with its own
formula). This is a mechanical-equivalence proof, not a byte-identity
proof of timing numbers.

### 1.4 What is measured and persisted

Per registered N in `PREFIX_SEQUENCE = (1, 2, 5, 10, 25, 50, 100)`:

- Clean total inference latency (one measurement, not per-N).
- Clean per-sample latency = clean total / n_samples (derived, not
  separately measured).
- Total TTA latency at N (one measurement per N).
- TTA per-sample latency at N = TTA total at N / n_samples.
- Compute multiplier at N = TTA total at N / clean total.

All five reuse the **existing, frozen formulas already implemented in
`evaluation/latency.py`** -- none is invented here.

### 1.5 Scope discipline

- Latency is measured using the **same** restored checkpoint, model
  object, validation population (`split.images`/`split.sample_indices`),
  device, and registered N values as the scientific evaluation already
  computed in the same attempt -- never a different checkpoint, subset,
  or device.
- Latency measurement runs strictly **after** the scientific probability
  bank (`compute_validation_evaluation()`'s output) is already fully
  computed, and never re-derives, mutates, or replaces any element of
  `predictions["clean_probs"]`, `predictions["view_probs"]`, or
  `predictions["bn_adapted_probs"]`. The view tensors generated for
  latency timing are independently regenerated via the same deterministic
  `evaluation/views.py::iter_deterministic_views()` call used elsewhere
  and are discarded after timing -- never stored, never fed into any
  aggregation/metric function.
- Latency values are descriptive only: no branch anywhere in the
  production path reads a latency value to select a policy, threshold,
  aggregator, early-stop condition, or "favorable" result. The function
  that computes latency (`compute_evaluation_latency_report()`) does not
  accept `labels` as a parameter at all, structurally forbidding any
  accuracy/TTA-effect quantity from reaching it.
- A latency-measurement or latency-persistence-validation failure must
  raise, causing the enclosing attempt to be recorded `status="failed"`
  (never `"completed"`) via the same exception-propagation-to-the-existing
  `except` block that already handles every other completion-blocking
  failure in `run_validation_evaluation()`. No new exception-handling
  branch is introduced for this -- the existing try/except already
  covers it once latency computation/validation is placed inside the
  existing `try` block, after checkpoint/data loading and before
  persistence.

### 1.6 Persistence location

The latency report is persisted as a new **required** structured section,
`metrics["latency"]`, inside the already-manifest-covered `metrics.json`
artifact -- not a new separate file. `metrics.json` is already one of
`REQUIRED_EVALUATION_ARTIFACTS` and is already included in
`artifact_manifest.json`'s hash/size verification
(`build_evaluation_artifact_manifest()` /
`verify_evaluation_artifact_manifest()`), so no new manifest-wiring code
is needed to satisfy "include the containing artifact in
artifact_manifest.json" -- it is already covered by construction. A
separate `latency.json` was considered and rejected: it would require
duplicating the manifest-inclusion/verification wiring for no benefit,
since `metrics.json` is already a small, JSON, schema-validated artifact
of exactly the right shape for a structured, required sub-section.

Exact persisted shape (`metrics["latency"]`):

```json
{
  "clean_latency_seconds": <float>,
  "n_samples": <int>,
  "by_n": {
    "1":   {"tta_latency_seconds": <float>, "per_sample_latency_seconds": <float>, "compute_multiplier": <float>},
    "2":   {...},
    "5":   {...},
    "10":  {...},
    "25":  {...},
    "50":  {...},
    "100": {...}
  }
}
```

(`by_n` keys are JSON string representations of the registered N values
-- JSON object keys must be strings; the underlying `LatencyReport`
dataclass itself keeps them as `int` in memory.)

### 1.7 Validation rules (enforced before `status="completed"` is ever possible)

1. `by_n` contains exactly the registered N values (`PREFIX_SEQUENCE`),
   each exactly once -- no missing, no extra, no duplicate.
2. `clean_latency_seconds`, every `tta_latency_seconds`, every
   `per_sample_latency_seconds`, and every `compute_multiplier` are
   finite (`math.isfinite`).
3. `clean_latency_seconds`, every `tta_latency_seconds`, and every
   `per_sample_latency_seconds` are `>= 0`.
4. `n_samples` matches the actual persisted prediction sample count
   (`predictions["labels"].shape[0]`) exactly.
5. For every N: `per_sample_latency_seconds == tta_latency_seconds /
   n_samples` (within floating-point tolerance, `math.isclose` with
   `rel_tol=1e-9`).
6. For every N: `compute_multiplier == tta_latency_seconds /
   clean_latency_seconds` (same tolerance), or `compute_multiplier ==
   float("inf")` if `clean_latency_seconds == 0`.

Any violation raises `EvaluationPersistenceError` before any file is
written, exactly mirroring how `validate_predictions_arrays()` already
gates `predictions.npz`.

## 2. Stable evaluation identity

### 2.1 The problem

`ValidationEvaluationConfig.source_commit` (hashed into `evaluation_id`
via `compute_evaluation_id()`) was set to `_git_commit_hash()` -- the
literal current `git rev-parse HEAD` at invocation time. Because this
repository's discipline is to commit ledger/incident/audit-documentation
changes as their own commits, HEAD advances on every such commit even
though nothing about *how the evaluation itself computes* changed. A
results-record commit (e.g. this very engineering-freeze commit, or a
future ledger-row commit) would therefore silently give an otherwise
byte-identical re-invocation a **different** `evaluation_id`, defeating
idempotent skip and, if actually re-executed, allocating a needless new
attempt.

### 2.2 The fix: separate three concepts

- **Scientific/evaluation identity** (`evaluation_id`, hashed): must
  depend only on what could change the scientific computation --
  checkpoint identity, frozen TTA-seed configuration (and its own
  freeze-commit/derivation provenance, unchanged), prefix sequence,
  aggregators, secondary analyses, policy identifier, frozen protocol
  commit constant, matrix hash, and the new **evaluator-implementation
  fingerprint** (2.3). It must NOT depend on the literal current HEAD.
- **Evaluator implementation identity** (`evaluator_fingerprint`,
  hashed into `evaluation_id`): a deterministic content fingerprint of an
  explicitly enumerated, frozen set of evaluation-relevant tracked files
  (2.3). Changing any byte of any manifested file changes this
  fingerprint, and therefore changes `evaluation_id`.
- **Execution provenance** (`source_commit`, NOT hashed): the literal
  `git rev-parse HEAD` at the moment a given attempt actually ran,
  recorded for audit/reproducibility purposes only, in
  `metadata.json`, exactly as it already was -- just no longer part of
  the identity hash.

### 2.3 Frozen evaluator-fingerprint file manifest

The fingerprint is `config_hash({"manifest_version": 1, "files": {path:
sha256(raw file bytes)}})` (reusing the existing, unchanged
`config.py::config_hash()` -- stable-sorted-key JSON, SHA-256), over
these 12 repo-relative paths, computed at invocation time by reading the
actual working-tree file content via the existing, unchanged
`artifacts.py::hash_file()`:

```
configs/validation_evaluation.yaml
src/when_tta_hurts/artifacts.py
src/when_tta_hurts/evaluation/aggregation.py
src/when_tta_hurts/evaluation/bn_adaptation.py
src/when_tta_hurts/evaluation/latency.py
src/when_tta_hurts/evaluation/validation_loader.py
src/when_tta_hurts/evaluation/views.py
src/when_tta_hurts/evaluation_result_artifacts.py
src/when_tta_hurts/metrics.py
src/when_tta_hurts/models/small_cnn.py
src/when_tta_hurts/transforms/policies.py
src/when_tta_hurts/validation_evaluation.py
```

Category coverage (per the task's required list):

| Category | File(s) |
|---|---|
| View generation | `evaluation/views.py` |
| Validation loading | `evaluation/validation_loader.py` |
| Evaluation computation / orchestration | `validation_evaluation.py` |
| Aggregation | `evaluation/aggregation.py` |
| Metrics | `metrics.py` |
| BN adaptation | `evaluation/bn_adaptation.py` |
| Latency | `evaluation/latency.py` |
| Evaluation persistence/schema | `evaluation_result_artifacts.py`, `artifacts.py` (atomic write/hash primitives used by persistence) |
| Frozen validation-evaluation configuration | `configs/validation_evaluation.yaml` |
| Model architecture (checkpoint-loading correctness) | `models/small_cnn.py` |
| Frozen augmentation policy | `transforms/policies.py` |

**Deliberately excluded, with rationale:**

- `src/when_tta_hurts/orchestrator.py` -- used only for canonical
  training-checkpoint *selection* (`check_confirmatory_skip`,
  `authorize_block_d_cell`, `_build_model` dispatch, clean-tree
  enforcement). Any change to selection logic that actually changed
  *which* checkpoint gets used is already captured downstream by a
  different `checkpoint_hash`/`training_attempt` in the hash; a change
  to unrelated orchestrator concerns (e.g. clean-tree ledger-path
  allow-listing, as in the immediately prior Phase 2B.4C commit) must
  not silently perturb evaluation identity, and is exactly the kind of
  drift this fingerprint design exists to prevent.
- `src/when_tta_hurts/matrix.py` -- its content-derived effect is
  already captured explicitly via `matrix_hash`
  (`expanded.source_config_hash`), already a separate hashed field in
  `ValidationEvaluationConfig`; including the file itself would be
  redundant.
- `src/when_tta_hurts/config.py` -- this is the hash **algorithm**
  used to compute the fingerprint itself; including it would be
  self-referential/bootstrapping and adds no practical protection (a
  change to `config_hash()`'s canonicalization would already change
  every downstream hash it touches, loudly, in essentially every other
  ledger/config-hash consistency check in the codebase).

If a manifested file does not exist on disk when the fingerprint is
computed, this is a hard failure (`EvaluatorFingerprintError`), never a
silent partial fingerprint.

### 2.4 Persistence

Both the stable fingerprint and its full file-manifest (the `{path:
sha256}` mapping used to compute it) are persisted in `metadata.json`
(new required keys `evaluator_fingerprint`, `evaluator_fingerprint_manifest`)
-- already one of `REQUIRED_EVALUATION_ARTIFACTS`, already
manifest-covered. `source_commit` remains in `metadata.json` exactly as
before, now explicitly documented as provenance-only (not hashed). No
change to the evaluation-ledger CSV schema
(`VALIDATION_EVALUATION_LEDGER_FIELDNAMES`) is made -- the ledger's
`evaluation_config_hash` column continues to hold the (now-stable)
`evaluation_id`, unchanged in meaning, just now computed without the
HEAD-drift defect.

### 2.5 Required behavior (frozen)

1. Same checkpoint + same frozen evaluation configuration + same
   evaluator fingerprint -> same `evaluation_id`.
2. A ledger/doc-only commit (touching neither `configs/validation_evaluation.yaml`
   nor any of the 12 manifested source files) after a completed run ->
   an identical re-invocation computes the **same** `evaluation_id` and
   is skip-eligible via `check_evaluation_skip()`, entirely before MPS
   initialization, checkpoint loading, dataset loading, model
   construction, or attempt allocation (`check_evaluation_skip()` is
   already called before all of those in `run_validation_evaluation()`
   -- unchanged control-flow position).
3. A change to any of the 12 manifested files -> a different
   `evaluator_fingerprint` -> a different `evaluation_id` -> the new
   invocation is treated as a **different** evaluation request; it does
   not skip against, and does not silently overwrite, an existing
   canonical completion under the old identity. (It also does not
   retroactively invalidate the old completion's ledger row -- that row
   remains readable exactly as recorded.)
4. The legacy aborted evaluation ID
   (`ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5`,
   recorded under the pre-correction hash formula that included
   `source_commit`) is **not** rewritten, recomputed, or reclassified by
   this change. Its `evaluation_attempt=1` row remains permanently
   reserved for `training_run_id=A-pathmnist-28px-batchnorm-policy-none-s0`
   exactly as already committed
   (`docs/phase2b_validation_evaluation_incident.md`,
   `artifacts/ledger_validation_evaluation.csv`).
5. `next_evaluation_attempt_number()` is unchanged -- it is keyed on
   `training_run_id` alone (union of directory and ledger evidence), not
   on `evaluation_id`, and was already correct under the old formula.
   The first eventual real canary for
   `A-pathmnist-28px-batchnorm-policy-none-s0` -- under whatever
   `evaluation_id` the corrected formula now computes -- still resolves
   to `evaluation_attempt=2`.

### 2.6 Completion-selection ambiguity (audit finding, folded into this freeze)

Auditing `check_evaluation_skip()` alongside this change surfaced a
pre-existing gap, independent of the identity-formula correction above:
if more than one **completed**, directory-backed, artifact-verified
attempt for the same `training_run_id` matched the same
`evaluation_config_hash`, the function silently returned the
numerically-first match rather than failing closed. This does not match
the discipline already established on the training side
(`orchestrator.check_confirmatory_skip()` raises
`AmbiguousCanonicalCompletionError` in the equivalent situation). This
freeze also fixes `check_evaluation_skip()` to raise a new
`AmbiguousEvaluationCompletionError` when more than one completed,
hash-matching, artifact-verified attempt exists, mirroring the training
side exactly. There is no evaluation-side amendments-ledger equivalent
(no mechanism yet exists to mark an evaluation attempt
canonical-ineligible), so unlike the training path this has no
eligibility-filtering step before the ambiguity check -- if this
situation is ever reached in practice, it requires the same kind of
explicit human reconciliation the training side already requires.
