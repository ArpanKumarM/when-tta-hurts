# Phase 2B.4D-Engineering Addendum: closing evaluator-identity gaps

This addendum corrects two gaps found in
`docs/phase2b_validation_evaluation_engineering_freeze.md` (Part 1 of
Phase 2B.4D-Engineering) before any real evaluation result exists. It
does **not** rewrite or remove that document -- the latency-persistence
design frozen there is unchanged and remains in force. This addendum only
replaces sec.2's evaluator-fingerprint manifest and adds a precise
incompatible-completion policy.

**No real evaluation result has yet been observed anywhere in this
project.** `configs/validation_evaluation.yaml` (frozen seed `1306178015`,
prefix sequence, primary N=50, aggregation, policy) is unchanged. No
scientific protocol document is modified by this addendum.

## 1. Why the first fingerprint manifest was incomplete

The Part-1 manifest (12 files) was assembled by manually reading
`validation_evaluation.py`'s direct imports and reasoning about
categories. That is an informal, non-exhaustive method. A **mechanical**
transitive local-import closure computed from `validation_evaluation.py`
(the production entry point) finds 26 locally-importable modules, several
of which are genuinely computation-relevant and were missing:

- `src/when_tta_hurts/models/resnet.py` -- ResNet-18 is the model
  architecture for every Block C cell and 4 of 6 Block D cells; only
  SmallCNN's file was manifested.
- `src/when_tta_hurts/orchestrator.py` -- `_build_model()` (the
  model-selection/construction dispatch: SmallCNN vs ResNet-18, channel
  count, normalization) lives here and was entirely unmanifested.
- `src/when_tta_hurts/matrix.py` -- resolves each matrix cell's
  `model`/`dataset`/`resolution`/`normalization` from the raw experiment-
  matrix YAML; this expansion logic is NOT captured by `matrix_hash`
  (which hashes only the pre-expansion raw YAML dict), so a bug in cell
  expansion could silently resolve a cell to the wrong architecture/
  resolution without changing any already-hashed field.
- `src/when_tta_hurts/data.py` -- dataset loading, class-count/channel
  metadata resolution (`get_dataset_metadata()`), and the preprocessing
  transform (`torchvision.transforms.ToTensor()`) applied to every clean
  and TTA-view input.
- `src/when_tta_hurts/devices.py` -- `select_device()` is the actual
  production `device_resolver` default; which physical device (MPS vs
  CPU) executes every forward pass can produce non-bit-identical floating
  point results.
- `src/when_tta_hurts/config.py` -- implements `config_hash()`, the
  algorithm `compute_evaluation_id()` itself (and `matrix_hash`,
  `seed_manifest_sha256`, and the fingerprint construction below) is
  built on. The original freeze document's exclusion rationale
  ("self-referential/bootstrapping") does not survive scrutiny:
  fingerprinting this file means computing `hash_file()` over its raw
  bytes, exactly like any other file -- there is no circularity, only an
  omission. Corrected here.
- `pyproject.toml`, `uv.lock` -- exact runtime dependency identity
  (`torch`, `torchvision`, `kornia`, `medmnist`, `numpy`, etc.) was not
  represented at all.

## 2. Acknowledgement: this closure audit is not, and cannot be, formally exhaustive

A static AST-based transitive-import closure (Part C's regression test)
catches every module reachable via `from when_tta_hurts.X import Y` /
`import when_tta_hurts.X` statements, computed transitively from
`validation_evaluation.py`. It does **not** and cannot catch:

- Dynamically constructed import paths (`importlib.import_module(f"...")`
  with a runtime-computed name) -- none exist anywhere in the closure
  audited here (verified by inspection; every import in every closure
  file is a static `import`/`from...import` statement).
- Third-party package internals (`torch`, `torchvision`, `kornia`,
  `medmnist`, `numpy`, `scipy`) -- these are represented at the
  dependency-identity level only, via `uv.lock`'s exact pinned versions
  (sec.3 below), not at the source-file level, since they are not local,
  tracked files this repository controls.
- A change that alters behavior without changing any tracked file's
  bytes (e.g. a different Python interpreter, a different CPU/GPU driver,
  non-determinism in an underlying library). These are genuinely outside
  what a content-fingerprint of tracked files can capture; `metadata.json`
  already separately records `source_commit`, MPS availability, and
  package versions captured at run time as descriptive provenance for
  exactly this residual class of variation.

This is stated plainly rather than implied: the corrected manifest and
its regression test raise the bar from "manually reasoned about" to
"mechanically verified against every statically-resolvable local import,"
which is a substantial, auditable improvement -- not a claim of absolute
completeness.

## 3. Corrected fingerprint manifest (20 files)

Frozen in `EVALUATOR_FINGERPRINT_MANIFEST`
(`src/when_tta_hurts/validation_evaluation.py`):

```
configs/validation_evaluation.yaml
pyproject.toml
src/when_tta_hurts/artifacts.py
src/when_tta_hurts/config.py
src/when_tta_hurts/data.py
src/when_tta_hurts/devices.py
src/when_tta_hurts/evaluation/aggregation.py
src/when_tta_hurts/evaluation/bn_adaptation.py
src/when_tta_hurts/evaluation/latency.py
src/when_tta_hurts/evaluation/validation_loader.py
src/when_tta_hurts/evaluation/views.py
src/when_tta_hurts/evaluation_result_artifacts.py
src/when_tta_hurts/matrix.py
src/when_tta_hurts/metrics.py
src/when_tta_hurts/models/resnet.py
src/when_tta_hurts/models/small_cnn.py
src/when_tta_hurts/orchestrator.py
src/when_tta_hurts/transforms/policies.py
src/when_tta_hurts/validation_evaluation.py
uv.lock
```

Construction rule (unchanged from Part 1): `config_hash({"manifest_version":
1, "files": {path: sha256(raw file bytes)}})`, `sha256` via
`artifacts.py::hash_file()`, `config_hash` via `config.py::config_hash()`
-- both reused unchanged.

### 3.1 `orchestrator.py`: a deliberately conservative, whole-file inclusion

`orchestrator.py` is large (1500+ lines) and contains both (a)
`_build_model()` -- genuinely computation-relevant -- and (b) canonical
training-attempt selection/clean-tree logic whose *output*
(`checkpoint_hash`, `training_attempt`, and for Block D the
`effective_config_hash`) is already separately hashed into
`evaluation_id`. A change to (b) alone (e.g. a future clean-tree
allow-list edit, as already happened once in Phase 2B.4C) will also
perturb `evaluation_id`, even though it does not touch (a). This is a
known, accepted cost: the addendum's explicit instruction is that false
invalidation is preferable to silently reusing a scientifically
incompatible completion, and splitting `_build_model()` into its own file
purely to narrow the fingerprint is judged an unnecessary refactor for
this task. If this collateral-invalidation cost becomes operationally
significant, extracting `_build_model()` into a dedicated,
narrowly-scoped module is the natural follow-up -- not done here.

### 3.2 Files considered and excluded, with reasons

See Part A's classification table in the accompanying report for the
full transitive closure. Summary of exclusions and why each is safe:

- `src/when_tta_hurts/__init__.py`, `src/when_tta_hurts/evaluation/__init__.py`
  -- trivial/thin re-export shims; no computation.
- `src/when_tta_hurts/evaluation/tta.py`, `src/when_tta_hurts/evaluation/cache.py`
  -- Phase 2A pilot-era modules, imported only as an inert side effect of
  `evaluation/__init__.py`'s package initialization; never called by any
  function reachable from the confirmatory evaluation path (verified: no
  call site anywhere outside their own definitions and the `__init__.py`
  re-export).
- `src/when_tta_hurts/ledger.py`, `src/when_tta_hurts/run_identity.py`,
  `src/when_tta_hurts/block_d_benchmark.py`, `src/when_tta_hurts/block_d_gate.py`,
  `src/when_tta_hurts/authorization.py` -- all participate in canonical
  training-checkpoint *selection*, never in evaluation *computation*.
  Their entire causal effect on evaluation results funnels through
  already-separately-hashed fields (`checkpoint_hash`, `training_attempt`,
  Block D's `effective_config_hash`) -- a selection-logic bug either picks
  a different (differently-hashed) checkpoint, which is already detected,
  or is independently caught by `hash_state_dict()` verification against
  the recorded `checkpoint_hash` before any checkpoint is used.
- `src/when_tta_hurts/dataset_verification.py`, `src/when_tta_hurts/reproducibility.py`,
  `src/when_tta_hurts/result_artifacts.py`, `src/when_tta_hurts/training.py`
  -- exclusively training-time code (dataset-checksum verification for
  the *training* loader path, training-time seeding, training-attempt
  persistence, the training loop itself). Verified by call-site inspection:
  none of these is invoked anywhere reachable from
  `run_validation_evaluation()`/`compute_validation_evaluation()`/
  `compute_evaluation_latency_report()`; evaluation's own dataset access
  goes through `data.py::load_pilot_split()` directly (no checksum
  verification on that path -- **noted as a separate, out-of-scope
  observation, not fixed here**; this addendum is about identity
  stability, not adding new integrity checks to the runner).
- `configs/experiment_matrix.yaml` -- deliberately NOT duplicated in the
  fingerprint manifest: its content is already captured via `matrix_hash`
  (`config_hash(raw)` of this exact file), a separate hashed field in
  `ValidationEvaluationConfig`. Adding the file itself would be pure
  redundancy, not a gap.

## 4. How runtime dependencies are represented

Not via `capture_environment()`/`EnvironmentManifest` (that is
descriptive, per-machine, per-run *provenance* -- already recorded in
training's own artifacts, structurally unsuitable for a stable identity
hash since it would make `evaluation_id` vary by machine even with
identical code). Instead: `pyproject.toml` (declared dependency
constraints) and `uv.lock` (the exact resolved, pinned versions of every
dependency, including `torch`, `torchvision`, `kornia`, `medmnist`,
`numpy`) are both in the fingerprint manifest. A `uv sync --frozen`
against a changed `uv.lock` changes the fingerprint; third-party package
*source code itself* is not separately fingerprinted (out of this
repository's control), but the exact version identifier that determines
which package code is installed is.

## 5. Incompatible-completion policy (precise, frozen here)

| Existing completion | Requested execution | Required result |
|---|---|---|
| Same frozen config and fingerprint | Identical | Skip before all heavy dependencies |
| Different fingerprint | Incompatible | Hard failure before attempt allocation |
| Different frozen evaluation config | Incompatible | Hard failure before attempt allocation |
| Different checkpoint hash | Conflicting training source | Hard failure |
| Multiple matching eligible completions | Ambiguous | Hard failure |
| Only failed/aborted attempts exist | No canonical completion | Next numbered attempt may proceed |

Implemented as `ConflictingEvaluationImplementationError` (new,
`validation_evaluation.py`), raised by `check_evaluation_skip()` when a
COMPLETED attempt exists for `training_run_id` under any
`evaluation_config_hash` different from the current request's, and no
exact match exists. This mirrors
`orchestrator.ConflictingCompletedRunError`'s discipline on the training
side. An aborted or failed attempt never counts as an existing canonical
completion and never triggers this -- consistent with the existing,
unchanged incident-recovery design
(`docs/phase2b_validation_evaluation_incident.md`): the recorded aborted
incident (`evaluation_attempt=1`,
`ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5`) is not
a completed canonical evaluation and therefore never blocks
`evaluation_attempt=2`.

This raise happens inside `check_evaluation_skip()`, which
`run_validation_evaluation()` already calls before
`require_clean_working_tree()`, attempt-number allocation, MPS
initialization, checkpoint loading, dataset loading, model construction,
or view generation -- unchanged control-flow position, so this hard
failure is structurally guaranteed to occur before all of those.
