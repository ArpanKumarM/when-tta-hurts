# Phase 2B.3E-Engineering: Block D native-128px benchmark operationalization

**Dated:** 2026-08-18T15:17:33Z, written **before any 128px artifact is
downloaded, before any real benchmark is run, and before the Block D
inclusion decision is made.**

**This document is a pre-measurement operational clarification necessitated
by a missing implementation -- not a response to any observed 128px
runtime or scientific result.** No 128px artifact and no 128px benchmark
result exists anywhere in this project as of this writing (reconfirmed:
`data/raw/` contains only 28/64px `pathmnist*`/`bloodmnist*` and
`dermamnist.npz`; no `artifacts/confirmatory/D` directory exists; no
`*128*`/`*block_d*` file exists under `artifacts/`).

It does **not** change the matrix, seeds, models, datasets, gate
thresholds, the 24-hour ceiling, or any scientific hypothesis. Every
numeric threshold below is already frozen in `docs/phase2b_protocol.md`
sec.6 and `src/when_tta_hurts/block_d_gate.py`; this document only fills
in the *operational* details (exact formulas, exact measurement
procedure) that the frozen protocol text left unspecified and that no
prior code implements for native 128px data.

## 1. Reconfirmed starting state

- No 128px official artifact exists locally.
- No 128px benchmark result exists (no file matching `*128*` under
  `artifacts/`).
- `artifacts/confirmatory/` contains only `A/`, `B/`, `C/` -- no `D/`.
- Mandatory matrix: A+B+C = 33/33 canonical. Ledger: 38 rows.
- Existing Block D gate criteria (`docs/phase2b_protocol.md` sec.6,
  `src/when_tta_hurts/block_d_gate.py`) are unchanged by this document.

## 2. Data rules (frozen, restated)

- Only official **native** PathMNIST-128 (`pathmnist_128.npz`) and
  BloodMNIST-128 (`bloodmnist_128.npz`) artifacts may be used for the
  real gate decision.
- Checksum must match `medmnist.INFO[dataset]["MD5_128"]` exactly.
- Resized proxies (e.g. upsampled 28px/64px images) and synthetic tensors
  are **prohibited** for the real decision -- they may only appear in
  tests, clearly labeled as non-production.
- The benchmark reads **training images/labels only**. Test keys
  (`test_images`, `test_labels`) must never be opened or read, in
  production or test code paths.
- Every real measurement record must carry `resized=False`, verified the
  same way `dataset_verification.verify_official_dataset_artifact`
  already verifies it for 28/64px (exact filename match to the resolution
  requested; no fallback to a different resolution file).

## 3. Device rules (frozen, restated)

- MPS only. No CPU fallback under any real-benchmark condition (mirrors
  `devices.select_device("mps")`'s existing no-fallback contract).
- float32, no mixed precision.
- `torch.mps.synchronize()` immediately before and immediately after every
  timed region (mirrors `scripts/benchmark_runtime.py`'s existing
  practice).

## 4. Model rules (frozen, restated)

- Exact Block D architecture: `SmallCNN` (`models/small_cnn.py`),
  `normalization="batchnorm"` (Block D's only registered normalization
  variant per `configs/experiment_matrix.yaml`).
- Dataset-specific output classes, derived from
  `data.get_dataset_metadata(dataset).n_classes` -- **never hardcoded**:
  PathMNIST=9, BloodMNIST=8.
- No training augmentation (Block D's `training_policy` is `none`).
- Optimizer/loss: Adam, lr=0.001, weight_decay=0, cross-entropy -- the
  same `FROZEN_TRAINING_SETTINGS` values used by every other block, read
  from `matrix.FROZEN_TRAINING_SETTINGS`, never re-declared as literals.

## 5. Batch-candidate selection rule (new operational detail)

- Fixed candidate set, deterministic ascending order: **64, 128, 256**
  (matches `scripts/benchmark_runtime.py`'s existing `BATCH_SIZES`).
- A candidate is **safe** iff: it completes all warm-up+measured steps
  without OOM, without non-finite loss, **and** peak
  `driver_allocated_bytes` stays `<= 0.7 x recommended_max_bytes` (the
  same `SAFE_MEMORY_FRACTION = 0.7` boundary already frozen in
  `scripts/benchmark_runtime.py`, generalized to 128px/both datasets --
  not a new threshold).
- The **largest safe candidate** is selected per dataset independently.
- If **no** candidate is safe for a dataset, that dataset fails the gate
  outright, and since the gate requires both datasets to pass, **the
  entire Block D gate fails** (per the existing frozen "if either dataset
  fails any gate, Block D is omitted in its entirety" rule).
- Selection uses runtime/memory evidence only -- no accuracy/prediction
  value is computed or could influence which batch is "safe."
- The selected batch size per dataset must be recorded in the benchmark
  output for later Block D training use, if the block is ever included.

## 6. Timing rules (new operational detail, extends existing pattern)

- 10 untimed warm-up training steps, 30 measured training steps (matches
  `scripts/benchmark_runtime.py`'s `WARMUP_STEPS`/`MEASURED_STEPS`).
- Every raw measured-step time is recorded (not just the mean).
- The frozen projection uses the **arithmetic mean** training-step time;
  the **median** is also reported, descriptively only, never used in the
  frozen formula.
- Validation is separately benchmarked: forward-only batches (no
  backward/optimizer step, `model.eval()`, `torch.no_grad()`), using the
  same 10-warmup/30-measured structure where the validation split has at
  least 40 batches at the selected batch size; where it has fewer, the
  closest complete-cycle equivalent is used (i.e. warm-up = min(10, one
  full epoch of validation batches) and measured = the remaining distinct
  batches up to one full validation epoch, cycling only if necessary to
  reach 30 measured steps) -- this is an explicit, honest accommodation
  for a validation split smaller than 40 batches, not a silent shortcut.
- No accuracy, prediction, or any other scientific metric is computed
  during either training-step or validation-step timing -- only
  loss-finiteness (a pass/fail check, not a stored score), step duration,
  and memory.

## 7. Projection formulas (new operational detail, frozen as of this document)

```
projected_training_seconds    = 30 * train_batches_per_epoch * mean_training_step_seconds
projected_validation_seconds  = 30 * validation_batches_per_epoch * mean_validation_step_seconds
```

`train_batches_per_epoch`/`validation_batches_per_epoch` are computed from
the real native 128px split sizes (`ceil(split_size / selected_batch)`),
exactly as `scripts/benchmark_runtime.py` already does for 28/64px
(`steps_per_epoch = -(-split_size // batch_size)`).

**One-time setup overhead** (checksum verification, DataLoader
construction, model + optimizer construction) is measured once per
dataset via wall-clock timing around that exact sequence.

**Persistence/verification overhead** (checkpoint serialization, hashing,
strict restoration, metadata/manifest writing and verification) is
measured once per dataset, using a **temporary, disposable benchmark
directory** (never `artifacts/confirmatory/`), by timing exactly the
sequence `result_artifacts.persist_and_verify_completion` performs on a
throwaway state_dict -- this reuses the real production persistence code
path for realism without touching any real attempt directory.

```
projected_end_to_end_seconds = setup_seconds
                              + projected_training_seconds
                              + projected_validation_seconds
                              + measured_persistence_verification_seconds
```

**No additional safety multiplier is introduced, because none was frozen**
in `docs/phase2b_protocol.md` sec.6 or `block_d_gate.py`. The only
existing safety margin is the `SAFE_MEMORY_FRACTION=0.7` memory boundary
(a batch-selection safety margin, not a runtime multiplier) and the
protocol's own 90-min/120-min/24h ceilings, which already contain
generous headroom relative to the measured 28/64px per-epoch rates.

**Gate checks:**
```
training gate: projected_training_seconds   <= 90 * 60   (90 minutes)
cell gate:     projected_end_to_end_seconds  <= 120 * 60  (120 minutes)
```

## 8. Total-budget rule (new operational detail)

The **binding** total-budget check uses the **preregistered pessimistic
A+B+C estimate of ~3.92 hours** (`docs/pilot_audit.md`, "Recomputed block
estimates" table, computed from measured per-epoch rates and multipliers
*before* any Phase 2B block was actually executed), **not** the lower
post-completion actual active-compute total (~124.85 min ~= 2.08h,
measured after A+B+C actually ran). Using the higher, pre-registered
pessimistic figure is intentionally conservative and consistent with the
frozen protocol's own "pessimistic projected total" framing -- it does not
relax the gate relative to using the actual figure, since actual < 
pessimistic here.

```
Block D contribution = 3 * PathMNIST projected_end_to_end_seconds
                      + 3 * BloodMNIST projected_end_to_end_seconds

binding total hours = 3.92 (frozen pessimistic A+B+C)
                     + Block D contribution / 3600
```

`binding total hours` must be **strictly below 24.0**. The
actual-compute-based total (measured A+B+C + Block D contribution) is
also reported, descriptively only, and **never** used to override the
binding result -- if the binding (pessimistic) total fails while the
actual-compute-based total would have passed, the gate still fails.

## 9. Consistency check against existing frozen documents

Cross-checked against `docs/phase2b_protocol.md` sec.6,
`src/when_tta_hurts/block_d_gate.py` (`MAX_TRAINING_MINUTES_PER_RUN=90`,
`MAX_END_TO_END_MINUTES_PER_CELL=120`, `MAX_PESSIMISTIC_TOTAL_HOURS=24.0`),
`configs/experiment_matrix.yaml` (Block D's `datasets`, `resolution=128`,
`normalization=batchnorm`, `seeds=[0,1,2]`, `training_runs=6`), and
`docs/pilot_audit.md`'s A+B+C pessimistic estimate. **No conflict found**
with any explicit frozen requirement -- every rule above either restates
an existing frozen number/threshold verbatim, or fills in a previously
unspecified operational detail (validation-benchmark structure,
persistence-overhead measurement, exact projection formula, which A+B+C
figure is binding) in a way that is strictly more conservative than or
equivalent to what the frozen text already required.

## 10. Explicit statement

This specification was written as a **pre-measurement operational
clarification necessitated by a missing implementation** -- confirmed at
the top of this document: no 128px artifact, no 128px benchmark result,
and no Block D attempt exists anywhere in the project as of this writing.
It is not a response to, and was not informed by, any observed 128px
runtime or any scientific/accuracy result, because none exists yet.
