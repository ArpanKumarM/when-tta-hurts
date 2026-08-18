# Phase 2B.3A canary audit: `attempt_001` of `A-pathmnist-28px-batchnorm-policy-none-s0`

**Status: `attempt_001` is preserved but declared noncanonical / ineligible
for confirmatory analysis.** This document records why, and the exact
chronology that led to the decision, before any validation accuracy, TTA
metric, or test metric was ever inspected.

## 1. Chronology

1. **Phase 2B.3A Part A** (commit `9bcc4a2fb5e5799466669e90886943d873bb287d`,
   `feat: enable Phase 2B train-validation execution`) enabled
   `scripts/run_confirmatory.py train-validation` as a strict single-cell
   canary CLI: exactly one `--run-id`, resolving to one approved
   unconditional Block A/B/C cell, gated on a clean working tree and a
   real MPS device (no CPU fallback), with production checksum
   verification before any `DataLoader` construction. Full test/lint/
   security suite passed before use (288 tests, ruff clean, gitleaks
   clean).
2. **Phase 2B.3A Part B** froze the canary target as the first literal
   cell in the committed matrix expansion: `A-pathmnist-28px-batchnorm-policy-none-s0`
   (Block A, PathMNIST, native 28px, SmallCNN, BatchNorm, training policy
   `none`, seed 0). Config hash `03154180ddda338c2b893d9938a45e21a0035884ea7a04e317392e54caab3059`.
   Matrix source hash `ed9d36e6b3e0fdb4561de6ad70e75720502c97538bf7e6fdb67dd2bd4cb9045a`.
   No prior completed/failed attempt existed for this run ID.
3. **Phase 2B.3A Part C** executed exactly one real train-validation cell
   via `uv run python3 scripts/run_confirmatory.py train-validation --run-id A-pathmnist-28px-batchnorm-policy-none-s0`.
   It completed successfully: exit status `completed`, checkpoint hash
   `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e`,
   runtime **75.94374823570251 seconds**.
4. **Immediately after completion**, before Part D could run, inspection
   of the produced artifacts revealed that the runner (as committed in
   `9bcc4a2fb5e5799466669e90886943d873bb287d`) does not persist per-epoch
   training history, best epoch, best validation accuracy, early-stopping
   outcome/reason, or peak MPS memory to any file. Only `best_checkpoint.pt`
   (model weights only) and a coarse `status.json` (attempt number,
   status, timestamps, config hash) exist. This was discovered by directly
   reading the artifact directory contents and the runner source code --
   **no validation accuracy, TTA metric, or test metric was inspected at
   any point in this discovery.**
5. An attempted idempotent-skip re-run (Part D) was **refused by design**:
   the CLI's clean-working-tree guard correctly detected that the ledger
   append from step 3 made the tree dirty, and refused to proceed before
   even resolving the run ID. No second invocation of the CLI's
   train-validation mode ever executed training, loaded data, or touched
   the model. This is recorded here as the **dirty-tree conflict**.
6. Separately, code inspection (not execution) of `run_canary_cell()`
   revealed a **skip-ordering defect**: the production data-loader factory
   (`default_train_validation_loader_factory`, which performs real
   checksum verification and real dataset loading) is invoked *before*
   `run_train_validation_cell()`'s "matching completed attempt: skip"
   check. This means even a legitimate, canonical-eligible completed run
   would currently trigger a real dataset reload before the skip decision
   is reached -- a defect, not a scientific-integrity violation (no
   retraining or overwrite occurs), but a violation of the intended "skip
   before dataset loading" contract.
7. Given both defects, the decision was made **not** to retry, not to
   change any code around the existing result, and not to treat the
   completed run as canonical. This document and the associated ledger
   amendment record that decision. **This decision was made entirely on
   the basis of missing observability infrastructure -- not on the basis
   of any inspected validation accuracy, TTA outcome, or test outcome,
   none of which were ever computed-and-inspected, computed only
   internally for early stopping, or accessed at all (test/TTA).** Any
   future re-execution of this run ID is therefore an **engineering
   retry** made necessary by fixing missing infrastructure, not a
   result-driven re-roll made because attempt 1's outcome was unfavorable
   -- attempt 1's outcome (validation accuracy, best epoch, etc.) was
   **never observed by anyone**, so it structurally cannot have motivated
   this decision.

## 2. Valid checksum / device / configuration facts (from Part B/C)

- Dataset artifact: `data/raw/pathmnist.npz` (native 28px)
- Expected MD5: `a8b06965200029087d5bd730944a56c1`
- Actual MD5: `a8b06965200029087d5bd730944a56c1` (verified match, `resized=False`)
- Device: MPS (real; `select_device("mps")` -- no CPU fallback path exists)
- Dependency versions: torch 2.13.0, kornia 0.8.3, medmnist 3.0.2
- Batch size: 256 (`FROZEN_TRAINING_SETTINGS.batch_size_28_64px`)
- Config hash: `03154180ddda338c2b893d9938a45e21a0035884ea7a04e317392e54caab3059`
- Matrix source hash: `ed9d36e6b3e0fdb4561de6ad70e75720502c97538bf7e6fdb67dd2bd4cb9045a`
- Phase 2B protocol commit: `ce4c962dba3ac29dec5aae7f1680385035322bd8`
- Runner enablement (source) commit: `9bcc4a2fb5e5799466669e90886943d873bb287d`

## 3. Runtime

**75.94374823570251 seconds** (`ended_at - started_at` from `status.json`:
`1787016828.5673192 - 1787016752.623571`).

## 4. Exact existing confirmatory-ledger row (unmodified, unmoved)

```
True,A-pathmnist-28px-batchnorm-policy-none-s0,1,A_core_normalization_resolution,03154180ddda338c2b893d9938a45e21a0035884ea7a04e317392e54caab3059,ce4c962,pathmnist,small_cnn,28,batchnorm,none,0,validation,completed,30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e,1787016752.623571,1787016828.5673192,75.94374823570251,,True,False
```

This row is **not** modified, removed, or reordered by this audit. The
eligibility correction is recorded exclusively as an append-only overlay
in `artifacts/ledger_amendments.csv` (see below).

## 5. Missing observability fields (the reason for ineligibility)

Not persisted to any artifact by `attempt_001`:

- Per-epoch training loss
- Per-epoch validation loss and accuracy
- Best epoch
- Best validation accuracy
- Early-stopping epoch and reason
- Peak MPS memory (current-allocated and driver-allocated)
- A `result.json` / `metadata.json` / `artifact_manifest.json` of any kind

Persisted (and preserved, unmodified, by this audit):

- `best_checkpoint.pt` (model weights only)
- `status.json` (attempt number, block/dataset/model/normalization/seed,
  config hash, status, `started_at`/`ended_at`, `failure_reason`)
- The confirmatory ledger row shown in section 4

## 6. Dirty-tree conflict

The single ledger append produced by `attempt_001`'s completion made the
working tree "dirty" under the Part A clean-tree guard, which refused a
subsequent identical CLI invocation before it reached matrix resolution.
This is documented as a design gap in the working-tree policy (frozen
protocol/config/source cleanliness was conflated with append-only
research-ledger cleanliness) -- addressed in Part 2 of the
Phase-2B.3A-correction plan (working-tree policy redefinition), not in
this document.

## 7. Skip-ordering defect

`run_canary_cell()`, as committed in `9bcc4a2fb5e5799466669e90886943d873bb287d`,
calls the data-loader factory (real checksum verification + real dataset
load) before `run_train_validation_cell()`'s completed-attempt skip check.
A legitimate skip would therefore still reload real data first. This is a
control-flow defect, not a scientific-integrity defect (no retraining,
no overwrite, no metric leakage results from it) -- addressed in Part 2
(skip-ordering fix), not in this document.

## 8. Disposition

`attempt_001` of `A-pathmnist-28px-batchnorm-policy-none-s0`:

- **Preserved** on disk, byte-for-byte, permanently. It is never deleted,
  renamed, or overwritten by any future action.
- **Not canonical.** It does not count as a completed confirmatory result
  for Phase 2B analysis purposes.
- **Does not block re-execution.** A future legitimate run of this exact
  run ID (after the Part 2 engineering fix lands) receives `attempt_002`,
  which may become canonical if it completes with full observability
  artifacts, verified hashes, and a successful ledger append.
- **The checkpoint is preserved, not discarded** -- it remains available
  for future inspection/comparison, but is not treated as an authoritative
  confirmatory result.
- **No unfavorable result was hidden.** No result -- favorable or
  unfavorable -- was ever computed-and-inspected in the first place. This
  disposition was decided purely on missing-infrastructure grounds,
  before any accuracy number of any kind was read by any person or
  process outside of PyTorch's own internal early-stopping comparison
  (which never left process memory and was never logged, printed, or
  persisted).

## 9. `attempt_002`: failed at post-training persistence verification

After the observability/skip-ordering correction (`fc40c4fc8f0505507c93a791a3c5a53c71576176`)
and the eligibility-boolean-parsing fix (`f2e2fe07c0b12287314a8cfcf680981c7a6474b4`)
landed, `attempt_002` of `A-pathmnist-28px-batchnorm-policy-none-s0` was
executed via the real production command:

```
uv run python3 scripts/run_confirmatory.py train-validation --run-id A-pathmnist-28px-batchnorm-policy-none-s0
```

**Scientific training completed successfully.** The failure occurred
strictly AFTER training, in the new post-training persistence-verification
step (`result_artifacts.py::persist_and_verify_completion`'s
checkpoint-restore comparison), which raised:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, mps:0 and cpu!
```

**Root cause:** `persist_and_verify_completion`'s `model_factory` callable
constructed a fresh model without moving it to the training device.
`torch.load()` restored the saved checkpoint's tensors onto their
originally-saved device (MPS), while the freshly-constructed comparison
model defaulted to CPU. `torch.equal()` then raised on the cross-device
comparison. This bug exists only in the new verification instrumentation
added in `fc40c4fc8`; it does not exist in, and had no effect on, the
training computation itself. It was never caught by any prior test
because every existing test ran on `torch.device("cpu")`, where the
model_factory's default device and the training device coincide -- this
was the first execution against real MPS.

### Complete eight-epoch history (from `training_history.json`, written
### successfully before the failure point)

| epoch | learning_rate | train_loss | val_loss | val_accuracy | epoch_runtime_s |
|---|---|---|---|---|---|
| 1 | 0.001000 | 0.6061511701 | 1.1447124557 | 0.5930627749 | 9.645014500 |
| 2 | 0.000997260948 | 0.3332840328 | 1.5202585471 | 0.5906637345 | 9.338128416 |
| 3 | 0.000989073800 | 0.2527653527 | 0.9149315368 | 0.7388044782 | 9.420314500 |
| 4 | 0.000975528258 | 0.2083446707 | 3.9892766944 | 0.3962415034 | 9.438764375 |
| 5 | 0.000956772729 | 0.1794501554 | 1.2563373124 | 0.6599360256 | 9.421177833 |
| 6 | 0.000933012702 | 0.1553598728 | 3.1791103030 | 0.5136945222 | 9.515376458 |
| 7 | 0.000904508497 | 0.1366218688 | 1.1107464026 | 0.6694322271 | 9.469812292 |
| 8 | 0.000871572413 | 0.1239525544 | 2.7124165786 | 0.5489804078 | 9.634175041 |

Best epoch: **3** (best_val_accuracy=0.7388044782087165,
best_val_loss=0.9149315368171121). Early-stopped after epoch 8: "no
validation-accuracy improvement (> min_delta=0.0) for 5 consecutive
epochs". Total runtime: 75.88329879200319 seconds (~75.96s including
attempt bookkeeping overhead). Peak MPS memory:
current_allocated_bytes=7,777,024, driver_allocated_bytes=1,170,292,736.

### All available artifact hashes

- `best_checkpoint.pt`: file MD5 `eb7cfb6e23b691f0ffc6a64f23b5a77f`, file
  SHA-256 `029c761a903a7a0386e37f03c8a8eb7ad3eb9ced61a7c8c0191efe922ee96eed`,
  tensor-content hash (`hash_state_dict`)
  `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e`
- `metadata.json`: file MD5 `2397747453b83338553ee29291d0ab0d`
- `result.json`: file MD5 `8618db86c9ce4d0110f614dba4e76ba4`
- `status.json`: file MD5 `e9118eb66c93d5af3fbf8aa355add762`
- `training_history.json`: file MD5 `33c35a5fdfb0ebf805762e079990eafa`

### Checkpoint equivalence with `attempt_001`

Independently re-verified (not merely trusted from the failed run's own
in-process record): `attempt_002`'s `best_checkpoint.pt` is **byte-for-byte
identical** to `attempt_001`'s -- identical file MD5
(`eb7cfb6e23b691f0ffc6a64f23b5a77f`), identical file SHA-256
(`029c761a903a7a0386e37f03c8a8eb7ad3eb9ced61a7c8c0191efe922ee96eed`), and
identical tensor-content hash
(`30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e`), with
every tensor confirmed equal via `torch.equal()` across the full
state_dict. This is bitwise reproducibility of the scientific training
computation across two independent executions with the same frozen
configuration and seed.

### Confirmations

- **`artifact_manifest.json` and `status=completed` were correctly
  withheld.** Both are the last two artifacts `persist_and_verify_completion`
  writes, strictly gated on the checkpoint-restore comparison succeeding.
  Since that comparison raised, neither was ever written --
  `attempt_002/`'s directory contains exactly five files
  (`best_checkpoint.pt`, `training_history.json`, `result.json`,
  `metadata.json`, `status.json`), never six, and `status.json` correctly
  reads `"status": "failed"`.
- **Validation accuracy was first inspected only after the runner had
  already failed.** `result.json` (containing `best_val_accuracy`) was
  written by the runner itself as part of its own artifact-persistence
  step, which completes BEFORE the checkpoint-restore comparison that
  actually failed; no human or downstream process read any validation
  metric until this audit document was written, after the run had already
  terminated with `status="failed"` on disk and in the ledger.
- **No scientific setting will change before `attempt_003`.** The Part 2
  fix (device-neutral checkpoint verification, to be committed separately)
  touches only `result_artifacts.py`'s post-training verification
  comparison -- not model construction, training, checkpoint selection,
  checkpoint saving, seeding, optimizer/scheduler, precision, or any
  frozen hyperparameter in `matrix.FROZEN_TRAINING_SETTINGS`.
- **`attempt_001` and `attempt_002` remain byte-for-byte unchanged** as of
  this writing -- re-verified immediately before this section was written
  (see hashes above and in section 4).
