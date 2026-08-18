# Phase 2B.3E-Gate Evaluation: Block D runtime-gate result

This document freezes the permanent, machine-checked INCLUDED/OMITTED
decision for Block D (native-128px PathMNIST/BloodMNIST), computed
mechanically from a single real MPS benchmark run per
`docs/phase2b_block_d_benchmark_spec.md`. **This document reports only
runtime, memory, checksum, and loss-finiteness evidence. No accuracy,
prediction, TTA, or test-split value was computed, read, or used at any
point in this evaluation.**

## 1. Frozen six-cell manifest

Expanded via `parse_and_validate_matrix(..., block_d_gate_passed=True)`,
`D_conditional_128px` block:

| # | run_id | dataset | res | model | norm | seed | n_classes |
|---|---|---|---|---|---|---|---|
| 1 | D-pathmnist-128px-batchnorm-policy-none-s0 | pathmnist | 128 | small_cnn | batchnorm | 0 | 9 |
| 2 | D-pathmnist-128px-batchnorm-policy-none-s1 | pathmnist | 128 | small_cnn | batchnorm | 1 | 9 |
| 3 | D-pathmnist-128px-batchnorm-policy-none-s2 | pathmnist | 128 | small_cnn | batchnorm | 2 | 9 |
| 4 | D-bloodmnist-128px-batchnorm-policy-none-s0 | bloodmnist | 128 | small_cnn | batchnorm | 0 | 8 |
| 5 | D-bloodmnist-128px-batchnorm-policy-none-s1 | bloodmnist | 128 | small_cnn | batchnorm | 1 | 8 |
| 6 | D-bloodmnist-128px-batchnorm-policy-none-s2 | bloodmnist | 128 | small_cnn | batchnorm | 2 | 8 |

Matrix hash: `ed9d36e6b3e0fdb4561de6ad70e75720502c97538bf7e6fdb67dd2bd4cb9045a`.
Protocol commit: `ce4c962`. Spec commit:
`3189580733581e41555b2cccda80366f58e22383`. Benchmark source commit:
`5f47d25dee6906f8f44e58ba411f406bc8339634` (HEAD at the moment the real
benchmark ran).

**No Block D cell in this manifest was trained.** This document, and the
benchmark that produced it, measure runtime/memory only.

## 2. Artifact provenance and checksums

Both artifacts were fetched via the explicit, separate
`scripts/run_block_d_benchmark.py prefetch` command (never implicitly by
plan or benchmark mode), then independently re-verified outside the
production code path (a second, standalone MD5 computation).

| Dataset | File | Size (bytes) | Expected MD5 (`medmnist.INFO[...]['MD5_128']`) | Actual MD5 (production) | Actual MD5 (independent re-check) | `resized` |
|---|---|---|---|---|---|---|
| PathMNIST-128 | `data/raw/pathmnist_128.npz` | 4,257,786,775 | `ac42d08fb904d92c244187169d1fd1d9` | `ac42d08fb904d92c244187169d1fd1d9` | `ac42d08fb904d92c244187169d1fd1d9` | `False` |
| BloodMNIST-128 | `data/raw/bloodmnist_128.npz` | 569,072,780 | `adace1e0ed228fccda1f39692059dd4c` | `adace1e0ed228fccda1f39692059dd4c` | `adace1e0ed228fccda1f39692059dd4c` | `False` |

Native shapes (from `train_images`/`val_images` only -- `test_images`/
`test_labels` were never indexed, only the two datasets' array-name lists
were enumerated via `np.load(...).files` to confirm the NPZ's split
structure, which is metadata, not split content):

- PathMNIST-128: `train_images.shape = (89996, 128, 128, 3)`,
  `val_images.shape = (10004, 128, 128, 3)` -- same train/val split sizes
  as the already-verified 28px/64px PathMNIST artifacts, confirming this
  is the same dataset at native higher resolution, not a derived resize.
- BloodMNIST-128: `train_images.shape = (11959, 128, 128, 3)`,
  `val_images.shape = (1712, 128, 128, 3)` -- same split sizes as the
  28px/64px BloodMNIST artifacts.
- `medmnist.INFO` identity: PathMNIST `python_class=PathMNIST`,
  `n_channels=3`, `task=multi-class`, 9 labels. BloodMNIST
  `python_class=BloodMNIST`, `n_channels=3`, `task=multi-class`, 8 labels
  -- matching Block D's frozen `n_classes` (9/8) exactly.

Both files are `.gitignore`d (`data/raw/`, `*.npz`); `git status` showed
no tracked file changed as a result of the prefetch.

## 3. Execution environment (single real run)

Device `mps`, `mps_built=True`, `mps_available=True`, chip Apple M3 Pro,
macOS 15.7.7, Python 3.12.2, arm64, torch 2.13.0, torchvision 0.28.0. The
benchmark command (`scripts/run_block_d_benchmark.py benchmark`) was
invoked **exactly once** and was not rerun for reproducibility or timing
stability, per protocol.

## 4. Raw measurements (per dataset, per batch candidate)

Fixed candidate set, deterministic ascending order (64, 128, 256); 10
warm-up + 30 measured steps per phase (training and validation, timed
separately); `SAFE_MEMORY_FRACTION = 0.7` boundary. All candidates
completed without OOM or non-finite loss for both datasets; the raw
`artifacts/benchmarks/block_d_native_128_benchmark.json` file (gitignored,
SHA-256 `e80123f9f6c0aafdbf7a846351c7294c12b3aa7a26a5f7fb6e52ab79c0adecb4`)
contains all 30 raw per-step timings for every candidate -- only the
mean/median/derived figures are reproduced below.

### PathMNIST-128

| Batch | Setup (s) | Train mean/median step (s) | Val mean/median step (s) | Peak current/driver bytes | Mem. fraction | Safe? | OOM | Finite loss |
|---|---|---|---|---|---|---|---|---|
| 64 | 32.929 | 0.078216 / 0.077929 | 0.019207 / 0.019091 | 14,147,840 / 1,296,973,824 | 0.1007 | yes | no | yes |
| 128 | 33.459 | 0.154954 / 0.154747 | 0.038269 / 0.038118 | 27,188,224 / 2,554,216,448 | 0.1982 | yes | no | yes |
| 256 | 33.787 | 0.310281 / 0.310064 | 0.076498 / 0.077534 | 52,279,296 / 5,087,576,064 | 0.3948 | yes | no | yes |

Throughput at the selected batch (256): 256 / 0.310281 s ≈ **825.06
images/sec** (training, forward+backward+step).

### BloodMNIST-128

| Batch | Setup (s) | Train mean/median step (s) | Val mean/median step (s) | Peak current/driver bytes | Mem. fraction | Safe? | OOM | Finite loss |
|---|---|---|---|---|---|---|---|---|
| 64 | 4.488 | 0.077270 / 0.076942 | 0.018706 / 0.018650 | 14,145,280 / 1,298,022,400 | 0.1007 | yes | no | yes |
| 128 | 4.444 | 0.152216 / 0.152123 | 0.035769 / 0.037046 | 27,183,360 / 2,554,216,448 | 0.1982 | yes | no | yes |
| 256 | 4.521 | 0.306939 / 0.307590 | 0.072999 / 0.073811 | 52,272,384 / 5,087,576,064 | 0.3948 | yes | no | yes |

Throughput at the selected batch (256): 256 / 0.306939 s ≈ **834.04
images/sec**.

## 5. Selected batches

All three candidates were safe for both datasets, so per the frozen
"largest safe candidate" rule, batch **256** was selected for **both**
PathMNIST-128 and BloodMNIST-128.

## 6. Formulas and independent recomputation

Frozen formulas (`docs/phase2b_block_d_benchmark_spec.md` sec.7-8):

```
train_batches_per_epoch      = ceil(train_split_size / selected_batch)
validation_batches_per_epoch = ceil(val_split_size / selected_batch)
projected_training_seconds   = 30 * train_batches_per_epoch * mean_training_step_seconds
projected_validation_seconds = 30 * validation_batches_per_epoch * mean_validation_step_seconds
projected_end_to_end_seconds = setup_seconds + projected_training_seconds
                              + projected_validation_seconds
                              + measured_persistence_verification_seconds
Block D contribution         = 3 * PathMNIST projected_end_to_end_seconds
                              + 3 * BloodMNIST projected_end_to_end_seconds
binding total hours          = 3.92 (frozen pessimistic A+B+C)
                              + Block D contribution / 3600
```

An independent Python recomputation (outside `block_d_benchmark.py`,
reading only the raw JSON's stored per-step means and split sizes)
reproduced every figure below to within 1e-6 relative tolerance:

| Quantity | PathMNIST-128 | BloodMNIST-128 |
|---|---|---|
| `train_batches_per_epoch` (independently: `ceil(89996/256)` / `ceil(11959/256)`) | 352 | 47 |
| `validation_batches_per_epoch` (`ceil(10004/256)` / `ceil(1712/256)`) | 40 | 7 |
| `projected_training_seconds` | 3276.570161 | 432.783937 |
| `projected_validation_seconds` | 91.797673 | 15.329829 |
| `persistence_verification_seconds` | 0.031478 | 0.008074 |
| `projected_end_to_end_seconds` | 3402.186432 | 452.642933 |
| 3-seed contribution (s) | 10206.559296 | 1357.928799 |

```
Block D contribution seconds = 10206.559296 + 1357.928799 = 11564.488095
                              (reported: 11564.488094942762 -- match)
binding total hours          = 3.92 + 11564.488095 / 3600 = 7.132358
                              (reported: 7.132357804150767 -- match)
```

**Descriptive-only** total using the measured (not pessimistic) A+B+C
active compute (Block A ~98.90 min + Block B ~20.34 min + Block C ~5.62
min = ~124.85 min = ~2.081 h, `docs/phase2b_block_c_audit.md` sec.6):
`2.081 + 11564.488095/3600 ≈ 5.293 h`. This descriptive figure is **not**
the binding total and did not affect the decision -- the binding total
(7.132 h) is higher and is what was actually gated on, per the frozen
"always use the pessimistic figure, never let a lower actual figure
relax the gate" rule.

## 7. Condition-by-condition gate table

| Condition | PathMNIST-128 | BloodMNIST-128 |
|---|---|---|
| Native 128px artifact (`resized=False`) | PASS | PASS |
| Checksum match | PASS | PASS |
| Device `mps` | PASS | PASS |
| No OOM | PASS | PASS |
| Finite loss throughout | PASS | PASS |
| Training ≤90 min/run (54.61 / 7.21 actual) | PASS | PASS |
| End-to-end ≤120 min/cell (56.70 / 7.54 actual) | PASS | PASS |
| ≥1 safe batch candidate | PASS (all 3) | PASS (all 3) |

| Combined condition | Result |
|---|---|
| Binding A+B+C+D total < 24.0 h (7.13 h actual) | PASS |

**Every condition passed for both datasets and the combined binding
total.** Per the frozen all-or-nothing rule, the result is:

## 8. Final decision: **INCLUDED**

This decision was **not overridden** -- it is exactly the programmatic
output of `evaluate_block_d_gate()`, reproduced independently in section
6 above and matching the canonical artifact bit-for-bit (section 9).

Raw benchmark output SHA-256:
`e80123f9f6c0aafdbf7a846351c7294c12b3aa7a26a5f7fb6e52ab79c0adecb4`
(file: `artifacts/benchmarks/block_d_native_128_benchmark.json`,
gitignored -- the canonical tracked artifact is
`artifacts/block_d_gate_decision.json`, which records this same hash in
its `raw_output_sha256` field).

## 9. Canonical decision artifact

`artifacts/block_d_gate_decision.json` (tracked, permanent, refuses
overwrite once written) was independently diffed field-by-field against
this document's audited figures (`final_decision`, `per_dataset_pass`,
`gate_condition_booleans`, `binding_total_hours`,
`block_d_contribution_seconds`, per-dataset checksums/`resized`/selected
batch/projections) -- **exact match**, no discrepancy.

## 10. No scientific evidence informed this decision

`DatasetBenchmarkRecord` (`src/when_tta_hurts/block_d_gate.py`) has no
accuracy, prediction, F1, NLL, ECE, Brier, or TTA-delta field by design.
The benchmark pipeline (`src/when_tta_hurts/block_d_benchmark.py`) never
computed one: training used `CrossEntropyLoss` solely to obtain a
realistic backward-pass timing and a finite/non-finite check, never
logged or compared to a label-derived correctness measure; validation
was forward-only timing with the same finite-loss check only. Both the
raw benchmark output and the canonical decision were schema-validated
(`_validate_output_schema`/`_validate_decision_schema`) to structurally
reject `accuracy`/`f1`/`nll`/`ece`/`brier`/`tta_delta`/`prediction`/
`test_metric` as whole-word tokens anywhere in the serialized JSON --
confirmed clean by an independent word-boundary scan (section 1's
"no accuracy/TTA/test value" claim was re-verified against the raw file,
not merely assumed). `test_images`/`test_labels` were never indexed in
either the prefetch or the benchmark run (`load_pilot_split` has no
test-split access mechanism at all).

## 11. Confirmation: no Block D training occurred

No `artifacts/confirmatory/D` directory exists. The confirmatory ledger
(`artifacts/ledger_confirmatory.csv`) remains exactly 38 data rows (35
completed / 1 failed / 2 aborted) -- unchanged by this gate evaluation.
No confirmatory attempt was allocated, no checkpoint was persisted
outside the disposable `artifacts/benchmarks/_block_d_disposable_persistence/`
timing harness (gitignored, not a real attempt directory), and
`load_and_verify_block_d_decision()` -- the only code path a future
Block D training entry point could use to read this decision -- remains
unwired from `orchestrator.py` and every other training entry point in
this repository. **Block D training was not started, unlocked, or
executed as part of this task, regardless of the INCLUDED result above.**
