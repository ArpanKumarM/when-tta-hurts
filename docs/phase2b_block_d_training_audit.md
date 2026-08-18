# Phase 2B.3G Part 2: Block D confirmatory training audit

This document is a mechanical audit of Block D's six confirmatory training
cells (native-128px PathMNIST/BloodMNIST), produced against the production
canonical-selection logic (`orchestrator.check_confirmatory_skip` /
`authorize_block_d_cell`), not a hand-maintained table.

**Validation results were not used for tuning.** No hyperparameter, seed,
batch size, architecture, or protocol setting was changed as a result of
any value observed during or after any Block D cell. Cells were run
strictly in frozen matrix order (PathMNIST s0, s1, s2, then BloodMNIST
s0, s1, s2); none was skipped, reordered, retried, or omitted based on an
earlier cell's outcome. **This document does not evaluate TTA and makes
no comparison to any accuracy/TTA/resolution hypothesis** -- Block D's
preregistered purpose (`docs/research_plan.md`, `configs/experiment_matrix.yaml`)
is to supply native-128px checkpoints for the later, still-unperformed
Validation-Gated TTA evaluation; this document reports only clean
(non-TTA) training/validation results.

## 1. Gate-decision provenance

`artifacts/block_d_gate_decision.json`: `final_decision=INCLUDED`, SHA-256
`8bcc2810371e161c34ee6dbf8b03cb722f6f767c373a26d195e972db8b9eb7cf`, tracked
and committed at `a90de21c365395771973748baab2790b2a914cc5` (`results:
freeze Phase 2B Block D runtime gate`). This file remained byte-identical
throughout all six training cells (re-verified after every cell and again
at the end of this audit).

| Field | Value |
|---|---|
| Gate-decision commit | `a90de21c365395771973748baab2790b2a914cc5` |
| Benchmark-source commit (recorded provenance, not code that trained) | `5f47d25dee6906f8f44e58ba411f406bc8339634` |
| Benchmark-specification commit | `3189580733581e41555b2cccda80366f58e22383` |
| Protocol commit | `ce4c962` |
| Matrix hash | `ed9d36e6b3e0fdb4561de6ad70e75720502c97538bf7e6fdb67dd2bd4cb9045a` |
| Selected batch (PathMNIST-128) | 256 |
| Selected batch (BloodMNIST-128) | 256 |
| PathMNIST-128 MD5 | `ac42d08fb904d92c244187169d1fd1d9` |
| BloodMNIST-128 MD5 | `adace1e0ed228fccda1f39692059dd4c` |

## 2. Training-source versus benchmark-source: corrected distinction

An earlier informal report described the canary as trained under "source/
benchmark commit 5f47d25," conflating two genuinely distinct, correctly
separated fields in the persisted artifacts. This section documents the
correction, verified mechanically against every cell's `result.json`/
`metadata.json`:

- **Actual training-source commit** -- the code HEAD that produced each
  checkpoint -- is the top-level, schema-required (`_RESULT_REQUIRED_KEYS`/
  `_METADATA_REQUIRED_KEYS` in `result_artifacts.py`), artifact-manifest-
  covered `source_commit` field. For all six Block D cells this is
  **`68410b0dd9bf01500255a1b08b459bac8c4216cf`** (`fix: route Block D CLI
  through gate authorization`) -- the HEAD at which every cell was
  actually executed.
- **Benchmark-source commit** -- the HEAD at which the runtime-gate
  benchmark measured PathMNIST-128/BloodMNIST-128 timing -- is recorded
  ONLY nested under `block_d_gate_provenance.block_d_benchmark_source_commit`,
  a clearly-labeled gate-provenance field, never presented as the code
  that trained the model. For all six cells this is
  `5f47d25dee6906f8f44e58ba411f406bc8339634`.

These two fields were never conflated in the persisted data -- only the
prose summary was imprecise. No code change, re-run, or amendment was
required; the seed-0 canary remained canonical throughout.

## 3. CLI-refusal chronology (pre-attempt engineering event, not a failure)

Before commit `68410b0`, the exact command
`uv run python3 scripts/run_confirmatory.py train-validation --run-id
D-pathmnist-128px-batchnorm-policy-none-s0` dispatched to
`run_canary_cell()`, whose `resolve_canary_run_id()` structurally rejects
Block D IDs. Invoking it raised `BlockDRunRejectedError` **before**
`check_confirmatory_skip()`, `require_clean_working_tree()`,
`device_resolver()`, or any loader/model code ran.

- No attempt was allocated (`start_attempt()` was never reached).
- No data, model, or MPS activity occurred.
- No ledger row was written or warranted (the confirmatory ledger records
  outcomes of allocated experimental attempts; none existed).
- `git status` and `artifacts/confirmatory/` were confirmed empty of any
  Block D content immediately afterward.

This is documented as a **pre-attempt engineering event** -- a CLI-dispatch
defect caught and corrected by commit `68410b0` (`fix: route Block D CLI
through gate authorization`) -- and must not be represented as an
experimental failure, incident, or ledger-worthy event of any kind.

## 4. Frozen six-cell manifest

| # | run_id | dataset | res | norm | seed | original matrix config hash |
|---|---|---|---|---|---|---|
| 1 | D-pathmnist-128px-batchnorm-policy-none-s0 | pathmnist | 128 | batchnorm | 0 | `228ac56a05f22739044dc3c9d7c249ef9aac81e7958867631591e577639b3311` |
| 2 | D-pathmnist-128px-batchnorm-policy-none-s1 | pathmnist | 128 | batchnorm | 1 | `26caa43ca7a9c75c21d7281c85312fc16c4fd9f6c6e660bfa91a530620aa0919` |
| 3 | D-pathmnist-128px-batchnorm-policy-none-s2 | pathmnist | 128 | batchnorm | 2 | `3172dd869af4742753341ef8a2d06890e501cec0073bd9b3cee9646228d001b4` |
| 4 | D-bloodmnist-128px-batchnorm-policy-none-s0 | bloodmnist | 128 | batchnorm | 0 | `f53e2978fc5c0cff4259d93ce079329193b6af96b7f73c454200853c9701450f` |
| 5 | D-bloodmnist-128px-batchnorm-policy-none-s1 | bloodmnist | 128 | batchnorm | 1 | `fa39b6e254145213d860a7e738eecacdd7de7587a520a4d4320a8d00bf68b7a7` |
| 6 | D-bloodmnist-128px-batchnorm-policy-none-s2 | bloodmnist | 128 | batchnorm | 2 | `8ca9f8ed9b9e0ade41678e9c78f1c72f8517a409aaab191cc9af6e9703871403` |

All `training_policy=none` (no training or validation augmentation --
confirmed for every cell: `metadata.json` carries no `augmentation_policy`
key, and `orchestrator.py` only ever constructs one when
`training_policy=="matched_to_approved_tta_policy"`). All six use
`model=small_cnn`, `normalization=batchnorm`, `selected_batch_size=256`.
PathMNIST cells: 9-class classifier. BloodMNIST cells: 8-class classifier.
Source commit for all six attempts: `68410b0dd9bf01500255a1b08b459bac8c4216cf`.

## 5. Canonical selection (production, not hand-maintained)

`verify_block_completions`: A **24/24**, B **6/6**, C **3/3** -- 0
missing/ambiguous/corrupt/stale for all three. Block D (via
`authorize_block_d_cell` + `check_confirmatory_skip` keyed on the
effective config hash, per cell): **6/6 canonical, all `attempt_001`, 0
missing/ambiguous/corrupt/stale**. Full matrix: **39/39 canonical**.

## 6. Per-cell verification

Every cell independently re-verified in this audit: correct run ID and
both config hashes, correct source commit (`68410b0`) and protocol commit
(`ce4c962`), gate-decision commit/benchmark-source/benchmark-spec commits
recorded as distinct provenance fields, native official dataset checksum
match (`ac42d08f...` / `adace1e0...`), `resized=False`, correct class
count (9 / 8), `selected_batch_size=256`, all 6 required files present,
`artifact_manifest.json` hash-verified with no exception, checkpoint
independently restored via `map_location="cpu"` + strict
`load_state_dict` with no exception, effective config hash independently
recomputed via `compute_block_d_effective_config_hash()` and matched
exactly, ledger row `confirmatory=True`, `split=validation`,
`status=completed`, `test_metrics_observed=False`.

| Dataset | Seed | Attempt | Best epoch | Best val acc | Best val loss | Epochs completed | Early-stopping outcome | Active runtime (s) | Wall-clock (s) | Peak MPS (current/driver) | Checkpoint tensor hash | Effective config hash |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pathmnist | 0 | 1 | 1 | 0.8095 | 0.5444 | 6 | early-stopped, 5 epochs no improvement | 810.29 | 810.59 | 36,420,096 / 6,463,504,384 | `276169a8...` | `7c06f47b...` |
| pathmnist | 1 | 1 | 9 | 0.9212 | 0.2319 | 14 | early-stopped, 5 epochs no improvement | 1894.74 | 1895.02 | 36,271,616 / 6,463,504,384 | `7fc1cf8f...` | `688e0585...` |
| pathmnist | 2 | 1 | 11 | 0.9481 | 0.1501 | 16 | early-stopped, 5 epochs no improvement | 2379.17 | 2379.41 | 36,271,616 / 6,463,504,384 | `2c302516...` | `911e0140...` |
| bloodmnist | 0 | 1 | 4 | 0.9416 | 0.2972 | 9 | early-stopped, 5 epochs no improvement | 156.26 | 156.34 | 53,044,224 / 8,845,869,056 | `ec97e1b8...` | `a6dc0018...` |
| bloodmnist | 1 | 1 | 6 | 0.8440 | 0.4624 | 11 | early-stopped, 5 epochs no improvement | 191.67 | 191.76 | 53,044,224 / 8,845,869,056 | `f3980a44...` | `bf96e48a...` |
| bloodmnist | 2 | 1 | 4 | 0.9241 | 0.3326 | 9 | early-stopped, 5 epochs no improvement | 157.63 | 157.73 | 53,044,224 / 8,845,869,056 | `f0ddb47a...` | `aafade59...` |

Full per-epoch active runtime, for completeness:

- pathmnist s0: `[132.531, 136.660, 134.397, 134.580, 137.947, 134.168]`
- pathmnist s1: `[135.324, 134.708, 135.885, 134.591, 135.828, 135.427, 135.131, 136.068, 135.448, 135.383, 135.757, 135.113, 135.023, 135.042]`
- pathmnist s2: `[133.420, 138.181, 136.834, 135.958, 135.720, 135.822, 135.833, 165.961, 161.900, 155.976, 159.563, 156.095, 162.310, 164.334, 159.483, 141.767]`
- bloodmnist s0: `[17.923, 17.303, 17.283, 17.242, 17.241, 17.328, 17.290, 17.313, 17.333]`
- bloodmnist s1: `[17.539, 17.291, 17.647, 17.393, 17.355, 17.347, 17.364, 17.400, 17.446, 17.480, 17.410]`
- bloodmnist s2: `[17.567, 17.530, 17.891, 18.011, 17.326, 17.257, 17.315, 17.393, 17.341]`

Dependency versions (identical for all six cells): torch 2.13.0, kornia
0.8.3, medmnist 3.0.2. Device: `mps` for all six (no CPU fallback).

## 7. Runtime accounting

**Block D active compute total:** 810.29 + 1894.74 + 2379.17 + 156.26 +
191.67 + 157.63 = **5589.76 seconds (~93.16 minutes, ~1.553 hours)**.

**Updated A+B+C+D active compute total:** Block A ~98.90 min + Block B
~20.34 min + Block C ~5.62 min + Block D ~93.16 min = **~218.02 minutes
(~3.634 hours)** -- well under both the frozen pessimistic 24-hour
ceiling and the gate's own binding pessimistic-total figure (7.13 hours),
since the gate's projection was computed for a full, non-early-stopped
30-epoch run per cell, while every actual cell early-stopped well before
epoch 30 (range: 6-16 epochs of a 30-epoch maximum).

**Projected versus actual, per dataset (30-epoch-run projection vs.
this run's actual early-stopped active runtime):**

| Dataset | Projected training (30 epochs) | Projected end-to-end (30 epochs) | Actual active runtime range (this run) |
|---|---|---|---|
| pathmnist | 54.61 min | 56.70 min | 13.51 - 39.65 min (seeds 0-2) |
| bloodmnist | 7.21 min | 7.54 min | 2.60 - 3.20 min (seeds 0-2) |

**Idle/suspended gaps:** wall-clock exceeds active runtime by 0.08-0.30
seconds for every cell -- ordinary post-training overhead (checkpoint
save, manifest hashing, ledger append), not an idle/suspended gap. No
individual cell shows an unexplained runtime discrepancy.

## 8. Observed training behavior (neutral description)

Best validation accuracy across the six cells ranges from 0.8440
(bloodmnist seed 1) to 0.9481 (pathmnist seed 2). PathMNIST seed 0 shows
its best epoch at epoch 1, with declining/fluctuating validation accuracy
over epochs 2-6 before early stopping; PathMNIST seeds 1 and 2 show
validation accuracy improving over more epochs (best epoch 9 and 11
respectively) before plateauing. All three BloodMNIST seeds show a short
training curve (9-11 epochs) with best epoch in the first half of the
run. This document does not characterize this seed-to-seed variation as
expected or attribute it to any specific mechanism, and draws **no
conclusion about resolution or TTA** -- that comparison requires the
still-unperformed Validation-Gated TTA evaluation. Values are reported as
observed only.

## 9. Isolation, split-firewall, and provenance confirmation

- **No training or validation augmentation occurred:** all six cells use
  `training_policy=none`; confirmed via `metadata.json` (no
  `augmentation_policy` key) for every cell.
- **No test or TTA code was accessed:** `load_pilot_split()` (the only
  loader used, via `default_block_d_train_validation_loader_factory`) has
  no test-split access mechanism of any kind; `authorize_block_d_cell()`/
  `run_block_d_train_validation_cell()` contain no `evaluation.*`
  reference (confirmed via source inspection).
- **No Block D cell was skipped, retried, or reordered** based on an
  observed validation result -- all six ran strictly in frozen matrix
  order, each invoked exactly once.
- **No other Block D attempt exists:** exactly one `attempt_001` per run
  ID, six run-ID directories total under `artifacts/confirmatory/D/`.
- **Gate decision byte-identical throughout:** SHA-256
  `8bcc2810371e161c34ee6dbf8b03cb722f6f767c373a26d195e972db8b9eb7cf`,
  re-verified after every cell and at the end of this audit.
- **A/B/C remained unchanged:** re-verified 24/6/3 canonical after all
  six D cells; spot-checked canonical checkpoint MD5 unchanged
  (`eb7cfb6e23b691f0ffc6a64f23b5a77f` for
  `A-pathmnist-28px-batchnorm-policy-none-s0`/attempt_003).
- **Phase 2A ledger MD5 unchanged:** `e2dbdcd757cb13d77201c24cd746c05a`.
- **Downloaded 128px datasets and the raw benchmark JSON were only read,
  never modified:** file mtimes unchanged across all six training
  invocations (still their original prefetch/benchmark timestamps).
- **A benign `git diff --check` finding, noted for completeness:** the
  confirmatory ledger has used CRLF (`\r\n`) line endings since before
  this task began (confirmed via `git show HEAD:artifacts/ledger_confirmatory.csv`
  on every prior committed row, not only the newly appended ones) --
  git's default whitespace heuristic flags a lone CR-before-LF as
  "trailing whitespace" for any diff against such a file when
  `core.whitespace` doesn't include `cr-at-eol`. This is a pre-existing
  property of the ledger's CSV writer (Python's `csv` module default line
  terminator), not a defect introduced by Block D, and does not indicate
  corrupted or malformed row content -- every appended row's fields were
  independently parsed and verified correct in this audit regardless.
