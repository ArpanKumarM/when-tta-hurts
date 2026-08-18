# Phase 2B.3B Block A audit

This document is a mechanical audit of every attempt directory produced by
Block A execution, run against the production canonical-selection logic
(`orchestrator.verify_block_completions` / `check_confirmatory_skip`) --
not a hand-maintained table. It also reconciles the two attempts that were
externally interrupted during unattended execution and accounts for the
wall-clock/idle-time discrepancy those interruptions produced.

**Validation results were not used for tuning.** Every fact in this
document was established from `status.json`, `result.json`,
`training_history.json`, `artifact_manifest.json`, and the ledger/amendment
CSVs -- never by choosing among candidates based on validation accuracy or
loss. No hyperparameter, seed, architecture, or protocol setting was
changed as a result of any value reported here.

## 1. Mechanical attempt inventory

Every attempt directory under `artifacts/confirmatory/A/` for all 24
Block A run IDs was enumerated directly from disk. 23 of 24 run IDs have
exactly one attempt (`attempt_001`, completed, ledger row present, no
amendment). The following run IDs have more than one attempt:

### `A-pathmnist-28px-batchnorm-policy-none-s0` (Cell 1) -- 4 attempts

| Attempt | status.json state | started_at | ended_at | Ledger row | Amendment |
|---|---|---|---|---|---|
| 1 | completed | (pre-Part-2 instrumentation; no result.json) | -- | completed, `checkpoint_hash=30bc1ca6...` | attempt 1: `canonical_eligible=false`, `engineering_observability_failure` |
| 2 | failed | 1787019624.106598 | 1787019700.070153 | failed, `runtime_seconds=75.96`, `failure_reason="Expected all tensors to be on the same device..."` | none |
| 3 | completed | 1787020418.885462 | 1787020494.54256 | completed, `checkpoint_hash=30bc1ca6...` | none |
| 4 | completed | 1787020534.221334 | 1787020610.516691 | completed, `checkpoint_hash=30bc1ca6...` | attempt 4: `canonical_eligible=false`, `unintended_duplicate_skip_search_defect` |

Config hash for all four attempts: `03154180ddda338c2b893d9938a45e21a0035884ea7a04e317392e54caab3059`.
All four attempts' checkpoints are bitwise-identical (`30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e`).

### `A-pathmnist-28px-groupnorm-policy-none-s1` -- 2 attempts

| Attempt | status.json state | started_at | ended_at | Artifacts present | Ledger row |
|---|---|---|---|---|---|
| 1 | **running** (never updated) | 1787022897.467121 | *(none -- never written)* | status.json only | none until this audit (see section 3) |
| 2 | completed | 1787023299.391535 | 1787023576.335745 | all 6 | completed, `checkpoint_hash=b63d8c9b9a38...` |

### `A-pathmnist-64px-groupnorm-policy-none-s2` -- 2 attempts

| Attempt | status.json state | started_at | ended_at | Artifacts present | Ledger row |
|---|---|---|---|---|---|
| 1 | **running** (never updated) | 1787034414.4075708 | *(none -- never written)* | status.json only | none until this audit (see section 3) |
| 2 | completed | 1787057301.221015 | 1787059715.4194229 | all 6 | completed, `checkpoint_hash=92c8739bfc10...` |

No facts were inferred or fabricated for the two `attempt_001` directories
above -- everything not directly readable from the existing `status.json`
(which was written exactly once, at attempt start, and never updated) is
reported as unknown in section 3 below.

## 2. Corrected canonical 24-cell table

Regenerated directly from `orchestrator.verify_block_completions("A", expected_total=24)`
(the exact production selection function used by the runner's own skip
logic), immediately after the section-3 reconciliation rows were appended.
Result: **24/24 canonical, 0 missing, 0 ambiguous, 0 corrupt.**

| run_id | canonical attempt | checkpoint_hash |
|---|---|---|
| A-pathmnist-28px-batchnorm-policy-none-s0 | **3** | 30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e |
| A-pathmnist-28px-batchnorm-policy-none-s1 | 1 | f3be88438078ce362528dc8c3919e7d395fb095902cca48159c861b6e308e6bf |
| A-pathmnist-28px-batchnorm-policy-none-s2 | 1 | b8b971407b6b149d8e64e4bf55ec87f95a753cad5256aa5f3a57d419145d9c06 |
| A-pathmnist-28px-groupnorm-policy-none-s0 | 1 | fcf6a2f41c136cadc012bab8726249062ed1a16290a98504b65903a96c234e98 |
| A-pathmnist-28px-groupnorm-policy-none-s1 | 2 | b63d8c9b9a38b3520f5aeb971a4afb60d760a7bffc707a63ce6c344abe5a4650 |
| A-pathmnist-28px-groupnorm-policy-none-s2 | 1 | 483eaa5957f570981a43933d3985b714b56911abc75ccbdebf5db8c6cdc9cdd3 |
| A-pathmnist-64px-batchnorm-policy-none-s0 | 1 | b5dea833b8985d7bf38e4092e5400a03452b0ec55a79ec7e52edd0091e6b3b54 |
| A-pathmnist-64px-batchnorm-policy-none-s1 | 1 | b898155b2c6ab101ca2940f30bceff8a23a7f67ef9adc56d7c07ab2275831c8e |
| A-pathmnist-64px-batchnorm-policy-none-s2 | 1 | 39950a5ac56c50aeee11d3a6ca879baabfcc19506ce041f046fbff10243fe293 |
| A-pathmnist-64px-groupnorm-policy-none-s0 | 1 | d54b5cd32b64882bc11030992db97ce57973df2078e37317871d1c66c92e484b |
| A-pathmnist-64px-groupnorm-policy-none-s1 | 1 | ebfef2c0e60ae1ee1095f6c47afd6dbad3c7abc2c32ebc5f21e502a9f33ee44e |
| A-pathmnist-64px-groupnorm-policy-none-s2 | **2** | 92c8739bfc1063b03782514c7c2b259cb3014ea5f0262f64e138be29e8c1ecd2 |
| A-bloodmnist-28px-batchnorm-policy-none-s0 | 1 | e96e3b5615c71fba0fd05d21439c3d2a551ddba00985dbcc0b8affcfda5c519f |
| A-bloodmnist-28px-batchnorm-policy-none-s1 | 1 | 1448dad9629f956c195e301dabe42047502ccf2981d6f4f3154f5994a35da98f |
| A-bloodmnist-28px-batchnorm-policy-none-s2 | 1 | 552c975c860585e9b57298d2e1fd3550420b9ad696066d2fe7bef49401a235f0 |
| A-bloodmnist-28px-groupnorm-policy-none-s0 | 1 | cd804a09e799f7d89c1e98dad10a63a953a9a5204d98d5fa4c5677b7937765bd |
| A-bloodmnist-28px-groupnorm-policy-none-s1 | 1 | d914f4ecd0d597b9cc422dd7e5b07413c900eaa084b0f0992575010dfd899928 |
| A-bloodmnist-28px-groupnorm-policy-none-s2 | 1 | 34a1b18158c3572f281035592f8a3af33eb2baa11055aa9a8c2e6a60d495741d |
| A-bloodmnist-64px-batchnorm-policy-none-s0 | 1 | 8572b19b0cb24b9c46ae07a1114eaa65ddad7690826265fe3af8984b2d0b6235 |
| A-bloodmnist-64px-batchnorm-policy-none-s1 | 1 | ff75298a2d1733dd1e43dd60d90a74a51450f80704c1e648e41eabf9ba7f4017 |
| A-bloodmnist-64px-batchnorm-policy-none-s2 | 1 | b2e83aa7e85050f6c3743c9e0a269428b3e1795d369a622328eb5b5e2abe2869 |
| A-bloodmnist-64px-groupnorm-policy-none-s0 | 1 | 6dc1eb925bf71525353ddec56ac41b78e4d3e04b761ad47fb13ca5ae8d1443b0 |
| A-bloodmnist-64px-groupnorm-policy-none-s1 | 1 | 2278d9cfd09a165210e53528eaa0f8567dba6b7303c0f5b21a09a20a652ec98b |
| A-bloodmnist-64px-groupnorm-policy-none-s2 | 1 | 9cbeb71c38c6d50873d5972ced6ca962cc1f38cde2bd080362351dd06df20fd9 |

Cell 1's canonical completion is confirmed as **attempt_003**;
`attempt_004` remains excluded by its pre-existing amendment
(`canonical_eligible=false`, `unintended_duplicate_skip_search_defect`).
Production selection was re-verified to choose attempt_003 (not
attempt_004, not attempt_1) before any ledger row in this document was
committed.

## 3. Interrupted-attempt reconciliation

Both interrupted attempts were audited from existing files only -- no
timestamp, duration, or outcome was inferred beyond what the files
directly contain.

### `A-pathmnist-28px-groupnorm-policy-none-s1` / `attempt_001`

- **Last recorded state:** `status.json`, written once, `status="running"`,
  `ended_at=null`. File mtime (`Aug 17 23:14:57 2026`) is identical to
  `started_at` (`1787022897.467121`), confirming the file was never
  rewritten after creation.
- **Start time:** `1787022897.467121`. **End time:** unknown -- `ended_at`
  was never written, and no other artifact exists to establish one.
- **Training completed:** No. No `training_history.json`, `result.json`,
  `metadata.json`, or `artifact_manifest.json` exists in this attempt
  directory -- only `status.json`.
- **Validation metrics persisted or observed:** No. No result artifact of
  any kind was ever written for this attempt.
- **Checkpoint exists:** No. `best_checkpoint.pt` does not exist in this
  attempt directory.
- **Resumed or fresh:** `attempt_002` was a **completely fresh run**, not
  a resume. The codebase has no checkpoint-resume capability at all --
  `run_train_validation_cell` always calls `seed_everything(cell.seed)`
  and constructs a brand-new model via `_build_model(cell)` before every
  attempt; `attempt_002`'s `started_at` (`1787023299.391535`) is 401.9
  seconds after `attempt_001`'s `started_at`, i.e. a normal-length training
  cell's worth of time later, consistent with a new invocation starting
  cleanly rather than resuming stale state.
- **Most specific evidence-backed interruption reason:** the invoking
  process was externally terminated by the Bash tool's 10-minute
  interactive-command timeout, which killed the very first
  `train-validation --block A` invocation of this session while this cell
  was mid-training (SIGTERM/SIGKILL, not a Python exception -- consistent
  with `run_identity.py`'s documented limitation that SIGKILL/external
  termination cannot be caught by the normal failure-handling path, so no
  incident/ledger row was written at the time).

### `A-pathmnist-64px-groupnorm-policy-none-s2` / `attempt_001`

- **Last recorded state:** `status.json`, written once, `status="running"`,
  `ended_at=null`. File mtime (`Aug 18 02:26:54 2026`) is identical to
  `started_at` (`1787034414.4075708`).
- **Start time:** `1787034414.4075708`. **End time:** unknown, for the
  same reason as above.
- **Training completed:** No -- same evidence pattern as above (only
  `status.json` exists in the attempt directory).
- **Validation metrics persisted or observed:** No.
- **Checkpoint exists:** No.
- **Resumed or fresh:** `attempt_002` was a **completely fresh run** (same
  structural argument as above). `attempt_002`'s `started_at`
  (`1787057301.221015`) is 22,886.8 seconds (~6.36 hours) after
  `attempt_001`'s `started_at` -- a much longer gap than the previous
  case, consistent with this attempt having been interrupted by an
  operator killing the background process, with the next invocation
  coming later in a separate terminal session rather than an automatic
  resume.
- **Most specific evidence-backed interruption reason:** the background
  training process was deliberately terminated (operator decision,
  recorded in this session), not a crash or exception -- `attempt_002` was
  only started once a person re-invoked the block command later.

### Reconciliation ledger rows appended

Two terminal `status=aborted` rows were appended to
`artifacts/ledger_confirmatory.csv` (via `ledger.append_confirmatory_entry`,
the same production function used for every other row, to guarantee exact
schema conformance):

```
confirmatory=True, run_id=A-pathmnist-28px-groupnorm-policy-none-s1, attempt_id=1,
status=aborted, checkpoint_hash=(blank), started_at=1787022897.467121,
ended_at=(blank), runtime_seconds=(blank),
validation_metrics_observed=False, test_metrics_observed=False

confirmatory=True, run_id=A-pathmnist-64px-groupnorm-policy-none-s2, attempt_id=1,
status=aborted, checkpoint_hash=(blank), started_at=1787034414.4075708,
ended_at=(blank), runtime_seconds=(blank),
validation_metrics_observed=False, test_metrics_observed=False
```

`ended_at` and `runtime_seconds` are left blank rather than estimated,
because they are genuinely unknown -- not merely unrecorded by an
oversight. Neither row is marked `completed` or canonical-eligible;
`status=aborted` structurally excludes both rows from
`check_confirmatory_skip`'s candidate set (which only considers rows with
`status=="completed"`), so no amendment-ledger entry was needed to exclude
them. Neither attempt directory was modified, renamed, or deleted.

## 4. Active-runtime versus wall-clock accounting

Two cells show a large gap between the confirmatory-ledger's wall-clock
span (`ended_at - started_at`) and the actual training compute time
recorded in `result.json`'s `total_runtime_seconds` (itself the sum of
`training_history.json`'s per-epoch `epoch_runtime_seconds`, confirmed
below). This gap is idle/suspended wall-clock time -- it is **not**
training compute and is reported separately, never merged into a runtime
figure.

| run_id (canonical attempt) | Ledger wall-clock span | Active training compute (= sum of per-epoch runtimes) | Idle/suspended time |
|---|---|---|---|
| A-pathmnist-64px-groupnorm-policy-none-s1 (attempt 1) | 7340.74s | 990.63s | 6350.10s (~105.8 min) |
| A-pathmnist-64px-groupnorm-policy-none-s2 (attempt 2) | 2414.20s | 941.07s | 1473.13s (~24.6 min) |

For both cells, `sum(epoch_runtime_seconds)` from `training_history.json`
matches `result.json`'s `total_runtime_seconds` to within floating-point
rounding (990.63s vs 990.63s; 941.06s vs 941.07s) -- **no individual
epoch's recorded runtime contains any part of either gap.** Per-epoch
runtimes for both cells range 31-55 seconds, consistent with every other
64px GroupNorm cell in Block A; neither cell has an anomalously large
single-epoch value. The idle time therefore occurred entirely outside the
`train_model()` epoch loop -- somewhere in checksum verification, dataset
loading, or artifact persistence/verification, or (most plausibly, given
the magnitude and this session's independently-recorded process-management
history) because the machine was asleep/suspended while the owning process
was between pipeline stages. No finer-grained timestamp exists in any
artifact to localize the gap further than "outside the epoch loop,"
and this document does not claim more precision than that.

Block-A-wide (23 newly-trained canonical attempts, excluding Cell 1's
pre-existing attempt_003):

- **Sum of active training compute:** ~98.9 minutes (unchanged from the
  prior session report -- this figure already used `result.json`'s
  `total_runtime_seconds`, never ledger wall-clock time).
- **Sum of ledger wall-clock spans:** ~232.6 minutes.
- **Total externally idle/suspended time:** ~133.7 minutes, entirely
  attributable to the two gaps documented above (105.8 + 24.6 = 130.4 min,
  plus normal per-cell overhead such as checksum verification and artifact
  persistence for the remaining 21 cells accounting for the residual ~3.3
  min).

## 5. Ledger counts and artifact verification

- `artifacts/ledger_confirmatory.csv`: **29 data rows** -- 26 `completed`,
  1 `failed`, 2 `aborted`.
- `artifacts/ledger_amendments.csv`: 2 rows, both for Cell 1
  (`attempt_id=1`, `attempt_id=4`), unchanged by this audit.
- `orchestrator.verify_block_completions("A", expected_total=24)`:
  **24/24 canonical-eligible, 0 missing, 0 ambiguous, 0 corrupt.**
- Every one of the 24 canonical completions was independently re-verified
  in this audit against: correct dataset-specific class count
  (PathMNIST=9, BloodMNIST=8, via `data.get_dataset_metadata`), correct
  native artifact and checksum (`dataset_verification.verify_official_dataset_artifact`),
  expected config hash (`run_identity.cell_config_hash`), expected matrix
  hash and protocol commit (from each attempt's own `result.json`), all
  six required artifacts present, artifact manifest hash-verified
  (`result_artifacts.verify_artifact_manifest`), `status.json` reads
  `completed`, and canonical selection uniquely resolved (zero ambiguous
  entries). All checks passed for all 24 cells.
- `artifacts/confirmatory/` contains only a Block A (`A/`) directory --
  no Block B, C, or D directories exist. No TTA-related file exists
  anywhere under `artifacts/confirmatory/`. No test-split access occurred
  at any point (structurally impossible via `load_pilot_split`, which has
  no test-split mechanism).
