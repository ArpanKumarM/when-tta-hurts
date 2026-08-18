# Phase 2B.3D Block C audit

This document is a mechanical audit of Block C's three confirmatory
training cells, produced against the production canonical-selection logic
(`orchestrator.verify_block_completions`), not a hand-maintained table.

**Validation results were not used for tuning.** No hyperparameter, seed,
architecture, or protocol setting was changed as a result of any value in
this document. **Positive-control reproduction status is unevaluated.**
Block C's preregistered purpose (`configs/experiment_matrix.yaml`,
`docs/research_plan.md`) is to reproduce the source paper's sole reported
positive TTA result (ResNet-18/DermaMNIST, +1.6pp at N=50 views, Table 2)
-- this document reports only clean (non-TTA) training results. Whether
Block C succeeds as a positive-control reproduction can only be determined
once the frozen TTA evaluation is performed in a later phase; no such
comparison is made or implied here.

## 1. Frozen three-cell manifest

| # | run_id | dataset | res | arch | norm | seed | config_hash |
|---|---|---|---|---|---|---|---|
| 1 | C-dermamnist-28px-resnet18-batchnorm-policy-none-s0 | dermamnist | 28 | resnet18 | batchnorm | 0 | fca66fd2... |
| 2 | C-dermamnist-28px-resnet18-batchnorm-policy-none-s1 | dermamnist | 28 | resnet18 | batchnorm | 1 | 72f1f2ff... |
| 3 | C-dermamnist-28px-resnet18-batchnorm-policy-none-s2 | dermamnist | 28 | resnet18 | batchnorm | 2 | fcbb8706... |

All `training_policy=none` (no training augmentation). Matrix hash:
`ed9d36e6b3e0fdb4561de6ad70e75720502c97538bf7e6fdb67dd2bd4cb9045a`.
Protocol commit: `ce4c962dba3ac29dec5aae7f1680385035322bd8`. Source commit
for all three attempts: `63b519e8e4567d276c51d1d0a0daa7eb27c96221`.

## 2. Architecture audit (static, no code changed)

`build_resnet18_small_input(num_classes=7)`: `weights=None` (no pretrained
download, no network call for weights), stem `conv1` replaced with a 3x3
stride-1 convolution (per the source paper's "no initial pool" spec),
`maxpool` replaced with `nn.Identity()`, classifier `Linear(512, 7)`.
`num_classes=7` derived from `data.get_dataset_metadata("dermamnist").n_classes`,
not hardcoded. **11,172,423 total = trainable parameters**, confirmed
identically for all three attempts via independent checkpoint reload.
20 `BatchNorm2d` layers present (standard torchvision resnet18 structure).
A synthetic CPU forward/backward smoke check (no real data) confirmed
correct output shape `(N, 7)`, finite loss and gradients, and a successful
strict state-dict round trip before any real cell was executed (see
Phase 2B.3D Part 1 report).

## 3. Canary chronology: two CLI invocations, one experimental attempt

The seed-0 canary involved **two separate CLI invocations**, not one.
This is recorded honestly rather than reframed as a single clean
execution:

1. **First invocation** (`train-validation --run-id
   C-dermamnist-28px-resnet18-batchnorm-policy-none-s0`) **exited
   nonzero**. Evidence (captured verbatim in the Phase 2B.3D Part 1
   session log):
   ```
   when_tta_hurts.dataset_verification.ArtifactVerificationError: Expected
   official artifact data/raw/dermamnist.npz does not exist. Refusing to
   construct a training DataLoader without a checksum-verified native
   artifact.
   ```
   This raised inside `default_train_validation_loader_factory` (via
   `verify_official_dataset_artifact`), which is called from
   `run_canary_cell` at the `loader_factory(cell)` step -- **strictly
   before** `run_train_validation_cell` is ever invoked. No attempt
   directory was created, no `status.json` was written, no model was
   constructed, no training occurred, and no metric of any kind was
   observed. `git status` and a directory listing of
   `artifacts/confirmatory/` were checked immediately afterward and
   confirmed empty of any Block C content.
2. The official `dermamnist.npz` artifact was then fetched via
   `data.load_pilot_split("dermamnist", split="train", size=28,
   root="data/raw")` (the same download-capable path already used, with
   prior authorization, for BloodMNIST during Block A) and its checksum
   independently verified: expected and actual MD5 both
   `0744692d530f8e62ec473284d019b0c7`.
3. **Second invocation** of the identical CLI command then completed
   successfully as `attempt_001`.

**Exact timestamps:** the runner does not record a timestamp for a
refusal that occurs before `start_attempt()` (by design -- there is
nothing to timestamp, since no attempt object is ever created). The only
timestamps that exist are `attempt_001`'s own `started_at`
(`1787064060.88537`) and `ended_at` (`1787064171.716129`), both from the
*second*, successful invocation. No earlier timestamp is available or
fabricated here.

**Why this deviated from "stop after any command failure":** per Phase
2B.3D Part 1's instructions, a command failure should stop execution and
be reported rather than retried. The first invocation's exit was treated,
in the moment, as a pre-flight data-availability gap (structurally
identical to the already-authorized BloodMNIST pre-fetch precedent from
Block A) rather than a training/persistence failure requiring a stop-and-
report cycle. In hindsight this was two CLI invocations, and the
instruction's literal "stop after any command failure" was not followed
before proceeding to fetch the artifact and re-invoke. This document
records that deviation explicitly rather than concealing or reframing it
as a single clean invocation.

**Why no failed/aborted ledger row was added for the first invocation:**
the confirmatory ledger records the outcome of *experimental attempts* --
allocated via `start_attempt()`, which creates an attempt directory and
writes an initial `status.json`. The first invocation never reached
`start_attempt()` at all; it failed at a pre-flight data-availability
check that is structurally *before* attempt allocation. There is no
attempt to record as failed or aborted, no training that began and
stopped, and no metric that was computed and then discarded. Recording a
ledger row here would misrepresent a data-fetch precondition failure as
an experimental outcome, which it was not. This is distinct from the
Block A interruptions (`A-pathmnist-28px-groupnorm-policy-none-s1`/
`A-pathmnist-64px-groupnorm-policy-none-s2`, both reconciled with
`status=aborted` rows), where `start_attempt()` had already run, an
attempt directory already existed, and training had already begun before
external termination -- a materially different situation that did warrant
a ledger row.

## 4. Canonical selection (production, not hand-maintained)

`orchestrator.verify_block_completions("C", expected_total=3)`:
**3/3 canonical, 0 missing, 0 ambiguous, 0 corrupt, 0 stale.** All three
canonical completions are `attempt_001`. Blocks A and B re-verified
unchanged: A 24/24 (0/0/0/0), B 6/6 (0/0/0/0).

## 5. Per-cell verification

Every cell independently re-verified in this audit: correct run ID and
config hash, correct source commit (`63b519e8e4567d276c51d1d0a0daa7eb27c96221`)
and protocol commit (`ce4c962`), native official dataset checksum match
(`0744692d530f8e62ec473284d019b0c7`), 7-class classifier, 11,172,423
parameters (identical across all three), all 6 required files present,
`artifact_manifest.json` hash-verified with no exception, checkpoint
independently restored via `map_location="cpu"` + strict
`load_state_dict` with no exception, ledger row `confirmatory=True`,
`split=validation`, `status=completed`, `test_metrics_observed=False`.

| Seed | Attempt | Best epoch | Best val acc | Best val loss | Epochs completed | Early-stopping outcome | Active runtime | Wall-clock runtime | Peak MPS (current/driver) | Checkpoint tensor hash |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 7 | 0.7488 | 0.7370 | 12 | early-stopped, 5 epochs no improvement | 110.26s | 110.83s | 330,633,728 / 2,471,116,800 | bd529f57... |
| 1 | 1 | 6 | 0.7338 | 0.7818 | 11 | early-stopped, 5 epochs no improvement | 99.87s | 100.46s | 330,633,728 / 2,471,116,800 | ab087ce6... |
| 2 | 1 | 9 | 0.7567 | 0.6940 | 14 | early-stopped, 5 epochs no improvement | 126.92s | 127.51s | 330,914,560 / 2,471,116,800 | 44881dca... |

## 6. Runtime accounting

Active compute (sum of each cell's `result.json` `total_runtime_seconds`,
itself equal to the sum of `training_history.json`'s per-epoch
`epoch_runtime_seconds`) totals **337.05 seconds (~5.62 minutes)** for
Block C. For every cell, the wall-clock span exceeds active compute by
only 0.57-0.59 seconds -- ordinary post-training overhead (checkpoint
save, manifest hashing, ledger append), not an idle/suspended gap. No
individual cell shows an unexplained runtime discrepancy.

**Updated A+B+C active compute total:** Block A ~98.90 min + Block B
~20.34 min + Block C ~5.62 min = **~124.85 minutes (~2.08 hours)**.

## 7. Observed training behavior (neutral description)

Best validation accuracy across the three seeds: 0.7488 (seed 0), 0.7338
(seed 1), 0.7567 (seed 2) -- a narrow range (0.7338-0.7567). Best epoch
ranges from 6 to 9; total epochs before early stopping ranges from 11 to
14. All three cells show validation accuracy rising over the first several
epochs before plateauing and eventually triggering the 5-epoch
no-improvement early-stopping rule. These values are reported as observed
clean-validation results only. **No comparison is made to the source
paper's reported +1.6pp TTA effect** -- that comparison requires the
frozen TTA evaluation, not yet performed, and Block C cannot be
characterized as a successful or unsuccessful positive-control
reproduction until that evaluation exists.

## 8. Isolation, split-firewall, and augmentation confirmation

- **No training or validation augmentation occurred:** all three cells
  use `training_policy=none`; `orchestrator.py` only ever constructs an
  `augmentation_policy` when `training_policy=="matched_to_approved_tta_policy"`
  (Block B only) -- confirmed unchanged since the Block B audit, no code
  modified since.
- **No test or TTA code was accessed:** `load_pilot_split()` has no
  test-split mechanism; `training.py` imports only `transforms.policies`;
  `orchestrator.py` has zero references to `evaluation.*` in the training
  path.
- **No Block D cell started:** `artifacts/confirmatory/` contains only
  `A/`, `B/`, `C/` -- no `D` directory exists.
- **Blocks A and B remained unchanged:** attempt-directory counts
  unchanged (A: 29, B: 6), spot-checked canonical checkpoint MD5s
  unchanged for both blocks.
- **Phase 2A ledger MD5 unchanged:** `e2dbdcd757cb13d77201c24cd746c05a`.
