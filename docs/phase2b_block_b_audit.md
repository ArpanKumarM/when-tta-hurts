# Phase 2B.3C Block B audit

This document is a mechanical audit of Block B's six confirmatory training
cells, produced against the production canonical-selection logic
(`orchestrator.verify_block_completions`), not a hand-maintained table.

**Validation results were not used for tuning.** No hyperparameter, seed,
architecture, augmentation parameter, or protocol setting was changed as a
result of any value in this document, at any point before or after
execution. Cells were run strictly in frozen matrix order; none was
skipped, reordered, or omitted based on an earlier cell's outcome.

## 1. Frozen six-cell manifest

Expanded via `parse_and_validate_matrix`, `training_matrix.B_policy_matching`
(`docs/phase2b_protocol.md` / `configs/experiment_matrix.yaml`):

| # | run_id | dataset | res | model | norm | seed | mapped Block A cell | config_hash |
|---|---|---|---|---|---|---|---|---|
| 1 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s0 | pathmnist | 28 | small_cnn | batchnorm | 0 | A-pathmnist-28px-batchnorm-policy-none-s0 | 35e38266... |
| 2 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s1 | pathmnist | 28 | small_cnn | batchnorm | 1 | A-pathmnist-28px-batchnorm-policy-none-s1 | 72cc68f7... |
| 3 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s2 | pathmnist | 28 | small_cnn | batchnorm | 2 | A-pathmnist-28px-batchnorm-policy-none-s2 | fe91da1e... |
| 4 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0 | bloodmnist | 28 | small_cnn | batchnorm | 0 | A-bloodmnist-28px-batchnorm-policy-none-s0 | 7efec8b1... |
| 5 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1 | bloodmnist | 28 | small_cnn | batchnorm | 1 | A-bloodmnist-28px-batchnorm-policy-none-s1 | a4282061... |
| 6 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2 | bloodmnist | 28 | small_cnn | batchnorm | 2 | A-bloodmnist-28px-batchnorm-policy-none-s2 | 0ab10924... |

All 6 use `training_policy=matched_to_approved_tta_policy` (the frozen
"mixed" augmentation policy). Matrix hash for all cells:
`ed9d36e6b3e0fdb4561de6ad70e75720502c97538bf7e6fdb67dd2bd4cb9045a`.
Protocol commit: `ce4c962dba3ac29dec5aae7f1680385035322bd8`. Source commit
for all six attempts: `a09758201b31b34b8f0cadd17d316c07b25ad6c0`.

## 2. Execution chronology

- **Cell 1 (canary), seed 0/pathmnist:** executed and independently audited
  in Phase 2B.3C Part 1 (prior session turn). Completed, attempt_001,
  checkpoint hash `f9d06b302a5a0a737e0476a01fa88cb1c309243f862de0f8f1a45e37ec88c47f`.
- **Cells 2-6:** executed sequentially in this session, strictly in frozen
  matrix order (pathmnist s1, pathmnist s2, bloodmnist s0, bloodmnist s1,
  bloodmnist s2), each via `scripts/run_confirmatory.py train-validation
  --run-id <exact_id>` (never a block-wide or broad command). Before each
  cell: confirmed no existing attempt directory, expected attempt number 1.
  After each cell, before starting the next: verified all 6 required
  artifacts, verified the artifact manifest, restored and verified the
  checkpoint, verified exactly one new completed ledger row, verified no
  stale/nonterminal attempt, verified no unexpected path. All 5 completed
  successfully on the first attempt -- no failure, interruption, staleness,
  OOM, non-finite value, or persistence error occurred, so the
  reconciliation mechanism was never invoked.

## 3. Canonical selection (production, not hand-maintained)

`orchestrator.verify_block_completions("B", expected_total=6)`:
**6/6 canonical, 0 missing, 0 ambiguous, 0 corrupt, 0 stale.** All six
canonical completions are `attempt_001`. `verify_block_completions("A",
expected_total=24)`: unchanged, still 24/24, 0/0/0/0.

## 4. Per-cell verification and results

Every cell independently re-verified in this audit (not merely trusted
from the runner's own report): correct full run ID and config hash,
correct source commit (`a09758201b31b34b8f0cadd17d316c07b25ad6c0`) and
protocol commit (`ce4c962`), native official dataset checksum match,
correct class count, all 6 required files present, `training_history.json`
complete, `result.json`/`metadata.json`/`status.json` schemas valid,
`artifact_manifest.json` hash-verified with no exception, checkpoint
independently restored via `map_location="cpu"` + strict
`load_state_dict` with no exception, ledger row `confirmatory=True`,
`split=validation`, `status=completed`, `test_metrics_observed=False`.

| Dataset | Seed | run_id / attempt | Best epoch | Best val acc | Best val loss | Epochs completed | Early-stopping outcome | Runtime (active) | Peak MPS (current/driver) | Checkpoint tensor hash | Mapped Block A cell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pathmnist | 0 | ...s0 / 1 | 1 | 0.6992 | 0.8419 | 6 | early-stopped, 5 epochs no improvement | 130.78s | 7,852,032 / 1,170,292,736 | f9d06b30... | A-pathmnist-28px-batchnorm-policy-none-s0 |
| pathmnist | 1 | ...s1 / 1 | 25 | 0.8561 | 0.4089 | 30 | early-stopped, 5 epochs no improvement | 644.95s | 7,777,024 / 1,170,292,736 | d8e1b91d... | A-pathmnist-28px-batchnorm-policy-none-s1 |
| pathmnist | 2 | ...s2 / 1 | 7 | 0.8047 | 0.5454 | 12 | early-stopped, 5 epochs no improvement | 261.33s | 7,777,024 / 1,170,292,736 | 44263c3e... | A-pathmnist-28px-batchnorm-policy-none-s2 |
| bloodmnist | 0 | ...s0 / 1 | 9 | 0.8557 | 0.3874 | 14 | early-stopped, 5 epochs no improvement | 34.51s | 5,964,032 / 1,155,612,672 | fe0bc5ac... | A-bloodmnist-28px-batchnorm-policy-none-s0 |
| bloodmnist | 1 | ...s1 / 1 | 23 | 0.8931 | 0.2868 | 28 | early-stopped, 5 epochs no improvement | 77.50s | 6,038,528 / 1,155,612,672 | 233311a2... | A-bloodmnist-28px-batchnorm-policy-none-s1 |
| bloodmnist | 2 | ...s2 / 1 | 20 | 0.9153 | 0.2674 | 25 | early-stopped, 5 epochs no improvement | 71.03s | 5,964,032 / 1,155,612,672 | 3d867a78... | A-bloodmnist-28px-batchnorm-policy-none-s2 |

## 5. Runtime accounting

Active compute time (sum of each cell's `result.json` `total_runtime_seconds`,
itself equal to the sum of `training_history.json`'s per-epoch
`epoch_runtime_seconds` for that cell) totals **1220.10 seconds (~20.34
minutes)** across all 6 cells.

For every cell, the ledger wall-clock span (`status.json`'s
`ended_at - started_at`) exceeds the active compute time by only
**0.08-0.09 seconds** -- this is attributable to ordinary post-training
overhead (checkpoint save, artifact persistence, manifest hashing, ledger
append) and does not constitute an idle/suspended gap requiring
investigation, unlike the two large multi-hour/multi-minute gaps found
during Block A's audit. No individual cell in Block B shows an
unexplained runtime discrepancy.

## 6. Ledger totals

`artifacts/ledger_confirmatory.csv`: **35 data rows** -- 32 `completed`
(26 from Block A + 6 from Block B), 1 `failed` (historical Block A
device-mismatch incident), 2 `aborted` (historical Block A interruption
reconciliations). Exactly one row per Block B run ID, all `attempt_id=1`.

## 7. Observed training curves and seed variation (neutral description)

Best validation accuracy across the six cells ranges from 0.6992
(pathmnist seed 0) to 0.9153 (bloodmnist seed 2). Best epoch ranges from
1 (pathmnist seed 0) to 25 (pathmnist seed 1); epochs completed before
early stopping ranges from 6 to 30. Pathmnist seed 0's training curve
shows its highest validation accuracy at epoch 1, with lower and
fluctuating accuracy in epochs 2-6 before early stopping triggers.
Pathmnist seed 1 and seed 2, and all three bloodmnist seeds, show
validation accuracy generally increasing over more epochs before
plateauing. This document does not characterize this variation as
expected, confirmatory, or attribute it to any specific mechanism --
doing so was not preregistered, and no post-hoc tuning, decomposition, or
follow-up analysis was performed to explain it. The values are reported
as observed.

## 8. Isolation, split-firewall, and augmentation-wiring confirmation

- **Augmentation applied exactly once, training only:** confirmed from
  code-path inspection in Phase 2B.3C Part 1's audit (`training.py`'s
  `x = sample_deterministic_view(x, ...)` replaces the training batch
  exactly once per step; `_evaluate_loss_accuracy`, used for validation,
  takes no `augmentation_policy` parameter at all). No code was changed
  between that audit and this execution, so the same wiring applies to
  all 6 cells including the 5 executed in this session.
- **No TTA/aggregation/BN-adaptation path ran:** `training.py` imports
  only `transforms.policies`; `orchestrator.py` has zero references to
  `evaluation.*` anywhere in the training path.
- **No test-split access occurred:** `load_pilot_split()` (the only
  loader used) has no test-split access mechanism of any kind.
- **No second block began:** `artifacts/confirmatory/` contains only `A/`
  and `B/` -- no `C` or `D` directory exists anywhere.
- **All Block A artifacts remained unchanged:** attempt-directory count
  unchanged (29), spot-checked canonical checkpoint MD5 unchanged
  (`eb7cfb6e23b691f0ffc6a64f23b5a77f` for `A-pathmnist-28px-batchnorm-policy-none-s0`/attempt_003).
- **Phase 2A ledger MD5 unchanged:** `e2dbdcd757cb13d77201c24cd746c05a`.
