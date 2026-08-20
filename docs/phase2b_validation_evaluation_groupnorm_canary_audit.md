# Phase 2B.4F GroupNorm Canary: corrected evaluation confirmed valid

**Recorded: 2026-08-20.** This document records the first technically
valid, canonical completion of the corrected GroupNorm persistence-
schema contract for `A-pathmnist-28px-groupnorm-policy-none-s0`,
together with the honest history of the failed attempt that preceded
it. No source code is changed by this document. The test split was
never accessed at any point. No remaining Block A cell was executed
while producing this record.

## 1. Attempt history

| Attempt | evaluation_id | Status | Runtime |
|---|---|---|---|
| 1 | `2bb65453d1d5fe03186ec008cbd4006416f889282d26e152cc0d09e59b8b7b4b` | `failed` | 4165.632166147232s |
| 2 | `db274d0aba7d32dc65ee9a6406d0842137602ad15ad9b9657115ff67485520ef` | `completed`, canonical-eligible | 4102.956974983215s |

Attempt 1 failed at the final persistence step with:
```
EvaluationSchemaValidationError: batching.bn_adaptation_microbatches_at_primary_n
must be a nonnegative integer, got None.
```
This was a genuine, previously-latent defect -- GroupNorm cells never
run BN-adaptation, so `bn_adaptation_microbatch_counts.get(PRIMARY_N)`
returned `None` (no `.get(..., 0)` default existed), and the schema
validator unconditionally required a nonnegative integer. This was the
first time any GroupNorm cell had ever reached the real production
persistence path. Full root-cause trace:
`docs/phase2b_validation_evaluation_groupnorm_persistence_incident.md`.

Attempt 1's directory (`.../attempt_001/`) contains only `status.json`
-- no partial or corrupt artifact was ever written, and it remains
byte-unchanged by this or any later attempt.

## 2. Correction commits

| Commit | Content |
|---|---|
| `e9f061b` | Research record: attempt-1 failure + the two successful sibling BatchNorm cells (`-s1`, `-s2`), preserved honestly, no amendment added |
| `14c5c19` | Documentation-only freeze: `docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md` defines the `bn_adaptation_applicable`/`bn_adaptation_microbatches_at_primary_n` contract, bound to `metadata.normalization`, before any code changed |
| `d5227d0` | Engineering fix: defaults the microbatch count to `0` (never `None`) when BN-adaptation did not run; adds `bn_adaptation_applicable` as a cross-checked, fail-closed field; adds `_validate_bn_adaptation_applicability_consistency()`; 14 new regression tests; full suite 783 passed |

The two already-completed BatchNorm cells
(`A-pathmnist-28px-batchnorm-policy-none-s1`, `-s2`) were **not**
rerun or amended -- the corrected code branch was never reachable for
them.

## 3. Evaluation identity

| Field | Value |
|---|---|
| Training run | `A-pathmnist-28px-groupnorm-policy-none-s0` |
| Canonical training attempt / checkpoint hash | `1` / `fcf6a2f41c136cadc012bab8726249062ed1a16290a98504b65903a96c234e98` |
| Evaluation attempt | `2` |
| Evaluation ID / evaluation-config hash | `db274d0aba7d32dc65ee9a6406d0842137602ad15ad9b9657115ff67485520ef` |
| Evaluator fingerprint | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` (new -- reflects the `validation_evaluation.py`/`evaluation_result_artifacts.py` correction) |
| Frozen TTA seed | `1306178015` |
| Metric-input contract | `probability_native_v1` |
| Dataset / resolution / split | `pathmnist` / 28 / `validation` |
| Runtime | `4102.956974983215` s (~1h8m) |

### Artifact manifest

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `predictions.npz` | 36,535,640 | `38f9368ebe5268da449d33b75a5a5e04c11c84ca151ccb5582b96a555c8de51a` |
| `metrics.json` | 13,546 | `5c28f32ec779621cd6211256c0cea6835d88f41e828a5f626d13380a5df37172` |
| `metadata.json` | 4,692 | `3fb69a70ec6a0e1cbf3363102616c5875cd4a3d5d7a65cbd8a0e024feca226a7` |
| `view_manifest.json` | 99,517 | `764f531cb816fa47cced92a7faa40090489bf46143bf39a92a31a0d8a7385b1e` |

Location: `artifacts/validation_evaluation/A-pathmnist-28px-groupnorm-policy-none-s0/attempt_002/`.

## 4. All 14 independent verification checks (passed)

1. Artifact manifest independently reverified (`verify_evaluation_artifact_manifest`) -- OK.
2. Clean-metrics semantic recomputation from persisted probabilities matches `metrics.json` exactly (accuracy `0.9302279088364654`).
3. N=50 mean-probability semantic recomputation matches exactly (accuracy `0.3666533386645342`).
4. `metadata.batching.bn_adaptation_applicable == False`.
5. `metadata.batching.bn_adaptation_microbatches_at_primary_n == 0` (never `None`).
6. `metrics.conditions.bn_adapted_tta is None` -- no BN-adapted metric of any kind reported.
7. `bn_adapted_probs` absent from `predictions.npz`.
8. `bn_adapted_prefix_sequence` absent from `predictions.npz`.
9. `dataset_verification.checksum_verified == True`.
10. `dataset_verification.resized == False`.
11. `checkpoint_hash` matches `resolve_canonical_training_completion()`'s canonical training result exactly.
12. `tta_seed == 1306178015`.
13. `metric_input_contract == "probability_native_v1"`.
14. `check_evaluation_skip(run_id, evaluation_id)` resolves to attempt 2 as the sole canonical-compatible completion (no ambiguity, no conflict).

`test_metrics_observed = False` on the ledger row. Attempt 1 confirmed byte-unchanged (same `failure_reason`, same `evaluation_id`, directory unchanged).

## 5. Complete results (attempt 2, exact persisted values)

### Clean

| Metric | Value |
|---|---|
| Accuracy | `0.9302279088364654` |
| Macro-F1 | `0.9302851099402862` |
| NLL | `0.2084379345178604` |
| ECE | `0.016133113375834052` |
| Brier | `0.10625410079371134` |

### Primary endpoint

```text
Clean accuracy:                          0.9302279088364654
TTA N=50 mean-probability accuracy:      0.3666533386645342
TTA N=50 mean-probability delta accuracy: -0.5635745701719312 (approximately -56.36pp)
```

### Mean-probability aggregation, all registered prefixes

| N | Accuracy | Macro-F1 | Delta-acc | Harm rate | Rescue rate | NLL | ECE | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.28328668532586965 | 0.24458270840038418 | -0.6469412235105958 | 0.7061035890823125 | 0.14183381088825214 | 4.588325023651123 | 0.49189828578434863 | 1.1275341306346207 |
| 2 | 0.3195721711315474 | 0.2700856771089571 | -0.610655737704918 | 0.6667741242209327 | 0.13753581661891118 | 3.3657419681549072 | 0.27197414303257295 | 0.9407610440481845 |
| 5 | 0.33966413434626147 | 0.28964622549232466 | -0.5905637744902039 | 0.643993122716527 | 0.12177650429799428 | 2.5111560821533203 | 0.1630753690361238 | 0.8268238888945699 |
| 10 | 0.3504598160735706 | 0.29788155036365827 | -0.5797680927628949 | 0.6319578766387277 | 0.11604584527220631 | 2.271489143371582 | 0.12387008023936477 | 0.7882461541387878 |
| 25 | 0.36165533786485404 | 0.3089733837997574 | -0.5685725709716114 | 0.6193853427895981 | 0.10888252148997135 | 2.12898588180542 | 0.1101748360375293 | 0.7643390278252763 |
| **50** | **0.3666533386645342** | **0.3154383206704984** | **-0.5635745701719312** | **0.614227380184827** | **0.11174785100286533** | **2.073582172393799** | **0.1123214699938649** | **0.756608395716899** |
| 100 | 0.36495401839264296 | 0.31290409210156234 | -0.5652738904438225 | 0.6155168708360198 | 0.10458452722063037 | 2.0453319549560547 | 0.1203384263793238 | 0.7528997420134467 |

(Majority-vote, confidence-weighted, and original-anchored full tables follow the identical established format and remain available in `metrics.json`; omitted here for brevity since Part D of the final Block A audit will consolidate all 24 cells together.)

### Batching/BN-adaptation applicability (explicit absence)

```json
{
  "bn_adaptation_algorithm": "sequential_microbatch_v1",
  "bn_adaptation_applicable": false,
  "bn_adaptation_batch_size": 256,
  "bn_adaptation_enumeration_order": "view_major_then_sample_major",
  "bn_adaptation_microbatches_at_primary_n": 0,
  "inference_batch_size": 256
}
```

### Dataset provenance

```json
{
  "actual_checksum_md5": "a8b06965200029087d5bd730944a56c1",
  "artifact_path": "data/raw/pathmnist.npz",
  "checksum_verified": true,
  "dataset": "pathmnist",
  "expected_checksum_md5": "a8b06965200029087d5bd730944a56c1",
  "resized": false,
  "resolution": 28,
  "verification_method": "dataset_verification.verify_official_dataset_artifact",
  "verification_version": 1
}
```

## 6. Interpretation, stated neutrally

This is one validation cell and one seed -- a pipeline canary
confirming the corrected GroupNorm persistence contract executes
end-to-end correctly on real data, not a confirmatory conclusion. No
threshold, policy, or analysis was changed because of this result. The
test split remains untouched throughout.

## 7. Confirmation: no remaining cell ran

Only this single authorized cell (`A-pathmnist-28px-groupnorm-policy-none-s0`,
attempt 2) executed. The remaining 20 Block A cells were not started
during this canary. The only tracked working-tree change resulting from
this canary is the strict, single-row append to
`artifacts/ledger_validation_evaluation.csv` recording attempt 2's
completion.
