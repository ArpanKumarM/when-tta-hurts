# Phase 2B.4F Block A Execution Incident: GroupNorm persistence-schema failure

**Recorded: 2026-08-19.** This document records, honestly and without
deletion, a real-execution failure encountered on the third of 23
planned Block A validation-evaluation cells, and preserves the research
record of the two cells that completed successfully before it. No source
code is changed by this document. No attempt directory is modified,
deleted, or recreated.

## 1. What happened, mechanically confirmed

Block A serial execution ran three cells in canonical matrix order:

1. `A-pathmnist-28px-batchnorm-policy-none-s1` -- **completed successfully**
   (evaluation attempt 1, runtime 6914.23s).
2. `A-pathmnist-28px-batchnorm-policy-none-s2` -- **completed successfully**
   (evaluation attempt 1, runtime 6963.30s).
3. `A-pathmnist-28px-groupnorm-policy-none-s0` -- **failed** (evaluation
   attempt 1, runtime 4165.63s) with:
   ```
   EvaluationSchemaValidationError: batching.bn_adaptation_microbatches_at_primary_n
   must be a nonnegative integer, got None.
   ```
   raised from `_validate_batching_schema()`
   (`src/when_tta_hurts/evaluation_result_artifacts.py:228`), called via
   `persist_and_verify_evaluation_completion()` ->
   `_validate_metadata_schema()`, at
   `src/when_tta_hurts/validation_evaluation.py:1720`.

Execution halted immediately after this failure, per the driver's
fail-fast design (no retry, no next cell). The remaining 20 planned
Block A cells were never started -- confirmed by the absence of any
attempt directory for each of them.

## 2. Root cause (traced, not yet fixed by this document)

`src/when_tta_hurts/validation_evaluation.py:1361`:
```python
"bn_adaptation_microbatches_at_primary_n": bn_adaptation_microbatch_counts.get(PRIMARY_N),
```
For a **GroupNorm** cell, BN-adaptation is structurally never run (this
is correct and already covered by
`test_compute_validation_evaluation_groupnorm_skips_bn_adapted`), so
`bn_adaptation_microbatch_counts` is an empty dict and `.get(PRIMARY_N)`
returns `None`. `_validate_batching_schema()` -- written during the
earlier bounded-memory-batching OOM correction, when only a BatchNorm
cell had ever been exercised for real -- requires this field to
unconditionally be a nonnegative `int`, with no exemption for the
BN-adaptation-not-applicable case.

**This is a genuine, previously-latent defect, not a new regression
introduced by this task.** Every real evaluation run to date (attempts
1-4 on `A-pathmnist-28px-batchnorm-policy-none-s0`, plus the two
successful Block A cells above) exercised only BatchNorm cells. This was
the first time a GroupNorm cell ever traversed the real production
persistence path, and it surfaced a schema-validation gap that no prior
real run could have exposed. The gap is fixed under a separate,
documentation-first, test-covered engineering correction (see
`docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md` and
the accompanying engineering commit) -- not by this record.

## 3. Disposition

- `A-pathmnist-28px-groupnorm-policy-none-s0` attempt 1 is recorded
  exactly as it failed: `status=failed`,
  `failure_reason="batching.bn_adaptation_microbatches_at_primary_n
  must be a nonnegative integer, got None."`. Its attempt directory
  (`artifacts/validation_evaluation/A-pathmnist-28px-groupnorm-policy-none-s0/attempt_001/`)
  contains **only** `status.json` -- schema validation failed *before*
  `predictions.npz`/`metrics.json`/`metadata.json`/`artifact_manifest.json`
  were ever written, so no partial or corrupt artifact exists.
- **No amendment row is added for this attempt.** A `status=failed`
  ledger row is already, structurally, never a completed evaluation and
  is never eligible for canonical selection -- the evaluation-amendments
  ledger exists to override the eligibility of a *completed* attempt
  (like attempt 3's double-softmax exclusion), not to mark a failed
  attempt as ineligible a second time. Adding an amendment here would be
  redundant and would misrepresent the amendments ledger's purpose.
- The two successful cells (`...-s1`, `...-s2`) **remain valid and
  canonical-compatible**, unaffected by this failure -- their own
  execution never touched the GroupNorm code path.
- No test split was accessed at any point.

## 4. Independently reverified: the two successful cells

| | `A-pathmnist-28px-batchnorm-policy-none-s1` | `A-pathmnist-28px-batchnorm-policy-none-s2` |
|---|---|---|
| Evaluation attempt | 1 | 1 |
| Evaluation ID / config hash | `d453bc9c9e13aac9d413c5827407ddfff87985796896fd70adf7401a78997f3c` | `add32ac4b38553726ad79cc207cfbeeeef6f52fda563d83f243235e91373e00a` |
| Checkpoint hash | `f3be88438078ce362528dc8c3919e7d395fb095902cca48159c861b6e308e6bf` | `b8b971407b6b149d8e64e4bf55ec87f95a753cad5256aa5f3a57d419145d9c06` |
| `predictions.npz` SHA-256 | `964a79f1b38485e5843d53313d006c2589af1f1aa8b1aaae4fddf215d22588f2` | `aa63c533109ca62e3b75be06eabe8278c6357007cdb37e19f4f773cc9286eef6` |
| Runtime (s) | 6914.231107950211 | 6963.30331993103 |
| Manifest independently verified | OK | OK |
| Dataset checksum verified / resized | True / False | True / False |
| Checkpoint hash matches canonical training completion | True | True |
| Clean + N=50 mean-probability semantic recomputation | exact match (atol=1e-6) | exact match (atol=1e-6) |
| Sole compatible canonical completion (`check_evaluation_skip`) | True | True |
| `test_metrics_observed` | False | False |
| `metric_input_contract` | `probability_native_v1` | `probability_native_v1` |

## 5. Evaluation ledger state after this incident

Row/status totals (8 total rows, including header):

| Status | Count |
|---|---|
| completed | 4 (attempts 3, 4 for `-s0`; attempt 1 for `-s1`; attempt 1 for `-s2`) |
| failed | 2 (attempt 2 for `-s0`; attempt 1 for `-s0`-groupnorm) |
| aborted | 1 (attempt 1 for `-s0`) |

Full appended rows (this incident's diff, strict append-only):

```
True,d453bc9c9e13aac9d413c5827407ddfff87985796896fd70adf7401a78997f3c,A-pathmnist-28px-batchnorm-policy-none-s1,1,f3be88438078ce362528dc8c3919e7d395fb095902cca48159c861b6e308e6bf,d453bc9c9e13aac9d413c5827407ddfff87985796896fd70adf7401a78997f3c,1,validation,completed,964a79f1b38485e5843d53313d006c2589af1f1aa8b1aaae4fddf215d22588f2,1787174234.073198,1787181148.304306,6914.231107950211,,False
True,add32ac4b38553726ad79cc207cfbeeeef6f52fda563d83f243235e91373e00a,A-pathmnist-28px-batchnorm-policy-none-s2,1,b8b971407b6b149d8e64e4bf55ec87f95a753cad5256aa5f3a57d419145d9c06,add32ac4b38553726ad79cc207cfbeeeef6f52fda563d83f243235e91373e00a,1,validation,completed,aa63c533109ca62e3b75be06eabe8278c6357007cdb37e19f4f773cc9286eef6,1787181151.6585732,1787188114.961893,6963.30331993103,,False
True,2bb65453d1d5fe03186ec008cbd4006416f889282d26e152cc0d09e59b8b7b4b,A-pathmnist-28px-groupnorm-policy-none-s0,1,fcf6a2f41c136cadc012bab8726249062ed1a16290a98504b65903a96c234e98,2bb65453d1d5fe03186ec008cbd4006416f889282d26e152cc0d09e59b8b7b4b,1,validation,failed,,1787188118.076252,1787192283.7084181,4165.632166147232,"batching.bn_adaptation_microbatches_at_primary_n must be a nonnegative integer, got None.",False
```

## 6. Next steps (implemented separately, after this record is committed)

`docs/phase2b_validation_evaluation_groupnorm_persistence_freeze.md`
freezes the corrected GroupNorm/BatchNorm persistence-schema contract.
A separate engineering commit implements it with regression tests. Only
after that correction, its tests, and the full quality suite pass will
`A-pathmnist-28px-groupnorm-policy-none-s0` be re-attempted -- as
**evaluation attempt 2** for that run_id (attempt 1 permanently reserved
as failed, never deleted or reused). The remaining 19 Block A cells
after that stay paused pending further authorization.
