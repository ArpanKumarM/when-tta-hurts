# Phase 2B.6F — Attempt-2 Pre-Access Failure Record

**Status: this document records a genuine engineering failure that
occurred strictly before any test-data access.** Attempt 2 for
`A-pathmnist-28px-batchnorm-policy-none-s0` is preserved permanently as
`status=failed` and is never reused, retried, deleted, or rewritten. No
test array, test metric, or scientific value of any kind was ever
computed, persisted, or observed as a result of this failure.

## 1. Timeline

| Time (UTC) | Event |
|---|---|
| 2026-08-22T18:31:49Z | Phase 2B.6E controlled-matrix start timestamp recorded. |
| 2026-08-22T18:31:56Z | Driver launched `uv run python3 scripts/run_final_test_evaluation.py evaluate-test --run-id A-pathmnist-28px-batchnorm-policy-none-s0` (cell 1 of 39, authorized attempt 2). |
| 2026-08-22T18:32:00.460290Z | `start_evaluation_attempt()` allocated `attempt_002/status.json`, `status="running"` (proven, from `status.json`'s `started_at`). |
| 2026-08-22T18:32:00.888511Z | Attempt 2 terminated `status="failed"` (proven, from `status.json`'s `ended_at`). Total runtime: 0.428 seconds. |
| 2026-08-22T18:32:01Z | Driver detected nonzero exit, halted the entire sequence immediately per its design. No cell 2 was ever started. |

## 2. Exact failure

**Command:**
```
uv run python3 scripts/run_final_test_evaluation.py evaluate-test --run-id A-pathmnist-28px-batchnorm-policy-none-s0
```

**Exit code:** 1

**Failure reason (verbatim, from `status.json` and the ledger row):**
```
authorized_cells['A-pathmnist-28px-batchnorm-policy-none-s0'].authorized_final_test_attempt=2 does not match the production runner's next allocatable attempt (3) -- refusing to authorize an attempt number that does not exactly match current final-test ledger/attempt-directory state.
```

**`failure_stage` recorded:** `test_data_load`

**Attempt 2 directory manifest** (unmodified since the failure):
```
artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_002/status.json  (583 bytes)
```
SHA-256: `1930cf2a0b471a072f7be61fd865875e9e1255ee530f0ac2d3c91a8b9237661d`

**`status.json` exact contents:**
```json
{
  "attempt_number": 2,
  "ended_at": 1787423520.888511,
  "evaluation_config_hash": "5d51d941b4473c3707c444df4f5c150e0c8a21566eebcbd91f14d7a762842814",
  "failure_reason": "authorized_cells['A-pathmnist-28px-batchnorm-policy-none-s0'].authorized_final_test_attempt=2 does not match the production runner's next allocatable attempt (3) -- refusing to authorize an attempt number that does not exactly match current final-test ledger/attempt-directory state.",
  "started_at": 1787423520.46029,
  "status": "failed",
  "training_run_id": "A-pathmnist-28px-batchnorm-policy-none-s0"
}
```

**Authorization in effect at the time of failure:** SHA-256
`960b54358a356442c58957cf2ecdec2da916e72d1a01b1d29d5c7d162f8afdc0`
(schema `phase2b.6d-v2`), commit `69fff1e2ebd569e6d017d80674ca2555086e668b`
-- unchanged; this authorization's own content was never at fault, only
the runner's use of it a second time mid-run (§4).

**HEAD at time of failure:** `69fff1e2ebd569e6d017d80674ca2555086e668b`
(unchanged throughout this investigation).

## 3. Corrected lifecycle table (mechanically re-verified)

A prior draft report (Phase 2B.6E's immediate post-failure summary)
stated "no test array, checkpoint, or MPS path was ever reached," which
was imprecise: device initialization and checkpoint restoration are
proven to have succeeded before the failure. That report was corrected
before this permanent record was written. The table below reflects a
line-by-line trace of `final_test_evaluation.py::run_final_test_evaluation()`'s
`try` block, keyed on `failure_stage`'s assignment sequence (each value
is only ever assigned strictly after the immediately preceding stage's
code path executes without raising).

| Stage | Reached | Basis |
|---|---|---|
| MPS device initialized (step 9) | **True** | `failure_stage` advanced to `"checkpoint_load"`, only possible after `device_resolver()` returned successfully. |
| Checkpoint restored (step 10) | **True** | `failure_stage` advanced to `"dataset_verification"`, only possible after `load_and_verify_canonical_checkpoint()` returned successfully. |
| Official dataset whole-file checksum verified (step 11) | **True** | `failure_stage` advanced to `"test_data_load"`, only possible after `verify_official_dataset_artifact()` returned successfully AND its checksum-consistency check passed. This is a whole-file MD5 hash of the `.npz` file's raw bytes -- never an array read, never sample indexing. |
| Test-only loader's own internal authorization re-check (inside `load_final_test_split()`, its first line) | **Reached, and failed here** | The exception's exact text matches only `verify_final_test_authorization()`'s per-cell exact-attempt-binding check; `load_final_test_split()` calls this as its very first operation, before its own checksum re-check or `load_dataset()`. |
| Test-only loader's own checksum re-check | **False** | Never reached -- the loader raised before this line. |
| `load_dataset(..., allow_test=True)` / medmnist test-split construction (test artifact opened) | **False** | Never reached. |
| Test array indexing / `DataLoader` materialization | **False** | Never reached. |
| Clean/TTA predictions computed | **False** | `test_predictions_computed=False` in the ledger row; step 13 (`compute_validation_evaluation`) was never reached. |
| Metrics computed | **False** | `test_metrics_computed=False`. |
| Metrics persisted | **False** | `test_metrics_persisted=False`; no `metadata.json`/`predictions.npz` exists for attempt 2. |
| Metrics observed (printed or otherwise exposed to a human) | **False** | `test_metrics_observed=False`; the CLI's only output was the redacted `REFUSED: ...` error line, containing no numeric value. |

**Distinction between device/checkpoint access and test-data access:**
device initialization and checkpoint restoration read only the
already-verified, already-canonical training checkpoint file (a
non-test artifact, read-only, unchanged since training) and initialize
the MPS backend -- neither touches the official test split in any way.
Dataset-checksum verification (step 11) reads the *whole test-artifact
file's raw bytes* for an MD5 hash, which is explicitly permitted and
required by the frozen design (`docs/phase2b_final_test_runner_engineering_freeze.md`
§3 step 11) specifically *because* it is not an array read and cannot
expose any sample's content -- a checksum reveals nothing about images
or labels. Test-DATA access begins only at `load_dataset()`'s
construction of the medmnist test-split object and the subsequent
`DataLoader` materialization, neither of which was ever reached.

## 4. Root cause

`evaluation/test_loader.py::load_final_test_split()` contains a
"belt-and-suspenders" re-invocation of the full, dynamic
`verify_final_test_authorization()` as its own first internal action --
added in Phase 2B.6A specifically so the loader could never load test
data even if some future caller forgot to check authorization first.
This was safe under schema `phase2b.6b-v1` (no per-cell attempt-number
binding existed, so nothing about a second call could go stale between
the orchestrator's initial check and the loader's re-check).

Schema `phase2b.6d-v2` (Phase 2B.6D) added an exact-attempt-binding check
inside `verify_final_test_authorization()`: it recomputes
`next_evaluation_attempt_number()` **live, on every call**. Between the
orchestrator's initial (successful) authorization check and the loader's
second (redundant) call, `start_evaluation_attempt()` had already
allocated the new attempt's directory on disk. The loader's second call
therefore observed a next-allocatable-attempt value one higher than what
the orchestrator's own (already-verified) authorization bound moments
earlier -- an artifact of calling a *dynamic, stateful* check twice
across a state-changing operation, not a defect in the authorization
content itself.

## 5. Universal impact on all 39 cells

This defect is **structural and cell-independent**. For any cell, the
moment `start_evaluation_attempt()` allocates its authorized attempt
number N, a second call to `verify_final_test_authorization()` from
inside the loader will observe next-allocatable-attempt = N+1, which can
never equal the authorization's static binding of N. **Every one of the
39 cells, including the 38 unaffected ones authorized at attempt 1,
would have failed identically** at the same `test_data_load` stage, for
the same reason, had the matrix continued. This is not specific to the
affected cell's attempt-2 recovery scenario.

## 6. Why retry is scientifically non-adaptive

No test prediction, metric, or scientific value of any kind was computed
at any point (§3). No model, protocol, configuration, hypothesis,
threshold, or analysis definition changed in response to this failure --
nothing *could* change in response to a value that was never computed.
The eventual fix (Phase 2B.6F Part D) removes a redundant, purely
mechanical re-verification call; it does not alter checkpoint selection,
preprocessing, TTA configuration, metrics, batching, or persistence in
any way. Re-authorizing and re-attempting the affected cell (as attempt
3) is therefore a procedural correction, not a result-dependent
adaptation.

## 7. Authorization disposition

* Schema-`phase2b.6d-v2` authorization (SHA-256 `960b54358a356442c58957cf2ecdec2da916e72d1a01b1d29d5c7d162f8afdc0`,
  commit `69fff1e2ebd569e6d017d80674ca2555086e668b`) is **suspended**
  effective immediately -- it remains historically valid and permanently
  preserved via Git history, but must not be used to authorize any
  further execution. It will be superseded by a schema-v3 authorization
  once the underlying defect is fixed (Phase 2B.6F Parts C-E).
* **Attempt 2 is permanently reserved** for
  `A-pathmnist-28px-batchnorm-policy-none-s0` -- it is never reused,
  retried, deleted, or rewritten.
* **Attempt 3 requires new, explicit authorization** (schema-v3, Part E)
  before it may be attempted. This document does not authorize it.
* Attempt 1 (the Phase 2B.6C-Incident's aborted accidental-access record)
  remains permanently reserved and untouched, unaffected by this second,
  independent failure.
