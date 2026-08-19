# Phase 2B.4D OOM Incident: evaluation attempt 2 failed at BN-adaptation N=100

This document records, honestly and without deletion, the failure of the
first real validation-evaluation canary attempt for a confirmatory cell.
No source code is changed by this document. It is committed together
with the strict, single-row ledger append that already resulted from the
production failure-handling path -- nothing here modifies or rewrites
that row.

## 1. Facts

| Field | Value |
|---|---|
| Training run | `A-pathmnist-28px-batchnorm-policy-none-s0` |
| Canonical training attempt | `3` |
| Canonical checkpoint hash | `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e` |
| Evaluation ID / evaluation_config_hash | `96fbf4705bf93f4e2115fb33b9837df1095c90549d1f86ed1b1c1c160cc7fffe` |
| Evaluation attempt | `2` |
| Status | `failed` |
| `started_at` | `1787138598.428125` |
| `ended_at` | `1787140196.116925` |
| `runtime_seconds` | `1597.6888000965118` (~26.6 minutes of real MPS compute) |
| `failure_reason` (ledger) | `Invalid buffer size: 9.35 GiB` |
| `test_metrics_observed` | `False` |
| `confirmatory` | `True` |
| `split` | `validation` |

## 2. Failure

```
RuntimeError: Invalid buffer size: 9.35 GiB
```

Traceback (abbreviated): `run_validation_evaluation()` →
`compute_validation_evaluation()` → the BN-adaptation loop (over
`PREFIX_SEQUENCE`) → `bn_adapt(model, adaptation_inputs)` →
`torch.nn.Conv2d.forward()` → `F.conv2d(...)`.

**Cause**: for each registered N, the BN-adaptation code path built
`adaptation_inputs` by concatenating every (sample, view) pair for that N
into a single tensor (`torch.cat(adaptation_views, dim=0)`) and ran ONE
unchunked forward pass over the whole thing. At the largest registered N
(100) over the full PathMNIST validation split (10,004 samples), this
produced a batch of approximately 1,000,400 images in one forward call --
far larger than the 256-image batch size validated anywhere in this
project's training or benchmark history. The resulting intermediate
convolutional activation required an invalid/oversized (9.35 GiB)
contiguous buffer, which MPS refused to allocate.

## 3. What happened before the failure (honest accounting)

Real computation genuinely occurred in memory for approximately 26.6
minutes before the crash: MPS was initialized, the canonical checkpoint
was loaded and verified, the official PathMNIST validation artifact was
checksum-verified and loaded, clean inference ran, the full 100-view
probability bank was computed, all `naive_tta` aggregator/prefix
combinations were computed, `original_anchored_tta` was computed, and the
BN-adaptation loop proceeded through smaller N values before failing at
N=100.

**None of this in-memory computation was ever persisted or observed as a
scientific result.** `persist_and_verify_evaluation_completion()` was
never reached -- the crash occurred entirely inside
`compute_validation_evaluation()`, before its return value exists, and
before any of `predictions.npz`, `metrics.json`, `metadata.json`,
`view_manifest.json`, or `artifact_manifest.json` could be constructed.
The attempt-2 directory
(`artifacts/validation_evaluation/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_002/`)
contains exactly one file: a terminal `status.json` with
`"status": "failed"`. No prediction, probability, accuracy, F1, NLL, ECE,
Brier, harm-rate, rescue-rate, or latency value was ever written to disk,
printed, or used to inform any decision, judgment, or scientific
conclusion.

## 4. No retry occurred

Per explicit instruction, the failed run was not retried, the failure was
not patched around, and the failed attempt's recorded state
(`status.json`, the ledger row) was not deleted or rewritten. This
document and its accompanying commit record the failure exactly as it
occurred.

## 5. Attempt-lifecycle status

- **Attempt 1** (`evaluation_id=ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5`,
  the Phase 2B.4B/4C test-harness-escape incident): unchanged,
  `status="aborted"`, permanently reserved. Not touched by this incident.
- **Attempt 2** (`evaluation_id=96fbf4705bf93f4e2115fb33b9837df1095c90549d1f86ed1b1c1c160cc7fffe`,
  this incident): `status="failed"`, permanently reserved via the
  existing append-only ledger discipline
  (`next_evaluation_attempt_number()` considers the union of ledger rows
  and attempt directories for `training_run_id`, regardless of
  `evaluation_id` -- see `docs/phase2b_validation_evaluation_incident.md`
  sec.7 for the identical reasoning already established for attempt 1).
  **Noncanonical**: a `status="failed"` attempt is never a skip-eligible
  or canonical completion (`check_evaluation_skip()` only ever returns a
  `status="completed"`, artifact-verified attempt).
- **Neither attempt 1 nor attempt 2 blocks a future attempt.** The next
  real evaluation attempt for this training run, once a memory-safe
  evaluator is frozen and implemented, resolves to **attempt 3**.

## 6. Why this is a genuine implementation defect, not a hardware fluke

The failure is deterministic and structural, not incidental: any
confirmatory cell whose validation split is large enough for
`n_samples x 100 > (a memory-safe batch size)` will hit the same
unbounded single-batch BN-adaptation forward pass. This is not specific
to PathMNIST, to 28px, to this machine, or to this run. It is addressed
by the bounded-memory operationalization frozen in
`docs/phase2b_validation_evaluation_batching_freeze.md` (Part 4 of this
same engineering correction) and implemented separately, after this
record is committed.
