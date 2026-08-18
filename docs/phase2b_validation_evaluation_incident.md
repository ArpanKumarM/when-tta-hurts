# Phase 2B.4C: aborted validation-evaluation attempt incident record

This document records, honestly and without deletion, an accidentally
started confirmatory validation-evaluation attempt that occurred during
Phase 2B.4B Part 2 development, before the Phase 2B.4C hardening below
was implemented. **No prediction, probability, or metric of any kind was
computed, persisted, or observed at any point during this incident.**

## 1. Chronology

1. Phase 2B.4B Part 1 (freezing the confirmatory TTA seed) was committed
   as `124bd5831f99c3caac0933b9826765916f9104d1`.
2. During Part 2 development, `src/when_tta_hurts/validation_evaluation.py`,
   `src/when_tta_hurts/evaluation_result_artifacts.py`,
   `scripts/run_validation_evaluation.py`, and
   `tests/test_validation_evaluation.py` were edited in the working tree
   (uncommitted) to remove the `--tta-seed` CLI parameter and load the
   seed exclusively from the frozen config. `tests/test_run_validation_evaluation_cli.py`
   had **not yet** been updated to match -- it still contained a test
   (`test_evaluate_validation_requires_tta_seed`, since removed) that
   invoked the CLI with a real, valid `--run-id` and no `--tta-seed`
   argument, expecting an argparse error that the (already-edited,
   uncommitted) CLI script no longer produced, because the corresponding
   required-flag check had already been removed from the script.
3. Running `uv run pytest tests/test_run_validation_evaluation_cli.py -q`
   against this dirty working tree caused that stale test to invoke
   `cli_module.main()` with `evaluate-validation --run-id
   A-pathmnist-28px-batchnorm-policy-none-s0` and no seed override --
   which, with no `--tta-seed` requirement left in the (edited) CLI and
   no clean-tree guard anywhere in `run_validation_evaluation()` at that
   time, proceeded into the real, unpatched production evaluation path.
4. The pytest process was observed via `ps aux` at ~100% CPU utilization,
   `R` (running) state, for approximately 2 minutes 24 seconds of
   accumulated CPU time before being identified as anomalous and
   terminated (`kill -9`).
5. Immediately after termination, `artifacts/validation_evaluation/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_001/status.json`
   was found to exist, containing `"status": "running"`,
   `"ended_at": null`. No other file existed anywhere under that attempt
   directory -- specifically, `predictions.npz`, `metrics.json`,
   `metadata.json`, `view_manifest.json`, and `artifact_manifest.json`
   were all absent.
6. The directory was deleted at the time (before this incident's
   ledger-recording significance was fully worked through). This document
   and the accompanying ledger row now correct that -- **deleting the
   directory did not, and does not, make `evaluation_attempt=1` for this
   `evaluation_id` available for reuse** (see section 8).

## 2. Evidence used

- The surviving `status.json` content (captured verbatim in the prior
  session transcript before deletion):
  ```json
  {
    "attempt_number": 1,
    "ended_at": null,
    "evaluation_config_hash": "ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5",
    "failure_reason": null,
    "started_at": 1787093621.476174,
    "status": "running",
    "training_run_id": "A-pathmnist-28px-batchnorm-policy-none-s0"
  }
  ```
- `ps aux` output captured before termination, showing the pytest process
  at ~100% CPU, `R` state, ~2:24 CPU time.
- The production code's execution order at that time (identical to the
  order in commit `bc84589f2faa5372cd72fbea16bd1200d74011ae`'s
  `run_validation_evaluation()`, since the incident-relevant portion was
  not changed between the incident and that commit): seed-config load ->
  canonical-checkpoint resolution -> evaluation-config/ID derivation ->
  skip check -> `device_resolver()` (MPS) -> `start_evaluation_attempt()`
  (directory/status.json creation) -> checkpoint load -> validation-split
  load -> view generation/inference -> persistence -> ledger append.
- Independent re-derivation, against the *current*, unchanged canonical
  training completion and the *current*, unchanged frozen seed
  configuration, of the evaluation ID that `A-pathmnist-28px-batchnorm-policy-none-s0`
  would produce under `source_commit=124bd5831f99c3caac0933b9826765916f9104d1`
  (the commit that was HEAD at the time, since the incident-causing
  changes were uncommitted): this recomputation yields
  `ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5`,
  an **exact match** to the value recorded in the surviving `status.json`.
  This is the strongest available evidence for every fact it depends on
  (training run ID, canonical training attempt/checkpoint, TTA seed,
  seed-config identity, matrix hash, and the HEAD commit at invocation
  time) -- an accidental match on all of these simultaneously is not
  plausible.
- Git history (`git log`) confirming commit `124bd5831f99c3caac0933b9826765916f9104d1`
  immediately precedes the incident and `bc84589f2faa5372cd72fbea16bd1200d74011ae`
  follows it.

## 3. Facts established (with confidence level)

| Field | Value | Confidence |
|---|---|---|
| Training run ID | `A-pathmnist-28px-batchnorm-policy-none-s0` | Certain -- directly recorded in `status.json` |
| Training attempt | `3` | Certain -- independently re-resolved via `resolve_canonical_training_completion()` against the unchanged canonical training ledger, and consistent with the evaluation-ID match |
| Training checkpoint hash | `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e` | Certain -- same basis as above |
| Evaluation ID / evaluation config hash | `ab2dfad0322e9e80cdb5005ff536e65f3cd7212b90464dd83a89b18a2dbd7ac5` | Certain -- directly recorded in `status.json`, independently reproduced |
| Evaluation attempt number | `1` | Certain -- directly recorded in `status.json` |
| TTA seed | `1306178015` | Certain -- the only value that reproduces the recorded evaluation ID; the frozen config was already committed and unchanged at incident time |
| Source HEAD at invocation | `124bd5831f99c3caac0933b9826765916f9104d1` | Certain -- the only source_commit value that reproduces the recorded evaluation ID |
| Working tree dirty at invocation | Yes | Certain -- the incident could only occur with uncommitted, in-progress Part 2 edits present (see section 1) |
| Start timestamp | `1787093621.476174` (unix time) | Certain -- directly recorded in `status.json` |
| Termination/end timestamp | **Unknown** | Not recorded -- the process never reached `finish_evaluation_attempt()`; no `ended_at` was ever written. Recorded as blank in the ledger row, not fabricated. |
| Approximate runtime | ~2 min 24 sec of CPU time (from `ps aux`, before termination) | Approximate only -- CPU time, not a wall-clock duration; recorded descriptively here, left blank in the ledger's `runtime_seconds` field rather than estimated as fact |
| MPS initialized | **Yes** | Certain -- `device_resolver()` (which resolves and initializes the MPS device) executes strictly before `start_evaluation_attempt()` in the code path that must have run for `status.json` to exist |
| Checkpoint loaded | **Unknown** | Not independently verifiable -- no artifact confirms this either way |
| Validation data loaded | **Unknown** | Not independently verifiable |
| Clean inference began | **Unknown** | Not independently verifiable |
| Any transformed view generated | **Unknown, but plausible** | The sustained ~100% single-process CPU utilization is *consistent with* CPU-bound kornia view generation (the pilot/confirmatory augmentation path runs on CPU unconditionally, per `evaluation/tta.py`'s documented MPS-performance workaround), but this is circumstantial, not a directly recorded fact, and is reported here as a plausible inference, not a certainty |
| Probabilities/predictions/metrics persisted | **No** | Certain -- `predictions.npz`, `metrics.json` never existed in the attempt directory at any point it was observed |
| Probabilities/predictions/metrics observed by anyone | **No** | Certain -- no such value was ever printed, read, or inspected during or after the incident |

No field above was inferred beyond what the listed evidence supports;
fields without direct or exact-match evidence are explicitly marked
Unknown rather than assumed.

## 4. Explicit classification

- **Scientific metrics observed:** false
- **Test metrics observed:** false
- **Training checkpoint modified:** false (the checkpoint file was only
  ever, at most, read; `load_and_verify_canonical_checkpoint()` never
  writes to it)
- **Attempt artifact directory:** inadvertently deleted after termination
- **Canonical eligibility:** false
- **Status:** aborted
- **Reason:** test-harness escape accidentally invoked production
  validation evaluation

## 5. Why the accidental directory deletion does not erase the attempt

Deleting `artifacts/validation_evaluation/.../attempt_001/` removed the
on-disk *evidence* of the attempt, but the attempt itself -- an
allocation of `evaluation_attempt=1` for this `evaluation_id`, made real
by a `start_evaluation_attempt()` call that genuinely executed -- is not
undone by deleting its directory. The append-only evaluation ledger
(`artifacts/ledger_validation_evaluation.csv`) is the authoritative,
permanent record of every allocated attempt, precisely so that a deleted
or corrupted directory can never silently free up an attempt number for
reuse. This document and its accompanying ledger row (recorded exactly
once, `status=aborted`) restore that permanence retroactively, following
the same reconciliation principle already established for confirmatory
*training* attempts (`orchestrator.reconcile_stale_attempt()`).

## 6. Why the seed freeze remains uncontaminated

The frozen confirmatory TTA seed (`configs/validation_evaluation.yaml`,
`confirmatory_tta_seed=1306178015`, committed `124bd5831f99c3caac0933b9826765916f9104d1`)
was **read-only** input to the incident, not touched, modified, or
influenced by it in any way. `load_frozen_tta_seed_config()` never writes
to that file; the incident's evaluation attempt used the seed exactly as
already frozen, and no observation from the incident (there being none to
observe) could have informed or altered the freeze, which predates the
incident by definition (Part 1 of Phase 2B.4B was committed first). The
seed configuration file is confirmed byte-identical (same SHA-256,
`b590be6b626ff6461368893662307f1dd0fef912274333be62f32ceedb16b9fa`)
before and after this incident and its recording.

## 7. Why `evaluation_attempt=1` for this evaluation ID can never be reused

Per Phase 2B.4C's hardened attempt-numbering rule (implemented in Part 2
of this same phase), evaluation attempt numbering considers **both**
attempt directories on disk **and** evaluation-ledger rows -- a ledger
row for `evaluation_attempt=N` permanently reserves that number for its
`evaluation_id`, whether or not attempt `N`'s directory currently exists.
The next real execution for this exact `evaluation_id` (should the
canonical training completion, frozen seed configuration, and everything
else that composes it remain unchanged) will therefore resolve to
`evaluation_attempt=2`, never re-allocating the aborted `attempt_001`.

## 8. Distinction from a completed scientific evaluation

This incident produced **zero** scientific content: no clean prediction,
no augmented view, no aggregated probability, no accuracy/F1/NLL/ECE/
Brier value, no harm/rescue count -- nothing that could inform, bias, or
even touch any future analysis, hypothesis, or Validation-Gated TTA
design decision. It is recorded here purely as a **process-integrity
event** (an allocated attempt number that must not be silently reused
and a testing-discipline gap that must be closed), not as any kind of
evaluation result, partial or otherwise. `test_metrics_observed=false`
and `confirmatory=true`/`status=aborted` in the ledger row reflect
exactly this: a real, permanently-recorded confirmatory *attempt*, with
no scientific outcome attached, ever.
