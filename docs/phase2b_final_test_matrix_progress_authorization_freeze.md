# Phase 2B.6H Part B — Matrix-Progress-Aware Final-Test Authorization Freeze

**Status: FROZEN before any engineering change.** This document
specifies the corrected authorization-verification design required to
resume the final-test matrix after cell 1's completion, per
`docs/phase2b_final_test_matrix_progress_halt.md` §4's root-cause
analysis. No production code is modified by this document.

## 1. What remains mandatory and unchanged

Global, static authorization validation is still required on every
verification call, for the authorization artifact as a whole:

* `schema_version` is a supported value and `status == "approved"`;
* the tracked authorization artifact is tracked-and-clean (byte-hash
  unchanged, present in the working tree, matching Git's index);
* the supersession chain (`supersedes_authorization_sha256`,
  `supersedes_authorization_commit`, `incident_record_commit`,
  `recovery_policy_commit`, `no_further_retry`) is all-or-nothing and,
  when present, verifies against Git history;
* every bound commit (`phase2b_protocol_commit`, `matrix_commit`,
  `cross_condition_addendum_commit`, and the supersession-chain commits)
  is an ancestor of HEAD;
* `evaluator_fingerprint`, `statistical_analysis_fingerprint`,
  `cross_condition_analysis_fingerprint`, and
  `final_test_runner_fingerprint` match current production computation;
* `authorized_cells` contains exactly the frozen matrix's current
  39-cell run-ID set -- no more, no fewer, no substitutions;
* for every cell, `training_attempt` and `checkpoint_hash` match the
  current canonical training completion, and the dataset checksum
  binding matches the official artifact.

None of the above changes. What changes is the exact-attempt check.

## 2. Per-cell dynamic classification (replaces the single global check)

Instead of a single boolean "does `authorized_final_test_attempt` equal
`next_evaluation_attempt_number()`" applied uniformly to every cell on
every call, each authorized cell is classified independently into
exactly one of three states, using only the final-test ledger and
attempt-directory evidence for *that* cell:

* **`pending`**: no ledger row exists at the authorized attempt number
  for this cell, no attempt directory exists at that number in a
  terminal or ambiguous state, and the authorized attempt number
  exactly equals `next_evaluation_attempt_number()` for this cell (i.e.
  nothing has been allocated yet, and the binding is still the correct
  next slot).
* **`completed_consumed`**: a ledger row exists for this cell at
  exactly the authorized attempt number, with `status == "completed"`,
  `confirmatory == True`, `split == "test"`, and lifecycle flags
  `test_split_accessed = test_predictions_computed =
  test_metrics_computed = test_metrics_persisted = True`
  (`test_metrics_observed` is expected `True` for any completed row and
  is not itself a validity signal). The attempt's on-disk artifacts
  (`predictions.npz`, `metrics.json`, `metadata.json`,
  `view_manifest.json`, `status.json`, `artifact_manifest.json`) all
  exist and the artifact manifest verifies. The row's checkpoint hash,
  training attempt, and all four fingerprints match the authorization's
  binding for this cell.
* **`invalid`**: anything else -- a `failed` or `aborted` row at the
  authorized attempt, a `running`/non-terminal attempt directory, a
  ledger/directory mismatch, a missing required artifact, an artifact
  manifest that fails to verify, a fingerprint or checkpoint mismatch, a
  later attempt number than authorized existing on disk or in the
  ledger with no corresponding `completed_consumed` classification at
  the authorized number, or any other state that does not cleanly match
  `pending` or `completed_consumed`.

## 3. Governing rules

1. A `completed_consumed` cell is accepted as fulfillment of its
   authorization. It is never required to satisfy
   `next_allocatable_attempt == authorized_attempt` after completion --
   that equality is a **pre-allocation** property, checked only while a
   cell is still `pending`, never re-checked afterward.
2. Only a `pending` requested cell may execute (allocate a new attempt
   and access MPS/checkpoint/test data).
3. Requesting a `completed_consumed` cell returns an idempotent
   "already completed" result derived entirely from the existing
   ledger row and artifacts. It must never re-invoke device selection,
   checkpoint loading, or `load_final_test_split()`.
4. Requesting an `invalid` cell fails closed with a descriptive error
   and blocks continued sequential execution of the matrix -- an
   `invalid` cell is a signal that something in the environment is
   inconsistent and must not be silently skipped or worked around.
5. Classification must be computed **per cell**, not derived from a
   single matrix-wide pass/fail boolean, and it must be computed **for
   every authorized cell**, not just the one being requested --
   scoping the check to only the requested cell is insufficient, because
   it would permit executing cell N+1 even if cell N (already
   attempted) were `invalid` (e.g. corrupted mid-write, or a
   ledger/directory mismatch left by an interrupted process). A single
   `invalid` cell anywhere in the matrix blocks all execution until
   resolved.
6. The immutable per-cell authorization receipt (`VerifiedFinalTestReceipt`,
   frozen at Phase 2B.6F) continues to govern the *requested* cell after
   its attempt is allocated. No dynamic `next_evaluation_attempt_number()`
   call may occur inside the loader, or anywhere else, after allocation
   -- this constraint from Phase 2B.6F Part C is unchanged and is not
   weakened by this design.

## 4. What this explicitly does not change

* The frozen 17-step execution order in
  `docs/phase2b_final_test_runner_engineering_freeze.md` §3.
* The `VerifiedFinalTestReceipt` mechanism and `verify_receipt_still_valid()`
  static recheck, frozen at Phase 2B.6F.
* The evaluator, statistical-analysis, and cross-condition-addendum
  fingerprints (no scientific computation is touched).
* The authorization JSON schema's field names or structure (schema
  remains `phase2b.6d-v2`; only the verification *behavior* changes).

## 5. Expected fingerprint consequence

Because this changes `final_test_evaluation.py` and/or
`final_test_authorization.py` (both covered by
`FINAL_TEST_RUNNER_MANIFEST`), `final_test_runner_fingerprint` will
change. The evaluator, statistical-analysis, and cross-condition
fingerprints are expected to remain unchanged, to be confirmed
independently after implementation, not assumed.
