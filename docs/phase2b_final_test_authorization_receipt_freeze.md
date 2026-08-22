# Phase 2B.6F — Immutable Final-Test Authorization Receipt (FROZEN)

**Status: FROZEN.** This document specifies the corrected
authorization-verification lifecycle before any code is changed. It
authorizes no execution by itself.

## 1. Root problem being corrected

`verify_final_test_authorization()`'s exact-attempt-binding check
(schema `phase2b.6d-v2`) recomputes `next_evaluation_attempt_number()`
**live, on every call**. Calling the full verifier a second time after
`start_evaluation_attempt()` has allocated the active attempt's directory
observes a next-allocatable-attempt value one higher than what the first
(correct) call observed -- a structural incompatibility between a
stateful, dynamic check and any code path that re-invokes it after the
state it depends on has changed
(`docs/phase2b_final_test_attempt2_preaccess_failure.md`).

## 2. Frozen design

1. **Full authorization verification occurs exactly once per attempt,
   before attempt allocation.** `run_final_test_evaluation()` calls
   `verify_final_test_authorization()` once, at step 2 (before step 6's
   idempotent-skip check and step 8's attempt allocation) -- unchanged
   from the existing frozen order.

2. **Successful verification returns an immutable verified authorization
   receipt.** A new, frozen dataclass (`VerifiedFinalTestReceipt`,
   `final_test_authorization.py`) captures, at the moment of successful
   verification, and never recomputed afterward:
   * authorization artifact SHA-256 and commit;
   * run ID;
   * the exact authorized attempt number for that run ID (a snapshot,
     not a live recomputation);
   * checkpoint hash and training attempt (checkpoint/training identity);
   * all four fingerprints (evaluator, statistical-analysis,
     cross-condition, final-test-runner);
   * the dataset's expected checksum for that cell's dataset/resolution;
   * protocol/matrix/cross-condition-addendum commit identities.

3. **The receipt is passed through the production orchestrator to the
   test loader.** `run_final_test_evaluation()` obtains the receipt once
   (step 2) and passes it explicitly into `load_final_test_split()` as a
   required parameter -- the loader never independently resolves
   authorization.

4. **After attempt allocation, no code may recompute "next allocatable
   attempt" for that active run.** `next_evaluation_attempt_number()` is
   called at most once per attempt, before `start_evaluation_attempt()`,
   as part of the single verification in step 1. No function reachable
   after step 8 may call it again for the same run within the same
   attempt's lifetime.

5. **Before test-data access, a static authorization-integrity recheck
   may confirm:**
   * the authorization file still exists;
   * its bytes/SHA-256 are unchanged since the receipt was issued;
   * it remains tracked and clean in git;
   * the receipt's run ID and attempt number match what is being loaded;
   * the receipt's fingerprints/dataset-checksum still match the current
     repository state.

   This recheck is static/comparative (receipt vs. current file bytes
   and current fingerprint recomputation) -- it never re-derives an
   attempt number from ledger/directory state.

6. **That static recheck must never recompute attempt allocation against
   the now-created directory.** No call to
   `next_evaluation_attempt_number()`, `list_evaluation_attempts()`, or
   any directory/ledger scan of the active run's attempt history is
   reachable from the static recheck.

7. **The loader may not independently invoke the full pre-allocation
   verifier.** `evaluation/test_loader.py` never calls
   `verify_final_test_authorization()` -- it accepts only a
   `VerifiedFinalTestReceipt` object and performs the static recheck
   (item 5) against it.

8. **A receipt for another run ID, attempt, dataset, or checksum must
   hard-fail.** The static recheck compares the receipt's bound `run_id`,
   `authorized_attempt`, `dataset`, `resolution`, and
   `dataset_expected_checksum_md5` against the loader's own call
   arguments -- any mismatch raises immediately, before any file access.

9. **No caller-controlled bypass or forged dictionary may substitute for
   a verified receipt.** The receipt type is a `frozen=True` dataclass
   constructible only by `verify_final_test_authorization()`'s own return
   path -- there is no public constructor, factory, or `from_dict()` that
   accepts caller-supplied field values, and the loader's parameter is
   type-annotated to require the real class (not a plain dict).

10. **Failure-stage reporting distinguishes:**
    * `authorization_preallocation` -- the single, full verification
      (step 2), before any attempt exists.
    * `device_initialization` (renamed from `device_init` for clarity).
    * `checkpoint_restore` (renamed from `checkpoint_load`).
    * `test_loader_authorization_receipt` -- the loader's static receipt
      recheck (item 5), reached only after an attempt has been allocated.
    * `dataset_checksum_verification` (renamed from `dataset_verification`).
    * `test_array_load` (renamed from `test_data_load`) -- reached only
      after the receipt recheck passes, covering the loader's own
      checksum re-check, `load_dataset()`, and `DataLoader` materialization.
    * `inference`.
    * `persistence`.

    This distinguishes a pre-allocation authorization failure (step 2,
    never reaches an attempt at all) from a post-allocation receipt
    failure (would have been the Phase 2B.6F incident's stage, had this
    design existed) from a genuine test-data-load failure (checksum
    mismatch, corrupt file, shape mismatch) -- three previously
    conflated failure modes.

## 3. What does not change

* Authorization-before-device/checkpoint/data ordering (steps 1-6 of the
  frozen order) is unchanged -- the single verification still occurs
  before any heavy dependency.
* No scientific computation, TTA configuration, dataset loading logic
  (beyond accepting a receipt parameter instead of re-deriving one),
  metrics, batching, or persistence changes in any way.
* No new CLI flag, environment variable, or bypass path is introduced.
  The receipt is an internal, in-process object -- never serialized to a
  file, never accepted from outside the single
  `run_final_test_evaluation()` call that produces and consumes it.
* The authorization artifact's own schema/content requirements (schema
  `phase2b.6d-v2`'s exact-attempt binding, supersession chain,
  fingerprint bindings) are unchanged -- only how many times, and at what
  point, the full verifier is invoked per attempt changes.
