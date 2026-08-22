# Phase 2B.6D — Final-Test Incident Recovery Policy (FROZEN)

**Status: FROZEN.** This document specifies the recovery policy for the
Phase 2B.6C-Incident before any code is changed or any new authorization
is issued. It authorizes no execution by itself.

## 1. Original authorization disposition

* Authorization SHA-256 `ccff976dda7d93e8fdd8c4bc2fe78eaecef5316bb011b1e420ed7d30a9637ec2`
  (commit `76c46e2a25e95a397198f4a03d56a9db58ab6877`) is **historically
  valid and permanently preserved** -- it was correctly constructed,
  correctly verified, and its content is not in question.
* It is considered **consumed** for
  `A-pathmnist-28px-batchnorm-policy-none-s0` (the affected cell) --
  attempt 1 exists, is permanently reserved, and will never be reused.
* It will be **superseded for the entire 39-cell matrix**, because Part C
  of this task corrects a fingerprint-manifested defect in
  `check_evaluation_skip()`'s final-test reuse (`ledger.py` and/or
  `final_test_evaluation.py`), which changes
  `compute_final_test_runner_fingerprint()`'s value -- the original
  authorization's `final_test_runner_fingerprint` binding necessarily
  stops matching the corrected repository state, so it can never again
  verify as `approved` once the fix lands. This is a consequence of the
  authorization's own fingerprint-binding design working exactly as
  intended (fail-closed on any runner drift), not a special case.
* **The old authorization must never become active again after
  supersession.** The new authorization's schema must record the exact
  hash/commit it supersedes, and production verification of the new
  authorization must confirm that binding.
* Its committed version remains permanently available and byte-recoverable
  from commit `76c46e2a25e95a397198f4a03d56a9db58ab6877` via Git history --
  it is never deleted, rewritten, or amended.

## 2. Affected cell: `A-pathmnist-28px-batchnorm-policy-none-s0`

* Attempt 1 = the aborted incident record (ledger row, `status="aborted"`,
  `failure_stage="unknown_externally_terminated"`), preserved exactly as
  committed in `1ee5f73`.
* `test_split_accessed` conservatively treated as `True` (governance
  treatment, not a forensic claim of certainty -- see
  `docs/phase2b_final_test_accidental_access_incident.md` §6-7).
* Predictions/metrics COMPUTATION state (`test_predictions_computed`,
  `test_metrics_computed`) remains recorded as unknown (blank) in the
  historical ledger row, exactly as originally committed -- this document
  does not retroactively change that row.
* `test_metrics_persisted=False` -- proven false (no `metadata.json`/
  `predictions.npz` exists anywhere for this attempt); already correctly
  recorded in the historical row.
* `test_metrics_observed=False` -- proven false as a POLICY STATEMENT for
  this recovery: the incident investigation established that a numeric
  test value can only ever be printed or otherwise exposed to a human
  strictly after `run_final_test_evaluation()` returns successfully
  (which never happened), and the captured process output contains zero
  metric-shaped text (`docs/phase2b_final_test_accidental_access_incident.md`
  §6, §5). This is the authoritative governance position for this
  recovery going forward; it does not require or trigger any edit to the
  historical ledger row, which remains blank on this field exactly as
  committed.
* Attempt 1 is **never reusable** -- it is not deleted, not rewritten, and
  not resolved as a valid completion for any future check.
* **Exactly one recovery attempt is authorized: attempt 2.**
* Recovery reason (bound into the new authorization):
  `accidental_pytest_execution_incident_1ee5f73`.
* **No further retry is authorized** beyond this single recovery attempt.
  A failure of attempt 2 requires separate incident adjudication and a
  new authorization decision, exactly like any other final-test failure.

## 3. Remaining 38 cells

* Authorized attempt number = **1** for every cell other than the
  affected one.
* No prior final-test attempt exists for any of them (mechanically
  confirmed: `find artifacts/final_test -mindepth 1 -maxdepth 1` shows
  only the affected cell's directory).
* No retry is authorized for any of them beyond their single attempt 1,
  under the same no-automatic-retry policy as the original authorization.

## 4. Scientific justification for recovery rather than exclusion

* No numeric test result was ever persisted, printed, or observed from
  attempt 1 (proven false on both counts, §2 above and the incident
  document).
* No model, protocol, configuration, hypothesis, threshold, or analysis
  changed in response to attempt 1 -- nothing could have changed in
  response to a value that was never computed-and-observed.
* Rerunning the affected cell (as attempt 2, under a freshly issued
  authorization) therefore introduces **no result-dependent adaptation**
  of any kind -- the frozen scientific computation, TTA configuration,
  and analysis definitions are entirely unchanged from what attempt 1
  would have used.
* The deviation from the original single-pass intention remains a
  **procedural incident**, not a scientific one, and must be disclosed as
  such in the eventual paper/audit (see §5).
* All eventual claims from this matrix remain subject to the same frozen
  test analysis (`docs/phase2b_statistical_analysis_engineering_freeze.md`,
  `docs/phase2b_final_test_cross_condition_addendum.md`) -- nothing about
  the analysis plan changes because of this incident.

## 5. Paper/audit disclosure (restated from the incident document)

The eventual paper/audit must state plainly: this cell's final-test
evaluation was preceded by an accidental, unauthorized-by-intent
execution during test-harness development (commit `1ee5f73`); that
execution never persisted or exposed any test metric; the cell was
subsequently re-authorized and executed once, for real, as attempt 2,
under a superseding authorization (this recovery). This must never be
described as if the original single-pass design had proceeded unbroken
for this one cell.

## 6. Matrix execution (corrected)

The corrected sequence remains the full 39-cell frozen matrix order
(`configs/experiment_matrix.yaml`'s cell expansion order). The affected
cell (position 1 of 39 in frozen order) executes as **attempt 2**; all
other 38 cells execute as **attempt 1**. Any new failure at any cell
halts the entire matrix immediately and requires separate incident
adjudication -- no automatic retry exists anywhere in this design, for
this recovery or for any future failure.
