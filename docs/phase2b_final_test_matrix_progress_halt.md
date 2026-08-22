# Phase 2B.6G/2B.6H — Matrix-Progress Authorization Halt Record

**Status: this document records the first real final-test cell
completion and the subsequent matrix-wide authorization halt.** It does
not itself access, open, print, or interpret any test-set scientific
value.

## 1. Cell 1's valid completion

`A-pathmnist-28px-batchnorm-policy-none-s0`, authorized and allocated
attempt 3, was evaluated for real under the schema-v3/generation-3
authorization (commit `f8e794053926a275d4eb503f2994668577435317`,
SHA-256 `0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a`).
The production CLI (`evaluate-test --run-id
A-pathmnist-28px-batchnorm-policy-none-s0`) exited 0 after 6404.8
seconds. The resulting ledger row:

| Field | Value |
|---|---|
| `evaluation_attempt` | 3 |
| `status` | `completed` |
| `confirmatory` | `True` |
| `split` | `test` |
| `test_split_accessed` | `True` |
| `test_predictions_computed` | `True` |
| `test_metrics_computed` | `True` |
| `test_metrics_persisted` | `True` |
| `test_metrics_observed` | `True` |
| `checkpoint_hash` | `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e` |
| `evaluator_fingerprint` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` |
| `statistical_analysis_fingerprint` | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` |
| `cross_condition_analysis_fingerprint` | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` |
| `final_test_runner_fingerprint` | `54bc3f58c8a91fc3d2b7a58c6336e722dbd4cb24a8a8239e5e17fd8591e95877` |
| `authorization_artifact_sha256` | `0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a` |
| `authorization_commit` | `f8e794053926a275d4eb503f2994668577435317` |
| `primary_artifact_hash` | `0841e7502cb8da05bfe58c56508197e18a3db0665f6033c24eb9a43a800551af` |

`test_metrics_observed=True` is **correct and expected** for any
completed final-test row. `final_test_evaluation.py`'s success path
hardcodes this value (a computed metric is conservatively treated as
observable regardless of whether a human inspected it); the "always
False" invariant documented in earlier Phase 2B.6A-era freezes applied
only to the pre-completion state, before any cell had ever finished.

## 2. Corrected static post-completion verification

A scratch verification script initially failed on cell 1 by calling the
full dynamic `verify_final_test_authorization()` after the cell had
already completed -- that function's exact-attempt check is a
pre-allocation check and necessarily disagrees with a completed cell's
frozen binding once the cell's own next-allocatable attempt has advanced
past it. This was a defect in ad hoc verification tooling, not in
`when_tta_hurts` production code, the ledger, or the authorization
artifact. A corrected, static-only verifier (comparing the tracked
authorization artifact's byte-identity, structural fields, and the
completed cell's frozen attempt binding, without any dynamic
recomputation) confirmed all 45 technical checks for cell 1, including
manifest integrity, dataset-checksum verification, probability-array
validity, independent recomputation of `clean.accuracy` (matched value
never printed), BN/GN consistency, and all four fingerprint bindings.

## 3. Cell 2's pre-allocation refusal

Resuming the sequence at cell 2
(`A-pathmnist-28px-batchnorm-policy-none-s1`, authorized attempt 1), the
production CLI itself (not scratch tooling) refused the cell before any
attempt allocation:

```
REFUSED: FinalTestAuthorizationError: authorized_cells['A-pathmnist-28px-batchnorm-policy-none-s0'].authorized_final_test_attempt=3 does not match the production runner's next allocatable attempt (4) -- refusing to authorize an attempt number that does not exactly match current final-test ledger/attempt-directory state.
```

CLI exit code 1.

## 4. Global next-attempt-check root cause

`verify_final_test_authorization()` (`final_test_authorization.py:446-475`)
loops over **every** authorized cell on every invocation and recomputes
`next_evaluation_attempt_number()` for each, comparing it against that
cell's frozen `authorized_final_test_attempt`. This check is global
across the whole matrix, not scoped to the cell currently being
processed. Once cell 1 completed attempt 3, its own binding permanently
disagreed with its new next-allocatable value (4) -- so calling this
function for **any** other cell (cell 2, or any of the remaining 37)
also failed, because it re-validates cell 1's now-stale binding along
the way. This is a distinct defect from the Phase 2B.6F loader
self-recheck issue: that fix addressed a cell's *own* redundant
post-allocation recheck within a single evaluation; this defect is the
orchestrator's upfront, matrix-wide verification call failing for
*every other* cell once *any* cell in the matrix has completed.

## 5. Proof cell 2 never reached MPS, checkpoint, or test-data access

* No ledger row exists for `A-pathmnist-28px-batchnorm-policy-none-s1`
  at any attempt -- confirmed by direct CSV inspection (3 total rows in
  the ledger, all for the affected cell).
* No `artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s1/`
  directory was ever created -- confirmed by directory listing (only
  the affected cell's directory exists).
* The refusal traceback originates entirely inside
  `verify_final_test_authorization()`, called at
  `final_test_evaluation.py:395`, which executes strictly before
  `receipt_for()` (step following, line 411),
  `check_final_test_evaluation_skip()` (line 447),
  `start_evaluation_attempt()` (line 467, which allocates the attempt
  directory), and `load_final_test_split()` (line 496, which is the
  only device/checkpoint/test-data access point). None of these later
  steps executed.
* `git status` shows only `artifacts/ledger_final_test.csv` modified
  (the cell-1 completion append) -- no other file changed.

## 6. Disposition

Cell 1 remains a valid, completed, consumed authorization and is not
rerun, amended, or deleted. No ledger row is added for cell 2 (no
attempt was ever allocated for it). The remaining 38 cells require a
corrected, matrix-progress-aware authorization-verification design
(Phase 2B.6H Parts B-D) before the sequence can safely resume.
