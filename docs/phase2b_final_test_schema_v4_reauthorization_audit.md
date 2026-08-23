# Phase 2B.6H Part D — Schema-v4 Final-Test Reauthorization Audit

**Status: this document records the fourth-generation superseding
final-test authorization, issued after the Phase 2B.6G matrix-progress
halt and the Phase 2B.6H matrix-progress-aware engineering fix.** Cell 1
is carried forward as a valid, completed, consumed result and is not
rerun. No further cell has been executed under this authorization.

## 1. Authorization chain

| Generation | SHA-256 | Commit | Disposition |
|---|---|---|---|
| 1 (`phase2b.6b-v1`) | `ccff976dda7d93e8fdd8c4bc2fe78eaecef5316bb011b1e420ed7d30a9637ec2` | `76c46e2a25e95a397198f4a03d56a9db58ab6877` | Superseded. Historically valid. |
| 2 (`phase2b.6d-v2`) | `960b54358a356442c58957cf2ecdec2da916e72d1a01b1d29d5c7d162f8afdc0` | `69fff1e2ebd569e6d017d80674ca2555086e668b` | Suspended after the attempt-2 pre-access failure. Historically valid. |
| 3 (`phase2b.6d-v2`, generation 3) | `0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a` | `f8e794053926a275d4eb503f2994668577435317` | **Suspended** after the matrix-progress authorization halt (Phase 2B.6G). Under this generation, cell 1 (`A-pathmnist-28px-batchnorm-policy-none-s0`, attempt 3) was evaluated for real and completed successfully. Historically valid, permanently preserved via Git history. |
| 4 (`phase2b.6d-v2`, generation 4) | `d7ad4a2739156dfdf336bef1da712d0d99e705e821a733fd73d685aefcb3a929` | *(this commit)* | **Active.** |

**Referenced provenance commits** (all independently confirmed ancestors
of HEAD):
* `1ee5f737c8eccd39ca0d1a49183cfffda1051c2a` -- accidental-access incident record.
* `965daebbff2af976ea05f88cffb1eda848d36b12` -- incident recovery-policy freeze.
* `28d0c3d7de477b83dda5819586e4d89a0ffd1c4b` -- attempt-2 pre-access failure record.
* `69a2c5902c5575f0bc9946be954ef980eccb1093` -- authorization-receipt design freeze.
* `dc97a047f694e5b4683d8a60b4379ca9186ab3f5` -- immutable-receipt engineering fix.
* `2563eb992f8fc4880d7812a7d4c3ba144e34ee21` -- record of cell 1's real completion and the matrix-progress authorization halt.
* `14bdc2c1019c023de4c598ffa01b074643f51fae` -- matrix-progress-aware authorization design freeze.
* `a67fe66b23b2d6efe613a2b86a1e1919d45d020b` -- matrix-progress-aware authorization engineering fix.

## 2. Cell 1 carried forward as `completed_consumed`

`authorized_cells` for run_id `A-pathmnist-28px-batchnorm-policy-none-s0`
now includes `"cell_status": "completed_consumed"` and a
`"consumed_binding"` object recording:

| Field | Value |
|---|---|
| `final_test_evaluation_id` | `e846df3f5a95fa8745599d1f71809a84ebf86e2fbfbe8bc5aee4c1e4f9b96b20` |
| `primary_artifact_hash` | `0841e7502cb8da05bfe58c56508197e18a3db0665f6033c24eb9a43a800551af` |
| training attempt / checkpoint hash | `3` / `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e` |
| `prior_authorization_sha256` | `0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a` |
| `prior_authorization_commit` | `f8e794053926a275d4eb503f2994668577435317` |
| `prior_final_test_runner_fingerprint` | `54bc3f58c8a91fc3d2b7a58c6336e722dbd4cb24a8a8239e5e17fd8591e95877` |
| `max_authorized_attempt_for_this_cell` | `3` |
| `no_further_attempt_authorized` | `true` |

These `consumed_binding` fields are informational/audit-only -- the
production classification function (`_classify_final_test_cell()`,
Phase 2B.6H) never reads them; it independently re-derives
`completed_consumed` status from the live ledger row, attempt directory,
and artifact manifest every time, so the authorization file's claims are
always cross-checked against real state rather than trusted at face
value. `authorized_final_test_attempt` for this cell remains `3` (its
actual completed attempt), and no attempt `4` is authorized anywhere in
this document.

## 3. 39-cell coverage

* `A-pathmnist-28px-batchnorm-policy-none-s0`: 1 cell, `completed_consumed`, attempt 3.
* All other 38 cells: `pending`, attempt 1.
* Total: 39 cells, no duplicates, omissions, or extras (mechanically
  asserted by the generation script against the frozen 39-cell matrix and
  independently re-verified by the production verifier's own
  `authorized_cells run_id set does not exactly match` check).

## 4. Fingerprints

| Fingerprint | Generation 3 | Generation 4 | Changed? |
|---|---|---|---|
| Evaluator | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | No |
| Statistical analysis | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` | No |
| Cross-condition addendum | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` | No |
| Final-test runner | `54bc3f58c8a91fc3d2b7a58c6336e722dbd4cb24a8a8239e5e17fd8591e95877` | `3659ddaaeae47a71251991e0cfac50ae16e621e802380988c2bdaa66b327ef12` | **Yes** -- expected, disclosed consequence of the matrix-progress-aware classification fix (commit `a67fe66`). |

No other fingerprint changed. This confirms the runner change is
authorization-lifecycle-only: nothing in the frozen scientific
computation (evaluator, statistical-analysis, or cross-condition-
addendum definitions) changed at any point in Phase 2B.6G/2B.6H.

## 5. Why cell 1's scientific computation remains valid and must not be rerun

Cell 1's actual evaluation ran under generation-3's runner fingerprint
(`54bc3f58...`), which governed the *identical* frozen scientific
computation as generation-4's runner fingerprint
(`3659ddaa...`) -- the only source-level change between the two
(commit `a67fe66`) was to `_classify_final_test_cell()` and its call
sites in the authorization-verification orchestration layer, never to
model inference, view generation, aggregation, metric computation, or
persistence. `run_final_test_evaluation()`'s scientific-computation
steps 9-16 (device init through artifact persistence) are byte-for-byte
unchanged. Re-running cell 1 under generation 4 would therefore not
produce a different scientific result -- it would only waste a real,
already-valid completion and violate the "no automatic retry" policy
that has governed this matrix since Phase 2B.6D. This is why generation
4 explicitly forbids any further attempt for this cell
(`no_further_attempt_authorized: true`, and the classification function
would independently refuse to allocate one regardless).

## 6. Production verification of plan-mode state

Verified after this authorization was committed (Part D closing check,
mechanically, no scientific values inspected):

* `authorization_status = "approved"`
* `execution_locked = False`
* `n_completed_consumed = 1`
* `n_pending = 38`
* `n_invalid = 0`
* Cell 1 (`A-pathmnist-28px-batchnorm-policy-none-s0`) is non-runnable --
  requesting it returns an idempotent `already_completed_consumed`
  result without device, checkpoint, or dataset access.
* Cell 2 (`A-pathmnist-28px-batchnorm-policy-none-s1`)'s classification
  is `pending` at authorized attempt 1, and its own next-allocatable
  attempt (independently, per-cell) is also 1.

## 7. Execution discipline (unchanged)

Resuming execution (a separately authorized future task) must proceed
sequentially in the frozen 39-cell matrix order, skipping cell 1 (already
`completed_consumed`), invoking `evaluate-test --run-id <run-id>` exactly
once per remaining cell at its authorized attempt (1 for every one of
the other 38 cells). Any new failure halts the entire remaining sequence
immediately -- no automatic retry exists anywhere in this design. No
cell's result may be interpreted, individually or collectively, until
the entire authorized 39-cell matrix has completed.
