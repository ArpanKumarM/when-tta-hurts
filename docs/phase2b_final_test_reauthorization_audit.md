# Phase 2B.6D — Final-Test Reauthorization Audit

**Status: this document records the superseding final-test authorization
(schema/version `phase2b.6d-v2`), issued after the Phase 2B.6C-Incident
and its recovery engineering.** No final-test cell has been executed
under this authorization. This document does not itself access, open,
or index any test array.

## 1. Original authorization provenance

* SHA-256: `ccff976dda7d93e8fdd8c4bc2fe78eaecef5316bb011b1e420ed7d30a9637ec2`
* Commit: `76c46e2a25e95a397198f4a03d56a9db58ab6877`
  ("results: authorize frozen Phase 2B final-test evaluation")
* Schema: `phase2b.6b-v1` -- lacked per-cell final-test attempt binding
  entirely, and is no longer accepted by the production verifier
  (`_SUPPORTED_SCHEMA_VERSIONS = {"phase2b.6d-v2"}`).
* Historically valid and permanently preserved -- its content is
  independently re-hashed from the real commit `76c46e2` and confirmed
  byte-identical to the recorded SHA-256 as part of this authorization's
  own supersession-chain verification.

## 2. Incident provenance

* Commit: `1ee5f737c8eccd39ca0d1a49183cfffda1051c2a`
  ("results: record accidental final-test access incident")
* Full forensic record: `docs/phase2b_final_test_accidental_access_incident.md`.
* Affected cell: `A-pathmnist-28px-batchnorm-policy-none-s0`, attempt 1,
  `status=aborted`, preserved permanently and untouched.

## 3. Engineering-fix provenance

* Recovery-policy freeze commit: `965daebbff2af976ea05f88cffb1eda848d36b12`
  ("docs: freeze final-test incident recovery policy").
* Stale-attempt/ledger-column defect fix commit:
  `e7e0235de0d26ab99329e22ed6d28d6ef817a033`
  ("fix: reconcile aborted final-test attempts and bind authorized recovery")
  -- corrects `check_evaluation_skip()`'s final-test reuse
  (`has_evaluation_row()` column-name/ledger-path mismatch,
  `docs/phase2b_final_test_accidental_access_incident.md` sec.8) via a
  new, self-contained `check_final_test_evaluation_skip()` in
  `final_test_evaluation.py`. Confined to `FINAL_TEST_RUNNER_MANIFEST`-
  covered files only -- deliberately never touches `validation_evaluation.py`,
  so the evaluator and statistical-analysis fingerprints are unaffected.
* Exact-attempt-binding/supersession schema commit:
  `d7de6ecbfd92033a38f9d7b038d6c35abe0485e6`
  ("feat: bind exact final-test attempt authorization and supersession")
  -- extends `final_test_authorization.py` to schema/version 2.

## 4. Old and new runner fingerprints

| Fingerprint | Old (at `76c46e2`) | New (current, at `d7de6ec`) | Changed? |
|---|---|---|---|
| Evaluator | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | No |
| Statistical analysis | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` | No |
| Cross-condition addendum | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` | No |
| Final-test runner | `0c1a6ac0f84765cdfa52b42cd0ef698df00bd78fa8c6d12fb7e5eb196c80840e` | `efc57a362da86f70b22b6120de9c87a599f418261a180ca56a4a3a9b9de93b4f` | **Yes** -- expected, disclosed consequence of the Part C/D fixes |

The final-test-runner fingerprint changed **twice** during this recovery:
once for the `check_evaluation_skip` reconciliation fix
(`0c1a6ac0...` -> `25a3d8f4...`) and once more for the schema-v2
authorization/supersession-binding logic (`25a3d8f4...` ->
`efc57a362da86f70b22b6120de9c87a599f418261a180ca56a4a3a9b9de93b4f`, the
value bound into this authorization). No other fingerprint changed at
any point in this recovery -- confirmed by direct recomputation after
each commit.

## 5. Exact per-cell attempt authorization

* `A-pathmnist-28px-batchnorm-policy-none-s0`: **attempt 2**
  (`prior_attempt_disposition: accidental_pytest_execution_incident_1ee5f73`).
* All other **38** cells: **attempt 1** (no prior attempt exists for any
  of them).
* Every value was computed via the production function
  `next_evaluation_attempt_number()` against the CURRENT, real
  `artifacts/final_test/` directory and `artifacts/ledger_final_test.csv`
  state -- not hand-typed -- and is re-verified fresh, mechanically,
  every time `verify_final_test_authorization()` runs (never cached).
  No attempt 3 is authorized for any cell; no attempt 2 is authorized for
  any cell other than the affected one.

## 6. Proof that no scientific setting changed

* `evaluator_fingerprint`, `statistical_analysis_fingerprint`, and
  `cross_condition_analysis_fingerprint` are bit-for-bit identical to the
  original authorization's values (§4) -- nothing in the frozen
  scientific computation, TTA configuration, statistical-analysis
  definitions, or cross-condition addendum changed during this recovery.
* `phase2b_protocol_commit`, `matrix_commit`, and
  `cross_condition_addendum_commit` are unchanged from the original
  authorization.
* All 39 cells' `training_attempt`/`checkpoint_hash` bindings are
  unchanged from the original authorization (independently re-verified
  against `resolve_canonical_training_completion()` at construction
  time).
* All 7 official dataset checksums are unchanged and independently
  re-verified against `expected_official_checksum()`.
* The only things that changed are: (a) the final-test-runner
  fingerprint (a defect fix + new binding logic, §4), and (b) the
  addition of per-cell exact-attempt binding and the supersession block
  -- both procedural/engineering, not scientific.

## 7. Proof that no result influenced the recovery

* No test prediction or test metric was ever computed-and-persisted from
  the incident's attempt 1 (proven false,
  `docs/phase2b_final_test_accidental_access_incident.md` §6).
* No test prediction or test metric was ever printed or otherwise
  exposed to a human (proven false, same document).
* The recovery-policy freeze
  (`docs/phase2b_final_test_incident_recovery_freeze.md`) was written and
  committed BEFORE any code was changed and BEFORE this authorization was
  constructed -- the exact attempt-number bindings in this authorization
  are a MECHANICAL CONSEQUENCE of that frozen policy plus the current
  ledger/directory state, not a judgment call made after observing any
  result.
* Nothing about the frozen scientific computation, TTA configuration, or
  statistical-analysis/addendum definitions was altered in response to
  the incident (§6).

## 8. Paper disclosure language

The eventual paper/audit must state: the final-test evaluation for
`A-pathmnist-28px-batchnorm-policy-none-s0` was preceded by an accidental,
unauthorized-by-intent execution during test-harness development
(`docs/phase2b_final_test_accidental_access_incident.md`); that execution
never persisted or exposed any test metric; the affected cell was
subsequently evaluated once, for real, as attempt 2, under this
superseding authorization
(`docs/phase2b_final_test_incident_recovery_freeze.md`). This must never
be described as if the original single-pass design had proceeded
unbroken for this one cell. All 39 cells' confirmatory and secondary
analyses remain governed by the same frozen statistical-analysis plan
and cross-condition addendum regardless of this procedural recovery.

## 9. Matrix execution discipline (restated, unchanged)

The authorized execution must proceed sequentially, in the frozen
39-cell matrix order, invoking each cell's `evaluate-test --run-id
<run-id>` exactly once at its authorized attempt number (2 for the
affected cell, 1 for every other cell). Any new failure halts the entire
matrix immediately and requires separate incident adjudication -- no
automatic retry exists anywhere in this design. No cell's result may be
interpreted, individually or collectively, until the entire authorized
39-cell matrix has completed.
