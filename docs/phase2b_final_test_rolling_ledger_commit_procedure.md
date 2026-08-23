# Phase 2B.6N Part C — Rolling Final-Test Ledger Commit Procedure Freeze

**Status: FROZEN before resuming cells 3-39.** This document resolves
the Phase 2B.6M clean-tree collision without any production-code
change and without any fingerprint cascade.

## 1. Root cause

`artifacts/ledger_final_test.csv` is not a member of
`require_clean_working_tree()`'s permitted-dirty-ledger whitelist
(`APPROVED_APPEND_ONLY_LEDGER_PATHS`, `orchestrator.py`) --
`['artifacts/ledger.csv', 'artifacts/ledger_amendments.csv',
'artifacts/ledger_confirmatory.csv', 'artifacts/ledger_incidents.csv',
'artifacts/ledger_validation_evaluation.csv']`. Any uncommitted append
to the final-test ledger therefore blocks every subsequent
`evaluate-test` invocation's step-7 clean-tree check, regardless of
which cell is being attempted.

## 2. Resolution: commit each row before the next invocation

1. `artifacts/ledger_final_test.csv` is not added to the clean-tree
   whitelist. No source patch is made to `orchestrator.py` or any other
   `EVALUATOR_FINGERPRINT_MANIFEST`/`ANALYSIS_FINGERPRINT_MANIFEST`/
   `CROSS_CONDITION_ADDENDUM_MANIFEST`/`FINAL_TEST_RUNNER_MANIFEST`-
   covered file during final-test execution, because that would trigger
   another fingerprint cascade of exactly the kind Phase 2B.6J/K/necessitated
   correcting.
2. Every successful cell is fully, independently verified (lifecycle
   flags, manifest, checksum, checkpoint/training binding, authorization
   and all four fingerprints, probability validity, sample alignment,
   independent semantic recomputation across every condition and
   prefix) BEFORE its ledger row is committed.
3. Each commit contains exactly one appended ledger row and no other
   path -- confirmed via `git diff` before every commit.
4. A failed or unverified cell halts execution immediately, with its
   row (if any was written) left uncommitted, exactly as in every prior
   final-test failure in this project's history.
5. Ledger-only and documentation-only commits (this document included)
   never alter the evaluator, statistical-analysis, cross-condition,
   reconciliation-implementation, or final-test-runner fingerprints --
   none of these commits touch any file listed in any fingerprint
   manifest. This is verified mechanically (§4 below), not assumed.
6. Because these commits change only `HEAD`'s provenance (which
   specific bytes of the ledger exist at which commit) and never the
   frozen scientific computation's source files, they cannot alter any
   already-computed or future scientific result -- they are pure record-
   keeping.
7. All previously bound, authorization-referenced commits
   (`d420657`, `6f012d1`, `bbbe7e2`, `6d68da1`, and now `f4de2be`)
   remain ancestors of `HEAD` at every step of this rolling-commit
   sequence -- a fast-forward-only history, never rewritten.

## 3. Commit granularity going forward

For cells 3-39: exactly one commit per successfully verified cell,
message `results: record final-test completion <exact-run-id>`,
containing only that cell's single new `artifacts/ledger_final_test.csv`
row. No batching, no squashing, no out-of-order commits.
