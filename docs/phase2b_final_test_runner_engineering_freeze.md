# Phase 2B.6A — Final-Test Evaluation Runner Engineering Freeze

**Status: FROZEN.** This document specifies the final-test-evaluation
runner and its authorization gate for the Phase 2B 39-cell confirmatory
matrix, before any implementing code is written. It does not authorize or
execute final-test evaluation, and it creates no authorization artifact.

## 0. Starting-state audit (Part A)

Confirmed before this document was written:

* HEAD = `24034a2` (the accepted Phase 2B.5C engineering commit).
* Working tree clean.
* `99b20be`, `429ac91`, `35e373a`, `24034a2` are all ancestors of HEAD.
* 39/39 canonical training and validation-evaluation completions
  unchanged (46 ledger rows: 43 completed/2 failed/1 aborted).
* Evaluator fingerprint `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef`
  confirmed unchanged; statistical-analysis fingerprint and cross-condition
  analysis fingerprint confirmed present and computable.
* No `artifacts/final_test_authorization.json` (or any authorization
  artifact) exists anywhere in the repository.
* No `artifacts/ledger_final_test.csv` or `artifacts/final_test/` exists.
* `artifacts/ledger_validation_evaluation_amendments.csv`'s
  `test_metrics_observed` column contains no `True` value anywhere (all
  historical amendments correctly disclose zero test-metric observation).
* The real statistical-analysis and cross-condition-addendum "analyze"/
  "real" modes have never been invoked outside their own test suites.

**Existing authorization infrastructure, disclosed:** the repository
already contains `src/when_tta_hurts/authorization.py` /
`configs/final_evaluation_authorization.yaml`, a Phase 2B.2-era gate for a
**different, still-draft, unrelated** future algorithm (Validation-Gated
TTA / H4's own eventual test pass), wired into `orchestrator.py` at a
call site that is provably unreachable today (H4 is draft, not approved).
This document's authorization gate is **new, independent, and additional**
-- it does not replace, extend, or share a path with that gate. Both gates
independently guard their own call site; neither is treated as
interchangeable with the other anywhere in this design.

## 1. Scientific computation (frozen: reuse, never fork)

The final-test evaluator reuses `validation_evaluation.py`'s scientific
computation **UNCHANGED**:

* `compute_validation_evaluation()` -- clean probabilities, the full
  100-view probability bank, every frozen condition/aggregator/prefix
  metric, BN adaptation.
* `compute_evaluation_latency_report()` -- clean/TTA latency.
* Checkpoint selection: `resolve_canonical_training_completion()` +
  `load_and_verify_canonical_checkpoint()`, unchanged.
* Preprocessing, deterministic view generation
  (`evaluation/views.py::iter_deterministic_views`), TTA seed
  `1306178015`, policy identifier, `MAX_VIEWS=100`, prefixes
  `[1,2,5,10,25,50,100]`, primary `N=50`, mean-probability primary
  aggregation, alternative aggregators (majority vote, confidence-
  weighted average), BN adaptation (`sequential_microbatch_v1`),
  inference batch size 256, BN-adaptation batch size 256,
  probability-native metric contract, latency boundaries: **all read
  from the same frozen `configs/validation_evaluation.yaml`
  (`load_frozen_tta_seed_config()`) and the same
  `EVALUATOR_FINGERPRINT_MANIFEST`-covered source files, with zero
  modification.**

**The only scientific-data difference is `split=test` rather than
`split=validation`.** This is achieved by a new, structurally test-only
loader (`evaluation/test_loader.py::load_final_test_split()`) that
returns the exact same `ValidationEvaluationSplit` dataclass
`evaluation/validation_loader.py` already defines and
`compute_validation_evaluation()`/`compute_evaluation_latency_report()`
already consume -- neither of those two functions is touched.
`evaluation/validation_loader.py` itself is **never broadened**; it
remains structurally validation-only (no split parameter), exactly as
its own docstring requires.

## 2. Authorization model

**Path (hardcoded, no override):** `artifacts/final_test_authorization.json`
(`FINAL_TEST_AUTHORIZATION_PATH` in `final_test_authorization.py`). No
CLI flag, environment variable, force flag, or alternate file path exists
anywhere that can substitute for this artifact -- `verify_final_test_authorization()`
reads no CLI argument and no environment variable; its only inputs are
the committed file's content plus git's own tracked/clean/ancestor state.

**Required, engineering-implemented, never-executed-for-real schema
(JSON object), every field checked against the CURRENT repository state
at verification time:**

| Field | Bound to |
|---|---|
| `status` | must be exactly `"approved"` |
| `approval_timestamp` | recorded, not verified against a clock |
| `phase2b_protocol_commit` | must be an ancestor of HEAD |
| `matrix_commit` | must be an ancestor of HEAD |
| `cross_condition_addendum_commit` | must be an ancestor of HEAD |
| `evaluator_fingerprint` | must equal `compute_evaluator_fingerprint()` |
| `statistical_analysis_fingerprint` | must equal `compute_analysis_fingerprint()` |
| `cross_condition_analysis_fingerprint` | must equal `compute_cross_condition_fingerprint()` |
| `final_test_runner_fingerprint` | must equal `compute_final_test_runner_fingerprint()` |
| `official_dataset_checksums` | `{"<dataset>@<resolution>": "<md5>"}`, each checked against `expected_official_checksum()` |
| `authorized_cells` | list of `{run_id, training_attempt, checkpoint_hash}`, one per current matrix cell; count and run_id set must exactly match the current 39-cell matrix; `training_attempt`/`checkpoint_hash` must exactly match each cell's CURRENT `resolve_canonical_training_completion()` result |

Authorization is **invalidated** by: any scientific/evaluator/analysis/
addendum/runner code drift (fingerprint mismatch), any bound commit
becoming unreachable (rewritten history), any dataset-checksum drift, or
any change to the canonical training identity of any of the 39 cells
(re-run, amendment, or matrix change) -- all checked freshly on every
call, never cached or trusted from a prior run.

No path override, CLI bypass, force flag, environment-variable bypass,
or alternate authorization file exists. `verify_final_test_authorization()`'s
signature is exactly `(artifact_path, matrix_path, repo_root)` -- no
`force`/`override`/`skip`/`unlock`/`bypass` parameter.

## 3. Ordering (frozen, enforced by `run_final_test_evaluation()`)

1. Resolve exact run ID (the function parameter itself).
2. Verify committed authorization (`verify_final_test_authorization()`).
3. Verify the cell is included in the authorization's authorized-cell set.
4. Resolve canonical training completion.
5. Recompute and bind every fingerprint/hash into a
   `FinalTestEvaluationConfig` and its `final_test_evaluation_id`.
6. Check existing final-test attempt/ledger state (idempotent skip,
   metadata-only, via the SAME `check_evaluation_skip()` validation
   already uses, pointed at the final-test ledger/root).
7. Enforce clean-tree policy (`require_clean_working_tree()`).
8. Allocate the attempt (`start_evaluation_attempt()`).
9. Initialize the requested device (`select_device("mps")`, no fallback).
10. Restore the checkpoint (`load_and_verify_canonical_checkpoint()`).
11. Verify the official dataset artifact checksum from raw file bytes
    (`verify_official_dataset_artifact()`).
12. Load only `test_images`/`test_labels`
    (`evaluation/test_loader.py::load_final_test_split()`).
13. Execute frozen inference (`compute_validation_evaluation()`).
14. Compute latency (`compute_evaluation_latency_report()`).
15. Validate and atomically persist artifacts
    (`persist_and_verify_final_test_completion()`).
16. Mark `status=completed`.
17. Append the final-test ledger row.

Steps (2)-(6) (every identity/authorization/idempotency check) complete
strictly before (9)-(12) (device/checkpoint/dataset access) -- a failure
in (1)-(7) creates zero files and zero ledger rows.

## 4. Observation and failure accounting

Truthful, independent lifecycle fields (never conflated): `test_split_accessed`,
`test_predictions_computed`, `test_metrics_computed`, `test_metrics_persisted`,
`test_metrics_observed`, `failure_stage`. `test_metrics_observed` is set
`True` as soon as metrics are **computed in memory** (`test_metrics_computed`),
independent of whether persistence later succeeds -- "not persisted" is
never conflated with "not accessed"/"not observed", since an in-memory
value is already observable to any caller.

`failure_stage` is one of: `device_init`, `checkpoint_load`,
`dataset_verification`, `test_data_load`, `inference`, `latency`,
`persistence` -- set to the furthest stage truthfully reached before an
exception, never rounded down or omitted.

No scientific metric value is ever printed to stdout. The CLI's
`evaluate-test` mode prints only: `status`, `training_run_id`,
`final_test_evaluation_id`, `attempt_number`, and artifact path/hash
pairs (`_redact_for_print()` in `scripts/run_final_test_evaluation.py`).

`status=completed` is reachable only after
`persist_and_verify_final_test_completion()` has independently
recomputed and verified every persisted metric, built and verified the
artifact manifest, and confirmed schema/finiteness/probability-array
validity -- exactly mirroring `evaluation_result_artifacts.py`'s
existing discipline for validation attempts. A failed attempt never
leaves partial predictions/metrics behind: persistence validates
in-memory before any file is written.

## 5. Retry policy

Mechanically derived: **no automatic retry** exists anywhere in this
design -- `run_final_test_evaluation()` has no retry loop, no
`retry`/`force`/`bypass` parameter (verified structurally by signature
inspection in tests), and the CLI has no `--retry`/`--force` flag. A
compatible completed attempt idempotently skips (step 6) strictly before
any heavy dependency (device/checkpoint/dataset) is touched. An
INCOMPATIBLE completed attempt (different identity) hard-fails via the
same `ConflictingEvaluationImplementationError` `check_evaluation_skip()`
already raises for validation -- it is never silently superseded. Any
retry after a real test-access failure would require a new, separately
authorized run (a new committed authorization artifact and/or a new
canonical training identity) -- this is a structural consequence of the
identity-binding design (§3 step 5-6), not a special-cased rule, and is
therefore not further "invented" here.

The task's own text raises the possibility that "exactly once" could be
ambiguous for a failure occurring before vs. after test-array access.
Resolution, mechanically derived rather than invented: the ordering in
§3 and the idempotent-skip check in step 6 already answer this
identically regardless of when the prior failure occurred -- a FAILED
attempt is never treated as "completed" by `check_evaluation_skip()`
(only a directory-and-manifest-verified `completed` status can be
returned as a skip target), so a failure at any stage, before or after
test-array access, is retried only by deliberately invoking the CLI
again -- there is no separate "pre-access failure -> silently retryable"
vs. "post-access failure -> blocked" branch. This does not require
stopping for a decision: no scientific computation or repeated-test-use
policy hinges on this distinction, since neither path is ever automatic.

## 6. Matrix-wide execution discipline

The engineered CLI provides only `evaluate-test --run-id <exact-run-id>`
-- there is no `--block`/`--all-cells` mode in this engineering task (not
already frozen elsewhere, so not added). A future matrix-wide real phase
would invoke this CLI once per cell, sequentially, in frozen matrix
order (`parse_and_validate_matrix()`'s cell order, the same order
`plan_final_test_evaluation()` reports), halting immediately on any
failure (no continuation to the next cell after a failure -- this
engineering task provides only the single-cell primitive; no
multi-cell/scripted-loop driver is implemented, so there is no
"continue past a failure" code path to disable). No parallel execution,
no result-dependent stopping, and no reading of scientific results
between cells is possible from this primitive alone, since it accepts
one `--run-id` and returns only redacted identity information.

## 7. Files (Part C implementation manifest)

* `src/when_tta_hurts/final_test_identity.py` -- `FINAL_TEST_RUNNER_MANIFEST`,
  `compute_final_test_runner_fingerprint()`, `FinalTestEvaluationConfig`,
  `compute_final_test_evaluation_id()`.
* `src/when_tta_hurts/final_test_authorization.py` -- the authorization
  gate (§2).
* `src/when_tta_hurts/evaluation/test_loader.py` -- the test-only loader
  (§1), the sole call site in the entire project passing
  `allow_test=True`.
* `src/when_tta_hurts/final_test_result_artifacts.py` -- final-test
  artifact schema/persistence, reusing
  `evaluation_result_artifacts.py`'s probability-array/latency/batching/
  BN-consistency validators unchanged; requires `split=="test"`
  (structurally disjoint from the validation schema's `split=="validation"`
  requirement).
* `src/when_tta_hurts/final_test_evaluation.py` -- `plan_final_test_evaluation()`
  (side-effect-free) and `run_final_test_evaluation()` (§3), reusing
  `validation_evaluation.py`'s attempt-lifecycle helpers
  (`start_evaluation_attempt`, `finish_evaluation_attempt`,
  `check_evaluation_skip`, `list_evaluation_attempts`) and scientific
  functions unchanged.
* `src/when_tta_hurts/ledger.py` -- `FINAL_TEST_LEDGER_PATH`,
  `FINAL_TEST_LEDGER_FIELDNAMES`, `ensure_final_test_ledger_exists()`,
  `append_final_test_entry()` (header-only during this engineering task).
* `scripts/run_final_test_evaluation.py` -- CLI (`plan`, `evaluate-test`),
  no scientific-metric printing, no bypass flags (§2, §6).
* `tests/test_final_test_authorization.py`,
  `tests/test_final_test_evaluation.py`, `tests/test_final_test_loader.py`,
  `tests/test_final_test_result_artifacts.py`,
  `tests/test_final_test_identity_and_ledger.py` -- synthetic tests.
* `tests/test_split_firewall_static.py` -- updated to recognize
  `verify_final_test_authorization()` as an acceptable guard alongside
  `verify_authorization()`, and to expect exactly one guarded
  `allow_test=True` call site (`evaluation/test_loader.py`) instead of
  zero.
* `.gitignore` -- narrow addition: `!artifacts/ledger_final_test.csv`
  (header-only ledger, tracked) and `!artifacts/final_test_authorization.json`
  (so a future, separately-authorized approval can be committed with a
  normal `git add`, never a force-add workaround); `artifacts/final_test/`
  itself stays ignored, mirroring `artifacts/validation_evaluation/`.

## 8. Fingerprint-identity disclosure

`src/when_tta_hurts/ledger.py` is a member of
`CROSS_CONDITION_ADDENDUM_MANIFEST` (added in Phase 2B.5C, load-bearing
for canonical-evaluation-identity resolution). Extending `ledger.py` with
the final-test ledger additions in this task therefore **changes the
cross-condition-analysis fingerprint** computed by
`compute_cross_condition_fingerprint()`, even though no cross-condition
addendum logic itself changed. This is the expected, correct behavior of
a content-hash-based fingerprint (it cannot selectively ignore
unrelated-but-manifested lines), and is disclosed here rather than
treated as a defect. Since no real cross-condition analysis has ever
been executed (`artifacts/ledger_statistical_analysis.csv` remains
nonexistent), nothing persisted is invalidated by this shift -- only the
identity a *future* cross-condition analysis run would compute changes.
`statistical_analysis_fingerprint` (`ANALYSIS_FINGERPRINT_MANIFEST`,
which does not include `ledger.py`) is unaffected. The evaluator
fingerprint (`EVALUATOR_FINGERPRINT_MANIFEST`, which also does not
include `ledger.py`) is likewise unaffected.

## 9. Not authorized by this document

This document does not authorize final-test evaluation, does not create
`artifacts/final_test_authorization.json`, and does not execute
`run_final_test_evaluation()` against real data. It specifies the design
that Part C implements and Part E tests, entirely against synthetic
fixtures.
