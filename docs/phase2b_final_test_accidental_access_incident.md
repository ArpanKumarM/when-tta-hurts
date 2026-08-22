# Phase 2B.6C-Incident — Accidental Final-Test Access Adjudication

**Status: this incident is CLOSED for engineering-fix purposes only after
Part D's test correction lands. The affected cell
(`A-pathmnist-28px-batchnorm-policy-none-s0`) and the current
authorization artifact are treated as CONSUMED/SUSPENDED pending a
separate reauthorization decision (see §7). No retry has occurred and
none is permitted until that decision is made.**

## 1. Timeline

| Time (local, America/New_York) | Event |
|---|---|
| 2026-08-22 ~12:47:00 | `uv run pytest -q > /tmp/pytest_2b6c_gate.log 2>&1` launched as the Part A gate-check for Phase 2B.6C (backgrounded automatically by the tool harness after exceeding a 300s foreground timeout). |
| 2026-08-22 12:48:41.836448 (proven, from `status.json`'s `started_at`) | `start_evaluation_attempt()` allocated `artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s0/attempt_001/` for real, with `status="running"`. |
| 2026-08-22 ~12:58:50 (approximate, reconstructed from `ps -o etime` readings of ~11:50-12:01 elapsed against the ~12:47:00 start; not an independently logged termination timestamp) | The pytest process (PID 38590) and its parent `uv run` wrapper (PID 38589) were terminated via `kill -9` after being observed stuck at the same progress point (37-38% of 948 collected tests) for over 5 minutes with 100% CPU utilization. |
| 2026-08-22 ~13:00-13:05 | Forensic investigation began: process list confirmed clean, evidence preserved without modification, root cause traced. |

## 2. Exact trigger

`tests/test_final_test_evaluation.py::test_cli_evaluate_test_refuses_before_heavy_dependencies_real_repo`
(test index 375 of 948 in the collected suite; the immediately preceding
test, index 371, `test_plan_mode_reports_real_repo_state`, also failed
for the same underlying reason -- see §5 -- but is not itself dangerous,
since `plan_final_test_evaluation()` is side-effect-free).

This test loads `scripts/run_final_test_evaluation.py` fresh via
`importlib.util.spec_from_file_location()`, sets `sys.argv` to
`["run_final_test_evaluation.py", "evaluate-test", "--run-id", "A-pathmnist-28px-batchnorm-policy-none-s0"]`
(a REAL matrix run ID), and calls the freshly-loaded module's `main()`
directly in-process -- with no mock of `verify_final_test_authorization`,
`resolve_canonical_training_completion`, `load_and_verify_canonical_checkpoint`,
`load_final_test_split`, or any other heavy dependency.

## 3. Root cause (two independent defects; see §6 Part B for the full audit)

1. **Stale test assumption.** The test's own docstring states it relies
   on "the REAL repo state (no authorization artifact exists)" to prove a
   refusal. That assumption was true when the test was written (Phase
   2B.6A) and became false the moment Phase 2B.6B's authorization was
   committed (`76c46e2`) -- at which point `verify_final_test_authorization()`
   legitimately succeeds for this run ID, and the CLI proceeds past the
   refusal point into the real orchestrator.
2. **Wrong monkeypatch target.** The test's only defensive patch,
   `monkeypatch.setattr("when_tta_hurts.devices.select_device", _raise_if_called)`,
   patches the attribute on the `when_tta_hurts.devices` module. But
   `src/when_tta_hurts/final_test_evaluation.py` does
   `from when_tta_hurts.devices import select_device` at module import
   time and calls the LOCALLY BOUND name (`fte.select_device`), not
   `when_tta_hurts.devices.select_device`, when resolving its default
   `device_resolver`. Patching the origin module's attribute never
   affects the already-imported reference in the consuming module's
   namespace -- the patch was a complete no-op.

Together: authorization succeeded for real, and the one intended safety
net (a device-call trap) never armed. The result was a real,
unauthorized-by-intent invocation of `run_final_test_evaluation()` against
a real, authorized matrix cell, run from inside a pytest process rather
than the controlled, single-planned matrix-execution sequence Phase
2B.6C actually authorizes.

## 4. Authorization state at the time of the incident

* Authorization commit: `76c46e2a25e95a397198f4a03d56a9db58ab6877` (unchanged, still current).
* Authorization artifact SHA-256: `ccff976dda7d93e8fdd8c4bc2fe78eaecef5316bb011b1e420ed7d30a9637ec2` (unchanged, still current -- verified byte-identical after the incident).
* `verify_final_test_authorization()` returned `status="approved"` for all 39 cells both before and after the incident (re-confirmed post-incident, read-only).
* `A-pathmnist-28px-batchnorm-policy-none-s0` is cell #1 of 39 in frozen matrix order, training_attempt=3, checkpoint_hash `30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e`.

## 5. Attempt identity and preserved evidence

**Directory manifest** (unmodified since discovery):

```
artifacts/final_test/
  A-pathmnist-28px-batchnorm-policy-none-s0/
    attempt_001/
      status.json   (291 bytes)
```

No `metadata.json`, `predictions.npz`, `metrics.json`, `view_manifest.json`,
or `artifact_manifest.json` exists anywhere under this attempt directory.

**`status.json` exact contents** (SHA-256:
`333191343b6a7dccb47ee7d17eec610ad76dd4ac369ee4635949ce9dc7bb91eb`):

```json
{
  "attempt_number": 1,
  "ended_at": null,
  "evaluation_config_hash": "a033986d1c20c2edd073cc41dd4a56466184cba373f8e05afec264d0f3aefc8b",
  "failure_reason": null,
  "started_at": 1787417321.836448,
  "status": "running",
  "training_run_id": "A-pathmnist-28px-batchnorm-policy-none-s0"
}
```

**Filesystem timestamps** (`stat`): directory and file both created and
last modified 2026-08-22 12:48:41 local time; the file's `accessed` time
reflects only this investigation's later read, not new writes.

**Final-test ledger state:** header-only both before and after the
incident (1 line, zero data rows) -- the accidental run never reached
step 17 (ledger append).

**Captured stdout/stderr** (`/tmp/pytest_2b6c_gate.log`, 5 lines total at
the moment of termination): five lines of pytest progress dots ending
mid-line at test index ~374; grepped exhaustively for any
metric-shaped/CLI-JSON-shaped output (`accuracy`, `delta`, `f1`,
`calibration`, `brier`, `nll`, `"clean"`, `"conditions"`, `REFUSED`,
`completed`, `artifact_hashes`) -- **zero matches**. Nothing resembling a
scientific value or the CLI's own (redacted) success output was ever
written to any captured stream.

**Approximate CPU time** (explicitly approximate): `ps -o etime` showed
the pytest process at 11 minutes 50 seconds elapsed immediately before
termination; `status.json`'s `started_at` places attempt-allocation at 1
minute 41 seconds into that same process's life. The window during which
the accidental orchestrator execution could have progressed past
attempt-allocation is therefore approximately **10 minutes**, but this
number is a reconstruction from independently-observed `ps` snapshots,
not a logged duration, and is not used to infer any specific completed
pipeline stage (see §6 explicitly rejecting elapsed-time-based inference).

## 6. Part A — Lifecycle fact classification

Traced against `run_final_test_evaluation()`'s frozen 17-step order
(`docs/phase2b_final_test_runner_engineering_freeze.md` §3). A step's
completion is classified **proven true** only when direct on-disk
evidence requires it; **proven false** only when direct on-disk evidence
excludes it; otherwise **unknown**.

| # | Stage | Classification | Basis |
|---|---|---|---|
| 1 | Resolve exact run ID | proven true | It is the literal CLI argument. |
| 2 | Verify committed authorization | proven true | Step 8 (below) could not have run without this succeeding first; independently, authorization is confirmed genuinely valid at this HEAD. |
| 3 | Verify cell in authorized set | proven true | Same reasoning as #2 -- this run ID is in the 39-cell authorized set. |
| 4 | Resolve canonical training completion | proven true | Required before step 5's identity binding, which is embedded in `status.json`'s `evaluation_config_hash`. |
| 5 | Bind fingerprints/identity | proven true | `evaluation_config_hash` is present and well-formed in `status.json`. |
| 6 | Check existing attempt/ledger state (idempotent skip) | proven true (returned None) | Execution proceeded past this point (attempt allocation happened), and no prior attempt existed at the time, so this necessarily returned `None`, not a skip. |
| 7 | Enforce clean-tree policy | proven true | Execution proceeded to attempt allocation, which is gated behind this check. |
| 8 | Allocate the attempt | **proven true** | `status.json` exists with `status="running"`, `attempt_number=1`. |
| 9 | Initialize device (MPS) | **unknown** | No on-disk evidence either way; not inferred from elapsed time. |
| 10 | Restore checkpoint | **unknown** | Same. |
| 11 | Verify dataset checksum | **unknown** | Same. |
| 12 | Load test_images/test_labels | **unknown forensically; treated as TRUE for governance** (see explicit instruction in §7) | `load_final_test_split()` is the only code path in this project that can reach the official test split; it cannot be forensically excluded from having executed. Per explicit governance instruction, `test_split_accessed` is operationally treated as `True` for all authorization/retry decisions regardless of the forensic "unknown." |
| 13a | Compute clean predictions | unknown | No file evidence. |
| 13b | Compute TTA predictions (partial) | unknown | No file evidence. |
| 13c | Compute TTA predictions (fully, all 100 views) | unknown, but this requires reaching the end of `compute_validation_evaluation()`, which is also required for step 13d | No file evidence. |
| 13d | Compute test metrics | unknown (requires 13c) | No file evidence. |
| 14 | Compute latency | unknown (requires 13 complete) | No file evidence. |
| 15 | Validate and persist artifacts | **proven false** | No `metadata.json`/`predictions.npz`/`metrics.json`/`view_manifest.json`/`artifact_manifest.json` exists anywhere. Persistence is atomic and all-or-nothing (`persist_and_verify_final_test_completion`) -- partial persistence is structurally impossible, so its total absence proves this step never completed. |
| 16 | Mark status=completed | **proven false** | `status.json` still reads `"status": "running"`. |
| 17 | Append ledger row | **proven false** | Ledger remains header-only. |
| -- | Test metric exposed to any human | **proven false** | The CLI only ever prints a value (via `_redact_for_print()`, itself containing no scientific field) strictly AFTER `run_final_test_evaluation()` returns successfully -- which never happened. The captured log contains zero metric-shaped or CLI-output-shaped text (§5). |

**Explicit statement, per instruction: real test-split access (step 12)
cannot be ruled out and is not claimed to be ruled out.** No inference
from elapsed CPU time was used to promote any "unknown" stage to
"proven" in either direction.

## 7. Governance treatment

* `test_split_accessed` is **operationally treated as `True`** for every
  authorization/retry decision concerning this cell and this attempt,
  regardless of the forensic "unknown" classification above. This is a
  conservative governance choice, not a forensic finding.
* No test prediction or test metric was ever persisted (proven false,
  §6).
* No numeric test metric was ever printed to any captured stream or
  otherwise exposed to a human (proven false, §6).
* No later cell started: the accidental execution addressed exactly one
  cell (`A-pathmnist-28px-batchnorm-policy-none-s0`); the authorized
  sequential matrix driver was never launched; no other run ID's
  attempt directory or ledger row exists anywhere under
  `artifacts/final_test/`.
* No scientific decision, hypothesis verdict, or interpretation of any
  kind was made from this incident -- this document contains no
  scientific value of any kind, only identity, timing, and process
  metadata.
* **The current authorization (`76c46e2`) is considered consumed/suspended
  with respect to this specific cell** -- attempt 1 of
  `A-pathmnist-28px-batchnorm-policy-none-s0` is permanently reserved and
  will never be reused, retried, deleted, or rewritten. Whether the
  remaining 38 cells may proceed under the existing authorization, or
  whether the entire authorization must be superseded, is a decision
  deferred to §Part E below and requires the user's explicit direction --
  this document does not make that decision.
* No retry of this cell, and no execution of any other final-test cell,
  is permitted until a new, explicit authorization decision is made.

## 8. Part B — Root-cause and exposure audit

**Defect 1 (test):** stale assumption that authorization would remain
absent forever (§3.1).

**Defect 2 (test):** monkeypatch targeted the definition module
(`when_tta_hurts.devices.select_device`) rather than the consuming
module's locally-bound name (`when_tta_hurts.final_test_evaluation.select_device`)
(§3.2).

**Mechanical exposure sweep** (every test file, searched for every
reachable path into the real runner):

* `grep` for `run_final_test_evaluation.py`, `scripts.run_final_test_evaluation`: found only in `tests/test_final_test_evaluation.py` (the one dangerous test) and the script itself.
* `grep` for `run_final_test_evaluation(` outside its own definition/import: all 6 call sites are in `tests/test_final_test_evaluation.py`; the other 5 (not the dangerous one) each call `_patch_common(monkeypatch)`, which unconditionally monkeypatches `fte.verify_final_test_authorization` to a fixed fake object -- these 5 are safe regardless of real authorization state, confirmed by direct code reading.
* `grep` for `load_final_test_split(` : only in `tests/test_final_test_loader.py` (all calls there monkeypatch `tl.verify_final_test_authorization`, `tl.load_dataset`, `tl.verify_official_dataset_artifact` -- safe) and one docstring mention in `tests/test_split_firewall_static.py`.
* `grep` for `allow_test=True` (literal): the sole call site in `src/`/`scripts/` is `evaluation/test_loader.py`'s own guarded call (by design, statically enforced). One additional literal call exists in `tests/test_data_firewall.py::test_load_dataset_allows_test_split_only_with_explicit_flag` -- a pre-existing (pre-Phase-2B.6A) firewall test that calls `data.py::load_dataset` directly with `root="/tmp/does_not_exist_xyz", download=False`; confirmed safe (this path cannot reach real data: `download=False` and the medmnist package will raise a file-not-found-shaped error, not `TestSplitAccessError`, which is exactly what the test asserts). This is unrelated to the incident and required no fix.
* No other file anywhere in `tests/`, `scripts/`, or `src/` references the real authorization artifact path or constructs an unmocked path into `run_final_test_evaluation()`.

**Conclusion: exactly one exposure path existed, and it has been fully
characterized. No second path was found.**

**Existing session-wide guard coverage** (`tests/conftest.py`):

An existing autouse, session-scoped fixture (`_no_real_validation_evaluation_side_effects`,
added after the unrelated Phase 2B.4B incident,
`docs/phase2b_validation_evaluation_incident.md`) asserts that
`artifacts/validation_evaluation/` and `artifacts/ledger_validation_evaluation.csv`
are unchanged across the whole test session. **This guard was never
extended to the final-test artifact directory or ledger when Phase
2B.6A was engineered.** It therefore could not have detected this
incident even in principle, since it does not inspect
`artifacts/final_test/` or `artifacts/ledger_final_test.csv` at all. This
is a genuine, mechanically-confirmed coverage gap, addressed in Part D.

**A second, independent limitation applies regardless of guard
coverage:** `tests/conftest.py`'s guard is a `yield`-based fixture whose
assertion runs only in normal session teardown (after all tests
complete, or pytest exits through its own normal exit path). A `kill -9`
against the pytest process bypasses all Python-level teardown code
entirely, including `yield`-based fixture finalizers. **No autouse
pytest fixture of any kind -- existing or newly added -- can detect or
prevent damage from an externally-terminated (SIGKILL'd) process**; such
a guard can only catch a *completed* test session that produced real
side effects, never an externally interrupted one. This limitation is
structural to pytest's fixture-teardown model, not fixable by adding
more assertions, and is disclosed here rather than papered over.

**A separate, independent production defect was discovered during this
investigation** (not part of the original incident's root cause, but
found while assessing whether the current state is safe): `check_evaluation_skip()`,
reused unchanged from `validation_evaluation.py` for the final-test
ledger (per `docs/phase2b_final_test_runner_engineering_freeze.md`'s
explicit reuse design), internally calls
`ledger_module.has_evaluation_row(evaluation_config_hash, attempt_number)`,
which reads CSV column `evaluation_id`. `FINAL_TEST_LEDGER_FIELDNAMES`
has no `evaluation_id` column (it uses `final_test_evaluation_id` and
`evaluation_config_hash` instead). **This means `has_evaluation_row()`
can never find a match against the final-test ledger, so
`check_evaluation_skip()` will unconditionally raise
`EvaluationStaleAttemptError` for any nonterminal leftover final-test
attempt directory, regardless of what is or is not recorded in the
ledger.** Mechanically confirmed twice, read-only: (1) calling
`check_evaluation_skip()` directly against the real, current state raises
`EvaluationStaleAttemptError`; (2) repeating the same call against a
temporary ledger copy containing a row whose `final_test_evaluation_id`/
`evaluation_config_hash`/`evaluation_attempt` exactly match the stray
directory's identity **still** raises the same error, proving the defect
is the column-name mismatch and not merely a missing row. This defect is
fail-safe in direction (it blocks reuse rather than silently permitting
it) and is **not fixed in this task** -- it is a genuine production
defect requiring separate authorization before any source change, per
the explicit instruction governing this incident response. It is also,
independently, why `A-pathmnist-28px-batchnorm-policy-none-s0` cannot
currently proceed to a new attempt through the production runner at all
(mechanically confirmed: `next_evaluation_attempt_number()` would
allocate attempt `2`, but `check_evaluation_skip()` would reject any
invocation before that allocation is ever reached).

## 9. Incident ledger row

The final-test ledger schema (`FINAL_TEST_LEDGER_FIELDNAMES`) stores every
field as a free-form CSV string with no strict-boolean parser (unlike the
amendments ledgers' `canonical_eligible` column) -- it CAN truthfully
represent an externally-terminated attempt with genuinely unknown fields
left blank. One row was appended via the production
`append_final_test_entry()` function (not hand-edited) with:

* `status="aborted"` (terminal, per instruction).
* `final_test_evaluation_id` / `evaluation_config_hash` =
  `a033986d1c20c2edd073cc41dd4a56466184cba373f8e05afec264d0f3aefc8b`
  (copied verbatim from the real, proven `status.json` -- not
  recomputed or fabricated).
* `training_attempt=3`, `checkpoint_hash=30bc1ca6ef364e2a8280d4f5d9df5c6860d839e92e8a619e979dd20dbd804b3e`
  (from the authorization manifest / canonical training resolution,
  independently reconfirmed unchanged).
* `evaluation_attempt=1`.
* `split="test"`, `confirmatory=True` (hardcoded by the production
  function).
* All four fingerprints and the authorization artifact SHA-256/commit
  set to their current, accepted, unchanged values (§4).
* `test_split_accessed=True` (governance treatment, §7 -- not a forensic
  claim).
* `test_predictions_computed`, `test_metrics_computed`,
  `test_metrics_observed` left **blank** (empty string) -- genuinely
  unknown, never claimed false, and not covered by the explicit
  governance override that applies only to `test_split_accessed`.
* `test_metrics_persisted=False` (proven false, §6).
* `primary_artifact_hash=""` (blank, per instruction).
* `started_at=1787417321.836448` (the real, proven value from
  `status.json`).
* `ended_at=""`, `runtime_seconds=""` (blank -- no fabricated end time or
  runtime, per instruction).
* `failure_stage="unknown_externally_terminated"` -- a distinct sentinel,
  not one of the runner's 7 real internal stage tokens, since which
  internal stage was reached is genuinely unknown and must never be
  guessed.
* `failure_reason="Externally terminated accidental pytest execution (kill -9 of PID 38590/38589 during Phase 2B.6C Part A gate-check pytest run); see docs/phase2b_final_test_accidental_access_incident.md for full forensic record."`
