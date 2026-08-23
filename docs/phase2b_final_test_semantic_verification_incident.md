# Phase 2B.6J Part A — Final-Test Semantic Verification Incident Record

**Status: this document preserves the exact, forensic record of cell 2's
first (attempt 1) real final-test execution failure.** It does not
interpret the numerical mismatch below as any TTA-efficacy result --
it is a technical integrity-check disagreement between a recomputed and
a persisted intermediate value, discovered strictly before persistence.

## 1. Exact command and outcome

```
uv run python3 scripts/run_final_test_evaluation.py evaluate-test --run-id A-pathmnist-28px-batchnorm-policy-none-s1
```

* Started: `2026-08-23T05:01:36.708462Z` (`started_at=1787461296.708462`)
* Ended: `2026-08-23T06:23:12.909428Z` (`ended_at=1787466192.909428`)
* Runtime: 4896.2 seconds (~1h22m)
* Exit code: 1
* `evaluation_config_hash`: `4dec9cc91cf28ab78073f3bbe1b9d4b81917417a96ed6cb40ad7f5288f66a51d`

## 2. Exact traceback (verbatim from the production log)

```
Traceback (most recent call last):
  File "/Users/arpanmahapatra/Desktop/Research/when-tta-hurts/scripts/run_final_test_evaluation.py", line 151, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/Users/arpanmahapatra/Desktop/Research/when-tta-hurts/scripts/run_final_test_evaluation.py", line 140, in main
    result = run_final_test_evaluation(args.run_id, matrix_path=args.matrix)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/arpanmahapatra/Desktop/Research/when-tta-hurts/src/when_tta_hurts/final_test_evaluation.py", line 652, in run_final_test_evaluation
    _verify_metrics_semantically(outcome["predictions"], metrics, PREFIX_SEQUENCE)
  File "/Users/arpanmahapatra/Desktop/Research/when-tta-hurts/src/when_tta_hurts/validation_evaluation.py", line 1145, in _verify_metrics_semantically
    _compare(f"original_anchored_tta.{n}.{k}", v, persisted_entry[k])
  File "/Users/arpanmahapatra/Desktop/Research/when-tta-hurts/src/when_tta_hurts/validation_evaluation.py", line 1122, in _compare
    raise EvaluationPersistenceError(
when_tta_hurts.evaluation_result_artifacts.EvaluationPersistenceError: Semantic metric verification failed at original_anchored_tta.1.negative_log_likelihood: recomputed=1.2021862268447876, persisted=1.2022227048873901 (atol=1e-06, rtol=1e-06).
```

This is the ONLY numerical value that appeared anywhere in the CLI
output. It is a diagnostic comparison produced by the integrity check
itself, not a TTA-efficacy measurement, and is preserved here strictly
as forensic evidence for Part B's engineering investigation -- it must
never be cited as, or treated as a step toward, any scientific
conclusion.

## 3. Lifecycle truth (ledger row, attempt 1, exact)

| Field | Value |
|---|---|
| `training_run_id` | `A-pathmnist-28px-batchnorm-policy-none-s1` |
| `evaluation_attempt` | `1` |
| `status` | `failed` |
| `test_split_accessed` | `True` |
| `test_predictions_computed` | `True` |
| `test_metrics_computed` | `True` |
| `test_metrics_persisted` | `False` |
| `test_metrics_observed` | `True` (conservative -- a computed metric is always treated as observable) |
| `failure_stage` | `persistence` |
| `runtime_seconds` | `4896.200965881348` |

`attempt_001/` contains ONLY `status.json` (474 bytes) -- no
`predictions.npz`, `metrics.json`, `metadata.json`, `view_manifest.json`,
or `artifact_manifest.json` were ever written, since
`_verify_metrics_semantically()` runs strictly before
`persist_and_verify_final_test_completion()` in the frozen execution
order.

## 4. Confirmed unaffected state

* Cell 1 (`A-pathmnist-28px-batchnorm-policy-none-s0`, attempt 3):
  `predictions.npz` SHA-256 `0841e7502cb8da05bfe58c56508197e18a3db0665f6033c24eb9a43a800551af`
  -- byte-identical to its original completion, confirmed after this
  incident.
* Cells 3-39: no attempt directories exist anywhere under
  `artifacts/final_test/` for any of them.
* No final-test evaluation process is running.
* The only tracked working-tree change is the single new `failed` row
  appended to `artifacts/ledger_final_test.csv` for cell 2 attempt 1.

## 5. Disposition

Cell 2 attempt 1 is permanently `failed` -- preserved exactly, never
amended, deleted, or retried. It permanently occupies attempt-number 1
for this run_id; any future recovery execution for this cell must
allocate attempt 2. No engineering correction has been implemented as
of this record; Part B's no-test forensic adjudication follows
separately, using only static source inspection, synthetic arrays, and
already-persisted artifacts -- never a new call to `evaluate-test` or
`load_final_test_split()`.
