# Phase 2B.6O Part B — Final-Test Matrix Closure Audit

**Status: this document formally closes the 39-cell Phase 2B confirmatory
final-test matrix.** It is generated mechanically from persisted ledger,
authorization, and artifact records. It contains no scientific value
(no accuracy, F1, NLL, ECE, Brier, harm/rescue, TTA delta, p-value,
confidence interval, or comparison of any kind) -- only operational,
identity, and integrity facts.

## 1. Closure state

* **Authorization:** `approved`, generation 5, `completed_consumed=39`,
  `pending=0`, `invalid=0`.
* **Ledger totals:** exactly 42 rows -- 39 `completed`, 2 `failed`, 1
  `aborted`.
* **Distinct completed run IDs:** 39 (matches the frozen 39-cell matrix
  exactly; no duplicate, ambiguous, stale, or amendment-excluded
  completion exists for any cell).
* **Working tree:** clean at closure.

## 2. Exact 39-cell run/attempt mapping

| Run ID | Attempt | Operational runtime (s) |
|---|---:|---:|
| A-pathmnist-28px-batchnorm-policy-none-s0 | 3 | 6404.78 |
| A-pathmnist-28px-batchnorm-policy-none-s1 | 2 | 4791.26 |
| A-pathmnist-28px-batchnorm-policy-none-s2 | 1 | 4717.89 |
| A-pathmnist-28px-groupnorm-policy-none-s0 | 1 | 2872.82 |
| A-pathmnist-28px-groupnorm-policy-none-s1 | 1 | 2829.66 |
| A-pathmnist-28px-groupnorm-policy-none-s2 | 1 | 2849.53 |
| A-pathmnist-64px-batchnorm-policy-none-s0 | 1 | 7203.02 |
| A-pathmnist-64px-batchnorm-policy-none-s1 | 1 | 5359.62 |
| A-pathmnist-64px-batchnorm-policy-none-s2 | 1 | 5269.58 |
| A-pathmnist-64px-groupnorm-policy-none-s0 | 1 | 3104.51 |
| A-pathmnist-64px-groupnorm-policy-none-s1 | 1 | 3154.39 |
| A-pathmnist-64px-groupnorm-policy-none-s2 | 1 | 3149.56 |
| A-bloodmnist-28px-batchnorm-policy-none-s0 | 1 | 2284.29 |
| A-bloodmnist-28px-batchnorm-policy-none-s1 | 1 | 2290.58 |
| A-bloodmnist-28px-batchnorm-policy-none-s2 | 1 | 2302.71 |
| A-bloodmnist-28px-groupnorm-policy-none-s0 | 1 | 1358.28 |
| A-bloodmnist-28px-groupnorm-policy-none-s1 | 1 | 1365.72 |
| A-bloodmnist-28px-groupnorm-policy-none-s2 | 1 | 1367.22 |
| A-bloodmnist-64px-batchnorm-policy-none-s0 | 1 | 2525.26 |
| A-bloodmnist-64px-batchnorm-policy-none-s1 | 1 | 2497.02 |
| A-bloodmnist-64px-batchnorm-policy-none-s2 | 1 | 2508.73 |
| A-bloodmnist-64px-groupnorm-policy-none-s0 | 1 | 1491.77 |
| A-bloodmnist-64px-groupnorm-policy-none-s1 | 1 | 1489.37 |
| A-bloodmnist-64px-groupnorm-policy-none-s2 | 1 | 1508.23 |
| B-pathmnist-28px-batchnorm-policy-matched_mixed-s0 | 1 | 4779.92 |
| B-pathmnist-28px-batchnorm-policy-matched_mixed-s1 | 1 | 4795.42 |
| B-pathmnist-28px-batchnorm-policy-matched_mixed-s2 | 1 | 5049.67 |
| B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0 | 1 | 2291.16 |
| B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1 | 1 | 2294.84 |
| B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2 | 1 | 2256.05 |
| C-dermamnist-28px-resnet18-batchnorm-policy-none-s0 | 1 | 1661.11 |
| C-dermamnist-28px-resnet18-batchnorm-policy-none-s1 | 1 | 1697.79 |
| C-dermamnist-28px-resnet18-batchnorm-policy-none-s2 | 1 | 1703.78 |
| D-pathmnist-128px-batchnorm-policy-none-s0 | 1 | 7590.24 |
| D-pathmnist-128px-batchnorm-policy-none-s1 | 1 | 7738.42 |
| D-pathmnist-128px-batchnorm-policy-none-s2 | 1 | 7519.36 |
| D-bloodmnist-128px-batchnorm-policy-none-s0 | 1 | 3601.84 |
| D-bloodmnist-128px-batchnorm-policy-none-s1 | 1 | 3427.06 |
| D-bloodmnist-128px-batchnorm-policy-none-s2 | 1 | 3449.38 |

**Block coverage:** A = 24/24, B = 6/6, C = 3/3, D = 6/6.

**Total operational runtime, completed cells:** 134,551.87 s (~37.38
hours). Two historical non-completed attempts recorded separately: cell
1 attempt 1 (`aborted`, externally terminated, no measurable runtime),
cell 1 attempt 2 (`failed`, 0.43 s), cell 2 attempt 1 (`failed`, 4896.20
s, failed before persistence).

## 3. Ledger lifecycle accounting

Every one of the 39 completed rows carries all five lifecycle flags
`True`: `test_split_accessed`, `test_predictions_computed`,
`test_metrics_computed`, `test_metrics_persisted`,
`test_metrics_observed`. `test_metrics_observed=True` is the correct,
expected value for a completed row (a computed metric is conservatively
treated as observable, per `final_test_evaluation.py`'s hardcoded
success-path assignment) -- this is a lifecycle/provenance flag, not a
disclosure of any metric value.

## 4. Authorization generation and hash

* **Active authorization:** generation 5 (schema `phase2b.6d-v2`),
  SHA-256 `1e217e7e678ce37cee5c2b51fbf76429aa0b3b5298e622b1bbcb5363a6969f32`,
  commit `6d68da1dc34c276374cdab133a03b171b8b45381`.
* **Supersession chain (all confirmed ancestors of HEAD):** generation 1
  (`76c46e2`) -> generation 2 (`69fff1e`) -> generation 3 (`f8e7940`,
  SHA-256 `0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a`)
  -> generation 4 (`4544780`, SHA-256
  `d7ad4a2739156dfdf336bef1da712d0d99e705e821a733fd73d685aefcb3a929`) ->
  generation 5 (`6d68da1`, active).

## 5. Final fingerprints

| Fingerprint | Value |
|---|---|
| Evaluator | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` |
| Statistical analysis | `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01` |
| Cross-condition addendum | `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51` |
| Final-test runner | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` |
| Reconciliation implementation | `1b70509568c79a7cc5162fe01b7ba5ba746763fc889b8e32383d40b61c6a74b3` |

**Cell 1's historical binding (documented discrepancy):** cell 1
(`A-pathmnist-28px-batchnorm-policy-none-s0`, attempt 3) actually
completed under **generation 3** (SHA-256 `0332f696...`, commit
`f8e7940`) -- its ledger row correctly preserves that historical
`authorization_artifact_sha256`/`authorization_commit`/
`final_test_runner_fingerprint`/`evaluator_fingerprint`/
`statistical_analysis_fingerprint`/`cross_condition_analysis_fingerprint`
binding rather than the current one, and this is correct (the
production classification function never requires fingerprint equality
for an already-`completed_consumed` cell; compatibility was established
independently via the 56/56-check recomputation in Phase 2B.6J/K, not
via fingerprint matching). **However**, the currently-committed
generation-5 authorization's `consumed_binding.prior_authorization_sha256`
/`prior_authorization_commit` field for cell 1 incorrectly records
generation **4** (`d7ad4a27.../4544780`) instead of the actual
generation **3** (`0332f696.../f8e7940`) cell 1 ran under -- a
provenance-metadata inaccuracy introduced by the Phase 2B.6K
reauthorization builder script reusing the wrong constant. This field is
purely descriptive: `_classify_final_test_cell()` never reads
`consumed_binding` when determining `completed_consumed` status (it
independently re-derives classification from the live ledger/attempt-
directory/manifest state every time), so this inaccuracy has **zero
functional or enforcement impact** on the closure verified in this
document. It is disclosed here rather than silently corrected, per this
project's no-invented/no-omitted-caveat discipline; correcting the
authorization JSON's descriptive field is out of scope for this
closure-only task and would require separate authorization.

## 6. Per-cell artifact/manifest/checksum/checkpoint verification status

All 39 completed cells independently re-verified in this task (Phase
2B.6O Part A), fresh, from persisted bytes only:

* Artifact manifest (`artifact_manifest.json`) hash/size verification: **39/39 PASS**.
* Dataset checksum verified, `resized=False`: **39/39 PASS**.
* Checkpoint hash and training attempt bound to the authorized cell: **39/39 PASS**.
* Probability arrays finite, bounded to [0,1], row-normalized, sample-index-aligned and complete: **39/39 PASS**.
* Independent semantic recomputation across every persisted condition and every registered prefix (naive_tta x 3 aggregators, original_anchored_tta, bn_adapted_tta where applicable, clean): **39/39 PASS**.
* Production classification resolves to `completed_consumed`: **39/39 PASS**.

## 7. Historical incident and recovery-attempt accounting

| Attempt | Status | Disposition |
|---|---|---|
| Cell 1 attempt 1 | `aborted` | Accidental execution during a pytest gate-check, externally terminated (`kill -9`); zero test-metric persistence. See `docs/phase2b_final_test_accidental_access_incident.md`. |
| Cell 1 attempt 2 | `failed` | Pre-access authorization-verification defect (loader's redundant dynamic re-check); zero test-split access. See `docs/phase2b_final_test_attempt2_preaccess_failure.md`. |
| Cell 1 attempt 3 | `completed` | Real, valid completion under generation-3 authorization. |
| Cell 2 attempt 1 | `failed` | Semantic metric verification failure (clipping-asymmetry defect in `original_anchored_tta`, since corrected) caught before persistence. See `docs/phase2b_final_test_semantic_verification_incident.md`. |
| Cell 2 attempt 2 | `completed` | Real, valid completion under generation-5 authorization, after the shared-aggregation correction. |
| Cells 3-39 | `completed` | First and only attempt each, under generation-5 authorization. |

All three incident/recovery documents remain the permanent, unedited
forensic record; none of their content is altered by this closure.

## 8. Rolling-commit provenance (Phase 2B.6N)

Cells 3-39 were each committed individually immediately after passing
full technical verification, resolving a clean-working-tree collision
discovered in Phase 2B.6M without any source-code change (see
`docs/phase2b_final_test_rolling_ledger_commit_procedure.md`). The 37
per-cell commits, in execution order, are:
`4371ea8, 456f1c6, 505bb1d, 387f7ec, 53a4d81, 1849b1b, b37c93d, ff55471,
4e9c1d2, d233763, f8682e5, d5e2a65, 065b1c7, b9274e5, 30beccc, 872ac5a,
165e31d, 7114405, 7233475, bdc5ffe, 645f47b, 18744d9, 96a2bac, 5290ba7,
6826ea2, 66c0e5a, d92f9a7, 7769daa, 1554cf2, 40c53ef, f5baca5, 902adf0,
d36da0b, c8d19c6, 99f20b9, aeac183, 88c3783`. Cell 2's attempt-2
completion was committed separately as `f4de2be` before the rolling
procedure began.

## 9. Validation-reconciliation binding

All 39 validation-stage evaluations were deterministically reconciled
under the corrected shared-aggregation contract (Phase 2B.6K,
`artifacts/ledger_validation_reconciliation.csv`, commit `bbbe7e2`).
Re-confirmed fresh in this closure task: 39/39 reconciliation records
present, all `status=completed`, all `unaffected_conditions_match=True`.
Validation reconciliation is evidentiary/provenance-only for the
statistical-analysis and cross-condition-addendum resolvers (Phase
2B.6O Part C examines whether those resolvers correctly treat it as
such, never as a substitute for a final-test outcome).

## 10. Proof no scientific value was exposed during matrix execution

* Every driver log line and CLI invocation was constrained to
  operational status, exit codes, runtimes, and PASS/FAIL integrity
  checks throughout Phases 2B.6E-2B.6N (confirmed by direct inspection
  of `matrix_2b6*.log` files during execution -- no accuracy, F1, NLL,
  ECE, Brier, harm/rescue, or delta value was ever printed to any log,
  chat message, or document).
* The one exception, disclosed at the time and preserved permanently, is
  a single diagnostic metric-verification mismatch that appeared in cell
  2 attempt 1's integrity-check traceback -- a byproduct of the
  semantic-verification mechanism itself, not a TTA-efficacy
  measurement, and never used for any scientific conclusion. Its exact
  numeric values are recorded only in
  `docs/phase2b_final_test_semantic_verification_incident.md`, not
  repeated here.
* This closure audit document itself contains no accuracy, F1, NLL,
  ECE, Brier, harm/rescue, TTA-delta, p-value, confidence-interval, or
  scientific-comparison value.

## 11. Readiness confirmation

All 39 authorized cells are `completed_consumed`, fully re-verified in
this task, with zero invalid or ambiguous state anywhere in the matrix.
The final-test result set is ready for a separately authorized, frozen
statistical-analysis pass -- subject to the statistical-runner
readiness findings in Phase 2B.6O Part C.
