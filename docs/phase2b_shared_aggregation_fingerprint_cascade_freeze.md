# Phase 2B.6K Part A — Shared Aggregation Fingerprint-Cascade Disposition Freeze

**Status: FROZEN before any further engineering.** This document
records the accepted disposition for the Phase 2B.6J fingerprint
cascade and freezes the reconciliation approach. It does not itself
access the test split, run statistical analysis, or report any
scientific value.

## 1. Pre-existing uncommitted engineering state (preserved exactly)

The following files, already modified in the working tree at the start
of this task (Phase 2B.6J Part E), are preserved byte-for-byte across
this document's commit -- confirmed by SHA-256 before and after:

| File | SHA-256 |
|---|---|
| `src/when_tta_hurts/evaluation/aggregation.py` | `d072eb5ae402e4d7552c119e0d95d9c20230611cce0be2dc367a77be03572ac2` |
| `src/when_tta_hurts/validation_evaluation.py` | `0b47a7b8254b47e512373bb39e8c48d6b2e6dfbf68d13592a505a049f493aa35` |
| `tests/test_aggregation.py` | `f75dbc7f4811a0cf9e21adb5e600e72cbd699fd0e868eb94db755d58e118d573` |
| `tests/test_metric_contract_correction.py` | `f31242411c8cc09c398a28127db4699ec375aa879d4e7bba0b9acb170d337881` |
| `tests/test_validation_evaluation.py` | `4722bc977afd63f2dcefce7aba865437b3346a0167a39adde931d102da31c830` |
| `tests/test_frozen_evaluation_n25.py` | `d9d6beb3642c2dfa4cc1579f1c34c1ec1aa6cd9d67a0a46c32594efc8fd32a59` |
| `tests/test_final_test_semantic_metric_contract_fix.py` (new) | `cad0f54bfc87fee97e90dac15887a5ed5c41505a033670625c2cffa7ac60f55e` |

## 2. Frozen decisions

1. **The shared correction is accepted; a final-test-only fork is
   prohibited.** `original_anchored_mean_probability()` and its two call
   sites in `validation_evaluation.py` are corrected in their one, shared
   location. Forking a final-test-only copy would violate this
   codebase's explicit, load-bearing invariant ("no scientific
   computation is forked" between validation and final-test evaluation)
   and would leave the identical defect live in the validation pipeline.
2. **The fingerprint cascade is expected and must not be bypassed.**
   `evaluator_fingerprint`, `statistical_analysis_fingerprint`,
   `cross_condition_analysis_fingerprint`, and
   `final_test_runner_fingerprint` all change as a structural
   consequence of `ANALYSIS_FINGERPRINT_MANIFEST` including
   `validation_evaluation.py` directly and
   `CROSS_CONDITION_ADDENDUM_MANIFEST` extending
   `ANALYSIS_FINGERPRINT_MANIFEST`. No manifest is edited to avoid this;
   no fingerprint computation is special-cased to hide it.
3. **Existing model predictions, labels, sample ordering, views,
   checkpoints, and dataset bindings remain valid.** Nothing about
   inference, view generation, checkpoint identity, or dataset checksum
   binding changed -- only how one already-computed probability array
   (the clean anchor) is subsequently reused inside one aggregation
   formula.
4. **The defect affects only original-anchored probability construction
   and downstream metrics derived from that condition.** Confirmed by
   source inspection (Phase 2B.6J Part B): `mean_probability`,
   `majority_vote`, and `confidence_weighted_average` never consume a
   clean-logit/clean-probability anchor at all -- they operate purely on
   `view_log_probs`, identically derived and reused in both the live and
   recompute call sites already, so they were never exposed to this
   divergence.
5. **Mean-probability, majority-vote, confidence-weighted, BN-adapted,
   clean predictions, and primary preregistered endpoints are unchanged
   unless mechanical comparison proves otherwise.** This is a
   requirement on Part B/C's reconciliation mechanism (§6 below), not an
   assumption -- every unaffected condition's persisted metric value
   must be independently re-verified equal, not presumed equal.
6. **Existing validation artifacts remain immutable.** No
   `predictions.npz`, `metrics.json`, `metadata.json`,
   `artifact_manifest.json`, or ledger row for any of the 39 validation
   evaluations is rewritten, amended, or deleted at any point in this
   task.
7. **The 39 validation evaluations will not be rerun.** Corrected
   original-anchored metrics are derived entirely offline from each
   evaluation's already-persisted `clean_probs`/`view_probs` arrays --
   no model, device, checkpoint, or dataset access occurs.
8. **The reconciliation is deterministic integrity repair** -- a pure,
   repeatable function of already-persisted bytes -- not new model
   inference, not hypothesis selection, not statistical analysis, and
   not test-set analysis of any kind.
9. **Cell 1's completed final-test result may be carried forward** only
   because all 56 compatibility checks (7 prefixes x 8 metric fields)
   independently passed under the corrected formula (Phase 2B.6J Part
   C). This finding is reconfirmed, not re-assumed, in Part D of this
   task.
10. **Cell 2 attempt 1 remains permanently failed**; any recovery must
    use attempt 2 under a new authorization generation. Attempt 1 is
    never amended, deleted, or retried.
11. **No tolerance may be loosened.** The frozen `atol=1e-6`/`rtol=1e-6`
    semantic-verification tolerance is unchanged everywhere in this
    task.
12. **No scientific conclusion may be generated during reconciliation.**
    The reconciliation mechanism (Part B/C) never prints, ranks,
    compares, or interprets a metric value -- it only proves hash/
    tolerance equality or inequality as a boolean integrity fact.

## 3. What remains open (addressed by Parts B-F of this task)

* The validation-evidence reconciliation mechanism itself (append-only
  record schema, resolver integration, fail-closed conditions) is
  specified in Part B and implemented there, not by this document.
* Execution of reconciliation across the real 39 validation cells is
  Part C, not this document.
* Final fingerprint computation and cell-1/cell-2 compatibility
  reconfirmation is Part D.
* New final-test authorization generation is Part E.
