# Phase 2B.7A — Final-Test Statistical-Analysis Runner: Frozen Design

**Status: FROZEN before any production code is written.** This document
governs the implementation that follows. It contains no scientific
values -- design and binding rules only.

## 0. Motivation

Phase 2B.6O Part C found that `statistical_analysis.py` and
`cross_condition_addendum.py` -- as engineered in Phase 2B.5A/2B.5C,
before the final-test matrix existed -- read exclusively from
`artifacts/validation_evaluation/` and
`artifacts/ledger_validation_evaluation.csv`. Neither module has any
final-test-aware code path. This document freezes how a final-test-aware
runner is added WITHOUT touching that existing, already-frozen,
validation-only behavior.

## 1. Non-negotiable: validation behavior is untouched

`plan_statistical_analysis()`, `compute_family_analysis()`,
`plan_cross_condition_addendum()`, `compute_hypothesis_did()`, and every
existing default parameter, ledger path, and root directory in
`statistical_analysis.py` / `cross_condition_addendum.py` remain byte-
for-byte as they are. No existing validation-mode function signature,
default, or return-value shape changes. This is enforced mechanically in
Part 2 of this phase by hashing both files before and after and
requiring the hash to match, and by the existing validation-mode test
suite passing unmodified.

## 2. No repoint, no shared default

The final-test runner is **additive**, not a substitution. It does not
change `validation_evaluation_root="artifacts/validation_evaluation"` or
`ledger_path="artifacts/ledger_validation_evaluation.csv"` to point
anywhere else, and it does not introduce a flag/branch inside the
existing functions that silently switches their target. All final-test
logic lives in new functions, in a new module
(`src/when_tta_hurts/final_test_statistical_analysis.py`), with its own
hardcoded final-test root (`artifacts/final_test`) and ledger
(`artifacts/ledger_final_test.csv`) -- the SAME paths the confirmatory
final-test runner (`final_test_evaluation.py`) already uses, never a new
or alternate path.

## 3. New, explicit final-test analysis modes

Two new library-level capabilities, mirroring the validation-mode split
exactly:

* **Preregistered within-cell analysis** (H1/H2/H3/BLOCK_C):
  `plan_final_test_statistical_analysis()` (plan, side-effect-free) and
  `compute_final_test_family_analysis()` (real analyze, persists on
  success).
* **Secondary cross-condition addendum** (H1/H2/H3 fixed-pair DiD):
  `plan_final_test_cross_condition_addendum()` (plan) and
  `compute_final_test_hypothesis_did()` (real analyze, persists on
  success).

Both read ONLY from `artifacts/final_test/<run_id>/attempt_NNN/` and
`artifacts/ledger_final_test.csv`. Neither has a
`validation_evaluation_root` parameter, an environment variable, or a
CLI flag that could redirect it elsewhere.

## 4. Validation and reconciliation are provenance-only

`validation_metric_reconciliation.py`'s ledger and the validation
evaluation artifacts are readable by the new module ONLY for cross-
referencing/provenance narration (e.g. confirming a training checkpoint
history matches what validation already saw) -- never as a substitute
input to a final-test statistic. No final-test analysis function may
compute a bootstrap CI, McNemar test, effect size, or DiD estimate from
a `predictions.npz` under `artifacts/validation_evaluation/`. This is
enforced by the new module never importing
`_load_pair_correctness`/`compute_family_analysis`'s validation-rooted
internals for computation -- only the pure, data-independent math
helpers are reused (see sec.5).

## 5. Frozen mathematics reused unchanged, never reimplemented

Imported directly from `statistical_analysis.py` and
`cross_condition_addendum.py`, with zero modification:

* `paired_bootstrap_ci` (primary N=50, mean-probability aggregation,
  >=10,000 resamples, 95% CI, joint-index resampling);
* `mcnemar_test` (exact binomial <25 discordant / continuity-corrected
  chi-square otherwise);
* `benjamini_hochberg` (FDR correction, no alpha, no verdict);
* `effect_sizes` (harm/rescue rate, delta accuracy);
* `did_point_estimate` / `did_bootstrap_ci` / `per_image_did` /
  `derive_bootstrap_seed` (cross-condition joint resampling DiD);
* `derive_family_cells` / `derive_fixed_pairs` / `load_addendum_spec`
  (matrix-derived, data-independent -- identical cell/pair sets apply to
  final-test since it is the same 39-cell matrix);
* `compute_analysis_id` (generic identity hash, family-agnostic).

No formula, threshold, resample count, or CI level is redefined,
copied, or approximated differently for the final-test path.

## 6. Final-test resolution requirements

Before any final-test cell may contribute to a plan report entry marked
eligible, or to a real analysis computation, ALL of the following must
hold, verified via `final_test_authorization.verify_final_test_authorization()`
and `_classify_final_test_cell()` (both reused unchanged, never
reimplemented):

* the authorization artifact is `approved` and current (git-tracked,
  clean, all bound commits are ancestors of HEAD, fingerprints match the
  current repository state);
* the cell's classification is exactly `completed_consumed` (not
  `pending`, not `invalid`);
* the cell's authorized attempt exactly matches the ledger's completed
  row for that run_id;
* all five final-test lifecycle flags on that ledger row are `True`
  (`test_split_accessed`, `test_predictions_computed`,
  `test_metrics_computed`, `test_metrics_persisted`,
  `test_metrics_observed`);
* the attempt directory's `artifact_manifest.json` verifies byte-for-
  byte against the files on disk;
* checkpoint hash and training attempt match the authorization's
  binding for that cell;
* the ledger row's `authorization_artifact_sha256` matches the run_id's
  EXPECTED historical binding (see sec.7 -- this is the current
  generation-5 hash for every cell except cell 1);
* for any paired (cross-condition) comparison, the two cells' `labels`
  and `sample_indices` arrays are checked for exact equality before any
  DiD is computed -- a mismatch raises, never silently drops or reorders
  a sample.

A family or hypothesis is `complete` (and eligible for real analysis)
only when EVERY one of its required cells/pairs independently satisfies
every bullet above. Partial families are reported, never computed.

## 7. Cell 1's carried-forward provenance, explicitly preserved

Cell 1 (`A-pathmnist-28px-batchnorm-policy-none-s0`) completed under
generation-3 authorization
(`0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a`,
commit `f8e794053926a275d4eb503f2994668577435317`), not generation-5. Its
ledger row and metadata legitimately record generation-3's fingerprints
and authorization hash. The final-test resolver therefore checks cell
1's `authorization_artifact_sha256` against this EXACT historical value,
via a small, explicit, hardcoded allow-list keyed by run_id -- every
other cell is checked against the CURRENT (generation-5) authorization's
`artifact_sha256`. This is a verification-time acknowledgment of real,
already-audited history (`docs/phase2b_final_test_matrix_closure_audit.md`
sec.5), not a correction of the generation-5 authorization JSON's own
`consumed_binding` metadata field, which remains uncorrected and
undisclosed-only-here-again for cross-reference. No other cell may ever
use this allow-list path; a run_id not present in it must match the
CURRENT authorization exactly or be rejected.

## 8. Plan mode: metadata-only, zero prediction loads

`plan_final_test_statistical_analysis()` and
`plan_final_test_cross_condition_addendum()` never call `np.load`, never
open `predictions.npz`, and never open `metrics.json`. They read only:
git state (via the reused authorization verifier), the final-test
ledger's CSV rows, and `metadata.json`/`artifact_manifest.json` (identity
and hash metadata, never a probability array). This is verified by a
dedicated test that monkeypatches `numpy.load` to raise if called during
either plan function's execution.

## 9. Real analysis: atomic, sealed persistence

A successful real-analysis run:

1. Resolves every required cell/pair per sec.6, hard-failing (no partial
   result) on the first unmet requirement.
2. Computes results using ONLY the reused math (sec.5).
3. Validates the result against the EXISTING, unmodified
   `statistical_analysis_artifacts.validate_analysis_result_schema()` /
   `validate_cross_condition_result_schema()` (both already require
   `test_split_accessed is False`, a non-empty result, and full-tree
   finiteness -- reused unchanged since a final-test analysis result has
   the identical shape and never touches the raw test-split data itself,
   only its own already-persisted, already-verified prediction arrays).
4. Persists atomically via the EXISTING, unmodified
   `persist_and_verify_analysis_completion()` /
   `persist_and_verify_cross_condition_completion()`, into a NEW root
   disjoint from every validation-mode path:
   `artifacts/final_test_analysis/<family>/attempt_NNN/` and
   `artifacts/final_test_cross_condition/<hypothesis>/attempt_NNN/`.
5. Appends exactly one row to a NEW, append-only ledger,
   `artifacts/ledger_final_test_analysis.csv`, only after step 4
   succeeds.
6. Never marks anything `completed` through a partial write -- any
   failure at any step above leaves no ledger row and no `status.json`
   claiming success.

## 10. CLI output stays sealed

Any new CLI surface (extending `scripts/run_statistical_analysis.py`
with final-test-specific plan/analyze subcommands) prints only:
authorization status, cell/pair counts, eligibility booleans, analysis
IDs, fingerprints, hashes, and file paths. No accuracy, F1, NLL, ECE,
Brier, harm/rescue, delta, CI bound, or p-value is ever printed by a
plan-mode invocation. A real analyze-mode invocation's output is
persisted to disk exactly as computed (it is a genuine scientific
artifact once real analysis is authorized and run) -- it is never echoed
to stdout by this engineering task, and no engineering-task test or
script here ever actually invokes analyze mode against real final-test
data.

## 11. No bypass

No CLI flag, environment variable, alternate root argument, or
validation-fallback path exists anywhere in the new module or its CLI
wiring. Every final-test-analysis function accepts fixed, hardcoded
final-test paths; the only parameterization that exists is for pointing
tests at synthetic `tmp_path` fixtures, exactly mirroring
`_resolve_canonical_evaluation_identity`'s existing test-seam discipline
in `statistical_analysis.py`.

## 12. Separate authorization gate before real analysis

A real final-test analysis (family or cross-condition) may not run
against real repository data until a SEPARATE, dedicated authorization
artifact (`artifacts/final_test_analysis_authorization.json`) exists,
is `approved`, is git-tracked-and-clean, and binds:

* the final-test closure commit (`581143e`, this phase's `results: close
  Phase 2B final-test matrix`);
* the generation-5 final-test authorization's hash and commit;
* the exact 39 final-test evaluation IDs, attempts, and artifact hashes;
* current evaluator and final-test-runner fingerprints;
* a new final-test-statistical-analysis fingerprint and a new final-
  test-cross-condition fingerprint (both computed the same
  content-hash-manifest way as every other fingerprint in this repo);
* the validation-reconciliation artifact hash (provenance only, per
  sec.4);
* the frozen SAP (`docs/statistical_analysis_plan.md`,
  `566840a15e11d3fafe4aa781e705e2c8ac005dd21c5c79c93da03bcb74b69fca`)
  and addendum config (`configs/final_test_cross_condition_addendum.yaml`,
  `bf2f1a260c2906d974659f65434babb8a3196fa5ca70948b84219edca36abfc5`)
  content hashes.

This gate is engineered in this phase but is never satisfied by this
phase's own commits alone -- creating `artifacts/final_test_analysis_authorization.json`
is itself a separate, later, explicitly-authorized step (Phase 2B.7A's
final part), and running real analysis against it is a further,
separately-authorized step after that. This document does not authorize
either.
