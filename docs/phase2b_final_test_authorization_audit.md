# Phase 2B.6B — Final-Test Authorization Audit

**Status: this document records the state at the moment
`artifacts/final_test_authorization.json` is committed as `approved` --
i.e., the point immediately before the official test split is ever
accessed for the first time in this project.** No final-test cell has
been evaluated. No test prediction or test metric of any kind has been
computed or observed. This authorization does not itself access, open,
or index any test array.

## 1. Disclosure: validation outcomes were already observed

Per the accepted history of this project (Phase 2B through Phase 2B.5C),
all 39 confirmatory validation-stage cells completed and their
descriptive validation-stage outcomes were visible before the
post-validation/pre-test secondary addendum (`docs/phase2b_final_test_cross_condition_addendum.md`)
was frozen. That addendum was, and remains, explicitly labeled
**post-validation, pre-test-specified** -- never originally
preregistered. This authorization step changes nothing about that
disclosure; it is repeated here because the three-way disclosure
taxonomy below governs every claim this project will ever make from the
test split forward.

**Authorization is not justified by, and does not reference, any
favorable validation outcome.** The only inputs to this authorization are
identity/provenance bindings (fingerprints, commits, checksums, training
attempt/checkpoint identities) -- never a validation accuracy, delta, or
any other observed metric.

## 2. Three-way disclosure taxonomy (unchanged, restated for the record)

1. **Originally preregistered, primary confirmatory:** the within-cell
   clean-vs-TTA test-set bootstrap and McNemar analyses
   (`docs/statistical_analysis_plan.md`, frozen before any result was
   observed).
2. **Post-validation, pre-test-specified, secondary:** the fixed-model,
   image-paired difference-in-differences estimates and intervals defined
   by `docs/phase2b_final_test_cross_condition_addendum.md` /
   `configs/final_test_cross_condition_addendum.yaml`.
3. **Descriptive/exploratory:** cross-seed summaries, any general
   method-level H1/H2/H3 comparison, Block C's external comparison, Block
   D's 128px trend, calibration metrics, alternative aggregators, BN
   adaptation, N-curves, and any analysis not explicitly registered as
   primary or secondary above.

## 3. Exact authorization scope

* **What is authorized:** exactly one final-test evaluation attempt per
  each of the 39 approved confirmatory matrix cells listed in §4, using
  the frozen scientific computation described in
  `docs/phase2b_final_test_runner_engineering_freeze.md` §1, split=test.
* **What is NOT authorized:** any statistical interpretation of results
  while the matrix is still in progress (§9); any retry beyond the
  no-automatic-retry policy (§7); any cell outside the 39 listed; any
  change to scientific computation, configuration, or protocol; any use
  of this authorization artifact for the separate, unrelated Phase 2B.2
  Validation-Gated TTA (H4) gate (`authorization.py`), which remains
  entirely its own, still-locked mechanism.
* **Authorization is self-invalidating** the moment any bound
  fingerprint, commit-ancestor relationship, dataset checksum, or
  per-cell canonical training identity drifts from what is recorded
  here -- re-verified fresh on every future call, never cached.

## 4. Exact 39-cell ordered manifest

In frozen matrix order (`configs/experiment_matrix.yaml`'s cell
expansion order, identical to `plan_final_test_evaluation()`'s reported
order):

| # | run_id | block | dataset | resolution | normalization | policy | seed | training_attempt | checkpoint_hash (first 16 hex) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | A-pathmnist-28px-batchnorm-policy-none-s0 | A | pathmnist | 28 | batchnorm | none | 0 | 3 | 30bc1ca6ef364e2a |
| 2 | A-pathmnist-28px-batchnorm-policy-none-s1 | A | pathmnist | 28 | batchnorm | none | 1 | 1 | f3be88438078ce36 |
| 3 | A-pathmnist-28px-batchnorm-policy-none-s2 | A | pathmnist | 28 | batchnorm | none | 2 | 1 | b8b971407b6b149d |
| 4 | A-pathmnist-28px-groupnorm-policy-none-s0 | A | pathmnist | 28 | groupnorm | none | 0 | 1 | fcf6a2f41c136cad |
| 5 | A-pathmnist-28px-groupnorm-policy-none-s1 | A | pathmnist | 28 | groupnorm | none | 1 | 2 | b63d8c9b9a38b352 |
| 6 | A-pathmnist-28px-groupnorm-policy-none-s2 | A | pathmnist | 28 | groupnorm | none | 2 | 1 | 483eaa5957f57098 |
| 7 | A-pathmnist-64px-batchnorm-policy-none-s0 | A | pathmnist | 64 | batchnorm | none | 0 | 1 | b5dea833b8985d7b |
| 8 | A-pathmnist-64px-batchnorm-policy-none-s1 | A | pathmnist | 64 | batchnorm | none | 1 | 1 | b898155b2c6ab101 |
| 9 | A-pathmnist-64px-batchnorm-policy-none-s2 | A | pathmnist | 64 | batchnorm | none | 2 | 1 | 39950a5ac56c50ae |
| 10 | A-pathmnist-64px-groupnorm-policy-none-s0 | A | pathmnist | 64 | groupnorm | none | 0 | 1 | d54b5cd32b64882b |
| 11 | A-pathmnist-64px-groupnorm-policy-none-s1 | A | pathmnist | 64 | groupnorm | none | 1 | 1 | ebfef2c0e60ae1ee |
| 12 | A-pathmnist-64px-groupnorm-policy-none-s2 | A | pathmnist | 64 | groupnorm | none | 2 | 2 | 92c8739bfc1063b0 |
| 13 | A-bloodmnist-28px-batchnorm-policy-none-s0 | A | bloodmnist | 28 | batchnorm | none | 0 | 1 | e96e3b5615c71fba |
| 14 | A-bloodmnist-28px-batchnorm-policy-none-s1 | A | bloodmnist | 28 | batchnorm | none | 1 | 1 | 1448dad9629f956c |
| 15 | A-bloodmnist-28px-batchnorm-policy-none-s2 | A | bloodmnist | 28 | batchnorm | none | 2 | 1 | 552c975c860585e9 |
| 16 | A-bloodmnist-28px-groupnorm-policy-none-s0 | A | bloodmnist | 28 | groupnorm | none | 0 | 1 | cd804a09e799f7d8 |
| 17 | A-bloodmnist-28px-groupnorm-policy-none-s1 | A | bloodmnist | 28 | groupnorm | none | 1 | 1 | d914f4ecd0d597b9 |
| 18 | A-bloodmnist-28px-groupnorm-policy-none-s2 | A | bloodmnist | 28 | groupnorm | none | 2 | 1 | 34a1b18158c3572f |
| 19 | A-bloodmnist-64px-batchnorm-policy-none-s0 | A | bloodmnist | 64 | batchnorm | none | 0 | 1 | 8572b19b0cb24b9c |
| 20 | A-bloodmnist-64px-batchnorm-policy-none-s1 | A | bloodmnist | 64 | batchnorm | none | 1 | 1 | ff75298a2d1733dd |
| 21 | A-bloodmnist-64px-batchnorm-policy-none-s2 | A | bloodmnist | 64 | batchnorm | none | 2 | 1 | b2e83aa7e85050f6 |
| 22 | A-bloodmnist-64px-groupnorm-policy-none-s0 | A | bloodmnist | 64 | groupnorm | none | 0 | 1 | 6dc1eb925bf71525 |
| 23 | A-bloodmnist-64px-groupnorm-policy-none-s1 | A | bloodmnist | 64 | groupnorm | none | 1 | 1 | 2278d9cfd09a1652 |
| 24 | A-bloodmnist-64px-groupnorm-policy-none-s2 | A | bloodmnist | 64 | groupnorm | none | 2 | 1 | 9cbeb71c38c6d508 |
| 25 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s0 | B | pathmnist | 28 | batchnorm | matched_to_approved_tta_policy | 0 | 1 | f9d06b302a5a0a73 |
| 26 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s1 | B | pathmnist | 28 | batchnorm | matched_to_approved_tta_policy | 1 | 1 | d8e1b91d421fd3c7 |
| 27 | B-pathmnist-28px-batchnorm-policy-matched_mixed-s2 | B | pathmnist | 28 | batchnorm | matched_to_approved_tta_policy | 2 | 1 | 44263c3e3db4a82d |
| 28 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s0 | B | bloodmnist | 28 | batchnorm | matched_to_approved_tta_policy | 0 | 1 | fe0bc5ac9371bca5 |
| 29 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s1 | B | bloodmnist | 28 | batchnorm | matched_to_approved_tta_policy | 1 | 1 | 233311a29208b194 |
| 30 | B-bloodmnist-28px-batchnorm-policy-matched_mixed-s2 | B | bloodmnist | 28 | batchnorm | matched_to_approved_tta_policy | 2 | 1 | 3d867a7872cb2a7f |
| 31 | C-dermamnist-28px-resnet18-batchnorm-policy-none-s0 | C | dermamnist | 28 | batchnorm | none | 0 | 1 | bd529f57be5f0604 |
| 32 | C-dermamnist-28px-resnet18-batchnorm-policy-none-s1 | C | dermamnist | 28 | batchnorm | none | 1 | 1 | ab087ce6cf7dfc7a |
| 33 | C-dermamnist-28px-resnet18-batchnorm-policy-none-s2 | C | dermamnist | 28 | batchnorm | none | 2 | 1 | 44881dca162455ee |
| 34 | D-pathmnist-128px-batchnorm-policy-none-s0 | D | pathmnist | 128 | batchnorm | none | 0 | 1 | 276169a842c64261 |
| 35 | D-pathmnist-128px-batchnorm-policy-none-s1 | D | pathmnist | 128 | batchnorm | none | 1 | 1 | 7fc1cf8f52c42029 |
| 36 | D-pathmnist-128px-batchnorm-policy-none-s2 | D | pathmnist | 128 | batchnorm | none | 2 | 1 | 2c302516e9cc7a6d |
| 37 | D-bloodmnist-128px-batchnorm-policy-none-s0 | D | bloodmnist | 128 | batchnorm | none | 0 | 1 | ec97e1b8b34af028 |
| 38 | D-bloodmnist-128px-batchnorm-policy-none-s1 | D | bloodmnist | 128 | batchnorm | none | 1 | 1 | f3980a44fb10c1bf |
| 39 | D-bloodmnist-128px-batchnorm-policy-none-s2 | D | bloodmnist | 128 | batchnorm | none | 2 | 1 | f0ddb47ab6d78491 |

The **complete, machine-generated, non-abbreviated** 39-row listing (every
`run_id`, `training_attempt`, `checkpoint_hash`, `block`, `dataset`,
`resolution`, `normalization`, `training_policy`, `seed`, and, for Block D
cells, `effective_training_config_hash`) is persisted verbatim in
`artifacts/final_test_authorization.json`'s `authorized_cells` array,
which is the authoritative record -- this table is a human-readable
excerpt, not a second source of truth. Block counts, mechanically
verified against the frozen matrix: **A=24, B=6, C=3, D=6, total=39**,
zero duplicates, zero missing cells, zero extra cells, zero occurrences
of the permanently-excluded seed 314159, and seeds `{0,1,2}` for every
block that requires all three.

## 5. Checkpoint and configuration bindings

For every one of the 39 cells, `authorized_cells[i].training_attempt` and
`authorized_cells[i].checkpoint_hash` were read directly from
`resolve_canonical_training_completion(run_id)` -- the same production
selection logic the validation-evaluation and final-test runners already
use -- at authorization-construction time, and independently re-verified
to match a fresh call to that same function immediately before
committing (§ mechanical verification, below). Block D cells additionally
carry `effective_training_config_hash`, computed via
`compute_block_d_effective_config_hash(authorize_block_d_cell(cell))`
-- the same effective-config binding Block D's own gate decision already
uses (`artifacts/block_d_gate_decision.json`, `final_decision: "INCLUDED"`,
tracked and clean).

## 6. Fingerprint values (bound and independently reconfirmed)

| Fingerprint | Value |
|---|---|
| Evaluator | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` |
| Statistical analysis | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` |
| Cross-condition addendum | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` |
| Final-test runner | `0c1a6ac0f84765cdfa52b42cd0ef698df00bd78fa8c6d12fb7e5eb196c80840e` |

All four were recomputed fresh via `compute_evaluator_fingerprint()`,
`compute_analysis_fingerprint()`, `compute_cross_condition_fingerprint()`,
and `compute_final_test_runner_fingerprint()` immediately before
constructing the authorization payload, and match the values accepted at
the close of Phase 2B.6A exactly (the cross-condition-addendum value's
prior shift, from `0935dd11...` to `5843f613...`, was already disclosed
and accepted in Phase 2B.6A as a mechanical consequence of extending
`ledger.py`, which is a member of `CROSS_CONDITION_ADDENDUM_MANIFEST`).

Provenance commits bound in the artifact -- `ce4c962` (protocol),
`72b63e2` (matrix/freeze-doc commit), `35e373a` (cross-condition-addendum
freeze commit) -- and the runner-source commit `c3849f471e0331679e9c1851adbf751d19a787cb`
were each independently confirmed to be ancestors of HEAD via
`git merge-base --is-ancestor`.

## 7. Dataset-checksum bindings

Official expected MD5 checksums (from `medmnist.INFO`, via
`expected_official_checksum()`), bound for every dataset/resolution
combination the 39-cell matrix actually requires:

| Dataset@Resolution | Expected MD5 |
|---|---|
| pathmnist@28 | `a8b06965200029087d5bd730944a56c1` |
| pathmnist@64 | `55aa9c1e0525abe5a6b9d8343a507616` |
| pathmnist@128 | `ac42d08fb904d92c244187169d1fd1d9` |
| bloodmnist@28 | `7053d0359d879ad8a5505303e11de1dc` |
| bloodmnist@64 | `2b94928a2ae4916078ca51e05b6b800b` |
| bloodmnist@128 | `adace1e0ed228fccda1f39692059dd4c` |
| dermamnist@28 | `0744692d530f8e62ec473284d019b0c7` |

All 7 combinations were independently re-verified against
`verify_official_dataset_artifact()` (whole-file MD5 against the locally
present artifact) immediately before this document was written; all 7
returned `checksum_verified=True`, `resized=False`. Exact values are
recorded verbatim in `artifacts/final_test_authorization.json`'s
`official_dataset_checksums` field -- not duplicated here to avoid a
second, driftable source of truth.

## 8. Retry/failure policy

**No automatic retry** exists anywhere in the runner or CLI (mechanically
confirmed: no `retry`/`force`/`bypass` parameter on
`run_final_test_evaluation()` or `verify_final_test_authorization()`, no
`--retry`/`--force` CLI flag). A compatible completed attempt
idempotently skips before any device/checkpoint/dataset access. An
incompatible completed attempt (different identity) hard-fails via
`ConflictingEvaluationImplementationError` rather than being silently
superseded. Every lifecycle/access flag
(`test_split_accessed`/`test_predictions_computed`/`test_metrics_computed`/
`test_metrics_persisted`/`test_metrics_observed`/`failure_stage`) is
recorded truthfully on both success and failure -- "not persisted" is
never conflated with "not accessed."

## 9. Execution discipline

The future execution phase (not authorized to begin by this document)
must: evaluate cells sequentially, in the frozen matrix order recorded in
§4, invoking each cell's `evaluate-test --run-id <run_id>` exactly once;
halt immediately on any failure rather than continuing to the next cell;
never run cells in parallel; never let a result from one cell influence
whether or how a later cell is run (no result-dependent stopping); never
read a scientific metric value between cells (only integrity metadata --
ledger status, attempt existence -- needed to confirm a prior cell's safe
completion); and **must not interpret any cell's result, individually or
collectively, until the entire authorized 39-cell matrix has completed.**
This is a discipline requirement on the future execution phase, not
something this authorization step itself performs or could violate,
since this document authorizes and this repository's engineering commits
contain no matrix-wide execution driver at all (only the single-cell CLI
primitive, per `docs/phase2b_final_test_runner_engineering_freeze.md`
§6).

## 10. Confirmation: authorization accesses no test array

`verify_final_test_authorization()` (the function invoked to construct
and validate this artifact) contains no import of `torch`, no call to
`select_device()`, no call to `np.load()` on a `predictions.npz`-shaped
path, and no reference to a test-array-loading function -- confirmed by
static source inspection (`tests/test_final_test_authorization.py::test_verify_final_test_authorization_never_imports_torch_or_touches_mps`)
and by this session's own commands, none of which invoked `evaluate-test`,
initialized MPS, loaded a checkpoint, or indexed a test array.

## 11. Confirmation: no test metric has yet been observed

* `artifacts/ledger_final_test.csv` remains header-only (1 line).
* No `artifacts/final_test/` directory exists.
* No `test_metrics_observed=True` row exists anywhere in this project's
  ledgers.
* No `evaluate-test` command has been executed against a real run ID
  with a valid authorization in place at any point in this session (the
  one prior `evaluate-test` invocation, in Phase 2B.6A, was run
  specifically to prove the pre-authorization refusal path, and correctly
  refused before any device/data access).

## 12. Confirmation: no scientific choice remains open after authorization

Every scientific parameter the final-test runner will use is already
frozen and bound into this authorization: checkpoint selection,
preprocessing, deterministic view generation, TTA seed `1306178015`,
policy identifier, `MAX_VIEWS=100`, prefixes `[1,2,5,10,25,50,100]`,
primary `N=50`, mean-probability primary aggregation, alternative
aggregators, BN adaptation, inference batch size 256, BN-adaptation
batch size 256, `sequential_microbatch_v1`, the probability-native metric
contract, and latency boundaries -- all read from the same frozen
`configs/validation_evaluation.yaml` the validation evaluator already
used, now bound by fingerprint into this authorization. The
post-validation/pre-test secondary addendum's estimand, pairing,
bootstrap parameters, and forbidden/permitted interpretations are
likewise already frozen and bound by fingerprint. No open scientific
decision remains for the execution phase to make.

## 13. Nonbinding runtime and storage projections

Derived **only** from already-recorded validation-evaluation runtimes and
artifact sizes (`artifacts/ledger_validation_evaluation.csv`,
`artifacts/validation_evaluation/`) -- **not** from any observed test-set
outcome, and **not** a guarantee, since the official test split's sample
count per dataset/resolution has not been verified equal to the
validation split's sample count in this session:

* 43 completed validation-evaluation attempts (39 unique cells + 4
  reconciliation reruns) totaled ~51.3 wall-clock hours; block-specific
  means were A≈68.4 min/cell, B≈86.4 min/cell, C≈13.7 min/cell,
  D≈101.1 min/cell.
* Applying these same per-block means to exactly 39 final-test cells
  (24×A + 6×B + 3×C + 6×D) projects to approximately **46.8 wall-clock
  hours** if run fully sequentially with no pauses -- an order-of-magnitude
  estimate only, since actual final-test runtime scales with the official
  test split's actual sample count, which may differ from validation's.
* `artifacts/validation_evaluation/` currently totals ~916 MB across 43
  attempts (~21 MB/attempt average, dominated by `predictions.npz`'s
  100-view probability bank, which scales with sample count and class
  count). Projecting the same per-attempt average to 39 final-test cells
  suggests an order-of-magnitude storage requirement of roughly
  **800 MB-1 GB**, again not a guarantee.

## 14. Disclosure: results must not be interpreted mid-matrix

Restated from §9 for emphasis, since it is a disclosure requirement, not
merely an execution-discipline note: whichever future session executes
the authorized matrix must not draw any scientific conclusion, adjust any
downstream decision, or report any interim finding from a subset of the
39 cells while cells remain outstanding. The confirmatory and secondary
analyses (§2) are defined over the complete, authorized set; a partial
read would be indistinguishable from -- and carries the same integrity
risk as -- optional stopping.
