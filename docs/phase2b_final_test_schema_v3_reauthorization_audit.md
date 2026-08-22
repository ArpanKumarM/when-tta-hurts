# Phase 2B.6F — Schema-v3 Final-Test Reauthorization Audit

**Status: this document records the third-generation superseding
final-test authorization, issued after the Phase 2B.6C-Incident, the
Phase 2B.6E attempt-2 pre-access failure, and the Phase 2B.6F engineering
correction.** No final-test cell has been executed under this
authorization. This document does not itself access, open, or index any
test array.

## 1. Note on "schema-v3"

The enforced `schema_version` literal in the authorization JSON remains
`"phase2b.6d-v2"` -- the production verifier's `_SUPPORTED_SCHEMA_VERSIONS`
allowlist was not modified in this task (Part E authorizes generating a
new authorization file, not a further production source change).
Phase 2B.6F's engineering fix (commit `dc97a047f694e5b4683d8a60b4379ca9186ab3f5`)
changed *runtime verification behavior* (the immutable-receipt mechanism)
but introduced no new *required authorization-file field* beyond what
schema `phase2b.6d-v2` already required -- `dataset`/`resolution` were
already present in every schema-v2 authorized-cell entry. This
authorization is therefore the **third content generation** in the
supersession chain, tracked explicitly via a new `authorization_generation: 3`
field (descriptive only, not schema-enforced), rather than a bumped
`schema_version` string. This distinction is disclosed here rather than
silently glossed over.

## 2. Full authorization chain

| Generation | SHA-256 | Commit | Disposition |
|---|---|---|---|
| 1 (schema `phase2b.6b-v1`) | `ccff976dda7d93e8fdd8c4bc2fe78eaecef5316bb011b1e420ed7d30a9637ec2` | `76c46e2a25e95a397198f4a03d56a9db58ab6877` | Superseded (Phase 2B.6D) -- historically valid, permanently preserved via Git history. |
| 2 (schema `phase2b.6d-v2`) | `960b54358a356442c58957cf2ecdec2da916e72d1a01b1d29d5c7d162f8afdc0` | `69fff1e2ebd569e6d017d80674ca2555086e668b` | **Suspended** (Phase 2B.6F) after the attempt-2 pre-access failure; superseded by this generation. Historically valid, permanently preserved via Git history. |
| 3 (schema `phase2b.6d-v2`, generation 3) | `0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a` | *(this commit)* | **Active.** |

**Referenced provenance commits** (all independently confirmed ancestors
of HEAD):
* `1ee5f737c8eccd39ca0d1a49183cfffda1051c2a` -- accidental-access incident record.
* `965daebbff2af976ea05f88cffb1eda848d36b12` -- incident recovery-policy freeze.
* `28d0c3d7de477b83dda5819586e4d89a0ffd1c4b` -- attempt-2 pre-access failure record.
* `69a2c5902c5575f0bc9946be954ef980eccb1093` -- authorization-receipt design freeze.
* `dc97a047f694e5b4683d8a60b4379ca9186ab3f5` -- engineering fix (receipt mechanism) and this authorization's bound `final_test_runner_source_commit`.

## 3. Exact reason attempt 3 is permitted

`next_evaluation_attempt_number()` (the same production function the
authorization verifier itself calls) returns **3** for
`A-pathmnist-28px-batchnorm-policy-none-s0`, computed as
`max(directory attempt numbers {1, 2}, ledger attempt numbers {1, 2}) + 1`.
Both prior attempts are terminal (`aborted`, `failed`) and fully
ledger-reconciled -- confirmed by the same
`check_final_test_evaluation_skip()` reconciliation logic Phase 2B.6F's
engineering fix corrected. Attempt 3 is therefore the mechanically
correct next allocation, not a manually chosen number.

## 4. Proof attempt 2 accessed no test data

Restated from `docs/phase2b_final_test_attempt2_preaccess_failure.md`
§3 (mechanically re-verified, unchanged since that document was
committed): device initialization and checkpoint restoration succeeded;
the official dataset whole-file checksum was verified (raw bytes only,
never an array read); the failure occurred in the test-only loader's own
authorization re-check, strictly before its own checksum re-check,
`load_dataset()`, or any `DataLoader` materialization. `test_split_accessed=False`,
`test_predictions_computed=False`, `test_metrics_computed=False`,
`test_metrics_persisted=False`, `test_metrics_observed=False` for
attempt 2 -- all proven, none inferred.

## 5. Old and new fingerprints

| Fingerprint | Generation 2 (`69fff1e`) | Generation 3 (current) | Changed? |
|---|---|---|---|
| Evaluator | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | `7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef` | No |
| Statistical analysis | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` | `fa0cb164f062253b58b4af37e6278f6aca005cf1c16ab29286e35ae3209e3450` | No |
| Cross-condition addendum | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` | `5843f613df4cac4bacef81bb4b6db420f8ae51d2c0c9efef539f6cc20b96b98c` | No |
| Final-test runner | `efc57a362da86f70b22b6120de9c87a599f418261a180ca56a4a3a9b9de93b4f` | `54bc3f58c8a91fc3d2b7a58c6336e722dbd4cb24a8a8239e5e17fd8591e95877` | **Yes** -- expected, disclosed consequence of the receipt-mechanism fix (Part D). |

No other fingerprint changed at any point across the Phase 2B.6E/2B.6F
recovery -- confirmed by direct recomputation after every commit.

## 6. Exact 39-cell attempt mapping

* `A-pathmnist-28px-batchnorm-policy-none-s0`: **attempt 3 only**.
  `prior_attempt_dispositions`: `{"1": "aborted_accidental_access_incident_1ee5f73", "2": "failed_preaccess_authorization_defect_28d0c3d"}`.
* All other **38** cells: **attempt 1 only**.
* No attempt 4 is authorized for the affected cell, and no attempt 2 is
  authorized for any other cell -- both mechanically enforced by the
  same exact-attempt binding check schema `phase2b.6d-v2` already
  requires (`authorized_final_test_attempt` must equal
  `next_evaluation_attempt_number()` exactly, recomputed fresh on every
  verification).

## 7. Proof scientific computation is unchanged

* `evaluator_fingerprint` and `statistical_analysis_fingerprint` are
  bit-for-bit identical across all three authorization generations --
  nothing in the frozen scientific computation, TTA configuration, or
  statistical-analysis/cross-condition-addendum definitions changed at
  any point during this incident/recovery sequence.
* `phase2b_protocol_commit`, `matrix_commit`, and
  `cross_condition_addendum_commit` are unchanged from generation 1.
* All 39 cells' `training_attempt`/`checkpoint_hash` bindings are
  unchanged and independently re-verified against
  `resolve_canonical_training_completion()` at construction time.
* All 7 official dataset checksums are unchanged and independently
  re-verified against `expected_official_checksum()`.
* The only things that changed across generations 2->3 are: (a) the
  final-test-runner fingerprint (the receipt-mechanism fix, §5), and (b)
  the affected cell's authorized attempt number (2 -> 3, a mechanical
  consequence of the prior attempt's termination, §3) -- both
  procedural/engineering, never scientific.

## 8. Required paper disclosure

The eventual paper/audit must state: the final-test evaluation for
`A-pathmnist-28px-batchnorm-policy-none-s0` was preceded by two
non-scientific incidents during test-harness development -- (1) an
accidental, unauthorized-by-intent execution
(`docs/phase2b_final_test_accidental_access_incident.md`) and (2) a
pre-access authorization-verification defect
(`docs/phase2b_final_test_attempt2_preaccess_failure.md`) -- neither of
which ever persisted or exposed any test metric. The cell was
subsequently evaluated once, for real, as attempt 3, under this
third-generation superseding authorization. This must never be described
as if the original single-pass design had proceeded unbroken for this
one cell.

## 9. Execution discipline (restated, unchanged)

The authorized execution must proceed sequentially, in the frozen
39-cell matrix order, invoking each cell's `evaluate-test --run-id
<run-id>` exactly once at its authorized attempt number (3 for the
affected cell, 1 for every other cell). Any new failure halts the entire
matrix immediately and requires separate incident adjudication -- no
automatic retry exists anywhere in this design. No cell's result may be
interpreted, individually or collectively, until the entire authorized
39-cell matrix has completed.
