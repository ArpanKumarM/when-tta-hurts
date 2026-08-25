# Phase 2B.9A Part B — Isolated Paper-Evidence Plotting Toolchain Freeze

**Status: FROZEN before any code or dependency change.** This document
contains no scientific values. It governs an isolated, separately-locked
plotting subproject that renders publication figures from the already-
sealed, already-committed generation-2 scientific summary. It does not
authorize any new analysis, any read of raw predictions/datasets/
checkpoints, or any change to the root package's scientific dependencies.

## 1. Plotting is downstream presentation only

Rendering figures from `artifacts/final_test_scientific_summary.json` is
a pure, read-only, post-hoc presentation step over already-frozen,
already-authorized scientific facts. It computes no new statistic,
resamples nothing, and touches no prediction array, checkpoint, or
dataset file. It is explicitly NOT scientific-computation infrastructure
and must never be treated as part of the evaluator/analysis/reporting
identity chain.

## 2. Root `pyproject.toml` and root `uv.lock` are immutable

Neither file may be modified by this work. Their content hashes,
recorded before any change in this phase, are:

| File | SHA-256 |
|---|---|
| `pyproject.toml` | `efc9ac9c85e313cfed29b81a08fe54938b829421a03845a18725f0ac4cb6b428` |
| `uv.lock` | `776fbd59567a0d5a5e80d9c46838ac761f10eecf0097e02dc580e4c9b50fb7c5` |

Both are re-verified unchanged after every step in this phase. If either
changes unexpectedly, the offending uncommitted change is reverted
immediately and the task stops.

## 3. Isolated environment location

The plotting environment lives only under:

* `tools/paper_evidence/pyproject.toml`
* `tools/paper_evidence/uv.lock`

Neither file is referenced by, or affects, the root project's
dependency resolution, build, or any existing fingerprint manifest.

## 4. Exact version pinning

Matplotlib is exactly version-pinned in `tools/paper_evidence/pyproject.toml`
(no floor-only `>=` range) and fully transitively locked in
`tools/paper_evidence/uv.lock`, so the isolated environment is
byte-for-byte reproducible on any machine that can run `uv sync
--project tools/paper_evidence --frozen`.

## 5. Local path dependency, one direction only

The isolated project may depend on this repository's own package
(`when_tta_hurts`) via a local path dependency, so the generator can
import `when_tta_hurts.paper_evidence`'s deterministic extraction
helpers. This dependency is one-directional: the root project never
depends on, imports, or is affected by anything under `tools/paper_evidence/`.
Creating or locking the isolated project must never write to, or modify
the resolution of, the root `pyproject.toml`/`uv.lock`.

## 6. Production commands

* `uv sync --project tools/paper_evidence --frozen`
* `uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py plan`
* `uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py generate`

No other invocation form is authorized. No environment variable,
alternate project path, or alternate lock file may be substituted.

## 7. New, disjoint fingerprint

The isolated toolchain and generator receive their own
`paper_evidence_fingerprint`, computed the same content-hash-manifest
way as every other fingerprint in this repository, covering (added in
Part C/D of this phase, per the frozen manifest membership list below)
exactly:

* `tools/paper_evidence/pyproject.toml`
* `tools/paper_evidence/uv.lock`
* `src/when_tta_hurts/paper_evidence.py`
* `scripts/generate_paper_evidence.py`
* this document and the Part B evidence-package freeze document
* the canonical-summary schema/verification code the generator actually
  calls (reused, not reimplemented)

This manifest is disjoint from `ANALYSIS_FINGERPRINT_MANIFEST`,
`CROSS_CONDITION_ADDENDUM_MANIFEST`, `FINAL_TEST_RUNNER_MANIFEST`,
`FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST`, and
`FINAL_TEST_REPORTING_MANIFEST` -- none of those existing manifests are
edited to include any paper-evidence file, and no paper-evidence
manifest includes any of their files beyond the read-only reuse noted
above.

## 8. Existing identities remain unchanged

Every existing fingerprint (evaluator, validation-analysis, cross-
condition, final-test-runner, reconciliation, final-test-analysis,
final-test-reporting) and every existing authorization (generation-5
final-test, generation-3 final-test-analysis, generation-2 unsealing)
must remain byte-identical throughout this phase. This is verified
mechanically before and after every commit.

## 9. No reauthorization required unless an existing fingerprint changes

Since the isolated toolchain's own files are outside every existing
manifest, no existing authorization becomes stale by this work. A new
paper-evidence-specific authorization is not required by this document
(the generator only ever reads the already-sealed, already-committed
canonical summary as opaque, hash-verified input -- it does not unseal
anything new). If, contrary to design, any existing fingerprint is found
to have changed, this is an immediate stop condition and no
reauthorization is performed silently.

## 10. Fail-closed on incidental root modification

If creating or locking the isolated project modifies the root
`pyproject.toml` or root `uv.lock` in any way (including whitespace or
lockfile-metadata changes), the task stops immediately and only those
incidental, uncommitted changes are reverted (via `git checkout --` on
exactly those two paths) -- no other file is touched by that recovery
step.
