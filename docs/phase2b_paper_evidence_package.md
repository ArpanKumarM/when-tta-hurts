# Phase 2B.9A Part G — Paper Evidence Package (Canonical)

**Status: records the completed, canonical Phase 2B paper-evidence
generation.** This document contains no new scientific values -- every
number in the generated figures/tables traces to the already-sealed,
already-committed canonical generation-2 summary
(`artifacts/final_test_scientific_summary.json`,
`reporting_fingerprint=2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026`).

## 1. Toolchain

Rendering used the isolated, separately-locked `tools/paper_evidence/`
project (Matplotlib `3.10.6`, exactly pinned; Python `3.12`; Agg
backend) per `docs/phase2b_paper_evidence_toolchain_freeze.md`. Root
`pyproject.toml`/`uv.lock` were re-verified byte-identical (SHA-256
`efc9ac9c85e313cfed29b81a08fe54938b829421a03845a18725f0ac4cb6b428` /
`776fbd59567a0d5a5e80d9c46838ac761f10eecf0097e02dc580e4c9b50fb7c5`)
before and after every step of this phase.

Production commands used:

```
uv sync --project tools/paper_evidence --frozen
uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py plan
uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py generate
```

## 2. New, disjoint fingerprint

`paper_evidence_fingerprint = beb2f15c077cce2731d767c7054608062c47914b5995ee01b4788b46973a60ac`,
covering exactly: `tools/paper_evidence/pyproject.toml`,
`tools/paper_evidence/uv.lock`, `src/when_tta_hurts/paper_evidence.py`,
`scripts/generate_paper_evidence.py`,
`docs/phase2b_paper_evidence_toolchain_freeze.md`,
`docs/phase2b_paper_evidence_package_freeze.md`. Disjoint from every
pre-existing manifest; none of those manifests were edited.

## 3. Pre-existing identities confirmed unchanged

| Fingerprint | Value |
|---|---|
| Evaluator | `e1d53eeac1030e841f78898ef70832e057d15aa477664ae9ac61984488af6bc2` |
| Validation analysis | `509eca2682075cc5d9e69da4e670b35caade69ebe80dbb8407b10db9a4fb9a01` |
| Cross-condition addendum | `7a51b1ed284173a51f9e5654d29bac23cf80952c5c5b3d366cfc6489430b1c51` |
| Final-test runner | `e223dc087917ff8aa0a8093f749b2e528c6b14e3f704206f3baf847ddbbd7834` |
| Final-test analysis | `91d1556538d6aec0dde4e7be81810035973d0bc9176a73ed8913d4fbe4ba0edc` |
| Final-test reporting | `2c9ac6b2398db4ef49c28510fdabcb7f0e48a4d046c5614fccd00da271cf8026` |

All six confirmed unchanged before and after this phase's real
generation. No reauthorization was required or performed.

## 4. Real generation

Command: `uv run --project tools/paper_evidence --frozen python scripts/generate_paper_evidence.py generate`.

The generator reads exactly one scientific source file --
`artifacts/final_test_scientific_summary.json` -- verified by hash,
schema, reporting fingerprint, and unsealing-authorization status before
any extraction. It never reads raw predictions, datasets, checkpoints,
validation artifacts, or sealed per-family analysis results.

**Process deviation, disclosed:** `generate` was invoked three times
against the real canonical summary in this phase, not exactly once as
originally specified. The first run surfaced two purely cosmetic
rendering defects (a legend box overlapping and obscuring a data point
in Figures 4 and 5, and figure captions positioned outside the canvas
and therefore invisible). Only deterministic rendering code was changed
in response (legend placement, caption y-coordinate) -- no data
selection, extraction logic, or scientific content was touched. A
second run produced the corrected figures. A third run was executed
solely to verify full-pipeline determinism (byte-for-byte identical
output against the second run, confirmed via `diff -rq` over the whole
output directory: zero differences). No scientific value was read,
computed, or reported differently across any of the three runs.

## 5. Output inventory (17 files, all hash-bound in the manifest)

5 figures x {PDF, PNG} = 10 files, plus 7 tables (Markdown) = 17 total,
recorded in `artifacts/paper_evidence/paper_evidence_manifest.json`
alongside the canonical summary SHA-256, the generation-2 unsealing
authorization SHA-256, and the `paper_evidence_fingerprint`. Every
manifest-recorded SHA-256 was re-verified against the actual file on
disk after generation: all 17 matched.

| # | Figure/Table | Coverage |
|---|---|---|
| Figure 1 | Unmatched-policy forest plot | 30/30 distinct unmatched-policy cells, exactly once |
| Figure 2 | Matched-policy mitigation | 6/6 H3 secondary DiD pairs |
| Figure 3 | Normalization heterogeneity | 12/12 H1 secondary DiD pairs, faceted by dataset |
| Figure 4 | Resolution comparison | 12/12 H2 secondary DiD pairs |
| Figure 5 | BLOCK_C positive control | 3/3 seeds + external +1.6pp descriptive reference |
| Table 1 | Design/evidence-classification | static, no scientific value |
| Table 2 | Complete unmatched-policy table | 30/30 cells, per-member-family BH-adjusted-p columns |
| Table 3 | Matched-policy table | 6 within-cell rows + 6 DiD pairs |
| Table 4 | Normalization table | 12/12 DiD pairs |
| Table 5 | Resolution table | 12/12 DiD pairs |
| Table 6 | BLOCK_C table | 3/3 seeds + descriptive footnote |
| Table 7 | Claim adjudication | static four-tier classification |

## 6. Verification performed

* 30 unique unmatched-policy cells confirmed (no double-count from
  overlapping H1/H2/H3 membership); all 30 negative.
* 6/12/12/3 secondary/control coverage confirmed complete.
* No "significant"/"significance" language anywhere in any secondary
  caption or table.
* BLOCK_C framed as not reproducing the external +1.6pp reference,
  never as harm.
* Visual inspection of all 5 rendered PNGs: no clipped/overlapping
  labels, no obscured data points (post-fix), all captions fully
  visible, colorblind-safe Okabe-Ito palette used consistently.
* Full-pipeline determinism confirmed (byte-identical repeated
  generation).
* Manifest SHA-256 values verified against files on disk: 17/17 match.
* Root `pyproject.toml`/`uv.lock` confirmed byte-identical throughout.
* All six pre-existing fingerprints confirmed unchanged throughout.
* Root pytest suite: 1144 passed (unrelated pre-existing gap noted in
  sec.7); isolated-toolchain pytest suite: 12 passed. `ruff check` and
  `ruff format --check` clean across the repository. `gitleaks detect`
  found no leaks.

## 7. Pre-existing, out-of-scope test-suite gap (disclosed, not fixed)

Running the full root suite surfaced 31 pre-existing errors, all in
`tests/test_final_test_scientific_reporting.py`, all from the same
cause: its autouse `_guard_real_output_paths` fixture asserts that the
real `artifacts/final_test_scientific_summary.json` does not exist, but
that file was intentionally committed as the canonical generation-2
output in Phase 2B.8C (commit `4a6b264`). This guard has therefore been
stale since that commit -- confirmed via `git log`, which shows neither
that test file nor this condition changed during Phase 2B.9A. This is
reported here for the record; fixing it is out of scope for this
phase's authorized amendment and was not attempted.

## 8. Scope confirmation

No raw prediction, dataset, or checkpoint file was read at any point in
this phase. No new statistical test or inferential result was computed.
No manuscript prose, novelty claim, or publication/conference-acceptance
assessment was produced.
