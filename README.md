# When TTA Hurts: A Causal Audit of Test-Time Augmentation in Medical Imaging

**Status: Planning / pre-registration stage (Phase 0). No experiments have
been run. No results exist. Nothing in this repository should be treated as
a finding.**

## What this is

This project reproduces and extends a specific empirical claim from a
recent preprint:

> *"I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts
> Medical Image Classification"* — arXiv:2604.09697 (April 2026, preprint —
> **not peer-reviewed**; see `docs/literature_review.md`).

That paper reports that test-time augmentation (TTA) *reduced* accuracy in
11 of 12 model-dataset combinations tested on MedMNIST datasets. This
project investigates *why*, and whether a simple validation-gated fallback
mechanism can reduce the harm.

This project does **not** claim to have invented test-time augmentation,
selective/adaptive TTA, or validation-based TTA gating — see
`docs/claims_and_risks.md` for an explicit novelty positioning against
prior work (Shanmugam et al. 2021, Lyzhov et al. 2020, learned-loss TTA,
BayTTA, and others).

## What this is not

- **Not clinically validated.** This project uses [MedMNIST](https://medmnist.com/),
  which is explicitly not intended for clinical use. Nothing produced by
  this project should be interpreted as diagnostic, clinically validated,
  or deployment-ready.
- **Not a claim of novelty for TTA or adaptive TTA itself.** See
  `docs/claims_and_risks.md`.
- **Not associated with, or a critique targeted at, the authors of the
  source preprint.** This is an independent reproduction and extension
  effort, in the normal spirit of scientific replication.

## Pre-registered hypotheses

Four hypotheses (H1–H4) covering normalization, resolution, augmentation
policy matching, and validation-gated TTA are pre-registered in
`docs/research_plan.md`, along with a critical evaluation of each
hypothesis's wording and limitations.

## Repository structure

```
docs/     research plan, literature review, protocol, statistics, compute
          budget, claims table, data/licensing notes
configs/  machine-readable experiment configuration (draft, unapproved)
data/     dataset placeholder — data is downloaded at runtime, git-ignored
results/  results placeholder — no results exist yet
paper/    placeholder for an eventual 4-6 page technical report
src/      when_tta_hurts package: config, devices, reproducibility, data,
          models, transforms, evaluation, artifacts (Phase 1)
tests/    offline unit tests (no dataset download required)
scripts/  smoke_test.py, verify_reproducibility.py
CLAUDE.md rules of engagement for AI-assisted work in this repo
```

## Reproducibility commitments

- **Implemented (Phase 1):** fixed seeds (`reproducibility.py`),
  machine-readable configs + content hashing (`config.py`), captured
  environment/hardware manifest (`devices.py`), dataset checksum
  verification (`data.py`), an engineering smoke test
  (`scripts/smoke_test.py`), and same-device reproducibility verification
  (`scripts/verify_reproducibility.py`).
- **Planned (Phase 2+):** append-only run ledger with failed runs never
  deleted (`artifacts.py` has the primitive; not yet wired into a real run
  loop), raw per-sample predictions saved, all headline metrics generated
  by scripts reading `results/` — never hand-typed, a strict
  train/validation-vs-test firewall (`docs/experimental_protocol.md`), and
  a single-command small reproduction / final approved experiment matrix.

## Current status

- **Phase 0 (preregistration) is frozen** in commit `a22db01` ("docs:
  preregister TTA causality study"). Hypotheses, protocol, and claims
  discipline are locked as of that snapshot; later changes are amendments,
  not silent rewrites — see `CLAUDE.md`.
- **Phase 1 (engineering infrastructure) is implemented**: a `uv`-managed,
  src-layout Python package (`src/when_tta_hurts/`) with device selection,
  reproducibility utilities, dataset loading via the official `medmnist`
  package, model and transform building blocks, a cache-key primitive, and
  an engineering smoke test — see `docs/research_plan.md` for phase
  definitions and the Phase 1 completion report for what was verified.
- **No pilot or research experiment has run.** No research results exist
  yet — the smoke test explicitly prints "Engineering smoke test — not an
  experimental result" and reports no accuracy.
- This project remains **not intended for clinical use** (see
  `docs/data_and_licensing.md`).

## License

Original code and text in this repository are MIT-licensed (see `LICENSE`).
This does **not** extend to the MedMNIST dataset or the cited papers, which
retain their own license/copyright status — see `docs/data_and_licensing.md`.
