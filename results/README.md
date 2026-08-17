# results/

## Phase 0 status

**No experiments have been run. No results exist.** Any numbers, plots, or
tables referencing this project outside of this repository (e.g. on
LinkedIn, in a README, in a report) before Phase 1+ artifacts exist here are
unauthorized and must not be produced or shared — see `CLAUDE.md`.

## Planned contents (Phase 1+, not yet created)

- `results/runs/<run_id>/` — one directory per training/eval run, containing:
  - `config.yaml` (frozen copy of the config used)
  - `env.json` (captured hardware/software environment)
  - `predictions_val.parquet`, `predictions_test.parquet` — raw per-sample
    predictions (never just aggregate metrics)
  - `metrics.json`
- `results/ledger.csv` — **append-only** experiment ledger. Every run,
  including failed ones, gets a row. Rows are never deleted or edited after
  the fact (see `CLAUDE.md` — "no deleting failed runs from the ledger").
- `results/figures/` — plots regenerated from raw result files only, never
  hand-edited or manually typed into documents.

All headline metrics reported anywhere in this repo (README, docs, paper/)
must be generated from files in this directory via a documented script —
never typed in by hand.
