# data/

No datasets are stored in this repository or committed to version control.

## Phase 0 status

**No data has been downloaded.** This directory currently contains only this
README. Phase 0 is planning-only (see `docs/research_plan.md`).

## Planned contents (Phase 1+, not yet created)

- `data/raw/` — untouched official MedMNIST `.npz` downloads, one per
  dataset/resolution. Git-ignored.
- `data/processed/` — any derived splits or cached tensors. Git-ignored.
- `data/checksums.json` — SHA256 checksums of every downloaded file, recorded
  at download time, so results can be tied to an exact dataset version.

## Source

All datasets will come from the official MedMNIST v2/MedMNIST+ release:
https://medmnist.com/ (see `docs/data_and_licensing.md` for verified license
terms, splits, and the official non-clinical-use disclaimer once confirmed).

No private, PHI-containing, or non-public data will be used at any phase of
this project.
