# Data and Licensing

**Status: Phase 0, correction pass applied. Findings below come from
directly opening medmnist.com, the MedMNIST GitHub repo, and
`on_medmnist_plus.md` (see `docs/literature_review.md` for full citation
details). No data has been downloaded.**

The repository must not redistribute MedMNIST data. Datasets are loaded at
runtime via the official `medmnist` package in Phase 1+; nothing under
`data/` is ever committed except checksums/manifests (see `data/README.md`
and `.gitignore`). DermaMNIST is CC BY-NC 4.0 — its downloaded data,
generated caches, and per-sample images must stay out of git and out of any
commercial use, and any result or figure derived from it must be labeled
non-commercial.

## Dataset license terms (verified)

| Dataset | License | Notes |
|---|---|---|
| PathMNIST | CC BY 4.0 | Attribution required; commercial use permitted. |
| BloodMNIST | CC BY 4.0 | Attribution required; commercial use permitted. |
| DermaMNIST | **CC BY-NC 4.0** | **Non-commercial only.** This differs from the other two datasets and must be respected — no commercial use of DermaMNIST-derived results, models trained on it, or figures built from it. |
| `medmnist` package (code) | Apache-2.0 | Covers the loader/utility code, not the data itself. |

Source: medmnist.com and github.com/MedMNIST/MedMNIST, fetched directly
during Phase 0 (see `docs/literature_review.md` §2).

## Attribution requirements

MedMNIST requires citing "the corresponding paper(s) of source data if you
use any subset" — i.e., citation of both the MedMNIST v2/MedMNIST+ paper(s)
*and* the original source dataset each subset is derived from. **The exact
original source dataset names/citations for PathMNIST, BloodMNIST, and
DermaMNIST were not found on the pages fetched in Phase 0 and are
UNVERIFIED.** MedMNIST provides a per-subset CSV mapping images to source
datasets; obtaining and recording these citations is a Phase 1 task and
must be completed before any public release or figure using these datasets.

## Non-clinical-use disclaimer (verified, quoted exactly)

> "Please note that this dataset is NOT intended for clinical use."

This must be repeated, verbatim or in equivalent strength, everywhere this
project's results are described (README, report, any public post). See
`CLAUDE.md` rule 7 and `docs/claims_and_risks.md`.

## Resolution provenance (CORRECTED — was previously flagged as a likely confound)

Verified directly against `github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md`:

| Dataset | Source resolution | Construction of 64/128px |
|---|---|---|
| PathMNIST | 3×224×224 | Resized directly from the 224×224 original, no crop. |
| BloodMNIST | 3×360×363 | Center-cropped to 3×200×200, then resized to target size. |
| DermaMNIST | 3×600×450 | Resized directly from the 600×450 original, no crop. |

Sample indices and train/val/test splits are preserved across all
resolutions ("The data in MedMNIST+ directly corresponds to that of
MedMNIST, maintaining the same dataset splits... and sample indices").
**The earlier concern that higher resolutions are upsampled from the 28px
files is retracted — it is false for these three datasets.** H2 evaluates
genuinely retained source-image information at each standardized
resolution. See `docs/research_plan.md`'s corrected H2 wording.

## Official split sizes (verified)

| Dataset | Train | Validation | Test |
|---|---|---|---|
| PathMNIST | 89,996 | 10,004 | 7,180 |
| BloodMNIST | 11,959 | 1,712 | 3,421 |
| DermaMNIST | 7,007 | 1,003 | 2,005 |

## Open questions (UNVERIFIED — must resolve before Phase 1 download)

- **Patient/source-level leakage:** no explicit statement was found
  confirming splits are leakage-free at the patient or source level. Do not
  assume independence between train/val/test at the patient level until
  this is checked against the MedMNIST Scientific Data (2023) paper.
- **Original source-dataset citations:** exact upstream dataset names for
  PathMNIST/BloodMNIST/DermaMNIST (needed for attribution) — not yet
  retrieved from MedMNIST's per-subset provenance CSVs.

## Target preprint license and code status (verified)

The target preprint (arXiv:2604.09697) is under **CC BY 4.0**, per its
arXiv abstract page. This permits reuse with attribution, consistent with
reproducing and extending its reported experiments.

**Code:** the paper provides a code URL in Appendix A —
`https://github.com/danielxmed/AI-Scientist-v3` — but the linked repository
was unavailable during verification (HTTP 404). This must be stated
precisely as such; it is not accurate to say "the paper provides no code
link." Our implementation is treated as independent of the source unless
and until that repository becomes available under a compatible license —
per `CLAUDE.md` rule 8, no code will be copied from it even if it
reappears, without first verifying its license.

## Third-party code — do not copy without verification

Per `CLAUDE.md` rule 8, no third-party source code is to be copied into this
repository. Status of code repos identified during the literature review
(`docs/literature_review.md`):

| Repo | License found? | Action |
|---|---|---|
| kakaobrain/learning-loss-for-tta | No LICENSE file found in the listing fetched — `UNVERIFIED`/likely unlicensed | Do not copy. Implement any comparable mechanism independently from the published method description if needed. |
| Z-Sherkat/BayTTA | Not checked in Phase 0 | `UNVERIFIED` — check before any reference use. |
| francescodisalvo05/medmnistc-api | License not stated on the page fetched | `UNVERIFIED` — check before any reference use, e.g. if borrowing corruption-transform definitions. |

Default posture: implement all pipeline code (dataloaders, augmentation
policies, training loops, TTA gating) independently from published method
descriptions, using only the official `medmnist` (Apache-2.0) package for
data loading.

## Repository license for original work

This repository's own code/text is MIT-licensed (see `LICENSE`), which
covers only what is authored here — it does not relicense MedMNIST data,
the target preprint, or any cited paper. See the note at the bottom of
`LICENSE`.
