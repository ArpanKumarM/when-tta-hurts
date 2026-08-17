# Data and Licensing

**Status: Phase 1 update. Findings below come from directly opening
medmnist.com, the MedMNIST GitHub repo, `on_medmnist_plus.md`, and — new in
this update — the `medmnist` Python package's own shipped `INFO` metadata
(a primary source distributed with the official package itself, inspected
directly via `get_dataset_metadata()` in `src/when_tta_hurts/data.py`, not
downloaded dataset images). The 28px PathMNIST `.npz` was downloaded for
smoke-testing purposes only (git-ignored, checksum-verified — see
`docs/` Phase 1 completion report) and its imagery was not inspected beyond
what the smoke test exercises (shapes/dtype/finiteness).**

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

## Attribution requirements — complete source-dataset citations

MedMNIST requires citing "the corresponding paper(s) of source data if you
use any subset." Full citations, verified against primary sources (or, for
HAM10000's full text, verified against the citation-metadata fields
reachable given a CAPTCHA/login wall on the publisher page — see caveat
below):

**PathMNIST source — NCT-CRC-HE-100K / CRC-VAL-HE-7K:**
Kather, J.N., Halama, N., Marx, A. (2018). *100,000 histological images of
human colorectal cancer and healthy tissue* [Data set]. Zenodo.
https://doi.org/10.5281/zenodo.1214456 (dataset DOI, standard citation for
NCT-CRC-HE-100K/CRC-VAL-HE-7K). Per the `medmnist` package's own
description: *"We resize the source images of 3×224×224 into 3×28×28, and
split NCT-CRC-HE-100K into training and validation set with a ratio of
9:1. The CRC-VAL-HE-7K is treated as the test set."*

**DermaMNIST source — HAM10000:**
Tschandl, P., Rosendahl, C., Kittler, H. (2018). *The HAM10000 dataset, a
large collection of multi-source dermatoscopic images of common pigmented
skin lesions.* Scientific Data, 5, 180161.
https://doi.org/10.1038/sdata.2018.161 —
primary source: https://www.nature.com/articles/sdata2018161 (page could
not be fetched directly in this pass; blocked by a login/CAPTCHA
redirect — see verification caveat below).

**BloodMNIST source — Acevedo et al. peripheral blood cell dataset:**
Acevedo, A., Merino, A., Alférez, S., Molina, Á., Boldú, L., Rodellar, J.
(2020). *A dataset for microscopic peripheral blood cell images for
development of automatic recognition systems.* Mendeley Data, V1.
https://doi.org/10.17632/snkd93bnjr.1 (verified directly — page confirms
17,092 individual normal cell images across 8 classes, "captured from
individuals without infection, hematologic or oncologic disease and free
of any pharmacologic treatment at the moment of blood collection." No
subject count or subject-grouping statement is given on the page.)

**Verification caveat on HAM10000's lesion-multiplicity claim:** the
Nature Scientific Data page for this paper redirected to a login wall and
could not be fetched directly in this pass. The facts recorded below (that
HAM10000 contains multiple images of some lesions, captured at different
magnifications/angles/cameras, and that the released metadata includes a
`lesion_id` field) are recorded per your instruction as verified facts to
document; they are consistent with this paper's well-established public
description but were not independently re-confirmed by this project
reading the primary page's full text in this pass. This should be
re-verified against the primary source directly (or the HAM10000 metadata
CSV itself, which is a more direct check) before this claim is used in any
public report.

## Provenance / leakage investigation

**Correction note:** the PathMNIST classification below was corrected —
"different clinical center" is NOT equated with "verified patient/slide
group separation." Center separation is real and documented, but patient-
and slide-identifier disjointness has not itself been checked against any
identifier data (this project has not obtained per-image patient/slide IDs
for either NCT-CRC-HE-100K/CRC-VAL-HE-7K). DermaMNIST was also
reclassified, from "unverifiable" to "potential lesion leakage," given the
HAM10000 facts recorded above.

Per the requirement not to infer "patient-independent" from a random
image-level split, each dataset is classified below using ONLY what is
explicitly stated in available primary/primary-adjacent sources — no
assumption of leakage-freedom is made where a source is silent.

| Dataset | What is stated | Classification |
|---|---|---|
| **PathMNIST — test vs. train+val** | Per `medmnist` package metadata: CRC-VAL-HE-7K (test) comes from a different clinical center than NCT-CRC-HE-100K (train/val). | **External-center test set: CRC-VAL-HE-7K comes from a different clinical center than NCT-CRC-HE-100K. Patient- and slide-identifier disjointness has not been independently verified from identifiers available to this project.** |
| **PathMNIST — train vs. val** | Described only as "split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1" — a ratio split of patches, no stated slide-grouping guarantee. | **Potential source leakage: the split is described as a ratio split of patches from NCT-CRC-HE-100K, without a verified slide-grouping guarantee.** |
| **BloodMNIST — all splits** | Acevedo et al.'s dataset contains 17,092 individual-cell images "captured from individuals without infection, hematologic or oncologic disease..."; neither the Mendeley Data page nor the `medmnist` package states how many subjects contributed images or whether MedMNIST's 7:1:2 split is subject-grouped. | **Subject-level independence unverifiable: public metadata does not provide this project with verified subject-grouped MedMNIST splits.** |
| **DermaMNIST — all splits** | HAM10000 (Tschandl et al. 2018) is recorded as containing multiple images of some of the same lesions, captured at different magnifications, angles, or with different cameras, with lesion identifiers (`lesion_id`) present in the source metadata (see verification caveat above). MedMNIST's own description states only a 7:1:2 split ratio and does not document lesion-grouped splitting. | **Potential lesion leakage: HAM10000 contains multiple images of some lesions under a shared lesion identifier, and MedMNIST does not document whether its 7:1:2 split groups by lesion — so the same lesion's images may appear across train/val/test.** |

**What generalization claims the split structure permits (restrained
interpretation):** For PathMNIST, test performance can be described as
reflecting an external-center evaluation — a meaningfully different data
source from train/val — but NOT as "patient-independent" or
"slide-independent," since that has not been verified from actual
identifiers. For BloodMNIST, test performance should be described as "held
-out accuracy under the official MedMNIST split" only. For DermaMNIST,
test performance should carry the same caveat plus an explicit note of
potential lesion-level leakage inflating apparent accuracy, since the same
lesion's images could appear in both train and test. No dataset in this
project should be described as "patient-independent" or
"subject-independent" without identifier-level verification this project
has not performed.

## Dataset-validity risk: low-level biases in NCT-CRC-HE (PathMNIST source)

**A preprint** (not established fact, not peer-reviewed — flagged
explicitly as such): Ignatov, A., Malivenko, G. (2024). *"NCT-CRC-HE: Not
All Histopathological Datasets Are Equally Useful."* arXiv:2409.11546,
submitted 17 Sep 2024, CC BY-NC-SA 4.0.
https://arxiv.org/abs/2409.11546

This preprint reports that NCT-CRC-HE-100K — PathMNIST's source dataset —
has inappropriate color normalization, severe class-inconsistent JPEG
artifacts, and some completely corrupted tissue samples. It reports that a
model using only the 3 raw RGB channel-mean features per image reaches over
50% accuracy on the 9-class task, and that color-histogram features alone
reach over 82% accuracy — and that a standard ImageNet-pretrained
EfficientNet-B0 reaches over 97.7% accuracy, exceeding prior specialized
histopathology models. This suggests classification success on this
dataset can be substantially explained by low-level color/artifact
statistics rather than tissue morphology.

**Relevance to this project (documented risk, not a hypothesis change):**
if PathMNIST classification is partly driven by such low-level, non
-morphological signals, then TTA's geometric and intensity transforms could
be disrupting those specific low-level statistics (color histograms, JPEG
artifact patterns) rather than — or in addition to — disrupting BatchNorm
statistics or genuine morphological feature extraction. This is a plausible
confound for interpreting PathMNIST's large reported TTA degradation
(H1/H2 in `docs/research_plan.md`): a TTA failure driven by color-jitter
disrupting a color-histogram shortcut is a different phenomenon from a TTA
failure driven by BatchNorm running-statistics mismatch, even though both
would show up as "TTA hurts PathMNIST accuracy." **The pre-registered
hypotheses (H1-H4) are not changed by this.** This is recorded as an
interpretive risk to flag explicitly when PathMNIST results are reported —
any causal claim about *why* TTA hurts PathMNIST should note this
alternative/contributing explanation and, where feasible, report whether
intensity-only vs. geometric-only TTA policies degrade PathMNIST
differently (the source paper's own finding that "intensity-only
augmentations outperform geometric transforms" is worth re-examining in
this light, though that is exploratory, not a new hypothesis).

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

## Official split sizes (verified twice: doc pages, then programmatically)

| Dataset | Train | Validation | Test |
|---|---|---|---|
| PathMNIST | 89,996 | 10,004 | 7,180 |
| BloodMNIST | 11,959 | 1,712 | 3,421 |
| DermaMNIST | 7,007 | 1,003 | 2,005 |

**Split-count verification (corrected — precise per-dataset wording):**

- **PathMNIST: split counts verified from the downloaded 28px artifact.**
  `data/raw/pathmnist.npz` (209MB, MD5-checksum-matched — see below) was
  downloaded for smoke testing, and its `train_images`/`val_images`/
  `test_images` array shapes were read directly and match
  `EXPECTED_SPLITS` exactly (see
  `src/when_tta_hurts/data.py::verify_split_counts_from_artifact()` and
  `tests/test_data_artifact.py`).
- **BloodMNIST: split counts verified against `medmnist` 3.0.2 `INFO`
  metadata; image artifact not downloaded.** No BloodMNIST `.npz` file has
  been fetched in this project. The counts above come only from the
  package's shipped metadata dict (see
  `src/when_tta_hurts/data.py::verify_split_counts()`), not from inspecting
  actual image data.
- **DermaMNIST: split counts verified against `medmnist` 3.0.2 `INFO`
  metadata; image artifact not downloaded.** Same caveat as BloodMNIST —
  metadata-only verification.

This distinction matters: metadata could in principle be stale or wrong
even if consistently repeated across doc pages and the package, whereas the
PathMNIST artifact check reads the actual array shapes. Do not describe
BloodMNIST or DermaMNIST as having been "empirically verified" or
"inspected" — only their metadata has been checked.

## Open questions — status after Phase 1 investigation

- **Patient/source-level leakage — investigated, not fully resolved.**
  See the "Provenance / leakage investigation" section above: PathMNIST's
  test split is verified center-disjoint from train/val; PathMNIST
  train-vs-val, BloodMNIST (all splits), and DermaMNIST (all splits) are
  classified as potential-leakage or unverifiable, not confirmed
  leakage-free. Any generalization claim in a future report must respect
  this per-dataset asymmetry.
- **Original source-dataset citations — dataset names resolved, full
  citations still pending.** NCT-CRC-HE-100K/CRC-VAL-HE-7K (PathMNIST) and
  HAM10000 (DermaMNIST) are now identified by name from the package's own
  metadata; exact BibTeX-form citations for these upstream papers still
  need to be added before any public release.

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
