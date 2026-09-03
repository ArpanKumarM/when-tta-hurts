# Citation Audit — Phase 2B.9B

This document records, for every entry in `paper/references.bib`, the
exact primary source fetched and read in this phase (not a search
snippet, blog, or AI-generated summary), what was verified, and any
gap. It supersedes nothing in `docs/literature_review.md` (Phase 0) --
it re-verifies that document's entries against the live sources before
they are used in `paper/manuscript.md`, since Phase 0's fetches
happened in an earlier session and pages can change.

Per `CLAUDE.md` rule 2, every citation used in the manuscript traces to
a primary source actually opened and read in this phase. No entry below
was accepted from a search-result snippet alone.

| BibTeX key | Primary source fetched | What was verified | Status |
|---|---|---|---|
| `medeiros2026tta` | `https://arxiv.org/abs/2604.09697` (fetched directly) | Exact title, sole author, submission date (6 Apr 2026), CC BY 4.0 license, abstract text confirming the 31.6pp maximum drop and the distribution-shift/BatchNorm causal claim. | Verified |
| `shanmugam2021better` | `https://arxiv.org/abs/2011.11156` (fetched directly) | Exact title, all four authors, version history, CC BY 4.0 license, ICCV 2021 peer-reviewed venue, arXiv DOI. | Verified |
| `lyzhov2020greedy` | `https://arxiv.org/abs/2002.09103` and `https://proceedings.mlr.press/v124/lyzhov20a.html` (both fetched directly) | Exact title confirmed on both; author order differs slightly between the arXiv listing (Molchanov first) and the official PMLR proceedings page (Lyzhov first) -- the PMLR proceedings page is the authoritative peer-reviewed record and is used for author order in `references.bib`. PMLR volume 124, pages 1308-1317, UAI 2020, editors Peters & Sontag confirmed directly from the official proceedings page, including its self-published BibTeX block. | Verified (author order resolved via the higher-authority proceedings record) |
| `kim2020learning` | `https://arxiv.org/abs/2010.11422` (fetched directly) | Exact title, three authors, submission date, NeurIPS 2020 acceptance note, arXiv DOI. | Verified |
| `sherkatghanad2024baytta` | `https://arxiv.org/abs/2406.17640` (fetched directly) | Exact title, all five authors, version history (v1 25 Jun 2024, v2 27 Aug 2024), CC0 1.0 license. | Verified |
| `disalvo2024medmnistc` | `https://arxiv.org/abs/2406.17536` (fetched directly) | Exact title, all three authors, version history (v1-v3), CC BY 4.0 license. | Verified |
| `schneider2020improving` | `https://arxiv.org/abs/2006.16971` (fetched directly) | Exact title, all six authors, version history, NeurIPS 2020 acceptance (page states "the Thirty-fourth Conference on Neural Information Processing Systems," the trademarked name "NeurIPS 2020" itself does not appear verbatim on the abstract page but the ordinal unambiguously identifies it). | Verified |
| `wu2018group` | `https://arxiv.org/abs/1803.08494` (fetched directly, fully verified: title, two authors, version history v1-v3); `https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html` and `https://dl.acm.org/doi/10.1007/978-3-030-01261-8_1` (both attempted, both returned HTTP 403 and could not be opened) | The arXiv preprint identity (title, authors, versions, license) is fully verified by direct fetch. The ECCV 2018 venue and the Springer DOI `10.1007/978-3-030-01261-8_1` were corroborated only via a `WebSearch` result set (surfacing the ACM DL and ResearchGate index listings) -- the actual publisher/proceedings pages could not be opened (403 Forbidden) in this session. | **Partially verified**: arXiv identity fully verified by direct fetch; ECCV 2018 venue/DOI is corroborated by independent index listings but not confirmed by directly opening the publisher's own page. The manuscript and `references.bib` cite this as an ECCV 2018 paper with the caveat recorded here; if a stricter standard is required, cite as the arXiv preprint only. |
| `yang2023medmnistv2` | `https://arxiv.org/abs/2110.14795` and `https://medmnist.com/` (both fetched directly) | Exact title, all eight authors, version history, CC BY 4.0 license (dataset code), Scientific Data 2023 publication and DOI `10.1038/s41597-022-01721-8` confirmed on the arXiv page; the official MedMNIST site independently confirmed the Scientific Data citation, the verbatim clinical-use disclaimer ("This dataset is NOT intended for clinical use"), and per-dataset licenses (PathMNIST CC BY 4.0, BloodMNIST CC BY 4.0, DermaMNIST CC BY-NC 4.0). | Verified |
| `yang2021medmnist` | `https://arxiv.org/abs/2010.14925` (fetched directly) | Exact title, all three authors, version history (v1-v4), CC BY 4.0 license, ISBI 2021 presentation venue. | Verified |
| `wang2021tent` | `https://arxiv.org/abs/2006.10726` (fetched directly) | Exact title, all five authors (Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell), submission June 2020 with final version March 2021, ICLR 2021 Spotlight venue stated on the page, method summary (online entropy minimization adapting BatchNorm statistics and channel-wise affine parameters). | Verified |
| `ayhan2018ttaug` | `https://github.com/berenslab/ttaug-midl2018` (the authors' own code repository, fetched directly) after `https://openreview.net/forum?id=rJZz-knjz` returned only a bot-check page and could not be opened -- disclosed exactly as for `wu2018group`'s blocked publisher pages | Exact title, both authors (Murat Seckin Ayhan, Philipp Berens), MIDL 2018 venue and year, and the authors' self-provided BibTeX block; purpose (TTA used to estimate heteroscedastic aleatoric uncertainty via prediction variability across augmented views). | **Partially verified**: bibliographic identity confirmed via the authors' own repository and its self-published BibTeX; the OpenReview proceedings page itself could not be opened (bot-check) in this session. |
| `kimura2024understanding` | `https://arxiv.org/abs/2402.06892` (fetched directly) | Exact title, sole author (Masanari Kimura), 2024, arXiv cs.LG identifier. The abstract states the paper gives theoretical guarantees for TTA and clarifies its behavior; the precise "ambiguity term" formulation is described in the manuscript only at the level the abstract supports, not quoted beyond it. | Verified (arXiv identity); theoretical-claim detail cited conservatively |

## Additional primary-source verification not yielding a separate BibTeX entry

* `https://github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md`
  (fetched directly) -- re-confirms the exact MedMNIST+ resolution
  construction used in `paper/manuscript.md`'s Methods section:
  PathMNIST 64/128px resized from independently-sourced 224x224
  originals; DermaMNIST 64/128px resized from 600x450 originals (no
  crop); BloodMNIST 64/128px center-cropped from 360x363 to 200x200
  then resized; splits and sample indices preserved across
  resolutions. This is project documentation, not a separately citable
  publication, and is referenced in the manuscript by its GitHub URL,
  not as a BibTeX entry.

## Sources considered but not cited

No blog post, marketing page, search-result summary, or AI-generated
summary was used as a citation source anywhere in the manuscript. Where
a `WebSearch` result surfaced a candidate source, the corresponding
primary page was fetched directly before any fact from it was used
(the sole exception being the two blocked ECCV/ACM pages noted above,
disclosed rather than silently treated as verified).

## Scope note

This audit covers the sources cited in `paper/manuscript.md`. It is not
an exhaustive systematic literature review of test-time augmentation or
medical image classification; `docs/literature_review.md` sec.10
already discloses this as a known, time-boxed scope limitation, and
`docs/phase2b_manuscript_claims_and_structure_freeze.md` sec.5 (Novelty
audit) states every novelty claim with the same non-exhaustive-search
caveat.
