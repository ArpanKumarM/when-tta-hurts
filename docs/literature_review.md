# Literature Review

**Status: Phase 0. All entries below were verified by directly opening the
primary source page (arXiv abstract pages, ICCV/UAI/NeurIPS proceedings
pages, or the linked GitHub repo), not from search-result snippets. Where a
fact could not be confirmed from the page actually opened, it is marked
`UNVERIFIED`. This document must be re-checked before any novelty claim is
finalized (see `docs/claims_and_risks.md`).**

## 1. Target preprint

**I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts
Medical Image Classification**

**Correction note:** This entry was rewritten after reading the complete
HTML paper (`arxiv.org/html/2604.09697v1`), including both appendices, not
just the abstract. All facts below are now sourced to the full text.

| Field | Value |
|---|---|
| Author | Daniel Nobrega Medeiros |
| Venue / status | **Preprint, arXiv:2604.09697, v1, submitted 6 Apr 2026.** No peer-review venue found. Treat as unreviewed. |
| License | CC BY 4.0 (per arXiv abs page) |
| **Code** | **The paper provides a code URL in Appendix A: `https://github.com/danielxmed/AI-Scientist-v3`. The linked repository was unavailable during verification (HTTP 404).** This is a corrected statement — it must not be reported as "the paper provides no code link." Our implementation is treated as independent of the source unless/until that repository becomes available under a compatible license. |
| Research question | Whether standard TTA pipelines help or hurt medical image classification accuracy, and why. |
| Datasets | PathMNIST, BloodMNIST, DermaMNIST (per Table 2 — the three datasets this project also uses). |
| Models tested | LogReg (~21K/16K/19K params depending on dataset), MLP (~670K), SmallCNN (~95K, 3 conv layers + BatchNorm + MaxPool), ResNet-18 (~11M, adapted for 28×28, no initial pooling layer). |
| Training setup | Adam (lr=1e-3), cosine annealing, 25-30 epochs, early stopping on validation accuracy, batch size 256 (Appendix A). |
| TTA setup | View counts {1,2,5,10,25,50,100}; policies geometric (flips, ±15° rotation, random resized crop 0.8-1.0×), intensity (brightness/contrast jitter ±0.3, Gaussian blur), mixed (geometric+intensity); aggregation via mean probability, majority vote, or confidence-weighted average. |
| **Verified headline result (Table 2, mixed policy, mean aggregation, N=50)** | TTA degraded accuracy in **11 of 12** model-dataset combinations. Sole improvement: **ResNet-18 on DermaMNIST, +1.6pp** (74.2%→75.8%). Largest drops: **ResNet-18/PathMNIST −31.6pp** (87.0%→55.4%), **SmallCNN/PathMNIST −28.6pp** (88.1%→59.6%), **SmallCNN/BloodMNIST −14.8pp** (94.5%→79.8%). Full 12-row table is in `docs/research_plan.md`'s reproduction target and should be re-derived from `results/` once reproduced, not hand-copied into any report. |
| Stated root cause (their claim) | Distribution shift between augmented inputs and training data, compounded by BatchNorm mismatches — **the source paper itself already names BatchNorm as a contributing factor**, directly overlapping with our H1. Our contribution, if any, is a controlled ablation isolating this rather than an observational claim (see H1's corrected wording in `docs/research_plan.md`). |
| **Appendix B — already-evaluated mitigations (must NOT be presented as our contribution)** | Three conditions compared: (1) augmented views only, (2) original image + augmented views, (3) BN adaptation + original + augmented views. **Clean-image anchoring**: reduced SmallCNN/PathMNIST's drop from −37.0% to −8.6%, and ResNet-18/BloodMNIST's from −21.1% to −2.6%; benefit diminishes at N≥10 as augmented views dominate the average. **Basic BN adaptation**: helped on BloodMNIST (ResNet-18: −9.2% augmentation-only → −7.2% with clean anchor → −1.3% with BN adaptation, all at N=10) but *hurt* on PathMNIST ("the augmented distribution is too far from the training distribution for reliable statistics estimation"). These are required baselines to reproduce (conditions 3-4 in `docs/experimental_protocol.md`), not novel mechanisms of this project. |
| Relationship to our work | Direct reproduction target; both of our proposed H4 baselines (clean anchoring, BN adaptation) are already in their appendix. |

## 2. MedMNIST v2 / MedMNIST+

**Correction note:** the resolution-construction claim below was rewritten
after directly reading
`github.com/MedMNIST/MedMNIST/blob/main/on_medmnist_plus.md`. The prior
version's "likely upsampled from 28px" concern is **incorrect and
retracted** — see corrected row below.

| Field | Value |
|---|---|
| Site / repo | https://medmnist.com/ , https://github.com/MedMNIST/MedMNIST |
| Data license | **CC BY 4.0 for PathMNIST and BloodMNIST. DermaMNIST is CC BY-NC 4.0 (non-commercial).** This asymmetry is important: DermaMNIST-derived results/figures cannot be used in any commercial context and must be labeled non-commercial wherever used (see `docs/data_and_licensing.md`). |
| Code license | Apache-2.0 (the `medmnist` package itself) |
| Peer-reviewed reference | Nature Scientific Data (2023) and ISBI (2021) — `UNVERIFIED` in detail (exact citation not opened directly; site states these venues). |
| Clinical use disclaimer | Confirmed, quoted verbatim: **"Please note that this dataset is NOT intended for clinical use."** |
| **Resolutions — CORRECTED** | 28×28 standard; MedMNIST+ adds 64×64, 128×128, 224×224. Per `on_medmnist_plus.md` (verified directly): **PathMNIST** 64/128px are resized directly from independently-sourced **224×224** originals (no crop). **DermaMNIST** 64/128px are resized directly from **600×450** originals (no crop). **BloodMNIST** 64/128px are constructed by center-cropping **360×363** originals to **200×200**, then resizing to target size. Sample indices and train/val/test splits are preserved across resolutions ("The data in MedMNIST+ directly corresponds to that of MedMNIST, maintaining the same dataset splits... and sample indices"). **H2 therefore measures retained source-image information at each standardized resolution, not 28px-interpolation artifacts** — the resolution confound previously flagged in `docs/research_plan.md` does not apply. |
| **Splits — CORRECTED (verified)** | PathMNIST: 89,996 train / 10,004 val / 7,180 test. BloodMNIST: 11,959 / 1,712 / 3,421. DermaMNIST: 7,007 / 1,003 / 2,005. |
| Source-dataset provenance | PathMNIST derives from 224×224 histopathology source images; BloodMNIST from 360×363 blood-cell images; DermaMNIST from 600×450 dermatoscopy images (resolutions confirmed via `on_medmnist_plus.md`). **Exact original dataset names/citations (e.g. the specific histopathology/blood-cell/dermatoscopy dataset each is drawn from) remain `UNVERIFIED`** — not found on the pages fetched; needed for `docs/data_and_licensing.md` attribution before any public release. |
| Patient/source-level leakage | **`UNVERIFIED`** — no explicit statement found on the pages fetched. Must check the Scientific Data paper's methodology section in Phase 1; do not assume splits are leakage-free. |

## 3. Better Aggregation in Test-Time Augmentation (Shanmugam et al.)

| Field | Value |
|---|---|
| Citation | Shanmugam, Blalock, Balakrishnan, Guttag. ICCV 2021, pp. 1214–1223. Peer-reviewed (ICCV). Also on arXiv:2011.11156 (CC BY 4.0). |
| Contribution | Shows naive averaging in TTA can flip many correct predictions to incorrect even when net accuracy improves (this is directly the "harm rate" concept underlying our H4 secondary endpoint — **not a novel framing on our part**). Proposes a **learned aggregation weighting method** as the fix. |
| Per-sample rejection + clean fallback? | Not confirmed as their mechanism — their fix is learned weighting, not documented as instance-level accept/reject-with-fallback. `UNVERIFIED` in detail; the full paper (not just abstract) should be read in Phase 1 before finalizing the novelty claim, since "harm rate" as an evaluated quantity is clearly prior art either way. |
| Code | Not found linked on the abstract page. `UNVERIFIED`. |
| Novelty impact on us | **High.** The "clean-correct → wrong" harm framing we propose as a secondary endpoint is essentially their framing already. We must cite this explicitly as the origin of that framing, not present it as ours. |

## 4. Greedy Policy Search (Lyzhov et al.)

| Field | Value |
|---|---|
| Citation | Lyzhov, Molchanova, Ashukha, Molchanov, Vetrov. UAI 2020, PMLR v124. Peer-reviewed (UAI). |
| Contribution | GPS: a simple, non-learned-network baseline that greedily searches over a policy of test-time augmentations to maximize validation performance — **this is validation-driven TTA policy selection**, directly adjacent to our proposed transform-level filtering step in H4. |
| Datasets/models | `UNVERIFIED` in detail from the fetched page (not listed). |
| Code | Not found linked on the fetched page. `UNVERIFIED`. |
| Novelty impact on us | **High.** Our step 2 (reject transforms that hurt validation accuracy) is conceptually a simplified/restricted case of greedy policy search. We should cite GPS explicitly and frame our transform-filtering step as "a minimal instance of the general GPS idea," not as new. |

## 5. Learning Loss for Test-Time Augmentation (Kim, Kim, Kim)

| Field | Value |
|---|---|
| Citation | Kim, Kim, Kim (Kakao Brain). NeurIPS 2020. arXiv:2010.11422 (nonexclusive distribution license). Peer-reviewed (NeurIPS). |
| Contribution | Learns an auxiliary module to predict per-transform loss on a given test input, then applies only the low-predicted-loss transformations before averaging — **this is exactly per-sample, per-transform selective TTA**, i.e., instance-level view rejection. |
| Code | https://github.com/kakaobrain/learning-loss-for-tta — fetched; **no LICENSE file found in the repo listing** (`UNVERIFIED`/likely unlicensed — do not copy code from this repo without further checking, per `CLAUDE.md` rule 8). |
| Clean-prediction fallback? | Not documented in the README excerpt fetched; `UNVERIFIED` — would require reading `eval_l2t.py` directly. |
| Novelty impact on us | **Very high.** This paper already does per-sample, per-transform selective TTA via a learned predictor. Our H4's "per-sample rejection" idea is not new in kind — our claimed difference is only that we use a **non-learned, validation-calibrated divergence threshold** instead of a trained auxiliary network, which is a simplicity/interpretability trade-off, not a new capability. This must be stated plainly in `docs/claims_and_risks.md`. |

## 6. BayTTA

| Field | Value |
|---|---|
| Citation | Sherkatghanad, Abdar, Bakhtyari, Plawiak, Makarenkov. arXiv:2406.17640, v1 25 Jun 2024, v2 27 Aug 2024. CC0 1.0 (public domain dedication). Peer-review status of the underlying venue `UNVERIFIED` (only arXiv page checked). |
| Contribution | Bayesian Model Averaging to combine TTA-view predictions weighted by posterior probability, evaluated **specifically on medical imaging** (skin cancer, breast cancer, chest X-ray) plus gene-editing data, using VGG-16/MobileNetV2/DenseNet201/ResNet152V2/InceptionResNetV2. |
| Code | https://github.com/Z-Sherkat/BayTTA (linked directly from the paper). |
| Novelty impact on us | **High and directly on-topic.** This is prior art for "TTA aggregation method evaluated specifically in medical imaging," which is close to our setting. It does not appear to do a BatchNorm/resolution causal ablation (not evaluated here in detail — `UNVERIFIED`, check full text in Phase 1), which is our main claimed differentiator; if BayTTA or similar already contains that ablation, our contribution shrinks further. |

## 7. MedMNIST-C

| Field | Value |
|---|---|
| Citation | Di Salvo, Doerrich, Ledig. arXiv:2406.17536v3, 23 Jul 2024. CC BY 4.0. |
| Contribution | A **corruption-robustness benchmark** (not a TTA-aggregation method) built on MedMNIST+, covering 12 datasets/9 modalities, with AlexNet/ResNet50/DenseNet121/ViT-B/16/VGG16. Shows their corruption augmentations improve robustness **as training-time data augmentation**, which is a different use of "augmentation" than test-time aggregation. |
| Code | github.com/francescodisalvo05/medmnistc-api (license not stated on the page fetched — `UNVERIFIED`). |
| Relevance to us | Useful as a source of a validated corruption/augmentation taxonomy for MedMNIST specifically (could inform our candidate transform pool), and as evidence the "MedMNIST + robustness" space is active, but it is not directly about TTA failure or gating — lower novelty overlap than sources 3–6. |

## 8. BatchNorm under distribution shift

**Improving robustness against common corruptions by covariate shift
adaptation** — Schneider, Rusak, Eck, Bringmann, Brendel, Bethge. NeurIPS
2020 (submitted 30 Jun 2020). Nonexclusive arXiv distribution license.
Proposes recomputing/adapting BatchNorm activation statistics at test time
using (unlabeled) test-distribution statistics instead of training
statistics, showing large corruption-robustness gains on ImageNet-C. This is
directly relevant background for H1: it supports the general premise that
BatchNorm's running statistics are a plausible mechanism for
distribution-shift sensitivity, and suggests a concrete alternative
condition worth testing (recomputed/adapted BatchNorm stats at test time)
beyond just BatchNorm-vs-GroupNorm. Should be cited in `research_plan.md`'s
H1 discussion.

## 9. Group Normalization

**Group Normalization** — Wu & He, arXiv:1803.08494 (March 2018; venue
line on ECCV 2018 not independently confirmed from the arXiv page —
`UNVERIFIED` on venue specifically, though widely known to be ECCV 2018).
Nonexclusive arXiv distribution license. Establishes GroupNorm as
batch-size-independent, splitting channels into groups for per-group
normalization instead of relying on batch (or running) statistics. This is
the mechanistic basis for treating GroupNorm as a "no running statistics"
comparison point in H1 — but as noted in `research_plan.md`, GroupNorm
differs from BatchNorm in more than just the presence of running
statistics, so citing this paper supports the mechanism story only
partially.

## 10. Other medical-imaging TTA studies

Only BayTTA (#6) was verified directly in Phase 0. A broader search for
additional medical-imaging-specific TTA studies was not completed
exhaustively in this phase — time-boxed given Phase 0 scope. **This is a
known gap**: before finalizing any novelty claim, Phase 1 should include a
more systematic search (e.g. via Google Scholar / Semantic Scholar citation
graphs of Shanmugam et al. and BayTTA) rather than relying on the
hand-picked list above. Flagged explicitly in
`docs/claims_and_risks.md` risk #1.

## Existing work on "validation-gated / selective TTA with clean fallback"

Based on the above (verified from primary sources, not search snippets):

- **Instance-level, learned selective TTA already exists** (Kim et al.,
  NeurIPS 2020) — auxiliary network predicts per-transform loss and filters
  transforms per sample.
- **Validation-driven, non-learned policy selection already exists**
  (Lyzhov et al., UAI 2020, Greedy Policy Search) — though GPS is described
  as policy-level search, not explicitly documented (from the page fetched)
  as having a per-sample divergence-based rejection step or an explicit
  "fall back to the clean/unaugmented prediction" rule.
- **Clean-image anchoring and basic BatchNorm adaptation are not just
  "discussed" but fully evaluated with quantitative results in the target
  preprint's own Appendix B** (verified from the full paper text — see §1
  above): clean anchoring reduced SmallCNN/PathMNIST's drop from −37.0% to
  −8.6%, and BN adaptation helped BloodMNIST but hurt PathMNIST. **These
  are not adjacent ideas — they are the exact mitigations this project must
  reproduce as required baselines, not present as its own contribution.**
  This is a stronger correction than the prior draft's "adjacent" framing.
- **No source verified here explicitly combines**: (a) a non-learned,
  validation-calibrated per-sample divergence threshold, (b) an explicit
  "insufficient views → fall back entirely to the clean prediction" rule,
  and (c) evaluation specifically against harm rate (not just accuracy) in
  the causal context of BatchNorm/resolution/policy-mismatch, **compared
  head-to-head against the source paper's own clean-anchoring and
  BN-adaptation baselines**. This last clause is now the narrowest
  defensible claim available — even it is `UNVERIFIED` as fully novel; it
  is simply the most specific gap found during this (non-exhaustive) Phase
  0 pass. See `docs/claims_and_risks.md` for exactly how this may and may
  not be described in any report.

## Verification gaps closed in this correction pass

- ✅ Full text (both appendices) of the target preprint read directly:
  exact dataset list, exact 12-row Table 2, exact augmentation/training
  specs, code URL (found — link is dead, see §1), Appendix B mitigation
  results.
- ✅ MedMNIST+ resolution construction confirmed via
  `on_medmnist_plus.md`: independently-sourced higher-resolution originals,
  not 28px upsampling (resolution confound retracted).
- ✅ Exact MedMNIST train/val/test sample counts for PathMNIST, BloodMNIST,
  DermaMNIST (see §2).

## Verification gaps still open

- Exact original source-dataset names/citations (the specific
  histopathology/blood-cell/dermatoscopy datasets each MedMNIST subset
  derives from) — needed for `docs/data_and_licensing.md` attribution.
- Patient/source-level leakage statement for MedMNIST splits.
- License terms for the Kim et al. and MedMNIST-C code repositories.
- A broader (non-hand-picked) search for medical-imaging-specific TTA
  robustness studies beyond BayTTA.
- Whether the dead `AI-Scientist-v3` code link becomes available later, or
  whether an archived/cached version exists (e.g. via the Wayback Machine)
  — worth a quick check in Phase 1 before assuming it is permanently gone.
