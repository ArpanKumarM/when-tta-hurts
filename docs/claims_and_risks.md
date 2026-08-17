# Claims, Novelty Positioning, and Risks

**Status: draft, Phase 0. No results exist yet — this document governs what
we will be *allowed* to claim once results exist, and is deliberately
skeptical about novelty.**

## Novelty positioning

**Correction note:** the target preprint's full text (both appendices) has
now been read directly (see `docs/literature_review.md` §1). This
materially shrinks the available novelty space and is reflected below.

**Already established, confirmed directly in the target preprint itself
(not just adjacent literature)**:
- **Clean-image anchoring** (including the original unaugmented image in
  the TTA aggregate) — evaluated in Appendix B with quantitative results
  (e.g. SmallCNN/PathMNIST drop reduced from −37.0% to −8.6%).
- **Basic BatchNorm-statistics adaptation at test time** — evaluated in
  Appendix B, helped on BloodMNIST but hurt on PathMNIST.
- BatchNorm is already named by the source paper as a contributing cause of
  TTA degradation, not a hypothesis original to this project.

**Already well established in the broader prior literature**
(`docs/literature_review.md`):
- Learned or heuristic TTA aggregation weighting (vs. naive mean) —
  Shanmugam et al. 2021.
- Learned selection/search over TTA policies — Lyzhov et al. 2020 (Greedy
  Policy Search).
- Learned per-sample loss/confidence prediction to weight or select TTA
  views — "Learning Loss for Test-Time Augmentation," NeurIPS 2020.
- Bayesian aggregation of TTA views — BayTTA.
- The general idea that TTA can hurt and that selective/adaptive TTA can
  help is **not new**. This project must not claim otherwise.

**Consequently, H4's Validation-Gated TTA must be evaluated against, and
explicitly framed as sitting alongside, the source paper's own
clean-anchoring and BN-adaptation baselines** — see the required comparison
set in `docs/research_plan.md`'s H4 and `docs/experimental_protocol.md`'s
conditions list. Presenting either mitigation as something this project
discovered would be a direct overclaim.

**Potentially differentiated (to be confirmed, not assumed):**
- A controlled *causal* isolation of normalization type, input resolution,
  and train/test augmentation-policy matching as explanatory factors,
  specifically in the setting of the 2026 MedMNIST TTA-failure result. Prior
  TTA-aggregation papers largely propose better aggregation methods without
  isolating *why* naive TTA fails via a normalization/resolution/policy
  ablation. If the literature review turns up prior work that already does
  this decomposition for medical imaging or MedMNIST specifically, this
  claim must be narrowed or dropped.

**Engineering contribution (not scientific novelty):**
- A reproducible, from-scratch implementation of a simple, non-learned
  validation-gated TTA fallback, released with code, configs, and raw
  predictions.

**Scientific contribution, if any, is the experiments and findings** —
not the existence of software with a particular name. A software artifact
alone is not a claimed contribution.

**If the literature review shows the causal-decomposition angle is already
covered** (e.g. an existing paper already ablates BatchNorm vs. GroupNorm
under TTA on medical imaging with a similar resolution sweep), this project
should be explicitly repositioned as a **replication + extension to a new
dataset/paper**, not a novel causal study, and the report should say so
plainly rather than searching for a thinner novelty angle.

## Claims table

| Proposed claim | Evidence required | Current status | Allowed wording | Prohibited overclaim |
|---|---|---|---|---|
| The source paper's headline TTA-failure result is reproducible | Independent re-run of source paper's fixed config (Table 2: LogReg/MLP/SmallCNN/ResNet-18 × PathMNIST/BloodMNIST/DermaMNIST, mixed policy, N=50, mean agg.) on official test set, compared to their reported 12 numbers | Not evaluated | "We reproduce a TTA accuracy drop of similar direction/magnitude to [source paper] on N/M of the 12 cells" | "We confirm/prove the source paper's finding" |
| TTA degradation differs between BatchNorm and GroupNorm (H1, corrected wording) | Confirmatory 3-seed results (block A), paired test, corrected p-value, effect size, across PathMNIST+BloodMNIST | Not evaluated | "In our controlled comparison, TTA degradation differed between BatchNorm and GroupNorm cells (effect size, CI)" | "BatchNorm causes TTA to fail"; "this proves running statistics are the mechanism" — BatchNorm vs. GroupNorm differs in more than running statistics, and the `no_running_stats` decomposition arm has been removed from the confirmatory matrix (see research_plan.md's H1 note) |
| Higher resolution reduces TTA degradation (H2) | Confirmatory 3-seed results, 28px vs. 64px (block A); 128px (block D) only strengthens a trend claim if all 3 seeds complete | Not evaluated | "TTA degradation was smaller at 64px than 28px in our experiments" (28/64 only, unless block D completes) | Claiming a general scaling law; presenting 128px results as confirmatory if block D's gate did not pass or seeds are incomplete. The upsampling-artifact confound previously flagged has been **retracted** — MedMNIST+ 64/128px are independently-sourced, not upsampled (verified, see data_and_licensing.md) |
| Matched train/test augmentation reduces TTA degradation (H3) | Confirmatory 3-seed results, block B, PathMNIST+BloodMNIST at 28px only | Not evaluated | "Matching augmentation policy reduced degradation relative to naive TTA at 28px" | Presenting this as a novel finding (it is expected under standard ML theory); generalizing beyond 28px without data |
| Validation-gated TTA reduces harm rate vs. other conditions (H4) | Confirmatory 3-seed results, harm-rate comparison against ALL of: clean, naive TTA, original-anchored TTA, BN-adapted TTA (paired tests) | Not evaluated | "A simple validation-gated fallback reduced the clean-correct→TTA-wrong harm rate relative to naive mean TTA and to [specific baseline] in our experiments" | "This guarantees TTA will not hurt"; "this is a novel algorithm"; "this is the first mitigation for TTA failure" — clean anchoring and BN adaptation are **already in the source paper's own appendix**, not novel to this project |
| This work is clinically relevant | N/A — out of scope by design | Not applicable | (no clinical claims permitted) | Any claim of clinical validity, diagnostic utility, or deployment readiness |
| This work is the first to study TTA failure causes in medical imaging | Full literature review completion | **Very likely false** — the source paper itself already attributes failure to BatchNorm/distribution-shift and evaluates two mitigations; BayTTA already studies TTA aggregation specifically in medical imaging | Narrow, hedged claims only about the *specific combination* tested (see literature_review.md's "Existing work" section) | "We are the first to study TTA failure in medical imaging" or any claim of this shape — directly contradicted by the source paper's own analysis |

## Five biggest scientific risks (updated after full-text verification)

1. **Novelty is substantially covered, and now confirmed rather than suspected.** The source paper's own Appendix B already evaluates clean anchoring and BN adaptation with quantitative results; BayTTA already does medical-imaging-specific TTA aggregation; Kim et al. (NeurIPS 2020) already does learned per-sample transform rejection. This is the single biggest risk to the project's contribution claim. The only remaining candidate claim is the narrow combination described in `docs/literature_review.md`'s "Existing work" section — the report must lead with this narrowness, not obscure it.
2. **Resolution confound — RETRACTED.** Verified directly against `on_medmnist_plus.md`: MedMNIST+ 64/128px images for PathMNIST/BloodMNIST/DermaMNIST are resized from independently-sourced higher-resolution originals (224×224, cropped-360×363, 600×450 respectively), not upsampled from 28px files. H2 is a stronger design than previously assumed; this risk is closed, not open.
3. **Compute budget — RESOLVED for the corrected matrix.** The corrected training matrix (blocks A+B+C = 33 runs, +D = 39) fits the 30-40 run target (see `docs/compute_budget.md`). Residual risk is unmeasured runtime/memory on the M4, not run-count design.
4. **H1's normalization comparison is not a clean single-factor ablation** — GroupNorm and BatchNorm differ in more than the presence of running statistics, so a "differs" result under-determines the mechanism (H1's wording was corrected to reflect this — see research_plan.md). The `no_running_stats` decomposition arm has been removed from the confirmatory matrix to fit budget, which *weakens* our ability to isolate the mechanism further; causal language must stay hedged regardless of outcome.
5. **Reproduction risk**: the source paper is an unreviewed, single-author April 2026 preprint with no working code link (the linked repo 404s). Its own numbers might not replicate exactly even with a faithful re-implementation of its stated hyperparameters (Adam 1e-3, cosine annealing, 25-30 epochs, batch 256). A failed exact reproduction is itself a valid and reportable finding, not a failure of this project — must be planned for and not treated as something to hide or force-fit.
6. **PathMNIST dataset-validity risk (added Phase 1, not a hypothesis change).** A preprint (Ignatov & Malivenko 2024, arXiv:2409.11546 — see `docs/data_and_licensing.md`) reports that NCT-CRC-HE-100K (PathMNIST's source) has color-normalization and JPEG-artifact biases strong enough that simple color-histogram features alone reach >82% accuracy on the 9-class task. If true, PathMNIST classification may be partly driven by low-level, non-morphological signals — meaning a TTA failure on PathMNIST could reflect disruption of those low-level statistics (by color-jitter/geometric transforms) rather than, or in addition to, the BatchNorm/resolution mechanisms in H1/H2. H1-H4 are unchanged; this is a documented interpretive risk to disclose when reporting PathMNIST results, not a new hypothesis.

## Rules for this document

- Update the "current status" column as results become available — never
  skip a row or delete a claim that turned out negative.
- Any claim not in this table must be added here (with evidence
  requirements defined) before it is used in any report or public post.
