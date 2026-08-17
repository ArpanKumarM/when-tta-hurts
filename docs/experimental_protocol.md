# Experimental Protocol

**Status: draft, Phase 0. Not yet approved for execution.**

## Models

Architecture and hyperparameter specs below are read directly from the
source paper's Table 1 / Section 4.2 / Appendix A (full text verified —
see `docs/literature_review.md`), used here as the reproduction target, not
as untested assumptions.

- **SmallCNN** — primary model for the causal ablation matrix (H1–H3).
  Source paper spec: ~95K parameters, 3 convolutional layers, BatchNorm,
  MaxPool. Our GroupNorm variant substitutes GroupNorm for BatchNorm in the
  same architecture; exact channel widths/depths to be finalized in Phase 1
  and frozen in a config file before any run, so they aren't tuned against
  results.
- **ResNet-18** — source paper spec: ~11M parameters, adapted for 28×28
  input (no initial pooling layer). Used only for (a) exact baseline
  reproduction of the source paper's reported numbers, and (b) the
  positive-control reproduction (block C: DermaMNIST, the source paper's
  sole reported improvement case, +1.6pp at N=50) — not the full causal
  matrix. Swapping normalization inside ResNet-18 is out of scope for the
  confirmatory matrix (see `configs/experiment_matrix.yaml`).
- **LogReg (~21K params) and MLP (~670K params)** — used only for exact
  baseline reproduction of the source paper's full Table 2 (all 12
  model-dataset combinations at N=50), not for the causal ablation matrix.
  Reproducing these two arms is comparatively cheap and directly checks
  whether our re-implementation matches the source paper before extending
  it.
- No foundation models, no additional architectures, in the initial study.

## Training hyperparameters (reproduction target, per source paper)

- Optimizer: Adam, learning rate 1e-3.
- Schedule: cosine annealing.
- Epochs: 25-30, with early stopping on validation accuracy.
- Batch size: 256 (per the source paper's Appendix A).
- These are the reproduction-target hyperparameters. Any deviation forced
  by MPS/memory constraints must be documented explicitly in
  `docs/compute_budget.md` and flagged in the run ledger, not silently
  substituted.

## TTA specification (reproduction target, per source paper)

- View counts: 1, 2, 5, 10, 25, 50, 100.
- Policies: **geometric** (flips, rotation ±15°, random resized crop
  0.8-1.0×); **intensity** (brightness/contrast jitter ±0.3, Gaussian
  blur); **mixed** (geometric + intensity). The source paper's headline
  failure result uses the **mixed** policy at **N=50** with **mean**
  aggregation — this is the primary reproduction target.
- Aggregation methods: mean probability, majority vote, confidence-weighted
  average.

## Conditions

1. **Clean baseline** — one unaugmented forward pass.
2. **Naive TTA** — mean probability over augmented views, mixed policy,
   primarily at N=50 (the source paper's headline condition), with the
   full 1/2/5/10/25/50/100 view-count curve as a secondary/exploratory
   sweep.
3. **Original-image-anchored TTA** — original (clean) image included
   alongside augmented views in the aggregate. **This is the source
   paper's own Appendix B condition, reproduced here as a required
   baseline, not a contribution of this project.**
4. **Basic BN-adapted TTA** — BatchNorm statistics adapted using the
   augmented-batch distribution, applicable only to BatchNorm cells.
   **Also the source paper's own Appendix B condition, reproduced as a
   required baseline, not a contribution of this project.**
5. **Matched TTA** — train-time augmentation sampled from the same approved
   policy used at test time (H3; block B cells only).
6. **Validation-selected (gated) TTA** — see algorithm below. Compared
   against all five conditions above, not naive TTA alone (H4's required
   comparison set — see `docs/research_plan.md`).

## Validation-gated TTA algorithm (draft — must be frozen before test eval)

This is a first draft for review, not final:

1. Candidate transform pool is fixed in advance (documented in the config,
   not chosen post hoc).
2. **Transform-level filtering:** for each candidate transform, measure its
   effect on validation accuracy (paired against the clean validation
   prediction). Reject any transform that reduces validation accuracy beyond
   a pre-specified tolerance. This uses validation data only.
3. **View aggregation:** average probabilities over the surviving transforms
   plus the clean view (clean view is always included).
4. **Optional per-sample rejection:** for a given test sample, compute a
   divergence (e.g. KL or L1 on predicted probability vectors) between each
   augmented view's prediction and the clean prediction. A per-sample view
   is excluded from the aggregate if its divergence exceeds a threshold
   calibrated on validation data (e.g. a percentile of the validation
   divergence distribution among *correctly retained* predictions).
5. **Fallback:** if fewer than a minimum number of views survive rejection,
   fall back to the clean prediction alone.
6. All thresholds (tolerance in step 2, divergence percentile in step 4,
   minimum view count in step 5) are selected using validation data only,
   then frozen. They are not re-tuned after seeing test results.

Open questions to resolve in Phase 1 before freezing: exact divergence
metric, exact percentile/tolerance values, and how ties/insufficient-views
edge cases are handled. These will be decided using validation data and
written into a versioned config, with the config hash recorded in the run
ledger.

## Test firewall

- Source-paper reproduction (blocks A/B/C training runs, plus exact
  Table-2-style reproduction using the source paper's fixed, published
  configuration) is kept **separate from method development** — reproducing
  a published, fixed configuration on the official test set is a
  verification check, not a design decision informed by test data, so it is
  the one case permitted to touch the test set directly.
- All gating rules, divergence thresholds, transformation-pool selection,
  and early-stopping/promotion rules for the validation-gated TTA method
  (H4) use **validation data only**. This includes every step of the
  algorithm below (transform filtering, per-sample divergence threshold,
  minimum-view fallback count).
- These rules are **frozen before any final method test evaluation** —
  recorded in `docs/experimental_protocol.md`/`configs/experiment_matrix.yaml`
  with `status: approved` and a git commit reference — and are not re-tuned
  afterward.
- **Per-sample test predictions and per-sample test failures must not be
  inspected while developing or tuning the method.** Development-time
  debugging, error analysis, and threshold calibration are done exclusively
  against validation-set predictions. If a bug is suspected in a component
  that touches test data (e.g. the reproduction pipeline), fix the code
  without reading individual test predictions, and re-run rather than
  patching based on an observed test-set failure.
- Every test-set evaluation, without exception, is recorded as a row in the
  append-only run ledger (`results/ledger.csv`), including the config hash,
  git commit, and timestamp, so it is possible to audit both that no
  test-set peeking occurred before freezing and how many times the test set
  was evaluated at all.

## Endpoints

**Primary:** Delta accuracy = TTA accuracy − clean single-pass accuracy
(paired per run, per condition).

**Secondary:**
- Macro-F1
- Negative log-likelihood
- Expected calibration error (ECE)
- Brier score
- Clean-correct → TTA-wrong harm rate
- Clean-wrong → TTA-correct rescue rate
- Number of accepted TTA views (validation-gated condition only)
- Inference latency and compute multiplier vs. clean baseline

All endpoints are computed from saved per-sample predictions, never
recomputed by hand or typed into documents directly.
