# Phase 2A Pilot Protocol — PathMNIST, 28px, SmallCNN/BatchNorm

**Status: PREREGISTERED, frozen before implementation or execution.**
**This document is committed BEFORE the pilot is run — see git log.**

## Purpose

A single validation-only engineering/scientific-sanity pilot to (a) verify
the training/evaluation pipeline produces sane, reproducible results before
committing to the confirmatory matrix, and (b) get a first, exploratory
read on clean vs. TTA accuracy for one cell. This is **not** a confirmatory
experiment and is not part of blocks A/B/C/D in
`configs/experiment_matrix.yaml`.

## Pilot seed

**314159.** Verified against `configs/experiment_matrix.yaml`'s
confirmatory seeds `[0, 1, 2]` — no conflict. **314159 is permanently
excluded from confirmatory results**: if this pilot's cell is later
promoted into the confirmatory matrix, it must be retrained with seeds 0,
1, 2, not reuse this pilot's checkpoint or seed.

## Frozen configuration

See `configs/pilot_pathmnist_28_bn.yaml` for the machine-readable version of
everything below. This document and that config are the authoritative,
frozen specification — per `CLAUDE.md`, they are not to be changed after
viewing pilot results without a documented amendment.

### Dataset

- PathMNIST, resolution 28×28.
- Official **training** split: used for training only.
- Official **validation** split: used for early-stopping monitoring and for
  all pilot TTA evaluation (clean + augmented).
- Official **test** split: **prohibited**. Not loaded, not evaluated, not
  inspected, at any point in this pilot. Enforced in code — see "Test-set
  firewall" in the Phase 2A completion report.

### Model

- Paper-constrained SmallCNN (`src/when_tta_hurts/models/small_cnn.py`),
  BatchNorm, 9 classes, the current frozen 94,857-parameter architecture
  (conv 32→64→128, AdaptiveAvgPool, Linear 128→9). No architecture changes
  in this pilot.

### Input preprocessing

- uint8 → float32, scaled to `[0, 1]` (`torchvision.transforms.ToTensor()`,
  already the default in `data.py::load_dataset`).
- No dataset-specific channel mean/std standardization.
- No training-time augmentation.
- No label smoothing.
- No class weighting.

### Training

- Optimizer: Adam, learning rate 0.001, weight decay 0.
- Loss: cross-entropy.
- Max epochs: 30.
- LR schedule: cosine annealing over the full 30-epoch budget (not
  restarted by early stopping).
- Early stopping: monitor validation accuracy, patience 5 epochs, minimum
  improvement 0 (any improvement resets patience), restore the checkpoint
  with the **best** validation accuracy (not the last epoch).
- Precision: float32 throughout. No mixed precision.
- Target batch size: **256**, subject to the runtime/memory gate in
  `scripts/benchmark_runtime.py` — see Part 4 of the Phase 2A completion
  report for the resolved value and why.

### Pilot TTA (validation split only)

- Policy: the frozen **mixed** policy from
  `docs/experimental_protocol.md`'s "Frozen augmentation parameters" table
  (implemented in `src/when_tta_hurts/transforms/policies.py` — unchanged
  in this pilot).
- Aggregation: **mean probability only** (majority vote and
  confidence-weighted are NOT evaluated in this pilot).
- View counts: 1, 2, 5, 10, 25, 50 — **augmented views only**, matching the
  source paper's main experiment. **The clean image is NOT included as an
  anchor in this pilot's TTA views** (that is the source paper's Appendix B
  condition, evaluated separately in a later phase, not here).
- Nested deterministic prefixes: a single ordered 50-view sequence is
  generated once per validation sample (seeded — see below); each tested
  view count (1/2/5/10/25/50) uses the first N views of that same sequence,
  so smaller-N results are literal prefixes of the 50-view result, not
  independently resampled.
- TTA seed: a **separately recorded deterministic seed**, distinct from the
  pilot's model/training seed 314159 — see `configs/pilot_pathmnist_28_bn.yaml`
  (`tta_seed: 271828`), so the view sequence is reproducible independent of
  the model's own training randomness.

### Pilot metrics (validation split only)

- Clean validation accuracy.
- TTA validation accuracy at each view count (1, 2, 5, 10, 25, 50).
- Delta accuracy = TTA accuracy − clean accuracy, per view count.
- Macro-F1 (clean and each TTA view count).
- Negative log-likelihood (clean and each TTA view count).
- Expected calibration error (clean and each TTA view count).
- Clean-correct → TTA-wrong harm rate (per view count).
- Clean-wrong → TTA-correct rescue rate (per view count).
- Training time (wall clock).
- Inference time (clean + TTA, wall clock).
- Peak MPS memory (current allocated + driver-allocated where supported).

## Explicit interpretation constraints

- **These are exploratory validation results, not confirmatory findings.**
  They come from one seed (314159, permanently excluded from the
  confirmatory matrix), evaluated only on the validation split.
- **They cannot be used in a paper abstract, a workshop submission, or a
  LinkedIn/social claim.** They exist to sanity-check the pipeline, not to
  report a scientific result.
- **No exact replication claim of the source paper is possible from this
  pilot** — the source paper's own code is unavailable (404, see
  `docs/data_and_licensing.md`), so this SmallCNN is a paper-constrained
  reimplementation, not a byte-for-byte reproduction, and this pilot uses
  only one seed on the validation split, not the source paper's own
  (test-set) evaluation protocol.
- **A non-negative (neutral or beneficial) TTA result must be reported
  honestly.** If this pilot finds TTA does not hurt PathMNIST/SmallCNN/
  BatchNorm at 28px, that is reported as-is. The augmentation policy,
  aggregation method, or view counts must NOT be silently modified to try
  to reproduce the source paper's reported degradation. Any such change
  would be test-set-blind p-hacking of the pilot itself and is prohibited
  by `CLAUDE.md`.
- Per `CLAUDE.md`: this configuration is frozen once committed. Any change
  after viewing pilot results requires a documented protocol amendment, not
  a silent edit.

## Relationship to the confirmatory matrix

This pilot's cell (PathMNIST, 28px, SmallCNN, BatchNorm) overlaps with
block A of `configs/experiment_matrix.yaml`, but is NOT one of block A's
three confirmatory seeds (0, 1, 2) and does NOT count toward the 33/39 run
budget. It is exploratory scaffolding only, per
`docs/statistical_analysis_plan.md`'s confirmatory/exploratory separation.
