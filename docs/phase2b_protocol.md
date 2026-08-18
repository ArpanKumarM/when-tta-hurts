# Phase 2B Confirmatory Protocol — FROZEN

**Status: Phase 2B.1 — baseline confirmatory protocol PREREGISTERED and
FROZEN. Not yet implemented (no runner exists). No training, no dataset
download, no test-split access has occurred under this protocol.**

This document resolves the configuration/protocol blockers identified in
the Phase 2B preflight audit. It does not implement the matrix runner, does
not touch Phase 2A's preregistration/audit/pilot/ledgers, and does not
authorize any training, evaluation, or dataset access. Runner
implementation is a separate, later step (Phase 2B.2) requiring its own
approval.

---

## 1. Frozen matrix (unchanged — reference only)

The matrix in `configs/experiment_matrix.yaml` is frozen **as-is**: no
cells added or removed by this document.

| Block | Runs | Type |
|---|---|---|
| A — core normalization × resolution | 24 | Mandatory |
| B — policy matching | 6 | Mandatory |
| C — positive-control reproduction | 3 | Mandatory |
| **A+B+C** | **33** | **Mandatory** |
| D — conditional 128px | 6 | Conditional (see §6) |
| **Maximum** | **39** | |

- Confirmatory seeds: **[0, 1, 2]** — unchanged.
- Pilot seed **314159** — permanently excluded from all confirmatory runs
  (see §7 for the hard-failure enforcement requirement).
- Pilot TTA seed **271828** — not reused for any confirmatory TTA view
  sequence; confirmatory runs require their own TTA seed(s), distinct from
  271828 (exact confirmatory TTA seed value(s) to be set when the runner is
  implemented — not fixed by this document, since it is a new, not-yet-
  existing parameter; see §7's determinism requirements for how it must be
  derived once chosen).

## 2. Frozen training settings (all confirmatory blocks)

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0 |
| Loss | Cross-entropy |
| Max epochs | 30 |
| LR schedule | Cosine annealing over the full 30-epoch maximum |
| Early-stopping monitor | Validation accuracy |
| Early-stopping patience | 5 epochs |
| Minimum improvement | 0 |
| Checkpoint restored | Best validation accuracy (not last epoch) |
| Batch size | 256, for native 28px and 64px |
| Precision | Float32 throughout, no mixed precision |
| Device | MPS |
| Label smoothing | None |
| Class weighting | None |
| Channel standardization | None (no dataset-specific mean/std) |
| Input scaling | uint8 → float32, scaled to [0,1] |
| Training-time augmentation | **None** in Blocks A, C, D |
| Training-time augmentation | Block B **only**: the exact frozen mixed policy from `docs/experimental_protocol.md`'s "Frozen augmentation parameters" table, applied **once per training sample per step** (not multiple augmented copies per step) |

Block D's batch size at 128px is **not** fixed by this table — it is
determined by Block D's own activation gate (§6), which requires a native
128px benchmark before any 128px training occurs.

**Explicit statement (required):** the source paper reports only "25-30
epochs" and does not report every optimizer detail (exact weight decay,
exact early-stopping patience, exact minimum-improvement threshold are all
unstated by the source). Selecting `max_epochs=30`, `patience=5`,
`min_delta=0`, and `weight_decay=0` is a **preregistered operational
choice** made here, not a value extracted from the source paper. **This
remains a paper-constrained reproduction study, not an exact replication**
— consistent with `docs/experimental_protocol.md`'s existing framing of the
SmallCNN architecture and `docs/literature_review.md`'s finding that the
source paper's own code is unavailable.

## 3. Frozen evaluation conditions

### Primary confirmatory endpoint

- Clean test accuracy **versus** augmented-only mixed-policy TTA.
- Mean-probability aggregation.
- **N=50** (the source paper's headline condition).
- Delta accuracy in percentage points.
- Evaluated across seeds [0, 1, 2], reported per-seed and summarized (per
  `docs/statistical_analysis_plan.md`'s existing seed-variance requirement).

### Frozen view sequence

**[1, 2, 5, 10, 25, 50, 100]** — deterministic nested prefixes of **one
registered 100-view sequence per sample/run** (i.e., each of the 7 tested
view counts is a literal prefix of the same underlying 100-view ordered
sequence, not independently resampled — matching the pattern already
implemented and audited for the pilot in `src/when_tta_hurts/evaluation/tta.py`,
extended here from a 50-view to a 100-view registered sequence for
confirmatory work per `configs/experiment_matrix.yaml`'s
`tta_view_counts: [1, 2, 5, 10, 25, 50, 100]`).

### Secondary/descriptive analyses (all preregistered, none confirmatory-primary)

1. **Scaling curve** — mixed policy, mean probability, all 7 registered
   view counts. Preregistered secondary/descriptive analysis.
2. **Augmentation-strategy ablation** — geometric, intensity, mixed; mean
   probability; **N=25**. Preregistered secondary analysis.
3. **Aggregation ablation** — mixed policy; mean probability, majority
   vote, confidence-weighted average; **N=25**. Preregistered secondary
   analysis.
4. **Original-anchored condition** — one clean view plus N augmented
   views, equal-weight mean probability over N+1 views, all 7 registered N
   values, applicable to every checkpoint. (Source paper's Appendix B
   condition, reproduced as a required baseline per `docs/research_plan.md`'s
   H4 comparison set — not a project contribution.)
5. **BN-adapted condition** — applicable only to BatchNorm models; mixed
   policy; mean-probability aggregation; all 7 registered N values; **not
   applicable to GroupNorm cells** (GroupNorm has no running statistics to
   adapt). See §4 for the exact adaptation procedure. (Also a source-paper
   Appendix B condition, reproduced as a required baseline.)

### Frozen aggregation definitions

- **Mean probability:** arithmetic mean of per-view softmax probabilities.
  (Already implemented: `src/when_tta_hurts/evaluation/tta.py::aggregate_mean_prefix`.)
- **Majority vote:** each view votes for its argmax class; ties are
  resolved by the highest mean probability among tied classes; any
  remaining exact tie is resolved by lowest class index. (Not yet
  implemented — see the Phase 2B preflight's implementation-readiness
  findings.)
- **Confidence-weighted average:** each view is weighted by its maximum
  softmax probability; weights are normalized to sum to one, then
  probability vectors are averaged using those normalized weights. (Not
  yet implemented.)

### Frozen multiclass Brier score definition

```
Brier = mean over samples of [ sum over classes of (predicted_probability - one_hot_label)^2 ]
```

(Not yet implemented — required secondary endpoint per
`docs/experimental_protocol.md`, previously specified only as "Brier
score" with no formula; this is now the frozen formula.)

### Unchanged definitions (kept as already implemented)

Accuracy, macro-F1, negative log-likelihood, expected calibration error,
harm rate, rescue rate — definitions unchanged from
`src/when_tta_hurts/metrics.py`, already independently validated (Phase 2A
audit, `tests/test_metrics_independent_validation.py`).

### Frozen inference latency specification

Inference latency measurement **must include device synchronization**
(`torch.mps.synchronize()` before/after timed regions, per the pattern
already used in `scripts/benchmark_runtime.py`) and must report:

- Clean latency (single forward pass).
- Total TTA latency at each registered N.
- Per-sample latency (total / sample count).
- Compute multiplier relative to clean inference (TTA latency / clean
  latency), per N.

## 4. Frozen BN-adaptation semantics

**Conflict check performed:** `docs/experimental_protocol.md`'s existing
description ("basic BatchNorm-statistics adaptation using the
augmented-batch distribution... applicable only to BatchNorm cells") is a
general description, not a precise algorithm. The exact procedure below is
a **refinement**, not a contradiction — **no conflict found**, nothing
silently changed.

**Exact procedure (frozen):**

1. Begin from an untouched **copy** of the frozen best checkpoint (the
   original checkpoint file/weights are never mutated in place).
2. Use only **unlabeled** inputs from the split being evaluated.
3. Perform **one deterministic, no-gradient pass** over the relevant
   augmented inputs.
4. Update **BatchNorm running mean/variance only**.
5. Do **not** update convolutional, linear, affine-BN (`weight`/`bias` of
   BatchNorm), or any other learned parameter.
6. Return the model to evaluation mode before prediction.
7. **Reset from the original checkpoint separately** for every
   split/N/seed/condition combination — adapted statistics are never
   carried across N values, datasets, seeds, or conditions.
8. **Never use labels during adaptation.**
9. This is a **paper-constrained operationalization**, recorded explicitly
   because the source paper's own implementation is unavailable (dead code
   link — see `docs/data_and_licensing.md`).

## 5. Strengthened test firewall

This **replaces** the draft protocol's test-firewall section in
`docs/experimental_protocol.md` with a strictly stricter version: **no
official test evaluation occurs during Phase 2B training or Phase 3 method
development.** This is a deliberate tightening relative to the earlier
draft, which permitted block A/B/C training-run reproduction to touch the
test set directly as a "verification check." Under this frozen order, even
that reproduction-style test touch is deferred to the very end, after
Validation-Gated TTA's algorithm is frozen — preventing baseline test
results from influencing development of the proposed mitigation in any
way, including implicitly.

**Frozen order:**

1. Implement and commit the Phase 2B runner.
2. Train Blocks A/B/C using train data and validation-based early stopping
   only.
3. Make the conditional Block D decision using runtime evidence only (§6).
4. Train Block D if activated.
5. Develop Validation-Gated TTA using validation data only.
6. Freeze and commit its algorithm and thresholds.
7. Only then unlock the official test split — once.
8. Run one final test evaluation for every frozen condition.
9. Never use test results to change configurations, thresholds, rerun
   decisions, or claims.

**Until Validation-Gated TTA is separately approved and committed:**
- Baseline checkpoints (blocks A/B/C/D) may be trained.
- Validation-only engineering checks may be performed.
- **No official test predictions or metrics may be generated, for any
  block, including A/B/C/D**, until step 7 above.

This closes the gap the preflight identified: `data.py::load_dataset`'s
`allow_test=True` mechanism exists but must remain uninvoked by any script
until an explicit, separately-committed authorization exists (see §7's
"final-evaluation authorization artifact" requirement).

## 6. Frozen Block D numeric trigger

Block D activates **as an entire six-run block** only if **all** of the
following pass, checked **before any 128px training**:

1. Official native PathMNIST-128 and BloodMNIST-128 training artifacts are
   downloaded.
2. Both official checksums match `medmnist.INFO` metadata exactly.
3. A native-real-data MPS benchmark (analogous to
   `scripts/benchmark_runtime.py`'s round-2 canonical benchmark, but at
   128px) completes without OOM or non-finite loss.
4. **No resized 28px/64px proxy is used for the activation decision** —
   only genuine native 128px artifact content.
5. Projected training time is **at most 90 minutes per 128px run**.
6. Projected end-to-end time (training + evaluation) is **at most 120
   minutes per 128px cell**.
7. Pessimistic projected total for A+B+C+D remains **below 24 hours**.
8. The decision uses **runtime/memory evidence only** — not accuracy or
   TTA outcomes.

**If either dataset (PathMNIST-128 or BloodMNIST-128) fails any gate,
Block D is omitted in its entirety** and the failed gate is reported
honestly (not silently dropped, not partially executed).

This resolves the preflight's finding that the "per-run kill criterion"
referenced by the original draft was never numerically specified
(`docs/compute_budget.md`'s "TBD"). It is now numerically frozen: **90
minutes/run training, 120 minutes/cell end-to-end, 24-hour pessimistic
total ceiling.**

**Runtime stops:**

- Stop and record a Block D attempt if training exceeds 90 minutes.
- Stop and record a cell if end-to-end execution exceeds 120 minutes.
- Cancel the remaining Block D cells if the pessimistic total crosses 24
  hours.
- **Do not remove completed A/B/C results** under any Block D stop
  condition.

## 7. Frozen reproducibility requirements (for the later runner)

These are requirements the Phase 2B.2 runner implementation must satisfy
— documented now, enforced later, per your instruction not to implement
the runner in this phase.

- **Deterministic run ID**, derived from: block, dataset, native
  resolution, model, normalization, training policy, and seed. (Not a
  random UUID, unlike the Phase 2A pilot's `run_pilot.py` — that scheme is
  explicitly pilot-only and must not be reused for confirmatory runs.)
- **Canonical configuration hash** (via `src/when_tta_hurts/config.py::config_hash`,
  already implemented) stored with every attempt.
- **Deterministic artifact directory**, derived from the run ID.
- **Atomic checkpoint and result writes** (via
  `src/when_tta_hurts/artifacts.py::save_checkpoint`/`atomic_write_json`/
  `atomic_write_npz`, already implemented).
- **Explicit attempt numbers** under each stable run ID (so retries are
  distinguishable from the canonical result).
- **Matching completed run: skip safely** — if a run with the same run ID
  and matching config hash has already completed, do not re-execute it.
- **Completed run with conflicting hash: hard failure** — if a run ID
  already has a completed result but the *new* attempt's config hash
  differs, this is a hard error (indicates the frozen protocol was
  changed), not a silent overwrite.
- **Failed/aborted attempt: append incident automatically** — unlike
  Phase 2A's manual incident-ledger pattern (`artifacts/ledger_incidents.csv`,
  written by hand after the fact), the Phase 2B runner must do this
  automatically on any crash/kill/timeout.
- **Never overwrite an existing completed result.**
- **First valid completed attempt becomes canonical** for that run ID.
- **No rerun after test metrics are observed merely because the result is
  unfavorable** — this is a direct restatement of the test-firewall
  discipline in §5 and `CLAUDE.md`.
- **Confirmatory ledger entries must use `confirmatory=true`** — this
  requires a new ledger function (`append_confirmatory_entry` or
  equivalent); the existing `append_pilot_entry` hardcodes
  `confirmatory=False` and must not be reused or monkey-patched for this
  purpose.
- **Pilot artifacts/checkpoints/predictions must be rejected as
  confirmatory inputs** — the runner must not load anything from
  `artifacts/pilots/` as a starting checkpoint, prior prediction, or
  aggregation input for any confirmatory run.
- **Seed 314159 must trigger a hard failure** in the confirmatory runner
  if requested for any confirmatory run, at validation time before any
  compute is spent.
- **Test access must require an explicit final-evaluation authorization
  artifact**, committed **after** Validation-Gated TTA is frozen (§5 step
  6) — a separate, explicit, committed file (not a config flag flipped
  inline) that the runner checks for before it will call
  `load_dataset(..., allow_test=True)` at all.

**Frozen deterministic execution order:** the **literal row order in
`configs/experiment_matrix.yaml`** — Block A → Block B → Block C →
conditional Block D. Within each block, the exact expanded order committed
in the matrix (dataset-major, then resolution, then normalization, then
seed, as the YAML's own field ordering implies) is preserved rather than
relying on dictionary iteration order, which is not guaranteed stable
across Python versions/implementations for this purpose.
