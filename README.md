# When Test-Time Augmentation Hurts: A Controlled Study in Medical Image Classification

An independent, **preregistered**, fully audited reproduction and
mechanistic characterization of a specific empirical claim from a recent
preprint:

> *"I Can't Believe TTA Is Not Better: When Test-Time Augmentation Hurts
> Medical Image Classification"* — arXiv:2604.09697 (April 2026,
> single-author preprint, **not peer-reviewed**).

That preprint reports that naive test-time augmentation (TTA) *reduced*
accuracy in 11 of 12 model/dataset combinations on MedMNIST. This project
independently verifies that claim on a preregistered confirmatory matrix,
characterizes how far the harm extends (normalization, resolution,
view count, aggregation rule), asks whether training/test augmentation-
policy matching mitigates it, and — in a post-review extension — localizes
the effect: it is a failure of augmentation robustness (a content-
preserving geometric-only policy causes it too), not a consequence of the
augmentation destroying labeled content.

The manuscript is in [`paper/manuscript.md`](paper/manuscript.md); the
TMLR-formatted build is in [`paper/tmlr/`](paper/tmlr/).

---

## For reviewers — start here

| To check… | Look at |
|---|---|
| **What was preregistered, and when** | [`docs/research_plan.md`](docs/research_plan.md) and [`docs/phase2b_protocol.md`](docs/phase2b_protocol.md); frozen in git history (Phase 0 = commit `a22db01`; the test split was not accessed until a later, separately authorized step) |
| **The manuscript** | [`paper/manuscript.md`](paper/manuscript.md) (prose) / [`paper/tmlr/main.tex`](paper/tmlr/main.tex) (build: `cd paper/tmlr && make`) |
| **That every number in the paper is real** | `uv run python3 paper/verify_manuscript_claims.py` — cross-checks every numeric claim in Results **and** the post-review "Extended Analyses" section against the sealed evidence artifacts; exits non-zero on any mismatch |
| **The primary result's evidence** | [`artifacts/final_test_scientific_summary.json`](artifacts/final_test_scientific_summary.json) (hash-bound) and [`artifacts/paper_evidence/`](artifacts/paper_evidence/) (figures + tables + manifest) |
| **The post-review extension** | protocols [`docs/phase2c_secondary_analysis_expansion_plan.md`](docs/phase2c_secondary_analysis_expansion_plan.md), [`docs/phase2c2_label_preservation_audit_protocol.md`](docs/phase2c2_label_preservation_audit_protocol.md), [`docs/phase2c2_component_ablation_addendum.md`](docs/phase2c2_component_ablation_addendum.md); results `docs/phase2c*_findings.md`; artifacts `artifacts/{secondary_analysis_expansion,label_preservation_audit,component_ablation}/` |
| **Citation provenance** | [`paper/citation_audit.md`](paper/citation_audit.md) — every cited work, the primary source fetched, what was verified |
| **A fast one-command reproduction of the headline numbers** | the companion repo `when-tta-hurts-minimal` (`reproduce.py`) — covers the primary matrix and the secondary conditions; the label-preservation audit and component ablation are reproduced from this repo |

### About the `docs/` directory

`docs/` has ~90 files because every protocol freeze, authorization step,
and process incident is written down and committed rather than kept as
tribal knowledge — that append-only record *is* the reproducibility
contribution. You do not need to read them all: the table above lists the
handful that matter. The rest are the audit trail, browsable if you want
to check a specific claim about the process.

---

## What this is not

- **Not clinically validated.** Uses [MedMNIST](https://medmnist.com/),
  which is explicitly not intended for clinical use. Nothing here is
  diagnostic, clinically validated, or deployment-ready.
- **Not a novelty claim for TTA, adaptive TTA, or TTA gating.** This is a
  replication + characterization study; it proposes no new method,
  aggregator, adaptation procedure, estimator, benchmark, or bound. See
  the manuscript's Related Work.
- **Not a critique of the source preprint's authors.** Independent
  replication in the normal spirit of scientific practice.

## Repository structure

```
paper/       manuscript (Markdown), references + citation audit,
             verify_manuscript_claims.py, reviews/, and tmlr/ (LaTeX build)
docs/        preregistration, protocol, statistical analysis plan, the
             Phase 2B audit trail, and the Phase 2C/2C.2 extension protocols
configs/     frozen machine-readable experiment configuration
src/         when_tta_hurts package: data, models, transforms, evaluation,
             statistics, sealed-pipeline / authorization / artifact plumbing
scripts/     analysis + figure-generation entry points (expand_secondary_analysis.py,
             label_preservation_audit.py, component_ablation.py, md_to_tex.py, ...)
artifacts/   sealed scientific summary, paper-evidence package, ledgers, and
             the Phase 2C summary/CSV deliverables (large per-run predictions,
             checkpoints, and view renders are git-ignored / regenerated)
tests/       offline unit tests
data/        MedMNIST cache — downloaded at runtime, git-ignored
```

## Reproducing

```bash
uv sync
uv run python3 paper/verify_manuscript_claims.py          # check every paper number
uv run python scripts/expand_secondary_analysis.py        # scaling curve + secondary conditions
uv run python scripts/label_preservation_audit.py --with-human   # (human_scores.csv committed)
uv run python scripts/component_ablation.py               # geometric/intensity decomposition (~2.5 h, resumable)
```

Training and final-test evaluation require the sealed pipeline and its
authorization artifacts (`docs/phase2b_*`); the analysis scripts above run
from the already-sealed predictions / trained checkpoints and do not
re-open the test split.

## License

Original code and text are MIT-licensed (see `LICENSE`). This does **not**
extend to the MedMNIST dataset or the cited papers, which keep their own
license/copyright — see `docs/data_and_licensing.md`.
