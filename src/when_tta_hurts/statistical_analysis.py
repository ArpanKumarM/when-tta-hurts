"""Phase 2B.5A statistical-analysis engineering.

Read-only, plan-mode-first infrastructure implementing exactly the
fully-specified subset of docs/statistical_analysis_plan.md, as mapped in
docs/phase2b_statistical_analysis_engineering_freeze.md. This module never
accesses the official test split (no such argument, flag, environment
variable, or loader exists here or is imported here), and its "real
analysis" mode is not invoked anywhere in the Phase 2B.5A engineering task
-- only `plan_statistical_analysis()` (side-effect-free, identity/manifest
metadata only) is exercised outside tests.

Deliberately NOT implemented (see the freeze doc sec.3 ambiguity gate):
- Any cross-condition "differs between X and Y" test for H1/H2/H3 (the
  frozen plan specifies only the within-cell clean-vs-TTA paired test).
- Any numeric alpha / accept-reject verdict (only CIs and p-values are
  exposed).
- Any pooled-dataset statistic, "meaningful harm" threshold, one-sided
  test, or post-hoc subgroup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.matrix import parse_and_validate_matrix
from when_tta_hurts.validation_evaluation import (
    compute_evaluator_fingerprint,
    resolve_canonical_training_completion,
)

# ---------------------------------------------------------------------------
# Analysis-implementation fingerprint
# ---------------------------------------------------------------------------
#
# Every file whose content could change a REPORTED statistical number
# (formulas, canonical-selection logic, or the frozen dependency lock).
# Deliberately excludes docs/ and ledgers -- a documentation- or
# ledger-only commit must never change this identity, exactly mirroring
# compute_evaluator_fingerprint()'s existing discipline.
ANALYSIS_FINGERPRINT_MANIFEST: tuple[str, ...] = (
    "pyproject.toml",
    "uv.lock",
    "src/when_tta_hurts/artifacts.py",
    "src/when_tta_hurts/config.py",
    "src/when_tta_hurts/matrix.py",
    "src/when_tta_hurts/metrics.py",
    "src/when_tta_hurts/validation_evaluation.py",
    "src/when_tta_hurts/evaluation_result_artifacts.py",
    "src/when_tta_hurts/statistical_analysis.py",
    "src/when_tta_hurts/statistical_analysis_artifacts.py",
)


class AnalysisFingerprintError(RuntimeError):
    """Raised when a file listed in ANALYSIS_FINGERPRINT_MANIFEST is
    missing. Fails closed -- never computes a partial fingerprint."""


def compute_analysis_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = ANALYSIS_FINGERPRINT_MANIFEST,
) -> tuple[str, dict[str, str]]:
    """Deterministic content fingerprint of every analysis-relevant
    tracked file, mirroring compute_evaluator_fingerprint()'s exact
    discipline: STABLE (independent of docs/ledger-only commits),
    computed from actual working-tree bytes, never partial."""
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise AnalysisFingerprintError(
                f"Analysis-fingerprint manifest file missing: {rel_path}. Refusing to compute a "
                f"partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


def compute_analysis_id(
    family: str,
    cell_evaluation_ids: tuple[str, ...],
    analysis_fingerprint: str,
) -> str:
    """Deterministic, STABLE analysis ID: hashes the family name, the
    sorted tuple of every input cell's evaluation_id (never a
    training_run_id alone -- evaluation_id already binds checkpoint +
    evaluator fingerprint + frozen TTA config), and the analysis
    implementation fingerprint. A docs/ledger-only commit changes none of
    these three inputs, so the analysis ID is stable across such commits
    (requirement 16); a change to any analysis-relevant source file
    changes analysis_fingerprint and therefore this ID (requirement 15's
    counterpart)."""
    return config_hash(
        {
            "family": family,
            "cell_evaluation_ids": tuple(sorted(cell_evaluation_ids)),
            "analysis_fingerprint": analysis_fingerprint,
        }
    )


# ---------------------------------------------------------------------------
# Analysis families -- mechanically derived from the frozen matrix
# ---------------------------------------------------------------------------
#
# Per docs/phase2b_statistical_analysis_engineering_freeze.md sec.2:
#   H1 (normalization): Block A cells, compared batchnorm vs groupnorm.
#   H2 (resolution):     Block A cells (28/64 primary) + Block D (128,
#                         trend-only).
#   H3 (policy matching): Block B cells + their Block A unmatched
#                         comparators (policy=none, same dataset/
#                         resolution/normalization/seed).
#   BLOCK_C: positive-control reproduction, not a formal H-test family.
# H4 is intentionally absent: Validation-Gated TTA is draft/not approved,
# per the freeze doc -- there is nothing to derive a family from yet.

KNOWN_FAMILIES: tuple[str, ...] = ("H1", "H2", "H3", "BLOCK_C")


@dataclass(frozen=True)
class FamilyCell:
    run_id: str
    block: str
    dataset: str
    resolution: int
    model: str
    normalization: str
    seed: int
    role: str  # e.g. "primary", "unmatched_comparator" (H3 only)


def _matrix_cells(matrix_path: str = "configs/experiment_matrix.yaml"):
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    return list(expanded.cells)


def derive_family_cells(
    family: str, matrix_path: str = "configs/experiment_matrix.yaml"
) -> tuple[FamilyCell, ...]:
    """Mechanically derive the cells belonging to `family` from the frozen
    matrix -- never hand-typed run-ID lists. Raises ValueError for any
    family not in KNOWN_FAMILIES (in particular, H4 -- draft, not yet
    derivable)."""
    if family not in KNOWN_FAMILIES:
        raise ValueError(
            f"Unknown or not-yet-derivable analysis family: {family!r}. "
            f"Known families: {KNOWN_FAMILIES}. H4 (Validation-Gated TTA) is "
            f"intentionally absent -- draft, not approved, per the frozen "
            f"engineering-mapping document."
        )

    cells = _matrix_cells(matrix_path)
    out: list[FamilyCell] = []

    if family == "H1":
        for c in cells:
            if c.block == "A_core_normalization_resolution":
                out.append(
                    FamilyCell(
                        c.run_id(),
                        c.block,
                        c.dataset,
                        c.resolution,
                        c.model,
                        c.normalization,
                        c.seed,
                        "primary",
                    )
                )
    elif family == "H2":
        for c in cells:
            if c.block in ("A_core_normalization_resolution", "D_conditional_128px"):
                role = "primary" if c.block == "A_core_normalization_resolution" else "trend_only"
                out.append(
                    FamilyCell(
                        c.run_id(), c.block, c.dataset, c.resolution, c.model, c.normalization, c.seed, role
                    )
                )
    elif family == "H3":
        matched = [c for c in cells if c.block == "B_policy_matching"]
        for c in matched:
            out.append(
                FamilyCell(
                    c.run_id(), c.block, c.dataset, c.resolution, c.model, c.normalization, c.seed, "matched"
                )
            )
        # Unmatched comparators: same dataset/resolution/normalization/seed,
        # policy=none, in Block A -- reused per
        # configs/experiment_matrix.yaml's `reuses_checkpoints_from`.
        for m in matched:
            for c in cells:
                if (
                    c.block == "A_core_normalization_resolution"
                    and c.dataset == m.dataset
                    and c.resolution == m.resolution
                    and c.normalization == m.normalization
                    and c.seed == m.seed
                ):
                    out.append(
                        FamilyCell(
                            c.run_id(),
                            c.block,
                            c.dataset,
                            c.resolution,
                            c.model,
                            c.normalization,
                            c.seed,
                            "unmatched_comparator",
                        )
                    )
    elif family == "BLOCK_C":
        for c in cells:
            if c.block == "C_positive_control_reproduction":
                out.append(
                    FamilyCell(
                        c.run_id(),
                        c.block,
                        c.dataset,
                        c.resolution,
                        c.model,
                        c.normalization,
                        c.seed,
                        "primary",
                    )
                )

    return tuple(sorted(out, key=lambda fc: fc.run_id))


# ---------------------------------------------------------------------------
# Plan mode -- side-effect-free, identity/manifest only, never reads a
# scientific metric value.
# ---------------------------------------------------------------------------


def plan_statistical_analysis(
    matrix_path: str = "configs/experiment_matrix.yaml",
    families: tuple[str, ...] = KNOWN_FAMILIES,
) -> dict[str, Any]:
    """SIDE-EFFECT-FREE. Never opens predictions.npz or metrics.json --
    only resolves each family's required cells to their canonical
    training/evaluation identity (run_id, attempt numbers, checkpoint
    hash, evaluation_id, evaluator fingerprint) via the SAME production
    selection logic the evaluation pipeline uses
    (resolve_canonical_training_completion, check_evaluation_skip).
    Reports missing/ambiguous/stale/incompatible cells explicitly rather
    than raising, exactly like plan_validation_evaluation()'s discipline."""
    current_fp, _ = compute_evaluator_fingerprint()
    analysis_fp, _ = compute_analysis_fingerprint()

    report: dict[str, Any] = {
        "current_evaluator_fingerprint": current_fp,
        "analysis_fingerprint": analysis_fp,
        "families": {},
    }

    for family in families:
        try:
            family_cells = derive_family_cells(family, matrix_path)
        except ValueError as e:
            report["families"][family] = {"error": str(e)}
            continue

        cell_reports = []
        eligible_evaluation_ids: list[str] = []
        for fc in family_cells:
            entry: dict[str, Any] = {
                "run_id": fc.run_id,
                "dataset": fc.dataset,
                "resolution": fc.resolution,
                "model": fc.model,
                "normalization": fc.normalization,
                "seed": fc.seed,
                "role": fc.role,
            }
            try:
                _, train_result = resolve_canonical_training_completion(fc.run_id, matrix_path)
                entry["checkpoint_hash"] = train_result.checkpoint_hash
                entry["training_attempt"] = train_result.attempt_number
            except Exception as e:  # noqa: BLE001 -- plan mode reports, never raises
                entry["training_error"] = f"{type(e).__name__}: {e}"
                cell_reports.append(entry)
                continue

            # Resolve the canonical evaluation completion for this run_id
            # purely from ledger/status metadata -- no metric value is
            # ever read here. check_evaluation_skip requires an explicit
            # evaluation_config_hash; we don't have one without building
            # the full ValidationEvaluationConfig, so instead we read the
            # ledger directly for the sole completed, current-fingerprint,
            # non-amendment-excluded row -- same eligibility rule,
            # expressed as a read rather than a recomputation.
            eval_entry = _resolve_canonical_evaluation_identity(fc.run_id, current_fp)
            entry.update(eval_entry)
            if eval_entry.get("evaluation_status") == "eligible":
                eligible_evaluation_ids.append(eval_entry["evaluation_id"])
            cell_reports.append(entry)

        n_eligible = sum(1 for c in cell_reports if c.get("evaluation_status") == "eligible")
        report["families"][family] = {
            "cells": cell_reports,
            "n_cells_required": len(family_cells),
            "n_cells_eligible": n_eligible,
            "complete": n_eligible == len(family_cells),
            "candidate_analysis_id": (
                compute_analysis_id(family, tuple(eligible_evaluation_ids), analysis_fp)
                if n_eligible == len(family_cells)
                else None
            ),
        }

    return report


def _resolve_canonical_evaluation_identity(
    run_id: str,
    current_fp: str,
    ledger_path: str | Path = "artifacts/ledger_validation_evaluation.csv",
    validation_evaluation_root: str | Path = "artifacts/validation_evaluation",
    amendments_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read-only: resolve run_id's sole eligible (current-fingerprint,
    completed, non-amendment-excluded) evaluation-ledger row, WITHOUT
    reading any metrics.json/predictions.npz scientific value. Reports
    'missing', 'stale' (older fingerprint), or 'ambiguous' explicitly
    rather than guessing. All paths are parameterized (defaulting to the
    real production paths) so tests can point this at synthetic
    tmp_path fixtures without ever touching real artifacts."""
    import csv
    import json as _json

    from when_tta_hurts.ledger import EVALUATION_AMENDMENTS_LEDGER_PATH, is_evaluation_canonical_ineligible

    if amendments_ledger_path is None:
        amendments_ledger_path = EVALUATION_AMENDMENTS_LEDGER_PATH

    ledger_path = Path(ledger_path)
    validation_evaluation_root = Path(validation_evaluation_root)
    if not ledger_path.exists():
        return {"evaluation_status": "missing"}

    with ledger_path.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["training_run_id"] == run_id and r["status"] == "completed"]

    eligible_rows = []
    stale_rows = []
    for r in rows:
        attempt = int(r["evaluation_attempt"])
        if is_evaluation_canonical_ineligible(
            r["evaluation_id"], attempt, ledger_path=amendments_ledger_path
        ):
            continue
        # Fingerprint compatibility is checked via the persisted metadata
        # file's evaluator_fingerprint field -- identity metadata only,
        # never a metric value.
        meta_path = validation_evaluation_root / run_id / f"attempt_{attempt:03d}" / "metadata.json"
        if not meta_path.exists():
            continue
        meta = _json.loads(meta_path.read_text())
        if meta.get("evaluator_fingerprint") == current_fp:
            eligible_rows.append((r, attempt))
        else:
            stale_rows.append((r, attempt))

    if len(eligible_rows) == 1:
        r, attempt = eligible_rows[0]
        return {
            "evaluation_status": "eligible",
            "evaluation_id": r["evaluation_id"],
            "evaluation_attempt": attempt,
        }
    if len(eligible_rows) > 1:
        return {
            "evaluation_status": "ambiguous",
            "evaluation_ids": [r["evaluation_id"] for r, _ in eligible_rows],
        }
    if stale_rows:
        return {
            "evaluation_status": "stale",
            "evaluation_ids": [r["evaluation_id"] for r, _ in stale_rows],
        }
    return {"evaluation_status": "missing"}


# ---------------------------------------------------------------------------
# Within-cell statistics -- fully specified, no invention
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    clean_correct: np.ndarray,
    tta_correct: np.ndarray,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Paired bootstrap CI on delta accuracy (TTA - clean), per
    docs/statistical_analysis_plan.md: 'resample test samples with
    replacement, >= 10,000 resamples'. clean_correct/tta_correct must be
    boolean arrays of identical length, paired index-for-index (same
    sample, same model). Fails closed on non-finite/mismatched input."""
    clean_correct = np.asarray(clean_correct, dtype=bool)
    tta_correct = np.asarray(tta_correct, dtype=bool)
    if clean_correct.shape != tta_correct.shape:
        raise ValueError(
            f"clean_correct and tta_correct must be paired (same shape); got "
            f"{clean_correct.shape} vs {tta_correct.shape}."
        )
    if clean_correct.ndim != 1 or clean_correct.size == 0:
        raise ValueError("clean_correct/tta_correct must be non-empty 1-D arrays.")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}.")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}.")

    rng = rng if rng is not None else np.random.default_rng()
    n = clean_correct.size
    point_delta = float(tta_correct.mean() - clean_correct.mean())

    resampled_deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_deltas[i] = tta_correct[idx].mean() - clean_correct[idx].mean()

    alpha = 1.0 - ci_level
    lo, hi = np.percentile(resampled_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta_accuracy": point_delta,
        "ci_level": ci_level,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_resamples": n_resamples,
        "n_samples": n,
    }


def mcnemar_test(clean_correct: np.ndarray, tta_correct: np.ndarray) -> dict[str, Any]:
    """McNemar's test on the paired (clean-correct, TTA-correct) 2x2
    contingency table, per docs/statistical_analysis_plan.md. Uses the
    exact binomial form when the discordant total is small (<25, the
    conventional threshold below which the chi-square approximation is
    unreliable -- see freeze doc sec.3 item 3: the frozen record does not
    pin an exact numeric threshold, so both the raw discordant counts and
    the chosen method are always reported for transparency), and the
    continuity-corrected chi-square form otherwise. Returns
    method='undefined' with p_value=None when there are zero discordant
    pairs (McNemar is degenerate in that case) -- never silently
    substitutes a p-value."""
    clean_correct = np.asarray(clean_correct, dtype=bool)
    tta_correct = np.asarray(tta_correct, dtype=bool)
    if clean_correct.shape != tta_correct.shape:
        raise ValueError("clean_correct and tta_correct must be paired (same shape).")
    if clean_correct.ndim != 1 or clean_correct.size == 0:
        raise ValueError("clean_correct/tta_correct must be non-empty 1-D arrays.")

    # b = clean correct, TTA wrong (harmed); c = clean wrong, TTA correct (rescued)
    b = int(np.sum(clean_correct & ~tta_correct))
    c = int(np.sum(~clean_correct & tta_correct))
    n_discordant = b + c

    if n_discordant == 0:
        return {"b": b, "c": c, "n_discordant": 0, "method": "undefined", "statistic": None, "p_value": None}

    if n_discordant < 25:
        # Exact two-sided binomial test: P(X <= min(b,c)) + P(X >= max(b,c))
        # under Binomial(n_discordant, 0.5).
        k = min(b, c)
        p_value = 0.0
        for x in range(0, k + 1):
            p_value += math.comb(n_discordant, x) * (0.5**n_discordant)
        p_value = min(1.0, 2.0 * p_value)
        return {
            "b": b,
            "c": c,
            "n_discordant": n_discordant,
            "method": "exact_binomial",
            "statistic": None,
            "p_value": p_value,
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = math.erfc(math.sqrt(statistic / 2.0))  # chi-square(1) survival via complementary error function
    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "method": "continuity_corrected_chi_square",
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR-corrected p-values (q-values), preserving
    input order. Does NOT apply any alpha or return accept/reject
    verdicts -- per the freeze doc, no numeric alpha is frozen anywhere;
    only corrected p-values are exposed, exactly as
    docs/statistical_analysis_plan.md requires ('report both corrected
    and uncorrected p-values')."""
    n = len(p_values)
    if n == 0:
        return []
    for p in p_values:
        if not (0.0 <= p <= 1.0) or not math.isfinite(p):
            raise ValueError(f"p-values must be finite and in [0, 1], got {p!r}.")

    order = sorted(range(n), key=lambda i: p_values[i])
    corrected = [0.0] * n
    prev = 1.0
    for rank, i in enumerate(reversed(order), start=1):
        k = n - rank + 1
        q = p_values[i] * n / k
        prev = min(prev, q)
        corrected[i] = prev
    return corrected


def effect_sizes(clean_correct: np.ndarray, tta_correct: np.ndarray) -> dict[str, float]:
    """Effect sizes on the paired correctness arrays. Harm/rescue-rate
    formulas are identical to, and were validated against,
    metrics.py::harm_rescue_rates (which operates on raw logits+labels
    rather than pre-computed correctness booleans -- the two share no
    code, satisfying the freeze doc's 'do not define the implementation
    and its test oracle with the same helper' requirement; both are
    exercised against the same hand-calculated examples in this module's
    test suite)."""
    clean_correct = np.asarray(clean_correct, dtype=bool)
    tta_correct = np.asarray(tta_correct, dtype=bool)
    n_clean_correct = int(clean_correct.sum())
    n_clean_wrong = int((~clean_correct).sum())
    harmed = int(np.sum(clean_correct & ~tta_correct))
    rescued = int(np.sum(~clean_correct & tta_correct))
    return {
        "delta_accuracy": float(tta_correct.mean() - clean_correct.mean()),
        "harm_rate": (harmed / n_clean_correct) if n_clean_correct > 0 else 0.0,
        "rescue_rate": (rescued / n_clean_wrong) if n_clean_wrong > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Real analysis mode -- implemented per Part E's requirement, but NEVER
# invoked anywhere in the Phase 2B.5A engineering task. Only exercised by
# this module's own tests against synthetic/temporary fixtures.
# ---------------------------------------------------------------------------


class AnalysisInputError(RuntimeError):
    """Raised when a family's inputs are missing, ambiguous, stale,
    corrupt, or otherwise incompatible with real analysis. Fails closed
    -- never silently substitutes a different cell or condition."""


def compute_family_analysis(
    family: str,
    condition: str = "naive_tta",
    aggregator: str = "mean_probability",
    n: int = 50,
    matrix_path: str = "configs/experiment_matrix.yaml",
    validation_evaluation_root: str | Path = "artifacts/validation_evaluation",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compute the fully-specified within-cell statistics (paired
    bootstrap CI, McNemar, effect sizes) for every cell in `family`, at
    the given condition/aggregator/N (default: the frozen primary
    N=50/mean_probability). Reads ONLY `artifacts/validation_evaluation/`
    predictions.npz/metrics.json (never a test-split path -- no such
    argument exists on this function or anywhere it calls). Requires ALL
    expected cells for the family to be eligible (current fingerprint,
    non-amendment-excluded, manifest-valid); raises AnalysisInputError on
    any missing/ambiguous/stale/corrupt cell rather than silently
    proceeding with a partial family. Never invoked outside this module's
    own tests during the Phase 2B.5A engineering task."""
    import json as _json

    import numpy as _np

    from when_tta_hurts.evaluation_result_artifacts import verify_evaluation_artifact_manifest

    current_fp, _ = compute_evaluator_fingerprint()
    analysis_fp, _ = compute_analysis_fingerprint()

    family_cells = derive_family_cells(family, matrix_path)
    per_cell_statistics: dict[str, Any] = {}
    evaluation_ids: list[str] = []
    input_manifest: dict[str, Any] = {}

    for fc in family_cells:
        identity = _resolve_canonical_evaluation_identity(fc.run_id, current_fp)
        status = identity.get("evaluation_status")
        if status != "eligible":
            raise AnalysisInputError(
                f"Cell {fc.run_id!r} is not eligible for real analysis (status={status!r}). "
                f"Refusing to compute a partial family result."
            )
        attempt = identity["evaluation_attempt"]
        attempt_dir = Path(validation_evaluation_root) / fc.run_id / f"attempt_{attempt:03d}"

        manifest = _json.loads((attempt_dir / "artifact_manifest.json").read_text())
        verify_evaluation_artifact_manifest(attempt_dir, manifest)

        metadata = _json.loads((attempt_dir / "metadata.json").read_text())
        if metadata.get("evaluator_fingerprint") != current_fp:
            raise AnalysisInputError(
                f"Cell {fc.run_id!r} metadata fingerprint does not match current fingerprint -- "
                f"stale input, refusing to use it."
            )

        predictions = dict(_np.load(attempt_dir / "predictions.npz"))
        metrics = _json.loads((attempt_dir / "metrics.json").read_text())

        labels = predictions["labels"]
        clean_probs = predictions["clean_probs"]
        clean_correct = clean_probs.argmax(axis=-1) == labels

        if condition == "naive_tta":
            view_probs = predictions["view_probs"][:n]
            agg_probs = view_probs.mean(axis=0) if aggregator == "mean_probability" else None
            if agg_probs is None:
                raise AnalysisInputError(
                    f"Aggregator {aggregator!r} not implemented in compute_family_analysis -- "
                    f"only 'mean_probability' is wired up for this engineering pass."
                )
            tta_correct = agg_probs.argmax(axis=-1) == labels
        else:
            raise AnalysisInputError(
                f"Condition {condition!r} not implemented in compute_family_analysis -- "
                f"only 'naive_tta' is wired up for this engineering pass."
            )

        bootstrap = paired_bootstrap_ci(clean_correct, tta_correct, rng=rng)
        mcnemar = mcnemar_test(clean_correct, tta_correct)
        effects = effect_sizes(clean_correct, tta_correct)

        per_cell_statistics[fc.run_id] = {
            "bootstrap": bootstrap,
            "mcnemar": mcnemar,
            "effect_sizes": effects,
            "n_samples": int(labels.shape[0]),
            "reported_delta_accuracy": metrics["conditions"]["naive_tta"][aggregator][str(n)][
                "delta_accuracy"
            ],
        }
        evaluation_ids.append(identity["evaluation_id"])
        input_manifest[fc.run_id] = {
            "evaluation_id": identity["evaluation_id"],
            "evaluation_attempt": attempt,
            "checkpoint_hash": metadata["checkpoint_hash"],
            "artifact_manifest_sha256s": {a["path"]: a["sha256"] for a in manifest["artifacts"]},
        }

    raw_p_values = [per_cell_statistics[fc.run_id]["mcnemar"]["p_value"] for fc in family_cells]
    definable = [p for p in raw_p_values if p is not None]
    corrected = benjamini_hochberg(definable) if definable else []
    corrected_iter = iter(corrected)
    corrected_p_values = [None if p is None else next(corrected_iter) for p in raw_p_values]

    analysis_id = compute_analysis_id(family, tuple(sorted(evaluation_ids)), analysis_fp)

    return {
        "family": family,
        "analysis_id": analysis_id,
        "analysis_fingerprint": analysis_fp,
        "current_evaluator_fingerprint": current_fp,
        "condition": condition,
        "aggregator": aggregator,
        "n": n,
        "cells": [fc.run_id for fc in family_cells],
        "input_manifest": input_manifest,
        "per_cell_statistics": per_cell_statistics,
        "multiplicity": {
            "family": family,
            "raw_p_values": raw_p_values,
            "corrected_p_values": corrected_p_values,
            "method": "benjamini_hochberg",
        },
        "status": "completed",
        "test_split_accessed": False,
    }
