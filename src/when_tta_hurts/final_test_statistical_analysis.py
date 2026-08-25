"""Phase 2B.7A: additive, final-test-only statistical-analysis and
cross-condition-addendum runner.

Discovered missing in Phase 2B.6O Part C: `statistical_analysis.py` and
`cross_condition_addendum.py` were engineered (Phase 2B.5A/2B.5C) before
the final-test matrix existed, and read exclusively from
`artifacts/validation_evaluation/` / `artifacts/ledger_validation_evaluation.csv`.
This module adds the final-test-aware counterpart WITHOUT modifying
either of those files or their defaults -- see
docs/phase2b_final_test_statistical_analysis_engineering_freeze.md.

Reuses the frozen mathematics (paired_bootstrap_ci, mcnemar_test,
benjamini_hochberg, effect_sizes, did_point_estimate, did_bootstrap_ci,
derive_bootstrap_seed, derive_family_cells, derive_fixed_pairs,
load_addendum_spec, compute_analysis_id) UNCHANGED, imported directly
from statistical_analysis.py / cross_condition_addendum.py. No formula,
threshold, resample count, or CI level is reimplemented here.

Plan mode (`plan_final_test_statistical_analysis`,
`plan_final_test_cross_condition_addendum`) is side-effect-free and never
calls `numpy.load` or opens `metrics.json` -- only git/ledger/manifest
metadata. Real analysis
(`compute_final_test_family_analysis`, `compute_final_test_hypothesis_did`)
is implemented per the frozen spec but is NEVER invoked outside this
module's own tests during the Phase 2B.7A engineering task.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.cross_condition_addendum import (
    KNOWN_HYPOTHESES,
    AddendumInputError,
    AddendumSpecError,
    FixedPair,
    derive_bootstrap_seed,
    derive_fixed_pairs,
    did_bootstrap_ci,
    load_addendum_spec,
)
from when_tta_hurts.final_test_analysis_ledger import (
    append_final_test_analysis_entry,
    ensure_final_test_analysis_ledger_exists,
    existing_completed_attempt,
    next_final_test_analysis_attempt_number,
)
from when_tta_hurts.final_test_authorization import (
    FinalTestAuthorization,
    FinalTestAuthorizationError,
    verify_final_test_authorization,
)
from when_tta_hurts.final_test_evaluation import DEFAULT_FINAL_TEST_ROOT
from when_tta_hurts.final_test_identity import FINAL_TEST_RUNNER_MANIFEST
from when_tta_hurts.final_test_result_artifacts import verify_final_test_artifact_manifest
from when_tta_hurts.ledger import FINAL_TEST_LEDGER_PATH
from when_tta_hurts.statistical_analysis import (
    KNOWN_FAMILIES,
    benjamini_hochberg,
    compute_analysis_id,
    derive_family_cells,
    effect_sizes,
    mcnemar_test,
    paired_bootstrap_ci,
)
from when_tta_hurts.statistical_analysis_artifacts import (
    persist_and_verify_analysis_completion,
    persist_and_verify_cross_condition_completion,
)

# ---------------------------------------------------------------------------
# Fingerprint -- additive, disjoint from ANALYSIS_FINGERPRINT_MANIFEST and
# CROSS_CONDITION_ADDENDUM_MANIFEST (this module never redefines those).
# ---------------------------------------------------------------------------

FINAL_TEST_ANALYSIS_ROOT = Path("artifacts/final_test_analysis")
FINAL_TEST_CROSS_CONDITION_ROOT = Path("artifacts/final_test_cross_condition")

FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST: tuple[str, ...] = FINAL_TEST_RUNNER_MANIFEST + (
    "src/when_tta_hurts/final_test_statistical_analysis.py",
    "src/when_tta_hurts/final_test_analysis_ledger.py",
)


class FinalTestAnalysisFingerprintError(RuntimeError):
    """Raised when a file listed in FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST
    is missing. Fails closed -- never computes a partial fingerprint."""


def compute_final_test_analysis_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST,
) -> tuple[str, dict[str, str]]:
    """Deterministic content fingerprint of every file that could change a
    reported final-test-analysis number, mirroring every other fingerprint
    in this repo's exact discipline (stable across docs/ledger-only
    commits, fails closed on a missing manifest file)."""
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise FinalTestAnalysisFingerprintError(
                f"Final-test-analysis fingerprint manifest file missing: {rel_path}. Refusing to "
                f"compute a partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


# ---------------------------------------------------------------------------
# Cell 1's real, historical (generation-3) authorization binding. See
# docs/phase2b_final_test_matrix_closure_audit.md sec.5 and
# docs/phase2b_final_test_statistical_analysis_engineering_freeze.md
# sec.7. Every OTHER cell must bind to the CURRENT authorization's
# artifact_sha256 -- this allow-list is deliberately narrow (one entry)
# and is never consulted for any run_id not explicitly listed here.
# ---------------------------------------------------------------------------

KNOWN_HISTORICAL_AUTHORIZATION_SHA256_BY_RUN_ID: dict[str, str] = {
    "A-pathmnist-28px-batchnorm-policy-none-s0": (
        "0332f696bea36ea92c45a3691147337a351c1990584c71a1cccb4da8b494343a"
    ),
}


class FinalTestAnalysisInputError(RuntimeError):
    """Raised when a final-test cell/pair's inputs are missing, ambiguous,
    stale, unauthorized, or otherwise incompatible with real analysis.
    Fails closed -- never silently substitutes a different cell/attempt or
    proceeds with a partial family/hypothesis."""


# ---------------------------------------------------------------------------
# Final-test-only canonical identity resolution -- read-only, no
# predictions.npz/metrics.json access. Requires an already-verified
# FinalTestAuthorization (never re-verifies authorization per cell).
# ---------------------------------------------------------------------------


def _final_test_ledger_row_for_attempt(
    run_id: str, attempt: int, ledger_path: str | Path
) -> dict[str, Any] | None:
    path = Path(ledger_path)
    if not path.exists():
        return None
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("training_run_id") == run_id and row.get("evaluation_attempt") == str(attempt):
                return row
    return None


def resolve_final_test_canonical_evaluation_identity(
    run_id: str,
    authorization: FinalTestAuthorization,
    final_test_ledger_path: str | Path = FINAL_TEST_LEDGER_PATH,
) -> dict[str, Any]:
    """Read-only: resolve run_id's sole eligible final-test completion,
    using ONLY the already-verified `authorization` object's own
    classification (never re-derives classification, never re-verifies
    authorization). Reports 'not_completed_consumed' or
    'unknown_run_id' explicitly rather than raising -- mirrors
    _resolve_canonical_evaluation_identity()'s plan-mode discipline.
    Never opens predictions.npz/metrics.json."""
    entry = authorization.authorized_cells_by_run_id.get(run_id)
    if entry is None:
        return {"evaluation_status": "unknown_run_id"}
    classification = authorization.cell_classifications.get(run_id)
    if classification != "completed_consumed":
        return {"evaluation_status": "not_completed_consumed", "classification": classification}

    attempt = entry["authorized_final_test_attempt"]
    row = _final_test_ledger_row_for_attempt(run_id, attempt, final_test_ledger_path)
    if row is None:
        return {"evaluation_status": "not_completed_consumed", "classification": classification}

    expected_sha256 = KNOWN_HISTORICAL_AUTHORIZATION_SHA256_BY_RUN_ID.get(
        run_id, authorization.artifact_sha256
    )
    if row.get("authorization_artifact_sha256") != expected_sha256:
        return {
            "evaluation_status": "unauthorized_binding",
            "expected_authorization_artifact_sha256": expected_sha256,
            "actual_authorization_artifact_sha256": row.get("authorization_artifact_sha256"),
        }

    return {
        "evaluation_status": "eligible",
        "evaluation_id": row["final_test_evaluation_id"],
        "evaluation_attempt": attempt,
        "authorization_artifact_sha256": expected_sha256,
    }


# ---------------------------------------------------------------------------
# Plan mode -- preregistered within-cell families. Side-effect-free,
# never opens predictions.npz/metrics.json.
# ---------------------------------------------------------------------------


def plan_final_test_statistical_analysis(
    matrix_path: str = "configs/experiment_matrix.yaml",
    families: tuple[str, ...] = KNOWN_FAMILIES,
) -> dict[str, Any]:
    """SIDE-EFFECT-FREE. Never opens predictions.npz or metrics.json.
    Verifies final-test authorization (git/ledger/manifest metadata only)
    and resolves each family's required cells to final-test canonical
    identity. Reports authorization failure explicitly rather than
    raising, exactly like every other plan-mode function in this repo."""
    try:
        authorization = verify_final_test_authorization()
    except FinalTestAuthorizationError as e:
        return {"authorization_status": "not_approved", "error": str(e), "families": {}}

    final_test_analysis_fp, _ = compute_final_test_analysis_fingerprint()

    n_completed_consumed = sum(
        1 for v in authorization.cell_classifications.values() if v == "completed_consumed"
    )
    n_pending = sum(1 for v in authorization.cell_classifications.values() if v == "pending")
    n_invalid = sum(1 for v in authorization.cell_classifications.values() if v == "invalid")

    report: dict[str, Any] = {
        "authorization_status": "approved",
        "authorization_artifact_sha256": authorization.artifact_sha256,
        "authorization_commit": authorization.authorization_commit,
        "final_test_analysis_fingerprint": final_test_analysis_fp,
        "n_cells_total": len(authorization.cell_classifications),
        "n_completed_consumed": n_completed_consumed,
        "n_pending": n_pending,
        "n_invalid": n_invalid,
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
            identity = resolve_final_test_canonical_evaluation_identity(fc.run_id, authorization)
            entry: dict[str, Any] = {"run_id": fc.run_id, "role": fc.role}
            entry.update(identity)
            if identity.get("evaluation_status") == "eligible":
                eligible_evaluation_ids.append(identity["evaluation_id"])
            cell_reports.append(entry)

        n_eligible = sum(1 for c in cell_reports if c.get("evaluation_status") == "eligible")
        report["families"][family] = {
            "cells": cell_reports,
            "n_cells_required": len(family_cells),
            "n_cells_eligible": n_eligible,
            "complete": n_eligible == len(family_cells),
            "candidate_final_test_analysis_id": (
                compute_analysis_id(family, tuple(sorted(eligible_evaluation_ids)), final_test_analysis_fp)
                if n_eligible == len(family_cells)
                else None
            ),
        }

    return report


# ---------------------------------------------------------------------------
# Plan mode -- secondary cross-condition addendum.
# ---------------------------------------------------------------------------


def plan_final_test_cross_condition_addendum(
    matrix_path: str = "configs/experiment_matrix.yaml",
    hypotheses: tuple[str, ...] = KNOWN_HYPOTHESES,
) -> dict[str, Any]:
    """SIDE-EFFECT-FREE. Never opens predictions.npz or metrics.json.
    Mirrors plan_final_test_statistical_analysis()'s discipline for the
    fixed-pair cross-condition addendum."""
    try:
        authorization = verify_final_test_authorization()
    except FinalTestAuthorizationError as e:
        return {"authorization_status": "not_approved", "error": str(e), "hypotheses": {}}

    try:
        spec = load_addendum_spec()
    except AddendumSpecError as e:
        return {"authorization_status": "approved", "error": str(e), "hypotheses": {}}

    final_test_analysis_fp, _ = compute_final_test_analysis_fingerprint()

    report: dict[str, Any] = {
        "authorization_status": "approved",
        "authorization_artifact_sha256": authorization.artifact_sha256,
        "final_test_analysis_fingerprint": final_test_analysis_fp,
        "hypotheses": {},
    }

    for hypothesis in hypotheses:
        pairs = derive_fixed_pairs(hypothesis, matrix_path, spec)
        pair_reports = []
        n_eligible = 0
        for p in pairs:
            id_a = resolve_final_test_canonical_evaluation_identity(p.condition_a_run_id, authorization)
            id_b = resolve_final_test_canonical_evaluation_identity(p.condition_b_run_id, authorization)
            entry: dict[str, Any] = {
                "pair_id": p.pair_id,
                "condition_a_run_id": p.condition_a_run_id,
                "condition_b_run_id": p.condition_b_run_id,
                "condition_a_status": id_a.get("evaluation_status"),
                "condition_b_status": id_b.get("evaluation_status"),
            }
            eligible = (
                id_a.get("evaluation_status") == "eligible" and id_b.get("evaluation_status") == "eligible"
            )
            entry["eligible"] = eligible
            pair_reports.append(entry)
            if eligible:
                n_eligible += 1

        report["hypotheses"][hypothesis] = {
            "pairs": pair_reports,
            "n_pairs_required": len(pairs),
            "n_pairs_eligible": n_eligible,
            "complete": n_eligible == len(pairs),
        }

    return report


# ---------------------------------------------------------------------------
# Real analysis -- preregistered within-cell families. Implemented per the
# frozen spec but NEVER invoked outside this module's own tests during the
# Phase 2B.7A engineering task.
# ---------------------------------------------------------------------------


def _load_final_test_cell_correctness(
    run_id: str,
    authorization: FinalTestAuthorization,
    final_test_root: str | Path,
    n: int,
    final_test_ledger_path: str | Path = FINAL_TEST_LEDGER_PATH,
) -> dict[str, Any]:
    """Read-only: resolve run_id's final-test canonical completion, verify
    its artifact manifest, and compute clean/TTA-correct boolean arrays
    plus aligned labels/sample_indices. Raises FinalTestAnalysisInputError
    on anything other than a clean, verified, authorized completion."""
    identity = resolve_final_test_canonical_evaluation_identity(run_id, authorization, final_test_ledger_path)
    status = identity.get("evaluation_status")
    if status != "eligible":
        raise FinalTestAnalysisInputError(
            f"Cell {run_id!r} is not eligible for final-test analysis (status={status!r})."
        )
    attempt = identity["evaluation_attempt"]
    attempt_dir = Path(final_test_root) / run_id / f"attempt_{attempt:03d}"

    manifest = json.loads((attempt_dir / "artifact_manifest.json").read_text())
    verify_final_test_artifact_manifest(attempt_dir, manifest)

    metadata = json.loads((attempt_dir / "metadata.json").read_text())

    predictions = dict(np.load(attempt_dir / "predictions.npz"))
    labels = predictions["labels"]
    sample_indices = predictions["sample_indices"]
    clean_probs = predictions["clean_probs"]
    view_probs = predictions["view_probs"][:n]
    agg_probs = view_probs.mean(axis=0)

    clean_correct = clean_probs.argmax(axis=-1) == labels
    tta_correct = agg_probs.argmax(axis=-1) == labels

    return {
        "run_id": run_id,
        "evaluation_id": identity["evaluation_id"],
        "evaluation_attempt": attempt,
        "checkpoint_hash": metadata["checkpoint_hash"],
        "labels": labels,
        "sample_indices": sample_indices,
        "clean_correct": clean_correct,
        "tta_correct": tta_correct,
        "artifact_manifest_sha256s": {a["path"]: a["sha256"] for a in manifest["artifacts"]},
    }


def compute_final_test_family_analysis(
    family: str,
    condition: str = "naive_tta",
    aggregator: str = "mean_probability",
    n: int = 50,
    matrix_path: str = "configs/experiment_matrix.yaml",
    final_test_root: str | Path = DEFAULT_FINAL_TEST_ROOT,
    final_test_ledger_path: str | Path = FINAL_TEST_LEDGER_PATH,
    rng: np.random.Generator | None = None,
    persist: bool = True,
    _authorization: FinalTestAuthorization | None = None,
) -> dict[str, Any]:
    """Compute the fully-specified within-cell statistics (paired
    bootstrap CI, McNemar, effect sizes) for every cell in `family`, at
    the frozen primary N=50/mean_probability naive_tta condition, reading
    ONLY `artifacts/final_test/` (never validation artifacts). Requires
    final-test authorization approved and ALL family cells
    completed_consumed; raises FinalTestAnalysisInputError on any
    ineligible cell rather than proceeding with a partial family. On
    success, persists atomically and appends one ledger row (idempotent:
    a prior identical completion short-circuits without recomputation).
    `_authorization` is a test-only seam (an already-built
    FinalTestAuthorization, bypassing a real verify_final_test_authorization()
    call) -- production code never passes it. Never invoked outside this
    module's own tests during Phase 2B.7A."""
    if condition != "naive_tta" or aggregator != "mean_probability":
        raise FinalTestAnalysisInputError(
            f"Only condition='naive_tta', aggregator='mean_probability' are wired up "
            f"(got condition={condition!r}, aggregator={aggregator!r})."
        )

    authorization = _authorization if _authorization is not None else verify_final_test_authorization()
    final_test_analysis_fp, _ = compute_final_test_analysis_fingerprint()

    family_cells = derive_family_cells(family, matrix_path)

    # Cheap pre-pass: resolve every cell's canonical identity (metadata
    # only, no prediction load) so the analysis_id can be computed and an
    # idempotency check performed BEFORE any expensive/real computation.
    evaluation_ids: list[str] = []
    for fc in family_cells:
        identity = resolve_final_test_canonical_evaluation_identity(
            fc.run_id, authorization, final_test_ledger_path
        )
        if identity.get("evaluation_status") != "eligible":
            raise FinalTestAnalysisInputError(
                f"Cell {fc.run_id!r} is not eligible for final-test analysis "
                f"(status={identity.get('evaluation_status')!r})."
            )
        evaluation_ids.append(identity["evaluation_id"])

    analysis_id = compute_analysis_id(family, tuple(sorted(evaluation_ids)), final_test_analysis_fp)

    existing_attempt = existing_completed_attempt(analysis_id)
    if existing_attempt is not None:
        attempt_dir = FINAL_TEST_ANALYSIS_ROOT / family / f"attempt_{existing_attempt:03d}"
        return json.loads((attempt_dir / "analysis_result.json").read_text())

    per_cell_statistics: dict[str, Any] = {}
    for fc in family_cells:
        cell = _load_final_test_cell_correctness(
            fc.run_id, authorization, final_test_root, n, final_test_ledger_path
        )
        bootstrap = paired_bootstrap_ci(cell["clean_correct"], cell["tta_correct"], rng=rng)
        mcnemar = mcnemar_test(cell["clean_correct"], cell["tta_correct"])
        effects = effect_sizes(cell["clean_correct"], cell["tta_correct"])
        per_cell_statistics[fc.run_id] = {
            "bootstrap": bootstrap,
            "mcnemar": mcnemar,
            "effect_sizes": effects,
            "n_samples": int(cell["labels"].shape[0]),
        }

    raw_p_values = [per_cell_statistics[fc.run_id]["mcnemar"]["p_value"] for fc in family_cells]
    definable = [p for p in raw_p_values if p is not None]
    corrected = benjamini_hochberg(definable) if definable else []
    corrected_iter = iter(corrected)
    corrected_p_values = [None if p is None else next(corrected_iter) for p in raw_p_values]

    result: dict[str, Any] = {
        "family": family,
        "analysis_id": analysis_id,
        "analysis_fingerprint": final_test_analysis_fp,
        "current_evaluator_fingerprint": authorization.evaluator_fingerprint,
        "cells": [fc.run_id for fc in family_cells],
        "per_cell_statistics": per_cell_statistics,
        "multiplicity": {
            "method": "benjamini_hochberg",
            "raw_p_values": raw_p_values,
            "corrected_p_values": corrected_p_values,
        },
        "status": "completed",
        "test_split_accessed": False,
    }

    if persist:
        _persist_final_test_family_analysis(result, authorization, final_test_analysis_fp)

    return result


def _persist_final_test_family_analysis(
    result: dict[str, Any], authorization: FinalTestAuthorization, final_test_analysis_fp: str
) -> dict[str, Any]:
    ensure_final_test_analysis_ledger_exists()
    analysis_id = result["analysis_id"]
    attempt = next_final_test_analysis_attempt_number(analysis_id)
    attempt_dir = FINAL_TEST_ANALYSIS_ROOT / result["family"] / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    manifest = persist_and_verify_analysis_completion(attempt_dir, result=result)
    ended_at = time.time()

    primary_artifact_hash = hash_file(attempt_dir / "analysis_result.json")
    append_final_test_analysis_entry(
        analysis_id=analysis_id,
        kind="family",
        identifier=result["family"],
        analysis_attempt=attempt,
        final_test_analysis_fingerprint=final_test_analysis_fp,
        final_test_authorization_sha256=authorization.artifact_sha256,
        final_test_authorization_commit=authorization.authorization_commit,
        current_evaluator_fingerprint=authorization.evaluator_fingerprint,
        status="completed",
        primary_artifact_hash=primary_artifact_hash,
        started_at=started_at,
        ended_at=ended_at,
        runtime_seconds=ended_at - started_at,
    )
    return manifest


# ---------------------------------------------------------------------------
# Real analysis -- secondary cross-condition addendum.
# ---------------------------------------------------------------------------


def compute_final_test_pair_did(
    pair: FixedPair,
    authorization: FinalTestAuthorization,
    analysis_fingerprint: str,
    n: int = 50,
    n_resamples: int = 10_000,
    ci_level: float = 0.95,
    final_test_root: str | Path = DEFAULT_FINAL_TEST_ROOT,
    final_test_ledger_path: str | Path = FINAL_TEST_LEDGER_PATH,
) -> dict[str, Any]:
    """Compute one fixed pair's DiD point estimate and bootstrap CI from
    final-test artifacts only. Hard-fails on label mismatch, sample-index
    mismatch, or ineligible/unauthorized inputs for either condition.
    Never invoked outside this module's own tests during Phase 2B.7A."""
    a = _load_final_test_cell_correctness(
        pair.condition_a_run_id, authorization, final_test_root, n, final_test_ledger_path
    )
    b = _load_final_test_cell_correctness(
        pair.condition_b_run_id, authorization, final_test_root, n, final_test_ledger_path
    )

    if a["labels"].shape != b["labels"].shape or not np.array_equal(a["labels"], b["labels"]):
        raise AddendumInputError(
            f"Pair {pair.pair_id!r}: label mismatch between {pair.condition_a_run_id!r} and "
            f"{pair.condition_b_run_id!r} -- refusing to compute a DiD on misaligned test images."
        )
    if a["sample_indices"].shape != b["sample_indices"].shape or not np.array_equal(
        a["sample_indices"], b["sample_indices"]
    ):
        raise AddendumInputError(
            f"Pair {pair.pair_id!r}: sample-index mismatch between {pair.condition_a_run_id!r} and "
            f"{pair.condition_b_run_id!r} -- refusing to compute a DiD on misaligned test images."
        )

    seed = derive_bootstrap_seed(pair.hypothesis, pair.pair_id, analysis_fingerprint)
    bootstrap = did_bootstrap_ci(
        a["clean_correct"],
        a["tta_correct"],
        b["clean_correct"],
        b["tta_correct"],
        n_resamples=n_resamples,
        ci_level=ci_level,
        seed=seed,
    )

    return {
        "pair_id": pair.pair_id,
        "hypothesis": pair.hypothesis,
        "condition_a": {"run_id": a["run_id"], "evaluation_id": a["evaluation_id"]},
        "condition_b": {"run_id": b["run_id"], "evaluation_id": b["evaluation_id"]},
        "bootstrap": bootstrap,
        "n_samples": int(a["labels"].shape[0]),
    }


def compute_final_test_hypothesis_did(
    hypothesis: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    final_test_root: str | Path = DEFAULT_FINAL_TEST_ROOT,
    final_test_ledger_path: str | Path = FINAL_TEST_LEDGER_PATH,
    persist: bool = True,
    _authorization: FinalTestAuthorization | None = None,
) -> dict[str, Any]:
    """Compute every eligible fixed pair's DiD for `hypothesis` from
    final-test artifacts only. Requires ALL expected pairs to be eligible;
    raises FinalTestAnalysisInputError on any missing/ambiguous/
    unauthorized/mismatched pair rather than proceeding with a partial
    result. Never pools pairs into a single p-value or model-population
    verdict. `_authorization` is a test-only seam, mirroring
    compute_final_test_family_analysis()'s. Never invoked outside this
    module's own tests during Phase 2B.7A."""
    authorization = _authorization if _authorization is not None else verify_final_test_authorization()
    spec = load_addendum_spec()
    final_test_analysis_fp, _ = compute_final_test_analysis_fingerprint()

    pairs = derive_fixed_pairs(hypothesis, matrix_path, spec)
    bootstrap_spec = spec["bootstrap"]

    # Cheap pre-check: the analysis_id for a cross-condition result is a
    # pure function of the pair-id set (never of predictions), so an
    # idempotency check can run BEFORE any real per-pair computation.
    analysis_id = compute_analysis_id(
        f"cross_condition_{hypothesis}", tuple(sorted(p.pair_id for p in pairs)), final_test_analysis_fp
    )
    existing_attempt = existing_completed_attempt(analysis_id)
    if existing_attempt is not None:
        attempt_dir = FINAL_TEST_CROSS_CONDITION_ROOT / hypothesis / f"attempt_{existing_attempt:03d}"
        return json.loads((attempt_dir / "cross_condition_result.json").read_text())

    per_pair_results: dict[str, Any] = {}
    for p in pairs:
        per_pair_results[p.pair_id] = compute_final_test_pair_did(
            p,
            authorization,
            final_test_analysis_fp,
            n=spec["endpoint"]["tta_view_count"],
            n_resamples=bootstrap_spec["n_resamples"],
            ci_level=bootstrap_spec["ci_level"],
            final_test_root=final_test_root,
            final_test_ledger_path=final_test_ledger_path,
        )

    result: dict[str, Any] = {
        "classification": "post_validation_pre_test_secondary",
        "hypothesis": hypothesis,
        "cross_condition_analysis_fingerprint": final_test_analysis_fp,
        "current_evaluator_fingerprint": authorization.evaluator_fingerprint,
        "pairs": [p.pair_id for p in pairs],
        "per_pair_results": per_pair_results,
        "status": "completed",
        "test_split_accessed": False,
    }

    if persist:
        _persist_final_test_cross_condition_analysis(result, authorization, final_test_analysis_fp)

    return result


def _persist_final_test_cross_condition_analysis(
    result: dict[str, Any], authorization: FinalTestAuthorization, final_test_analysis_fp: str
) -> dict[str, Any]:
    ensure_final_test_analysis_ledger_exists()
    hypothesis = result["hypothesis"]
    analysis_id = compute_analysis_id(
        f"cross_condition_{hypothesis}",
        tuple(sorted(p["pair_id"] for p in result["per_pair_results"].values())),
        final_test_analysis_fp,
    )
    attempt = next_final_test_analysis_attempt_number(analysis_id)
    attempt_dir = FINAL_TEST_CROSS_CONDITION_ROOT / hypothesis / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    manifest = persist_and_verify_cross_condition_completion(attempt_dir, result=result)
    ended_at = time.time()

    primary_artifact_hash = hash_file(attempt_dir / "cross_condition_result.json")
    append_final_test_analysis_entry(
        analysis_id=analysis_id,
        kind="cross_condition",
        identifier=hypothesis,
        analysis_attempt=attempt,
        final_test_analysis_fingerprint=final_test_analysis_fp,
        final_test_authorization_sha256=authorization.artifact_sha256,
        final_test_authorization_commit=authorization.authorization_commit,
        current_evaluator_fingerprint=authorization.evaluator_fingerprint,
        status="completed",
        primary_artifact_hash=primary_artifact_hash,
        started_at=started_at,
        ended_at=ended_at,
        runtime_seconds=ended_at - started_at,
    )
    return manifest


__all__ = [
    "FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST",
    "FinalTestAnalysisFingerprintError",
    "FinalTestAnalysisInputError",
    "KNOWN_HISTORICAL_AUTHORIZATION_SHA256_BY_RUN_ID",
    "compute_final_test_analysis_fingerprint",
    "resolve_final_test_canonical_evaluation_identity",
    "plan_final_test_statistical_analysis",
    "plan_final_test_cross_condition_addendum",
    "compute_final_test_family_analysis",
    "compute_final_test_pair_did",
    "compute_final_test_hypothesis_did",
]
