"""Phase 2B.6K Part B: deterministic, offline reconciliation of the 39
validation evaluations' `original_anchored_tta` metrics after the
shared aggregation-contract correction
(docs/phase2b_final_test_semantic_metric_contract_freeze.md,
docs/phase2b_shared_aggregation_fingerprint_cascade_freeze.md).

Reads ONLY already-persisted predictions/metadata/metrics for each
canonical, pre-fix-fingerprint completed validation evaluation -- NEVER
initializes MPS, loads a checkpoint, loads a dataset, generates views,
or performs inference. Recomputes corrected `original_anchored_tta`
metrics directly from persisted `clean_probs`/`view_probs`,
independently reverifies every OTHER (unaffected) persisted metric
still matches exactly, and writes an APPEND-ONLY reconciliation record.
Never rewrites the original ledger, metrics.json, predictions.npz,
metadata.json, or artifact_manifest.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts import ledger as ledger_module
from when_tta_hurts.artifacts import atomic_write_json, hash_file
from when_tta_hurts.evaluation_result_artifacts import (
    ALL_REQUIRED_EVALUATION_ARTIFACT_FILENAMES,
    EvaluationPersistenceError,
    verify_evaluation_artifact_manifest,
)
from when_tta_hurts.ledger import EVALUATION_AMENDMENTS_LEDGER_PATH, VALIDATION_EVALUATION_LEDGER_PATH
from when_tta_hurts.metrics import compute_metrics_from_probabilities
from when_tta_hurts.validation_evaluation import (
    AGGREGATORS,
    DEFAULT_EVALUATION_ROOT,
    PREFIX_SEQUENCE,
    _recompute_all_conditions_from_predictions,
    compute_evaluator_fingerprint,
)

# The evaluator_fingerprint value recorded by every canonical validation
# evaluation BEFORE the Phase 2B.6J/K shared-aggregation correction --
# confirmed (Phase 2B.6J) as the exact value present in every affected
# cell's metadata.json prior to this fix, and independently reconfirmed
# in cell A-pathmnist-28px-batchnorm-policy-none-s0's attempt 5
# metadata.json. Reconciliation targets ONLY rows bound to this exact
# prior fingerprint -- never a guess, never "any non-current fingerprint".
OLD_EVALUATOR_FINGERPRINT = "7fdce1db496ffb149e2adf608e5b811f8b9dfb5f944daa3462cda1b5d87c6bef"

RECONCILIATION_ROOT = Path("artifacts/validation_evaluation_reconciliation")
RECONCILIATION_LEDGER_PATH = Path("artifacts/ledger_validation_reconciliation.csv")

RECONCILIATION_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "training_run_id",
    "evaluation_id",
    "evaluation_attempt",
    "old_evaluator_fingerprint",
    "corrected_evaluator_fingerprint",
    "predictions_sha256",
    "metrics_sha256",
    "metadata_sha256",
    "manifest_sha256",
    "reconciliation_record_sha256",
    "unaffected_conditions_match",
    "reconciliation_code_fingerprint",
    "reconciliation_source_commit",
    "status",
    "reconciled_at",
)

# Files whose content determines the reconciliation LOGIC's own
# identity -- distinct from EVALUATOR_FINGERPRINT_MANIFEST, scoped only
# to the mechanism this module implements.
RECONCILIATION_CODE_MANIFEST: tuple[str, ...] = (
    "src/when_tta_hurts/validation_metric_reconciliation.py",
    "src/when_tta_hurts/evaluation/aggregation.py",
    "src/when_tta_hurts/validation_evaluation.py",
    "src/when_tta_hurts/metrics.py",
)


class ReconciliationError(RuntimeError):
    """Raised for ANY reconciliation failure -- missing, duplicate,
    stale, malformed, hash-mismatched, amendment-excluded, or
    semantically inconsistent evidence. Always fails closed; never
    silently reconciles a cell it cannot fully verify."""


def compute_reconciliation_code_fingerprint(
    repo_root: str | Path = ".", manifest: tuple[str, ...] = RECONCILIATION_CODE_MANIFEST
) -> str:
    repo_root = Path(repo_root)
    h = hashlib.sha256()
    for rel in manifest:
        h.update(rel.encode())
        h.update(hash_file(repo_root / rel).encode())
    return h.hexdigest()


def _git_commit_hash(repo_root: str | Path = ".") -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def ensure_reconciliation_ledger_exists(ledger_path: str | Path = RECONCILIATION_LEDGER_PATH) -> bool:
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RECONCILIATION_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


def _reconciliation_rows_for_run(
    run_id: str, ledger_path: str | Path = RECONCILIATION_LEDGER_PATH
) -> list[dict[str, Any]]:
    return [r for r in ledger_module._read_existing_rows(ledger_path) if r.get("training_run_id") == run_id]


def resolve_canonical_pre_fix_row(
    run_id: str,
    ledger_path: str | Path = VALIDATION_EVALUATION_LEDGER_PATH,
    validation_root: str | Path = DEFAULT_EVALUATION_ROOT,
    amendments_ledger_path: str | Path = EVALUATION_AMENDMENTS_LEDGER_PATH,
    old_evaluator_fingerprint: str = OLD_EVALUATOR_FINGERPRINT,
) -> tuple[dict[str, Any], int]:
    """Resolve run_id's SOLE eligible (pre-fix-fingerprint, completed,
    non-amendment-excluded) evaluation-ledger row. Raises
    ReconciliationError on missing/ambiguous -- never guesses."""
    validation_root = Path(validation_root)
    rows = [
        r
        for r in ledger_module._read_existing_rows(ledger_path)
        if r["training_run_id"] == run_id and r["status"] == "completed"
    ]

    eligible: list[tuple[dict[str, Any], int]] = []
    for r in rows:
        attempt = int(r["evaluation_attempt"])
        if ledger_module.is_evaluation_canonical_ineligible(
            r["evaluation_id"], attempt, ledger_path=amendments_ledger_path
        ):
            continue
        meta_path = validation_root / run_id / f"attempt_{attempt:03d}" / "metadata.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get("evaluator_fingerprint") == old_evaluator_fingerprint:
            eligible.append((r, attempt))

    if len(eligible) == 0:
        raise ReconciliationError(
            f"No pre-fix-fingerprint-eligible completed evaluation found for {run_id!r}."
        )
    if len(eligible) > 1:
        raise ReconciliationError(
            f"Ambiguous: {len(eligible)} pre-fix-fingerprint-eligible completed evaluations for {run_id!r}."
        )
    return eligible[0]


def reconcile_validation_cell(
    run_id: str,
    *,
    validation_root: str | Path = DEFAULT_EVALUATION_ROOT,
    ledger_path: str | Path = VALIDATION_EVALUATION_LEDGER_PATH,
    amendments_ledger_path: str | Path = EVALUATION_AMENDMENTS_LEDGER_PATH,
    reconciliation_root: str | Path = RECONCILIATION_ROOT,
    reconciliation_ledger_path: str | Path = RECONCILIATION_LEDGER_PATH,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Deterministic, offline reconciliation of ONE validation cell.
    Never initializes MPS, loads a checkpoint, loads a dataset,
    generates views, or performs inference -- reads only already-
    persisted predictions/metadata/metrics. Fails closed on ANY
    inconsistency (see ReconciliationError)."""
    validation_root = Path(validation_root)
    reconciliation_root = Path(reconciliation_root)

    row, attempt = resolve_canonical_pre_fix_row(run_id, ledger_path, validation_root, amendments_ledger_path)

    if _reconciliation_rows_for_run(run_id, reconciliation_ledger_path):
        raise ReconciliationError(f"{run_id!r} already has a reconciliation record -- refusing to duplicate.")

    attempt_dir = validation_root / run_id / f"attempt_{attempt:03d}"
    if not all((attempt_dir / fn).exists() for fn in ALL_REQUIRED_EVALUATION_ARTIFACT_FILENAMES):
        raise ReconciliationError(f"{run_id!r} attempt {attempt}: missing required artifact file(s).")

    # Verify the ORIGINAL manifest BEFORE reading predictions.
    manifest = json.loads((attempt_dir / "artifact_manifest.json").read_text())
    try:
        verify_evaluation_artifact_manifest(attempt_dir, manifest)
    except EvaluationPersistenceError as e:
        raise ReconciliationError(f"{run_id!r} attempt {attempt}: manifest verification failed: {e}") from e

    metadata = json.loads((attempt_dir / "metadata.json").read_text())
    metrics = json.loads((attempt_dir / "metrics.json").read_text())
    npz = np.load(attempt_dir / "predictions.npz")
    predictions = {k: npz[k] for k in npz.files}

    old_fp = metadata.get("evaluator_fingerprint")
    if old_fp != OLD_EVALUATOR_FINGERPRINT:
        raise ReconciliationError(
            f"{run_id!r} attempt {attempt}: metadata evaluator_fingerprint {old_fp!r} does not match "
            f"the expected pre-fix fingerprint {OLD_EVALUATOR_FINGERPRINT!r} -- refusing to reconcile."
        )
    corrected_fp, _ = compute_evaluator_fingerprint(repo_root=repo_root)

    recomputed = _recompute_all_conditions_from_predictions(predictions, PREFIX_SEQUENCE)
    persisted_conditions = metrics["conditions"]

    # Verify EVERY unaffected condition still matches exactly -- fail
    # closed if "only original_anchored_tta is affected" is ever wrong
    # for this cell, rather than silently reconciling anyway.
    unaffected_match = True
    for agg in AGGREGATORS:
        for n in PREFIX_SEQUENCE:
            persisted_entry = persisted_conditions["naive_tta"][agg][str(n)]
            for k, v in recomputed["naive_tta"][agg][n].items():
                if not np.isclose(v, persisted_entry[k], atol=1e-6, rtol=1e-6):
                    unaffected_match = False
    if recomputed["bn_adapted_tta"] is not None:
        persisted_bn = persisted_conditions.get("bn_adapted_tta")
        if persisted_bn is None:
            unaffected_match = False
        else:
            for n, entry in recomputed["bn_adapted_tta"].items():
                persisted_entry = persisted_bn[str(n)]
                for k, v in entry.items():
                    if not np.isclose(v, persisted_entry[k], atol=1e-6, rtol=1e-6):
                        unaffected_match = False
    elif persisted_conditions.get("bn_adapted_tta") is not None:
        unaffected_match = False

    clean_recomputed = compute_metrics_from_probabilities(predictions["clean_probs"], predictions["labels"])
    for k, v in clean_recomputed.items():
        if not np.isclose(v, metrics["clean"][k], atol=1e-6, rtol=1e-6):
            unaffected_match = False

    if not unaffected_match:
        raise ReconciliationError(
            f"{run_id!r} attempt {attempt}: an UNAFFECTED metric diverged beyond tolerance -- the "
            f"'only original_anchored_tta is affected' assumption does not hold for this cell. "
            f"Refusing to reconcile."
        )

    corrected_original_anchored = {str(n): recomputed["original_anchored_tta"][n] for n in PREFIX_SEQUENCE}

    reconciliation_code_fp = compute_reconciliation_code_fingerprint(repo_root)
    source_commit = _git_commit_hash(repo_root)

    record = {
        "training_run_id": run_id,
        "evaluation_id": row["evaluation_id"],
        "evaluation_attempt": attempt,
        "old_evaluator_fingerprint": old_fp,
        "corrected_evaluator_fingerprint": corrected_fp,
        "original_anchored_tta_corrected": corrected_original_anchored,
        "unaffected_conditions_match": unaffected_match,
        "predictions_sha256": hash_file(attempt_dir / "predictions.npz"),
        "metrics_sha256": hash_file(attempt_dir / "metrics.json"),
        "metadata_sha256": hash_file(attempt_dir / "metadata.json"),
        "manifest_sha256": hash_file(attempt_dir / "artifact_manifest.json"),
        "reconciliation_code_fingerprint": reconciliation_code_fp,
        "reconciliation_source_commit": source_commit,
    }

    record_dir = reconciliation_root / run_id / f"attempt_{attempt:03d}"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / "reconciliation.json"
    if record_path.exists():
        raise ReconciliationError(f"{run_id!r} attempt {attempt}: reconciliation.json already exists.")
    atomic_write_json(record, record_path)
    record_sha256 = hash_file(record_path)

    ensure_reconciliation_ledger_exists(reconciliation_ledger_path)
    with open(reconciliation_ledger_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RECONCILIATION_LEDGER_FIELDNAMES))
        writer.writerow(
            {
                "training_run_id": run_id,
                "evaluation_id": row["evaluation_id"],
                "evaluation_attempt": attempt,
                "old_evaluator_fingerprint": old_fp,
                "corrected_evaluator_fingerprint": corrected_fp,
                "predictions_sha256": record["predictions_sha256"],
                "metrics_sha256": record["metrics_sha256"],
                "metadata_sha256": record["metadata_sha256"],
                "manifest_sha256": record["manifest_sha256"],
                "reconciliation_record_sha256": record_sha256,
                "unaffected_conditions_match": unaffected_match,
                "reconciliation_code_fingerprint": reconciliation_code_fp,
                "reconciliation_source_commit": source_commit,
                "status": "completed",
                "reconciled_at": time.time(),
            }
        )

    return {
        "status": "completed",
        "training_run_id": run_id,
        "evaluation_attempt": attempt,
        "record_path": str(record_path),
    }


def is_reconciled_compatible(
    run_id: str, current_fp: str, reconciliation_ledger_path: str | Path = RECONCILIATION_LEDGER_PATH
) -> tuple[bool, dict[str, Any] | None]:
    """Returns (True, row) iff EXACTLY ONE valid reconciliation record
    exists for run_id binding old_evaluator_fingerprint -> current_fp
    with status='completed' and unaffected_conditions_match='True'.
    Used by statistical-analysis/cross-condition resolvers to recognize
    an old evaluation plus a valid reconciliation record as current-
    contract-compatible."""
    rows = _reconciliation_rows_for_run(run_id, reconciliation_ledger_path)
    matching = [
        r
        for r in rows
        if r.get("corrected_evaluator_fingerprint") == current_fp
        and r.get("status") == "completed"
        and r.get("unaffected_conditions_match") == "True"
    ]
    if len(matching) == 1:
        return True, matching[0]
    return False, None


def get_reconciled_original_anchored_metric(
    run_id: str,
    prefix_n: int,
    metric_key: str,
    reconciliation_root: str | Path = RECONCILIATION_ROOT,
    reconciliation_ledger_path: str | Path = RECONCILIATION_LEDGER_PATH,
) -> float:
    """Reads the CORRECTED original_anchored_tta metric value from the
    reconciliation record (never from the original, uncorrected
    metrics.json) -- for use whenever an affected original-anchored
    endpoint is requested after reconciliation. Fails closed if no
    reconciliation record exists."""
    rows = _reconciliation_rows_for_run(run_id, reconciliation_ledger_path)
    if len(rows) != 1:
        raise ReconciliationError(
            f"Expected exactly one reconciliation record for {run_id!r}, found {len(rows)}."
        )
    attempt = int(rows[0]["evaluation_attempt"])
    record_path = Path(reconciliation_root) / run_id / f"attempt_{attempt:03d}" / "reconciliation.json"
    if not record_path.exists():
        raise ReconciliationError(f"Reconciliation record file missing for {run_id!r}: {record_path}")
    record = json.loads(record_path.read_text())
    return record["original_anchored_tta_corrected"][str(prefix_n)][metric_key]
