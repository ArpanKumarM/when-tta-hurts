"""Append-only experiment ledger. Thin, typed wrapper around
artifacts.append_ledger_row so every run type (pilot, confirmatory, ...)
writes a consistent, documented row shape. Never deletes or edits rows,
regardless of run outcome -- see CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import append_ledger_row

DEFAULT_LEDGER_PATH = Path("artifacts/ledger.csv")

# Separate append-only ledger for aborted/incident runs. Kept in its own
# file rather than mixed into DEFAULT_LEDGER_PATH because csv.DictWriter
# fixes column order/count from whichever row wrote the header first;
# an incident row has a different field set (reason, checkpoint status,
# approx runtime, etc.) than a completed pilot row, and appending it to
# ledger.csv would misalign columns against the header already written
# there. This keeps ledger.csv's existing completed row byte-for-byte
# untouched while still recording the incident append-only. See
# docs/pilot_audit.md section A for the cross-reference.
INCIDENTS_LEDGER_PATH = Path("artifacts/ledger_incidents.csv")


def append_pilot_entry(
    *,
    run_id: str,
    dataset: str,
    resolution: int,
    model: str,
    normalization: str,
    seed: int,
    tta_seed: int,
    config_hash: str,
    git_commit: str,
    best_epoch: int,
    epochs_completed: int,
    early_stopped: bool,
    clean_val_accuracy: float,
    status: str,
    artifact_dir: str,
    ledger_path: str | Path = DEFAULT_LEDGER_PATH,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append one row for a Phase 2A-style pilot run. Always tags
    phase=pilot, confirmatory=false, split=validation, per your
    requirement -- these are not caller-overridable to avoid a pilot
    accidentally being logged as confirmatory.
    """
    row = {
        "phase": "pilot",
        "confirmatory": False,
        "split": "validation",
        "run_id": run_id,
        "dataset": dataset,
        "resolution": resolution,
        "model": model,
        "normalization": normalization,
        "seed": seed,
        "tta_seed": tta_seed,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "early_stopped": early_stopped,
        "clean_val_accuracy": clean_val_accuracy,
        "status": status,
        "artifact_dir": artifact_dir,
    }
    if extra:
        row.update(extra)
    append_ledger_row(row, ledger_path)


def append_incident_entry(
    *,
    run_id: str,
    dataset: str,
    resolution: int,
    model: str,
    normalization: str,
    seed: int,
    tta_seed: int,
    config_hash: str,
    git_commit: str,
    status: str,
    reason: str,
    training_completed: bool,
    tta_metrics_observed: bool,
    checkpoint_status: str,
    approx_runtime_seconds: float,
    notes: str = "",
    ledger_path: str | Path = INCIDENTS_LEDGER_PATH,
) -> None:
    """Append one row for an aborted/incident run (e.g. killed due to a
    performance pathology) to the separate incidents ledger -- see
    INCIDENTS_LEDGER_PATH docstring for why this is a separate file from
    the completed-run ledger. Always tags phase=pilot, confirmatory=False,
    split=validation, matching append_pilot_entry's convention.
    """
    row = {
        "phase": "pilot",
        "confirmatory": False,
        "split": "validation",
        "run_id": run_id,
        "dataset": dataset,
        "resolution": resolution,
        "model": model,
        "normalization": normalization,
        "seed": seed,
        "tta_seed": tta_seed,
        "config_hash": config_hash,
        "git_commit": git_commit,
        "status": status,
        "reason": reason,
        "training_completed": training_completed,
        "tta_metrics_observed": tta_metrics_observed,
        "checkpoint_status": checkpoint_status,
        "approx_runtime_seconds": approx_runtime_seconds,
        "notes": notes,
    }
    append_ledger_row(row, ledger_path)
