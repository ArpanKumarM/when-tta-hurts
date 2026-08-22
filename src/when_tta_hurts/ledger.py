"""Append-only experiment ledger. Thin, typed wrapper around
artifacts.append_ledger_row so every run type (pilot, confirmatory, ...)
writes a consistent, documented row shape. Never deletes or edits rows,
regardless of run outcome -- see CLAUDE.md.
"""

from __future__ import annotations

import csv
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

# Separate append-only ledger for confirmatory (Phase 2B) runs -- kept
# separate from both DEFAULT_LEDGER_PATH (pilot schema) and
# INCIDENTS_LEDGER_PATH (incident schema) for the same column-alignment
# reason documented above: a confirmatory row has a materially different
# field set (attempt_id, block, protocol_commit, artifact hashes, ...)
# than a pilot row, and csv.DictWriter's header is fixed by whichever row
# writes it first. Existing ledger.csv/ledger_incidents.csv rows and
# schemas are completely unaffected by this file's existence.
CONFIRMATORY_LEDGER_PATH = Path("artifacts/ledger_confirmatory.csv")

# Single source of truth for the confirmatory ledger's column order --
# used both by append_confirmatory_entry (row construction) and
# ensure_confirmatory_ledger_exists (header-only file creation), so the
# two can never drift apart.
CONFIRMATORY_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "confirmatory",
    "run_id",
    "attempt_id",
    "block",
    "config_hash",
    "protocol_commit",
    "dataset",
    "model",
    "resolution",
    "normalization",
    "training_policy",
    "seed",
    "split",
    "status",
    "checkpoint_hash",
    "started_at",
    "ended_at",
    "runtime_seconds",
    "failure_reason",
    "validation_metrics_observed",
    "test_metrics_observed",
)


def ensure_confirmatory_ledger_exists(ledger_path: str | Path = CONFIRMATORY_LEDGER_PATH) -> bool:
    """Create a HEADER-ONLY confirmatory ledger file at `ledger_path` if it
    does not already exist. Writes NO data row -- only the column header,
    using CONFIRMATORY_LEDGER_FIELDNAMES so it can never drift from what
    append_confirmatory_entry actually writes. Returns True if the file was
    created, False if it already existed (no-op).
    """
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CONFIRMATORY_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


class LedgerConflictError(RuntimeError):
    """Raised when append_confirmatory_entry is called for a (run_id,
    attempt_id) that already has a row with DIFFERENT content -- this is a
    hard failure, never a silent overwrite."""


def _read_existing_rows(ledger_path: str | Path) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def has_confirmatory_row(
    run_id: str, attempt_id: int, ledger_path: str | Path = CONFIRMATORY_LEDGER_PATH
) -> bool:
    """True iff ANY row (any status -- completed/failed/aborted) exists in
    the confirmatory ledger for (run_id, attempt_id). Used to detect
    attempts that are nonterminal on disk but were never reconciled."""
    for row in _read_existing_rows(ledger_path):
        if row.get("run_id") == run_id and row.get("attempt_id") == str(attempt_id):
            return True
    return False


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


def append_confirmatory_entry(
    *,
    run_id: str,
    attempt_id: int,
    block: str,
    config_hash: str,
    protocol_commit: str,
    dataset: str,
    model: str,
    resolution: int,
    normalization: str,
    training_policy: str,
    seed: int,
    split: str,
    status: str,
    checkpoint_hash: str,
    started_at: float,
    ended_at: float,
    runtime_seconds: float,
    failure_reason: str = "",
    validation_metrics_observed: bool = False,
    test_metrics_observed: bool = False,
    ledger_path: str | Path = CONFIRMATORY_LEDGER_PATH,
) -> str:
    """Append one confirmatory-run row. Always tags confirmatory=true.

    Idempotency: keyed on (run_id, attempt_id).
    - No existing row for this (run_id, attempt_id): appends, returns "appended".
    - Existing row with IDENTICAL content: no-op, returns "duplicate_ignored"
      (idempotent -- calling this twice with the same facts is safe).
    - Existing row with DIFFERENT content: raises LedgerConflictError (hard
      failure -- never silently overwritten).

    Per your requirement, `split` must never be "test" unless
    test_metrics_observed reflects a properly authorized final evaluation
    -- this function does not itself enforce the authorization gate (that
    lives in authorization.py); it only records what is asserted by the
    caller, and is not a substitute for that gate.
    """
    row = {
        "confirmatory": True,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "block": block,
        "config_hash": config_hash,
        "protocol_commit": protocol_commit,
        "dataset": dataset,
        "model": model,
        "resolution": resolution,
        "normalization": normalization,
        "training_policy": training_policy,
        "seed": seed,
        "split": split,
        "status": status,
        "checkpoint_hash": checkpoint_hash,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "failure_reason": failure_reason,
        "validation_metrics_observed": validation_metrics_observed,
        "test_metrics_observed": test_metrics_observed,
    }
    row_str = {k: str(v) for k, v in row.items()}

    existing_rows = _read_existing_rows(ledger_path)
    for existing in existing_rows:
        if existing.get("run_id") == run_id and existing.get("attempt_id") == str(attempt_id):
            if existing == row_str:
                return "duplicate_ignored"
            raise LedgerConflictError(
                f"Confirmatory ledger already has a DIFFERENT row for run_id={run_id}, "
                f"attempt_id={attempt_id}. Existing: {existing}. New: {row_str}. "
                f"Hard failure -- refusing to append a conflicting duplicate."
            )

    append_ledger_row(row, ledger_path)
    return "appended"


# Append-only eligibility-overlay ledger (Phase 2B.3A Part 2B). NEVER
# rewrites a row in CONFIRMATORY_LEDGER_PATH -- a completed attempt that
# turns out to be scientifically/observability ineligible is recorded here
# instead, as an additional row layered on top of (never replacing) the
# original confirmatory-ledger row. See docs/phase2b_canary_audit.md for
# the first real use of this mechanism.
AMENDMENTS_LEDGER_PATH = Path("artifacts/ledger_amendments.csv")

AMENDMENTS_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "run_id",
    "attempt_id",
    "original_status",
    "canonical_eligible",
    "amendment_type",
    "reason",
    "validation_metrics_computed",
    "validation_metrics_persisted",
    "validation_metrics_inspected",
    "test_metrics_observed",
    "tta_metrics_observed",
    "source_commit",
    "checkpoint_hash",
    "recorded_at",
)


def ensure_amendments_ledger_exists(ledger_path: str | Path = AMENDMENTS_LEDGER_PATH) -> bool:
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AMENDMENTS_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


def append_amendment_entry(
    *,
    run_id: str,
    attempt_id: int,
    original_status: str,
    canonical_eligible: bool,
    amendment_type: str,
    reason: str,
    validation_metrics_computed: bool,
    validation_metrics_persisted: bool,
    validation_metrics_inspected: bool,
    test_metrics_observed: bool,
    tta_metrics_observed: bool,
    source_commit: str,
    checkpoint_hash: str,
    recorded_at: str,
    ledger_path: str | Path = AMENDMENTS_LEDGER_PATH,
) -> str:
    """Append one eligibility-overlay row. Idempotent on (run_id,
    attempt_id): identical duplicate -> 'duplicate_ignored'; conflicting
    duplicate -> LedgerConflictError (never silently overwritten), exactly
    like append_confirmatory_entry."""
    row = {
        "run_id": run_id,
        "attempt_id": attempt_id,
        "original_status": original_status,
        "canonical_eligible": canonical_eligible,
        "amendment_type": amendment_type,
        "reason": reason,
        "validation_metrics_computed": validation_metrics_computed,
        "validation_metrics_persisted": validation_metrics_persisted,
        "validation_metrics_inspected": validation_metrics_inspected,
        "test_metrics_observed": test_metrics_observed,
        "tta_metrics_observed": tta_metrics_observed,
        "source_commit": source_commit,
        "checkpoint_hash": checkpoint_hash,
        "recorded_at": recorded_at,
    }
    row_str = {k: str(v) for k, v in row.items()}

    existing_rows = _read_existing_rows(ledger_path)
    for existing in existing_rows:
        if existing.get("run_id") == run_id and existing.get("attempt_id") == str(attempt_id):
            if existing == row_str:
                return "duplicate_ignored"
            raise LedgerConflictError(
                f"Amendments ledger already has a DIFFERENT row for run_id={run_id}, "
                f"attempt_id={attempt_id}. Existing: {existing}. New: {row_str}. "
                f"Hard failure -- refusing to append a conflicting duplicate."
            )

    append_ledger_row(row, ledger_path)
    return "appended"


# Separate append-only ledger for validation-only TTA-evaluation attempts
# (Phase 2B.4A) -- kept separate from CONFIRMATORY_LEDGER_PATH for the same
# column-alignment reason documented above (a different field set: training
# run/attempt provenance, evaluation config hash, primary artifact hash,
# rather than training-loop fields like block/checkpoint_hash-as-training-
# result). This ledger NEVER records a test-split evaluation -- `split` is
# always "validation" and `test_metrics_observed` is always False (see
# append_evaluation_entry's docstring).
VALIDATION_EVALUATION_LEDGER_PATH = Path("artifacts/ledger_validation_evaluation.csv")

VALIDATION_EVALUATION_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "confirmatory",
    "evaluation_id",
    "training_run_id",
    "training_attempt",
    "checkpoint_hash",
    "evaluation_config_hash",
    "evaluation_attempt",
    "split",
    "status",
    "primary_artifact_hash",
    "started_at",
    "ended_at",
    "runtime_seconds",
    "failure_reason",
    "test_metrics_observed",
)


def ensure_evaluation_ledger_exists(ledger_path: str | Path = VALIDATION_EVALUATION_LEDGER_PATH) -> bool:
    """Create a HEADER-ONLY validation-evaluation ledger file at
    `ledger_path` if it does not already exist. Writes NO data row -- only
    the column header, using VALIDATION_EVALUATION_LEDGER_FIELDNAMES so it
    can never drift from what append_evaluation_entry actually writes.
    Uses the identical csv.DictWriter + newline="" convention as
    ensure_confirmatory_ledger_exists(), so the byte-level line-ending
    behavior matches the existing ledgers exactly -- nothing is
    "normalized" differently for this new file. Returns True if the file
    was created, False if it already existed (no-op).
    """
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(VALIDATION_EVALUATION_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


def has_evaluation_row(
    evaluation_id: str, evaluation_attempt: int, ledger_path: str | Path = VALIDATION_EVALUATION_LEDGER_PATH
) -> bool:
    """True iff ANY row (any status) exists for (evaluation_id,
    evaluation_attempt) -- used to detect nonterminal attempts left over
    from a prior invocation, mirroring has_confirmatory_row()."""
    for row in _read_existing_rows(ledger_path):
        if row.get("evaluation_id") == evaluation_id and row.get("evaluation_attempt") == str(
            evaluation_attempt
        ):
            return True
    return False


def append_evaluation_entry(
    *,
    evaluation_id: str,
    training_run_id: str,
    training_attempt: int,
    checkpoint_hash: str,
    evaluation_config_hash: str,
    evaluation_attempt: int,
    status: str,
    primary_artifact_hash: str,
    started_at: float,
    ended_at: float,
    runtime_seconds: float,
    failure_reason: str = "",
    ledger_path: str | Path = VALIDATION_EVALUATION_LEDGER_PATH,
) -> str:
    """Append one validation-evaluation row. `split` is hardcoded to
    "validation" and `test_metrics_observed` is hardcoded to False --
    neither is a caller-overridable parameter, so this function cannot be
    used to record a test-split evaluation even by mistake; a genuine
    test-split evaluation (once authorized, post-Phase-2B.5) requires its
    own, separately-designed ledger entry point.

    Idempotency: keyed on (evaluation_id, evaluation_attempt) -- identical
    duplicate -> "duplicate_ignored"; conflicting duplicate -> hard failure
    (LedgerConflictError), exactly like append_confirmatory_entry().
    """
    row = {
        "confirmatory": True,
        "evaluation_id": evaluation_id,
        "training_run_id": training_run_id,
        "training_attempt": training_attempt,
        "checkpoint_hash": checkpoint_hash,
        "evaluation_config_hash": evaluation_config_hash,
        "evaluation_attempt": evaluation_attempt,
        "split": "validation",
        "status": status,
        "primary_artifact_hash": primary_artifact_hash,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "failure_reason": failure_reason,
        "test_metrics_observed": False,
    }
    row_str = {k: str(v) for k, v in row.items()}

    existing_rows = _read_existing_rows(ledger_path)
    for existing in existing_rows:
        if existing.get("evaluation_id") == evaluation_id and existing.get("evaluation_attempt") == str(
            evaluation_attempt
        ):
            if existing == row_str:
                return "duplicate_ignored"
            raise LedgerConflictError(
                f"Validation-evaluation ledger already has a DIFFERENT row for "
                f"evaluation_id={evaluation_id}, evaluation_attempt={evaluation_attempt}. "
                f"Existing: {existing}. New: {row_str}. Hard failure -- refusing to append "
                f"a conflicting duplicate."
            )

    append_ledger_row(row, ledger_path)
    return "appended"


# Append-only eligibility-overlay ledger for validation-evaluation attempts
# (Phase 2B.4D Metric-Contract Correction) -- mirrors AMENDMENTS_LEDGER_PATH's
# discipline exactly (never rewrites/reorders the base ledger; a completed
# attempt with no amendment row is eligible by default; amendments are
# opt-in exceptions, not a default-deny allowlist). Kept separate from
# AMENDMENTS_LEDGER_PATH because its keying and field set are
# evaluation-specific (evaluation_id/evaluation_attempt, not
# run_id/attempt_id).
EVALUATION_AMENDMENTS_LEDGER_PATH = Path("artifacts/ledger_validation_evaluation_amendments.csv")

EVALUATION_AMENDMENTS_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "evaluation_id",
    "evaluation_attempt",
    "historical_status",
    "canonical_eligible",
    "reason",
    "validation_metrics_observed",
    "test_metrics_observed",
    "artifacts_preserved",
    "rerun_required",
    "predictions_sha256",
    "source_commit",
    "recorded_at",
)


def ensure_evaluation_amendments_ledger_exists(
    ledger_path: str | Path = EVALUATION_AMENDMENTS_LEDGER_PATH,
) -> bool:
    """Create a HEADER-ONLY evaluation-amendments ledger file if it does
    not already exist. Writes NO data row. Returns True if created, False
    if it already existed (no-op)."""
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(EVALUATION_AMENDMENTS_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


def append_evaluation_amendment_entry(
    *,
    evaluation_id: str,
    evaluation_attempt: int,
    historical_status: str,
    canonical_eligible: bool,
    reason: str,
    validation_metrics_observed: bool,
    test_metrics_observed: bool,
    artifacts_preserved: bool,
    rerun_required: bool,
    predictions_sha256: str,
    source_commit: str,
    recorded_at: str,
    ledger_path: str | Path = EVALUATION_AMENDMENTS_LEDGER_PATH,
) -> str:
    """Append one evaluation eligibility-overlay row. Idempotent on
    (evaluation_id, evaluation_attempt): identical duplicate ->
    'duplicate_ignored'; conflicting duplicate -> LedgerConflictError
    (never silently overwritten), exactly like append_amendment_entry()."""
    row = {
        "evaluation_id": evaluation_id,
        "evaluation_attempt": evaluation_attempt,
        "historical_status": historical_status,
        "canonical_eligible": canonical_eligible,
        "reason": reason,
        "validation_metrics_observed": validation_metrics_observed,
        "test_metrics_observed": test_metrics_observed,
        "artifacts_preserved": artifacts_preserved,
        "rerun_required": rerun_required,
        "predictions_sha256": predictions_sha256,
        "source_commit": source_commit,
        "recorded_at": recorded_at,
    }
    row_str = {k: str(v) for k, v in row.items()}

    existing_rows = _read_existing_rows(ledger_path)
    for existing in existing_rows:
        if existing.get("evaluation_id") == evaluation_id and existing.get("evaluation_attempt") == str(
            evaluation_attempt
        ):
            if existing == row_str:
                return "duplicate_ignored"
            raise LedgerConflictError(
                f"Evaluation-amendments ledger already has a DIFFERENT row for "
                f"evaluation_id={evaluation_id}, evaluation_attempt={evaluation_attempt}. "
                f"Existing: {existing}. New: {row_str}. Hard failure -- refusing to append "
                f"a conflicting duplicate."
            )

    append_ledger_row(row, ledger_path)
    return "appended"


class LedgerSchemaError(RuntimeError):
    """Raised when a ledger row contains a malformed value for a field
    that must be strictly boolean (e.g. canonical_eligible). Fails
    closed -- never silently defaults to eligible or ineligible."""


def _parse_canonical_bool(value: str | None, *, context: str) -> bool:
    """Strict, case-insensitive boolean parsing for ledger CSV fields.
    Accepts only 'true'/'false' (any capitalization, with surrounding
    whitespace stripped). Anything else -- missing, empty, or any other
    token -- is a hard failure, never a silent default."""
    if value is None:
        raise LedgerSchemaError(f"{context}: missing required boolean field.")
    token = value.strip().lower()
    if token == "true":
        return True
    if token == "false":
        return False
    raise LedgerSchemaError(f"{context}: expected 'true' or 'false' (case-insensitive), got {value!r}.")


def is_canonical_ineligible(
    run_id: str, attempt_id: int, ledger_path: str | Path = AMENDMENTS_LEDGER_PATH
) -> bool:
    """True iff the amendments ledger has a row for (run_id, attempt_id)
    with canonical_eligible == False. A completed attempt with NO
    amendment row is eligible by default (amendments are opt-in
    exceptions, not a default-deny allowlist). Raises LedgerSchemaError
    (fails closed -- never permits silent execution or silent skip) if a
    matching row's canonical_eligible field is not a valid boolean token."""
    for row in _read_existing_rows(ledger_path):
        if row.get("run_id") == run_id and row.get("attempt_id") == str(attempt_id):
            eligible = _parse_canonical_bool(
                row.get("canonical_eligible"),
                context=(
                    f"amendments ledger row (run_id={run_id}, attempt_id={attempt_id}), "
                    f"field 'canonical_eligible'"
                ),
            )
            if not eligible:
                return True
    return False


# ---------------------------------------------------------------------------
# Phase 2B.5A: header-only ledger for read-only statistical-analysis
# completions. Mirrors VALIDATION_EVALUATION_LEDGER_PATH's exact
# discipline: append-only, header-only until a real analysis actually
# runs (which does not happen during the Phase 2B.5A engineering task --
# no row is ever appended by this repository's tests or by this task).
# `test_split_accessed` is always False -- this ledger records analyses
# of already-persisted validation-evaluation artifacts only.
# ---------------------------------------------------------------------------

STATISTICAL_ANALYSIS_LEDGER_PATH = Path("artifacts/ledger_statistical_analysis.csv")

STATISTICAL_ANALYSIS_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "analysis_id",
    "family",
    "analysis_attempt",
    "analysis_fingerprint",
    "current_evaluator_fingerprint",
    "status",
    "primary_artifact_hash",
    "started_at",
    "ended_at",
    "runtime_seconds",
    "failure_reason",
    "test_split_accessed",
)


def ensure_statistical_analysis_ledger_exists(
    ledger_path: str | Path = STATISTICAL_ANALYSIS_LEDGER_PATH,
) -> bool:
    """Create a HEADER-ONLY statistical-analysis ledger file if it does
    not already exist. Writes NO data row. Returns True if created, False
    if it already existed (no-op)."""
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(STATISTICAL_ANALYSIS_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


def is_evaluation_canonical_ineligible(
    evaluation_id: str, evaluation_attempt: int, ledger_path: str | Path = EVALUATION_AMENDMENTS_LEDGER_PATH
) -> bool:
    """True iff the evaluation-amendments ledger has a row for
    (evaluation_id, evaluation_attempt) with canonical_eligible == False.
    A completed evaluation attempt with NO amendment row is eligible by
    default. Raises LedgerSchemaError (fails closed) if a matching row's
    canonical_eligible field is not a valid boolean token."""
    for row in _read_existing_rows(ledger_path):
        if row.get("evaluation_id") == evaluation_id and row.get("evaluation_attempt") == str(
            evaluation_attempt
        ):
            eligible = _parse_canonical_bool(
                row.get("canonical_eligible"),
                context=(
                    f"evaluation-amendments ledger row (evaluation_id={evaluation_id}, "
                    f"evaluation_attempt={evaluation_attempt}), field 'canonical_eligible'"
                ),
            )
            if not eligible:
                return True
    return False
