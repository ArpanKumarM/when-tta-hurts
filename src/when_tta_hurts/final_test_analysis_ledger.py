"""Phase 2B.7A: append-only ledger for real final-test-analysis
completions (preregistered within-cell families AND the secondary
cross-condition addendum, disambiguated by `kind`/`identifier`).

Deliberately kept OUT of ledger.py: ledger.py is itself a member of
CROSS_CONDITION_ADDENDUM_MANIFEST / FINAL_TEST_RUNNER_MANIFEST (see
final_test_identity.py), so any edit to it -- even a purely additive,
unrelated new function -- changes cross_condition_analysis_fingerprint
and final_test_runner_fingerprint, invalidating the already-approved
generation-5 final-test authorization and forcing a reauthorization
cascade exactly like the one Phase 2B.6J/K had to remediate. This module
is a new, disjoint file, listed in no existing fingerprint manifest, so
adding it changes nothing that verify_final_test_authorization() checks.

Structurally separate from STATISTICAL_ANALYSIS_LEDGER_PATH (validation-
mode, never mixed) and from FINAL_TEST_LEDGER_PATH (records EVALUATION
completions, not ANALYSIS completions). `test_split_accessed` is always
False -- this ledger records analyses of already-persisted final-test-
evaluation artifacts only, never a direct read of the official test
split. No row is ever appended by this repository's Phase 2B.7A
engineering task or its own tests against the real path.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import append_ledger_row

FINAL_TEST_ANALYSIS_LEDGER_PATH = Path("artifacts/ledger_final_test_analysis.csv")

FINAL_TEST_ANALYSIS_LEDGER_FIELDNAMES: tuple[str, ...] = (
    "analysis_id",
    "kind",
    "identifier",
    "analysis_attempt",
    "final_test_analysis_fingerprint",
    "final_test_authorization_sha256",
    "final_test_authorization_commit",
    "current_evaluator_fingerprint",
    "status",
    "primary_artifact_hash",
    "started_at",
    "ended_at",
    "runtime_seconds",
    "failure_reason",
    "test_split_accessed",
)


class FinalTestAnalysisLedgerConflictError(RuntimeError):
    """Raised when append_final_test_analysis_entry is called for an
    (analysis_id, analysis_attempt) that already has a row with DIFFERENT
    content -- this is a hard failure, never a silent overwrite."""


def _read_existing_rows(ledger_path: str | Path) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ensure_final_test_analysis_ledger_exists(
    ledger_path: str | Path = FINAL_TEST_ANALYSIS_LEDGER_PATH,
) -> bool:
    """Create a HEADER-ONLY final-test-analysis ledger file if it does not
    already exist. Writes NO data row. Returns True if created, False if
    it already existed (no-op)."""
    path = Path(ledger_path)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FINAL_TEST_ANALYSIS_LEDGER_FIELDNAMES))
        writer.writeheader()
    return True


def append_final_test_analysis_entry(
    *,
    analysis_id: str,
    kind: str,
    identifier: str,
    analysis_attempt: int,
    final_test_analysis_fingerprint: str,
    final_test_authorization_sha256: str,
    final_test_authorization_commit: str,
    current_evaluator_fingerprint: str,
    status: str,
    primary_artifact_hash: str,
    started_at: float,
    ended_at: float,
    runtime_seconds: float,
    failure_reason: str = "",
    ledger_path: str | Path = FINAL_TEST_ANALYSIS_LEDGER_PATH,
) -> str:
    """Append one final-test-analysis row. `kind` must be 'family' or
    'cross_condition'; `identifier` is the family name (H1/H2/H3/BLOCK_C)
    or addendum hypothesis (H1/H2/H3). `test_split_accessed` is hardcoded
    to False -- not caller-overridable -- since this ledger only ever
    records analyses of already-persisted final-test-evaluation artifacts.

    Idempotency: keyed on (analysis_id, analysis_attempt) -- identical
    duplicate -> "duplicate_ignored"; conflicting duplicate -> hard
    failure (FinalTestAnalysisLedgerConflictError)."""
    if kind not in ("family", "cross_condition"):
        raise ValueError(f"kind must be 'family' or 'cross_condition', got {kind!r}.")
    row = {
        "analysis_id": analysis_id,
        "kind": kind,
        "identifier": identifier,
        "analysis_attempt": analysis_attempt,
        "final_test_analysis_fingerprint": final_test_analysis_fingerprint,
        "final_test_authorization_sha256": final_test_authorization_sha256,
        "final_test_authorization_commit": final_test_authorization_commit,
        "current_evaluator_fingerprint": current_evaluator_fingerprint,
        "status": status,
        "primary_artifact_hash": primary_artifact_hash,
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": runtime_seconds,
        "failure_reason": failure_reason,
        "test_split_accessed": False,
    }
    row_str = {k: str(v) for k, v in row.items()}

    existing_rows = _read_existing_rows(ledger_path)
    for existing in existing_rows:
        if existing.get("analysis_id") == analysis_id and existing.get("analysis_attempt") == str(
            analysis_attempt
        ):
            if existing == row_str:
                return "duplicate_ignored"
            raise FinalTestAnalysisLedgerConflictError(
                f"Final-test-analysis ledger already has a DIFFERENT row for "
                f"analysis_id={analysis_id}, analysis_attempt={analysis_attempt}. "
                f"Existing: {existing}. New: {row_str}. Hard failure -- refusing to append a "
                f"conflicting duplicate."
            )

    append_ledger_row(row, ledger_path)
    return "appended"


def next_final_test_analysis_attempt_number(
    analysis_id: str, ledger_path: str | Path = FINAL_TEST_ANALYSIS_LEDGER_PATH
) -> int:
    """max(existing analysis_attempt values for this analysis_id) + 1, or
    1 if none exist."""
    attempts = [
        int(row["analysis_attempt"])
        for row in _read_existing_rows(ledger_path)
        if row.get("analysis_id") == analysis_id
    ]
    return max(attempts) + 1 if attempts else 1


def existing_completed_attempt(
    analysis_id: str, ledger_path: str | Path = FINAL_TEST_ANALYSIS_LEDGER_PATH
) -> int | None:
    """Returns the attempt number of an already-`completed` row for
    `analysis_id`, or None if none exists. Used to make real-analysis
    entry points idempotent: a prior completed analysis is read back from
    its persisted artifact rather than recomputed."""
    for row in _read_existing_rows(ledger_path):
        if row.get("analysis_id") == analysis_id and row.get("status") == "completed":
            return int(row["analysis_attempt"])
    return None
