"""Regression tests for the canonical_eligible boolean-parsing bug found
during Phase 2B.3A attempt_002 pre-run verification: is_canonical_ineligible()
previously compared against the literal string "False" (capitalized),
which never matched the real amendment row's committed lowercase "false"
value -- silently treating a canonical-ineligible attempt as eligible.

Synthetic tests use temporary ledger files only. The read-only checks
against the REAL artifacts/ledger_amendments.csv are explicitly authorized
(read-only verification that the fix resolves the actual reported bug);
they never write to that file."""

from __future__ import annotations

import csv

import pytest

from when_tta_hurts.ledger import (
    AMENDMENTS_LEDGER_FIELDNAMES,
    AMENDMENTS_LEDGER_PATH,
    LedgerSchemaError,
    is_canonical_ineligible,
)

RUN_ID = "A-pathmnist-28px-batchnorm-policy-none-s0"


def _write_amendment_row(path, canonical_eligible_raw: str, attempt_id: int = 1):
    row = {name: "" for name in AMENDMENTS_LEDGER_FIELDNAMES}
    row.update(
        {
            "run_id": RUN_ID,
            "attempt_id": attempt_id,
            "original_status": "completed",
            "canonical_eligible": canonical_eligible_raw,
            "amendment_type": "engineering_observability_failure",
            "reason": "test",
            "validation_metrics_computed": "true",
            "validation_metrics_persisted": "false",
            "validation_metrics_inspected": "false",
            "test_metrics_observed": "false",
            "tta_metrics_observed": "false",
            "source_commit": "deadbeef",
            "checkpoint_hash": "abc123",
            "recorded_at": "2026-01-01T00:00:00Z",
        }
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AMENDMENTS_LEDGER_FIELDNAMES))
        writer.writeheader()
        writer.writerow(row)


@pytest.mark.parametrize("token", ["true", "True", "TRUE", " true ", " True "])
def test_true_tokens_mean_eligible(tmp_path, token):
    path = tmp_path / "ledger_amendments.csv"
    _write_amendment_row(path, token)
    assert is_canonical_ineligible(RUN_ID, 1, path) is False


@pytest.mark.parametrize("token", ["false", "False", "FALSE", " false ", " False "])
def test_false_tokens_mean_ineligible(tmp_path, token):
    path = tmp_path / "ledger_amendments.csv"
    _write_amendment_row(path, token)
    assert is_canonical_ineligible(RUN_ID, 1, path) is True


def test_empty_value_hard_fails(tmp_path):
    path = tmp_path / "ledger_amendments.csv"
    _write_amendment_row(path, "")
    with pytest.raises(LedgerSchemaError):
        is_canonical_ineligible(RUN_ID, 1, path)


def test_missing_field_hard_fails(tmp_path):
    """A row with the column entirely absent (malformed header/row)."""
    path = tmp_path / "ledger_amendments.csv"
    fieldnames = [n for n in AMENDMENTS_LEDGER_FIELDNAMES if n != "canonical_eligible"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": RUN_ID,
                "attempt_id": 1,
                "original_status": "completed",
                "amendment_type": "x",
                "reason": "x",
                "validation_metrics_computed": "true",
                "validation_metrics_persisted": "false",
                "validation_metrics_inspected": "false",
                "test_metrics_observed": "false",
                "tta_metrics_observed": "false",
                "source_commit": "deadbeef",
                "checkpoint_hash": "abc123",
                "recorded_at": "2026-01-01T00:00:00Z",
            }
        )
    with pytest.raises(LedgerSchemaError):
        is_canonical_ineligible(RUN_ID, 1, path)


@pytest.mark.parametrize("token", ["0", "1", "no", "yes", "fals", "unknown", "null", "None"])
def test_invalid_tokens_hard_fail(tmp_path, token):
    path = tmp_path / "ledger_amendments.csv"
    _write_amendment_row(path, token)
    with pytest.raises(LedgerSchemaError):
        is_canonical_ineligible(RUN_ID, 1, path)


def test_no_amendment_row_means_eligible_by_default(tmp_path):
    path = tmp_path / "ledger_amendments.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(AMENDMENTS_LEDGER_FIELDNAMES))
        writer.writeheader()
    assert is_canonical_ineligible(RUN_ID, 1, path) is False


# --- read-only verification against the REAL committed amendment row ---


def test_real_committed_amendment_row_is_lowercase_false():
    """Sanity: confirms we're testing against the actual reported defect,
    not a hypothetical -- the real row's raw value is lowercase 'false'."""
    with AMENDMENTS_LEDGER_PATH.open() as f:
        rows = list(csv.DictReader(f))
    matching = [r for r in rows if r["run_id"] == RUN_ID and r["attempt_id"] == "1"]
    assert len(matching) == 1
    assert matching[0]["canonical_eligible"] == "false"


def test_real_amendment_row_now_correctly_reports_ineligible():
    assert is_canonical_ineligible(RUN_ID, 1, AMENDMENTS_LEDGER_PATH) is True


def test_real_attempt_001_alone_would_not_trigger_completed_run_skip():
    """attempt_001 in isolation must never trigger a skip -- verified
    directly against is_canonical_ineligible rather than the full
    check_confirmatory_skip (which, against the real repo state, now
    correctly selects the later canonical attempt_003 -- see
    test_real_canonical_selection_finds_attempt_3 below)."""
    assert is_canonical_ineligible(RUN_ID, 1, AMENDMENTS_LEDGER_PATH) is True


def test_real_canonical_selection_finds_attempt_3():
    """Against the real repo state (attempt_001 ineligible, attempt_002
    failed, attempt_003 canonical, attempt_004 ineligible),
    check_confirmatory_skip must select attempt_003 -- not attempt_001,
    and not raise ambiguity."""
    from when_tta_hurts.matrix import parse_and_validate_matrix
    from when_tta_hurts.orchestrator import check_confirmatory_skip

    expanded = parse_and_validate_matrix("configs/experiment_matrix.yaml", block_d_gate_passed=False)
    cell = expanded.cells[0]
    assert cell.run_id() == RUN_ID
    skip = check_confirmatory_skip(cell, "artifacts/confirmatory", AMENDMENTS_LEDGER_PATH)
    assert skip is not None
    assert skip.status == "skipped_completed"
    assert skip.attempt_number == 3


def test_real_next_attempt_number_exceeds_ineligible_attempt_001():
    """Regardless of how many attempts currently exist on disk, the next
    attempt number must always be strictly greater than 1 -- attempt_001's
    ineligibility must never cause its slot to be reused."""
    from when_tta_hurts.matrix import parse_and_validate_matrix
    from when_tta_hurts.run_identity import next_attempt_number

    expanded = parse_and_validate_matrix("configs/experiment_matrix.yaml", block_d_gate_passed=False)
    cell = expanded.cells[0]
    assert next_attempt_number(cell, "artifacts/confirmatory") > 1
