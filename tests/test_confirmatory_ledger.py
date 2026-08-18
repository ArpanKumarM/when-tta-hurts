"""Tests for ledger.py::append_confirmatory_entry. All use temporary CSV
files only -- never artifacts/ledger.csv or artifacts/ledger_incidents.csv."""

import csv

import pytest

from when_tta_hurts.ledger import LedgerConflictError, append_confirmatory_entry, append_pilot_entry

BASE_KWARGS = dict(
    run_id="A-pathmnist-28px-batchnorm-s0",
    attempt_id=1,
    block="A_core_normalization_resolution",
    config_hash="deadbeef",
    protocol_commit="ce4c962",
    dataset="pathmnist",
    model="small_cnn",
    resolution=28,
    normalization="batchnorm",
    training_policy="none",
    seed=0,
    split="validation",
    status="completed",
    checkpoint_hash="abc123",
    started_at=1000.0,
    ended_at=1010.0,
    runtime_seconds=10.0,
)


def test_append_confirmatory_entry_tags_confirmatory_true(tmp_path):
    path = tmp_path / "ledger_confirmatory.csv"
    result = append_confirmatory_entry(ledger_path=path, **BASE_KWARGS)
    assert result == "appended"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["confirmatory"] == "True"
    assert rows[0]["split"] == "validation"


def test_duplicate_identical_append_is_idempotent(tmp_path):
    path = tmp_path / "ledger_confirmatory.csv"
    r1 = append_confirmatory_entry(ledger_path=path, **BASE_KWARGS)
    r2 = append_confirmatory_entry(ledger_path=path, **BASE_KWARGS)
    assert r1 == "appended"
    assert r2 == "duplicate_ignored"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # NOT appended twice


def test_conflicting_duplicate_is_hard_failure(tmp_path):
    path = tmp_path / "ledger_confirmatory.csv"
    append_confirmatory_entry(ledger_path=path, **BASE_KWARGS)
    conflicting = dict(BASE_KWARGS)
    conflicting["config_hash"] = "different-hash"
    with pytest.raises(LedgerConflictError):
        append_confirmatory_entry(ledger_path=path, **conflicting)


def test_different_attempt_id_is_a_separate_row(tmp_path):
    path = tmp_path / "ledger_confirmatory.csv"
    append_confirmatory_entry(ledger_path=path, **BASE_KWARGS)
    second = dict(BASE_KWARGS)
    second["attempt_id"] = 2
    result = append_confirmatory_entry(ledger_path=path, **second)
    assert result == "appended"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_test_metrics_observed_field_recorded(tmp_path):
    path = tmp_path / "ledger_confirmatory.csv"
    kwargs = dict(BASE_KWARGS)
    kwargs["test_metrics_observed"] = True
    kwargs["split"] = "test"
    append_confirmatory_entry(ledger_path=path, **kwargs)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["test_metrics_observed"] == "True"
    assert rows[0]["split"] == "test"


def test_pilot_entries_remain_confirmatory_false(tmp_path):
    ledger_path = tmp_path / "ledger.csv"
    append_pilot_entry(
        run_id="pilot-x",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        seed=314159,
        tta_seed=271828,
        config_hash="hash",
        git_commit="commit",
        best_epoch=1,
        epochs_completed=1,
        early_stopped=False,
        clean_val_accuracy=0.5,
        status="completed",
        artifact_dir="artifacts/pilots/pilot-x",
        ledger_path=ledger_path,
    )
    with ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["confirmatory"] == "False"


def test_existing_pilot_ledger_schema_unaffected_by_confirmatory_ledger(tmp_path):
    """Writing to a confirmatory ledger must never touch a pilot ledger file."""
    pilot_ledger = tmp_path / "ledger.csv"
    confirmatory_ledger = tmp_path / "ledger_confirmatory.csv"

    append_pilot_entry(
        run_id="pilot-x",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        seed=314159,
        tta_seed=271828,
        config_hash="hash",
        git_commit="commit",
        best_epoch=1,
        epochs_completed=1,
        early_stopped=False,
        clean_val_accuracy=0.5,
        status="completed",
        artifact_dir="artifacts/pilots/pilot-x",
        ledger_path=pilot_ledger,
    )
    before_content = pilot_ledger.read_text()

    append_confirmatory_entry(ledger_path=confirmatory_ledger, **BASE_KWARGS)

    assert pilot_ledger.read_text() == before_content  # byte-for-byte unaffected
