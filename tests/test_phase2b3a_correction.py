"""Phase 2B.3A Part 2F: synthetic end-to-end tests for the observability/
skip-ordering/working-tree-policy correction. Temporary directories and
synthetic data ONLY -- never touches the real attempt_001 of
A-pathmnist-28px-batchnorm-policy-none-s0 or any real dataset."""

from __future__ import annotations

import csv
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import when_tta_hurts.orchestrator as orch
from when_tta_hurts import ledger as ledger_module
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.result_artifacts import (
    ALL_REQUIRED_ARTIFACT_FILENAMES,
    REQUIRED_COMPLETION_ARTIFACTS,
    PersistenceVerificationError,
)
from when_tta_hurts.run_identity import run_directory

A_CELL = MatrixCell(
    block="A_core_normalization_resolution",
    dataset="pathmnist",
    resolution=28,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)


def _make_loader(n, num_classes=9, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 28, 28, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=g)


def _run(cell, tmp_path, ledger_path, amendments_path, **kwargs):
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    return orch.run_train_validation_cell(
        cell,
        train_loader,
        val_loader,
        device,
        root=str(tmp_path),
        confirmatory_ledger_path=ledger_path,
        amendments_ledger_path=amendments_path,
        **kwargs,
    )


def _ledger_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


# --- artifact persistence content ---


def test_complete_history_best_epoch_early_stopping_peak_memory_persisted(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    result = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    assert result.status == "completed"

    attempt_dir = run_directory(A_CELL, root=str(tmp_path)) / "attempt_001"
    history = json.loads((attempt_dir / "training_history.json").read_text())
    assert len(history) > 0
    required_keys = (
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "epoch_runtime_seconds",
    )
    for entry in history:
        for key in required_keys:
            assert key in entry

    result_json = json.loads((attempt_dir / "result.json").read_text())
    assert result_json["best_epoch"] >= 1
    assert result_json["best_val_accuracy"] >= 0.0
    assert result_json["early_stopped"] in (True, False)
    assert result_json["early_stopping_reason"] != ""
    assert "peak_mps_memory" in result_json  # None on CPU, present as a key regardless
    assert result_json["peak_mps_memory"] is None  # CPU device in this test

    metadata = json.loads((attempt_dir / "metadata.json").read_text())
    assert metadata["frozen_training_settings"]["optimizer"] == "adam"
    assert metadata["block"] == A_CELL.block


def test_artifact_manifest_hashes_verify(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _run(A_CELL, tmp_path, ledger_path, amendments_path)

    attempt_dir = run_directory(A_CELL, root=str(tmp_path)) / "attempt_001"
    manifest = json.loads((attempt_dir / "artifact_manifest.json").read_text())
    from when_tta_hurts.result_artifacts import verify_artifact_manifest

    verify_artifact_manifest(attempt_dir, manifest)  # must not raise

    covered = {e["path"] for e in manifest["artifacts"]}
    assert covered == set(REQUIRED_COMPLETION_ARTIFACTS)

    for filename in ALL_REQUIRED_ARTIFACT_FILENAMES:
        assert (attempt_dir / filename).exists()


def test_atomic_completion_ordering_status_completed_only_after_artifacts(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _run(A_CELL, tmp_path, ledger_path, amendments_path)
    attempt_dir = run_directory(A_CELL, root=str(tmp_path)) / "attempt_001"
    status = json.loads((attempt_dir / "status.json").read_text())
    assert status["status"] == "completed"
    # All required artifacts exist alongside a completed status.
    for filename in ALL_REQUIRED_ARTIFACT_FILENAMES:
        assert (attempt_dir / filename).exists()


def test_persistence_failure_makes_attempt_failed_not_completed(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"

    def _broken_persist(*args, **kwargs):
        raise PersistenceVerificationError("simulated persistence failure")

    monkeypatch.setattr(orch, "persist_and_verify_completion", _broken_persist)

    with pytest.raises(PersistenceVerificationError):
        _run(A_CELL, tmp_path, ledger_path, amendments_path)

    attempt_dir = run_directory(A_CELL, root=str(tmp_path)) / "attempt_001"
    status = json.loads((attempt_dir / "status.json").read_text())
    assert status["status"] == "failed"
    assert "persistence verification failed" in status["failure_reason"]

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"


# --- eligibility overlay ---


def test_amendment_makes_completed_attempt_noncanonical_and_next_attempt_is_2(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    first = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    assert first.status == "completed" and first.attempt_number == 1

    ledger_module.append_amendment_entry(
        run_id=A_CELL.run_id(),
        attempt_id=1,
        original_status="completed",
        canonical_eligible=False,
        amendment_type="engineering_observability_failure",
        reason="synthetic test amendment",
        validation_metrics_computed=True,
        validation_metrics_persisted=False,
        validation_metrics_inspected=False,
        test_metrics_observed=False,
        tta_metrics_observed=False,
        source_commit="deadbeef",
        checkpoint_hash=first.checkpoint_hash,
        recorded_at="2026-01-01T00:00:00Z",
        ledger_path=amendments_path,
    )

    second = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    assert second.status == "completed"
    assert second.attempt_number == 2  # not skipped, not attempt 1 again

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 2
    assert rows[0]["attempt_id"] == "1"
    assert rows[1]["attempt_id"] == "2"


def test_ineligible_completion_does_not_skip(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    first = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    ledger_module.append_amendment_entry(
        run_id=A_CELL.run_id(),
        attempt_id=1,
        original_status="completed",
        canonical_eligible=False,
        amendment_type="engineering_observability_failure",
        reason="synthetic test amendment",
        validation_metrics_computed=True,
        validation_metrics_persisted=False,
        validation_metrics_inspected=False,
        test_metrics_observed=False,
        tta_metrics_observed=False,
        source_commit="deadbeef",
        checkpoint_hash=first.checkpoint_hash,
        recorded_at="2026-01-01T00:00:00Z",
        ledger_path=amendments_path,
    )
    skip = orch.check_confirmatory_skip(A_CELL, root=str(tmp_path), amendments_ledger_path=amendments_path)
    assert skip is None  # must NOT skip -- ineligible


def test_matching_eligible_completion_skips_before_loader_model_device_factories(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    first = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    assert first.status == "completed"

    def _forbidden_loader_factory(cell):
        raise AssertionError("loader_factory must never be called on an eligible skip")

    def _forbidden_device_resolver():
        raise AssertionError("device_resolver must never be called on an eligible skip")

    result = orch.run_canary_cell(
        A_CELL.run_id(),
        loader_factory=_forbidden_loader_factory,
        device_resolver=_forbidden_device_resolver,
        require_clean_tree=False,
        root=str(tmp_path),
        confirmatory_ledger_path=ledger_path,
        amendments_ledger_path=amendments_path,
    )
    assert result.status == "skipped_completed"
    assert result.attempt_number == 1


# --- working-tree policy ---


def test_append_only_ledger_dirtiness_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        orch,
        "_git_status_porcelain",
        lambda: " M artifacts/ledger_confirmatory.csv\n",
    )
    head_content = "header\n"
    working_content = "header\nnew_row\n"
    monkeypatch.setattr(orch, "_git_show_head", lambda path: head_content)
    monkeypatch.setattr(orch.Path, "read_text", lambda self: working_content)
    orch.require_clean_working_tree()  # must not raise


def test_dirty_source_or_config_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M src/when_tta_hurts/matrix.py\n")
    with pytest.raises(orch.DirtyWorkingTreeError):
        orch.require_clean_working_tree()


def test_modified_ledger_content_not_a_strict_prefix_is_rejected(monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M artifacts/ledger_confirmatory.csv\n")
    monkeypatch.setattr(orch, "_git_show_head", lambda path: "header\nrow1\n")
    # working content EDITS row1 instead of appending -- not a prefix extension.
    monkeypatch.setattr(orch.Path, "read_text", lambda self: "header\nrow1_edited\n")
    with pytest.raises(orch.DirtyWorkingTreeError):
        orch.require_clean_working_tree()


def test_deleted_ledger_content_is_rejected(monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M artifacts/ledger_confirmatory.csv\n")
    monkeypatch.setattr(orch, "_git_show_head", lambda path: "header\nrow1\nrow2\n")
    monkeypatch.setattr(orch.Path, "read_text", lambda self: "header\nrow1\n")  # row2 deleted
    with pytest.raises(orch.DirtyWorkingTreeError):
        orch.require_clean_working_tree()


def test_reordered_ledger_content_is_rejected(monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M artifacts/ledger_confirmatory.csv\n")
    monkeypatch.setattr(orch, "_git_show_head", lambda path: "header\nrow1\nrow2\n")
    monkeypatch.setattr(orch.Path, "read_text", lambda self: "header\nrow2\nrow1\n")  # reordered
    with pytest.raises(orch.DirtyWorkingTreeError):
        orch.require_clean_working_tree()


def test_untracked_file_is_rejected(monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: "?? some_new_file.py\n")
    with pytest.raises(orch.DirtyWorkingTreeError):
        orch.require_clean_working_tree()


# --- sequential synthetic cells, no committing between them ---


def test_sequential_synthetic_cells_without_committing(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"

    cell_b = MatrixCell(
        block="A_core_normalization_resolution",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        training_policy="none",
        seed=1,
    )

    r1 = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    r2 = _run(cell_b, tmp_path, ledger_path, amendments_path)
    assert r1.status == "completed"
    assert r2.status == "completed"
    assert r1.run_id != r2.run_id

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 2
    run_ids = {row["run_id"] for row in rows}
    assert run_ids == {A_CELL.run_id(), cell_b.run_id()}


def test_duplicate_ledger_rows_not_created_on_repeated_eligible_skip(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _run(A_CELL, tmp_path, ledger_path, amendments_path)
    second = _run(A_CELL, tmp_path, ledger_path, amendments_path)
    assert second.status == "skipped_completed"
    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1


# --- final-test / test-firewall unaffected ---


def test_final_test_still_locked():
    from when_tta_hurts.authorization import AuthorizationError

    with pytest.raises(AuthorizationError):
        orch.run_final_test()


def test_no_allow_test_true_call_site_in_orchestrator_or_result_artifacts():
    import inspect

    from when_tta_hurts import result_artifacts

    for module in (orch, result_artifacts):
        source = inspect.getsource(module)
        assert "allow_test=True" not in source
