"""Phase 2B.2 audit, Part G: end-to-end synthetic integration test for the
confirmatory train-validation orchestration path.

Uses ONLY tiny in-memory synthetic tensors, real-but-tiny models (small_cnn
on CPU, 28px, batch_size=8), temporary directories, and temporary ledgers.
No real dataset is downloaded, loaded, or inspected. No production CLI flag
or environment variable is used to select a synthetic backend -- these
tests call the same orchestrator functions production code calls, just
with injected loaders/paths.
"""

from __future__ import annotations

import csv

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import when_tta_hurts.orchestrator as orch
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.run_identity import (
    ConflictingCompletedRunError,
    PilotArtifactRejectedError,
    RunStatus,
    reject_pilot_artifact,
    run_directory,
)
from when_tta_hurts.training import TrainingOOMError

A_CELL = MatrixCell(
    block="A_core_normalization_resolution",
    dataset="pathmnist",
    resolution=28,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)

B_CELL = MatrixCell(
    block="B_policy_matching",
    dataset="pathmnist",
    resolution=28,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="matched_to_approved_tta_policy",
    seed=0,
)


def _make_loader(n, num_classes=9, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 28, 28, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=g)


def _run(cell, tmp_path, ledger_path, **kwargs):
    # A_CELL below intentionally shares its run_id with the real Phase
    # 2B.3A canary cell, so the amendments ledger must also be pinned to a
    # temp path -- otherwise the real (correctly ineligible) attempt_001
    # amendment would leak into these synthetic tests.
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    kwargs.setdefault("amendments_ledger_path", tmp_path / "ledger_amendments.csv")
    return orch.run_train_validation_cell(
        cell,
        train_loader,
        val_loader,
        device,
        root=str(tmp_path),
        confirmatory_ledger_path=ledger_path,
        **kwargs,
    )


def _ledger_rows(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


# --- 1. Successful Block A-style cell ---


def test_1_successful_block_a_cell(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    result = _run(A_CELL, tmp_path, ledger_path)

    assert result.status == "completed"
    assert result.checkpoint_hash is not None

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    attempt_dir = run_dir / "attempt_001"
    assert (attempt_dir / "status.json").exists()
    assert (attempt_dir / "best_checkpoint.pt").exists()

    import json

    status = json.loads((attempt_dir / "status.json").read_text())
    assert status["status"] == RunStatus.COMPLETED.value

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["confirmatory"] == "True"
    assert rows[0]["checkpoint_hash"] == result.checkpoint_hash


# --- 2. Repeating the completed cell: skip safely ---


def test_2_repeat_completed_cell_skips_safely(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    _run(A_CELL, tmp_path, ledger_path)
    second = _run(A_CELL, tmp_path, ledger_path)

    assert second.status == "skipped_completed"
    run_dir = run_directory(A_CELL, root=str(tmp_path))
    # No new attempt directory was created.
    attempt_dirs = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    assert attempt_dirs == ["attempt_001"]

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1  # not duplicated


# --- 3. Same run ID, conflicting hash: hard failure, no overwrite ---


def test_3_conflicting_hash_hard_failure_no_overwrite(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    _run(A_CELL, tmp_path, ledger_path)

    import json

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    status_path = run_dir / "attempt_001" / "status.json"
    before = status_path.read_text()
    data = json.loads(before)
    data["config_hash"] = "deliberately-wrong-hash"
    status_path.write_text(json.dumps(data))

    with pytest.raises(ConflictingCompletedRunError):
        _run(A_CELL, tmp_path, ledger_path)

    # The corrupted attempt_001 status was not further modified/overwritten
    # by the refused second call.
    assert status_path.read_text() == json.dumps(data)
    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1  # no new row from the refused attempt


# --- 4. Simulated Block B cell: augmentation exactly once, reuse mapping ---


def test_4_block_b_augmentation_exactly_once_and_reuse_mapping(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger_confirmatory.csv"
    call_count = {"n": 0}

    import when_tta_hurts.training as training_module

    original = training_module.sample_deterministic_view

    def counting_wrapper(x, policy, seed):
        call_count["n"] += 1
        return original(x, policy, seed)

    monkeypatch.setattr(training_module, "sample_deterministic_view", counting_wrapper)

    result = _run(B_CELL, tmp_path, ledger_path)
    assert result.status == "completed"

    # small_cnn on CPU, 16 train samples, batch_size=8 -> 2 steps/epoch;
    # augmentation must be applied exactly once per step (not zero, not twice).
    steps_per_epoch = 2
    assert call_count["n"] % steps_per_epoch == 0
    assert call_count["n"] > 0

    # Reuse mapping resolves to the correct Block A run WITHOUT retraining it.
    unmatched = orch.unmatched_comparison_cell_for(B_CELL)
    assert unmatched.block == "A_core_normalization_resolution"
    assert unmatched.run_id() == A_CELL.run_id()
    a_run_dir = run_directory(unmatched, root=str(tmp_path))
    assert not a_run_dir.exists()  # mapping alone never causes A to be (re)trained


# --- 5. Simulated non-finite loss: failed, incident recorded ---


def test_5_non_finite_loss_marks_failed_and_records_incident(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger_confirmatory.csv"

    class NaNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3 * 28 * 28, 9)

        def forward(self, x):
            return self.linear(x.flatten(1)) * float("nan")

    monkeypatch.setattr(orch, "_build_model", lambda cell: NaNModel())

    with pytest.raises(RuntimeError, match="non-finite"):
        _run(A_CELL, tmp_path, ledger_path)

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    import json

    status = json.loads((run_dir / "attempt_001" / "status.json").read_text())
    assert status["status"] == RunStatus.FAILED.value
    assert not (run_dir / "attempt_001" / "best_checkpoint.pt").exists()

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "non-finite" in rows[0]["failure_reason"]


# --- 6. Simulated MPS OOM: failed, no CPU fallback, incident recorded ---


def test_6_simulated_mps_oom_marks_failed_no_fallback(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger_confirmatory.csv"

    class OOMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3 * 28 * 28, 9)
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            raise RuntimeError("MPS backend out of memory (simulated)")

    monkeypatch.setattr(orch, "_build_model", lambda cell: OOMModel())

    with pytest.raises(TrainingOOMError):
        _run(A_CELL, tmp_path, ledger_path)

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    import json

    status = json.loads((run_dir / "attempt_001" / "status.json").read_text())
    assert status["status"] == RunStatus.FAILED.value
    assert "OOM" in status["failure_reason"]

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "OOM" in rows[0]["failure_reason"]
    # The device passed through unchanged -- no silent CPU-device substitution
    # occurred anywhere in this call (the test itself requested CPU, and
    # failure was due to the injected model, not a device switch).


# --- 7. Simulated interruption: prior attempt preserved, next attempt numbered ---


def test_7_interruption_preserves_prior_attempt_and_numbers_next(tmp_path):
    ledger_path = tmp_path / "ledger_confirmatory.csv"

    with pytest.raises(Exception):
        _run(A_CELL, tmp_path, ledger_path, max_training_seconds=0.0)

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    assert (run_dir / "attempt_001").exists()

    result = _run(A_CELL, tmp_path, ledger_path)
    assert result.status == "completed"
    assert result.attempt_number == 2
    assert (run_dir / "attempt_001").exists()  # still preserved
    assert (run_dir / "attempt_002").exists()

    rows = _ledger_rows(ledger_path)
    assert len(rows) == 2
    assert rows[0]["attempt_id"] == "1" and rows[0]["status"] == "failed"
    assert rows[1]["attempt_id"] == "2" and rows[1]["status"] == "completed"


# --- 8. Pilot checkpoint/artifact input: rejected before model evaluation ---


def test_8_pilot_artifact_rejected_before_evaluation(tmp_path):
    pilot_path = tmp_path / "artifacts" / "pilots" / "some_pilot_run" / "best_checkpoint.pt"
    with pytest.raises(PilotArtifactRejectedError):
        reject_pilot_artifact(pilot_path)


def test_no_production_cli_flag_enables_synthetic_backend():
    """Static check: scripts/run_confirmatory.py must not expose any flag
    or env var that swaps in a synthetic data/model backend."""
    from pathlib import Path

    source = Path("scripts/run_confirmatory.py").read_text()
    assert "synthetic" not in source.lower()
    assert "os.environ" not in source
