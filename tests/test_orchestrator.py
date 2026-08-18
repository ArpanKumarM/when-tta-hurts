"""Tests for orchestrator.py. Plan mode is exercised for real (it's
side-effect-free by design); train-validation mode is exercised ONLY with
synthetic tensors/temp dirs; final-test mode is exercised only to confirm
it fails closed without touching data."""

import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from when_tta_hurts.authorization import AuthorizationError
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.orchestrator import (
    CellTrainResult,
    FinalTestNotYetImplementedError,
    UnfavorableRerunRefusedError,
    print_plan,
    run_final_test,
    run_train_validation_cell,
    unmatched_comparison_cell_for,
)
from when_tta_hurts.run_identity import ConflictingCompletedRunError


def _run_cell(cell, train_loader, val_loader, device, **kwargs):
    """Wrapper forcing EVERY call in this file onto a temporary
    confirmatory ledger derived from `root` -- never
    artifacts/ledger_confirmatory.csv."""
    from pathlib import Path

    kwargs.setdefault("confirmatory_ledger_path", Path(kwargs["root"]) / "ledger_confirmatory.csv")
    return run_train_validation_cell(cell, train_loader, val_loader, device, **kwargs)


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


# --- Plan mode: real matrix, must be side-effect-free ---


def test_plan_mode_no_filesystem_side_effects(tmp_path, capsys):
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        # matrix path is relative to repo root normally; pass absolute path
        # to the real committed matrix so parsing succeeds from a clean tmp cwd.
        lines = print_plan(os.path.join(cwd, "configs/experiment_matrix.yaml"))
        assert len(lines) == 33
        # confirm NOTHING was created in the tmp cwd we're standing in.
        assert list(tmp_path.iterdir()) == []
    finally:
        os.chdir(cwd)


def test_plan_mode_no_artifacts_confirmatory_dir_created():
    assert not os.path.exists("artifacts/confirmatory")
    print_plan()
    assert not os.path.exists("artifacts/confirmatory")


def test_plan_mode_returns_39_with_block_d():
    lines = print_plan(block_d_gate_passed=True)
    assert len(lines) == 39


# --- unmatched checkpoint reuse mapping ---


def test_unmatched_comparison_cell_for_maps_to_block_a():
    unmatched = unmatched_comparison_cell_for(B_CELL)
    assert unmatched.block == "A_core_normalization_resolution"
    assert unmatched.training_policy == "none"
    assert unmatched.dataset == B_CELL.dataset
    assert unmatched.resolution == B_CELL.resolution
    assert unmatched.normalization == B_CELL.normalization
    assert unmatched.seed == B_CELL.seed


def test_unmatched_comparison_cell_for_rejects_non_block_b():
    a_cell = MatrixCell(
        block="A_core_normalization_resolution",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        training_policy="none",
        seed=0,
    )
    with pytest.raises(ValueError):
        unmatched_comparison_cell_for(a_cell)


# --- train-validation mode: synthetic data + temp dirs ONLY ---

A_CELL = MatrixCell(
    block="A_core_normalization_resolution",
    dataset="pathmnist",
    resolution=28,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)


def test_run_train_validation_cell_completes_with_synthetic_data(tmp_path):
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    result = _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path))
    assert isinstance(result, CellTrainResult)
    assert result.status == "completed"
    assert result.checkpoint_hash is not None


def test_run_train_validation_cell_skips_matching_completed(tmp_path):
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path))
    second = _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path))
    assert second.status == "skipped_completed"


def test_run_train_validation_cell_rejects_seed_314159(tmp_path):
    bad_cell = MatrixCell(**{**A_CELL.__dict__, "seed": 314159})
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    with pytest.raises(ValueError, match="314159"):
        _run_cell(bad_cell, train_loader, val_loader, device, root=str(tmp_path))


def test_run_train_validation_cell_refuses_rerun_after_test_metrics(tmp_path):
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    with pytest.raises(UnfavorableRerunRefusedError):
        _run_cell(
            A_CELL,
            train_loader,
            val_loader,
            device,
            root=str(tmp_path),
            ledger_check_test_metrics=lambda run_id: True,  # simulate test metrics already observed
        )


def test_run_train_validation_cell_conflicting_hash_hard_failure(tmp_path):
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path))

    # Corrupt the stored hash to simulate protocol drift.
    import json

    from when_tta_hurts.run_identity import run_directory

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    status_path = run_dir / "attempt_001" / "status.json"
    data = json.loads(status_path.read_text())
    data["config_hash"] = "deliberately-wrong"
    status_path.write_text(json.dumps(data))

    with pytest.raises(ConflictingCompletedRunError):
        _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path))


def test_run_train_validation_cell_preserves_failed_attempt(tmp_path):
    """A failed attempt (e.g. from a bad max_training_seconds) must remain
    on disk, and a subsequent successful call gets a new attempt number."""
    device = torch.device("cpu")
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)

    with pytest.raises(Exception):
        _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path), max_training_seconds=0.0)

    from when_tta_hurts.run_identity import run_directory

    run_dir = run_directory(A_CELL, root=str(tmp_path))
    assert (run_dir / "attempt_001").exists()  # preserved

    result = _run_cell(A_CELL, train_loader, val_loader, device, root=str(tmp_path))
    assert result.status == "completed"
    assert result.attempt_number == 2


# --- final-test mode: must fail closed before touching any data ---


def test_run_final_test_fails_closed_without_authorization():
    with pytest.raises(AuthorizationError):
        run_final_test()


def test_run_final_test_never_reaches_locked_error():
    """Confirms the failure happens at the authorization check, not later
    -- i.e. no dataset-related code executes first, and the domain-specific
    locked error is unreachable while unauthorized."""
    try:
        run_final_test()
        raise AssertionError("expected AuthorizationError")
    except AuthorizationError as e:
        assert "does not exist" in str(e)
    except FinalTestNotYetImplementedError:
        raise AssertionError(
            "reached FinalTestNotYetImplementedError -- authorization gate was bypassed!"
        ) from None


def test_final_test_locked_error_is_domain_specific_not_generic():
    """FinalTestNotYetImplementedError must not be a bare NotImplementedError
    -- reaching this lock must be distinguishable from an accidental stub."""
    assert not issubclass(FinalTestNotYetImplementedError, NotImplementedError)
    assert issubclass(FinalTestNotYetImplementedError, RuntimeError)
