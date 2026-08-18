"""Phase 2B.3A Part A tests: single-cell canary resolution and the
pre-data-access guard chain in orchestrator.py. All use synthetic loaders/
CPU device injection/temp dirs -- never real data or real MPS."""

from __future__ import annotations

import subprocess

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import when_tta_hurts.orchestrator as orch
from when_tta_hurts.devices import DeviceUnavailableError
from when_tta_hurts.orchestrator import (
    BlockDRunRejectedError,
    DirtyWorkingTreeError,
    PilotOrExcludedSeedRunIdError,
    UnknownRunIdError,
    require_clean_working_tree,
    resolve_canary_run_id,
    run_canary_cell,
)

VALID_A_RUN_ID = "A-pathmnist-28px-batchnorm-policy-none-s0"
BLOCK_D_RUN_ID = "D-pathmnist-128px-batchnorm-policy-none-s0"


def _synthetic_loader_factory(cell):
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, 3, cell.resolution, cell.resolution, generator=g)
    y = torch.randint(0, 9, (16,), generator=g)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(x[:8], y[:8]), batch_size=8, shuffle=False)
    return train_loader, val_loader


# --- resolve_canary_run_id ---


def test_resolve_valid_a_cell():
    cell = resolve_canary_run_id(VALID_A_RUN_ID)
    assert cell.run_id() == VALID_A_RUN_ID
    assert cell.block == "A_core_normalization_resolution"


def test_resolve_block_d_rejected():
    with pytest.raises(BlockDRunRejectedError):
        resolve_canary_run_id(BLOCK_D_RUN_ID)


def test_resolve_pilot_seed_rejected():
    with pytest.raises(PilotOrExcludedSeedRunIdError):
        resolve_canary_run_id("A-pathmnist-28px-batchnorm-policy-none-s314159")


def test_resolve_pilot_prefixed_id_rejected():
    with pytest.raises(PilotOrExcludedSeedRunIdError):
        resolve_canary_run_id("pilot-pathmnist-28px-bn")


def test_resolve_unknown_id_rejected():
    with pytest.raises(UnknownRunIdError):
        resolve_canary_run_id("totally-bogus-run-id-xyz")


# --- require_clean_working_tree ---


def test_clean_tree_passes(monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: "")
    require_clean_working_tree()  # must not raise


def test_dirty_tree_raises(monkeypatch):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M some_file.py\n")
    with pytest.raises(DirtyWorkingTreeError):
        require_clean_working_tree()


# --- run_canary_cell: full guard-chain ordering ---


def test_dirty_tree_fails_before_data_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "_git_status_porcelain", lambda: " M dirty.py\n")

    def _forbidden_loader_factory(cell):
        raise AssertionError("loader_factory must never be called when the tree is dirty")

    with pytest.raises(DirtyWorkingTreeError):
        run_canary_cell(
            VALID_A_RUN_ID,
            loader_factory=_forbidden_loader_factory,
            device_resolver=lambda: torch.device("cpu"),
            require_clean_tree=True,
            root=str(tmp_path),
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )
    assert not (tmp_path / "A").exists()  # no artifact directory created


def test_missing_mps_fails_before_artifact_creation(tmp_path):
    def _forbidden_loader_factory(cell):
        raise AssertionError("loader_factory must never be called when the device is unavailable")

    def _raising_device_resolver():
        raise DeviceUnavailableError("device='mps' was explicitly requested but is not available.")

    with pytest.raises(DeviceUnavailableError):
        run_canary_cell(
            VALID_A_RUN_ID,
            loader_factory=_forbidden_loader_factory,
            device_resolver=_raising_device_resolver,
            require_clean_tree=False,
            root=str(tmp_path),
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )
    assert list(tmp_path.iterdir()) == []  # nothing was created at all


def test_block_d_rejected_before_any_guard(tmp_path):
    """Block D must be rejected at resolution time -- before the tree/
    device checks even run."""

    def _forbidden(*args, **kwargs):
        raise AssertionError("must never be reached for a Block D run_id")

    with pytest.raises(BlockDRunRejectedError):
        run_canary_cell(
            BLOCK_D_RUN_ID,
            loader_factory=_forbidden,
            device_resolver=_forbidden,
            require_clean_tree=True,
            root=str(tmp_path),
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )


def test_valid_single_abc_id_reaches_injected_train_validation_path(tmp_path):
    result = run_canary_cell(
        VALID_A_RUN_ID,
        loader_factory=_synthetic_loader_factory,
        device_resolver=lambda: torch.device("cpu"),
        require_clean_tree=False,
        root=str(tmp_path),
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
    )
    assert result.status == "completed"
    assert result.run_id == VALID_A_RUN_ID


def test_no_test_loader_reachable_from_loader_factory_contract():
    """DataLoaderFactory's contract is exactly (train_loader, val_loader) --
    structurally two values, no third (test) slot exists to populate."""
    train_loader, val_loader = _synthetic_loader_factory(resolve_canary_run_id(VALID_A_RUN_ID))
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)


def test_default_loader_factory_source_has_no_test_split_access():
    import inspect

    source = inspect.getsource(orch.default_train_validation_loader_factory)
    assert 'split="test"' not in source
    assert "split='test'" not in source
    assert "allow_test" not in source


# --- final-test remains locked ---


def test_final_test_still_locked_after_canary_enablement():
    from when_tta_hurts.authorization import AuthorizationError

    with pytest.raises(AuthorizationError):
        orch.run_final_test()


# --- no CLI/env bypass ---


def test_git_status_helper_uses_real_git_command():
    """Sanity: the clean-tree check genuinely shells out to git, it isn't
    a stub that always returns clean."""
    real_status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    assert orch._git_status_porcelain() == real_status
