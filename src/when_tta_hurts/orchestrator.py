"""Confirmatory-matrix orchestrator core logic. Three modes:

- plan: parse + validate + expand the matrix, print it, NO side effects
  (no filesystem writes, no dataset access, no model construction, no
  ledger writes). This is the ONLY mode exercised in Phase 2B.2.
- train-validation: train/evaluate blocks A/B/C(/D) using train+validation
  data ONLY -- implemented here for Phase 2B.2's testing requirements, but
  NOT invoked against real data in this phase (tests use synthetic tensors
  and temporary directories only).
- final-test: requires a committed, tracked-and-clean final-evaluation
  authorization artifact (authorization.py) -- since that artifact does
  not exist, this mode always raises AuthorizationError before touching
  any data. Not invoked at all in Phase 2B.2.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from when_tta_hurts.authorization import verify_authorization
from when_tta_hurts.matrix import MatrixCell, parse_and_validate_matrix
from when_tta_hurts.models.resnet import build_resnet18_small_input
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything
from when_tta_hurts.run_identity import (
    ConflictingCompletedRunError,
    RunStatus,
    cell_config_hash,
    find_completed_attempt,
    finish_attempt,
    start_attempt,
)
from when_tta_hurts.training import EarlyStoppingConfig, TrainingOOMError, train_model
from when_tta_hurts.transforms.policies import build_policy

# Confirmatory runs never load any pretrained/existing checkpoint as a
# starting point (_build_model always constructs fresh, untrained weights)
# -- this structurally satisfies "reject pilot checkpoints/artifacts" for
# training. run_identity.reject_pilot_artifact() is the explicit guard
# available for any future code path that DOES load a checkpoint path
# (e.g. a resume feature), and is tested directly in test_run_identity.py.


class UnfavorableRerunRefusedError(RuntimeError):
    """Raised if a rerun is requested for a run_id whose confirmatory
    ledger already shows test_metrics_observed=True -- reruns after test
    metrics exist are never permitted, regardless of the stated reason."""


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def print_plan(
    matrix_path: str = "configs/experiment_matrix.yaml", block_d_gate_passed: bool = False
) -> list[str]:
    """PLAN MODE. Parses and expands the matrix and returns/prints a list
    of one-line descriptions, in committed order. Performs NO filesystem
    writes, NO dataset access, NO model construction, NO ledger writes --
    verified by tests/test_orchestrator_plan_mode.py.
    """
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=block_d_gate_passed)
    lines = []
    for cell in expanded.cells:
        lines.append(
            f"{cell.block}\t{cell.run_id()}\t{cell.dataset}\t{cell.resolution}px\t"
            f"{cell.model}\t{cell.normalization}\tpolicy={cell.training_policy}\tseed={cell.seed}"
        )
    for line in lines:
        print(line)
    print(f"\nTotal cells: {len(expanded.cells)} (block D included: {expanded.block_d_included})")
    return lines


def unmatched_comparison_cell_for(block_b_cell: MatrixCell) -> MatrixCell:
    """Block B's matched-policy checkpoints are compared against Block A's
    ALREADY-EXISTING unmatched checkpoint for the same dataset/resolution/
    normalization/seed -- this returns that Block A cell WITHOUT causing
    any retraining. Per docs/phase2b_protocol.md / configs/experiment_matrix.yaml:
    reuses_checkpoints_from: A_core_normalization_resolution.
    """
    if block_b_cell.block != "B_policy_matching":
        raise ValueError("unmatched_comparison_cell_for() only applies to Block B cells")
    return MatrixCell(
        block="A_core_normalization_resolution",
        dataset=block_b_cell.dataset,
        resolution=block_b_cell.resolution,
        model=block_b_cell.model,
        normalization=block_b_cell.normalization,
        training_policy="none",
        seed=block_b_cell.seed,
    )


def _build_model(cell: MatrixCell) -> torch.nn.Module:
    if cell.model == "small_cnn":
        return build_small_cnn(num_classes=9, normalization=cell.normalization)
    if cell.model == "resnet18":
        return build_resnet18_small_input(num_classes=9)
    raise ValueError(f"Unknown model '{cell.model}'")


@dataclass
class CellTrainResult:
    status: str  # "skipped_completed" | "completed" | "failed"
    run_id: str
    attempt_number: int | None
    checkpoint_hash: str | None
    reason: str | None = None


def run_train_validation_cell(
    cell: MatrixCell,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    root: str = "artifacts/confirmatory",
    max_training_seconds: float | None = None,
    ledger_check_test_metrics: Callable[[str], bool] | None = None,
) -> CellTrainResult:
    """TRAIN-VALIDATION MODE for one cell. Uses train_loader/val_loader
    ONLY -- never a test loader (no test-loader parameter exists on this
    function at all, structurally impossible to pass one). NOT invoked
    against real data in Phase 2B.2 -- exercised only with synthetic
    tensors in tests.

    `ledger_check_test_metrics`: optional callable(run_id) -> bool, used to
    enforce the "no rerun after test metrics observed" rule; if it returns
    True for this cell's run_id, refuses to proceed.
    """
    if cell.seed == 314159:
        raise ValueError("Refusing to train: seed 314159 is permanently excluded from confirmatory runs.")

    run_id = cell.run_id()
    if ledger_check_test_metrics is not None and ledger_check_test_metrics(run_id):
        raise UnfavorableRerunRefusedError(
            f"Run {run_id} already has test metrics observed in the confirmatory ledger -- "
            f"reruns after test metrics exist are never permitted."
        )

    existing = find_completed_attempt(cell, root)
    this_hash = cell_config_hash(cell)
    if existing is not None:
        if existing["config_hash"] == this_hash:
            return CellTrainResult(
                status="skipped_completed",
                run_id=run_id,
                attempt_number=existing["attempt_number"],
                checkpoint_hash=None,
                reason="matching completed attempt already exists",
            )
        raise ConflictingCompletedRunError(
            f"Run {run_id} has a completed attempt with a different config hash."
        )

    attempt_dir, status = start_attempt(cell, root)
    try:
        seed_everything(cell.seed)
        model = _build_model(cell).to(device)

        augmentation_policy = None
        augmentation_seed = None
        if cell.training_policy == "matched_to_approved_tta_policy":
            augmentation_policy = build_policy("mixed")
            augmentation_seed = cell.seed * 1000  # deterministic, distinct per seed

        result = train_model(
            model,
            train_loader,
            val_loader,
            device,
            early_stopping=EarlyStoppingConfig(patience=5, min_delta=0.0),
            augmentation_policy=augmentation_policy,
            augmentation_seed=augmentation_seed,
            max_training_seconds=max_training_seconds,
        )

        from when_tta_hurts.artifacts import save_checkpoint

        ckpt_hash = save_checkpoint(result.best_state_dict, attempt_dir / "best_checkpoint.pt")
        finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
        return CellTrainResult(
            status="completed", run_id=run_id, attempt_number=status.attempt_number, checkpoint_hash=ckpt_hash
        )
    except TrainingOOMError as e:
        finish_attempt(attempt_dir, status, RunStatus.FAILED, failure_reason=f"OOM: {e}")
        raise
    except Exception as e:
        finish_attempt(attempt_dir, status, RunStatus.FAILED, failure_reason=str(e))
        raise


def run_final_test(authorization_artifact_path: str = "configs/final_evaluation_authorization.yaml") -> None:
    """FINAL-TEST MODE. Verifies authorization FIRST -- before any dataset
    construction. Since the real authorization artifact does not exist in
    Phase 2B.2, this always raises AuthorizationError. Never invoked in
    this phase.
    """
    verify_authorization(authorization_artifact_path)  # will raise -- artifact does not exist
    raise NotImplementedError(
        "Final-test evaluation logic is intentionally not implemented in Phase 2B.2 -- "
        "reaching this point would require Validation-Gated TTA to be frozen and a real "
        "authorization artifact to exist, neither of which is true yet."
    )
