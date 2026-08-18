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
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from when_tta_hurts import ledger as ledger_module
from when_tta_hurts.authorization import verify_authorization
from when_tta_hurts.data import load_pilot_split
from when_tta_hurts.dataset_verification import DEFAULT_DATA_ROOT, verify_official_dataset_artifact
from when_tta_hurts.devices import select_device
from when_tta_hurts.matrix import FROZEN_TRAINING_SETTINGS, MatrixCell, parse_and_validate_matrix
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

# The commit at which docs/phase2b_protocol.md and configs/experiment_matrix.yaml
# were frozen (Phase 2B.1) -- recorded on every confirmatory ledger row so a
# row is always traceable to the exact frozen protocol version that governed
# it, independent of whatever commit later touched orchestrator.py itself.
FROZEN_PROTOCOL_COMMIT = "ce4c962"

DataLoaderFactory = Callable[[MatrixCell], tuple[DataLoader, DataLoader]]

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


class UnknownRunIdError(RuntimeError):
    """Raised when a --run-id does not match any cell in the approved
    unconditional (Block A/B/C) matrix expansion -- covers genuinely
    unknown IDs as well as any malformed/hand-edited ID that happens not
    to collide with a real cell."""


class BlockDRunRejectedError(RuntimeError):
    """Raised when a --run-id resolves to a Block D (conditional, 128px)
    cell. Block D requires its own gate evaluation (block_d_gate.py) and
    is never authorized for single-cell canary/production execution via
    this path -- Phase 2B.3A explicitly excludes Block D."""


class PilotOrExcludedSeedRunIdError(RuntimeError):
    """Raised immediately, before any matrix lookup, when a --run-id
    references the permanently-excluded pilot seed (314159) or otherwise
    looks like a pilot identifier -- these can never be valid confirmatory
    targets regardless of what the matrix contains."""


class DirtyWorkingTreeError(RuntimeError):
    """Raised when the working tree is not clean at canary-execution time
    -- training must never run against an uncommitted/uncertain code
    state, so this is checked before any dataset access."""


def _git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def require_clean_working_tree() -> None:
    status = _git_status_porcelain()
    if status.strip():
        raise DirtyWorkingTreeError(
            f"Refusing to execute against a dirty working tree. `git status --porcelain` output:\n{status}"
        )


def resolve_canary_run_id(run_id: str, matrix_path: str = "configs/experiment_matrix.yaml") -> MatrixCell:
    """Resolve EXACTLY one --run-id to its approved unconditional (Block
    A/B/C) matrix cell. Rejects: pilot/permanently-excluded-seed IDs
    (before any matrix parsing), Block D IDs (explicitly, with a distinct
    error from "unknown"), and anything not present in the committed
    matrix expansion. Never matches more than one cell (run IDs are
    unique by construction -- see matrix.py::MatrixCell.run_id())."""
    if "-s314159" in run_id or run_id.startswith("pilot"):
        raise PilotOrExcludedSeedRunIdError(
            f"Refusing run_id '{run_id}': pilot/permanently-excluded-seed identifiers "
            f"are never valid confirmatory targets."
        )
    full = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    for cell in full.cells:
        if cell.run_id() == run_id:
            if cell.block == "D_conditional_128px":
                raise BlockDRunRejectedError(
                    f"Run '{run_id}' belongs to Block D, which is not authorized for "
                    f"single-cell canary execution -- Block D requires its own gate "
                    f"evaluation and is out of scope for Phase 2B.3A."
                )
            return cell
    raise UnknownRunIdError(
        f"'{run_id}' does not match any approved unconditional (Block A/B/C) matrix cell."
    )


class FinalTestNotYetImplementedError(RuntimeError):
    """Raised by run_final_test() AFTER authorization has been verified
    (so this is never reachable while unauthorized -- it can only ever
    fire once an authorization artifact legitimately exists). Deliberately
    NOT a generic/accidental NotImplementedError: reaching this point is an
    intentional, documented lock, not a bug. Final-test evaluation logic
    must not be implemented until Validation-Gated TTA is frozen -- see
    docs/phase2b_protocol.md. Implementing it earlier would mean writing
    test-split-touching code before the analysis method it evaluates is
    even decided, which is exactly the kind of test-split proximity this
    project's firewall discipline exists to prevent."""


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


def default_train_validation_loader_factory(
    cell: MatrixCell, root: str | Path = DEFAULT_DATA_ROOT
) -> tuple[DataLoader, DataLoader]:
    """PRODUCTION data-loading path for a confirmatory matrix cell.

    Verifies the official checksummed NATIVE-resolution artifact BEFORE
    constructing any DataLoader -- fails closed (ArtifactVerificationError)
    on a missing file, a checksum mismatch, or an unsupported dataset/
    resolution, so a resized proxy or corrupt download can never reach a
    training loop. Loads train/val splits ONLY via load_pilot_split(),
    which has no test-split access mechanism of any kind -- no test loader
    is reachable through this factory.

    This is real, wired production code, not a stub -- but it is not
    invoked against real data in Phase 2B.2 (scripts/run_confirmatory.py's
    train-validation mode still refuses to run in this phase). Tests must
    inject a synthetic factory (see DataLoaderFactory) instead of calling
    this one; no production code path allows a synthetic backend to be
    selected via CLI flag or environment variable.
    """
    verify_official_dataset_artifact(cell.dataset, cell.resolution, root=root)
    train_ds = load_pilot_split(cell.dataset, split="train", size=cell.resolution, root=str(root))
    val_ds = load_pilot_split(cell.dataset, split="val", size=cell.resolution, root=str(root))
    batch_size = FROZEN_TRAINING_SETTINGS.batch_size_28_64px
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


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
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
    protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
) -> CellTrainResult:
    """TRAIN-VALIDATION MODE for one cell. Uses train_loader/val_loader
    ONLY -- never a test loader (no test-loader parameter exists on this
    function at all, structurally impossible to pass one). NOT invoked
    against real data in Phase 2B.2 -- exercised only with synthetic
    tensors and temporary ledgers in tests.

    All training hyperparameters (learning rate, weight decay, epochs,
    early-stopping patience/min-delta) are read explicitly from
    matrix.FROZEN_TRAINING_SETTINGS -- the single frozen-protocol source of
    truth -- rather than relying on train_model()'s own defaults, which
    happen to coincidentally match but must never be the thing actually
    trusted.

    On both success and failure, appends exactly one row to the
    confirmatory ledger (`confirmatory_ledger_path`) tagged
    confirmatory=True: status="completed" with validation_metrics_observed
    =True on success, or status="failed" with `failure_reason` set on any
    exception (including OOM and non-finite loss). test_metrics_observed
    is always False here -- this function never touches the test split.

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
    settings = FROZEN_TRAINING_SETTINGS
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
            max_epochs=settings.max_epochs,
            learning_rate=settings.learning_rate,
            weight_decay=settings.weight_decay,
            early_stopping=EarlyStoppingConfig(
                patience=settings.early_stopping_patience,
                min_delta=settings.early_stopping_min_delta,
            ),
            augmentation_policy=augmentation_policy,
            augmentation_seed=augmentation_seed,
            max_training_seconds=max_training_seconds,
        )

        from when_tta_hurts.artifacts import save_checkpoint

        ckpt_hash = save_checkpoint(result.best_state_dict, attempt_dir / "best_checkpoint.pt")
        finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
        ledger_module.append_confirmatory_entry(
            ledger_path=confirmatory_ledger_path,
            run_id=run_id,
            attempt_id=status.attempt_number,
            block=cell.block,
            config_hash=this_hash,
            protocol_commit=protocol_commit,
            dataset=cell.dataset,
            model=cell.model,
            resolution=cell.resolution,
            normalization=cell.normalization,
            training_policy=cell.training_policy,
            seed=cell.seed,
            split="validation",
            status="completed",
            checkpoint_hash=ckpt_hash,
            started_at=status.started_at,
            ended_at=status.ended_at,
            runtime_seconds=status.ended_at - status.started_at,
            validation_metrics_observed=True,
            test_metrics_observed=False,
        )
        return CellTrainResult(
            status="completed", run_id=run_id, attempt_number=status.attempt_number, checkpoint_hash=ckpt_hash
        )
    except TrainingOOMError as e:
        finish_attempt(attempt_dir, status, RunStatus.FAILED, failure_reason=f"OOM: {e}")
        _append_failed_confirmatory_row(
            cell, status, this_hash, f"OOM: {e}", confirmatory_ledger_path, protocol_commit
        )
        raise
    except Exception as e:
        finish_attempt(attempt_dir, status, RunStatus.FAILED, failure_reason=str(e))
        _append_failed_confirmatory_row(
            cell, status, this_hash, str(e), confirmatory_ledger_path, protocol_commit
        )
        raise


def _append_failed_confirmatory_row(
    cell: MatrixCell,
    status,
    config_hash: str,
    failure_reason: str,
    confirmatory_ledger_path: str | Path,
    protocol_commit: str,
) -> None:
    ledger_module.append_confirmatory_entry(
        ledger_path=confirmatory_ledger_path,
        run_id=cell.run_id(),
        attempt_id=status.attempt_number,
        block=cell.block,
        config_hash=config_hash,
        protocol_commit=protocol_commit,
        dataset=cell.dataset,
        model=cell.model,
        resolution=cell.resolution,
        normalization=cell.normalization,
        training_policy=cell.training_policy,
        seed=cell.seed,
        split="validation",
        status="failed",
        checkpoint_hash="",
        started_at=status.started_at,
        ended_at=status.ended_at,
        runtime_seconds=status.ended_at - status.started_at,
        failure_reason=failure_reason,
        validation_metrics_observed=False,
        test_metrics_observed=False,
    )


def run_canary_cell(
    run_id: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    loader_factory: DataLoaderFactory = default_train_validation_loader_factory,
    device_resolver: Callable[[], torch.device] = lambda: select_device("mps"),
    require_clean_tree: bool = True,
    root: str = "artifacts/confirmatory",
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
) -> CellTrainResult:
    """Single entry point for the Phase 2B.3A canary CLI path (and its
    tests). Enforces, IN ORDER, before any dataset access or artifact
    creation:
    1. run_id resolves to exactly one approved unconditional A/B/C cell
       (rejects pilot/excluded-seed IDs, Block D IDs, and unknown IDs --
       see resolve_canary_run_id()).
    2. Working tree is clean (skippable only for injected tests that don't
       care about repo state, via require_clean_tree=False).
    3. Device resolves to MPS with NO silent CPU fallback (device_resolver
       defaults to select_device('mps'), which raises DeviceUnavailableError
       rather than substituting CPU).

    Only after all three checks pass does it call loader_factory(cell) --
    which performs official-checksum verification before constructing any
    DataLoader -- and then run_train_validation_cell(). Tests inject a
    synthetic loader_factory and a CPU device_resolver; production code
    (scripts/run_confirmatory.py) uses the real defaults.
    """
    cell = resolve_canary_run_id(run_id, matrix_path)
    if require_clean_tree:
        require_clean_working_tree()
    device = device_resolver()
    train_loader, val_loader = loader_factory(cell)
    return run_train_validation_cell(
        cell,
        train_loader,
        val_loader,
        device,
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
    )


def run_final_test(authorization_artifact_path: str = "configs/final_evaluation_authorization.yaml") -> None:
    """FINAL-TEST MODE. Verifies authorization FIRST -- before any dataset
    construction. Since the real authorization artifact does not exist in
    Phase 2B.2, this always raises AuthorizationError. Never invoked in
    this phase.
    """
    verify_authorization(authorization_artifact_path)  # will raise -- artifact does not exist
    raise FinalTestNotYetImplementedError(
        "Final-test evaluation is locked: authorization has been verified, but final-test "
        "evaluation logic is intentionally not yet implemented. It may only be implemented "
        "AFTER Validation-Gated TTA is designed and frozen (see docs/phase2b_protocol.md) -- "
        "reaching this point in Phase 2B.2 is impossible in practice, since verify_authorization() "
        "above always raises first (no real authorization artifact exists in this phase)."
    )
