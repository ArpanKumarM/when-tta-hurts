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

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from when_tta_hurts import ledger as ledger_module
from when_tta_hurts.authorization import verify_authorization
from when_tta_hurts.data import get_dataset_metadata, load_pilot_split
from when_tta_hurts.dataset_verification import DEFAULT_DATA_ROOT, verify_official_dataset_artifact
from when_tta_hurts.devices import select_device
from when_tta_hurts.matrix import FROZEN_TRAINING_SETTINGS, MatrixCell, parse_and_validate_matrix
from when_tta_hurts.models.resnet import build_resnet18_small_input
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything
from when_tta_hurts.result_artifacts import (
    PersistenceVerificationError,
    persist_and_verify_completion,
    verify_artifact_manifest,
)
from when_tta_hurts.run_identity import (
    ConflictingCompletedRunError,
    RunStatus,
    cell_config_hash,
    finish_attempt,
    list_attempts,
    run_directory,
    start_attempt,
)
from when_tta_hurts.training import EarlyStoppingConfig, TrainingOOMError, train_model
from when_tta_hurts.transforms.policies import build_policy

try:
    import importlib.metadata as _importlib_metadata

    _TORCH_VERSION = _importlib_metadata.version("torch")
    _KORNIA_VERSION = _importlib_metadata.version("kornia")
    _MEDMNIST_VERSION = _importlib_metadata.version("medmnist")
except Exception:
    _TORCH_VERSION = _KORNIA_VERSION = _MEDMNIST_VERSION = "unknown"

# The commit at which docs/phase2b_protocol.md and configs/experiment_matrix.yaml
# were frozen (Phase 2B.1) -- recorded on every confirmatory ledger row so a
# row is always traceable to the exact frozen protocol version that governed
# it, independent of whatever commit later touched orchestrator.py itself.
FROZEN_PROTOCOL_COMMIT = "ce4c962"

DataLoaderFactory = Callable[[MatrixCell], "TrainValidationLoaders"]

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


class AmbiguousCanonicalCompletionError(RuntimeError):
    """Raised when more than one completed, canonical-eligible, matching-
    config-hash, artifact-valid attempt exists for the same run ID.
    Selection must NEVER silently pick earliest/latest/best in this case
    -- it is always a hard failure requiring manual/amendment-ledger
    resolution."""


class DirtyWorkingTreeError(RuntimeError):
    """Raised when the working tree contains anything other than a
    verified, append-only extension of an approved research ledger --
    training must never run against an uncommitted/uncertain code,
    config, protocol, or test state, so this is checked before any
    dataset access."""


# Part 2D: an absolutely-clean-tree rule is impossible to satisfy in
# practice, because every successful confirmatory run appends a row to a
# tracked ledger as its own authorized side effect. The policy below
# distinguishes that expected, verified append from any other dirtiness:
# source/scripts/tests/configs/protocol docs/dependency files/unexpected
# paths must be perfectly clean; these four ledgers may be modified ONLY
# via a strict append (existing header + rows byte-identical, new rows
# added at the end, nothing edited/deleted/reordered).
APPROVED_APPEND_ONLY_LEDGER_PATHS = frozenset(
    {
        "artifacts/ledger.csv",
        "artifacts/ledger_incidents.csv",
        "artifacts/ledger_confirmatory.csv",
        "artifacts/ledger_amendments.csv",
    }
)


def _git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], text=True)


def _git_show_head(path: str) -> str:
    try:
        return subprocess.check_output(["git", "show", f"HEAD:{path}"], text=True)
    except subprocess.CalledProcessError:
        return ""


def require_clean_working_tree() -> None:
    """See APPROVED_APPEND_ONLY_LEDGER_PATHS docstring above. Any dirty
    path outside that set -- tracked or untracked, source or otherwise --
    is a hard failure. Any dirty path inside that set must be a strict
    append: the working-tree content must start with the exact HEAD
    content (guaranteeing header/existing rows/ordering are byte-for-byte
    unchanged and only new rows were added at the end); anything else
    (edit, deletion, reordering, malformed append) is a hard failure.
    """
    status_output = _git_status_porcelain()
    for line in status_output.splitlines():
        if not line.strip():
            continue
        code = line[:2]
        path = line[3:].strip()
        if path not in APPROVED_APPEND_ONLY_LEDGER_PATHS:
            raise DirtyWorkingTreeError(
                f"Refusing to execute: unexpected dirty path '{path}' (git status '{code}'). "
                f"Only appends to {sorted(APPROVED_APPEND_ONLY_LEDGER_PATHS)} are permitted; "
                f"everything else (source, scripts, tests, configs, protocol docs, dependency "
                f"files) must be perfectly clean."
            )
        if code.strip() != "M":
            raise DirtyWorkingTreeError(
                f"Refusing to execute: ledger '{path}' has git status '{code}' -- only an "
                f"in-place modification (a verified append) is ever permitted for this path; "
                f"additions/deletions/renames of the ledger file itself are not."
            )
        head_content = _git_show_head(path)
        working_content = Path(path).read_text()
        if not working_content.startswith(head_content):
            raise DirtyWorkingTreeError(
                f"Refusing to execute: '{path}' working-tree content is not a strict "
                f"append-only extension of its committed HEAD content -- an edit, deletion, "
                f"or reordering of existing content was detected."
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
    """num_classes is always derived from the dataset's own registered
    metadata (data.py::get_dataset_metadata(), backed by medmnist.INFO's
    label dict) -- NEVER hardcoded. PathMNIST=9, BloodMNIST=8,
    DermaMNIST=7. get_dataset_metadata() itself hard-fails for any
    unregistered dataset name before this function returns anything."""
    num_classes = get_dataset_metadata(cell.dataset).n_classes
    if cell.model == "small_cnn":
        return build_small_cnn(num_classes=num_classes, normalization=cell.normalization)
    if cell.model == "resnet18":
        return build_resnet18_small_input(num_classes=num_classes)
    raise ValueError(f"Unknown model '{cell.model}'")


@dataclass(frozen=True)
class TrainValidationLoaders:
    """Train-validation DataLoader bundle plus the dataset-provenance facts
    needed for result.json -- structurally two loaders only (no test-loader
    field exists on this type at all, so no test loader is ever reachable
    through it)."""

    train_loader: DataLoader
    val_loader: DataLoader
    dataset_artifact_filename: str
    dataset_expected_checksum_md5: str
    dataset_actual_checksum_md5: str


def default_train_validation_loader_factory(
    cell: MatrixCell, root: str | Path = DEFAULT_DATA_ROOT
) -> TrainValidationLoaders:
    """PRODUCTION data-loading path for a confirmatory matrix cell.

    Verifies the official checksummed NATIVE-resolution artifact BEFORE
    constructing any DataLoader -- fails closed (ArtifactVerificationError)
    on a missing file, a checksum mismatch, or an unsupported dataset/
    resolution, so a resized proxy or corrupt download can never reach a
    training loop. Loads train/val splits ONLY via load_pilot_split(),
    which has no test-split access mechanism of any kind -- no test loader
    is reachable through this factory.

    This is real, wired production code, not a stub. Tests must inject a
    synthetic factory (see DataLoaderFactory) instead of calling this one;
    no production code path allows a synthetic backend to be selected via
    CLI flag or environment variable.
    """
    verification = verify_official_dataset_artifact(cell.dataset, cell.resolution, root=root)
    train_ds = load_pilot_split(cell.dataset, split="train", size=cell.resolution, root=str(root))
    val_ds = load_pilot_split(cell.dataset, split="val", size=cell.resolution, root=str(root))
    batch_size = FROZEN_TRAINING_SETTINGS.batch_size_28_64px
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return TrainValidationLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_artifact_filename=Path(verification.artifact_path).name,
        dataset_expected_checksum_md5=verification.expected_checksum_md5,
        dataset_actual_checksum_md5=verification.actual_checksum_md5,
    )


@dataclass
class CellTrainResult:
    status: str  # "skipped_completed" | "completed" | "failed"
    run_id: str
    attempt_number: int | None
    checkpoint_hash: str | None
    reason: str | None = None
    config_hash: str | None = None
    manifest_verified: bool | None = None


def check_confirmatory_skip(
    cell: MatrixCell,
    root: str | Path = "artifacts/confirmatory",
    amendments_ledger_path: str | Path = ledger_module.AMENDMENTS_LEDGER_PATH,
) -> CellTrainResult | None:
    """Determine whether a canonical-eligible matching completion already
    exists, WITHOUT touching MPS, checksums, datasets, DataLoaders,
    models, or ledger append -- this function only reads run_identity's
    status.json files, artifact_manifest.json files, and the (small,
    local) amendments ledger.

    Considers EVERY attempt for this run ID, in numeric order (not just
    the first) -- see list_attempts(). This matters because an early
    attempt can be completed-but-canonical-ineligible (e.g. attempt_001 of
    A-pathmnist-28px-batchnorm-policy-none-s0) while a LATER attempt is
    the true canonical completion; considering only the first attempt
    would incorrectly conclude "not eligible, proceed to train" forever.

    Algorithm:
    1. Enumerate every attempt, numeric order.
    2. Ignore anything not status=="completed" (failed/aborted/running/
       planned attempts are never skip candidates).
    3. Any completed attempt with a DIFFERENT config_hash than the current
       cell is a hard failure (protocol drift), regardless of eligibility.
    4. Among completed, matching-hash attempts, exclude any marked
       canonical_eligible=false in the amendments ledger.
    5. For each remaining (eligible, matching-hash) candidate, fully
       verify its artifact_manifest.json -- missing or corrupt artifacts
       are a hard failure for that candidate (never silently retrained).
    6. Exactly one such candidate -> return it as the skip target.
       Zero -> return None (no skip, a new attempt is needed).
       More than one -> AmbiguousCanonicalCompletionError (never silently
       chosen by recency/favorability).

    Never uses any metric value (validation accuracy, loss, etc.) in this
    decision.
    """
    run_id = cell.run_id()
    this_hash = cell_config_hash(cell)

    eligible_matching_candidates = []
    for status in list_attempts(cell, root):
        if status.get("status") != RunStatus.COMPLETED.value:
            continue  # failed/aborted/running/planned attempts are never skip candidates
        if status["config_hash"] != this_hash:
            raise ConflictingCompletedRunError(
                f"Run {run_id} has a completed attempt_{status['attempt_number']:03d} with a "
                f"different config hash ({status['config_hash']}) than the current cell "
                f"({this_hash}). This indicates the frozen protocol changed underneath an "
                f"existing result. Hard failure -- will not silently overwrite or ignore."
            )
        if ledger_module.is_canonical_ineligible(run_id, status["attempt_number"], amendments_ledger_path):
            continue  # ineligible: does not block a new attempt, not a skip candidate

        attempt_dir = run_directory(cell, root) / f"attempt_{status['attempt_number']:03d}"
        manifest_path = attempt_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise PersistenceVerificationError(
                f"Eligible candidate attempt_{status['attempt_number']:03d} for run {run_id} is "
                f"missing artifact_manifest.json -- cannot verify it as canonical. Hard failure, "
                f"not an automatic retrain."
            )
        manifest = json.loads(manifest_path.read_text())
        verify_artifact_manifest(attempt_dir, manifest)  # raises PersistenceVerificationError on corruption
        eligible_matching_candidates.append(status)

    if len(eligible_matching_candidates) == 0:
        return None
    if len(eligible_matching_candidates) > 1:
        attempt_numbers = sorted(s["attempt_number"] for s in eligible_matching_candidates)
        raise AmbiguousCanonicalCompletionError(
            f"Multiple eligible, matching-hash, artifact-valid completed attempts exist for "
            f"run {run_id}: attempts {attempt_numbers}. Refusing to silently choose "
            f"earliest/latest/best -- resolve via the amendments ledger before proceeding."
        )

    selected = eligible_matching_candidates[0]
    selected_attempt_dir = run_directory(cell, root) / f"attempt_{selected['attempt_number']:03d}"
    result_data = json.loads((selected_attempt_dir / "result.json").read_text())
    return CellTrainResult(
        status="skipped_completed",
        run_id=run_id,
        attempt_number=selected["attempt_number"],
        checkpoint_hash=result_data["checkpoint_hash"],
        reason="matching completed, canonical-eligible, artifact-verified attempt already exists",
        config_hash=selected["config_hash"],
        manifest_verified=True,
    )


def run_train_validation_cell(
    cell: MatrixCell,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    root: str = "artifacts/confirmatory",
    max_training_seconds: float | None = None,
    ledger_check_test_metrics: Callable[[str], bool] | None = None,
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
    amendments_ledger_path: str | Path = ledger_module.AMENDMENTS_LEDGER_PATH,
    protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
    matrix_path: str = "configs/experiment_matrix.yaml",
    dataset_artifact_filename: str = "unknown",
    dataset_expected_checksum_md5: str = "unknown",
    dataset_actual_checksum_md5: str = "unknown",
) -> CellTrainResult:
    """TRAIN-VALIDATION MODE for one cell. Uses train_loader/val_loader
    ONLY -- never a test loader (no test-loader parameter exists on this
    function at all, structurally impossible to pass one). NOT invoked
    against real data except via the frozen Phase 2B.3A canary process --
    exercised otherwise only with synthetic tensors and temporary ledgers
    in tests.

    All training hyperparameters (learning rate, weight decay, epochs,
    early-stopping patience/min-delta) are read explicitly from
    matrix.FROZEN_TRAINING_SETTINGS -- the single frozen-protocol source of
    truth -- rather than relying on train_model()'s own defaults, which
    happen to coincidentally match but must never be the thing actually
    trusted.

    status="completed" (and therefore canonical-eligible, absent a later
    amendment) is written ONLY after persist_and_verify_completion()
    succeeds -- see result_artifacts.py. ANY persistence/verification
    failure is treated as a training failure: status="failed", an incident
    is recorded, and PersistenceVerificationError propagates.

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

    skip = check_confirmatory_skip(cell, root, amendments_ledger_path)
    if skip is not None:
        return skip
    this_hash = cell_config_hash(cell)

    # check_confirmatory_skip already ruled out a hash mismatch (it would
    # have raised ConflictingCompletedRunError above); any completed
    # attempt still on disk at this point is necessarily eligibility-
    # excluded (amendments-ledger ineligible), so a new attempt is
    # legitimately needed -- see run_identity.start_attempt() docstring.
    attempt_dir, status = start_attempt(cell, root, allow_new_attempt_despite_matching_hash=True)
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

        matrix_hash = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True).source_config_hash
        result_fields = {
            "run_id": run_id,
            "attempt_id": status.attempt_number,
            "best_epoch": result.best_epoch,
            "best_val_accuracy": result.best_val_accuracy,
            "best_val_loss": result.best_val_loss,
            "epochs_completed": result.epochs_completed,
            "early_stopped": result.early_stopped,
            "early_stopping_reason": result.early_stopping_reason,
            "total_runtime_seconds": result.training_time_seconds,
            "peak_mps_memory": result.peak_mps_memory,
            "checkpoint_hash": ckpt_hash,
            "config_hash": this_hash,
            "matrix_hash": matrix_hash,
            "protocol_commit": protocol_commit,
            "source_commit": _git_commit_hash(),
            "dataset_artifact_filename": dataset_artifact_filename,
            "dataset_expected_checksum_md5": dataset_expected_checksum_md5,
            "dataset_actual_checksum_md5": dataset_actual_checksum_md5,
            "device": str(device),
            "dependency_versions": {
                "torch": _TORCH_VERSION,
                "kornia": _KORNIA_VERSION,
                "medmnist": _MEDMNIST_VERSION,
            },
        }
        metadata_fields = {
            "run_id": run_id,
            "attempt_id": status.attempt_number,
            "block": cell.block,
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "model": cell.model,
            "normalization": cell.normalization,
            "training_policy": cell.training_policy,
            "seed": cell.seed,
            "frozen_training_settings": {
                "optimizer": settings.optimizer,
                "learning_rate": settings.learning_rate,
                "weight_decay": settings.weight_decay,
                "loss": settings.loss,
                "max_epochs": settings.max_epochs,
                "lr_schedule": settings.lr_schedule,
                "early_stopping_patience": settings.early_stopping_patience,
                "early_stopping_min_delta": settings.early_stopping_min_delta,
                "restore_best": settings.restore_best,
                "batch_size_28_64px": settings.batch_size_28_64px,
                "precision": settings.precision,
                "mixed_precision": settings.mixed_precision,
                "device": settings.device,
                "label_smoothing": settings.label_smoothing,
                "class_weighting": settings.class_weighting,
                "channel_standardization": settings.channel_standardization,
            },
            "matrix_hash": matrix_hash,
            "protocol_commit": protocol_commit,
            "source_commit": _git_commit_hash(),
        }

        persist_and_verify_completion(
            attempt_dir,
            history=result.history,
            result_fields=result_fields,
            metadata_fields=metadata_fields,
            best_state_dict=result.best_state_dict,
            model_factory=lambda: _build_model(cell),
        )

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
    except PersistenceVerificationError as e:
        finish_attempt(
            attempt_dir, status, RunStatus.FAILED, failure_reason=f"persistence verification failed: {e}"
        )
        _append_failed_confirmatory_row(
            cell,
            status,
            this_hash,
            f"persistence verification failed: {e}",
            confirmatory_ledger_path,
            protocol_commit,
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
    amendments_ledger_path: str | Path = ledger_module.AMENDMENTS_LEDGER_PATH,
) -> CellTrainResult:
    """Single entry point for the Phase 2B.3A canary CLI path (and its
    tests). Enforces, IN ORDER:
    1. run_id resolves to exactly one approved unconditional A/B/C cell
       (rejects pilot/excluded-seed IDs, Block D IDs, and unknown IDs --
       see resolve_canary_run_id()).
    2. A canonical-eligible matching completion is checked and, if found,
       returned as an IMMEDIATE skip -- before the working-tree check, MPS
       initialization, checksum verification, dataset loading, DataLoader
       construction, model construction, artifact-attempt creation, or any
       ledger append (see check_confirmatory_skip()). A completed-but-
       ineligible attempt (e.g. one recorded in the amendments ledger) does
       NOT trigger this skip.
    3. Only if execution is actually needed: working tree is clean
       (skippable only for injected tests that don't care about repo
       state, via require_clean_tree=False), then device resolves to MPS
       with NO silent CPU fallback (device_resolver defaults to
       select_device('mps'), which raises DeviceUnavailableError rather
       than substituting CPU), then loader_factory(cell) (real checksum
       verification before any DataLoader), then run_train_validation_cell().

    Tests inject a synthetic loader_factory and a CPU device_resolver;
    production code (scripts/run_confirmatory.py) uses the real defaults.
    """
    cell = resolve_canary_run_id(run_id, matrix_path)

    skip = check_confirmatory_skip(cell, root, amendments_ledger_path)
    if skip is not None:
        return skip

    if require_clean_tree:
        require_clean_working_tree()
    device = device_resolver()
    bundle = loader_factory(cell)
    return run_train_validation_cell(
        cell,
        bundle.train_loader,
        bundle.val_loader,
        device,
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
        amendments_ledger_path=amendments_ledger_path,
        matrix_path=matrix_path,
        dataset_artifact_filename=bundle.dataset_artifact_filename,
        dataset_expected_checksum_md5=bundle.dataset_expected_checksum_md5,
        dataset_actual_checksum_md5=bundle.dataset_actual_checksum_md5,
    )


BLOCK_SHORT_TO_FULL = {
    "A": "A_core_normalization_resolution",
    "B": "B_policy_matching",
    "C": "C_positive_control_reproduction",
}
# Block D is deliberately absent from this mapping -- resolve_block_full_name()
# rejects it explicitly (Phase 2B.3B execution is A/B/C only; Block D
# requires its own gate evaluation and is out of scope for this path).


class UnsupportedBlockError(RuntimeError):
    """Raised for --block D or any unrecognized block letter passed to the
    sequential block-execution or verify-completions paths."""


def resolve_block_full_name(block: str) -> str:
    if block == "D":
        raise UnsupportedBlockError(
            "Block D is not authorized for sequential block execution via this path -- "
            "it requires its own gate evaluation (block_d_gate.py) and is out of scope."
        )
    if block not in BLOCK_SHORT_TO_FULL:
        raise UnsupportedBlockError(
            f"Unrecognized block '{block}' -- expected one of "
            f"{sorted(BLOCK_SHORT_TO_FULL)} (or 'D', rejected)."
        )
    return BLOCK_SHORT_TO_FULL[block]


def run_block_cells(
    block: str,
    expected_total: int,
    expected_pending: int,
    matrix_path: str = "configs/experiment_matrix.yaml",
    loader_factory: DataLoaderFactory = default_train_validation_loader_factory,
    device_resolver: Callable[[], torch.device] = lambda: select_device("mps"),
    require_clean_tree: bool = True,
    root: str = "artifacts/confirmatory",
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
    amendments_ledger_path: str | Path = ledger_module.AMENDMENTS_LEDGER_PATH,
) -> list[CellTrainResult]:
    """Execute every cell of `block` (A/B/C only) sequentially, in the
    exact committed matrix order. NEVER parallel.

    Guard order (all metadata-only, before any MPS/data/artifact touch
    for ANY cell in the block):
    1. Resolve the block letter (rejects D and unknown letters).
    2. Parse the matrix; the block's actual cell count must equal
       `expected_total`, or this is a hard failure before anything else.
    3. Compute the skip/pending decision for EVERY cell up front (pure
       metadata reads via check_confirmatory_skip -- no MPS, no
       checksums, no datasets, no models). The count of cells requiring
       execution must equal `expected_pending`, or this is a hard
       failure before any cell is touched.

    Only after both counts are confirmed does real execution begin, cell
    by cell, in order:
    - A cell with an existing eligible canonical completion is skipped
      immediately (no MPS/data/model for that cell).
    - Otherwise: working-tree check, fresh device resolution, fresh
      loader_factory() call, fresh model/optimizer/scheduler construction
      (all internal to run_train_validation_cell -- every cell starts
      from a completely fresh state, reseeded from ITS OWN registered
      seed via seed_everything(cell.seed) before model construction and
      before the training loop's first DataLoader iteration).
    - After each cell that actually trained, cell-local references are
      released and the MPS cache is cleared (torch.mps.empty_cache())
      before moving to the next cell -- this is memory hygiene only and
      never alters any scientific computation (confirmed across attempts
      1-4 of the canary cell, which reproduced bit-identically despite
      this same cache-clearing behavior already occurring within
      benchmark/training code paths elsewhere in this codebase).
    - The FIRST cell that fails/aborts stops the entire block immediately
      -- no further cells are attempted, no retry, and the exception
      propagates to the caller. All previously completed cells in this
      block invocation, and all pre-existing attempts, remain untouched.
    """
    full_block_name = resolve_block_full_name(block)
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cells = expanded.cells_by_block[full_block_name]

    if len(cells) != expected_total:
        raise ValueError(
            f"Block {block} expected-total mismatch: matrix has {len(cells)} cells, "
            f"--expected-total={expected_total}. Refusing to proceed before touching any data."
        )

    # Pure metadata pass: decide skip vs. pending for every cell BEFORE any
    # MPS/checksum/dataset/model/DataLoader activity for any cell.
    precomputed_skips: dict[str, CellTrainResult | None] = {}
    for cell in cells:
        precomputed_skips[cell.run_id()] = check_confirmatory_skip(cell, root, amendments_ledger_path)

    actual_pending = sum(1 for skip in precomputed_skips.values() if skip is None)
    if actual_pending != expected_pending:
        raise ValueError(
            f"Block {block} expected-pending mismatch: {actual_pending} cells actually "
            f"pending, --expected-pending={expected_pending}. Refusing to proceed before "
            f"touching any data."
        )

    results: list[CellTrainResult] = []
    for cell in cells:
        run_id = cell.run_id()
        skip = precomputed_skips[run_id]
        if skip is not None:
            results.append(skip)
            continue

        if require_clean_tree:
            require_clean_working_tree()
        device = device_resolver()
        bundle = loader_factory(cell)
        result = run_train_validation_cell(
            cell,
            bundle.train_loader,
            bundle.val_loader,
            device,
            root=root,
            confirmatory_ledger_path=confirmatory_ledger_path,
            amendments_ledger_path=amendments_ledger_path,
            matrix_path=matrix_path,
            dataset_artifact_filename=bundle.dataset_artifact_filename,
            dataset_expected_checksum_md5=bundle.dataset_expected_checksum_md5,
            dataset_actual_checksum_md5=bundle.dataset_actual_checksum_md5,
        )
        results.append(result)

        # Release cell-local references and MPS cache -- memory hygiene
        # only, never affects scientific computation.
        del bundle, device
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        if result.status == "failed":
            break  # stop the entire block immediately; never retry, never continue

    return results


def verify_block_completions(
    block: str,
    expected_total: int,
    matrix_path: str = "configs/experiment_matrix.yaml",
    root: str = "artifacts/confirmatory",
    amendments_ledger_path: str | Path = ledger_module.AMENDMENTS_LEDGER_PATH,
) -> dict:
    """METADATA-ONLY block completion report. Never initializes MPS, never
    verifies/downloads/loads any dataset, never constructs a model or
    DataLoader, and makes NO filesystem or ledger changes -- it only calls
    check_confirmatory_skip() (itself metadata-only) once per cell and
    classifies the outcome.
    """
    full_block_name = resolve_block_full_name(block)
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cells = expanded.cells_by_block[full_block_name]
    if len(cells) != expected_total:
        raise ValueError(
            f"Block {block} expected-total mismatch: matrix has {len(cells)} cells, "
            f"--expected-total={expected_total}."
        )

    cell_reports = []
    missing, ambiguous, corrupt = [], [], []
    canonical_count = 0
    for cell in cells:
        run_id = cell.run_id()
        try:
            skip = check_confirmatory_skip(cell, root, amendments_ledger_path)
        except AmbiguousCanonicalCompletionError as e:
            ambiguous.append(run_id)
            cell_reports.append({"run_id": run_id, "status": "ambiguous", "error": str(e)})
            continue
        except (ConflictingCompletedRunError, PersistenceVerificationError) as e:
            corrupt.append(run_id)
            cell_reports.append({"run_id": run_id, "status": "corrupt", "error": str(e)})
            continue

        if skip is None:
            missing.append(run_id)
            cell_reports.append({"run_id": run_id, "status": "missing"})
        else:
            canonical_count += 1
            cell_reports.append(
                {
                    "run_id": run_id,
                    "status": "canonical",
                    "attempt_number": skip.attempt_number,
                    "checkpoint_hash": skip.checkpoint_hash,
                    "config_hash": skip.config_hash,
                }
            )

    return {
        "block": block,
        "expected_total": expected_total,
        "actual_total": len(cells),
        "canonical_count": canonical_count,
        "missing": missing,
        "ambiguous": ambiguous,
        "corrupt": corrupt,
        "failed_canonical": [],  # reserved; corrupt/ambiguous cover all current failure modes
        "cells": cell_reports,
    }


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
