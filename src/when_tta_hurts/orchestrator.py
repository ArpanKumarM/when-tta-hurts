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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from when_tta_hurts import ledger as ledger_module
from when_tta_hurts.authorization import verify_authorization
from when_tta_hurts.block_d_benchmark import (
    DEFAULT_DECISION_PATH as _BLOCK_D_DEFAULT_DECISION_PATH,
)
from when_tta_hurts.block_d_benchmark import (
    FROZEN_PROTOCOL_COMMIT as _BLOCK_D_FROZEN_PROTOCOL_COMMIT,
)
from when_tta_hurts.block_d_benchmark import (
    SPEC_COMMIT as _BLOCK_D_SPEC_COMMIT,
)
from when_tta_hurts.block_d_benchmark import (
    BlockDDecisionError,
    _default_git_tracked_and_clean,
    _sha256_file,
    load_and_verify_block_d_decision,
)
from when_tta_hurts.block_d_gate import (
    MAX_TRAINING_MINUTES_PER_RUN as _BLOCK_D_MAX_TRAINING_MINUTES_PER_RUN,
)
from when_tta_hurts.data import get_dataset_metadata, load_pilot_split
from when_tta_hurts.dataset_verification import (
    DEFAULT_DATA_ROOT,
    ArtifactVerification,
    verify_official_dataset_artifact,
)
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


class StaleAttemptError(RuntimeError):
    """Raised when an earlier attempt for this run ID is nonterminal
    (status 'running' or 'planned' -- i.e. never reached completed/failed/
    aborted) AND has no confirmatory-ledger row of any kind. The runner
    must NEVER silently start a new attempt in this situation -- doing so
    is exactly what caused an unintended duplicate real-MPS training run
    during Phase 2B.3B. Call reconcile_stale_attempt() first, with
    independent evidence that the stale attempt has genuinely stopped."""


# Terminal confirmatory-attempt statuses -- anything else (running,
# planned) is nonterminal and, if unledgered, blocks further execution
# for that run ID until explicitly reconciled.
_TERMINAL_ATTEMPT_STATUSES = frozenset({"completed", "failed", "aborted"})


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
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
    effective_config_hash: str | None = None,
) -> CellTrainResult | None:
    """Determine whether a canonical-eligible matching completion already
    exists, WITHOUT touching MPS, checksums, datasets, DataLoaders,
    models, or ledger append -- this function only reads run_identity's
    status.json files, artifact_manifest.json files, and the (small,
    local) amendments/confirmatory ledgers.

    `effective_config_hash`: if given, used INSTEAD of cell_config_hash(cell)
    as the hash a completed attempt must match to be skip-eligible -- for
    Block D cells (see orchestrator.py::compute_block_d_effective_config_hash()).
    A/B/C call sites leave this None, so their behavior is unchanged. This
    is also how an idempotent Block D skip stays gate-decision-aware: if
    the recorded gate decision has since changed (different selected
    batch, different decision commit, etc.), the effective hash changes
    too, so a stale completed attempt is no longer treated as a silent
    match -- it instead hits the existing hash-mismatch hard failure below
    (ConflictingCompletedRunError), exactly like any other protocol-drift
    case, rather than a new attempt silently reusing an outdated result.

    Considers EVERY attempt for this run ID, in numeric order (not just
    the first) -- see list_attempts(). This matters because an early
    attempt can be completed-but-canonical-ineligible (e.g. attempt_001 of
    A-pathmnist-28px-batchnorm-policy-none-s0) while a LATER attempt is
    the true canonical completion; considering only the first attempt
    would incorrectly conclude "not eligible, proceed to train" forever.

    Algorithm:
    0. FIRST, for every attempt (regardless of status): if it is
       nonterminal (status "running" or "planned") AND has no
       confirmatory-ledger row of any kind, raise StaleAttemptError. Since
       this function always runs before start_attempt() for the CURRENT
       invocation, any nonterminal attempt found here can only be a
       leftover from a PRIOR invocation -- reconcile_stale_attempt() must
       be called first. This is what prevents the runner from silently
       starting attempt_N+1 while attempt_N sits abandoned mid-training.
    1. Enumerate every attempt, numeric order.
    2. Ignore anything not status=="completed" (failed/aborted attempts
       are never skip candidates).
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
    this_hash = effective_config_hash if effective_config_hash is not None else cell_config_hash(cell)

    all_attempts = list_attempts(cell, root)
    for status in all_attempts:
        if status.get("status") not in _TERMINAL_ATTEMPT_STATUSES:
            if not ledger_module.has_confirmatory_row(
                run_id, status["attempt_number"], confirmatory_ledger_path
            ):
                raise StaleAttemptError(
                    f"Run {run_id} attempt_{status['attempt_number']:03d} is nonterminal "
                    f"(status='{status.get('status')}') and has no confirmatory-ledger row -- "
                    f"refusing to start a new attempt. Call reconcile_stale_attempt() first, "
                    f"with independent evidence that this attempt has genuinely stopped."
                )

    eligible_matching_candidates = []
    for status in all_attempts:
        if status.get("status") != RunStatus.COMPLETED.value:
            continue  # failed/aborted attempts are never skip candidates
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


def reconcile_stale_attempt(
    cell: MatrixCell,
    attempt_number: int,
    reason: str,
    confirmed_not_running: bool,
    recorded_at: str,
    root: str | Path = "artifacts/confirmatory",
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
    protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
) -> str:
    """Explicit reconciliation for an attempt that check_confirmatory_skip
    has identified (via StaleAttemptError) as nonterminal and unledgered.

    Appends exactly ONE terminal status="aborted" confirmatory-ledger row
    for (cell.run_id(), attempt_number), via ledger.append_confirmatory_entry
    -- which is itself idempotent: an identical repeat call (same
    reason/recorded_at) returns "duplicate_ignored"; a call with DIFFERENT
    content for the same (run_id, attempt_number) raises LedgerConflictError
    (never silently overwritten). This function never edits, deletes, or
    creates anything in the attempt directory -- it only reads status.json
    and appends to the ledger. status="aborted" structurally can never be
    selected as canonical (check_confirmatory_skip only considers
    status=="completed" as a candidate).

    Fails closed:
    - `confirmed_not_running` must be explicitly True. Reconciliation
      always refuses by default -- the caller must have independent
      evidence (outside this codebase, e.g. confirming no process is
      executing the attempt) that it has genuinely stopped, not merely
      assume so because it looks old.
    - `reason` must be non-empty (an evidence-based explanation).
    - The targeted attempt must exist and currently be nonterminal
      (status "running" or "planned"); an already-terminal attempt must
      go through its normal path, not reconciliation.
    """
    if not confirmed_not_running:
        raise StaleAttemptError(
            f"Refusing to reconcile {cell.run_id()} attempt_{attempt_number:03d}: the "
            f"attempt may still be running. Pass confirmed_not_running=True only after "
            f"obtaining independent evidence that it has genuinely stopped."
        )
    if not reason or not reason.strip():
        raise ValueError("reconcile_stale_attempt requires a non-empty, evidence-based reason.")

    run_id = cell.run_id()
    attempt_dir = run_directory(cell, root) / f"attempt_{attempt_number:03d}"
    status_path = attempt_dir / "status.json"
    if not status_path.exists():
        raise StaleAttemptError(
            f"No such attempt: {run_id} attempt_{attempt_number:03d} (status.json not found)."
        )
    status = json.loads(status_path.read_text())
    if status.get("status") in _TERMINAL_ATTEMPT_STATUSES:
        raise StaleAttemptError(
            f"{run_id} attempt_{attempt_number:03d} already has terminal status "
            f"'{status.get('status')}' -- reconciliation is only for nonterminal "
            f"(running/planned) attempts."
        )

    return ledger_module.append_confirmatory_entry(
        ledger_path=confirmatory_ledger_path,
        run_id=run_id,
        attempt_id=attempt_number,
        block=cell.block,
        config_hash=status.get("config_hash", ""),
        protocol_commit=protocol_commit,
        dataset=cell.dataset,
        model=cell.model,
        resolution=cell.resolution,
        normalization=cell.normalization,
        training_policy=cell.training_policy,
        seed=cell.seed,
        split="validation",
        status="aborted",
        checkpoint_hash="",
        started_at=status.get("started_at", ""),
        ended_at="",
        runtime_seconds="",
        failure_reason=reason,
        validation_metrics_observed=False,
        test_metrics_observed=False,
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
    effective_config_hash: str | None = None,
    extra_result_fields: dict[str, Any] | None = None,
    extra_metadata_fields: dict[str, Any] | None = None,
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

    `effective_config_hash`: if given, used as this attempt's config hash
    instead of cell_config_hash(cell) -- for Block D cells, whose identity
    must also cover gate-decision provenance. A/B/C call sites leave this
    None, so their hashing and behavior are completely unchanged.

    `extra_result_fields`/`extra_metadata_fields`: optional dicts merged
    into result.json/metadata.json ON TOP OF the required fields below
    (never removing or overriding a required key) -- used by Block D to
    persist gate provenance (selected batch, decision/benchmark/spec
    commits, decision-artifact SHA-256) alongside the normal record. A/B/C
    call sites leave these None, so their persisted schema is unchanged.
    """
    if cell.seed == 314159:
        raise ValueError("Refusing to train: seed 314159 is permanently excluded from confirmatory runs.")

    run_id = cell.run_id()
    if ledger_check_test_metrics is not None and ledger_check_test_metrics(run_id):
        raise UnfavorableRerunRefusedError(
            f"Run {run_id} already has test metrics observed in the confirmatory ledger -- "
            f"reruns after test metrics exist are never permitted."
        )

    skip = check_confirmatory_skip(
        cell,
        root,
        amendments_ledger_path,
        confirmatory_ledger_path,
        effective_config_hash=effective_config_hash,
    )
    if skip is not None:
        return skip
    this_hash = effective_config_hash if effective_config_hash is not None else cell_config_hash(cell)

    # check_confirmatory_skip already ruled out a hash mismatch (it would
    # have raised ConflictingCompletedRunError above); any completed
    # attempt still on disk at this point is necessarily eligibility-
    # excluded (amendments-ledger ineligible), so a new attempt is
    # legitimately needed -- see run_identity.start_attempt() docstring.
    attempt_dir, status = start_attempt(
        cell, root, allow_new_attempt_despite_matching_hash=True, effective_config_hash=effective_config_hash
    )
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
        if extra_result_fields:
            result_fields = {**result_fields, **extra_result_fields}
        if extra_metadata_fields:
            metadata_fields = {**metadata_fields, **extra_metadata_fields}

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

    skip = check_confirmatory_skip(cell, root, amendments_ledger_path, confirmatory_ledger_path)
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
        precomputed_skips[cell.run_id()] = check_confirmatory_skip(
            cell, root, amendments_ledger_path, confirmatory_ledger_path
        )

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
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
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
    missing, ambiguous, corrupt, stale = [], [], [], []
    canonical_count = 0
    for cell in cells:
        run_id = cell.run_id()
        try:
            skip = check_confirmatory_skip(cell, root, amendments_ledger_path, confirmatory_ledger_path)
        except AmbiguousCanonicalCompletionError as e:
            ambiguous.append(run_id)
            cell_reports.append({"run_id": run_id, "status": "ambiguous", "error": str(e)})
            continue
        except StaleAttemptError as e:
            stale.append(run_id)
            cell_reports.append({"run_id": run_id, "status": "stale", "error": str(e)})
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
        "stale": stale,
        "failed_canonical": [],  # reserved; corrupt/ambiguous cover all current failure modes
        "cells": cell_reports,
    }


### --- Phase 2B.3F: Block D gate-authorized training -------------------- ###
#
# Block D (native-128px PathMNIST/BloodMNIST) is never reachable through
# resolve_canary_run_id()/run_canary_cell()/run_block_cells() -- those
# structurally reject it (BlockDRunRejectedError / UnsupportedBlockError),
# and NOTHING below changes that. This section adds a SEPARATE, PARALLEL
# entry point that is the only way to train a Block D cell, and it can
# never proceed without first verifying the committed, tracked
# artifacts/block_d_gate_decision.json says INCLUDED for that dataset.
#
# Authorization happens BEFORE: attempt allocation, attempt-directory
# creation, ledger writing, dirty-tree acceptance, MPS initialization,
# dataset loading, model construction, and training -- see
# run_block_d_train_validation_cell()'s docstring for the exact order.
#
# No CLI flag, environment variable, or dependency-injection parameter
# bypasses this -- every injectable parameter below (loader_factory,
# device_resolver, dataset_verifier, git_tracked_and_clean,
# last_commit_for_path, commit_is_ancestor) exists ONLY so tests can
# supply synthetic/fake implementations; none of them can skip the
# authorization CALLS themselves, which are unconditional. There is no
# force/override flag anywhere in this section.


class NotBlockDRunIdError(RuntimeError):
    """Raised when a run_id resolves to an Block A/B/C cell (or does not
    resolve at all) via resolve_block_d_run_id() -- this entry point trains
    ONLY Block D cells."""


class BlockDAuthorizationError(RuntimeError):
    """Raised whenever Block D gate authorization fails, for ANY reason --
    missing/untracked/dirty/malformed/OMITTED decision, hash/checksum
    mismatch, non-ancestor provenance commit, or a dataset with no
    recorded selected batch. Always raised BEFORE attempt allocation,
    ledger writing, dirty-tree acceptance, MPS initialization, dataset
    loading, or model construction -- see run_block_d_train_validation_cell().
    """


def resolve_block_d_run_id(run_id: str, matrix_path: str = "configs/experiment_matrix.yaml") -> MatrixCell:
    """Resolve run_id to EXACTLY one Block D matrix cell. Rejects pilot/
    excluded-seed IDs (before any matrix parsing, mirroring
    resolve_canary_run_id()), any A/B/C run_id (NotBlockDRunIdError,
    distinct from "unknown"), and anything not present in the matrix."""
    if "-s314159" in run_id or run_id.startswith("pilot"):
        raise PilotOrExcludedSeedRunIdError(
            f"Refusing run_id '{run_id}': pilot/permanently-excluded-seed identifiers "
            f"are never valid confirmatory targets."
        )
    full = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    for cell in full.cells:
        if cell.run_id() == run_id:
            if cell.block != "D_conditional_128px":
                raise NotBlockDRunIdError(
                    f"Run '{run_id}' belongs to block '{cell.block}', not Block D -- "
                    f"use resolve_canary_run_id()/run_canary_cell() for Block A/B/C instead."
                )
            return cell
    raise UnknownRunIdError(f"'{run_id}' does not match any Block D matrix cell.")


@dataclass(frozen=True)
class BlockDEffectiveConfig:
    """Everything that determines a Block D attempt's true identity: the
    frozen matrix cell PLUS the gate-decision provenance that authorized
    training it. Two attempts of the "same" cell trained under a
    DIFFERENT gate decision (different selected batch, different decision
    commit, etc.) are NOT the same attempt as far as config hashing is
    concerned -- see compute_block_d_effective_config_hash()."""

    cell: MatrixCell
    selected_batch_size: int
    decision_artifact_sha256: str
    gate_decision_commit: str
    benchmark_source_commit: str
    benchmark_spec_commit: str
    protocol_commit: str
    matrix_hash: str
    dataset_checksum_md5: str
    native_resolution: int = 128
    resized: bool = False


def compute_block_d_effective_config_hash(effective_config: BlockDEffectiveConfig) -> str:
    """Config hash for a Block D cell, covering the same (cell +
    FROZEN_TRAINING_SETTINGS) payload cell_config_hash() uses for A/B/C
    PLUS full gate-decision provenance. Does NOT alter cell_config_hash()
    itself or any A/B/C call site."""
    payload = {
        "cell": asdict(effective_config.cell),
        "training_settings": asdict(FROZEN_TRAINING_SETTINGS),
        "block_d_gate": {
            "selected_batch_size": effective_config.selected_batch_size,
            "decision_artifact_sha256": effective_config.decision_artifact_sha256,
            "gate_decision_commit": effective_config.gate_decision_commit,
            "benchmark_source_commit": effective_config.benchmark_source_commit,
            "benchmark_spec_commit": effective_config.benchmark_spec_commit,
            "protocol_commit": effective_config.protocol_commit,
            "matrix_hash": effective_config.matrix_hash,
            "dataset_checksum_md5": effective_config.dataset_checksum_md5,
            "native_resolution": effective_config.native_resolution,
            "resized": effective_config.resized,
        },
    }
    from when_tta_hurts.config import config_hash as _config_hash

    return _config_hash(payload)


def default_block_d_train_validation_loader_factory(
    cell: MatrixCell, batch_size: int, root: str | Path = DEFAULT_DATA_ROOT
) -> TrainValidationLoaders:
    """PRODUCTION Block D data-loading path -- structurally identical to
    default_train_validation_loader_factory() except the batch size is the
    gate-authorized SELECTED batch (never FROZEN_TRAINING_SETTINGS'
    28/64px default), and it is only ever called AFTER authorization has
    already verified the official checksum matches the gate decision's
    recorded checksum. Still independently re-verifies the checksum itself
    (fails closed on a missing/corrupt/resized artifact) before
    constructing any DataLoader -- never accepts a resized proxy, and
    load_pilot_split() has no test-split access mechanism of any kind."""
    verification = verify_official_dataset_artifact(cell.dataset, cell.resolution, root=root)
    train_ds = load_pilot_split(cell.dataset, split="train", size=cell.resolution, root=str(root))
    val_ds = load_pilot_split(cell.dataset, split="val", size=cell.resolution, root=str(root))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return TrainValidationLoaders(
        train_loader=train_loader,
        val_loader=val_loader,
        dataset_artifact_filename=Path(verification.artifact_path).name,
        dataset_expected_checksum_md5=verification.expected_checksum_md5,
        dataset_actual_checksum_md5=verification.actual_checksum_md5,
    )


def _default_last_commit_for_path(path: Path) -> str | None:
    """The most recent commit that touched `path` in the current repository
    history, or None if `path` has never been committed."""
    try:
        out = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path)], text=True).strip()
    except subprocess.CalledProcessError:
        return None
    return out or None


def _default_commit_is_ancestor(commit: str, head: str) -> bool:
    """True iff `commit` is an ancestor of (or equal to) `head` in the
    current repository history."""
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def authorize_block_d_cell(
    cell: MatrixCell,
    decision_path: str | Path = _BLOCK_D_DEFAULT_DECISION_PATH,
    expected_matrix_hash: str | None = None,
    expected_protocol_commit: str = _BLOCK_D_FROZEN_PROTOCOL_COMMIT,
    expected_spec_commit: str = _BLOCK_D_SPEC_COMMIT,
    expected_batch_size: int | None = None,
    matrix_path: str = "configs/experiment_matrix.yaml",
    git_tracked_and_clean=_default_git_tracked_and_clean,
    last_commit_for_path=_default_last_commit_for_path,
    commit_is_ancestor=_default_commit_is_ancestor,
    head_commit: str | None = None,
) -> tuple[dict[str, Any], BlockDEffectiveConfig]:
    """Verify Block D training is authorized for `cell` and derive its
    effective configuration. Raises BlockDAuthorizationError (wrapping the
    underlying cause) if ANY of the following hold:

    - the decision is missing, untracked, dirty, or malformed (delegated
      to load_and_verify_block_d_decision(), require_git_tracked=True,
      raw_output_path=None -- the raw benchmark file need not be present)
    - the decision is not INCLUDED
    - the decision's matrix/protocol/specification hashes don't match the
      currently-frozen matrix/protocol/spec
    - the decision has never been committed (no commit in history touches
      decision_path)
    - the gate-decision commit, benchmark source commit, or benchmark
      spec commit is not an ancestor of the current HEAD (current HEAD is
      NOT required to equal any of them -- later audited engineering
      commits are fine)
    - `cell.dataset` has no valid (positive int) selected_batch_size
      recorded in the decision
    - `expected_batch_size` is given and differs from the recorded
      selected batch (defense-in-depth self-consistency check; the
      decision's own recorded batch is always what training actually
      uses, regardless of whether this optional check is exercised)

    Does NOT touch MPS, the real dataset artifact, or attempt/ledger state
    -- purely decision-file + git-history verification.
    """
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    matrix_hash = expected_matrix_hash if expected_matrix_hash is not None else expanded.source_config_hash

    try:
        decision = load_and_verify_block_d_decision(
            decision_path,
            expected_matrix_hash=matrix_hash,
            expected_protocol_commit=expected_protocol_commit,
            expected_spec_commit=expected_spec_commit,
            dataset=cell.dataset,
            requested_batch_size=expected_batch_size,
            require_git_tracked=True,
            git_tracked_and_clean=git_tracked_and_clean,
            raw_output_path=None,
        )
    except BlockDDecisionError as e:
        raise BlockDAuthorizationError(f"Block D authorization failed for {cell.run_id()}: {e}") from e

    decision_path = Path(decision_path)
    head = head_commit if head_commit is not None else _git_commit_hash()

    gate_decision_commit = last_commit_for_path(decision_path)
    if gate_decision_commit is None:
        raise BlockDAuthorizationError(
            f"Block D authorization failed for {cell.run_id()}: {decision_path} has no commit in "
            f"repository history -- refusing to trust a decision that was never committed."
        )

    for label, commit in (
        ("gate-decision", gate_decision_commit),
        ("benchmark source", decision.get("source_commit")),
        ("benchmark spec", decision.get("spec_commit")),
    ):
        if not commit or not commit_is_ancestor(commit, head):
            raise BlockDAuthorizationError(
                f"Block D authorization failed for {cell.run_id()}: {label} commit "
                f"'{commit}' is not an ancestor of current HEAD '{head}'."
            )

    dataset_provenance = decision.get("per_dataset", {}).get(cell.dataset)
    selected_batch = dataset_provenance.get("selected_batch_size") if dataset_provenance else None
    if not isinstance(selected_batch, int) or selected_batch <= 0:
        raise BlockDAuthorizationError(
            f"Block D authorization failed for {cell.run_id()}: no valid selected_batch_size "
            f"recorded for dataset '{cell.dataset}' in the gate decision."
        )

    decision_sha256 = _sha256_file(decision_path)

    effective_config = BlockDEffectiveConfig(
        cell=cell,
        selected_batch_size=selected_batch,
        decision_artifact_sha256=decision_sha256,
        gate_decision_commit=gate_decision_commit,
        benchmark_source_commit=decision["source_commit"],
        benchmark_spec_commit=decision["spec_commit"],
        protocol_commit=decision["protocol_commit"],
        matrix_hash=decision["matrix_hash"],
        dataset_checksum_md5=dataset_provenance["checksum_actual"],
        native_resolution=128,
        resized=False,
    )
    return decision, effective_config


def run_block_d_train_validation_cell(
    run_id: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    decision_path: str | Path = _BLOCK_D_DEFAULT_DECISION_PATH,
    loader_factory: Callable[[MatrixCell, int, str | Path], TrainValidationLoaders] = (
        default_block_d_train_validation_loader_factory
    ),
    dataset_verifier: Callable[
        [str, int, str | Path], ArtifactVerification
    ] = verify_official_dataset_artifact,
    device_resolver: Callable[[], torch.device] = lambda: select_device("mps"),
    require_clean_tree: bool = True,
    root: str = "artifacts/confirmatory",
    data_root: str | Path = DEFAULT_DATA_ROOT,
    confirmatory_ledger_path: str | Path = ledger_module.CONFIRMATORY_LEDGER_PATH,
    amendments_ledger_path: str | Path = ledger_module.AMENDMENTS_LEDGER_PATH,
    expected_batch_size: int | None = None,
    git_tracked_and_clean=_default_git_tracked_and_clean,
    last_commit_for_path=_default_last_commit_for_path,
    commit_is_ancestor=_default_commit_is_ancestor,
) -> CellTrainResult:
    """Single entry point for Block D gate-authorized training. Enforces,
    IN ORDER:

    1. run_id resolves to exactly one Block D matrix cell
       (resolve_block_d_run_id()).
    2. The committed gate decision is loaded and verified
       (authorize_block_d_cell() -- see its docstring for the full list of
       hard-fail conditions). NOTHING past this point is reachable if
       authorization fails.
    3. The effective Block D configuration (cell + gate provenance) and
       its hash are derived (authorize_block_d_cell()).
    4. A canonical-eligible matching completion is checked
       (check_confirmatory_skip(), keyed on the EFFECTIVE hash -- so an
       existing completion under a now-superseded gate decision is never
       silently treated as still valid; see check_confirmatory_skip()'s
       docstring). If found, returned as an IMMEDIATE skip -- before the
       working-tree check, MPS initialization, checksum verification,
       dataset loading, DataLoader construction, model construction,
       attempt-directory creation, or any ledger append.
    5. Only if execution is actually needed: working tree is clean
       (require_clean_working_tree(), skippable only for injected tests).
    6. Device resolves to MPS with NO silent CPU fallback
       (device_resolver defaults to select_device('mps')).
    7. The official native dataset artifact is independently re-verified
       (dataset_verifier) and its checksum/resized flag are compared
       against the gate decision's recorded values -- a dataset that
       changed on disk since the gate ran, or that is no longer native,
       hard-fails here even though authorize_block_d_cell() already
       passed.
    8. loader_factory(cell, selected_batch, data_root) constructs
       train/validation DataLoaders at the gate-selected batch size.
    9-11. run_train_validation_cell() constructs the model, trains
       (max_training_seconds bounded by the frozen 90-minute per-run
       training limit, and inheriting its existing OOM/non-finite/
       interruption handling unchanged), persists/verifies artifacts
       (with Block D gate provenance merged into result.json/
       metadata.json), and appends the terminal confirmatory-ledger row
       (with the EFFECTIVE config hash).

    A failed authorization (steps 1-3) creates ZERO files and ZERO ledger
    rows -- it raises before any of run_identity.start_attempt(),
    require_clean_working_tree(), device_resolver(), loader_factory(), or
    ledger append are ever called.
    """
    cell = resolve_block_d_run_id(run_id, matrix_path)

    decision, effective_config = authorize_block_d_cell(
        cell,
        decision_path=decision_path,
        matrix_path=matrix_path,
        expected_batch_size=expected_batch_size,
        git_tracked_and_clean=git_tracked_and_clean,
        last_commit_for_path=last_commit_for_path,
        commit_is_ancestor=commit_is_ancestor,
    )
    effective_hash = compute_block_d_effective_config_hash(effective_config)

    skip = check_confirmatory_skip(
        cell, root, amendments_ledger_path, confirmatory_ledger_path, effective_config_hash=effective_hash
    )
    if skip is not None:
        return skip

    if require_clean_tree:
        require_clean_working_tree()
    device = device_resolver()

    verification = dataset_verifier(cell.dataset, cell.resolution, data_root)
    if verification.resized is not False:
        raise BlockDAuthorizationError(
            f"Block D authorization failed for {run_id}: dataset artifact reports "
            f"resized={verification.resized!r}, not exactly False."
        )
    if verification.actual_checksum_md5 != effective_config.dataset_checksum_md5:
        raise BlockDAuthorizationError(
            f"Block D authorization failed for {run_id}: current dataset checksum "
            f"'{verification.actual_checksum_md5}' differs from the gate decision's recorded "
            f"checksum '{effective_config.dataset_checksum_md5}' -- the artifact changed since "
            f"the gate ran."
        )

    bundle = loader_factory(cell, effective_config.selected_batch_size, data_root)

    gate_provenance = {
        "block_d_gate_decision_path": str(decision_path),
        "block_d_gate_decision_sha256": effective_config.decision_artifact_sha256,
        "block_d_gate_decision_commit": effective_config.gate_decision_commit,
        "block_d_benchmark_source_commit": effective_config.benchmark_source_commit,
        "block_d_benchmark_spec_commit": effective_config.benchmark_spec_commit,
        "block_d_selected_batch_size": effective_config.selected_batch_size,
        "block_d_native_resolution": effective_config.native_resolution,
        "block_d_resized": effective_config.resized,
        "block_d_gate_final_decision": decision["final_decision"],
    }

    return run_train_validation_cell(
        cell,
        bundle.train_loader,
        bundle.val_loader,
        device,
        root=root,
        max_training_seconds=_BLOCK_D_MAX_TRAINING_MINUTES_PER_RUN * 60,
        confirmatory_ledger_path=confirmatory_ledger_path,
        amendments_ledger_path=amendments_ledger_path,
        matrix_path=matrix_path,
        dataset_artifact_filename=bundle.dataset_artifact_filename,
        dataset_expected_checksum_md5=bundle.dataset_expected_checksum_md5,
        dataset_actual_checksum_md5=bundle.dataset_actual_checksum_md5,
        effective_config_hash=effective_hash,
        extra_result_fields={"block_d_gate_provenance": gate_provenance},
        extra_metadata_fields={"block_d_gate_provenance": gate_provenance},
    )


### --- end Phase 2B.3F section ------------------------------------------ ###


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
