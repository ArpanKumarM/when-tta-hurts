"""Phase 2B.4A: frozen, validation-only TTA evaluation runner.

Evaluates an existing, canonical, completed confirmatory training
checkpoint against the VALIDATION split only, using the frozen aggregation
methods (evaluation/aggregation.py), frozen BN-adaptation procedure
(evaluation/bn_adaptation.py), frozen metrics (metrics.py), frozen
inference-latency spec (evaluation/latency.py), and the frozen mixed
augmentation policy (transforms/policies.py) -- all reused UNCHANGED from
Phase 2A. The only genuinely new logic here is: canonical-checkpoint
resolution reuse, deterministic per-sample view generation
(evaluation/views.py), evaluation identity/attempt bookkeeping, and
artifact/ledger persistence.

**This module never accesses the official test split.** The only loader
used is evaluation/validation_loader.py::load_validation_evaluation_split,
which has no split parameter at all -- 'val' is hardcoded. Every produced
artifact/ledger row records split="validation" and
test_metrics_observed=False, neither of which is caller-overridable (see
ledger.py::append_evaluation_entry and evaluation_result_artifacts.py's
schema validators).

**No metric ever influences orchestration.** check_evaluation_skip() and
the attempt state machine below never read a metric value -- only
metadata (config hashes, attempt status, manifest validity), exactly
mirroring orchestrator.py::check_confirmatory_skip()'s discipline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from when_tta_hurts.artifacts import atomic_write_json, hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.evaluation.aggregation import (
    confidence_weighted_average,
    majority_vote,
    mean_probability,
    original_anchored_mean_probability,
)
from when_tta_hurts.evaluation.bn_adaptation import BNAdaptationNotApplicableError, bn_adapt
from when_tta_hurts.evaluation.validation_loader import load_validation_evaluation_split
from when_tta_hurts.evaluation.views import build_view_seed_manifest, iter_deterministic_views
from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    persist_and_verify_evaluation_completion,
    recompute_clean_accuracy,
    recompute_mean_probability_prefix,
)
from when_tta_hurts.matrix import MatrixCell, parse_and_validate_matrix
from when_tta_hurts.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    harm_rescue_rates,
    macro_f1,
    negative_log_likelihood,
    softmax,
)
from when_tta_hurts.orchestrator import (
    CellTrainResult,
    PilotOrExcludedSeedRunIdError,
    UnknownRunIdError,
    _build_model,
    authorize_block_d_cell,
    check_confirmatory_skip,
    compute_block_d_effective_config_hash,
)
from when_tta_hurts.transforms.policies import build_policy

FROZEN_PROTOCOL_COMMIT = "ce4c962"
CONFIRMATORY_TTA_SEED = 271828  # NOTE: this is the PILOT's TTA seed -- see
# NoConfirmatoryTTASeedYetError below. docs/phase2b_protocol.md sec.1 is
# explicit that the confirmatory TTA seed must be DISTINCT from the pilot's
# 271828 and "is to be set when the runner is implemented." This module IS
# that implementation, so it must not silently reuse 271828. See
# resolve_confirmatory_tta_seed() -- there is no default value here.

PREFIX_SEQUENCE: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)
MAX_VIEWS = 100
PRIMARY_N = 50
AGGREGATORS: tuple[str, ...] = ("mean_probability", "majority_vote", "confidence_weighted_average")
SECONDARY_ANALYSES: tuple[str, ...] = (
    "scaling_curve",
    "augmentation_ablation",
    "aggregation_ablation",
    "original_anchored",
    "bn_adapted",
)

DEFAULT_EVALUATION_ROOT = Path("artifacts/validation_evaluation")


class NoConfirmatoryTTASeedYetError(RuntimeError):
    """Raised by resolve_confirmatory_tta_seed() -- docs/phase2b_protocol.md
    sec.1 requires a confirmatory TTA seed DISTINCT from the pilot's 271828,
    "to be set when the runner is implemented." This function is that
    decision point; it does not invent a value silently. A caller must
    supply an explicit tta_seed distinct from 271828 (e.g. via CLI/config),
    or this raises -- there is no implicit default."""


def resolve_confirmatory_tta_seed(requested: int | None) -> int:
    if requested is None:
        raise NoConfirmatoryTTASeedYetError(
            "No confirmatory TTA seed was supplied. docs/phase2b_protocol.md sec.1 requires "
            "a confirmatory TTA seed distinct from the pilot's 271828, chosen explicitly -- "
            "this function refuses to invent one silently."
        )
    if requested == 271828:
        raise NoConfirmatoryTTASeedYetError(
            "271828 is the Phase 2A PILOT's TTA seed and must not be reused for confirmatory "
            "TTA view sequences, per docs/phase2b_protocol.md sec.1."
        )
    return requested


class NoCanonicalTrainingCompletionError(RuntimeError):
    """Raised when no canonical, completed confirmatory training attempt
    exists for the requested run_id -- evaluation cannot proceed."""


class EvaluationStaleAttemptError(RuntimeError):
    """Raised when an earlier evaluation attempt is nonterminal (status
    'running'/'planned') and has no ledger row of any kind -- mirrors
    orchestrator.StaleAttemptError for training attempts."""


_TERMINAL_EVAL_STATUSES = frozenset({"completed", "failed", "aborted"})


class EvaluationRunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class EvaluationAttemptStatus:
    training_run_id: str
    attempt_number: int
    status: str
    evaluation_config_hash: str
    started_at: float | None = None
    ended_at: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluation_run_directory(training_run_id: str, root: str | Path = DEFAULT_EVALUATION_ROOT) -> Path:
    return Path(root) / training_run_id


def _list_attempt_dirs(run_dir: Path) -> list[Path]:
    if not run_dir.exists():
        return []
    return sorted(
        (p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")),
        key=lambda p: p.name,
    )


def _read_status(attempt_dir: Path) -> dict[str, Any] | None:
    status_path = attempt_dir / "status.json"
    if not status_path.exists():
        return None
    return json.loads(status_path.read_text())


def list_evaluation_attempts(
    training_run_id: str, root: str | Path = DEFAULT_EVALUATION_ROOT
) -> list[dict[str, Any]]:
    run_dir = evaluation_run_directory(training_run_id, root)
    statuses = [s for d in _list_attempt_dirs(run_dir) if (s := _read_status(d)) is not None]
    statuses.sort(key=lambda s: s["attempt_number"])
    return statuses


def next_evaluation_attempt_number(training_run_id: str, root: str | Path = DEFAULT_EVALUATION_ROOT) -> int:
    run_dir = evaluation_run_directory(training_run_id, root)
    existing = _list_attempt_dirs(run_dir)
    if not existing:
        return 1
    numbers = [int(p.name.split("_")[1]) for p in existing]
    return max(numbers) + 1


def start_evaluation_attempt(
    training_run_id: str,
    evaluation_config_hash: str,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
) -> tuple[Path, EvaluationAttemptStatus]:
    attempt_number = next_evaluation_attempt_number(training_run_id, root)
    run_dir = evaluation_run_directory(training_run_id, root)
    attempt_dir = run_dir / f"attempt_{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)

    status = EvaluationAttemptStatus(
        training_run_id=training_run_id,
        attempt_number=attempt_number,
        status=EvaluationRunStatus.RUNNING.value,
        evaluation_config_hash=evaluation_config_hash,
        started_at=time.time(),
    )
    atomic_write_json(status.to_dict(), attempt_dir / "status.json")
    return attempt_dir, status


def finish_evaluation_attempt(
    attempt_dir: Path,
    status: EvaluationAttemptStatus,
    final_status: EvaluationRunStatus,
    failure_reason: str | None = None,
) -> EvaluationAttemptStatus:
    if final_status not in (
        EvaluationRunStatus.COMPLETED,
        EvaluationRunStatus.FAILED,
        EvaluationRunStatus.ABORTED,
    ):
        raise ValueError(f"finish_evaluation_attempt() requires a terminal status, got {final_status}")
    status.status = final_status.value
    status.ended_at = time.time()
    status.failure_reason = failure_reason
    atomic_write_json(status.to_dict(), attempt_dir / "status.json")
    return status


# ---------------------------------------------------------------------------
# Canonical training-checkpoint resolution (reuses orchestrator's existing,
# already-tested canonical-selection logic -- no new selection algorithm)
# ---------------------------------------------------------------------------


def resolve_canonical_training_completion(
    run_id: str, matrix_path: str = "configs/experiment_matrix.yaml"
) -> tuple[MatrixCell, CellTrainResult]:
    """Resolve `run_id` to its approved matrix cell and canonical, verified,
    completed training attempt. Rejects (by construction/delegation, never
    by re-implementing the checks): pilot/seed-314159 run_ids (explicit,
    before any matrix parsing); unknown run_ids; noncanonical/failed/
    aborted attempts (check_confirmatory_skip only ever returns a
    "completed" attempt or None); amendments-ineligible attempts (excluded
    inside check_confirmatory_skip); ambiguous completions
    (AmbiguousCanonicalCompletionError propagates); missing/corrupt
    manifests (PersistenceVerificationError propagates); checkpoint/config
    hash conflicts (ConflictingCompletedRunError propagates); a Block D
    cell whose gate is not INCLUDED (BlockDAuthorizationError propagates,
    from authorize_block_d_cell -- called before any training-completion
    lookup for Block D cells).
    """
    if "-s314159" in run_id or run_id.startswith("pilot"):
        raise PilotOrExcludedSeedRunIdError(
            f"Refusing run_id '{run_id}': pilot/permanently-excluded-seed identifiers "
            f"are never valid evaluation targets."
        )
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cell = next((c for c in expanded.cells if c.run_id() == run_id), None)
    if cell is None:
        raise UnknownRunIdError(f"'{run_id}' does not match any approved matrix cell.")

    if cell.block == "D_conditional_128px":
        _decision, effective = authorize_block_d_cell(cell, matrix_path=matrix_path)
        effective_hash = compute_block_d_effective_config_hash(effective)
    else:
        effective_hash = None

    result = check_confirmatory_skip(cell, effective_config_hash=effective_hash)
    if result is None:
        raise NoCanonicalTrainingCompletionError(
            f"No canonical, completed confirmatory training attempt exists for '{run_id}' -- "
            f"cannot evaluate an incomplete or nonexistent training result."
        )
    return cell, result


# ---------------------------------------------------------------------------
# Deterministic evaluation identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationEvaluationConfig:
    training_run_id: str
    training_attempt: int
    checkpoint_hash: str
    split: str
    tta_seed: int
    prefix_sequence: tuple[int, ...]
    aggregators: tuple[str, ...]
    secondary_analyses: tuple[str, ...]
    policy: str
    protocol_commit: str
    matrix_hash: str
    source_commit: str


def compute_evaluation_id(cfg: ValidationEvaluationConfig) -> str:
    """Deterministic evaluation ID hashing every field listed in
    ValidationEvaluationConfig -- training run/attempt/checkpoint identity,
    split (always 'validation'), TTA seed, prefix sequence, aggregators/
    secondary-analysis configuration, and protocol/matrix/source
    provenance. Two evaluation requests differing in ANY of these fields
    get a different evaluation_id."""
    return config_hash(asdict(cfg))


def build_validation_evaluation_config(
    cell: MatrixCell,
    training_result: CellTrainResult,
    tta_seed: int,
    matrix_hash: str,
    source_commit: str,
) -> ValidationEvaluationConfig:
    return ValidationEvaluationConfig(
        training_run_id=cell.run_id(),
        training_attempt=training_result.attempt_number,
        checkpoint_hash=training_result.checkpoint_hash,
        split="validation",
        tta_seed=tta_seed,
        prefix_sequence=PREFIX_SEQUENCE,
        aggregators=AGGREGATORS,
        secondary_analyses=SECONDARY_ANALYSES,
        policy="mixed",
        protocol_commit=FROZEN_PROTOCOL_COMMIT,
        matrix_hash=matrix_hash,
        source_commit=source_commit,
    )


# ---------------------------------------------------------------------------
# Idempotent skip
# ---------------------------------------------------------------------------


def check_evaluation_skip(
    training_run_id: str,
    evaluation_config_hash: str,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
) -> dict[str, Any] | None:
    """Metadata-only: does NOT touch MPS, the checkpoint, the dataset, view
    generation, or metric calculation -- only reads status.json/
    artifact_manifest.json files, mirroring
    orchestrator.check_confirmatory_skip()'s discipline exactly (never
    uses a metric value in this decision).

    First, for every attempt (any status): if nonterminal AND unledgered,
    raise EvaluationStaleAttemptError (mirrors StaleAttemptError). Then:
    among completed attempts, the FIRST with a matching
    evaluation_config_hash and a verified artifact_manifest.json is
    returned as the skip target; a completed attempt with a DIFFERENT
    hash is a hard failure (the evaluation config changed underneath an
    existing result)."""
    from when_tta_hurts import ledger as ledger_module
    from when_tta_hurts.evaluation_result_artifacts import verify_evaluation_artifact_manifest

    run_dir = evaluation_run_directory(training_run_id, root)
    all_attempts = list_evaluation_attempts(training_run_id, root)

    for status in all_attempts:
        if status.get("status") not in _TERMINAL_EVAL_STATUSES:
            if not ledger_module.has_evaluation_row(evaluation_config_hash, status["attempt_number"]):
                raise EvaluationStaleAttemptError(
                    f"Evaluation of {training_run_id} attempt_{status['attempt_number']:03d} is "
                    f"nonterminal (status='{status.get('status')}') and has no ledger row -- "
                    f"refusing to start a new attempt without explicit reconciliation."
                )

    for status in all_attempts:
        if status.get("status") != EvaluationRunStatus.COMPLETED.value:
            continue
        if status["evaluation_config_hash"] != evaluation_config_hash:
            continue  # a different config's completed attempt is irrelevant to this request
        attempt_dir = run_dir / f"attempt_{status['attempt_number']:03d}"
        manifest_path = attempt_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise EvaluationPersistenceError(
                f"Completed evaluation attempt_{status['attempt_number']:03d} for "
                f"{training_run_id} is missing artifact_manifest.json."
            )
        manifest = json.loads(manifest_path.read_text())
        verify_evaluation_artifact_manifest(attempt_dir, manifest)
        return status
    return None


# ---------------------------------------------------------------------------
# Plan mode (side-effect-free)
# ---------------------------------------------------------------------------


def plan_validation_evaluation(matrix_path: str = "configs/experiment_matrix.yaml") -> list[dict[str, Any]]:
    """SIDE-EFFECT-FREE. Never initializes MPS, opens a dataset/checkpoint,
    creates a file/directory, writes a ledger, or computes a prediction --
    only reads matrix/ledger/status metadata already on disk. Lists all 39
    eligible training cells with their canonical attempt (if any),
    checkpoint hash (if canonical), a *planned* evaluation ID (using
    tta_seed=None is not possible -- plan mode reports the evaluation ID
    formula and required fields, not a computed ID, since the confirmatory
    TTA seed is a caller-supplied parameter this function does not assume),
    and the required analyses/prefix sequence.
    """
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    rows = []
    for cell in expanded.cells:
        run_id = cell.run_id()
        canonical_status: dict[str, Any] = {"canonical": False}
        try:
            if cell.block == "D_conditional_128px":
                _decision, effective = authorize_block_d_cell(cell, matrix_path=matrix_path)
                effective_hash = compute_block_d_effective_config_hash(effective)
            else:
                effective_hash = None
            result = check_confirmatory_skip(cell, effective_config_hash=effective_hash)
            if result is not None:
                canonical_status = {
                    "canonical": True,
                    "attempt_number": result.attempt_number,
                    "checkpoint_hash": result.checkpoint_hash,
                }
        except Exception as e:  # noqa: BLE001 -- plan mode reports, never raises, on a per-cell basis
            canonical_status = {"canonical": False, "error": f"{type(e).__name__}: {e}"}

        rows.append(
            {
                "run_id": run_id,
                "dataset": cell.dataset,
                "resolution": cell.resolution,
                "model": cell.model,
                "normalization": cell.normalization,
                "seed": cell.seed,
                "training_completion": canonical_status,
                "prefix_sequence": list(PREFIX_SEQUENCE),
                "aggregators": list(AGGREGATORS),
                "secondary_analyses": list(SECONDARY_ANALYSES),
                "estimated_artifact_dir": str(evaluation_run_directory(run_id)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Checkpoint loading (read-only, verified, never mutated)
# ---------------------------------------------------------------------------


def _training_checkpoint_path(
    cell: MatrixCell, training_result: CellTrainResult, root: str | Path = "artifacts/confirmatory"
) -> Path:
    from when_tta_hurts.run_identity import run_directory

    attempt_dir = run_directory(cell, root) / f"attempt_{training_result.attempt_number:03d}"
    return attempt_dir / "best_checkpoint.pt"


def load_and_verify_canonical_checkpoint(
    cell: MatrixCell, training_result: CellTrainResult, root: str | Path = "artifacts/confirmatory"
) -> torch.nn.Module:
    """Load the canonical training checkpoint READ-ONLY (never mutated,
    never re-saved) and verify its on-disk content hash matches the
    training result's recorded checkpoint_hash before returning a model
    with it strictly loaded."""
    from when_tta_hurts.artifacts import hash_state_dict

    ckpt_path = _training_checkpoint_path(cell, training_result, root)
    if not ckpt_path.exists():
        raise EvaluationPersistenceError(f"Canonical checkpoint {ckpt_path} does not exist.")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    actual_hash = hash_state_dict(state_dict)
    if actual_hash != training_result.checkpoint_hash:
        raise EvaluationPersistenceError(
            f"Checkpoint hash mismatch for {cell.run_id()}: recorded "
            f"{training_result.checkpoint_hash}, on-disk file hashes to {actual_hash}."
        )
    model = _build_model(cell)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Real evaluation computation (per-condition metrics from stored per-view
# probabilities wherever mathematically possible)
# ---------------------------------------------------------------------------


def _per_prefix_metrics(clean_probs: np.ndarray, agg_probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    clean_logp = np.log(np.clip(clean_probs, 1e-12, 1.0))
    hr = harm_rescue_rates(clean_logp, agg_probs, labels)
    return {
        "accuracy": accuracy(agg_probs, labels),
        "macro_f1": macro_f1(agg_probs, labels),
        "negative_log_likelihood": negative_log_likelihood(agg_probs, labels),
        "expected_calibration_error": expected_calibration_error(agg_probs, labels),
        "brier_score": brier_score(agg_probs, labels),
        "delta_accuracy": accuracy(agg_probs, labels) - accuracy(clean_logp, labels),
        "harm_rate": hr["harm_rate"],
        "rescue_rate": hr["rescue_rate"],
    }


def compute_validation_evaluation(
    model: torch.nn.Module,
    split,
    tta_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Compute clean probabilities, the full 100-view probability bank, and
    every frozen condition/aggregator/prefix metric. `split`: a
    ValidationEvaluationSplit (evaluation/validation_loader.py). Returns a
    dict with 'predictions' (arrays for persistence) and 'metrics' (the
    full nested metrics structure) -- never persists images.
    """
    model = model.to(device)
    model.eval()

    x = split.images.to(device)
    labels = split.labels
    sample_indices = split.sample_indices.tolist()

    with torch.no_grad():
        clean_logits = model(x).detach().cpu().numpy()
    clean_probs = softmax(clean_logits)
    n_classes = clean_probs.shape[-1]  # derived from the model's actual output, not re-looked-up metadata

    policy = build_policy("mixed", output_size=(split.resolution, split.resolution))

    view_probs = np.empty((MAX_VIEWS, len(sample_indices), n_classes), dtype=np.float32)
    for view_index, view_batch in iter_deterministic_views(
        split.images, policy, tta_seed, split.dataset, split.resolution, sample_indices, MAX_VIEWS
    ):
        with torch.no_grad():
            view_logits = model(view_batch.to(device)).detach().cpu().numpy()
        view_probs[view_index] = softmax(view_logits)
    view_log_probs = np.log(np.clip(view_probs, 1e-12, 1.0))  # for aggregation.py's log-space convention

    conditions: dict[str, Any] = {"naive_tta": {agg: {} for agg in AGGREGATORS}}
    for n in PREFIX_SEQUENCE:
        conditions["naive_tta"]["mean_probability"][n] = _per_prefix_metrics(
            clean_probs, softmax(mean_probability(view_log_probs, n)), labels
        )
        _, vote_log_probs = majority_vote(view_log_probs, n)
        conditions["naive_tta"]["majority_vote"][n] = _per_prefix_metrics(
            clean_probs, softmax(vote_log_probs), labels
        )
        conditions["naive_tta"]["confidence_weighted_average"][n] = _per_prefix_metrics(
            clean_probs, softmax(confidence_weighted_average(view_log_probs, n)), labels
        )

    conditions["original_anchored_tta"] = {
        n: _per_prefix_metrics(
            clean_probs, softmax(original_anchored_mean_probability(clean_logits, view_log_probs, n)), labels
        )
        for n in PREFIX_SEQUENCE
    }

    bn_adapted_probs_by_n: dict[int, np.ndarray] = {}
    try:
        conditions["bn_adapted_tta"] = {}
        for n in PREFIX_SEQUENCE:
            adaptation_views = [
                view_batch
                for _view_index, view_batch in iter_deterministic_views(
                    split.images, policy, tta_seed, split.dataset, split.resolution, sample_indices, n
                )
            ]
            adaptation_inputs = torch.cat(adaptation_views, dim=0).to(device)
            adapted_model = bn_adapt(model, adaptation_inputs)
            with torch.no_grad():
                adapted_logits = adapted_model(x).detach().cpu().numpy()
            adapted_probs = softmax(adapted_logits)
            bn_adapted_probs_by_n[n] = adapted_probs
            conditions["bn_adapted_tta"][n] = _per_prefix_metrics(clean_probs, adapted_probs, labels)
    except BNAdaptationNotApplicableError:
        conditions["bn_adapted_tta"] = (
            None  # GroupNorm -- explicitly recorded as not applicable, never silent
        )

    clean_log_probs = np.log(np.clip(clean_probs, 1e-12, 1.0))
    predictions: dict[str, np.ndarray] = {
        "labels": labels,
        "sample_indices": split.sample_indices,
        "clean_probs": clean_probs.astype(np.float32),
        "view_probs": view_probs,
    }
    if bn_adapted_probs_by_n:
        predictions["bn_adapted_probs"] = bn_adapted_probs_by_n[max(bn_adapted_probs_by_n)].astype(np.float32)

    return {
        "predictions": predictions,
        "metrics": {
            "clean": {
                "accuracy": accuracy(clean_log_probs, labels),
                "macro_f1": macro_f1(clean_log_probs, labels),
                "negative_log_likelihood": negative_log_likelihood(clean_log_probs, labels),
                "expected_calibration_error": expected_calibration_error(clean_log_probs, labels),
                "brier_score": brier_score(clean_log_probs, labels),
            },
            "conditions": conditions,
            "primary_endpoint": {
                "clean_accuracy": recompute_clean_accuracy(clean_probs, labels),
                "tta_accuracy_at_n50_mean_probability": conditions["naive_tta"]["mean_probability"][
                    PRIMARY_N
                ]["accuracy"],
                "delta_accuracy_pp": 100.0
                * conditions["naive_tta"]["mean_probability"][PRIMARY_N]["delta_accuracy"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_validation_evaluation(
    run_id: str,
    tta_seed: int,
    matrix_path: str = "configs/experiment_matrix.yaml",
    device_resolver=None,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
    training_root: str | Path = "artifacts/confirmatory",
    data_root: str | Path = "data/raw",
    evaluation_ledger_path=None,
) -> dict[str, Any]:
    """Single entry point for validation-only TTA evaluation. Enforces, IN
    ORDER: (1) resolve the canonical training completion (rejects every
    ineligible attempt type -- see resolve_canonical_training_completion());
    (2) derive the deterministic evaluation config/ID; (3) idempotent skip
    check (metadata-only, before MPS/dataset/checkpoint/view-generation/
    metric-calculation); (4) MPS device resolution (no CPU fallback --
    device_resolver defaults to select_device('mps')); (5) load + verify
    the canonical checkpoint read-only; (6) load the validation split ONLY;
    (7) compute clean + per-view probabilities and every frozen condition/
    aggregator/prefix metric; (8) persist + independently verify artifacts;
    (9) append the terminal evaluation-ledger row.

    A failed resolution/skip/authorization step (1)-(3) creates ZERO files
    and ZERO ledger rows.
    """
    from when_tta_hurts import ledger as ledger_module
    from when_tta_hurts.devices import select_device

    if device_resolver is None:
        device_resolver = lambda: select_device("mps")  # noqa: E731
    if evaluation_ledger_path is None:
        evaluation_ledger_path = ledger_module.VALIDATION_EVALUATION_LEDGER_PATH

    resolved_tta_seed = resolve_confirmatory_tta_seed(tta_seed)

    cell, training_result = resolve_canonical_training_completion(run_id, matrix_path)

    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cfg = build_validation_evaluation_config(
        cell, training_result, resolved_tta_seed, expanded.source_config_hash, _git_commit_hash()
    )
    evaluation_id = compute_evaluation_id(cfg)

    skip = check_evaluation_skip(run_id, evaluation_id, root)
    if skip is not None:
        return {
            "status": "skipped_completed",
            "training_run_id": run_id,
            "evaluation_id": evaluation_id,
            **skip,
        }

    device = device_resolver()
    attempt_dir, status = start_evaluation_attempt(run_id, evaluation_id, root)
    try:
        model = load_and_verify_canonical_checkpoint(cell, training_result, training_root)
        split = load_validation_evaluation_split(cell.dataset, cell.resolution, data_root)
        outcome = compute_validation_evaluation(model, split, resolved_tta_seed, device)

        seed_manifest = build_view_seed_manifest(
            resolved_tta_seed, cell.dataset, cell.resolution, split.sample_indices.tolist(), MAX_VIEWS
        )
        seed_manifest_hash = config_hash({"entries": [asdict(e) for e in seed_manifest]})

        metadata = {
            "evaluation_id": evaluation_id,
            "training_run_id": run_id,
            "training_attempt": training_result.attempt_number,
            "checkpoint_hash": training_result.checkpoint_hash,
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "model": cell.model,
            "normalization": cell.normalization,
            "training_policy": cell.training_policy,
            "seed": cell.seed,
            "tta_seed": resolved_tta_seed,
            "prefix_sequence": list(PREFIX_SEQUENCE),
            "aggregators": list(AGGREGATORS),
            "secondary_analyses": list(SECONDARY_ANALYSES),
            "protocol_commit": cfg.protocol_commit,
            "matrix_hash": cfg.matrix_hash,
            "source_commit": cfg.source_commit,
            "evaluation_config_hash": evaluation_id,
            "split": "validation",
            "n_validation_samples": len(split.sample_indices),
        }
        view_manifest = {
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "tta_seed": resolved_tta_seed,
            "n_views": MAX_VIEWS,
            "seed_formula": (
                "sha256(tta_seed|dataset|resolution|sample_index|view_index)[:8 bytes] % (2**31-1)"
            ),
            "sample_indices": split.sample_indices.tolist(),
            "seed_manifest_sha256": seed_manifest_hash,
        }
        metrics = {"training_run_id": run_id, "evaluation_config_hash": evaluation_id, **outcome["metrics"]}

        def _recompute_n50_mean_probability_accuracy() -> float:
            n50_probs = recompute_mean_probability_prefix(outcome["predictions"]["view_probs"], PRIMARY_N)
            n50_log_probs = np.log(np.clip(n50_probs, 1e-12, 1.0))
            return accuracy(n50_log_probs, split.labels)

        metric_recomputers = {
            "clean.accuracy": (
                metrics["clean"]["accuracy"],
                lambda: recompute_clean_accuracy(outcome["predictions"]["clean_probs"], split.labels),
            ),
            "naive_tta.mean_probability.N50.accuracy": (
                metrics["conditions"]["naive_tta"]["mean_probability"][PRIMARY_N]["accuracy"],
                _recompute_n50_mean_probability_accuracy,
            ),
        }

        manifest = persist_and_verify_evaluation_completion(
            attempt_dir,
            predictions=outcome["predictions"],
            metrics=metrics,
            metadata=metadata,
            view_manifest=view_manifest,
            metric_recomputers=metric_recomputers,
        )

        finish_evaluation_attempt(attempt_dir, status, EvaluationRunStatus.COMPLETED)
        primary_artifact_hash = hash_file(attempt_dir / "predictions.npz")
        ledger_module.append_evaluation_entry(
            evaluation_id=evaluation_id,
            training_run_id=run_id,
            training_attempt=training_result.attempt_number,
            checkpoint_hash=training_result.checkpoint_hash,
            evaluation_config_hash=evaluation_id,
            evaluation_attempt=status.attempt_number,
            status="completed",
            primary_artifact_hash=primary_artifact_hash,
            started_at=status.started_at,
            ended_at=status.ended_at,
            runtime_seconds=status.ended_at - status.started_at,
            ledger_path=evaluation_ledger_path,
        )
        return {
            "status": "completed",
            "training_run_id": run_id,
            "evaluation_id": evaluation_id,
            "attempt_number": status.attempt_number,
            "artifact_manifest": manifest,
        }
    except Exception as e:
        finish_evaluation_attempt(attempt_dir, status, EvaluationRunStatus.FAILED, failure_reason=str(e))
        ledger_module.append_evaluation_entry(
            evaluation_id=evaluation_id,
            training_run_id=run_id,
            training_attempt=training_result.attempt_number,
            checkpoint_hash=training_result.checkpoint_hash,
            evaluation_config_hash=evaluation_id,
            evaluation_attempt=status.attempt_number,
            status="failed",
            primary_artifact_hash="",
            started_at=status.started_at,
            ended_at=status.ended_at or time.time(),
            runtime_seconds=(status.ended_at or time.time()) - status.started_at,
            failure_reason=str(e),
            ledger_path=evaluation_ledger_path,
        )
        raise


def _git_commit_hash() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
