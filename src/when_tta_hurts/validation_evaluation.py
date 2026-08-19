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

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from when_tta_hurts.artifacts import atomic_write_json, hash_file
from when_tta_hurts.block_d_benchmark import _sha256_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.evaluation.aggregation import (
    confidence_weighted_average,
    majority_vote,
    mean_probability,
    original_anchored_mean_probability,
)
from when_tta_hurts.evaluation.bn_adaptation import BNAdaptationNotApplicableError, bn_adapt
from when_tta_hurts.evaluation.latency import LatencyReport, measure_clean_latency, measure_tta_latency
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
    _default_commit_is_ancestor,
    _default_git_tracked_and_clean,
    _default_last_commit_for_path,
    authorize_block_d_cell,
    check_confirmatory_skip,
    compute_block_d_effective_config_hash,
)
from when_tta_hurts.transforms.policies import build_policy

FROZEN_PROTOCOL_COMMIT = "ce4c962"

PREFIX_SEQUENCE: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)
MAX_VIEWS = 100
PRIMARY_N = 50
PRIMARY_AGGREGATION = "mean_probability"
POLICY_IDENTIFIER = "mixed"
AGGREGATORS: tuple[str, ...] = ("mean_probability", "majority_vote", "confidence_weighted_average")
SECONDARY_ANALYSES: tuple[str, ...] = (
    "scaling_curve",
    "augmentation_ablation",
    "aggregation_ablation",
    "original_anchored",
    "bn_adapted",
)

DEFAULT_EVALUATION_ROOT = Path("artifacts/validation_evaluation")

# --- Phase 2B.4D-Engineering: evaluator-implementation fingerprint -------
#
# See docs/phase2b_validation_evaluation_engineering_freeze.md sec.2.3 for
# the frozen rationale/category coverage of every entry below. Content
# fingerprint of these files (never the literal current git HEAD) is what
# makes evaluation_id stable across ledger/doc-only commits while still
# failing closed on a real evaluation-relevant code/config change.
# Corrected in Phase 2B.4D-Engineering Addendum
# (docs/phase2b_validation_evaluation_engineering_addendum.md) after a
# mechanical transitive-closure audit found this manifest was missing:
# models/resnet.py (Block C/D use ResNet-18, not just SmallCNN),
# orchestrator.py (_build_model() -- model-architecture dispatch --
# lives here; included as a whole file, a deliberately conservative
# choice per the addendum's "false invalidation is preferable to silent
# reuse" principle, even though it also contains unrelated clean-tree/
# selection logic whose OWN effect is already captured via checkpoint_
# hash/training_attempt), matrix.py (cell-expansion logic that resolves
# model/dataset/resolution/normalization per cell -- NOT fully captured
# by matrix_hash, which only hashes the pre-expansion raw YAML), data.py
# (dataset loading/preprocessing/class-count resolution), devices.py
# (MPS/CPU device resolution -- affects which physical device executes
# forward passes), config.py (the config_hash() algorithm itself -- the
# earlier "self-referential" exclusion rationale did not hold up:
# fingerprinting a file's raw bytes via hash_file() has no circularity),
# and pyproject.toml/uv.lock (exact runtime dependency versions).
EVALUATOR_FINGERPRINT_MANIFEST: tuple[str, ...] = (
    "configs/validation_evaluation.yaml",
    "pyproject.toml",
    "src/when_tta_hurts/artifacts.py",
    "src/when_tta_hurts/config.py",
    "src/when_tta_hurts/data.py",
    "src/when_tta_hurts/devices.py",
    "src/when_tta_hurts/evaluation/aggregation.py",
    "src/when_tta_hurts/evaluation/bn_adaptation.py",
    "src/when_tta_hurts/evaluation/latency.py",
    "src/when_tta_hurts/evaluation/validation_loader.py",
    "src/when_tta_hurts/evaluation/views.py",
    "src/when_tta_hurts/evaluation_result_artifacts.py",
    "src/when_tta_hurts/matrix.py",
    "src/when_tta_hurts/metrics.py",
    "src/when_tta_hurts/models/resnet.py",
    "src/when_tta_hurts/models/small_cnn.py",
    "src/when_tta_hurts/orchestrator.py",
    "src/when_tta_hurts/transforms/policies.py",
    "src/when_tta_hurts/validation_evaluation.py",
    "uv.lock",
)


class EvaluatorFingerprintError(RuntimeError):
    """Raised when a file listed in EVALUATOR_FINGERPRINT_MANIFEST does
    not exist on disk at fingerprint-computation time -- never silently
    fingerprints a partial manifest."""


def compute_evaluator_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = EVALUATOR_FINGERPRINT_MANIFEST,
) -> tuple[str, dict[str, str]]:
    """Deterministic content fingerprint of every evaluation-relevant
    tracked file in `manifest`, computed from the actual working-tree
    bytes via the existing, unchanged artifacts.py::hash_file(). Returns
    (fingerprint, {path: sha256}) -- both are persisted (see
    metadata.json's evaluator_fingerprint/evaluator_fingerprint_manifest
    fields). This is a STABLE identity input (unlike source_commit, which
    is provenance-only and not hashed into evaluation_id)."""
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise EvaluatorFingerprintError(
                f"Evaluator-fingerprint manifest file missing: {rel_path}. Refusing to compute a "
                f"partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


# --- Phase 2B.4B: frozen confirmatory TTA seed ---------------------------
#
# docs/phase2b_protocol.md sec.1 deferred the confirmatory TTA seed to
# "when the runner is implemented." That freeze happened in a SEPARATE,
# committed, tracked artifact -- configs/validation_evaluation.yaml (see
# docs/phase2b_validation_evaluation_freeze.md) -- not as a hardcoded
# constant here and not as a CLI parameter. This module never accepts a
# seed value from a caller in production; load_frozen_tta_seed_config()
# below is the ONLY path to a confirmatory TTA seed, and it always
# verifies the seed equals 1306178015, differs from the pilot TTA seed
# (271828), the pilot training seed (314159), and the confirmatory
# training seeds (0, 1, 2), and that the loaded prefix/N/aggregation/
# policy configuration matches this module's own frozen constants above.
_FROZEN_TTA_SEED = 1306178015
_EXCLUDED_TTA_SEEDS: frozenset[int] = frozenset({271828, 314159, 0, 1, 2})
DEFAULT_TTA_SEED_CONFIG_PATH = Path("configs/validation_evaluation.yaml")


class FrozenTTASeedConfigError(RuntimeError):
    """Raised whenever the frozen confirmatory TTA-seed configuration
    (configs/validation_evaluation.yaml) fails validation for ANY reason
    -- missing, untracked, dirty, malformed, draft/unapproved, wrong
    split, a seed other than 1306178015, a seed colliding with an
    excluded value, view/prefix/aggregation/policy configuration
    diverging from this module's frozen constants, no commit in
    repository history, or a freeze commit that is not an ancestor of
    the current HEAD. ALWAYS raised before evaluation identity/attempt
    creation -- see run_validation_evaluation()."""


@dataclass(frozen=True)
class FrozenTTASeedConfig:
    confirmatory_tta_seed: int
    prefix_sequence: tuple[int, ...]
    total_generated_views: int
    primary_prefix: int
    primary_aggregation: str
    policy_identifier: str
    derivation_namespace: str
    derivation_sha256: str
    config_file_sha256: str
    freeze_commit: str
    config_path: str


_REQUIRED_TTA_SEED_CONFIG_FIELDS = {
    "schema_version",
    "status",
    "split",
    "confirmatory_tta_seed",
    "derivation",
    "excluded_seeds",
    "prefix_sequence",
    "total_generated_views",
    "primary_prefix",
    "primary_aggregation",
    "policy_identifier",
}


def load_frozen_tta_seed_config(
    config_path: str | Path = DEFAULT_TTA_SEED_CONFIG_PATH,
    git_tracked_and_clean=_default_git_tracked_and_clean,
    last_commit_for_path=_default_last_commit_for_path,
    commit_is_ancestor=_default_commit_is_ancestor,
    head_commit: str | None = None,
    expected_config_sha256: str | None = None,
) -> FrozenTTASeedConfig:
    """Load and exhaustively verify configs/validation_evaluation.yaml.
    The injectable git_tracked_and_clean/last_commit_for_path/
    commit_is_ancestor/head_commit parameters exist ONLY for synthetic
    tests (mirroring authorize_block_d_cell()'s identical pattern) -- no
    production call site (CLI or otherwise) overrides them."""
    path = Path(config_path)
    if not path.exists():
        raise FrozenTTASeedConfigError(
            f"{path} does not exist -- the confirmatory TTA seed has not been frozen. "
            f"See docs/phase2b_validation_evaluation_freeze.md."
        )
    if not git_tracked_and_clean(path):
        raise FrozenTTASeedConfigError(
            f"{path} is not a clean, git-tracked artifact -- refusing to trust an "
            f"untracked, uncommitted, or locally-modified TTA-seed configuration."
        )

    import yaml

    try:
        raw = yaml.safe_load(path.read_text())
    except Exception as e:
        raise FrozenTTASeedConfigError(f"{path} is malformed: {e}") from e
    if not isinstance(raw, dict):
        raise FrozenTTASeedConfigError(f"{path} must parse to a mapping.")

    missing = _REQUIRED_TTA_SEED_CONFIG_FIELDS - set(raw.keys())
    if missing:
        raise FrozenTTASeedConfigError(f"{path} missing required field(s): {sorted(missing)}")

    if raw["status"] != "approved":
        raise FrozenTTASeedConfigError(f"{path} status is '{raw['status']}', not 'approved'.")
    if raw["split"] != "validation":
        raise FrozenTTASeedConfigError(f"{path} split must be 'validation', got {raw['split']!r}.")

    seed = raw["confirmatory_tta_seed"]
    if seed != _FROZEN_TTA_SEED:
        raise FrozenTTASeedConfigError(
            f"{path} confirmatory_tta_seed must be {_FROZEN_TTA_SEED}, got {seed!r}."
        )
    if seed in _EXCLUDED_TTA_SEEDS:
        raise FrozenTTASeedConfigError(f"confirmatory_tta_seed {seed} collides with an excluded seed.")

    derivation = raw["derivation"]
    namespace = derivation["namespace"]
    digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
    if digest != derivation["sha256_digest"]:
        raise FrozenTTASeedConfigError(
            f"{path}: recomputed SHA-256 of the derivation namespace does not match the "
            f"recorded digest -- refusing to trust an inconsistent derivation."
        )
    if int(digest[:8], 16) != seed:
        raise FrozenTTASeedConfigError(
            f"{path}: derived integer int(digest[:8], 16)={int(digest[:8], 16)} does not "
            f"match confirmatory_tta_seed={seed}."
        )

    if tuple(raw["prefix_sequence"]) != PREFIX_SEQUENCE:
        raise FrozenTTASeedConfigError(
            f"{path} prefix_sequence {raw['prefix_sequence']} diverges from the frozen "
            f"PREFIX_SEQUENCE {PREFIX_SEQUENCE}."
        )
    if raw["total_generated_views"] != MAX_VIEWS:
        raise FrozenTTASeedConfigError(
            f"{path} total_generated_views={raw['total_generated_views']} diverges from "
            f"MAX_VIEWS={MAX_VIEWS}."
        )
    if raw["primary_prefix"] != PRIMARY_N:
        raise FrozenTTASeedConfigError(
            f"{path} primary_prefix={raw['primary_prefix']} diverges from PRIMARY_N={PRIMARY_N}."
        )
    if raw["primary_aggregation"] != PRIMARY_AGGREGATION:
        raise FrozenTTASeedConfigError(
            f"{path} primary_aggregation={raw['primary_aggregation']!r} diverges from "
            f"{PRIMARY_AGGREGATION!r}."
        )
    if raw["policy_identifier"] != POLICY_IDENTIFIER:
        raise FrozenTTASeedConfigError(
            f"{path} policy_identifier={raw['policy_identifier']!r} diverges from {POLICY_IDENTIFIER!r}."
        )

    freeze_commit = last_commit_for_path(path)
    if freeze_commit is None:
        raise FrozenTTASeedConfigError(f"{path} has no commit in repository history.")
    head = head_commit if head_commit is not None else _git_commit_hash()
    if not commit_is_ancestor(freeze_commit, head):
        raise FrozenTTASeedConfigError(
            f"Freeze commit {freeze_commit} for {path} is not an ancestor of current HEAD {head}."
        )

    config_file_sha256 = _sha256_file(path)
    if expected_config_sha256 is not None and config_file_sha256 != expected_config_sha256:
        raise FrozenTTASeedConfigError(
            f"{path} SHA-256 {config_file_sha256} does not match expected {expected_config_sha256}."
        )

    return FrozenTTASeedConfig(
        confirmatory_tta_seed=seed,
        prefix_sequence=tuple(raw["prefix_sequence"]),
        total_generated_views=raw["total_generated_views"],
        primary_prefix=raw["primary_prefix"],
        primary_aggregation=raw["primary_aggregation"],
        policy_identifier=raw["policy_identifier"],
        derivation_namespace=namespace,
        derivation_sha256=digest,
        config_file_sha256=config_file_sha256,
        freeze_commit=freeze_commit,
        config_path=str(path),
    )


class NoCanonicalTrainingCompletionError(RuntimeError):
    """Raised when no canonical, completed confirmatory training attempt
    exists for the requested run_id -- evaluation cannot proceed."""


class EvaluationStaleAttemptError(RuntimeError):
    """Raised when an earlier evaluation attempt is nonterminal (status
    'running'/'planned') and has no ledger row of any kind -- mirrors
    orchestrator.StaleAttemptError for training attempts."""


class EvaluationLedgerConflictError(RuntimeError):
    """Raised when an evaluation attempt directory and the evaluation
    ledger disagree about the same (training_run_id, attempt_number) in
    a way that is NOT the explicit, sanctioned "ledger-recorded
    aborted/failed attempt whose directory was deleted" case (see
    docs/phase2b_validation_evaluation_incident.md) -- e.g. a directory
    and ledger row for the same attempt number with different
    evaluation_config_hash values, a terminal-status directory with no
    corresponding ledger row at all, or a ledger row claiming
    status="completed" with no directory to back it. Always a hard
    failure requiring explicit investigation, never silently resolved."""


class AmbiguousEvaluationCompletionError(RuntimeError):
    """Raised when more than one completed, directory-backed, artifact-
    verified evaluation attempt for the same training_run_id matches the
    same evaluation_config_hash -- mirrors
    orchestrator.AmbiguousCanonicalCompletionError's discipline exactly.
    There is no evaluation-side amendments-ledger equivalent (no
    eligibility-filtering step), so this is never silently resolved by
    earliest/latest recency -- it always requires explicit human
    reconciliation."""


class ConflictingEvaluationImplementationError(RuntimeError):
    """Raised when a training_run_id already has a COMPLETED evaluation
    attempt recorded under a DIFFERENT evaluation_config_hash than the
    current request -- i.e. the evaluator-implementation fingerprint, the
    frozen evaluation configuration, or the resolved checkpoint identity
    changed underneath an existing canonical completion (Phase
    2B.4D-Engineering Addendum). Mirrors
    orchestrator.ConflictingCompletedRunError's discipline on the training
    side exactly: a completed result under one implementation must never
    silently permit a second, different-identity "canonical" completion
    for the same training run -- this is always a hard failure requiring
    explicit reconciliation, never a silent "proceed with a new attempt."
    An aborted/failed attempt never triggers this -- only a COMPLETED one
    counts as an existing canonical result."""


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


def _evaluation_ledger_rows_for_run(training_run_id: str, ledger_path=None) -> list[dict[str, Any]]:
    from when_tta_hurts import ledger as ledger_module

    if ledger_path is None:
        ledger_path = ledger_module.VALIDATION_EVALUATION_LEDGER_PATH
    return [
        row
        for row in ledger_module._read_existing_rows(ledger_path)
        if row.get("training_run_id") == training_run_id
    ]


def next_evaluation_attempt_number(
    training_run_id: str,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
    ledger_path=None,
) -> int:
    """Considers BOTH attempt directories on disk AND evaluation-ledger
    rows for `training_run_id` -- a ledger-recorded attempt number stays
    permanently reserved even if its directory was deleted (see
    docs/phase2b_validation_evaluation_incident.md). Returns
    max(all known attempt numbers) + 1, or 1 if none exist."""
    run_dir = evaluation_run_directory(training_run_id, root)
    dir_numbers = [int(p.name.split("_")[1]) for p in _list_attempt_dirs(run_dir)]
    ledger_numbers = [
        int(row["evaluation_attempt"])
        for row in _evaluation_ledger_rows_for_run(training_run_id, ledger_path)
    ]
    all_numbers = dir_numbers + ledger_numbers
    if not all_numbers:
        return 1
    return max(all_numbers) + 1


def start_evaluation_attempt(
    training_run_id: str,
    evaluation_config_hash: str,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
    ledger_path=None,
) -> tuple[Path, EvaluationAttemptStatus]:
    attempt_number = next_evaluation_attempt_number(training_run_id, root, ledger_path)
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
    evaluator_fingerprint: str
    tta_seed_config_sha256: str
    tta_seed_freeze_commit: str
    tta_seed_derivation_sha256: str


def compute_evaluation_id(cfg: ValidationEvaluationConfig) -> str:
    """Deterministic, STABLE evaluation ID hashing every field listed in
    ValidationEvaluationConfig -- training run/attempt/checkpoint identity,
    split (always 'validation'), the frozen TTA seed AND its config-file
    SHA-256/freeze commit/derivation digest, prefix sequence, aggregators/
    secondary-analysis configuration, protocol/matrix provenance, and the
    evaluator-implementation fingerprint (compute_evaluator_fingerprint()).
    Deliberately does NOT hash the literal current git HEAD (that is
    execution provenance -- see source_commit, persisted separately,
    unhashed, in metadata.json) -- see
    docs/phase2b_validation_evaluation_engineering_freeze.md sec.2 for the
    stability rationale. Two evaluation requests differing in ANY hashed
    field get a different evaluation_id; a ledger/doc-only commit that
    changes none of them does not."""
    return config_hash(asdict(cfg))


def build_validation_evaluation_config(
    cell: MatrixCell,
    training_result: CellTrainResult,
    tta_seed_config: FrozenTTASeedConfig,
    matrix_hash: str,
    evaluator_fingerprint: str,
) -> ValidationEvaluationConfig:
    return ValidationEvaluationConfig(
        training_run_id=cell.run_id(),
        training_attempt=training_result.attempt_number,
        checkpoint_hash=training_result.checkpoint_hash,
        split="validation",
        tta_seed=tta_seed_config.confirmatory_tta_seed,
        prefix_sequence=tta_seed_config.prefix_sequence,
        aggregators=AGGREGATORS,
        secondary_analyses=SECONDARY_ANALYSES,
        policy=tta_seed_config.policy_identifier,
        protocol_commit=FROZEN_PROTOCOL_COMMIT,
        matrix_hash=matrix_hash,
        evaluator_fingerprint=evaluator_fingerprint,
        tta_seed_config_sha256=tta_seed_config.config_file_sha256,
        tta_seed_freeze_commit=tta_seed_config.freeze_commit,
        tta_seed_derivation_sha256=tta_seed_config.derivation_sha256,
    )


# ---------------------------------------------------------------------------
# Idempotent skip
# ---------------------------------------------------------------------------


def check_evaluation_skip(
    training_run_id: str,
    evaluation_config_hash: str,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
    ledger_path=None,
) -> dict[str, Any] | None:
    """Metadata-only: does NOT touch MPS, the checkpoint, the dataset, view
    generation, or metric calculation -- only reads status.json/
    artifact_manifest.json/ledger-row metadata, mirroring
    orchestrator.check_confirmatory_skip()'s discipline exactly (never
    uses a metric value in this decision).

    In order:
    1. For every attempt DIRECTORY (any status): if nonterminal AND
       unledgered, raise EvaluationStaleAttemptError.
    2. Ledger/directory consistency, per (training_run_id, attempt_number):
       - directory + ledger row both exist for the same attempt number
         but with DIFFERENT evaluation_config_hash -> EvaluationLedgerConflictError.
       - a ledger row exists with NO directory: sanctioned ONLY if the
         ledger row's status is "aborted" or "failed" (the explicit
         deleted-terminal-attempt case -- see
         docs/phase2b_validation_evaluation_incident.md); a "completed"
         ledger row with no directory is NOT sanctioned and hard-fails,
         since a completed evaluation's artifacts must never simply
         vanish.
       - a directory exists with terminal status but NO ledger row ->
         EvaluationLedgerConflictError (a completed/failed/aborted
         attempt must always have been ledger-recorded by the production
         path; its absence indicates a crash between finish and append
         that needs explicit reconciliation, not a silent retry).
    3. Among completed attempts (directory-backed only, by construction
       of check 2 above): if more than one matches evaluation_config_hash
       (verified artifact_manifest.json each), raise
       AmbiguousEvaluationCompletionError; if exactly one matches, return
       it as the skip target. If NONE matches but at least one completed
       attempt exists under a DIFFERENT evaluation_config_hash, raise
       ConflictingEvaluationImplementationError -- a completed evaluation
       under one implementation/config/checkpoint identity must never
       silently permit a second, different-identity "canonical"
       completion for the same training_run_id (Phase 2B.4D-Engineering
       Addendum). Only if there is truly no completed attempt at all
       (only failed/aborted, or none) does this return None, permitting a
       new numbered attempt to proceed.
    """
    from when_tta_hurts import ledger as ledger_module
    from when_tta_hurts.evaluation_result_artifacts import verify_evaluation_artifact_manifest

    run_dir = evaluation_run_directory(training_run_id, root)
    all_attempts = list_evaluation_attempts(training_run_id, root)
    dir_by_number = {s["attempt_number"]: s for s in all_attempts}

    for status in all_attempts:
        if status.get("status") not in _TERMINAL_EVAL_STATUSES:
            if not ledger_module.has_evaluation_row(evaluation_config_hash, status["attempt_number"]):
                raise EvaluationStaleAttemptError(
                    f"Evaluation of {training_run_id} attempt_{status['attempt_number']:03d} is "
                    f"nonterminal (status='{status.get('status')}') and has no ledger row -- "
                    f"refusing to start a new attempt without explicit reconciliation."
                )

    ledger_rows = _evaluation_ledger_rows_for_run(training_run_id, ledger_path)
    ledger_by_number = {int(row["evaluation_attempt"]): row for row in ledger_rows}

    for number in sorted(set(dir_by_number) | set(ledger_by_number)):
        dir_status = dir_by_number.get(number)
        ledger_row = ledger_by_number.get(number)
        if dir_status is not None and ledger_row is not None:
            if dir_status["evaluation_config_hash"] != ledger_row["evaluation_config_hash"]:
                raise EvaluationLedgerConflictError(
                    f"{training_run_id} attempt_{number:03d}: directory evaluation_config_hash "
                    f"{dir_status['evaluation_config_hash']} does not match ledger row's "
                    f"{ledger_row['evaluation_config_hash']}."
                )
        elif dir_status is None and ledger_row is not None:
            if ledger_row["status"] not in ("aborted", "failed"):
                raise EvaluationLedgerConflictError(
                    f"{training_run_id} attempt_{number:03d}: ledger records status="
                    f"{ledger_row['status']!r} but no attempt directory exists -- only aborted/"
                    f"failed attempts may have a deleted directory."
                )
        elif dir_status is not None and ledger_row is None:
            if dir_status.get("status") in _TERMINAL_EVAL_STATUSES:
                raise EvaluationLedgerConflictError(
                    f"{training_run_id} attempt_{number:03d}: directory has terminal status "
                    f"{dir_status.get('status')!r} but no ledger row exists -- requires explicit "
                    f"reconciliation, not a silent retry."
                )

    matching_completed = []
    conflicting_completed = []
    for status in all_attempts:
        if status.get("status") != EvaluationRunStatus.COMPLETED.value:
            continue
        if status["evaluation_config_hash"] != evaluation_config_hash:
            # NOT "irrelevant" -- an existing completed evaluation under a
            # DIFFERENT hash (evaluator fingerprint, frozen config, or
            # checkpoint identity changed) is a conflicting canonical
            # result, checked below. Never silently treated as unrelated.
            conflicting_completed.append(status)
            continue
        attempt_dir = run_dir / f"attempt_{status['attempt_number']:03d}"
        manifest_path = attempt_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise EvaluationPersistenceError(
                f"Completed evaluation attempt_{status['attempt_number']:03d} for "
                f"{training_run_id} is missing artifact_manifest.json."
            )
        manifest = json.loads(manifest_path.read_text())
        verify_evaluation_artifact_manifest(attempt_dir, manifest)
        matching_completed.append(status)

    if len(matching_completed) > 1:
        attempt_numbers = sorted(s["attempt_number"] for s in matching_completed)
        raise AmbiguousEvaluationCompletionError(
            f"Multiple completed, artifact-verified evaluation attempts for {training_run_id} match "
            f"evaluation_config_hash={evaluation_config_hash}: attempts {attempt_numbers}. Refusing "
            f"to silently choose earliest/latest -- resolve via explicit reconciliation before "
            f"proceeding."
        )
    if matching_completed:
        return matching_completed[0]

    if conflicting_completed:
        attempt_numbers = sorted(s["attempt_number"] for s in conflicting_completed)
        conflicting_hashes = sorted({s["evaluation_config_hash"] for s in conflicting_completed})
        raise ConflictingEvaluationImplementationError(
            f"{training_run_id} already has a COMPLETED evaluation (attempt(s) {attempt_numbers}, "
            f"evaluation_config_hash(es) {conflicting_hashes}) that does NOT match the current "
            f"request's evaluation_config_hash={evaluation_config_hash}. This means the evaluator-"
            f"implementation fingerprint, the frozen evaluation configuration, or the resolved "
            f"checkpoint identity changed underneath an existing canonical result. Hard failure -- "
            f"will not silently create a second, different-identity canonical completion for the "
            f"same training run. Requires explicit reconciliation before proceeding."
        )
    return None


# ---------------------------------------------------------------------------
# Plan mode (side-effect-free)
# ---------------------------------------------------------------------------


def plan_validation_evaluation(matrix_path: str = "configs/experiment_matrix.yaml") -> dict[str, Any]:
    """SIDE-EFFECT-FREE. Never initializes MPS, opens a dataset/checkpoint,
    creates a file/directory, writes a ledger, or computes a prediction --
    only reads matrix/ledger/status metadata and the (small, local, git-
    tracked) TTA-seed config file already on disk. Returns
    {"tta_seed_config": {...} or {"error": ...}, "cells": [...]} -- the
    frozen seed/config hash/freeze commit are reported once at the top
    level (loading configs/validation_evaluation.yaml is itself a pure
    local file+git-metadata read, never MPS/dataset/checkpoint access),
    and each of the 39 eligible training cells lists its canonical
    attempt (if any), checkpoint hash (if canonical), and the required
    analyses/prefix sequence. A missing/invalid seed config is reported
    as an error field, never raised -- plan mode never raises.
    """
    try:
        seed_cfg = load_frozen_tta_seed_config()
        tta_seed_report: dict[str, Any] = {
            "confirmatory_tta_seed": seed_cfg.confirmatory_tta_seed,
            "config_file_sha256": seed_cfg.config_file_sha256,
            "freeze_commit": seed_cfg.freeze_commit,
            "config_path": seed_cfg.config_path,
        }
    except FrozenTTASeedConfigError as e:
        tta_seed_report = {"error": str(e)}

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
    return {"tta_seed_config": tta_seed_report, "cells": rows}


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
# Inference-latency measurement (Phase 2B.4D-Engineering) -- reuses
# evaluation/latency.py's measure_clean_latency()/measure_tta_latency()
# UNCHANGED. Does not call evaluation/latency.py::build_latency_report()
# directly: that convenience wrapper requires every registered N's view
# list to be materialized simultaneously (~193 view-batches total), which
# is materially larger than necessary. This function calls the SAME
# frozen primitives in a loop that generates and discards each N's views
# immediately after timing, bounding peak memory to the single largest
# registered N -- see
# docs/phase2b_validation_evaluation_engineering_freeze.md sec.1.2/1.3.
# ---------------------------------------------------------------------------


def compute_evaluation_latency_report(
    model: torch.nn.Module,
    split,
    tta_seed: int,
    device: torch.device,
) -> LatencyReport:
    """Measure clean + per-registered-N TTA inference latency using the
    SAME restored checkpoint (`model`), validation population (`split`),
    device, and registered N values (PREFIX_SEQUENCE) as the scientific
    evaluation already computed in this attempt. Never accepts labels or
    any prediction/metric value -- structurally cannot influence or be
    influenced by scientific results. Never mutates or reads the
    already-computed scientific probability bank; regenerates its own
    deterministic view tensors via evaluation/views.py and discards them
    immediately after timing."""
    model.eval()
    x = split.images.to(device)
    n_samples = x.shape[0]
    clean_latency = measure_clean_latency(model, x, device)

    policy = build_policy(POLICY_IDENTIFIER, output_size=(split.resolution, split.resolution))
    sample_indices = split.sample_indices.tolist()

    tta_latency_by_n: dict[int, float] = {}
    per_sample_by_n: dict[int, float] = {}
    multiplier_by_n: dict[int, float] = {}
    for n in PREFIX_SEQUENCE:
        views = [
            view_batch.to(device)
            for _view_index, view_batch in iter_deterministic_views(
                split.images, policy, tta_seed, split.dataset, split.resolution, sample_indices, n
            )
        ]
        total = measure_tta_latency(model, views, device)
        del views
        tta_latency_by_n[n] = total
        per_sample_by_n[n] = total / n_samples if n_samples > 0 else 0.0
        multiplier_by_n[n] = (total / clean_latency) if clean_latency > 0 else float("inf")

    return LatencyReport(
        clean_latency_seconds=clean_latency,
        tta_latency_seconds_by_n=tta_latency_by_n,
        per_sample_latency_seconds_by_n=per_sample_by_n,
        compute_multiplier_by_n=multiplier_by_n,
        n_samples=n_samples,
    )


def _latency_report_to_dict(report: LatencyReport) -> dict[str, Any]:
    """JSON-safe serialization of a LatencyReport for metrics["latency"]
    -- JSON object keys must be strings, so registered N values become
    string keys under "by_n" (see
    docs/phase2b_validation_evaluation_engineering_freeze.md sec.1.6)."""
    return {
        "clean_latency_seconds": report.clean_latency_seconds,
        "n_samples": report.n_samples,
        "by_n": {
            str(n): {
                "tta_latency_seconds": report.tta_latency_seconds_by_n[n],
                "per_sample_latency_seconds": report.per_sample_latency_seconds_by_n[n],
                "compute_multiplier": report.compute_multiplier_by_n[n],
            }
            for n in report.tta_latency_seconds_by_n
        },
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_validation_evaluation(
    run_id: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    device_resolver=None,
    root: str | Path = DEFAULT_EVALUATION_ROOT,
    training_root: str | Path = "artifacts/confirmatory",
    data_root: str | Path = "data/raw",
    evaluation_ledger_path=None,
    tta_seed_config_path: str | Path = DEFAULT_TTA_SEED_CONFIG_PATH,
    tta_seed_git_tracked_and_clean=_default_git_tracked_and_clean,
    tta_seed_last_commit_for_path=_default_last_commit_for_path,
    tta_seed_commit_is_ancestor=_default_commit_is_ancestor,
    require_clean_tree: bool = True,
    evaluator_fingerprint_repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Single entry point for validation-only TTA evaluation. Enforces, IN
    ORDER: (1) load and exhaustively verify the frozen confirmatory TTA-
    seed configuration (configs/validation_evaluation.yaml -- see
    load_frozen_tta_seed_config()); (2) resolve the canonical training
    completion (rejects every ineligible attempt type -- see
    resolve_canonical_training_completion()); (3) compute the evaluator-
    implementation fingerprint (compute_evaluator_fingerprint() -- content
    hash of EVALUATOR_FINGERPRINT_MANIFEST, NOT the current git HEAD) and
    derive the deterministic, STABLE evaluation config/ID from it (now
    including the seed config's SHA-256/freeze commit/derivation digest --
    see docs/phase2b_validation_evaluation_engineering_freeze.md sec.2);
    (4) idempotent skip check (metadata-only, ledger/directory-consistency-
    checked, ambiguity-checked -- before MPS/dataset/checkpoint/view-
    generation/metric-calculation); (5) verify permitted working-tree
    state (require_clean_working_tree() -- dirty source/scripts/tests/
    configs/docs/dependency files fail HERE, before any attempt is
    allocated; only strict byte-prefix appends to the approved ledgers,
    including this evaluation ledger, are permitted); (6) allocate the
    attempt number and create the attempt directory/status.json; (7) MPS
    device resolution (no CPU fallback -- device_resolver defaults to
    select_device('mps') -- now INSIDE the try block, so a device failure
    is recorded as a terminal "failed" attempt with a ledger row, never
    left as a dangling unledgered "running" status); (8) load + verify the
    canonical checkpoint read-only; (9) load the validation split ONLY;
    (10) compute clean + per-view probabilities and every frozen
    condition/aggregator/prefix metric, THEN measure clean/TTA inference
    latency (compute_evaluation_latency_report() -- descriptive only,
    never influences (10)'s already-computed results); (11) persist +
    independently verify artifacts, including the required latency
    section (a malformed/incomplete/non-finite latency report fails this
    step, never producing status="completed"); (12) append the terminal
    evaluation-ledger row.

    The `tta_seed_config_*` injectable parameters exist ONLY for synthetic
    tests, mirroring authorize_block_d_cell()'s identical pattern -- there
    is no CLI flag or environment variable that overrides them.
    `require_clean_tree=False` exists only for injected tests that don't
    care about repo state (mirrors run_canary_cell()'s identical
    parameter); the production CLI never passes it.
    `evaluator_fingerprint_repo_root` defaults to "." (the real repo root
    in production) and exists ONLY so synthetic tests can chdir into an
    isolated temporary git repo (for require_clean_working_tree() testing)
    while still pointing evaluator-fingerprint file reads at the real,
    unchanged evaluation source tree -- the production CLI never passes
    it.

    A failed seed-config/resolution/skip/clean-tree step (1)-(5) creates
    ZERO files and ZERO ledger rows.
    """
    from when_tta_hurts import ledger as ledger_module
    from when_tta_hurts.devices import select_device
    from when_tta_hurts.orchestrator import require_clean_working_tree

    if device_resolver is None:
        device_resolver = lambda: select_device("mps")  # noqa: E731
    if evaluation_ledger_path is None:
        evaluation_ledger_path = ledger_module.VALIDATION_EVALUATION_LEDGER_PATH

    seed_cfg = load_frozen_tta_seed_config(
        tta_seed_config_path,
        git_tracked_and_clean=tta_seed_git_tracked_and_clean,
        last_commit_for_path=tta_seed_last_commit_for_path,
        commit_is_ancestor=tta_seed_commit_is_ancestor,
    )
    resolved_tta_seed = seed_cfg.confirmatory_tta_seed

    cell, training_result = resolve_canonical_training_completion(run_id, matrix_path)

    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    evaluator_fingerprint, evaluator_fingerprint_manifest = compute_evaluator_fingerprint(
        repo_root=evaluator_fingerprint_repo_root
    )
    cfg = build_validation_evaluation_config(
        cell, training_result, seed_cfg, expanded.source_config_hash, evaluator_fingerprint
    )
    evaluation_id = compute_evaluation_id(cfg)
    source_commit = _git_commit_hash()  # execution provenance only -- NOT hashed into evaluation_id

    skip = check_evaluation_skip(run_id, evaluation_id, root, evaluation_ledger_path)
    if skip is not None:
        # NOTE: "status" is placed AFTER the **skip spread so the constant
        # "skipped_completed" marker always wins -- `skip` is the underlying
        # attempt's own status dict (status.json content), whose "status"
        # key holds "completed", not "skipped_completed". Putting the
        # literal key first would let dict-literal evaluation order
        # silently overwrite it with "completed" (audit finding, Phase
        # 2B.4D-Engineering).
        return {
            "training_run_id": run_id,
            "evaluation_id": evaluation_id,
            **skip,
            "status": "skipped_completed",
        }

    if require_clean_tree:
        require_clean_working_tree()

    attempt_dir, status = start_evaluation_attempt(run_id, evaluation_id, root, evaluation_ledger_path)
    try:
        device = device_resolver()
        model = load_and_verify_canonical_checkpoint(cell, training_result, training_root)
        split = load_validation_evaluation_split(cell.dataset, cell.resolution, data_root)
        outcome = compute_validation_evaluation(model, split, resolved_tta_seed, device)
        latency_report = compute_evaluation_latency_report(model, split, resolved_tta_seed, device)

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
            "tta_seed_config_sha256": seed_cfg.config_file_sha256,
            "tta_seed_freeze_commit": seed_cfg.freeze_commit,
            "tta_seed_derivation_sha256": seed_cfg.derivation_sha256,
            "prefix_sequence": list(PREFIX_SEQUENCE),
            "aggregators": list(AGGREGATORS),
            "secondary_analyses": list(SECONDARY_ANALYSES),
            "protocol_commit": cfg.protocol_commit,
            "matrix_hash": cfg.matrix_hash,
            "source_commit": source_commit,
            "evaluator_fingerprint": evaluator_fingerprint,
            "evaluator_fingerprint_manifest": evaluator_fingerprint_manifest,
            "evaluation_config_hash": evaluation_id,
            "split": "validation",
            "n_validation_samples": len(split.sample_indices),
        }
        view_manifest = {
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "tta_seed": resolved_tta_seed,
            "tta_seed_config_sha256": seed_cfg.config_file_sha256,
            "tta_seed_freeze_commit": seed_cfg.freeze_commit,
            "tta_seed_derivation_sha256": seed_cfg.derivation_sha256,
            "n_views": MAX_VIEWS,
            "seed_formula": (
                "sha256(tta_seed|dataset|resolution|sample_index|view_index)[:8 bytes] % (2**31-1)"
            ),
            "sample_indices": split.sample_indices.tolist(),
            "seed_manifest_sha256": seed_manifest_hash,
        }
        metrics = {
            "training_run_id": run_id,
            "evaluation_config_hash": evaluation_id,
            **outcome["metrics"],
            "latency": _latency_report_to_dict(latency_report),
        }

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
            prefix_sequence=PREFIX_SEQUENCE,
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
