"""Deterministic run/attempt identity, artifact-directory layout, and the
planned/running/completed/failed/aborted state machine for confirmatory
runs. Phase 2B.2 -- implementation only, not executed against real data.

Design:
- Run ID is deterministic (block, dataset, resolution, model, norm,
  training_policy, seed) -- see matrix.py::MatrixCell.run_id(). Same cell
  always produces the same run ID, unlike the Phase 2A pilot's
  uuid.uuid4()-based scheme (which is explicitly NOT reused here).
- Each run ID gets its own directory; each *attempt* at that run ID gets a
  numbered subdirectory (attempt_001, attempt_002, ...), so partial/failed
  attempts are preserved rather than overwritten.
- A run's canonical config hash is computed once (from the frozen matrix
  cell + FROZEN_TRAINING_SETTINGS) and stored in every attempt's status
  file, so a hash mismatch on a "completed" run is detectable.

IMPORTANT LIMITATION (documented, not solved by this module): SIGKILL and
power loss cannot be caught by any signal handler or exception path -- an
attempt killed that way will be left with status.json still reading
"running" forever, with no incident row ever written for it. This is NOT
something normal exception/SIGINT/SIGTERM handling (implemented at the
runner-call-site level, see scripts/run_confirmatory.py and
training.py) can fix. Reconciliation of stale "running" attempts (e.g. a
status.json whose started_at is implausibly old and whose process is no
longer alive) is a NECESSARY FUTURE OPERATIONAL STEP, not implemented in
this module -- a future runner invocation encountering a stale "running"
attempt directory should treat it as abandoned (new attempt, do not resume
into it) rather than silently trusting the stale state.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import atomic_write_json
from when_tta_hurts.config import config_hash
from when_tta_hurts.matrix import FROZEN_TRAINING_SETTINGS, MatrixCell

DEFAULT_CONFIRMATORY_ROOT = Path("artifacts/confirmatory")


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class RunIdentityError(RuntimeError):
    """Base class for hard failures in run/attempt identity handling."""


class ConflictingCompletedRunError(RunIdentityError):
    """Raised when a run ID already has a COMPLETED attempt whose config
    hash does not match the new attempt's config hash -- this indicates
    the frozen protocol changed underneath an existing result, which is
    never allowed to silently overwrite."""


class PilotArtifactRejectedError(RunIdentityError):
    """Raised when a confirmatory run attempts to reuse anything from
    artifacts/pilots/ as a starting point."""


def cell_config_hash(cell: MatrixCell) -> str:
    """Canonical config hash for one matrix cell, combining the cell's own
    fields with the frozen (shared) training settings -- so a change to
    either changes the hash."""
    payload = {
        "cell": asdict(cell),
        "training_settings": asdict(FROZEN_TRAINING_SETTINGS),
    }
    return config_hash(payload)


def run_directory(cell: MatrixCell, root: str | Path = DEFAULT_CONFIRMATORY_ROOT) -> Path:
    block_short = cell.block.split("_")[0]
    return Path(root) / block_short / cell.run_id()


def reject_pilot_artifact(path: str | Path) -> None:
    """Hard-fail if `path` is inside artifacts/pilots/ -- confirmatory runs
    must never reuse pilot checkpoints, predictions, or artifacts."""
    resolved = Path(path).resolve()
    if "pilots" in resolved.parts and "artifacts" in resolved.parts:
        raise PilotArtifactRejectedError(
            f"Refusing to use pilot artifact as confirmatory input: {path}. "
            f"Phase 2A pilot artifacts (seed 314159) are permanently excluded "
            f"from confirmatory results."
        )


@dataclass
class AttemptStatus:
    run_id: str
    attempt_number: int
    status: str  # RunStatus value
    config_hash: str
    block: str
    dataset: str
    resolution: int
    model: str
    normalization: str
    seed: int
    started_at: float | None = None
    ended_at: float | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    with status_path.open() as f:
        return json.load(f)


def find_completed_attempt(cell: MatrixCell, root: str | Path = DEFAULT_CONFIRMATORY_ROOT) -> dict | None:
    """Return the first (canonical) COMPLETED attempt's status dict for this
    cell, if one exists, else None. 'First valid completed attempt becomes
    canonical' -- attempt_001 before attempt_002, etc."""
    run_dir = run_directory(cell, root)
    for attempt_dir in _list_attempt_dirs(run_dir):
        status = _read_status(attempt_dir)
        if status and status.get("status") == RunStatus.COMPLETED.value:
            return status
    return None


def next_attempt_number(cell: MatrixCell, root: str | Path = DEFAULT_CONFIRMATORY_ROOT) -> int:
    run_dir = run_directory(cell, root)
    existing = _list_attempt_dirs(run_dir)
    if not existing:
        return 1
    numbers = [int(p.name.split("_")[1]) for p in existing]
    return max(numbers) + 1


def start_attempt(
    cell: MatrixCell, root: str | Path = DEFAULT_CONFIRMATORY_ROOT
) -> tuple[Path, AttemptStatus]:
    """Begin a new attempt for `cell`. Raises:
    - ConflictingCompletedRunError if a completed attempt exists with a
      DIFFERENT config hash than this cell's current hash (protocol drift).
    Returns (attempt_dir, status) with status.status == RUNNING and a
    status.json already written (atomically) marking it as such -- caller
    is responsible for writing COMPLETED/FAILED/ABORTED when done.

    If a matching-hash completed attempt already exists, the caller should
    check `find_completed_attempt` FIRST and skip -- this function does not
    perform that skip itself, to keep "check" and "start" separable for
    callers/tests.
    """
    this_hash = cell_config_hash(cell)
    existing_completed = find_completed_attempt(cell, root)
    if existing_completed is not None:
        if existing_completed["config_hash"] != this_hash:
            raise ConflictingCompletedRunError(
                f"Run {cell.run_id()} already has a COMPLETED attempt with config_hash="
                f"{existing_completed['config_hash']}, but the current cell's hash is "
                f"{this_hash}. This indicates the frozen protocol changed underneath an "
                f"existing result. Hard failure -- will not silently overwrite."
            )
        # Matching hash: caller should have skipped via find_completed_attempt;
        # we still refuse to start a redundant attempt here as a second guard.
        raise RunIdentityError(
            f"Run {cell.run_id()} already has a matching-hash COMPLETED attempt -- "
            f"use find_completed_attempt() to skip safely instead of starting a new attempt."
        )

    attempt_number = next_attempt_number(cell, root)
    run_dir = run_directory(cell, root)
    attempt_dir = run_dir / f"attempt_{attempt_number:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an existing attempt dir

    status = AttemptStatus(
        run_id=cell.run_id(),
        attempt_number=attempt_number,
        status=RunStatus.RUNNING.value,
        config_hash=this_hash,
        block=cell.block,
        dataset=cell.dataset,
        resolution=cell.resolution,
        model=cell.model,
        normalization=cell.normalization,
        seed=cell.seed,
        started_at=time.time(),
    )
    atomic_write_json(status.to_dict(), attempt_dir / "status.json")
    return attempt_dir, status


def finish_attempt(
    attempt_dir: Path, status: AttemptStatus, final_status: RunStatus, failure_reason: str | None = None
) -> AttemptStatus:
    """Atomically transition an attempt to COMPLETED/FAILED/ABORTED.
    Never called for RUNNING/PLANNED -- those are terminal-only.
    """
    if final_status not in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.ABORTED):
        raise ValueError(f"finish_attempt() requires a terminal status, got {final_status}")
    status.status = final_status.value
    status.ended_at = time.time()
    status.failure_reason = failure_reason
    atomic_write_json(status.to_dict(), attempt_dir / "status.json")
    return status
