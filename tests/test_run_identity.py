"""Tests for run_identity.py -- all use temporary directories only, never
artifacts/confirmatory/ or any real ledger."""

import pytest

from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.run_identity import (
    AttemptStatus,
    ConflictingCompletedRunError,
    PilotArtifactRejectedError,
    RunIdentityError,
    RunStatus,
    cell_config_hash,
    find_completed_attempt,
    finish_attempt,
    next_attempt_number,
    reject_pilot_artifact,
    run_directory,
    start_attempt,
)

CELL_A = MatrixCell(
    block="A_core_normalization_resolution",
    dataset="pathmnist",
    resolution=28,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)


def test_run_id_deterministic_across_instances():
    other = MatrixCell(
        block="A_core_normalization_resolution",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        training_policy="none",
        seed=0,
    )
    assert CELL_A.run_id() == other.run_id()


def test_config_hash_deterministic():
    assert cell_config_hash(CELL_A) == cell_config_hash(CELL_A)


def test_config_hash_changes_with_seed():
    other = MatrixCell(**{**CELL_A.__dict__, "seed": 1})
    assert cell_config_hash(CELL_A) != cell_config_hash(other)


def test_start_attempt_creates_running_status(tmp_path):
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    assert attempt_dir.exists()
    assert status.status == RunStatus.RUNNING.value
    assert (attempt_dir / "status.json").exists()


def test_first_attempt_is_numbered_001(tmp_path):
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    assert attempt_dir.name == "attempt_001"
    assert status.attempt_number == 1


def test_finish_attempt_transitions_to_completed(tmp_path):
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    finished = finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
    assert finished.status == RunStatus.COMPLETED.value
    assert finished.ended_at is not None


def test_find_completed_attempt_returns_none_before_completion(tmp_path):
    start_attempt(CELL_A, root=tmp_path)  # left running
    assert find_completed_attempt(CELL_A, root=tmp_path) is None


def test_find_completed_attempt_returns_status_after_completion(tmp_path):
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
    found = find_completed_attempt(CELL_A, root=tmp_path)
    assert found is not None
    assert found["status"] == RunStatus.COMPLETED.value


def test_matching_completed_run_is_skippable(tmp_path):
    """Simulates the 'safe skip' behavior a caller implements: check
    find_completed_attempt, and if config_hash matches, do not start a new
    attempt."""
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
    found = find_completed_attempt(CELL_A, root=tmp_path)
    assert found["config_hash"] == cell_config_hash(CELL_A)  # caller would skip here


def test_conflicting_completed_hash_is_hard_failure(tmp_path):
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
    # Corrupt the stored hash to simulate protocol drift.
    import json

    status_path = attempt_dir / "status.json"
    data = json.loads(status_path.read_text())
    data["config_hash"] = "deliberately-different-hash"
    status_path.write_text(json.dumps(data))

    with pytest.raises(ConflictingCompletedRunError):
        start_attempt(CELL_A, root=tmp_path)


def test_matching_hash_completed_refuses_redundant_start(tmp_path):
    attempt_dir, status = start_attempt(CELL_A, root=tmp_path)
    finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
    with pytest.raises(RunIdentityError):
        start_attempt(CELL_A, root=tmp_path)


def test_partial_attempt_preserved_next_attempt_numbered(tmp_path):
    attempt_dir_1, status_1 = start_attempt(CELL_A, root=tmp_path)
    finish_attempt(attempt_dir_1, status_1, RunStatus.FAILED, failure_reason="simulated crash")
    assert attempt_dir_1.exists()  # preserved, not deleted

    assert next_attempt_number(CELL_A, root=tmp_path) == 2
    attempt_dir_2, status_2 = start_attempt(CELL_A, root=tmp_path)
    assert attempt_dir_2.name == "attempt_002"
    assert attempt_dir_1.exists()  # still preserved after second attempt starts


def test_never_overwrites_existing_attempt_dir(tmp_path):
    attempt_dir, _status = start_attempt(CELL_A, root=tmp_path)
    (attempt_dir / "marker.txt").write_text("do not delete me")
    # Directly calling mkdir on the same path again must fail (exist_ok=False).
    with pytest.raises(FileExistsError):
        attempt_dir.mkdir(parents=True, exist_ok=False)
    assert (attempt_dir / "marker.txt").read_text() == "do not delete me"


def test_reject_pilot_artifact_raises_for_pilot_path():
    with pytest.raises(PilotArtifactRejectedError):
        reject_pilot_artifact("artifacts/pilots/pilot-pathmnist-28-bn-8f4a5024/best_checkpoint.pt")


def test_reject_pilot_artifact_allows_confirmatory_path(tmp_path):
    reject_pilot_artifact(tmp_path / "confirmatory" / "A" / "run" / "attempt_001" / "best_checkpoint.pt")
    # no exception -- passes


def test_run_directory_layout():
    d = run_directory(CELL_A, root="artifacts/confirmatory")
    assert str(d) == f"artifacts/confirmatory/A/{CELL_A.run_id()}"


def test_attempt_status_serializes_to_dict():
    status = AttemptStatus(
        run_id="x",
        attempt_number=1,
        status=RunStatus.PLANNED.value,
        config_hash="abc",
        block="A_core_normalization_resolution",
        dataset="pathmnist",
        resolution=28,
        model="small_cnn",
        normalization="batchnorm",
        seed=0,
    )
    d = status.to_dict()
    assert d["run_id"] == "x"
    assert d["status"] == "planned"
