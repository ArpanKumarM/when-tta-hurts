"""Regression tests for the completed-attempt selection fix (Phase 2B.3A
attempt_004 correction): check_confirmatory_skip() must consider EVERY
attempt for a run ID, not just the first, and must never invoke MPS/
dataset/loader/model factories during a skip decision.

Synthetic temporary artifacts ONLY -- never touches the real
attempt_001-004 of A-pathmnist-28px-batchnorm-policy-none-s0."""

from __future__ import annotations

import pytest
from torch import nn

from when_tta_hurts import ledger as ledger_module
from when_tta_hurts.artifacts import atomic_write_json, save_checkpoint
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.orchestrator import (
    AmbiguousCanonicalCompletionError,
    check_confirmatory_skip,
)
from when_tta_hurts.result_artifacts import (
    PersistenceVerificationError,
    build_artifact_manifest,
)
from when_tta_hurts.run_identity import (
    ConflictingCompletedRunError,
    RunStatus,
    finish_attempt,
    run_directory,
    start_attempt,
)

CELL = MatrixCell(
    block="A_core_normalization_resolution",
    dataset="synthetic",
    resolution=4,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)


def _tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 5))


def _write_full_completed_attempt(root, cell, config_hash_override=None, corrupt_manifest=False):
    """Creates one fully-valid (or optionally corrupted) completed attempt
    directory with all 6 required artifacts, via the real start_attempt/
    finish_attempt state machine plus manually-built result artifacts
    (mirroring what persist_and_verify_completion produces)."""
    attempt_dir, status = start_attempt(cell, root, allow_new_attempt_despite_matching_hash=True)
    model = _tiny_model()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(state_dict, attempt_dir / "best_checkpoint.pt")
    atomic_write_json([{"epoch": 1}], attempt_dir / "training_history.json")
    atomic_write_json({"result": True}, attempt_dir / "result.json")
    atomic_write_json({"metadata": True}, attempt_dir / "metadata.json")
    manifest = build_artifact_manifest(attempt_dir)
    if corrupt_manifest:
        manifest["artifacts"][0]["sha256"] = "0" * 64  # deliberately wrong
    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
    if config_hash_override is not None:
        status.config_hash = config_hash_override
    finish_attempt(attempt_dir, status, RunStatus.COMPLETED)
    return status.attempt_number


def _write_failed_attempt(root, cell):
    attempt_dir, status = start_attempt(cell, root, allow_new_attempt_despite_matching_hash=True)
    finish_attempt(attempt_dir, status, RunStatus.FAILED, failure_reason="simulated failure")
    return status.attempt_number


def _mark_ineligible(amendments_path, run_id, attempt_id):
    ledger_module.append_amendment_entry(
        run_id=run_id,
        attempt_id=attempt_id,
        original_status="completed",
        canonical_eligible=False,
        amendment_type="test_amendment",
        reason="synthetic test",
        validation_metrics_computed=True,
        validation_metrics_persisted=True,
        validation_metrics_inspected=False,
        test_metrics_observed=False,
        tta_metrics_observed=False,
        source_commit="deadbeef",
        checkpoint_hash="irrelevant",
        recorded_at="2026-01-01T00:00:00Z",
        ledger_path=amendments_path,
    )


def test_earlier_ineligible_later_eligible_selects_later(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    n1 = _write_full_completed_attempt(root, CELL)
    _mark_ineligible(amendments_path, CELL.run_id(), n1)
    n2 = _write_full_completed_attempt(root, CELL)

    skip = check_confirmatory_skip(CELL, root, amendments_path)
    assert skip is not None
    assert skip.attempt_number == n2 == 2


def test_real_state_sequence_ineligible_failed_eligible_ineligible(tmp_path):
    """Mirrors the real 1/2/3/4 state: ineligible, failed, eligible, ineligible."""
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    n1 = _write_full_completed_attempt(root, CELL)
    _mark_ineligible(amendments_path, CELL.run_id(), n1)
    _write_failed_attempt(root, CELL)
    n3 = _write_full_completed_attempt(root, CELL)
    n4 = _write_full_completed_attempt(root, CELL)
    _mark_ineligible(amendments_path, CELL.run_id(), n4)

    skip = check_confirmatory_skip(CELL, root, amendments_path)
    assert skip is not None
    assert skip.attempt_number == n3 == 3


def test_only_ineligible_completions_no_skip(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    n1 = _write_full_completed_attempt(root, CELL)
    _mark_ineligible(amendments_path, CELL.run_id(), n1)

    assert check_confirmatory_skip(CELL, root, amendments_path) is None


def test_failed_attempts_no_skip(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _write_failed_attempt(root, CELL)
    _write_failed_attempt(root, CELL)

    assert check_confirmatory_skip(CELL, root, amendments_path) is None


def test_one_eligible_completion_skips(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    n1 = _write_full_completed_attempt(root, CELL)

    skip = check_confirmatory_skip(CELL, root, amendments_path)
    assert skip is not None
    assert skip.attempt_number == n1


def test_multiple_eligible_completions_hard_fail(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _write_full_completed_attempt(root, CELL)
    _write_full_completed_attempt(root, CELL)

    with pytest.raises(AmbiguousCanonicalCompletionError):
        check_confirmatory_skip(CELL, root, amendments_path)


def test_conflicting_config_hash_hard_fail(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _write_full_completed_attempt(root, CELL, config_hash_override="deliberately-wrong-hash")

    with pytest.raises(ConflictingCompletedRunError):
        check_confirmatory_skip(CELL, root, amendments_path)


def test_missing_manifest_hard_fail(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    n1 = _write_full_completed_attempt(root, CELL)
    manifest_path = run_directory(CELL, root) / f"attempt_{n1:03d}" / "artifact_manifest.json"
    manifest_path.unlink()

    with pytest.raises(PersistenceVerificationError, match="missing artifact_manifest"):
        check_confirmatory_skip(CELL, root, amendments_path)


def test_corrupt_artifact_hard_fail(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _write_full_completed_attempt(root, CELL, corrupt_manifest=True)

    with pytest.raises(PersistenceVerificationError):
        check_confirmatory_skip(CELL, root, amendments_path)


def test_numeric_ordering_attempt_2_vs_attempt_10(tmp_path):
    """attempt_2 must sort before attempt_10 numerically, not lexically."""
    from when_tta_hurts.run_identity import list_attempts

    root = tmp_path / "confirmatory"
    for _ in range(11):
        _write_failed_attempt(root, CELL)  # attempts 1..11, all failed
    statuses = list_attempts(CELL, root)
    numbers = [s["attempt_number"] for s in statuses]
    assert numbers == sorted(numbers)
    assert numbers == list(range(1, 12))


def test_no_factories_invoked_during_skip(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    n1 = _write_full_completed_attempt(root, CELL)

    # check_confirmatory_skip takes no device/loader/model-factory
    # parameters at all -- it is structurally impossible for it to invoke
    # any of them. This test documents/locks that contract.
    import inspect

    sig = inspect.signature(check_confirmatory_skip)
    assert set(sig.parameters) == {"cell", "root", "amendments_ledger_path"}

    skip = check_confirmatory_skip(CELL, root, amendments_path)
    assert skip is not None and skip.attempt_number == n1


def test_no_new_attempt_or_ledger_row_during_skip(tmp_path):
    root = tmp_path / "confirmatory"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _write_full_completed_attempt(root, CELL)

    before_dirs = sorted(p.name for p in run_directory(CELL, root).iterdir())
    check_confirmatory_skip(CELL, root, amendments_path)
    after_dirs = sorted(p.name for p in run_directory(CELL, root).iterdir())
    assert before_dirs == after_dirs

    confirmatory_ledger_path = tmp_path / "ledger_confirmatory.csv"
    assert not confirmatory_ledger_path.exists()  # check_confirmatory_skip never touches it
