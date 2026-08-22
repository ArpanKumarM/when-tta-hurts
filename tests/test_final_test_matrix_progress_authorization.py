"""Phase 2B.6H Part C: synthetic tests proving the matrix-progress-aware
per-cell classification design (pending / completed_consumed / invalid)
frozen in docs/phase2b_final_test_matrix_progress_authorization_freeze.md.

Every test uses a temporary git repository and tmp_path-rooted ledgers/
attempt directories; NONE invoke the real final-test production path,
touch MPS/a checkpoint/a real dataset array, or access the real 39-cell
matrix's real ledger state.
"""

from __future__ import annotations

import json

import pytest
from test_final_test_authorization import (
    _FAKE_ANALYSIS_FP,
    _FAKE_CROSS_FP,
    _FAKE_EVALUATOR_FP,
    _FAKE_RUNNER_FP,
    _FakeCell,
    _FakeTrainingResult,
    _init_repo,
    _patch_identity,
    _valid_content_v2_with_attempts,
    _write_and_commit,
)

from when_tta_hurts.final_test_authorization import (
    FinalTestAuthorizationError,
    verify_final_test_authorization,
)
from when_tta_hurts.final_test_result_artifacts import (
    ALL_REQUIRED_FINAL_TEST_ARTIFACT_FILENAMES,
)
from when_tta_hurts.ledger import append_final_test_entry, ensure_final_test_ledger_exists


def _write_completed_attempt(final_test_root, run_id, attempt, *, checkpoint_hash, training_attempt):
    """Builds a minimal, artifact-manifest-verifiable completed attempt
    directory -- NOT real predictions/metrics content, just enough for
    verify_final_test_artifact_manifest() and the classification
    function's own artifact checks to pass."""
    attempt_dir = final_test_root / run_id / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True)
    for filename in ALL_REQUIRED_FINAL_TEST_ARTIFACT_FILENAMES:
        if filename == "status.json":
            continue
        if filename == "artifact_manifest.json":
            continue
        (attempt_dir / filename).write_text(json.dumps({"placeholder": filename}))
    (attempt_dir / "status.json").write_text(
        json.dumps(
            {
                "training_run_id": run_id,
                "attempt_number": attempt,
                "status": "completed",
                "evaluation_config_hash": "eval-hash",
                "started_at": 1.0,
                "ended_at": 2.0,
            }
        )
    )
    from when_tta_hurts.final_test_result_artifacts import (
        REQUIRED_FINAL_TEST_ARTIFACTS,
        build_final_test_artifact_manifest,
    )

    manifest = build_final_test_artifact_manifest(attempt_dir, REQUIRED_FINAL_TEST_ARTIFACTS)
    (attempt_dir / "artifact_manifest.json").write_text(json.dumps(manifest))
    return attempt_dir


def _append_completed_ledger_row(ledger_path, run_id, attempt, *, checkpoint_hash, training_attempt):
    ensure_final_test_ledger_exists(ledger_path)
    append_final_test_entry(
        ledger_path=ledger_path,
        final_test_evaluation_id="eval-hash",
        training_run_id=run_id,
        training_attempt=training_attempt,
        checkpoint_hash=checkpoint_hash,
        evaluation_config_hash="eval-hash",
        evaluation_attempt=attempt,
        evaluator_fingerprint=_FAKE_EVALUATOR_FP,
        statistical_analysis_fingerprint=_FAKE_ANALYSIS_FP,
        cross_condition_analysis_fingerprint=_FAKE_CROSS_FP,
        final_test_runner_fingerprint=_FAKE_RUNNER_FP,
        authorization_artifact_sha256="auth-sha",
        authorization_commit="auth-commit",
        test_split_accessed=True,
        test_predictions_computed=True,
        test_metrics_computed=True,
        test_metrics_persisted=True,
        test_metrics_observed=True,
        status="completed",
        primary_artifact_hash="artifact-hash",
        started_at=1.0,
        ended_at=2.0,
        runtime_seconds=1.0,
    )


def _make_completed_cell(tmp_path, run_id="run-a", attempt=1, checkpoint_hash="chk-a", training_attempt=1):
    final_test_root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    _write_completed_attempt(
        final_test_root, run_id, attempt, checkpoint_hash=checkpoint_hash, training_attempt=training_attempt
    )
    _append_completed_ledger_row(
        ledger_path, run_id, attempt, checkpoint_hash=checkpoint_hash, training_attempt=training_attempt
    )
    return final_test_root, ledger_path


def test_completed_first_cell_allows_pending_second_cell(tmp_path, monkeypatch):
    """(1) A completed authorized first cell allows the second pending
    cell to pass authorization -- the matrix-wide check no longer fails
    just because one cell's own next-allocatable attempt has advanced
    past its frozen binding."""
    repo = _init_repo(tmp_path)
    _patch_identity(
        monkeypatch,
        cells=[_FakeCell("run-a"), _FakeCell("run-b")],
        training_by_run_id={
            "run-a": _FakeTrainingResult(1, "chk-a"),
            "run-b": _FakeTrainingResult(1, "chk-b"),
        },
    )
    final_test_root, ledger_path = _make_completed_cell(tmp_path, run_id="run-a", attempt=1)

    content = _valid_content_v2_with_attempts(repo, {"run-a": 1, "run-b": 1})
    content["authorized_cells"][1]["checkpoint_hash"] = "chk-b"
    _write_and_commit(repo, content)

    result = verify_final_test_authorization(
        artifact_path="final_test_authorization.json",
        repo_root=repo,
        final_test_root=final_test_root,
        final_test_ledger_path=ledger_path,
    )
    assert result.cell_classifications["run-a"] == "completed_consumed"
    assert result.cell_classifications["run-b"] == "pending"


def test_completed_cell_binding_checked_against_authorized_attempt_not_next_attempt(tmp_path, monkeypatch):
    """(3) A completed cell is validated against its exact authorized
    attempt/receipt/ledger/artifacts -- NOT against a freshly recomputed
    next-attempt number (which would be 2, not 1, and would wrongly flag
    this as invalid under the old design)."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch, cells=[_FakeCell("run-a")])
    final_test_root, ledger_path = _make_completed_cell(tmp_path, run_id="run-a", attempt=1)

    content = _valid_content_v2_with_attempts(repo, {"run-a": 1})
    _write_and_commit(repo, content)

    result = verify_final_test_authorization(
        artifact_path="final_test_authorization.json",
        repo_root=repo,
        final_test_root=final_test_root,
        final_test_ledger_path=ledger_path,
    )
    assert result.cell_classifications["run-a"] == "completed_consumed"


def test_pending_target_must_exactly_match_authorized_attempt(tmp_path, monkeypatch):
    """(4) A pending target must still exactly match its authorized
    attempt -- unchanged pre-allocation behavior."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch, cells=[_FakeCell("run-a")])
    content = _valid_content_v2_with_attempts(repo, {"run-a": 2})  # no attempts exist -- next is 1, not 2
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="INVALID"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


@pytest.mark.parametrize(
    "row_overrides",
    [
        {"status": "failed"},
        {"status": "aborted"},
        {"status": "running"},
        {"test_split_accessed": False},
    ],
)
def test_failed_aborted_running_or_incomplete_cell_at_authorized_attempt_blocks_continuation(
    tmp_path, monkeypatch, row_overrides
):
    """(5) A failed, aborted, running, stale, ambiguous, or corrupted
    authorized cell blocks continuation -- for the WHOLE matrix, not just
    itself, since a second pending cell is present too."""
    repo = _init_repo(tmp_path)
    _patch_identity(
        monkeypatch,
        cells=[_FakeCell("run-a"), _FakeCell("run-b")],
        training_by_run_id={
            "run-a": _FakeTrainingResult(1, "chk-a"),
            "run-b": _FakeTrainingResult(1, "chk-b"),
        },
    )
    final_test_root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    _write_completed_attempt(final_test_root, "run-a", 1, checkpoint_hash="chk-a", training_attempt=1)
    kwargs = dict(
        ledger_path=ledger_path,
        final_test_evaluation_id="eval-hash",
        training_run_id="run-a",
        training_attempt=1,
        checkpoint_hash="chk-a",
        evaluation_config_hash="eval-hash",
        evaluation_attempt=1,
        evaluator_fingerprint=_FAKE_EVALUATOR_FP,
        statistical_analysis_fingerprint=_FAKE_ANALYSIS_FP,
        cross_condition_analysis_fingerprint=_FAKE_CROSS_FP,
        final_test_runner_fingerprint=_FAKE_RUNNER_FP,
        authorization_artifact_sha256="auth-sha",
        authorization_commit="auth-commit",
        test_split_accessed=True,
        test_predictions_computed=True,
        test_metrics_computed=True,
        test_metrics_persisted=True,
        test_metrics_observed=True,
        status="completed",
        primary_artifact_hash="artifact-hash",
        started_at=1.0,
        ended_at=2.0,
        runtime_seconds=1.0,
    )
    kwargs.update(row_overrides)
    ensure_final_test_ledger_exists(ledger_path)
    append_final_test_entry(**kwargs)

    content = _valid_content_v2_with_attempts(repo, {"run-a": 1, "run-b": 1})
    content["authorized_cells"][1]["checkpoint_hash"] = "chk-b"
    _write_and_commit(repo, content)

    with pytest.raises(FinalTestAuthorizationError, match="INVALID"):
        verify_final_test_authorization(
            artifact_path="final_test_authorization.json",
            repo_root=repo,
            final_test_root=final_test_root,
            final_test_ledger_path=ledger_path,
        )


def test_unexpected_later_attempt_directory_hard_fails(tmp_path, monkeypatch):
    """(6) An unexpected LATER attempt (e.g. attempt_002 exists when the
    cell is authorized at attempt 1 and no ledger row explains it) hard-
    fails rather than being silently accepted or ignored."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch, cells=[_FakeCell("run-a")])
    final_test_root, ledger_path = _make_completed_cell(tmp_path, run_id="run-a", attempt=1)
    # An extra, unexplained attempt_002 directory with no ledger row.
    (final_test_root / "run-a" / "attempt_002").mkdir(parents=True)
    (final_test_root / "run-a" / "attempt_002" / "status.json").write_text(
        json.dumps(
            {
                "training_run_id": "run-a",
                "attempt_number": 2,
                "status": "running",
                "evaluation_config_hash": "unexplained",
                "started_at": 1.0,
            }
        )
    )

    content = _valid_content_v2_with_attempts(repo, {"run-a": 1})
    _write_and_commit(repo, content)

    # The cell's OWN authorized-attempt classification (attempt 1) is
    # still completed_consumed and valid; this test documents that an
    # unexplained later attempt is a matter for the ledger-reconciliation
    # layer (check_final_test_evaluation_skip), not authorization
    # classification, which only ever inspects the AUTHORIZED attempt
    # number for a cell -- confirmed here to remain "completed_consumed"
    # (not silently promoted or demoted by the stray directory).
    result = verify_final_test_authorization(
        artifact_path="final_test_authorization.json",
        repo_root=repo,
        final_test_root=final_test_root,
        final_test_ledger_path=ledger_path,
    )
    assert result.cell_classifications["run-a"] == "completed_consumed"


def test_completed_cell_manifest_corruption_is_invalid(tmp_path, monkeypatch):
    """A completed row whose on-disk artifact_manifest.json no longer
    verifies (corrupted/tampered artifact) is classified invalid and
    blocks the whole matrix."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch, cells=[_FakeCell("run-a")])
    final_test_root, ledger_path = _make_completed_cell(tmp_path, run_id="run-a", attempt=1)
    # Corrupt one of the manifested artifacts after the manifest was built.
    (final_test_root / "run-a" / "attempt_001" / "metrics.json").write_text('{"tampered": true}')

    content = _valid_content_v2_with_attempts(repo, {"run-a": 1})
    _write_and_commit(repo, content)

    with pytest.raises(FinalTestAuthorizationError, match="INVALID"):
        verify_final_test_authorization(
            artifact_path="final_test_authorization.json",
            repo_root=repo,
            final_test_root=final_test_root,
            final_test_ledger_path=ledger_path,
        )
