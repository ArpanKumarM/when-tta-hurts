"""Phase 2B.6D: synthetic regression tests reproducing and fixing the
final-test stale-attempt/ledger-column defect discovered in
docs/phase2b_final_test_accidental_access_incident.md
(check_evaluation_skip()'s internal has_evaluation_row() call reads CSV
column "evaluation_id", which does not exist in
FINAL_TEST_LEDGER_FIELDNAMES, and never receives the caller's
ledger_path). Covers the corrected check_final_test_evaluation_skip().

All fixtures are tmp_path-rooted; none touch the real final-test ledger,
authorization artifact, or attempt directory.
"""

from __future__ import annotations

import json

import pytest

from when_tta_hurts.final_test_evaluation import check_final_test_evaluation_skip
from when_tta_hurts.final_test_result_artifacts import build_final_test_artifact_manifest
from when_tta_hurts.ledger import (
    FINAL_TEST_LEDGER_FIELDNAMES,
    append_final_test_entry,
    ensure_final_test_ledger_exists,
)
from when_tta_hurts.validation_evaluation import (
    AmbiguousEvaluationCompletionError,
    ConflictingEvaluationImplementationError,
    EvaluationLedgerConflictError,
    EvaluationStaleAttemptError,
    next_evaluation_attempt_number,
)


def _write_status(root, run_id, attempt, status, evaluation_config_hash):
    d = root / run_id / f"attempt_{attempt:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "status.json").write_text(
        json.dumps(
            {
                "training_run_id": run_id,
                "attempt_number": attempt,
                "status": status,
                "evaluation_config_hash": evaluation_config_hash,
                "started_at": 1.0,
                "ended_at": None,
                "failure_reason": None,
            }
        )
    )
    return d


def _append_row(ledger_path, **overrides):
    base = dict(
        final_test_evaluation_id="hash-a",
        training_run_id="run-a",
        training_attempt=1,
        checkpoint_hash="chk",
        evaluation_config_hash="hash-a",
        evaluation_attempt=1,
        evaluator_fingerprint="e",
        statistical_analysis_fingerprint="s",
        cross_condition_analysis_fingerprint="c",
        final_test_runner_fingerprint="r",
        authorization_artifact_sha256="a",
        authorization_commit="ac",
        test_split_accessed=True,
        test_predictions_computed="",
        test_metrics_computed="",
        test_metrics_persisted=False,
        test_metrics_observed="",
        status="aborted",
        primary_artifact_hash="",
        started_at=1.0,
        ended_at="",
        runtime_seconds="",
        failure_stage="unknown_externally_terminated",
        failure_reason="synthetic test row",
    )
    base.update(overrides)
    return append_final_test_entry(ledger_path=ledger_path, **base)


def _write_completed_attempt(root, run_id, attempt, evaluation_config_hash):
    d = _write_status(root, run_id, attempt, "completed", evaluation_config_hash)
    (d / "predictions.npz").write_bytes(b"\x00")
    manifest = build_final_test_artifact_manifest(d, filenames=("predictions.npz",))
    (d / "artifact_manifest.json").write_text(json.dumps(manifest))
    return d


def test_ledger_schema_column_mismatch_confirmed():
    """Direct schema-level confirmation of the defect's root cause: the
    final-test ledger has no 'evaluation_id' column (the OLD, generic
    has_evaluation_row() reads exactly that column), so any code path
    relying on it can never find a match against this ledger."""
    assert "evaluation_id" not in FINAL_TEST_LEDGER_FIELDNAMES
    assert "final_test_evaluation_id" in FINAL_TEST_LEDGER_FIELDNAMES
    assert "evaluation_config_hash" in FINAL_TEST_LEDGER_FIELDNAMES


def test_exact_incident_sequence_reconciles_not_stale(tmp_path):
    """Running directory + matching terminal (aborted) ledger row for the
    SAME attempt number -- exactly the real incident's shape -- must
    reconcile (return None, proceed to allocate a new attempt), never
    raise, and never be selected as completed."""
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 1, "running", "hash-a")
    _append_row(ledger_path, status="aborted")

    result = check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)
    assert result is None


def test_unledgered_nonterminal_directory_hard_fails(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 1, "running", "hash-a")

    with pytest.raises(EvaluationStaleAttemptError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_mismatched_hash_between_directory_and_ledger_row_hard_fails(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 1, "running", "hash-a")
    _append_row(
        ledger_path, evaluation_config_hash="different-hash", final_test_evaluation_id="different-hash"
    )

    with pytest.raises(EvaluationLedgerConflictError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_mismatched_attempt_number_hard_fails(tmp_path):
    """A ledger row exists but for a DIFFERENT attempt number than the
    directory -- the directory's own attempt number remains unledgered."""
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 2, "running", "hash-a")
    _append_row(ledger_path, evaluation_attempt=1)

    with pytest.raises(EvaluationStaleAttemptError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_mismatched_evaluation_id_within_same_attempt_hard_fails(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 1, "running", "hash-a")
    _append_row(ledger_path, final_test_evaluation_id="other-hash", evaluation_config_hash="other-hash")

    with pytest.raises(EvaluationLedgerConflictError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_conflicting_terminal_ledger_row_with_no_directory_hard_fails(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _append_row(ledger_path, status="completed")  # no directory at all

    with pytest.raises(EvaluationLedgerConflictError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_directory_terminal_with_no_ledger_row_hard_fails(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 1, "failed", "hash-a")

    with pytest.raises(EvaluationLedgerConflictError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_completed_attempt_still_skips(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_completed_attempt(root, "run-a", 1, "hash-a")
    _append_row(ledger_path, status="completed")

    result = check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)
    assert result is not None
    assert result["attempt_number"] == 1


def test_attempt_number_union_across_directory_and_ledger_resolves_to_2(tmp_path):
    """The exact real-incident resolution this fix exists to restore:
    reconciled aborted attempt 1 -> next attempt allocated is 2."""
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_status(root, "run-a", 1, "aborted", "hash-a")
    _append_row(ledger_path, status="aborted")

    assert next_evaluation_attempt_number("run-a", root, ledger_path) == 2


def test_ambiguous_multiple_completed_matches_raises(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    for attempt in (1, 2):
        _write_completed_attempt(root, "run-a", attempt, "hash-a")
        _append_row(ledger_path, evaluation_attempt=attempt, status="completed")

    with pytest.raises(AmbiguousEvaluationCompletionError):
        check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path)


def test_conflicting_completed_under_different_hash_raises(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    _write_completed_attempt(root, "run-a", 1, "old-hash")
    _append_row(
        ledger_path,
        evaluation_config_hash="old-hash",
        final_test_evaluation_id="old-hash",
        status="completed",
    )

    with pytest.raises(ConflictingEvaluationImplementationError):
        check_final_test_evaluation_skip("run-a", "new-hash", root, ledger_path)


def test_no_ledger_no_directory_returns_none(tmp_path):
    root = tmp_path / "final_test"
    ledger_path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(ledger_path)
    assert check_final_test_evaluation_skip("run-a", "hash-a", root, ledger_path) is None


def test_run_final_test_evaluation_uses_the_corrected_skip_function(monkeypatch, tmp_path):
    """Structural: run_final_test_evaluation() must call
    check_final_test_evaluation_skip -- not the generic, defective
    validation_evaluation.check_evaluation_skip -- for its final-test
    idempotent-skip step. Self-contained fake identity chain (mirrors
    test_final_test_evaluation.py's _patch_common pattern) rather than
    importing across test modules."""
    import when_tta_hurts.final_test_evaluation as fte

    class _FakeCell:
        dataset = "pathmnist"
        resolution = 28
        model = "small_cnn"
        normalization = "batchnorm"
        training_policy = "none"
        seed = 0

        def run_id(self):
            return "fake-run-a"

    class _FakeTrainingResult:
        attempt_number = 1
        checkpoint_hash = "chk-a"

    class _FakeExpanded:
        def __init__(self, cells):
            self.cells = cells
            self.source_config_hash = "fake-matrix-hash"

    class _FakeReceipt:
        dataset = "pathmnist"
        resolution = 28

    class _FakeAuthorization:
        authorized_cells_by_run_id = {"fake-run-a": {"training_attempt": 1, "checkpoint_hash": "chk-a"}}
        artifact_sha256 = "fake-artifact-sha"
        authorization_commit = "fake-authorization-commit"

        def receipt_for(self, run_id):
            return _FakeReceipt()

    class _FakeSeedConfig:
        confirmatory_tta_seed = 1306178015
        config_file_sha256 = "fake-seed-sha"
        freeze_commit = "fake-freeze-commit"
        derivation_sha256 = "fake-derivation-sha"
        metric_input_contract = "probability_native_v1"

    cell = _FakeCell()
    monkeypatch.setattr(fte, "verify_final_test_authorization", lambda **k: _FakeAuthorization())
    monkeypatch.setattr(
        fte, "resolve_canonical_training_completion", lambda rid, mp: (cell, _FakeTrainingResult())
    )
    monkeypatch.setattr(fte, "load_frozen_tta_seed_config", lambda path: _FakeSeedConfig())
    monkeypatch.setattr(fte, "parse_and_validate_matrix", lambda mp, **k: _FakeExpanded([cell]))
    monkeypatch.setattr(fte, "compute_evaluator_fingerprint", lambda: ("fake-evaluator-fp", {}))
    monkeypatch.setattr(fte, "compute_analysis_fingerprint", lambda: ("fake-analysis-fp", None))
    monkeypatch.setattr(fte, "compute_cross_condition_fingerprint", lambda: ("fake-cross-fp", None))
    monkeypatch.setattr(fte, "compute_final_test_runner_fingerprint", lambda: ("fake-runner-fp", None))
    monkeypatch.setattr(fte, "expected_official_checksum", lambda dataset, resolution: "0" * 32)
    monkeypatch.setattr(fte, "_git_commit_hash", lambda: "fake-source-commit")
    monkeypatch.setattr(fte, "require_clean_working_tree", lambda: None)

    called = {"new": False}

    def _spy(*a, **k):
        called["new"] = True
        return {"attempt_number": 1, "status": "completed"}

    monkeypatch.setattr(fte, "check_final_test_evaluation_skip", _spy)

    result = fte.run_final_test_evaluation(
        "fake-run-a",
        device_resolver=lambda: (_ for _ in ()).throw(AssertionError("must not reach device init")),
        root=tmp_path / "final_test",
        final_test_ledger_path=tmp_path / "ledger.csv",
    )
    assert called["new"] is True
    assert result["status"] == "skipped_completed"
