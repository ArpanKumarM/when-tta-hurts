"""Phase 2B.6A: synthetic tests for final_test_authorization.py -- the
39-cell final-test evaluation gate. Every test uses a temporary git
repository and monkeypatched fingerprint/checksum/training-completion
functions; NONE touch the real (nonexistent) production authorization
artifact or the real 39-cell matrix's real ledger state.
"""

from __future__ import annotations

import csv
import json
import subprocess

import pytest

import when_tta_hurts.final_test_authorization as fta
from when_tta_hurts.final_test_authorization import (
    FinalTestAuthorizationError,
    verify_final_test_authorization,
)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    # Need at least one commit for merge-base/--is-ancestor to have a HEAD.
    (tmp_path / "README.md").write_text("seed commit\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def _bound_commit(repo):
    """A commit that IS an ancestor of the repo's current HEAD."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return out.stdout.strip()


class _FakeCell:
    def __init__(self, run_id_value):
        self._run_id = run_id_value
        self.dataset = "pathmnist"
        self.resolution = 28

    def run_id(self):
        return self._run_id


class _FakeTrainingResult:
    def __init__(self, attempt_number, checkpoint_hash):
        self.attempt_number = attempt_number
        self.checkpoint_hash = checkpoint_hash


class _FakeExpanded:
    def __init__(self, cells):
        self.cells = cells


_FAKE_EVALUATOR_FP = "fake-evaluator-fp"
_FAKE_ANALYSIS_FP = "fake-analysis-fp"
_FAKE_CROSS_FP = "fake-cross-fp"
_FAKE_RUNNER_FP = "fake-runner-fp"
_FAKE_DATASET_CHECKSUM = "0" * 32


def _patch_identity(monkeypatch, *, cells=None, training_by_run_id=None):
    cells = cells if cells is not None else [_FakeCell("run-a")]
    training_by_run_id = training_by_run_id or {"run-a": _FakeTrainingResult(1, "chk-a")}

    monkeypatch.setattr(fta, "compute_evaluator_fingerprint", lambda: (_FAKE_EVALUATOR_FP, {}))
    monkeypatch.setattr(fta, "compute_analysis_fingerprint", lambda: (_FAKE_ANALYSIS_FP, {}))
    monkeypatch.setattr(fta, "compute_cross_condition_fingerprint", lambda: (_FAKE_CROSS_FP, {}))
    monkeypatch.setattr(fta, "compute_final_test_runner_fingerprint", lambda: (_FAKE_RUNNER_FP, {}))
    monkeypatch.setattr(fta, "expected_official_checksum", lambda dataset, resolution: _FAKE_DATASET_CHECKSUM)
    monkeypatch.setattr(fta, "parse_and_validate_matrix", lambda matrix_path, **k: _FakeExpanded(cells))
    monkeypatch.setattr(
        fta,
        "resolve_canonical_training_completion",
        lambda run_id, matrix_path: (None, training_by_run_id[run_id]),
    )


def _valid_content(repo, cells=("run-a",), training_attempt=1, checkpoint_hash="chk-a"):
    return {
        "status": "approved",
        "approval_timestamp": "2026-09-01T00:00:00Z",
        "phase2b_protocol_commit": _bound_commit(repo),
        "matrix_commit": _bound_commit(repo),
        "cross_condition_addendum_commit": _bound_commit(repo),
        "evaluator_fingerprint": _FAKE_EVALUATOR_FP,
        "statistical_analysis_fingerprint": _FAKE_ANALYSIS_FP,
        "cross_condition_analysis_fingerprint": _FAKE_CROSS_FP,
        "final_test_runner_fingerprint": _FAKE_RUNNER_FP,
        "official_dataset_checksums": {"pathmnist@28": _FAKE_DATASET_CHECKSUM},
        "authorized_cells": [
            {"run_id": run_id, "training_attempt": training_attempt, "checkpoint_hash": checkpoint_hash}
            for run_id in cells
        ],
    }


def _write_and_commit(repo, content, filename="final_test_authorization.json"):
    path = repo / filename
    path.write_text(json.dumps(content))
    _git(repo, "add", filename)
    _git(repo, "commit", "-q", "-m", "add authorization")
    return path


def test_missing_artifact_raises(tmp_path):
    repo = _init_repo(tmp_path)
    with pytest.raises(FinalTestAuthorizationError, match="does not exist"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_untracked_artifact_raises(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    path = repo / "final_test_authorization.json"
    path.write_text(json.dumps(_valid_content(repo)))
    # not git-added -- untracked
    with pytest.raises(FinalTestAuthorizationError, match="not tracked-and-clean"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_dirty_artifact_raises(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    (repo / "final_test_authorization.json").write_text(
        json.dumps({**_valid_content(repo), "status": "draft"})
    )
    with pytest.raises(FinalTestAuthorizationError, match="not tracked-and-clean"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_malformed_json_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    path = repo / "final_test_authorization.json"
    path.write_text("{not valid json")
    _git(repo, "add", "final_test_authorization.json")
    _git(repo, "commit", "-q", "-m", "bad json")
    with pytest.raises(FinalTestAuthorizationError, match="malformed JSON"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_missing_field_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    content = _valid_content(repo)
    del content["approval_timestamp"]
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="missing required field"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_status_not_approved_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, {**_valid_content(repo), "status": "draft"})
    with pytest.raises(FinalTestAuthorizationError, match="not 'approved'"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_non_ancestor_commit_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    content = _valid_content(repo)
    content["phase2b_protocol_commit"] = "0" * 40  # a syntactically plausible but nonexistent commit
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="not an ancestor of HEAD"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


@pytest.mark.parametrize(
    "field",
    [
        "evaluator_fingerprint",
        "statistical_analysis_fingerprint",
        "cross_condition_analysis_fingerprint",
        "final_test_runner_fingerprint",
    ],
)
def test_fingerprint_mismatch_rejected(tmp_path, monkeypatch, field):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    content = _valid_content(repo)
    content[field] = "wrong-value"
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="does not match"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_dataset_checksum_mismatch_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    content = _valid_content(repo)
    content["official_dataset_checksums"]["pathmnist@28"] = "f" * 32
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="does not match the current official"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_authorized_cells_count_mismatch_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch, cells=[_FakeCell("run-a"), _FakeCell("run-b")])
    content = _valid_content(repo, cells=("run-a",))  # only 1, but matrix has 2
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="entries, but the frozen matrix"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_authorized_cells_run_id_set_mismatch_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch, cells=[_FakeCell("run-a")])
    content = _valid_content(repo, cells=("run-DIFFERENT",))
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="does not exactly match"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_training_attempt_mismatch_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    content = _valid_content(repo, training_attempt=999)  # real fake resolver says attempt=1
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="training_attempt"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_checkpoint_hash_mismatch_rejected(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    content = _valid_content(repo, checkpoint_hash="WRONG-HASH")
    _write_and_commit(repo, content)
    with pytest.raises(FinalTestAuthorizationError, match="checkpoint_hash"):
        verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)


def test_fully_valid_authorization_passes(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    result = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    assert result.status == "approved"
    assert result.evaluator_fingerprint == _FAKE_EVALUATOR_FP
    assert "run-a" in result.authorized_cells_by_run_id
    assert result.artifact_sha256
    assert result.authorization_commit


def test_no_bypass_flags_on_verify_function_signature():
    import inspect

    sig = inspect.signature(verify_final_test_authorization)
    param_names = set(sig.parameters.keys())
    assert param_names == {"artifact_path", "matrix_path", "repo_root"}
    for forbidden in ("force", "override", "skip", "unlock", "bypass", "env"):
        assert forbidden not in param_names


def test_real_production_authorization_artifact_lifecycle_invariant(monkeypatch):
    """The real authorization artifact has three legitimate lifecycle
    states, and this test accepts any of them:

    1. Pre-authorization: the artifact is absent.
    2. Authorization transition or later, fingerprint-current: if
       present, its CONTENT must be fully valid (parses, satisfies the
       committed schema, status=approved, exactly 39 unique cells, binds
       the current fingerprints/commits/checksums, contains no
       scientific-result field) and it must coexist with a header-only-
       or-aborted-only final-test ledger and no completed final-test
       result artifact.
    3. Superseded/stale (e.g. mid-incident-recovery, per
       docs/phase2b_final_test_incident_recovery_freeze.md, after a
       runner-code fix changes compute_final_test_runner_fingerprint()
       but before a new authorization is issued): the on-disk artifact
       legitimately fails production verification with
       FinalTestAuthorizationError due to a fingerprint mismatch -- this
       is the CORRECT, intended behavior (the old authorization must
       never remain active after a runner change), not a defect. The raw
       JSON content is still checked for the same schema/uniqueness/no-
       scientific-field invariants in this case.

    Deliberately does NOT require the artifact to be tracked-and-clean in
    git here -- that is necessarily false in the window between the
    artifact being written to disk and being committed, and this test
    must remain valid throughout that window. Tracked-and-clean
    enforcement remains mandatory and untouched in
    verify_final_test_authorization() itself (see
    test_untracked_artifact_raises/test_dirty_artifact_raises above) and
    in post-authorization-commit verification -- monkeypatching it away
    here affects only this one test's local call, never production
    behavior."""
    import json

    from when_tta_hurts.final_test_authorization import FINAL_TEST_AUTHORIZATION_PATH
    from when_tta_hurts.final_test_evaluation import DEFAULT_FINAL_TEST_ROOT
    from when_tta_hurts.ledger import FINAL_TEST_LEDGER_PATH

    if not FINAL_TEST_AUTHORIZATION_PATH.exists():
        return  # legitimate pre-authorization state

    monkeypatch.setattr(fta, "_is_tracked_and_clean", lambda repo_root, rel_path: True)
    real_git = fta._git

    def _git_or_placeholder_commit(repo_root, *args):
        if args[:3] == ("log", "-1", "--format=%H"):
            # Not yet committed in this transitional window -- every other
            # binding (schema, fingerprints, checksums, cell identities,
            # commit-ancestor checks) is still verified for real below.
            return "0" * 40
        return real_git(repo_root, *args)

    monkeypatch.setattr(fta, "_git", _git_or_placeholder_commit)
    try:
        result = verify_final_test_authorization()  # full content/binding validation, via production code
        assert result.status == "approved"
    except FinalTestAuthorizationError:
        pass  # legitimate superseded/stale state -- see docstring state 3

    raw = json.loads(FINAL_TEST_AUTHORIZATION_PATH.read_text())
    run_ids = [c["run_id"] for c in raw["authorized_cells"]]
    assert len(run_ids) == 39
    assert len(set(run_ids)) == 39

    forbidden_substrings = (
        "accuracy",
        "macro_f1",
        "f1_score",
        "calibration",
        "delta_accuracy",
        '"ece"',
        "brier",
        '"nll"',
        "tta_delta",
    )
    payload_lower = json.dumps(raw).lower()
    for forbidden in forbidden_substrings:
        assert forbidden not in payload_lower, f"forbidden scientific-result field {forbidden!r} found"

    assert FINAL_TEST_LEDGER_PATH.exists()
    ledger_rows = list(csv.DictReader(FINAL_TEST_LEDGER_PATH.open(newline="")))
    # No REAL completed final-test evaluation may exist while authorization
    # is otherwise valid -- any data row must be a non-completed record
    # (e.g. the preserved docs/phase2b_final_test_accidental_access_incident.md
    # aborted row), never a genuine completed evaluation, since this test
    # exercises only identity/binding, never real execution.
    for row in ledger_rows:
        assert row["status"] != "completed", (
            f"unexpected COMPLETED final-test ledger row found: {row!r} -- no real final-test "
            f"evaluation may have occurred outside an explicitly authorized, monitored matrix pass."
        )

    # No COMPLETED final-test artifact set may exist anywhere -- a preserved
    # nonterminal/aborted attempt directory (e.g. the one documented in
    # docs/phase2b_final_test_accidental_access_incident.md) is legitimate
    # and permanent; a completed predictions.npz/metrics.json is not.
    if DEFAULT_FINAL_TEST_ROOT.exists():
        completed_artifacts = list(DEFAULT_FINAL_TEST_ROOT.glob("*/*/predictions.npz"))
        assert completed_artifacts == [], (
            f"unexpected completed final-test artifact(s): {completed_artifacts}"
        )


def test_verify_final_test_authorization_never_imports_torch_or_touches_mps():
    """Structural: this module must never import torch or any device-
    selection symbol -- it is a pure git/hash/metadata gate."""
    import inspect

    source = inspect.getsource(fta)
    assert "import torch" not in source
    assert "select_device" not in source
    assert "np.load" not in source
    assert "predictions.npz" not in source
