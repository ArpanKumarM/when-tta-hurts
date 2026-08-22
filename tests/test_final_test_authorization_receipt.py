"""Phase 2B.6F: synthetic regression tests for the immutable
VerifiedFinalTestReceipt mechanism
(docs/phase2b_final_test_authorization_receipt_freeze.md), which
corrects the exact defect documented in
docs/phase2b_final_test_attempt2_preaccess_failure.md: a second, dynamic
call to verify_final_test_authorization() after attempt allocation
always observed a next-allocatable-attempt one higher than the first
(correct) call did. All fixtures are tmp_path-rooted; none touch the
real authorization artifact or final-test ledger/directory.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import when_tta_hurts.final_test_authorization as fta
from when_tta_hurts.final_test_authorization import (
    FinalTestAuthorizationError,
    VerifiedFinalTestReceipt,
    verify_final_test_authorization,
    verify_receipt_still_valid,
)


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "README.md").write_text("seed commit\n")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


def _bound_commit(repo):
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


def _valid_content(repo, run_id="run-a", authorized_attempt=1):
    return {
        "schema_version": "phase2b.6d-v2",
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
            {
                "run_id": run_id,
                "training_attempt": 1,
                "checkpoint_hash": "chk-a",
                "authorized_final_test_attempt": authorized_attempt,
                "dataset": "pathmnist",
                "resolution": 28,
            }
        ],
    }


def _write_and_commit(repo, content, filename="final_test_authorization.json"):
    path = repo / filename
    path.write_text(json.dumps(content))
    _git(repo, "add", filename)
    _git(repo, "commit", "-q", "-m", "add authorization")
    return path


# ---------------------------------------------------------------------------
# receipt_for()
# ---------------------------------------------------------------------------


def test_receipt_for_returns_correct_fields(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo, authorized_attempt=1))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)

    receipt = auth.receipt_for("run-a")
    assert isinstance(receipt, VerifiedFinalTestReceipt)
    assert receipt.run_id == "run-a"
    assert receipt.authorized_attempt == 1
    assert receipt.checkpoint_hash == "chk-a"
    assert receipt.dataset == "pathmnist"
    assert receipt.resolution == 28
    assert receipt.dataset_expected_checksum_md5 == _FAKE_DATASET_CHECKSUM
    assert receipt.evaluator_fingerprint == _FAKE_EVALUATOR_FP
    assert receipt.authorization_artifact_sha256 == auth.artifact_sha256


def test_receipt_for_unknown_run_id_raises(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)

    with pytest.raises(FinalTestAuthorizationError, match="No authorized cell"):
        auth.receipt_for("not-a-real-run-id")


def test_receipt_has_no_public_constructor_bypass():
    """Structural: VerifiedFinalTestReceipt is a plain frozen dataclass --
    it CAN technically be constructed directly in Python (dataclasses
    don't enforce private construction), but there is no factory
    function or deserialization helper anywhere in the module that would
    let a caller build one from untrusted data; the only path any
    production code actually uses is FinalTestAuthorization.receipt_for().
    Checked via actual function/method definitions, not a substring
    match (which would false-positive on this test file's own
    explanatory docstrings)."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fta))
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "from_dict" not in function_names
    assert "make_receipt" not in function_names
    assert function_names & {"receipt_for"} == {"receipt_for"}


# ---------------------------------------------------------------------------
# verify_receipt_still_valid() -- the core fix
# ---------------------------------------------------------------------------


def test_dynamic_next_attempt_becoming_n_plus_1_does_not_invalidate_active_receipt(tmp_path, monkeypatch):
    """THE central regression test: reproduces the exact real-incident
    shape. A receipt is issued authorizing attempt N. The attempt's
    directory is then created (simulating start_evaluation_attempt()
    having just allocated it) -- which would make a FRESH dynamic call to
    verify_final_test_authorization() observe next-allocatable-attempt =
    N+1. verify_receipt_still_valid() must NOT recompute this and must
    still pass."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo, authorized_attempt=1))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    # Simulate start_evaluation_attempt() having just allocated attempt_001
    # for "run-a" -- a FRESH dynamic verify_final_test_authorization() call
    # would now see next_evaluation_attempt_number()==2, which would NOT
    # match the receipt's authorized_attempt==1 (exactly the incident).
    (tmp_path / "final_test" / "run-a" / "attempt_001").mkdir(parents=True)
    (tmp_path / "final_test" / "run-a" / "attempt_001" / "status.json").write_text(
        json.dumps({"training_run_id": "run-a", "attempt_number": 1, "status": "running"})
    )

    monkeypatch.setattr(fta, "_is_tracked_and_clean", lambda repo_root, rel_path: True)

    # Must NOT raise -- the static recheck never touches attempt-directory
    # state at all.
    verify_receipt_still_valid(
        receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
    )


def test_static_recheck_never_calls_next_evaluation_attempt_number(tmp_path, monkeypatch):
    """Mechanical proof the static recheck cannot reach the dynamic
    attempt-allocation logic at all."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    monkeypatch.setattr(fta, "_is_tracked_and_clean", lambda repo_root, rel_path: True)

    source_has_no_dynamic_calls = (
        "next_evaluation_attempt_number"
        not in fta.__dict__.get("verify_receipt_still_valid").__code__.co_names
    )
    assert source_has_no_dynamic_calls

    verify_receipt_still_valid(
        receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
    )


def test_receipt_rejected_for_wrong_dataset(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    with pytest.raises(FinalTestAuthorizationError, match="different cell"):
        verify_receipt_still_valid(
            receipt, "bloodmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_rejected_for_wrong_resolution(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    with pytest.raises(FinalTestAuthorizationError, match="different cell"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 64, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_rejected_after_authorization_bytes_changed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    # Authorization bytes change after the receipt was issued (still
    # committed, just different content -- e.g. amended or superseded).
    _write_and_commit(repo, _valid_content(repo, authorized_attempt=2))

    with pytest.raises(FinalTestAuthorizationError, match="bytes have changed"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_rejected_after_authorization_becomes_dirty(tmp_path, monkeypatch):
    """An uncommitted local edit necessarily changes the file's bytes, so
    the byte/hash check (checked first, matching the frozen order in
    docs/phase2b_final_test_authorization_receipt_freeze.md item 5) is
    what actually fires here -- both are legitimate rejections of the
    same underlying "the committed artifact is no longer what the
    receipt was issued against" condition."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    (repo / "final_test_authorization.json").write_text(
        json.dumps({**_valid_content(repo), "status": "draft"})
    )

    with pytest.raises(FinalTestAuthorizationError, match="bytes have changed|not tracked-and-clean"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_rejected_when_untracked_but_bytes_still_match(tmp_path, monkeypatch):
    """Isolates the tracked-and-clean check specifically: git-remove the
    file from the index (git rm --cached) WITHOUT changing its on-disk
    bytes, so the hash check passes and only the tracked-and-clean check
    can catch it."""
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    subprocess.run(
        ["git", "-C", str(repo), "rm", "--cached", "-q", "final_test_authorization.json"],
        check=True,
        capture_output=True,
    )

    with pytest.raises(FinalTestAuthorizationError, match="tracked-and-clean"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_rejected_after_authorization_removed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    (repo / "final_test_authorization.json").unlink()

    with pytest.raises(FinalTestAuthorizationError, match="no longer exists"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


@pytest.mark.parametrize(
    "fp_field,fake_fn_name",
    [
        ("evaluator_fingerprint", "compute_evaluator_fingerprint"),
        ("statistical_analysis_fingerprint", "compute_analysis_fingerprint"),
        ("cross_condition_analysis_fingerprint", "compute_cross_condition_fingerprint"),
        ("final_test_runner_fingerprint", "compute_final_test_runner_fingerprint"),
    ],
)
def test_receipt_rejected_on_fingerprint_drift(tmp_path, monkeypatch, fp_field, fake_fn_name):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    monkeypatch.setattr(fta, "_is_tracked_and_clean", lambda repo_root, rel_path: True)
    monkeypatch.setattr(fta, fake_fn_name, lambda: ("DRIFTED-VALUE", {}))

    with pytest.raises(FinalTestAuthorizationError, match="drifted"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_rejected_on_dataset_checksum_drift(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    monkeypatch.setattr(fta, "_is_tracked_and_clean", lambda repo_root, rel_path: True)
    monkeypatch.setattr(fta, "expected_official_checksum", lambda dataset, resolution: "DRIFTED" * 4)

    with pytest.raises(FinalTestAuthorizationError, match="drifted"):
        verify_receipt_still_valid(
            receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
        )


def test_receipt_still_valid_when_nothing_changed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _patch_identity(monkeypatch)
    _write_and_commit(repo, _valid_content(repo))
    auth = verify_final_test_authorization(artifact_path="final_test_authorization.json", repo_root=repo)
    receipt = auth.receipt_for("run-a")

    # Should not raise.
    verify_receipt_still_valid(
        receipt, "pathmnist", 28, artifact_path="final_test_authorization.json", repo_root=repo
    )
