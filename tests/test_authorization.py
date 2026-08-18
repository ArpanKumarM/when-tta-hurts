"""Tests for authorization.py -- the final-test-evaluation gate. Every
test mocks/uses temporary files; NONE touch real data or the real
(nonexistent) authorization artifact."""

import subprocess

import pytest
import yaml

from when_tta_hurts.authorization import AuthorizationError, verify_authorization


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    return tmp_path


VALID_CONTENT = {
    "status": "approved",
    "phase2b_protocol_commit": "ce4c962",
    "validation_gated_tta_freeze_commit": "abcdef1",
    "matrix_config_hash": "somehash",
    "approved_test_conditions": ["clean", "naive_tta"],
    "approval_timestamp": "2026-09-01T00:00:00Z",
}


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(AuthorizationError, match="does not exist"):
        verify_authorization(tmp_path / "does_not_exist.yaml")


def test_untracked_artifact_raises(tmp_path):
    repo = _init_repo(tmp_path)
    artifact = repo / "auth.yaml"
    artifact.write_text(yaml.dump(VALID_CONTENT))
    # not git-added -- untracked
    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(AuthorizationError, match="not tracked-and-clean"):
            verify_authorization(artifact)
    finally:
        os.chdir(cwd)


def test_dirty_artifact_raises(tmp_path):
    repo = _init_repo(tmp_path)
    artifact = repo / "auth.yaml"
    artifact.write_text(yaml.dump(VALID_CONTENT))
    _git(repo, "add", "auth.yaml")
    _git(repo, "commit", "-q", "-m", "add auth")
    # now modify it without committing -> dirty
    artifact.write_text(yaml.dump({**VALID_CONTENT, "status": "draft"}))

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(AuthorizationError, match="not tracked-and-clean"):
            verify_authorization(artifact)
    finally:
        os.chdir(cwd)


def test_committed_valid_artifact_passes(tmp_path):
    repo = _init_repo(tmp_path)
    artifact = repo / "auth.yaml"
    artifact.write_text(yaml.dump(VALID_CONTENT))
    _git(repo, "add", "auth.yaml")
    _git(repo, "commit", "-q", "-m", "add auth")

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        result = verify_authorization(artifact)
        assert result.status == "approved"
    finally:
        os.chdir(cwd)


def test_draft_status_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    artifact = repo / "auth.yaml"
    artifact.write_text(yaml.dump({**VALID_CONTENT, "status": "draft"}))
    _git(repo, "add", "auth.yaml")
    _git(repo, "commit", "-q", "-m", "add auth")

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(AuthorizationError, match="not 'approved'"):
            verify_authorization(artifact)
    finally:
        os.chdir(cwd)


def test_missing_field_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    incomplete = dict(VALID_CONTENT)
    del incomplete["approval_timestamp"]
    artifact = repo / "auth.yaml"
    artifact.write_text(yaml.dump(incomplete))
    _git(repo, "add", "auth.yaml")
    _git(repo, "commit", "-q", "-m", "add auth")

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(AuthorizationError, match="missing required field"):
            verify_authorization(artifact)
    finally:
        os.chdir(cwd)


def test_malformed_yaml_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    artifact = repo / "auth.yaml"
    artifact.write_text("not: valid: yaml: [structure")
    _git(repo, "add", "auth.yaml")
    _git(repo, "commit", "-q", "-m", "add bad auth")

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(AuthorizationError, match="malformed"):
            verify_authorization(artifact)
    finally:
        os.chdir(cwd)


def test_matrix_hash_mismatch_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    artifact = repo / "auth.yaml"
    artifact.write_text(yaml.dump(VALID_CONTENT))
    _git(repo, "add", "auth.yaml")
    _git(repo, "commit", "-q", "-m", "add auth")

    import os

    cwd = os.getcwd()
    try:
        os.chdir(repo)
        with pytest.raises(AuthorizationError, match="does not match"):
            verify_authorization(artifact, expected_matrix_config_hash="a-different-hash")
    finally:
        os.chdir(cwd)


def test_real_authorization_artifact_does_not_exist():
    """Confirms the REAL (production) authorization artifact path does not
    exist -- Phase 2B.2 must not create it."""
    from when_tta_hurts.authorization import AUTHORIZATION_ARTIFACT_PATH

    assert not AUTHORIZATION_ARTIFACT_PATH.exists()


def test_cli_flag_alone_cannot_unlock():
    """Static check: verify_authorization() takes no CLI-flag-like or
    env-var-like boolean override parameter."""
    import inspect

    sig = inspect.signature(verify_authorization)
    param_names = set(sig.parameters.keys())
    assert param_names == {"artifact_path", "expected_matrix_config_hash"}
    for forbidden in ("force", "override", "skip", "unlock", "bypass"):
        assert forbidden not in param_names
