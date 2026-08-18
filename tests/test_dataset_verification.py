"""Tests for dataset_verification.py -- uses ONLY temporary fabricated
files and mocked medmnist.INFO entries. Never downloads or inspects a real
dataset artifact."""

import hashlib

import pytest

from when_tta_hurts.dataset_verification import (
    ArtifactVerificationError,
    verify_official_dataset_artifact,
)


def _write_fake_artifact(path, content: bytes) -> str:
    path.write_bytes(content)
    return hashlib.md5(content).hexdigest()


def test_matching_checksum_passes(tmp_path, monkeypatch):
    from medmnist import INFO

    path = tmp_path / "pathmnist.npz"
    real_md5 = _write_fake_artifact(path, b"fake pathmnist 28px content")
    monkeypatch.setitem(INFO["pathmnist"], "MD5", real_md5)

    result = verify_official_dataset_artifact("pathmnist", 28, root=tmp_path)
    assert result.checksum_verified is True
    assert result.resized is False
    assert result.native_resolution == 28
    assert result.actual_checksum_md5 == result.expected_checksum_md5


def test_checksum_mismatch_fails(tmp_path, monkeypatch):
    from medmnist import INFO

    path = tmp_path / "pathmnist.npz"
    path.write_bytes(b"corrupted content")
    monkeypatch.setitem(INFO["pathmnist"], "MD5", "0" * 32)  # deliberately wrong

    with pytest.raises(ArtifactVerificationError, match="CHECKSUM MISMATCH"):
        verify_official_dataset_artifact("pathmnist", 28, root=tmp_path)


def test_wrong_native_resolution_fails(tmp_path):
    """Resolution 999 has no MD5 key convention registered at all."""
    with pytest.raises(ArtifactVerificationError, match="Unsupported/unregistered resolution"):
        verify_official_dataset_artifact("pathmnist", 999, root=tmp_path)


def test_resized_proxy_cannot_claim_native_status(tmp_path, monkeypatch):
    """A 28px file's bytes, even if placed at the 64px filename, will NOT
    match the official 64px checksum (resizing/copying changes file bytes
    -- the checksum check catches any substitution, proving a resized
    proxy can never pass as native)."""
    from medmnist import INFO

    # Simulate: someone takes 28px content and drops it in as if it were 64px.
    fake_64px_path = tmp_path / "pathmnist_64.npz"
    fake_64px_path.write_bytes(b"this is actually 28px content, not real 64px")

    # The real medmnist.INFO MD5_64 checksum (unmodified) will NOT match
    # this substituted content.
    real_expected_64_md5 = INFO["pathmnist"]["MD5_64"]
    actual_md5 = hashlib.md5(fake_64px_path.read_bytes()).hexdigest()
    assert actual_md5 != real_expected_64_md5  # sanity: genuinely different

    with pytest.raises(ArtifactVerificationError, match="CHECKSUM MISMATCH"):
        verify_official_dataset_artifact("pathmnist", 64, root=tmp_path)


def test_unsupported_artifact_missing_file_fails(tmp_path, monkeypatch):
    from medmnist import INFO

    monkeypatch.setitem(INFO["pathmnist"], "MD5", "deadbeef" * 4)
    # tmp_path is empty -- no pathmnist.npz file exists.
    with pytest.raises(ArtifactVerificationError, match="does not exist"):
        verify_official_dataset_artifact("pathmnist", 28, root=tmp_path)


def test_unsupported_dataset_name_fails(tmp_path):
    with pytest.raises(ArtifactVerificationError, match="Unsupported dataset"):
        verify_official_dataset_artifact("not_a_real_dataset", 28, root=tmp_path)


def test_generalizes_across_all_three_registered_datasets(tmp_path, monkeypatch):
    from medmnist import INFO

    for name in ("pathmnist", "bloodmnist", "dermamnist"):
        path = tmp_path / f"{name}.npz"
        real_md5 = _write_fake_artifact(path, f"fake {name} content".encode())
        monkeypatch.setitem(INFO[name], "MD5", real_md5)
        result = verify_official_dataset_artifact(name, 28, root=tmp_path)
        assert result.dataset == name
        assert result.checksum_verified is True


def test_real_medmnist_checksums_are_distinct_per_resolution():
    """Sanity check against the REAL (unmocked) medmnist.INFO: the 28px and
    64px checksums for pathmnist must differ (confirming they are genuinely
    different artifacts, not the same file referenced twice)."""
    from medmnist import INFO

    assert INFO["pathmnist"]["MD5"] != INFO["pathmnist"]["MD5_64"]
