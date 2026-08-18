"""Tests for scripts/benchmark_runtime.py's checksum/native-data guarantees
(Phase 2A audit round 2). Import via importlib since scripts/ isn't a package."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_runtime.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_runtime", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT_PATH.parent.parent / "src"))
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bench_module():
    return _load_module()


def test_resolutions_are_native_28_and_64_only(bench_module):
    assert bench_module.RESOLUTIONS == (28, 64)


def test_verify_official_artifact_delegates_to_shared_module(bench_module, tmp_path, monkeypatch):
    """Checksum verification now lives in when_tta_hurts.dataset_verification
    (Phase 2B.2 audit fix) -- this script's verify_official_artifact() is a
    thin wrapper. Confirm it still works end-to-end via the shared module."""
    fake_path = tmp_path / "pathmnist.npz"
    fake_path.write_bytes(b"fake content")
    import hashlib

    real_md5 = hashlib.md5(b"fake content").hexdigest()

    monkeypatch.setattr(bench_module, "DATA_ROOT", tmp_path)
    from medmnist import INFO

    monkeypatch.setitem(INFO["pathmnist"], "MD5", real_md5)

    result = bench_module.verify_official_artifact(28)
    assert result["resized"] is False
    assert result["checksum_verified"] is True
    assert result["native_resolution"] == 28
    assert result["actual_checksum_md5"] == result["expected_checksum_md5"]


def test_verify_official_artifact_fails_closed_on_checksum_mismatch(bench_module, tmp_path, monkeypatch):
    fake_path = tmp_path / "pathmnist.npz"
    fake_path.write_bytes(b"corrupted content")

    monkeypatch.setattr(bench_module, "DATA_ROOT", tmp_path)
    from medmnist import INFO

    monkeypatch.setitem(INFO["pathmnist"], "MD5", "0" * 32)  # deliberately wrong

    from when_tta_hurts.dataset_verification import ArtifactVerificationError

    with pytest.raises(ArtifactVerificationError, match="CHECKSUM MISMATCH"):
        bench_module.verify_official_artifact(28)


def test_verify_official_artifact_fails_closed_on_missing_file(bench_module, tmp_path, monkeypatch):
    monkeypatch.setattr(bench_module, "DATA_ROOT", tmp_path)  # empty dir, no artifact
    from when_tta_hurts.dataset_verification import ArtifactVerificationError

    with pytest.raises(ArtifactVerificationError, match="does not exist"):
        bench_module.verify_official_artifact(28)


def test_verify_official_artifact_unknown_resolution_raises(bench_module):
    from when_tta_hurts.dataset_verification import ArtifactVerificationError

    with pytest.raises(ArtifactVerificationError):
        bench_module.verify_official_artifact(999)


def test_no_interpolation_helper_used_in_benchmark_cell_source():
    """Static check: benchmark_cell's source must not call F.interpolate or
    torch.rand -- it must use real batches straight from the DataLoader."""
    source = SCRIPT_PATH.read_text()
    cell_start = source.index("def benchmark_cell")
    cell_end = source.index("\ndef main")
    cell_source = source[cell_start:cell_end]
    assert "interpolate" not in cell_source
    assert "torch.rand" not in cell_source


def test_only_train_split_requested_in_main_source():
    """Static check: this script must only ever request split='train'."""
    source = SCRIPT_PATH.read_text()
    assert 'split="test"' not in source
    assert "split='test'" not in source
    assert 'split="train"' in source


def test_64px_artifact_shape_is_native_not_upsampled_28px():
    """If the real artifact is present locally, confirm its native shape --
    a genuinely-64px array, not a resize of the 28px array (which would
    also technically be 64x64 in shape but derived, not native)."""
    path = Path("data/raw/pathmnist_64.npz")
    if not path.exists():
        pytest.skip("pathmnist_64.npz not downloaded locally")
    with np.load(path) as data:
        assert data["train_images"].shape[1:3] == (64, 64)
        assert data["train_images"].shape[0] == 89996


def test_condition_type_field_distinguishes_native(bench_module):
    """The result dict template used by benchmark_cell must tag itself as
    native official real data (not synthetic, not resized proxy)."""
    source = SCRIPT_PATH.read_text()
    assert '"condition_type": "native_official_real_data"' in source
