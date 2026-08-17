"""Empirical split-count verification against a downloaded .npz artifact.

Skipped automatically if the artifact isn't present -- CI must not download
full datasets (see CLAUDE.md), so this only runs locally after
scripts/smoke_test.py (or an equivalent download) has fetched pathmnist.npz.
"""

from pathlib import Path

import pytest

from when_tta_hurts.data import EXPECTED_SPLITS, verify_split_counts_from_artifact

PATHMNIST_NPZ = Path("data/raw/pathmnist.npz")


@pytest.mark.skipif(not PATHMNIST_NPZ.exists(), reason="pathmnist.npz not downloaded locally")
def test_pathmnist_split_counts_match_downloaded_artifact():
    result = verify_split_counts_from_artifact("pathmnist")
    assert result["verification_level"] == "empirical_artifact"
    assert result["matches"], result["mismatches"]
    assert result["actual"] == EXPECTED_SPLITS["pathmnist"]


def test_verify_split_counts_from_artifact_raises_if_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_split_counts_from_artifact("pathmnist", root=tmp_path)


def test_verify_split_counts_from_artifact_invalid_dataset_raises():
    with pytest.raises(ValueError):
        verify_split_counts_from_artifact("not_a_real_dataset", root="data/raw")
