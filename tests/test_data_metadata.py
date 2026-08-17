"""Dataset metadata tests -- read medmnist's shipped INFO dict only.
No dataset files are downloaded by these tests (see CLAUDE.md / this repo's
CI must not download full datasets)."""

import pytest

from when_tta_hurts.data import (
    EXPECTED_SPLITS,
    SUPPORTED_DATASETS,
    get_dataset_metadata,
    verify_split_counts,
)


@pytest.mark.parametrize("name", SUPPORTED_DATASETS)
def test_get_dataset_metadata_has_expected_fields(name):
    meta = get_dataset_metadata(name)
    assert meta.n_channels == 3
    assert meta.n_classes > 0
    assert meta.license in ("CC BY 4.0", "CC BY-NC 4.0")
    assert 28 in meta.md5_by_resolution


def test_dermamnist_is_noncommercial_license():
    meta = get_dataset_metadata("dermamnist")
    assert meta.license == "CC BY-NC 4.0"


def test_pathmnist_and_bloodmnist_are_cc_by():
    assert get_dataset_metadata("pathmnist").license == "CC BY 4.0"
    assert get_dataset_metadata("bloodmnist").license == "CC BY 4.0"


@pytest.mark.parametrize("name", SUPPORTED_DATASETS)
def test_verify_split_counts_matches_recorded_expectation(name):
    result = verify_split_counts(name)
    assert result["matches"], result["mismatches"]
    assert result["actual"] == EXPECTED_SPLITS[name]


def test_get_dataset_metadata_invalid_name_raises():
    with pytest.raises(ValueError):
        get_dataset_metadata("not_a_real_dataset")
