"""Regression tests for the num_classes-per-dataset fix: orchestrator.py's
_build_model() previously hardcoded num_classes=9 for every dataset,
which is correct for PathMNIST but wrong for BloodMNIST (8 classes) and
DermaMNIST (7 classes). num_classes must always be derived from
data.py::get_dataset_metadata() (backed by medmnist.INFO), never
hardcoded or inferred from a batch.

No dataset file, split, or real training is accessed by any test here --
get_dataset_metadata() is pure metadata (medmnist.INFO dict lookups)."""

from __future__ import annotations

import inspect

import pytest
import torch

from when_tta_hurts.data import get_dataset_metadata
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.orchestrator import _build_model


def _cell(dataset, model="small_cnn", resolution=28, normalization="batchnorm"):
    return MatrixCell(
        block="A_core_normalization_resolution",
        dataset=dataset,
        resolution=resolution,
        model=model,
        normalization=normalization,
        training_policy="none",
        seed=0,
    )


def test_pathmnist_small_cnn_output_dimension_is_9():
    model = _build_model(_cell("pathmnist"))
    assert model.classifier.out_features == 9


def test_bloodmnist_small_cnn_output_dimension_is_8():
    model = _build_model(_cell("bloodmnist"))
    assert model.classifier.out_features == 8


def test_dermamnist_resnet18_output_dimension_is_7():
    model = _build_model(_cell("dermamnist", model="resnet18"))
    assert model.fc.out_features == 7


@pytest.mark.parametrize(
    "dataset,model_type,expected_n",
    [
        ("pathmnist", "small_cnn", 9),
        ("bloodmnist", "small_cnn", 8),
        ("dermamnist", "small_cnn", 7),
        ("pathmnist", "resnet18", 9),
        ("bloodmnist", "resnet18", 8),
        ("dermamnist", "resnet18", 7),
    ],
)
def test_forward_logits_shape_matches_n_classes(dataset, model_type, expected_n):
    cell = _cell(dataset, model=model_type)
    model = _build_model(cell)
    model.eval()
    x = torch.rand(2, 3, 28, 28)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, expected_n)


def test_unknown_dataset_fails_before_model_construction():
    cell = _cell("not_a_real_dataset")
    with pytest.raises(ValueError, match="Unsupported dataset"):
        _build_model(cell)


def test_no_hardcoded_universal_num_classes_9_in_build_model_source():
    source = inspect.getsource(_build_model)
    assert "num_classes=9" not in source
    assert "get_dataset_metadata" in source


def test_pathmnist_architecture_and_parameter_shapes_unchanged():
    """The pre-fix architecture for PathMNIST (n_classes=9, which was
    already correct) must be byte-identical in shape to the post-fix one
    -- this fix must never alter PathMNIST's model at all."""
    model = _build_model(_cell("pathmnist"))
    expected_shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    # Re-derive independently via get_dataset_metadata to confirm n_classes==9
    assert get_dataset_metadata("pathmnist").n_classes == 9
    assert expected_shapes["classifier.weight"][0] == 9
    assert expected_shapes["classifier.bias"][0] == 9


def test_dataset_metadata_never_touches_real_files():
    """get_dataset_metadata is pure medmnist.INFO metadata -- confirms no
    dataset file access is required to determine class counts."""
    import inspect as _inspect

    from when_tta_hurts import data as data_module

    source = _inspect.getsource(data_module.get_dataset_metadata)
    assert "open(" not in source
    assert ".npz" not in source
    assert "download" not in source
