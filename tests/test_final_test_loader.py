"""Phase 2B.6A: synthetic tests for evaluation/test_loader.py -- the
test-ONLY evaluation loader. Every test monkeypatches authorization,
checksum verification, and dataset construction; NONE touch a real
dataset file or the real (nonexistent) authorization artifact.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import when_tta_hurts.evaluation.test_loader as tl
from when_tta_hurts.evaluation.test_loader import TestLoaderError, load_final_test_split
from when_tta_hurts.evaluation.validation_loader import ValidationEvaluationSplit


class _FakeVerification:
    def __init__(self, resized=False):
        self.resized = resized


class _CalledUnexpectedly(RuntimeError):
    pass


def _raise_if_called(*_a, **_k):
    raise _CalledUnexpectedly("should not have been called")


class _FakeTorchDataset:
    def __init__(self, n=4, resolution=28, n_classes=9):
        self.n = n
        self.resolution = resolution
        self.n_classes = n_classes

    def __len__(self):
        return self.n


def _patch_success(monkeypatch, *, n=4, resolution=28, n_classes=9, bad_labels=False, bad_shape=False):
    monkeypatch.setattr(tl, "verify_final_test_authorization", lambda **k: None)
    monkeypatch.setattr(
        tl, "verify_official_dataset_artifact", lambda *a, **k: _FakeVerification(resized=False)
    )

    fake_ds = _FakeTorchDataset(n=n, resolution=resolution)
    monkeypatch.setattr(tl, "load_dataset", lambda *a, **k: fake_ds)

    labels = np.full(n, n_classes, dtype=np.int64) if bad_labels else np.zeros(n, dtype=np.int64)
    img_res = resolution + 1 if bad_shape else resolution
    images = torch.zeros(n, 3, img_res, img_res)

    class _FakeLoader:
        def __init__(self, dataset, batch_size, shuffle):
            pass

        def __iter__(self):
            return iter([(images, torch.from_numpy(labels))])

    monkeypatch.setattr(tl, "DataLoader", _FakeLoader)
    monkeypatch.setattr(tl, "INFO", {"pathmnist": {"label": {str(i): "x" for i in range(n_classes)}}})


def test_authorization_checked_before_checksum_and_dataset_access(monkeypatch):
    monkeypatch.setattr(tl, "verify_final_test_authorization", _raise_if_called)
    monkeypatch.setattr(tl, "verify_official_dataset_artifact", _raise_if_called)
    monkeypatch.setattr(tl, "load_dataset", _raise_if_called)

    with pytest.raises(_CalledUnexpectedly):
        load_final_test_split("pathmnist", 28)


def test_checksum_verified_before_dataset_load(monkeypatch):
    calls = []
    monkeypatch.setattr(tl, "verify_final_test_authorization", lambda **k: None)

    def _checksum(*a, **k):
        calls.append("checksum")
        return _FakeVerification(resized=False)

    def _load(*a, **k):
        calls.append("load")
        return _FakeTorchDataset()

    monkeypatch.setattr(tl, "verify_official_dataset_artifact", _checksum)
    monkeypatch.setattr(tl, "load_dataset", _load)
    _patch_success(monkeypatch)
    # override checksum/load with call-order-tracking versions after _patch_success
    monkeypatch.setattr(tl, "verify_official_dataset_artifact", _checksum)
    monkeypatch.setattr(tl, "load_dataset", _load)

    load_final_test_split("pathmnist", 28)
    assert calls == ["checksum", "load"]


def test_resized_proxy_rejected(monkeypatch):
    monkeypatch.setattr(tl, "verify_final_test_authorization", lambda **k: None)
    monkeypatch.setattr(
        tl, "verify_official_dataset_artifact", lambda *a, **k: _FakeVerification(resized=True)
    )
    monkeypatch.setattr(tl, "load_dataset", _raise_if_called)

    with pytest.raises(TestLoaderError, match="resized/proxy"):
        load_final_test_split("pathmnist", 28)


def test_only_split_test_is_ever_requested(monkeypatch):
    captured = {}

    def _load(dataset, split, size, root, allow_test):
        captured["split"] = split
        captured["allow_test"] = allow_test
        return _FakeTorchDataset()

    monkeypatch.setattr(tl, "verify_final_test_authorization", lambda **k: None)
    monkeypatch.setattr(tl, "verify_official_dataset_artifact", lambda *a, **k: _FakeVerification())
    monkeypatch.setattr(tl, "load_dataset", _load)
    _patch_success(monkeypatch)
    monkeypatch.setattr(tl, "load_dataset", _load)

    load_final_test_split("pathmnist", 28)
    assert captured["split"] == "test"
    assert captured["allow_test"] is True


def test_no_split_parameter_on_public_function():
    sig = inspect.signature(load_final_test_split)
    assert "split" not in sig.parameters


def test_shape_mismatch_rejected(monkeypatch):
    _patch_success(monkeypatch, bad_shape=True)
    with pytest.raises(TestLoaderError, match="resolution"):
        load_final_test_split("pathmnist", 28)


def test_label_out_of_range_rejected(monkeypatch):
    _patch_success(monkeypatch, bad_labels=True, n_classes=9)
    with pytest.raises(TestLoaderError, match="out of range"):
        load_final_test_split("pathmnist", 28)


def test_successful_load_returns_validation_evaluation_split(monkeypatch):
    _patch_success(monkeypatch, n=5, resolution=28)
    split = load_final_test_split("pathmnist", 28)
    assert isinstance(split, ValidationEvaluationSplit)
    assert split.dataset == "pathmnist"
    assert split.resolution == 28
    assert split.labels.shape[0] == 5
    assert list(split.sample_indices) == [0, 1, 2, 3, 4]


def test_no_train_or_val_identifiers_referenced_in_source():
    source = inspect.getsource(tl)
    assert "train_images" not in source
    assert "val_images" not in source
    assert 'split="train"' not in source
    assert 'split="val"' not in source


def test_no_reachable_split_override_in_source():
    source = inspect.getsource(tl)
    # The only string literal split value in this module must be "test".
    import ast

    tree = ast.parse(source)
    literal_splits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "split" and isinstance(node.value, ast.Constant):
            literal_splits.add(node.value.value)
    assert literal_splits == {"test"}
