"""Phase 2B.6F: synthetic tests for evaluation/test_loader.py -- the
test-ONLY evaluation loader. Every test monkeypatches the static receipt
recheck and dataset construction; NONE touch a real dataset file or the
real (nonexistent) authorization artifact. Uses a real
VerifiedFinalTestReceipt (the only way to construct one is via
FinalTestAuthorization.receipt_for(), so tests build a minimal fake
FinalTestAuthorization and call the real receipt_for() on it, rather
than constructing a receipt directly).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

import when_tta_hurts.evaluation.test_loader as tl
from when_tta_hurts.evaluation.test_loader import TestLoaderError, load_final_test_split
from when_tta_hurts.evaluation.validation_loader import ValidationEvaluationSplit
from when_tta_hurts.final_test_authorization import FinalTestAuthorization


def _make_receipt(dataset="pathmnist", resolution=28):
    """Builds a receipt the ONLY sanctioned way: via
    FinalTestAuthorization.receipt_for()."""
    auth = FinalTestAuthorization(
        status="approved",
        schema_version="phase2b.6d-v2",
        approval_timestamp="2026-01-01T00:00:00Z",
        phase2b_protocol_commit="proto",
        matrix_commit="matrix",
        cross_condition_addendum_commit="addendum",
        evaluator_fingerprint="eval-fp",
        statistical_analysis_fingerprint="sa-fp",
        cross_condition_analysis_fingerprint="cc-fp",
        final_test_runner_fingerprint="runner-fp",
        authorized_cells_by_run_id={
            "run-a": {
                "run_id": "run-a",
                "training_attempt": 1,
                "checkpoint_hash": "chk-a",
                "authorized_final_test_attempt": 1,
                "dataset": dataset,
                "resolution": resolution,
            }
        },
        official_dataset_checksums={f"{dataset}@{resolution}": "0" * 32},
        artifact_sha256="auth-sha",
        authorization_commit="auth-commit",
        supersedes_authorization_sha256=None,
        supersedes_authorization_commit=None,
        incident_record_commit=None,
        recovery_policy_commit=None,
        no_further_retry=None,
    )
    return auth.receipt_for("run-a")


class _CalledUnexpectedly(RuntimeError):
    pass


def _raise_if_called(*_a, **_k):
    raise _CalledUnexpectedly("should not have been called")


class _FakeVerification:
    def __init__(self, resized=False):
        self.resized = resized


class _FakeTorchDataset:
    def __init__(self, n=4, resolution=28, n_classes=9):
        self.n = n
        self.resolution = resolution
        self.n_classes = n_classes

    def __len__(self):
        return self.n


def _patch_success(monkeypatch, *, n=4, resolution=28, n_classes=9, bad_labels=False, bad_shape=False):
    monkeypatch.setattr(tl, "verify_receipt_still_valid", lambda receipt, dataset, resolution: None)
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


def test_receipt_recheck_before_checksum_and_dataset_access(monkeypatch):
    monkeypatch.setattr(tl, "verify_receipt_still_valid", _raise_if_called)
    monkeypatch.setattr(tl, "verify_official_dataset_artifact", _raise_if_called)
    monkeypatch.setattr(tl, "load_dataset", _raise_if_called)

    with pytest.raises(_CalledUnexpectedly):
        load_final_test_split("pathmnist", 28, receipt=_make_receipt())


def test_checksum_verified_before_dataset_load(monkeypatch):
    calls = []

    def _receipt_check(receipt, dataset, resolution):
        calls.append("receipt")

    def _checksum(*a, **k):
        calls.append("checksum")
        return _FakeVerification(resized=False)

    def _load(*a, **k):
        calls.append("load")
        return _FakeTorchDataset()

    _patch_success(monkeypatch)
    monkeypatch.setattr(tl, "verify_receipt_still_valid", _receipt_check)
    monkeypatch.setattr(tl, "verify_official_dataset_artifact", _checksum)
    monkeypatch.setattr(tl, "load_dataset", _load)

    load_final_test_split("pathmnist", 28, receipt=_make_receipt())
    assert calls == ["receipt", "checksum", "load"]


def test_resized_proxy_rejected(monkeypatch):
    monkeypatch.setattr(tl, "verify_receipt_still_valid", lambda receipt, dataset, resolution: None)
    monkeypatch.setattr(
        tl, "verify_official_dataset_artifact", lambda *a, **k: _FakeVerification(resized=True)
    )
    monkeypatch.setattr(tl, "load_dataset", _raise_if_called)

    with pytest.raises(TestLoaderError, match="resized/proxy"):
        load_final_test_split("pathmnist", 28, receipt=_make_receipt())


def test_only_split_test_is_ever_requested(monkeypatch):
    captured = {}

    def _load(dataset, split, size, root, allow_test):
        captured["split"] = split
        captured["allow_test"] = allow_test
        return _FakeTorchDataset()

    _patch_success(monkeypatch)
    monkeypatch.setattr(tl, "load_dataset", _load)

    load_final_test_split("pathmnist", 28, receipt=_make_receipt())
    assert captured["split"] == "test"
    assert captured["allow_test"] is True


def test_receipt_is_required_keyword_only():
    sig = inspect.signature(load_final_test_split)
    assert "split" not in sig.parameters
    assert sig.parameters["receipt"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["receipt"].default is inspect.Parameter.empty


def test_shape_mismatch_rejected(monkeypatch):
    _patch_success(monkeypatch, bad_shape=True)
    with pytest.raises(TestLoaderError, match="resolution"):
        load_final_test_split("pathmnist", 28, receipt=_make_receipt())


def test_label_out_of_range_rejected(monkeypatch):
    _patch_success(monkeypatch, bad_labels=True, n_classes=9)
    with pytest.raises(TestLoaderError, match="out of range"):
        load_final_test_split("pathmnist", 28, receipt=_make_receipt())


def test_successful_load_returns_validation_evaluation_split(monkeypatch):
    _patch_success(monkeypatch, n=5, resolution=28)
    split = load_final_test_split("pathmnist", 28, receipt=_make_receipt())
    assert isinstance(split, ValidationEvaluationSplit)
    assert split.dataset == "pathmnist"
    assert split.resolution == 28
    assert split.labels.shape[0] == 5
    assert list(split.sample_indices) == [0, 1, 2, 3, 4]


def test_receipt_for_wrong_dataset_or_resolution_rejected(monkeypatch):
    """A receipt bound to a DIFFERENT cell than the one being loaded must
    hard-fail -- verified against the REAL verify_receipt_still_valid()
    (not mocked here)."""
    receipt = _make_receipt(dataset="pathmnist", resolution=28)
    from when_tta_hurts.final_test_authorization import FinalTestAuthorizationError

    monkeypatch.setattr(tl, "verify_official_dataset_artifact", _raise_if_called)
    monkeypatch.setattr(tl, "load_dataset", _raise_if_called)
    with pytest.raises(FinalTestAuthorizationError, match="different cell"):
        load_final_test_split("bloodmnist", 28, receipt=receipt)


def test_loader_cannot_invoke_full_verifier():
    """Structural: this module must never IMPORT
    verify_final_test_authorization (only mention it in prose/docstrings
    explaining why not) -- checked via actual import statements, not a
    substring match over the whole file (which would false-positive on
    this module's own explanatory docstring)."""
    import ast

    tree = ast.parse(inspect.getsource(tl))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)
    assert "verify_final_test_authorization" not in imported_names
    assert "verify_receipt_still_valid" in imported_names


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
