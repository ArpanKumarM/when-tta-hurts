"""Tests for the Phase 2B.4D Data-Integrity Addendum: checksum-verified
evaluation dataset loading. Uses ONLY synthetic NPZ fixtures, temporary
repositories/ledgers, injected fakes, and mocked heavy dependencies. Never
touches a real dataset artifact under data/raw/, never initializes real
MPS, never loads a real checkpoint, and never accesses the official test
split.
"""

from __future__ import annotations

import hashlib
import inspect
import json

import numpy as np
import pytest
import torch

from when_tta_hurts.dataset_verification import (
    ArtifactVerificationError,
    expected_official_checksum,
    verify_official_dataset_artifact,
)
from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    EvaluationSchemaValidationError,
    persist_and_verify_evaluation_completion,
)
from when_tta_hurts.validation_evaluation import (
    EVALUATOR_FINGERPRINT_MANIFEST,
    ValidationEvaluationConfig,
    compute_evaluation_id,
    compute_evaluator_fingerprint,
    finish_evaluation_attempt,
    load_frozen_tta_seed_config,
    run_validation_evaluation,
    start_evaluation_attempt,
)
from when_tta_hurts.validation_evaluation import (
    EvaluationRunStatus as _RunStatus,
)

_VALID_PREFIX_SEQUENCE = (1, 2, 5, 10, 25, 50, 100)

_VALID_YAML_TEXT = """
schema_version: "1.0"
status: approved
split: validation
confirmatory_tta_seed: 1306178015
derivation:
  namespace: "when-tta-hurts|phase2b|confirmatory-tta|v1"
  sha256_digest: "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd"
  conversion_rule: "int(digest[:8], 16)"
excluded_seeds:
  pilot_tta_seed: 271828
  pilot_training_seed: 314159
  confirmatory_training_seeds: [0, 1, 2]
prefix_sequence: [1, 2, 5, 10, 25, 50, 100]
total_generated_views: 100
primary_prefix: 50
primary_aggregation: mean_probability
policy_identifier: mixed
inference_batch_size: 256
bn_adaptation_batch_size: 256
bn_adaptation_algorithm: sequential_microbatch_v1
bn_adaptation_enumeration_order: view_major_then_sample_major
metric_input_contract: probability_native_v1
"""


def _always_tracked_clean(path):
    return True


def _commit_for(commit):
    return lambda path: commit


def _all_ancestors(commit, head):
    return True


def _valid_latency(n_samples=3, prefix_sequence=_VALID_PREFIX_SEQUENCE, clean_latency=0.01):
    by_n = {}
    for n in prefix_sequence:
        tta = clean_latency * n
        by_n[str(n)] = {
            "tta_latency_seconds": tta,
            "per_sample_latency_seconds": tta / n_samples,
            "compute_multiplier": tta / clean_latency,
        }
    return {"clean_latency_seconds": clean_latency, "n_samples": n_samples, "by_n": by_n}


def _valid_batching():
    return {
        "inference_batch_size": 256,
        "bn_adaptation_batch_size": 256,
        "bn_adaptation_algorithm": "sequential_microbatch_v1",
        "bn_adaptation_enumeration_order": "view_major_then_sample_major",
        "bn_adaptation_applicable": False,
        "bn_adaptation_microbatches_at_primary_n": 0,
    }


def _valid_dataset_verification(dataset="pathmnist", resolution=28, checksum="a" * 32):
    return {
        "dataset": dataset,
        "resolution": resolution,
        "expected_checksum_md5": checksum,
        "actual_checksum_md5": checksum,
        "checksum_verified": True,
        "resized": False,
        "verification_method": "dataset_verification.verify_official_dataset_artifact",
        "verification_version": 1,
        "artifact_path": f"data/raw/{dataset}.npz",
    }


def _valid_metadata(dataset="pathmnist", resolution=28, checksum="a" * 32):
    return {
        "evaluation_id": "e1",
        "training_run_id": "r1",
        "training_attempt": 1,
        "checkpoint_hash": "c1",
        "dataset": dataset,
        "resolution": resolution,
        "model": "small_cnn",
        "normalization": "groupnorm",
        "training_policy": "none",
        "seed": 0,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "cfgsha",
        "tta_seed_freeze_commit": "c" * 40,
        "tta_seed_derivation_sha256": "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
        "prefix_sequence": [1, 2, 5, 10, 25, 50, 100],
        "aggregators": ["mean_probability"],
        "secondary_analyses": ["scaling_curve"],
        "protocol_commit": "ce4c962",
        "matrix_hash": "m1",
        "source_commit": "s1",
        "evaluator_fingerprint": "fp1",
        "evaluator_fingerprint_manifest": {"src/when_tta_hurts/metrics.py": "abc123"},
        "dataset_expected_checksum_md5": checksum,
        "dataset_verification": _valid_dataset_verification(dataset, resolution, checksum),
        "batching": _valid_batching(),
        "evaluation_config_hash": "e1",
        "split": "validation",
        "n_validation_samples": 3,
        "metric_input_contract": "probability_native_v1",
    }


def _valid_view_manifest():
    return {
        "dataset": "pathmnist",
        "resolution": 28,
        "tta_seed": 1306178015,
        "tta_seed_config_sha256": "cfgsha",
        "tta_seed_freeze_commit": "c" * 40,
        "tta_seed_derivation_sha256": "4ddab1df75616fbff1543665667d24ccb0b047f37dca42a8ae2bbaad55d81acd",
        "n_views": 100,
        "seed_formula": "sha256(...)",
        "sample_indices": [0, 1, 2],
        "seed_manifest_sha256": "abc",
    }


def _valid_predictions(n=3, c=3):
    return {
        "labels": np.arange(n) % c,
        "sample_indices": np.arange(n),
        "clean_probs": np.full((n, c), 1.0 / c, dtype=np.float32),
        "view_probs": np.full((100, n, c), 1.0 / c, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# dataset_verification.py core behavior (synthetic, no real datasets)
# ---------------------------------------------------------------------------


def _fake_info_entry(md5_by_key):
    return dict(md5_by_key)


def test_correct_checksum_verifies_successfully(tmp_path, monkeypatch):
    content = b"synthetic official artifact bytes\x00\x01\x02"
    real_md5 = hashlib.md5(content).hexdigest()
    (tmp_path / "fakeds.npz").write_bytes(content)

    import when_tta_hurts.dataset_verification as dv

    monkeypatch.setitem(dv.INFO, "fakeds", _fake_info_entry({"MD5": real_md5}))

    result = verify_official_dataset_artifact("fakeds", 28, root=tmp_path)
    assert result.checksum_verified is True
    assert result.resized is False
    assert result.expected_checksum_md5 == real_md5
    assert result.actual_checksum_md5 == real_md5


def test_missing_artifact_fails_closed(tmp_path, monkeypatch):
    import when_tta_hurts.dataset_verification as dv

    monkeypatch.setitem(dv.INFO, "fakeds", _fake_info_entry({"MD5": "0" * 32}))
    with pytest.raises(ArtifactVerificationError):
        verify_official_dataset_artifact("fakeds", 28, root=tmp_path)
    # no file was created as a side effect (no download fallback)
    assert not (tmp_path / "fakeds.npz").exists()


def test_checksum_mismatch_fails_closed(tmp_path, monkeypatch):
    import when_tta_hurts.dataset_verification as dv

    (tmp_path / "fakeds.npz").write_bytes(b"corrupt or resized bytes")
    monkeypatch.setitem(dv.INFO, "fakeds", _fake_info_entry({"MD5": "f" * 32}))  # deliberately wrong
    with pytest.raises(ArtifactVerificationError):
        verify_official_dataset_artifact("fakeds", 28, root=tmp_path)


def test_resized_proxy_rejected_via_checksum_mismatch(tmp_path, monkeypatch):
    """A resized/proxy file placed at the expected native filename is
    rejected -- resizing changes file bytes, so its MD5 can never match
    the official native-resolution checksum recorded in medmnist.INFO."""
    import when_tta_hurts.dataset_verification as dv

    native_content = b"the real native 64px artifact bytes"
    native_md5 = hashlib.md5(native_content).hexdigest()
    resized_content = b"a resized-down 28px proxy masquerading as 64px"
    assert hashlib.md5(resized_content).hexdigest() != native_md5

    (tmp_path / "fakeds_64.npz").write_bytes(resized_content)  # the PROXY, not the real file
    monkeypatch.setitem(dv.INFO, "fakeds", _fake_info_entry({"MD5_64": native_md5}))

    with pytest.raises(ArtifactVerificationError):
        verify_official_dataset_artifact("fakeds", 64, root=tmp_path)


def test_unsupported_dataset_rejected():
    with pytest.raises(ArtifactVerificationError):
        expected_official_checksum("not-a-real-dataset", 28)
    with pytest.raises(ArtifactVerificationError):
        verify_official_dataset_artifact("not-a-real-dataset", 28)


def test_unsupported_resolution_rejected(monkeypatch):
    import when_tta_hurts.dataset_verification as dv

    monkeypatch.setitem(dv.INFO, "fakeds", _fake_info_entry({"MD5": "0" * 32}))
    with pytest.raises(ArtifactVerificationError):
        expected_official_checksum("fakeds", 999)
    with pytest.raises(ArtifactVerificationError):
        verify_official_dataset_artifact("fakeds", 999)


def test_no_download_logic_exists_anywhere_in_verification():
    """Structural proof: neither the expected-checksum lookup nor the
    full verification function contains any download/network/fetch
    logic -- a missing artifact is always a hard failure, never a
    trigger to fetch one."""
    import when_tta_hurts.dataset_verification as dv

    for fn in (expected_official_checksum, verify_official_dataset_artifact):
        source = inspect.getsource(fn)
        for forbidden in ("download", "urlopen", "requests.", "urllib", "http://", "https://"):
            assert forbidden not in source.lower(), f"{fn.__name__} contains forbidden token {forbidden!r}"
    module_source = inspect.getsource(dv)
    assert "download=True" not in module_source
    assert "download=False" not in module_source or "download" not in inspect.getsource(
        verify_official_dataset_artifact
    )


def test_verification_never_indexes_npz_arrays_only_hashes_raw_bytes():
    """Structural proof: verification computes an MD5 over raw file bytes
    only -- it never calls np.load()/np.savez or indexes into any array
    key (train_images/val_images/test_images/etc). This is what makes
    'hash the whole container' safe and NOT test-split access."""
    import when_tta_hurts.dataset_verification as dv

    source = inspect.getsource(dv)
    for forbidden in ("np.load", "numpy.load", "test_images", "test_labels", "['test", '["test'):
        assert forbidden not in source, f"dataset_verification.py contains forbidden token {forbidden!r}"


def test_expected_checksum_lookup_takes_no_path_or_root_parameter():
    """Structural proof that scientific identity cannot depend on the
    absolute local artifact path: expected_official_checksum()'s
    signature has no path/root parameter at all -- it is purely a
    (dataset, resolution) -> medmnist.INFO metadata lookup."""
    sig = inspect.signature(expected_official_checksum)
    assert set(sig.parameters) == {"dataset", "resolution"}


@pytest.mark.parametrize(
    "dataset,resolution",
    [
        ("pathmnist", 28),
        ("pathmnist", 64),
        ("pathmnist", 128),
        ("bloodmnist", 28),
        ("bloodmnist", 64),
        ("bloodmnist", 128),
        ("dermamnist", 28),
    ],
)
def test_all_matrix_dataset_resolution_combinations_resolve_official_checksums(dataset, resolution):
    """Every (dataset, resolution) combination that appears in the real
    39-cell confirmatory matrix must resolve an expected checksum through
    the official medmnist.INFO metadata path, not a duplicated hardcoded
    constant -- cross-checked directly against medmnist.INFO here."""
    from medmnist import INFO

    md5_key = {28: "MD5", 64: "MD5_64", 128: "MD5_128"}[resolution]
    expected_from_info = INFO[dataset][md5_key]
    assert expected_official_checksum(dataset, resolution) == expected_from_info


def test_matrix_combinations_match_real_matrix_cells():
    """Confirms the parametrized set above is exactly what the real,
    committed 39-cell matrix actually contains -- not a stale/assumed
    list."""
    from when_tta_hurts.matrix import parse_and_validate_matrix

    expanded = parse_and_validate_matrix("configs/experiment_matrix.yaml", block_d_gate_passed=True)
    combos = {(c.dataset, c.resolution) for c in expanded.cells}
    assert combos == {
        ("pathmnist", 28),
        ("pathmnist", 64),
        ("pathmnist", 128),
        ("bloodmnist", 28),
        ("bloodmnist", 64),
        ("bloodmnist", 128),
        ("dermamnist", 28),
    }


# ---------------------------------------------------------------------------
# Persisted dataset_verification schema validation
# ---------------------------------------------------------------------------


def test_persist_accepts_valid_dataset_verification(tmp_path):
    predictions = _valid_predictions()
    metadata = _valid_metadata()
    metrics = {
        "training_run_id": "r1",
        "evaluation_config_hash": "e1",
        "clean": {"accuracy": 1.0 / 3},
        "conditions": {},
        "latency": _valid_latency(),
    }
    manifest = persist_and_verify_evaluation_completion(
        tmp_path,
        predictions=predictions,
        metrics=metrics,
        metadata=metadata,
        view_manifest=_valid_view_manifest(),
        prefix_sequence=_VALID_PREFIX_SEQUENCE,
    )
    assert len(manifest["artifacts"]) == 4


@pytest.mark.parametrize(
    "mutate",
    [
        lambda md: md.__delitem__("dataset_verification"),
        lambda md: md["dataset_verification"].__delitem__("expected_checksum_md5"),
        lambda md: md["dataset_verification"].update(dataset="bloodmnist"),  # mismatched dataset
        lambda md: md["dataset_verification"].update(resolution=64),  # mismatched resolution
        lambda md: md["dataset_verification"].update(resolution=999),  # unsupported resolution
        lambda md: md["dataset_verification"].update(actual_checksum_md5="b" * 32),  # unequal checksums
        lambda md: md["dataset_verification"].update(resized=True),  # must reject resized
        lambda md: md["dataset_verification"].update(checksum_verified=False),
        lambda md: md["dataset_verification"].update(expected_checksum_md5="not-a-valid-md5"),
        lambda md: md["dataset_verification"].update(expected_checksum_md5="a" * 31),  # wrong length
        lambda md: md.update(dataset_expected_checksum_md5="mismatched" + "0" * 22),  # binding inconsistency
    ],
)
def test_persist_rejects_malformed_dataset_verification(tmp_path, mutate):
    predictions = _valid_predictions()
    metadata = _valid_metadata()
    mutate(metadata)
    metrics = {
        "training_run_id": "r1",
        "evaluation_config_hash": "e1",
        "clean": {"accuracy": 1.0 / 3},
        "conditions": {},
        "latency": _valid_latency(),
    }
    with pytest.raises((EvaluationSchemaValidationError, EvaluationPersistenceError)):
        persist_and_verify_evaluation_completion(
            tmp_path,
            predictions=predictions,
            metrics=metrics,
            metadata=metadata,
            view_manifest=_valid_view_manifest(),
            prefix_sequence=_VALID_PREFIX_SEQUENCE,
        )


# ---------------------------------------------------------------------------
# Evaluator fingerprint: dataset_verification.py is now manifested
# ---------------------------------------------------------------------------


def test_dataset_verification_is_in_the_fingerprint_manifest():
    assert "src/when_tta_hurts/dataset_verification.py" in EVALUATOR_FINGERPRINT_MANIFEST


def test_fingerprint_changes_when_dataset_verification_changes(tmp_path):
    import shutil
    from pathlib import Path

    real_root = Path(__file__).resolve().parent.parent
    for rel in EVALUATOR_FINGERPRINT_MANIFEST:
        src = real_root / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)

    baseline_fp, _ = compute_evaluator_fingerprint(repo_root=tmp_path)
    target = tmp_path / "src/when_tta_hurts/dataset_verification.py"
    target.write_bytes(target.read_bytes() + b"\n# perturbed for test\n")
    perturbed_fp, _ = compute_evaluator_fingerprint(repo_root=tmp_path)
    assert baseline_fp != perturbed_fp


def test_evaluation_config_has_dataset_expected_checksum_field():
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(ValidationEvaluationConfig)}
    assert "dataset_expected_checksum_md5" in field_names
    # no absolute-path field of any kind participates in identity
    assert not any("path" in name.lower() for name in field_names)


# ---------------------------------------------------------------------------
# Full production-path integration (synthetic checkpoint/split, no MPS)
# ---------------------------------------------------------------------------


def _fake_cell_and_result(dataset="pathmnist", resolution=28, checkpoint_hash="ckpt-fixed", attempt=1):
    cell = type(
        "FakeCell",
        (),
        {
            "run_id": lambda self: "fake-run",
            "dataset": dataset,
            "resolution": resolution,
            "block": "A_core_normalization_resolution",
            "model": "small_cnn",
            "normalization": "batchnorm",
            "training_policy": "none",
            "seed": 0,
        },
    )()
    result = type(
        "R", (), {"attempt_number": attempt, "checkpoint_hash": checkpoint_hash, "status": "completed"}
    )()
    return cell, result


def test_production_order_dataset_verification_before_split_load():
    """Static check: verify_official_dataset_artifact() must appear
    before load_validation_evaluation_split() in run_validation_evaluation()'s
    source."""
    source = inspect.getsource(run_validation_evaluation)
    assert source.index("verify_official_dataset_artifact(") < source.index(
        "load_validation_evaluation_split("
    )


def test_verification_failure_prevents_array_loading_and_produces_failed_status(tmp_path, monkeypatch):
    """Missing artifact -> ArtifactVerificationError, caught by the
    existing failure mechanism -> status="failed", never "completed", and
    load_validation_evaluation_split()/array loading is never reached."""
    import when_tta_hurts.validation_evaluation as ve

    monkeypatch.setattr(ve, "resolve_canonical_training_completion", lambda *a, **k: _fake_cell_and_result())
    monkeypatch.setattr(
        ve, "parse_and_validate_matrix", lambda *a, **k: type("E", (), {"source_config_hash": "m"})()
    )

    def _explode(*a, **k):
        raise AssertionError("load_validation_evaluation_split reached despite verification failure")

    monkeypatch.setattr(ve, "load_validation_evaluation_split", _explode)
    monkeypatch.setattr(ve, "load_and_verify_canonical_checkpoint", lambda *a, **k: object())

    seed_cfg_path = tmp_path / "validation_evaluation.yaml"
    seed_cfg_path.write_text(_VALID_YAML_TEXT)
    root = tmp_path / "eval_root"
    ledger_path = tmp_path / "ledger.csv"
    empty_data_root = tmp_path / "no_data_here"  # deliberately does not exist / is empty

    with pytest.raises(ArtifactVerificationError):
        ve.run_validation_evaluation(
            "fake-run",
            device_resolver=lambda: torch.device("cpu"),
            root=root,
            data_root=empty_data_root,
            evaluation_ledger_path=ledger_path,
            tta_seed_config_path=seed_cfg_path,
            tta_seed_git_tracked_and_clean=_always_tracked_clean,
            tta_seed_last_commit_for_path=_commit_for("c" * 40),
            tta_seed_commit_is_ancestor=_all_ancestors,
            require_clean_tree=False,
        )

    import csv

    with ledger_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"

    attempt_dir = root / "fake-run" / "attempt_001"
    assert not (attempt_dir / "predictions.npz").exists()
    assert not (attempt_dir / "metrics.json").exists()
    status = json.loads((attempt_dir / "status.json").read_text())
    assert status["status"] == "failed"


def test_checksum_mismatch_prevents_array_loading_and_produces_failed_status(tmp_path, monkeypatch):
    import when_tta_hurts.validation_evaluation as ve

    data_root = tmp_path / "data_raw"
    data_root.mkdir()
    (data_root / "pathmnist.npz").write_bytes(b"corrupt bytes, does not match official checksum")
    # deliberately leave medmnist.INFO's real pathmnist MD5 untouched -- a
    # corrupt/wrong-content file at the official filename will never match it.

    monkeypatch.setattr(ve, "resolve_canonical_training_completion", lambda *a, **k: _fake_cell_and_result())
    monkeypatch.setattr(
        ve, "parse_and_validate_matrix", lambda *a, **k: type("E", (), {"source_config_hash": "m"})()
    )

    def _explode(*a, **k):
        raise AssertionError("load_validation_evaluation_split reached despite checksum mismatch")

    monkeypatch.setattr(ve, "load_validation_evaluation_split", _explode)
    monkeypatch.setattr(ve, "load_and_verify_canonical_checkpoint", lambda *a, **k: object())

    seed_cfg_path = tmp_path / "validation_evaluation.yaml"
    seed_cfg_path.write_text(_VALID_YAML_TEXT)
    root = tmp_path / "eval_root"
    ledger_path = tmp_path / "ledger.csv"

    with pytest.raises(ArtifactVerificationError):
        ve.run_validation_evaluation(
            "fake-run",
            device_resolver=lambda: torch.device("cpu"),
            root=root,
            data_root=data_root,
            evaluation_ledger_path=ledger_path,
            tta_seed_config_path=seed_cfg_path,
            tta_seed_git_tracked_and_clean=_always_tracked_clean,
            tta_seed_last_commit_for_path=_commit_for("c" * 40),
            tta_seed_commit_is_ancestor=_all_ancestors,
            require_clean_tree=False,
        )

    import csv

    with ledger_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"


def test_completed_evaluation_skips_before_checksum_computation_and_dataset_loading(tmp_path, monkeypatch):
    """A compatible, already-completed evaluation skips before
    verify_official_dataset_artifact() and load_validation_evaluation_split()
    are ever called."""
    import when_tta_hurts.validation_evaluation as ve
    from when_tta_hurts.ledger import append_evaluation_entry

    monkeypatch.setattr(
        ve, "resolve_canonical_training_completion", lambda *a, **k: _fake_cell_and_result(attempt=3)
    )
    monkeypatch.setattr(
        ve, "parse_and_validate_matrix", lambda *a, **k: type("E", (), {"source_config_hash": "matrixhash"})()
    )

    seed_cfg_path = tmp_path / "validation_evaluation.yaml"
    seed_cfg_path.write_text(_VALID_YAML_TEXT)
    root = tmp_path / "eval_root"
    ledger_path = tmp_path / "ledger.csv"

    cell, training_result = _fake_cell_and_result(attempt=3)
    seed_cfg = load_frozen_tta_seed_config(
        seed_cfg_path,
        git_tracked_and_clean=_always_tracked_clean,
        last_commit_for_path=_commit_for("c" * 40),
        commit_is_ancestor=_all_ancestors,
    )
    fingerprint, _ = compute_evaluator_fingerprint()
    real_expected_checksum = expected_official_checksum(cell.dataset, cell.resolution)
    from when_tta_hurts.validation_evaluation import build_validation_evaluation_config

    cfg = build_validation_evaluation_config(
        cell, training_result, seed_cfg, "matrixhash", fingerprint, real_expected_checksum
    )
    evaluation_id = compute_evaluation_id(cfg)

    attempt_dir, status = start_evaluation_attempt(
        "fake-run", evaluation_id, root=root, ledger_path=ledger_path
    )
    predictions = _valid_predictions()
    metadata = _valid_metadata(checksum=real_expected_checksum)
    metadata.update(
        evaluation_id=evaluation_id,
        training_run_id="fake-run",
        checkpoint_hash="ckpt-fixed",
        evaluation_config_hash=evaluation_id,
        n_validation_samples=3,
    )
    metrics = {
        "training_run_id": "fake-run",
        "evaluation_config_hash": evaluation_id,
        "clean": {"accuracy": 1 / 3},
        "conditions": {},
        "latency": _valid_latency(),
    }
    persist_and_verify_evaluation_completion(
        attempt_dir,
        predictions=predictions,
        metrics=metrics,
        metadata=metadata,
        view_manifest=_valid_view_manifest(),
        prefix_sequence=_VALID_PREFIX_SEQUENCE,
    )
    finish_evaluation_attempt(attempt_dir, status, _RunStatus.COMPLETED)
    append_evaluation_entry(
        evaluation_id=evaluation_id,
        training_run_id="fake-run",
        training_attempt=3,
        checkpoint_hash="ckpt-fixed",
        evaluation_config_hash=evaluation_id,
        evaluation_attempt=status.attempt_number,
        status="completed",
        primary_artifact_hash="art",
        started_at=status.started_at,
        ended_at=status.ended_at,
        runtime_seconds=1.0,
        ledger_path=ledger_path,
    )

    def _explode(*a, **k):
        raise AssertionError("heavy dependency reached -- skip failed to short-circuit")

    monkeypatch.setattr(ve, "verify_official_dataset_artifact", _explode)
    monkeypatch.setattr(ve, "load_validation_evaluation_split", _explode)

    result = ve.run_validation_evaluation(
        "fake-run",
        device_resolver=_explode,
        root=root,
        evaluation_ledger_path=ledger_path,
        tta_seed_config_path=seed_cfg_path,
        tta_seed_git_tracked_and_clean=_always_tracked_clean,
        tta_seed_last_commit_for_path=_commit_for("c" * 40),
        tta_seed_commit_is_ancestor=_all_ancestors,
    )
    assert result["status"] == "skipped_completed"
    assert result["evaluation_id"] == evaluation_id


def test_only_validation_arrays_indexed_by_validation_loader():
    """Structural proof: evaluation/validation_loader.py never references
    'test' split arrays -- 'val' is hardcoded, and load_pilot_split() has
    no test-split mechanism at all."""
    import when_tta_hurts.evaluation.validation_loader as vl

    source = inspect.getsource(vl)
    assert 'split="val"' in source or "split='val'" in source
    for forbidden in ("test_images", "test_labels", 'split="test"', "split='test'"):
        assert forbidden not in source
