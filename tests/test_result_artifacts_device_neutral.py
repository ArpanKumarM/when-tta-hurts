"""Regression tests for the device-neutral checkpoint-verification fix
(Phase 2B.3A Part 2 of the attempt_002 correction): persist_and_verify_completion's
checkpoint-restore comparison previously constructed its comparison model
on the default (CPU) device while the saved checkpoint's tensors remained
on whatever device training used (e.g. MPS), causing a spurious
RuntimeError on cross-device torch.equal(). The fix always compares on
CPU regardless of the training device.

Tiny synthetic models/data and temporary directories ONLY -- never
touches real attempt_001/attempt_002 or any real dataset."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from when_tta_hurts.artifacts import hash_state_dict, save_checkpoint
from when_tta_hurts.result_artifacts import (
    PersistenceVerificationError,
    _to_cpu_detached,
    persist_and_verify_completion,
)


def _tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 5))


def _valid_result_fields():
    return {
        "run_id": "x",
        "attempt_id": 1,
        "best_epoch": 1,
        "best_val_accuracy": 0.5,
        "best_val_loss": 0.5,
        "epochs_completed": 1,
        "early_stopped": False,
        "early_stopping_reason": "max_epochs=1 reached without triggering early stopping",
        "total_runtime_seconds": 1.0,
        "peak_mps_memory": None,
        "checkpoint_hash": "irrelevant",
        "config_hash": "irrelevant",
        "matrix_hash": "irrelevant",
        "protocol_commit": "irrelevant",
        "source_commit": "irrelevant",
        "dataset_artifact_filename": "synthetic.npz",
        "dataset_expected_checksum_md5": "synthetic",
        "dataset_actual_checksum_md5": "synthetic",
        "device": "cpu",
        "dependency_versions": {"torch": "x", "kornia": "x", "medmnist": "x"},
    }


def _valid_metadata_fields():
    return {
        "run_id": "x",
        "attempt_id": 1,
        "block": "A_core_normalization_resolution",
        "dataset": "synthetic",
        "resolution": 4,
        "model": "tiny",
        "normalization": "none",
        "training_policy": "none",
        "seed": 0,
        "frozen_training_settings": {},
        "matrix_hash": "irrelevant",
        "protocol_commit": "irrelevant",
        "source_commit": "irrelevant",
    }


def _valid_history():
    return [
        {
            "epoch": 1,
            "learning_rate": 0.001,
            "train_loss": 0.1,
            "val_loss": 0.1,
            "val_accuracy": 0.9,
            "epoch_runtime_seconds": 0.01,
        }
    ]


def test_expected_on_cpu_restored_on_cpu(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    manifest = persist_and_verify_completion(
        tmp_path,
        history=_valid_history(),
        result_fields=_valid_result_fields(),
        metadata_fields=_valid_metadata_fields(),
        best_state_dict=best_state_dict,
        model_factory=_tiny_model,
    )
    assert manifest["artifacts"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available on this machine")
def test_expected_on_mps_restored_on_cpu_when_mps_available(tmp_path):
    """Tiny synthetic MPS smoke test: initializes/saves/restores a tiny
    synthetic model on MPS. No real data, no real training."""
    device = torch.device("mps")
    model = _tiny_model().to(device)
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}  # tensors on MPS
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    # model_factory intentionally returns a CPU-default model (matching
    # production usage: _build_model() never calls .to(device)) -- the fix
    # must handle this without raising a device-mismatch error.
    manifest = persist_and_verify_completion(
        tmp_path,
        history=_valid_history(),
        result_fields=_valid_result_fields(),
        metadata_fields=_valid_metadata_fields(),
        best_state_dict=best_state_dict,
        model_factory=_tiny_model,
    )
    assert manifest["artifacts"]


def test_map_location_cpu_is_used(tmp_path, monkeypatch):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    captured = {}
    real_load = torch.load

    def spy_load(*args, **kwargs):
        captured["map_location"] = kwargs.get("map_location")
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy_load)

    persist_and_verify_completion(
        tmp_path,
        history=_valid_history(),
        result_fields=_valid_result_fields(),
        metadata_fields=_valid_metadata_fields(),
        best_state_dict=best_state_dict,
        model_factory=_tiny_model,
    )
    assert captured["map_location"] == "cpu"


def test_exact_successful_restore_content_matches(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    persist_and_verify_completion(
        tmp_path,
        history=_valid_history(),
        result_fields=_valid_result_fields(),
        metadata_fields=_valid_metadata_fields(),
        best_state_dict=best_state_dict,
        model_factory=_tiny_model,
    )
    restored = torch.load(tmp_path / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    for key in best_state_dict:
        assert torch.equal(best_state_dict[key].cpu(), restored[key])


def test_key_mismatch_fails_closed(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    # Save a checkpoint with an EXTRA key the reference doesn't have.
    corrupted = dict(best_state_dict)
    corrupted["extra.unexpected_key"] = torch.zeros(3)
    torch.save(corrupted, tmp_path / "best_checkpoint.pt")

    with pytest.raises(PersistenceVerificationError, match="keys do not match"):
        persist_and_verify_completion(
            tmp_path,
            history=_valid_history(),
            result_fields=_valid_result_fields(),
            metadata_fields=_valid_metadata_fields(),
            best_state_dict=best_state_dict,
            model_factory=_tiny_model,
        )


def test_shape_mismatch_fails_closed(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    # Corrupt one tensor's shape directly in the saved file.
    key = next(iter(best_state_dict))
    corrupted = {k: (v.unsqueeze(0) if k == key else v.clone()) for k, v in best_state_dict.items()}
    torch.save(corrupted, tmp_path / "best_checkpoint.pt")

    with pytest.raises(PersistenceVerificationError, match="shape mismatch"):
        persist_and_verify_completion(
            tmp_path,
            history=_valid_history(),
            result_fields=_valid_result_fields(),
            metadata_fields=_valid_metadata_fields(),
            best_state_dict=best_state_dict,
            model_factory=_tiny_model,
        )


def test_dtype_mismatch_fails_closed(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    # Corrupt the saved checkpoint file's dtype directly.
    corrupted = {k: v.double() for k, v in best_state_dict.items()}
    torch.save(corrupted, tmp_path / "best_checkpoint.pt")

    with pytest.raises(PersistenceVerificationError):
        persist_and_verify_completion(
            tmp_path,
            history=_valid_history(),
            result_fields=_valid_result_fields(),
            metadata_fields=_valid_metadata_fields(),
            best_state_dict=best_state_dict,
            model_factory=_tiny_model,
        )


def test_tensor_content_mismatch_fails_closed(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    # Save a DIFFERENT (perturbed) state dict than what's passed as the reference.
    perturbed = {k: v.clone() + 1.0 for k, v in best_state_dict.items()}
    save_checkpoint(perturbed, tmp_path / "best_checkpoint.pt")

    with pytest.raises(PersistenceVerificationError):
        persist_and_verify_completion(
            tmp_path,
            history=_valid_history(),
            result_fields=_valid_result_fields(),
            metadata_fields=_valid_metadata_fields(),
            best_state_dict=best_state_dict,
            model_factory=_tiny_model,
        )


def test_device_neutral_content_hash_equality(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    persist_and_verify_completion(
        tmp_path,
        history=_valid_history(),
        result_fields=_valid_result_fields(),
        metadata_fields=_valid_metadata_fields(),
        best_state_dict=best_state_dict,
        model_factory=_tiny_model,
    )
    restored = torch.load(tmp_path / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    assert hash_state_dict(_to_cpu_detached(best_state_dict)) == hash_state_dict(_to_cpu_detached(restored))


def test_no_mutation_of_original_in_memory_state_dict(tmp_path):
    model = _tiny_model()
    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    before = {k: v.clone() for k, v in best_state_dict.items()}
    before_devices = {k: v.device for k, v in best_state_dict.items()}
    save_checkpoint(best_state_dict, tmp_path / "best_checkpoint.pt")

    persist_and_verify_completion(
        tmp_path,
        history=_valid_history(),
        result_fields=_valid_result_fields(),
        metadata_fields=_valid_metadata_fields(),
        best_state_dict=best_state_dict,
        model_factory=_tiny_model,
    )
    for key in best_state_dict:
        assert torch.equal(best_state_dict[key], before[key])
        assert best_state_dict[key].device == before_devices[key]


def test_to_cpu_detached_never_mutates_input():
    model = _tiny_model()
    sd = model.state_dict()
    before = {k: v.clone() for k, v in sd.items()}
    _ = _to_cpu_detached(sd)
    for key in sd:
        assert torch.equal(sd[key], before[key])


def test_real_attempt_001_and_002_artifacts_untouched():
    """Read-only check: importing/using this module must not touch real
    project artifacts."""
    from pathlib import Path

    attempt_dir = Path("artifacts/confirmatory/A/A-pathmnist-28px-batchnorm-policy-none-s0")
    if not attempt_dir.exists():
        pytest.skip("real canary artifacts not present in this environment")
    import hashlib

    def md5_of(path):
        return hashlib.md5(path.read_bytes()).hexdigest()

    assert md5_of(attempt_dir / "attempt_001" / "best_checkpoint.pt") == "eb7cfb6e23b691f0ffc6a64f23b5a77f"
    assert md5_of(attempt_dir / "attempt_001" / "status.json") == "dad4841fc6add5984f2676804025de34"
