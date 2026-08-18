"""Tests for training.py's Phase 2B.2 extensions: augmentation hook, OOM
detection, time limit. All use synthetic tensors on CPU."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.training import (
    EarlyStoppingConfig,
    TrainingOOMError,
    TrainingTimeoutError,
    train_model,
)
from when_tta_hurts.transforms.policies import build_policy


def _make_loader(n, num_classes=9, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 28, 28, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=g)


def test_no_augmentation_behavior_unchanged():
    """Pilot's existing call signature (no augmentation_policy) must
    produce identical behavior to before this extension."""
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    result = train_model(model, train_loader, val_loader, device, max_epochs=2)
    assert result.epochs_completed == 2


def test_augmentation_hook_requires_seed_when_policy_given():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    policy = build_policy("mixed")
    with pytest.raises(ValueError, match="augmentation_seed"):
        train_model(model, train_loader, val_loader, device, max_epochs=1, augmentation_policy=policy)


def test_augmentation_applied_exactly_once_per_step(monkeypatch):
    """Spy on sample_deterministic_view to confirm it's called exactly
    once per training step (batch), not zero or multiple times."""
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=8)  # 2 steps/epoch
    val_loader = _make_loader(8, batch_size=8)
    policy = build_policy("mixed")

    call_count = 0
    import when_tta_hurts.training as training_module

    real_fn = training_module.sample_deterministic_view

    def spy(x, pol, seed):
        nonlocal call_count
        call_count += 1
        return real_fn(x, pol, seed)

    monkeypatch.setattr(training_module, "sample_deterministic_view", spy)

    train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=1,
        augmentation_policy=policy,
        augmentation_seed=0,
    )
    assert call_count == 2  # exactly 2 steps in 1 epoch, one augmentation call each


def test_augmentation_seeds_are_distinct_per_step(monkeypatch):
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(24, batch_size=8)  # 3 steps/epoch
    val_loader = _make_loader(8, batch_size=8)
    policy = build_policy("mixed")

    seeds_used = []
    import when_tta_hurts.training as training_module

    real_fn = training_module.sample_deterministic_view

    def spy(x, pol, seed):
        seeds_used.append(seed)
        return real_fn(x, pol, seed)

    monkeypatch.setattr(training_module, "sample_deterministic_view", spy)

    train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=1,
        augmentation_policy=policy,
        augmentation_seed=100,
    )
    assert seeds_used == [100, 101, 102]  # distinct, incrementing


def test_non_finite_loss_raises():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    with pytest.raises((RuntimeError,)):
        train_model(model, train_loader, val_loader, device, max_epochs=1, learning_rate=1e30)


def test_oom_error_is_distinct_type(monkeypatch):
    """Simulate an OOM by monkeypatching the model's forward to raise the
    characteristic RuntimeError message -- confirms it's caught and
    re-raised as TrainingOOMError, not a generic RuntimeError."""
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(8, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)

    def raise_oom(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory (allocated 100 MB)")

    monkeypatch.setattr(model, "forward", raise_oom)

    with pytest.raises(TrainingOOMError):
        train_model(model, train_loader, val_loader, device, max_epochs=1)


def test_max_training_seconds_raises_timeout():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    with pytest.raises(TrainingTimeoutError):
        train_model(
            model,
            train_loader,
            val_loader,
            device,
            max_epochs=1000,
            early_stopping=EarlyStoppingConfig(patience=1000),
            max_training_seconds=0.0,  # expires immediately at the epoch-2 boundary check
        )


def test_no_time_limit_by_default():
    """max_training_seconds=None (default) must not raise regardless of
    how long training takes -- confirms backward compatibility."""
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(8, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)
    result = train_model(model, train_loader, val_loader, device, max_epochs=2)
    assert result.epochs_completed == 2
