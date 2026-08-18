"""Phase 2B.3A Part 2E regression test: the instrumented train_model()
(with per-epoch learning-rate/runtime capture, best_val_loss tracking,
early-stopping-reason strings, and peak-MPS-memory read-only snapshotting
added) must produce BIT-IDENTICAL model weights to a reference
implementation containing only the pre-instrumentation scientific
computation -- proving the added instrumentation is read-only/observational
and never alters training dynamics.

Uses a tiny synthetic dataset/model on CPU, deterministic seeding, no real
data."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from when_tta_hurts.training import EarlyStoppingConfig, train_model


def _make_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(32, 3, 8, 8, generator=g)
    y = torch.randint(0, 4, (32,), generator=g)
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)
    val_loader = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=8, shuffle=False)
    return train_loader, val_loader


def _make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.ReLU(), nn.Linear(16, 4))


def _reference_pre_instrumentation_train(
    model, train_loader, val_loader, device, max_epochs, learning_rate, weight_decay, early_stopping
):
    """Deliberately minimal reimplementation of ONLY the pre-instrumentation
    scientific computation from training.py::train_model -- no learning-rate
    capture, no epoch-runtime capture, no best_val_loss tracking, no
    early_stopping_reason, no peak-memory snapshot. Used as the ground
    truth to diff against the instrumented version."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_accuracy = -1.0
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        total_correct, total_n = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).long().view(-1)
                logits = model(x)
                total_correct += (logits.argmax(dim=-1) == y).sum().item()
                total_n += x.size(0)
        val_accuracy = total_correct / total_n

        if val_accuracy > best_val_accuracy + early_stopping.min_delta:
            best_val_accuracy = val_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping.patience:
                break

    return best_state_dict


def test_instrumented_trainer_matches_reference_weights_exactly():
    device = torch.device("cpu")
    early_stopping = EarlyStoppingConfig(patience=3, min_delta=0.0)

    train_loader_a, val_loader_a = _make_data(seed=0)
    model_a = _make_model()
    result = train_model(
        model_a,
        train_loader_a,
        val_loader_a,
        device,
        max_epochs=10,
        learning_rate=1e-2,
        weight_decay=0.0,
        early_stopping=early_stopping,
    )

    train_loader_b, val_loader_b = _make_data(seed=0)
    model_b = _make_model()
    reference_best_state_dict = _reference_pre_instrumentation_train(
        model_b,
        train_loader_b,
        val_loader_b,
        device,
        max_epochs=10,
        learning_rate=1e-2,
        weight_decay=0.0,
        early_stopping=early_stopping,
    )

    assert reference_best_state_dict is not None
    assert set(result.best_state_dict.keys()) == set(reference_best_state_dict.keys())
    for key in result.best_state_dict:
        assert torch.equal(result.best_state_dict[key], reference_best_state_dict[key]), (
            f"instrumented vs reference weight mismatch at '{key}' -- instrumentation must be "
            f"purely observational and must never alter training dynamics"
        )


def test_instrumentation_fields_present_without_altering_epoch_count():
    """Sanity: the new fields exist and are internally consistent, without
    asserting anything about their numeric values affecting training."""
    device = torch.device("cpu")
    train_loader, val_loader = _make_data(seed=1)
    model = _make_model()
    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=6,
        early_stopping=EarlyStoppingConfig(patience=2, min_delta=0.0),
    )
    assert len(result.history) == result.epochs_completed
    for entry in result.history:
        assert "learning_rate" in entry
        assert "epoch_runtime_seconds" in entry
        assert entry["epoch_runtime_seconds"] >= 0.0
    assert result.best_val_accuracy >= 0.0
    assert result.best_val_loss is not None
    assert result.early_stopping_reason != ""
    assert result.peak_mps_memory is None  # CPU device -- no MPS snapshot
