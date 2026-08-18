import torch
from torch.utils.data import DataLoader, TensorDataset

from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.training import EarlyStoppingConfig, train_model


def _make_loader(n, num_classes=9, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 3, 28, 28, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True, generator=g)


def test_train_one_synthetic_mini_epoch():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(32)
    val_loader = _make_loader(16)

    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=1,
        early_stopping=EarlyStoppingConfig(patience=5),
    )
    assert result.epochs_completed == 1
    assert result.best_epoch == 1
    assert len(result.history) == 1
    assert "val_accuracy" in result.history[0]


def test_best_checkpoint_restoration():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(32)
    val_loader = _make_loader(16)

    result = train_model(model, train_loader, val_loader, device, max_epochs=3)

    # After training, model's live weights must match the returned best_state_dict.
    live_state = model.state_dict()
    for k in result.best_state_dict:
        assert torch.equal(live_state[k], result.best_state_dict[k])


def test_early_stopping_triggers_with_low_patience():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=16)
    val_loader = _make_loader(16, batch_size=16)

    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=30,
        early_stopping=EarlyStoppingConfig(patience=1, min_delta=1.0),  # impossible to satisfy -> stops fast
    )
    assert result.epochs_completed <= 2
    assert result.early_stopped is True


def test_scheduler_runs_full_max_epochs_without_early_stop():
    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    train_loader = _make_loader(16, batch_size=16)
    val_loader = _make_loader(16, batch_size=16)

    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=2,
        early_stopping=EarlyStoppingConfig(patience=100, min_delta=0.0),
    )
    assert result.epochs_completed == 2
    assert result.early_stopped is False


def test_training_raises_on_nonfinite_loss():
    import pytest

    device = torch.device("cpu")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    # >=2 batches per epoch: the huge LR corrupts weights after step 1, so
    # step 2's forward pass produces a non-finite loss and must raise.
    train_loader = _make_loader(16, batch_size=8)
    val_loader = _make_loader(8, batch_size=8)

    with pytest.raises(RuntimeError):
        train_model(model, train_loader, val_loader, device, max_epochs=1, learning_rate=1e30)
