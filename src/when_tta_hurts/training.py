"""Training loop for the Phase 2A pilot: Adam + cosine annealing + early
stopping on validation accuracy, restoring the best checkpoint. Deliberately
minimal -- only what docs/pilot_protocol.md's frozen training spec requires,
not a general-purpose trainer.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    min_delta: float = 0.0


@dataclass
class TrainResult:
    best_state_dict: dict
    best_epoch: int  # 1-indexed
    epochs_completed: int
    early_stopped: bool
    history: list[dict] = field(
        default_factory=list
    )  # per-epoch: {epoch, train_loss, val_loss, val_accuracy}
    training_time_seconds: float = 0.0


def _evaluate_loss_accuracy(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    early_stopping: EarlyStoppingConfig | None = None,
) -> TrainResult:
    if early_stopping is None:
        early_stopping = EarlyStoppingConfig()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_accuracy = -1.0
    best_state_dict = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    t_start = time.perf_counter()
    epochs_completed = 0
    early_stopped = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.size(0)
            train_n += x.size(0)
        scheduler.step()

        train_loss = train_loss_sum / train_n
        val_loss, val_accuracy = _evaluate_loss_accuracy(model, val_loader, device, criterion)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy}
        )
        epochs_completed = epoch

        if val_accuracy > best_val_accuracy + early_stopping.min_delta:
            best_val_accuracy = val_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping.patience:
                early_stopped = True
                break

    training_time_seconds = time.perf_counter() - t_start

    if best_state_dict is None:
        # Should not happen (first epoch always "improves" over -1.0), but
        # guard rather than silently return an untrained model.
        raise RuntimeError("training completed with no best checkpoint recorded")

    model.load_state_dict(best_state_dict)

    return TrainResult(
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        epochs_completed=epochs_completed,
        early_stopped=early_stopped,
        history=history,
        training_time_seconds=training_time_seconds,
    )
