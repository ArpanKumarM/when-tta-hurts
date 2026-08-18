"""Training loop for Phase 2A pilot AND Phase 2B confirmatory runs:
Adam + cosine annealing + early stopping on validation accuracy, restoring
the best checkpoint. Extended (Phase 2B.2) with an OPTIONAL training-time
augmentation hook (Block B only), OOM detection, and an optional wall-clock
time limit (Block D's 90-minute gate) -- all backward compatible: with no
augmentation_policy and no max_training_seconds, behavior for existing
callers (the Phase 2A pilot) is completely unchanged.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from when_tta_hurts.transforms.policies import sample_deterministic_view


class TrainingOOMError(RuntimeError):
    """Raised when an MPS/CUDA out-of-memory error is detected during
    training, distinct from a generic RuntimeError so callers can route it
    to incident logging specifically."""


class TrainingTimeoutError(RuntimeError):
    """Raised when training exceeds max_training_seconds (checked at epoch
    boundaries) -- used for Block D's 90-minute-per-run stop."""


@dataclass
class EarlyStoppingConfig:
    patience: int = 5
    min_delta: float = 0.0


def _mps_memory_snapshot(device: torch.device) -> dict | None:
    """Read-only introspection, never affects computation. None on non-MPS
    devices (torch.mps.* is only meaningful when device.type == 'mps')."""
    if device.type != "mps":
        return None
    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }


@dataclass
class TrainResult:
    best_state_dict: dict
    best_epoch: int  # 1-indexed
    epochs_completed: int
    early_stopped: bool
    early_stopping_reason: str = ""
    best_val_accuracy: float = -1.0
    best_val_loss: float | None = None
    history: list[dict] = field(default_factory=list)
    # per-epoch: {epoch, learning_rate, train_loss, val_loss, val_accuracy,
    # epoch_runtime_seconds}
    training_time_seconds: float = 0.0
    peak_mps_memory: dict | None = None


def _is_oom_error(e: RuntimeError) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


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
    augmentation_policy: nn.Module | None = None,
    augmentation_seed: int | None = None,
    max_training_seconds: float | None = None,
) -> TrainResult:
    """
    augmentation_policy: if given (Block B only), applied EXACTLY ONCE per
      training sample per step -- one augmented view replaces the clean
      batch for that step, not an additional copy. Augmentation runs on
      CPU (per the measured MPS performance fix in evaluation/tta.py),
      then the single augmented batch is moved to `device`. If None
      (default, matching all prior callers including the Phase 2A pilot),
      training is completely unaugmented -- unchanged behavior.
    augmentation_seed: required if augmentation_policy is given; advances
      by one per training step so augmentation is deterministic and
      reproducible but not identical across steps.
    max_training_seconds: if given, raises TrainingTimeoutError as soon as
      an epoch boundary is crossed after this many seconds have elapsed
      (Block D's 90-minute-per-run stop). None (default) = no limit,
      matching all prior callers.
    """
    if early_stopping is None:
        early_stopping = EarlyStoppingConfig()
    if augmentation_policy is not None and augmentation_seed is None:
        raise ValueError("augmentation_seed is required when augmentation_policy is given")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_accuracy = -1.0
    best_val_loss = None
    best_state_dict = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    t_start = time.perf_counter()
    epochs_completed = 0
    early_stopped = False
    early_stopping_reason = ""
    step_counter = 0

    for epoch in range(1, max_epochs + 1):
        if max_training_seconds is not None and (time.perf_counter() - t_start) > max_training_seconds:
            raise TrainingTimeoutError(
                f"Training exceeded max_training_seconds={max_training_seconds} "
                f"at epoch {epoch} (elapsed={time.perf_counter() - t_start:.1f}s)."
            )

        t_epoch_start = time.perf_counter()
        lr_this_epoch = optimizer.param_groups[0]["lr"]  # read-only, pre-scheduler.step()

        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for x, y in train_loader:
            if augmentation_policy is not None:
                # Exactly once per sample per step: one augmented view
                # replaces x, no double application, CPU augmentation then
                # single device transfer.
                x = sample_deterministic_view(x, augmentation_policy, seed=augmentation_seed + step_counter)
                step_counter += 1
            x = x.to(device)
            y = y.to(device).long().view(-1)
            optimizer.zero_grad()
            try:
                logits = model(x)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}: {loss.item()}")
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if _is_oom_error(e):
                    raise TrainingOOMError(f"OOM during training at epoch {epoch}: {e}") from e
                raise
            train_loss_sum += loss.item() * x.size(0)
            train_n += x.size(0)
        scheduler.step()

        train_loss = train_loss_sum / train_n
        val_loss, val_accuracy = _evaluate_loss_accuracy(model, val_loader, device, criterion)
        epoch_runtime_seconds = time.perf_counter() - t_epoch_start
        history.append(
            {
                "epoch": epoch,
                "learning_rate": lr_this_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch_runtime_seconds": epoch_runtime_seconds,
            }
        )
        epochs_completed = epoch

        if val_accuracy > best_val_accuracy + early_stopping.min_delta:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping.patience:
                early_stopped = True
                early_stopping_reason = (
                    f"no validation-accuracy improvement (> min_delta={early_stopping.min_delta}) "
                    f"for {early_stopping.patience} consecutive epochs"
                )
                break

    if not early_stopped:
        early_stopping_reason = f"max_epochs={max_epochs} reached without triggering early stopping"

    training_time_seconds = time.perf_counter() - t_start
    peak_mps_memory = _mps_memory_snapshot(device)

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
        early_stopping_reason=early_stopping_reason,
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        history=history,
        training_time_seconds=training_time_seconds,
        peak_mps_memory=peak_mps_memory,
    )
