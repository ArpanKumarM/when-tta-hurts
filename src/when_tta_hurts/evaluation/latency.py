"""Synchronized inference-latency measurement, per docs/phase2b_protocol.md
sec.3's frozen inference-latency specification. Includes device
synchronization (torch.mps.synchronize()) before/after timed regions, so
MPS's async dispatch doesn't understate measured time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class LatencyReport:
    clean_latency_seconds: float
    tta_latency_seconds_by_n: dict[int, float]
    per_sample_latency_seconds_by_n: dict[int, float]
    compute_multiplier_by_n: dict[int, float]
    n_samples: int


@torch.no_grad()
def measure_clean_latency(model: nn.Module, x: torch.Tensor, device: torch.device) -> float:
    model.eval()
    _sync(device)
    t0 = time.perf_counter()
    _ = model(x)
    _sync(device)
    return time.perf_counter() - t0


@torch.no_grad()
def measure_tta_latency(model: nn.Module, views: list[torch.Tensor], device: torch.device) -> float:
    """Total latency for a full TTA pass over `views` (one forward pass per
    view), synchronized before and after the whole sequence."""
    model.eval()
    _sync(device)
    t0 = time.perf_counter()
    for v in views:
        _ = model(v)
    _sync(device)
    return time.perf_counter() - t0


def build_latency_report(
    model: nn.Module,
    x: torch.Tensor,
    ordered_views_by_n: dict[int, list[torch.Tensor]],
    device: torch.device,
) -> LatencyReport:
    """ordered_views_by_n: {N: [view_1, ..., view_N]} -- nested prefixes of
    the same underlying view sequence, per the frozen protocol. Reports
    clean latency, total/per-sample TTA latency at each N, and the compute
    multiplier (TTA latency / clean latency) at each N.
    """
    n_samples = x.shape[0]
    clean_latency = measure_clean_latency(model, x, device)

    tta_latency_by_n: dict[int, float] = {}
    per_sample_by_n: dict[int, float] = {}
    multiplier_by_n: dict[int, float] = {}

    for n, views in ordered_views_by_n.items():
        total = measure_tta_latency(model, views, device)
        tta_latency_by_n[n] = total
        per_sample_by_n[n] = total / n_samples if n_samples > 0 else 0.0
        multiplier_by_n[n] = (total / clean_latency) if clean_latency > 0 else float("inf")

    return LatencyReport(
        clean_latency_seconds=clean_latency,
        tta_latency_seconds_by_n=tta_latency_by_n,
        per_sample_latency_seconds_by_n=per_sample_by_n,
        compute_multiplier_by_n=multiplier_by_n,
        n_samples=n_samples,
    )
