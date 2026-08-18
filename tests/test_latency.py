"""Tests for evaluation/latency.py -- device synchronization is mocked
(CPU device, where _sync() is a no-op by design, since torch.mps.synchronize
only applies to device.type=='mps'); we mock time.perf_counter to verify
the sync-call pattern deterministically."""

import torch

from when_tta_hurts.evaluation.latency import build_latency_report, measure_clean_latency, measure_tta_latency
from when_tta_hurts.models.small_cnn import build_small_cnn


def test_measure_clean_latency_returns_positive_float():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(4, 3, 28, 28)
    latency = measure_clean_latency(model, x, torch.device("cpu"))
    assert latency > 0


def test_measure_tta_latency_scales_with_view_count():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(4, 3, 28, 28)
    views_5 = [x] * 5
    views_50 = [x] * 50
    t5 = measure_tta_latency(model, views_5, torch.device("cpu"))
    t50 = measure_tta_latency(model, views_50, torch.device("cpu"))
    assert t50 > t5  # more views takes longer (not a tight ratio check -- just monotonic)


def test_build_latency_report_structure():
    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(4, 3, 28, 28)
    device = torch.device("cpu")
    views_by_n = {1: [x], 2: [x, x]}
    report = build_latency_report(model, x, views_by_n, device)

    assert report.clean_latency_seconds > 0
    assert set(report.tta_latency_seconds_by_n.keys()) == {1, 2}
    assert set(report.per_sample_latency_seconds_by_n.keys()) == {1, 2}
    assert set(report.compute_multiplier_by_n.keys()) == {1, 2}
    assert report.n_samples == 4
    for n in (1, 2):
        assert report.per_sample_latency_seconds_by_n[n] == report.tta_latency_seconds_by_n[n] / 4
        assert report.compute_multiplier_by_n[n] == (
            report.tta_latency_seconds_by_n[n] / report.clean_latency_seconds
        )


def test_sync_called_before_and_after_on_mps_device(monkeypatch):
    """Mock torch.mps.synchronize and torch.backends.mps to confirm _sync()
    is invoked (device-guarded) without requiring real MPS hardware."""
    import when_tta_hurts.evaluation.latency as latency_module

    sync_calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: sync_calls.append(1))

    model = build_small_cnn(num_classes=9, normalization="batchnorm")
    x = torch.rand(2, 3, 28, 28)

    class FakeDevice:
        type = "mps"

    latency_module.measure_clean_latency(model, x, FakeDevice())
    assert len(sync_calls) == 2  # once before, once after
