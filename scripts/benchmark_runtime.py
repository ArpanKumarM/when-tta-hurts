#!/usr/bin/env python3
"""CANONICAL runtime/memory benchmark -- NOT a research result.

CORRECTED (Phase 2A audit, round 2): this is now the protocol-compliant
benchmark, using REAL, OFFICIAL PathMNIST training images at their NATIVE
resolutions -- 28x28 (`pathmnist.npz`) and 64x64 (`pathmnist_64.npz`), both
loaded via `load_pilot_split(..., size=28|64)`. Neither resolution is
produced by resizing/interpolating the other; `pathmnist_64.npz` is a
separately-sourced, natively-64px official MedMNIST+ artifact (verified:
`train_images.shape == (89996, 64, 64, 3)`, not derived from the 28px
array at load time -- see docs/data_and_licensing.md for how MedMNIST+
constructs it upstream).

Before timing each resolution, this script independently computes and
verifies the downloaded artifact's MD5 against medmnist.INFO's published
checksums, and FAILS CLOSED (raises, does not proceed to timing) on any
mismatch -- in addition to the `medmnist` package's own checksum-on-download.

The previous version of this script (round 1 of the audit) used
`torch.rand(...)` synthetic tensors for ALL resolutions, including the
64px condition -- meaning the official 64px artifact was downloaded and
checksummed but its pixel content was never actually measured. That defect
is what this rewrite corrects. See docs/pilot_audit.md for the full
chronological disclosure.

Uses train-split batches only. The official test split is never loaded by
this script (only `load_pilot_split`, which has no test-access mechanism,
is used).

Disposable model/optimizer state per condition: optimizer steps are taken
ONLY to produce a realistic forward+backward+step timing; weights are
never saved, evaluated, or reused as a checkpoint.

Usage:
    uv run python scripts/benchmark_runtime.py [--device mps]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.data import load_pilot_split
from when_tta_hurts.dataset_verification import verify_official_dataset_artifact
from when_tta_hurts.devices import capture_environment, select_device
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything

RESOLUTIONS = (28, 64)  # native, official artifacts only -- see module docstring
BATCH_SIZES = (64, 128, 256)
WARMUP_STEPS = 10
MEASURED_STEPS = 30
BENCHMARK_SEED = 0  # arbitrary, fixed; not the pilot seed, this is engineering-only
SAFE_MEMORY_FRACTION = 0.7
MAX_EPOCHS_FOR_ESTIMATE = 30
DATA_ROOT = Path("data/raw")

ARTIFACT_PATH = Path("artifacts/benchmarks/runtime_benchmark.json")

# Checksum verification (Phase 2B.2 audit): moved to the reusable
# src/when_tta_hurts/dataset_verification.py module, which this script and
# the confirmatory training path (orchestrator.py) both now import --
# previously this logic existed ONLY here, unavailable to production
# training. verify_official_artifact() below is now a thin wrapper for
# this script's own reporting shape, delegating all real verification.


def verify_official_artifact(resolution: int) -> dict:
    verification = verify_official_dataset_artifact("pathmnist", resolution, root=DATA_ROOT)
    return {
        "dataset": verification.dataset,
        "split": "train",
        "native_resolution": verification.native_resolution,
        "artifact_filename": Path(verification.artifact_path).name,
        "expected_checksum_md5": verification.expected_checksum_md5,
        "actual_checksum_md5": verification.actual_checksum_md5,
        "checksum_verified": verification.checksum_verified,
        "resized": verification.resized,
    }


def _mps_memory_snapshot() -> dict:
    if not torch.backends.mps.is_available():
        return {
            "current_allocated_bytes": None,
            "driver_allocated_bytes": None,
            "recommended_max_bytes": None,
        }
    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "recommended_max_bytes": torch.mps.recommended_max_memory(),
    }


def _infinite_batches(loader: DataLoader):
    while True:
        yield from loader


def benchmark_cell(resolution: int, batch_size: int, loader: DataLoader, device: torch.device) -> dict:
    """Fresh, disposable model/optimizer per condition. Real training
    batches only. No checkpoint saving, no research metrics, no TTA."""
    seed_everything(BENCHMARK_SEED)
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if device.type == "mps":
        torch.mps.empty_cache()

    it = _infinite_batches(loader)

    result = {
        "resolution": resolution,
        "batch_size": batch_size,
        "condition_type": "native_official_real_data",
        "status": "ok",
        "non_finite_loss": False,
        "oom_or_memory_pressure": False,
        "error": None,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
    }

    try:
        model.train()
        for _ in range(WARMUP_STEPS):
            x, y = next(it)
            x = x.to(device)
            y = y.to(device).long().view(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()

        step_times = []
        for _ in range(MEASURED_STEPS):
            x, y = next(it)
            x = x.to(device)
            y = y.to(device).long().view(-1)

            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                result["non_finite_loss"] = True
                raise RuntimeError(f"non-finite loss encountered: {loss.item()}")
            loss.backward()
            optimizer.step()

            if device.type == "mps":
                torch.mps.synchronize()
            step_times.append(time.perf_counter() - t0)

        peak_after = _mps_memory_snapshot()

        mean_step_time = statistics.mean(step_times)
        median_step_time = statistics.median(step_times)
        samples_per_sec = batch_size / mean_step_time
        steps_per_epoch = -(-89996 // batch_size)  # PathMNIST train split size, ceil div
        estimated_epoch_time = steps_per_epoch * mean_step_time
        estimated_30_epoch_time = estimated_epoch_time * MAX_EPOCHS_FOR_ESTIMATE

        result.update(
            {
                "mean_step_time_seconds": mean_step_time,
                "median_step_time_seconds": median_step_time,
                "samples_per_second": samples_per_sec,
                "steps_per_epoch_estimate": steps_per_epoch,
                "estimated_epoch_time_seconds": estimated_epoch_time,
                "estimated_30_epoch_time_seconds": estimated_30_epoch_time,
                "peak_memory_after_measured_steps": peak_after,
            }
        )

        rec_max = peak_after.get("recommended_max_bytes")
        driver_alloc = peak_after.get("driver_allocated_bytes")
        if rec_max and driver_alloc is not None:
            frac = driver_alloc / rec_max
            result["memory_fraction_of_recommended_max"] = frac
            result["within_safe_memory_fraction"] = frac <= SAFE_MEMORY_FRACTION
        else:
            result["memory_fraction_of_recommended_max"] = None
            result["within_safe_memory_fraction"] = None

    except RuntimeError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        if "out of memory" in str(e).lower() or "mps" in str(e).lower():
            result["oom_or_memory_pressure"] = True
        if device.type == "mps":
            torch.mps.empty_cache()

    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    if args.device.lower() == "mps" and device.type != "mps":
        raise RuntimeError("--device mps was requested but select_device did not return an MPS device.")
    print(f"=== CANONICAL benchmark (native official real data). Device: {device} ===")
    if device.type != "mps":
        print(
            f"WARNING: not running on MPS (device={device}). Per protocol, MPS failures must be "
            f"reported and stopped on, not silently substituted with CPU.",
            file=sys.stderr,
        )

    manifest = capture_environment(device)

    results = []
    artifact_verifications = []
    for resolution in RESOLUTIONS:
        print(f"--- Verifying official artifact for resolution={resolution} ---")
        # load_pilot_split downloads (if needed) + the medmnist package's own
        # download path checksums internally; we ALSO independently verify below.
        _ = load_pilot_split("pathmnist", split="train", size=resolution, root=str(DATA_ROOT))
        verification = verify_official_artifact(resolution)
        artifact_verifications.append(verification)
        print(
            f"    artifact={verification['artifact_filename']} "
            f"checksum_verified={verification['checksum_verified']} "
            f"resized={verification['resized']}"
        )

        train_ds = load_pilot_split("pathmnist", split="train", size=resolution, root=str(DATA_ROOT))

        for batch_size in BATCH_SIZES:
            print(f"--- resolution={resolution} (native) batch_size={batch_size} ---")
            loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            cell = benchmark_cell(resolution, batch_size, loader, device)
            cell["artifact_verification"] = verification
            results.append(cell)
            if cell["status"] == "ok":
                print(
                    f"    OK: {cell['samples_per_second']:.1f} samples/s, "
                    f"mean_step={cell['mean_step_time_seconds'] * 1000:.1f}ms, "
                    f"median_step={cell['median_step_time_seconds'] * 1000:.1f}ms, "
                    f"est_epoch={cell['estimated_epoch_time_seconds']:.1f}s, "
                    f"est_30ep={cell['estimated_30_epoch_time_seconds'] / 60:.1f}min, "
                    f"mem_frac={cell.get('memory_fraction_of_recommended_max')}"
                )
            else:
                print(f"    FAILED: {cell['error']}")

    def cell_ok(c):
        return c["status"] == "ok" and c.get("within_safe_memory_fraction") is not False

    chosen_batch_size = {}
    for resolution in RESOLUTIONS:
        res_cells = [c for c in results if c["resolution"] == resolution]
        ok_256 = next((c for c in res_cells if c["batch_size"] == 256 and cell_ok(c)), None)
        if ok_256:
            chosen_batch_size[resolution] = 256
        else:
            safe_cells = [c for c in res_cells if cell_ok(c)]
            chosen_batch_size[resolution] = max((c["batch_size"] for c in safe_cells), default=None)

    output = {
        "benchmark_type": "canonical_native_official_real_data",
        "resolutions_covered": list(RESOLUTIONS),
        "note": (
            "Uses REAL, OFFICIAL PathMNIST training images at their native "
            "resolution for both 28px (pathmnist.npz) and 64px (pathmnist_64.npz) "
            "-- no torch.rand synthetic tensors, no resizing/interpolation. "
            "Supersedes the round-1 synthetic 64px measurement -- see docs/pilot_audit.md."
        ),
        "artifact_verifications": artifact_verifications,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "environment": manifest.to_dict(),
        "results": results,
        "chosen_batch_size_per_resolution": chosen_batch_size,
    }

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ARTIFACT_PATH.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        import json

        json.dump(output, f, indent=2, default=str)
    tmp_path.replace(ARTIFACT_PATH)

    print(f"=== Benchmark artifact written to {ARTIFACT_PATH} ===")
    print(f"Chosen batch size per resolution: {chosen_batch_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
