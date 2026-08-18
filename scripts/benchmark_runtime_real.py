#!/usr/bin/env python3
"""Real-data augmentation/architecture runtime benchmark -- NOT a research result.

Measures, using REAL, OFFICIAL PathMNIST 28px training data at batch=256:

  1. DataLoader + host->device transfer + SmallCNN forward/backward, 28px
     (native official data -- condition_type=native_official_real_data).
  2. Same path WITH the registered matched-training augmentation enabled
     (CPU augmentation, per the MPS fix in evaluation/tta.py) -- also
     native official 28px data.
  3. ResNet-18 forward/backward, 28px, real official batches (runtime
     proxy for architecture cost only -- still native official data, but
     "proxy" in the sense that ResNet-18 itself is not the pilot's model).

  4. SmallCNN at 128px ONLY, using REAL 28px PathMNIST images resized up
     for THROUGHPUT MEASUREMENT ONLY (condition_type=resized_proxy_not_official).
     This is a compute/runtime proxy, NOT the scientifically valid
     MedMNIST+ 128px dataset (independently-sourced higher-resolution
     images per docs/data_and_licensing.md, not upsampled 28px images) --
     do not use these numbers as evidence about the actual 128px
     experiments' content, only their runtime. It is NOT a native-resolution
     measurement and NOT equivalent to measuring an official 128px artifact
     (no official 128px artifact is measured anywhere in this project).

The 64px condition previously here has been REMOVED: a genuine native,
official-64px-artifact benchmark now exists in scripts/benchmark_runtime.py
(round 2 of the Phase 2A audit correction) and supersedes any 64px number
this script could produce via resizing. See docs/pilot_audit.md.

Usage:
    uv run python scripts/benchmark_runtime_real.py [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.artifacts import atomic_write_json
from when_tta_hurts.data import load_pilot_split
from when_tta_hurts.devices import capture_environment, select_device
from when_tta_hurts.models.resnet import build_resnet18_small_input
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything
from when_tta_hurts.transforms import build_policy, sample_deterministic_view

BATCH_SIZE = 256
WARMUP_STEPS = 10
MEASURED_STEPS = 30
BENCHMARK_SEED = 0
OUTPUT_PATH = Path("artifacts/benchmarks/runtime_benchmark_real.json")


def _mps_memory_snapshot() -> dict:
    if not torch.backends.mps.is_available():
        return {"current_allocated_bytes": None, "driver_allocated_bytes": None}
    return {
        "current_allocated_bytes": torch.mps.current_allocated_memory(),
        "driver_allocated_bytes": torch.mps.driver_allocated_memory(),
    }


def _real_batch_iterator(loader: DataLoader):
    """Infinite iterator over a real DataLoader (re-starts when exhausted),
    so warmup+measured steps can exceed one epoch's batch count."""
    while True:
        yield from loader


def benchmark_condition(
    name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    condition_type: str,
    resize_to: int | None = None,
    augment_policy=None,
    aug_seed: int = 0,
) -> dict:
    seed_everything(BENCHMARK_SEED)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    it = _real_batch_iterator(loader)

    if device.type == "mps":
        torch.mps.empty_cache()

    aug_time_total = 0.0
    result = {"condition": name, "condition_type": condition_type, "status": "ok", "error": None}

    try:
        model.train()
        for step in range(WARMUP_STEPS + MEASURED_STEPS):
            x, y = next(it)
            if resize_to is not None and x.shape[-1] != resize_to:
                x = F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False)

            t_aug0 = time.perf_counter()
            if augment_policy is not None:
                x = sample_deterministic_view(x, augment_policy, seed=aug_seed + step)
            aug_elapsed = time.perf_counter() - t_aug0

            x = x.to(device)
            y = y.to(device).long().view(-1)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            optimizer.step()

            if step >= WARMUP_STEPS:
                aug_time_total += aug_elapsed
                if step == WARMUP_STEPS:
                    if device.type == "mps":
                        torch.mps.synchronize()
                    t0 = time.perf_counter()

        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        step_time = elapsed / MEASURED_STEPS
        samples_per_sec = BATCH_SIZE / step_time
        avg_aug_time = aug_time_total / MEASURED_STEPS

        result.update(
            {
                "batch_size": BATCH_SIZE,
                "step_time_seconds": step_time,
                "samples_per_second": samples_per_sec,
                "avg_augmentation_time_seconds_per_step": avg_aug_time,
                "augmentation_fraction_of_step_time": (avg_aug_time / step_time) if augment_policy else 0.0,
                "peak_mps_memory": _mps_memory_snapshot(),
            }
        )
    except RuntimeError as e:
        result["status"] = "failed"
        result["error"] = str(e)
        if device.type == "mps":
            torch.mps.empty_cache()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"=== Real-data end-to-end benchmark. Device: {device} ===")
    manifest = capture_environment(device)

    train_ds = load_pilot_split("pathmnist", split="train", size=28)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    results = []

    # 1. SmallCNN, 28px, no augmentation -- native official data
    print("--- 1: SmallCNN 28px, native official data, no augmentation ---")
    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    r = benchmark_condition("small_cnn_28px_no_aug", model, loader, device, "native_official_real_data")
    results.append(r)
    print(f"    {r.get('samples_per_second', r.get('error'))}")

    # 2. SmallCNN, 28px, WITH matched training augmentation (mixed policy, CPU) -- native official data
    print("--- 2: SmallCNN 28px, native official data, matched-policy augmentation enabled ---")
    model2 = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    policy = build_policy("mixed")
    r = benchmark_condition(
        "small_cnn_28px_matched_augmentation",
        model2,
        loader,
        device,
        "native_official_real_data",
        augment_policy=policy,
        aug_seed=12345,
    )
    results.append(r)
    print(f"    {r.get('samples_per_second', r.get('error'))}")

    # 3. ResNet-18, 28px, native official real batches (architecture-cost proxy)
    print("--- 3: ResNet-18 28px, native official data (architecture-cost proxy) ---")
    model3 = build_resnet18_small_input(num_classes=9).to(device)
    r = benchmark_condition("resnet18_28px_no_aug", model3, loader, device, "native_official_real_data")
    results.append(r)
    print(f"    {r.get('samples_per_second', r.get('error'))}")

    # 4. SmallCNN, real 28px images resized to 128px for THROUGHPUT PROXY ONLY.
    # 64px is NOT benchmarked here -- see scripts/benchmark_runtime.py for the
    # native, official-artifact 64px measurement that supersedes any
    # resized-proxy 64px number.
    target_res = 128
    print(f"--- 4: SmallCNN {target_res}px, real 28px images resized (RESIZED PROXY, NOT OFFICIAL) ---")
    model4 = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    r = benchmark_condition(
        f"small_cnn_{target_res}px_resized_proxy_no_aug",
        model4,
        loader,
        device,
        "resized_proxy_not_official",
        resize_to=target_res,
    )
    r["WARNING"] = (
        f"Images are real 28px PathMNIST samples upsampled to {target_res}px via bilinear "
        "interpolation for THROUGHPUT MEASUREMENT ONLY. This is NOT official 128px PathMNIST "
        "data, NOT a native-resolution measurement, and NOT equivalent to measuring an official "
        "128px artifact (no official 128px artifact has been downloaded or measured anywhere in "
        "this project). It is NOT the scientifically valid MedMNIST+ dataset at this resolution "
        "(which uses independently-sourced higher-resolution originals, not upsampled 28px "
        "images -- see docs/data_and_licensing.md). Do not use these images' content for any "
        "accuracy claim; runtime only."
    )
    results.append(r)
    print(f"    {r.get('samples_per_second', r.get('error'))}")

    output = {
        "benchmark_type": "real_data_augmentation_and_architecture_diagnostics",
        "note": (
            "Complements, does not replace, scripts/benchmark_runtime.py (the "
            "canonical native 28px/64px benchmark). Measures augmentation "
            "overhead and ResNet-18 architecture cost on native 28px data, "
            "plus a clearly-labeled 128px resized-proxy diagnostic."
        ),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "batch_size": BATCH_SIZE,
        "environment": manifest.to_dict(),
        "results": results,
    }
    atomic_write_json(output, OUTPUT_PATH)
    print(f"=== Real-data benchmark written to {OUTPUT_PATH} ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
