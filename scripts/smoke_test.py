#!/usr/bin/env python3
"""Engineering smoke test for the when_tta_hurts pipeline.

This is NOT an experimental result. It only checks that the pipeline runs
end-to-end: environment capture, dataset metadata/checksum validation,
loading one small batch, three model forward passes, a loss + at most two
optimizer steps, two seeded TTA views, and shape validation. It uses
train/validation data only -- see docs/experimental_protocol.md's test
firewall. No accuracy number is computed or reported by this script.

Usage:
    uv run python scripts/smoke_test.py [--config configs/smoke.yaml]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.artifacts import write_environment_manifest
from when_tta_hurts.config import load_config
from when_tta_hurts.data import get_dataset_metadata, verify_split_counts
from when_tta_hurts.devices import capture_environment, select_device
from when_tta_hurts.models import build_resnet18_small_input
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything
from when_tta_hurts.transforms import build_policy, sample_deterministic_view

BANNER = "Engineering smoke test -- not an experimental result."


def main() -> int:
    parser = argparse.ArgumentParser(description=BANNER)
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args()

    t_start = time.time()
    print(f"=== {BANNER} ===")

    config = load_config(args.config)
    seed_everything(config["seed"])
    device = select_device(config["device"])
    print(f"[1/9] Device resolved: {device}")

    # --- Environment capture ---
    manifest = capture_environment(device)
    manifest_path = config["output_manifest_path"]
    write_environment_manifest(manifest, manifest_path)
    print(f"[2/9] Environment manifest written to {manifest_path}")

    # --- Dataset metadata / split-count / checksum validation (no download here) ---
    dataset_name = config["dataset"]["name"]
    meta = get_dataset_metadata(dataset_name)
    split_check = verify_split_counts(dataset_name)
    print(
        f"[3/9] Dataset metadata for {dataset_name}: "
        f"{meta.n_classes} classes, splits={meta.splits}, "
        f"split_counts_match_expected={split_check['matches']}, "
        f"md5(28px)={meta.md5_by_resolution.get(28)}"
    )
    if not split_check["matches"]:
        print(f"      MISMATCH: {split_check['mismatches']}", file=sys.stderr)
        return 1
    if dataset_name == "dermamnist":
        from when_tta_hurts.data import DERMAMNIST_LICENSE_NOTICE

        print(f"      NOTICE: {DERMAMNIST_LICENSE_NOTICE}")

    # --- Load a small training batch (train/val only, per test firewall) ---
    from when_tta_hurts.data import load_dataset

    ds_size = config["dataset"]["size"]
    train_ds = load_dataset(dataset_name, split="train", size=ds_size, root=config["dataset"]["root"])
    batch_size = config["dataset"]["batch_size"]
    images = []
    labels = []
    for i in range(batch_size):
        img, label = train_ds[i]
        images.append(img)
        labels.append(int(label.item()) if hasattr(label, "item") else int(label))
    x = torch.stack(images).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    print(f"[4/9] Loaded one training batch: x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")

    num_classes = meta.n_classes

    # --- Model forward passes ---
    small_bn = build_small_cnn(num_classes=num_classes, input_size=ds_size, normalization="batchnorm").to(
        device
    )
    small_gn = build_small_cnn(num_classes=num_classes, input_size=ds_size, normalization="groupnorm").to(
        device
    )
    resnet = build_resnet18_small_input(num_classes=num_classes).to(device)

    out_bn = small_bn(x)
    out_gn = small_gn(x)
    out_resnet = resnet(x)
    assert out_bn.shape == (batch_size, num_classes), out_bn.shape
    assert out_gn.shape == (batch_size, num_classes), out_gn.shape
    assert out_resnet.shape == (batch_size, num_classes), out_resnet.shape
    print(
        f"[5/9] Forward passes OK: SmallCNN(BN)={tuple(out_bn.shape)}, "
        f"SmallCNN(GN)={tuple(out_gn.shape)}, ResNet18={tuple(out_resnet.shape)}"
    )

    n_params_bn = sum(p.numel() for p in small_bn.parameters())
    n_params_gn = sum(p.numel() for p in small_gn.parameters())
    n_params_resnet = sum(p.numel() for p in resnet.parameters())
    print(
        f"      Param counts: SmallCNN(BN)={n_params_bn:,}, "
        f"SmallCNN(GN)={n_params_gn:,}, ResNet18={n_params_resnet:,}"
    )

    # --- Loss + at most two optimizer steps (SmallCNN/BatchNorm only, as the
    #     smoke-test model) ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(small_bn.parameters(), lr=config["optimizer"]["lr"])
    max_steps = min(config["optimizer"]["max_steps"], 2)
    for step in range(max_steps):
        optimizer.zero_grad()
        logits = small_bn(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    print(
        f"[6/9] Loss computed and {max_steps} optimizer step(s) taken (loss value not reported as a result)"
    )

    # --- Two seeded TTA views ---
    policy = build_policy(config["tta_smoke"]["policy"], output_size=(ds_size, ds_size)).to(device)
    num_views = min(config["tta_smoke"]["num_views"], 2)
    views = [sample_deterministic_view(x, policy, seed=config["seed"] + i) for i in range(num_views)]
    for v in views:
        assert v.shape == x.shape, v.shape
        assert torch.isfinite(v).all(), "non-finite values in a TTA view"
    view_logits = [small_bn(v) for v in views]
    for vl in view_logits:
        assert vl.shape == (batch_size, num_classes), vl.shape
    print(f"[7/9] Generated {num_views} seeded TTA view(s), shapes and finiteness OK")

    print("[8/9] Logit/output shape validation passed for all models and views")

    # --- Clean teardown ---
    del small_bn, small_gn, resnet, x, y, views, view_logits
    if device.type == "mps":
        torch.mps.empty_cache()
    elapsed = time.time() - t_start
    print(f"[9/9] Clean teardown complete. Elapsed: {elapsed:.2f}s")
    print(f"=== {BANNER} ===")
    print("No accuracy or experimental result is reported by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
