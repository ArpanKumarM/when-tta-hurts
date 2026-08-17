#!/usr/bin/env python3
"""Same-device (MPS-to-MPS) reproducibility verification -- Phase 1 requirement.

Does NOT require or check bitwise equivalence between MPS and CPU (that is
not guaranteed -- see reproducibility.py). It checks that, on the SAME
device, seeding twice with the same seed produces identical results for:
  - a seeded augmented view
  - initial (pre-training) model parameters
  - clean forward-pass logits before any optimizer step
  - config hashes and cache keys

If any check fails, this script reports exactly which operation diverged
and exits non-zero, rather than silently passing.

Usage:
    uv run python scripts/verify_reproducibility.py [--device mps]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.config import config_hash
from when_tta_hurts.devices import select_device
from when_tta_hurts.evaluation.cache import CacheKey, cache_key_hash
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything
from when_tta_hurts.transforms import build_policy, sample_deterministic_view


def run_once(seed: int, device: torch.device):
    seed_everything(seed)

    model = build_small_cnn(num_classes=9, normalization="batchnorm").to(device)
    initial_params = [p.detach().clone() for p in model.parameters()]

    x = torch.rand(4, 3, 28, 28, device=device)
    with torch.no_grad():
        clean_logits = model(x).detach().clone()

    policy = build_policy("mixed").to(device)
    view = sample_deterministic_view(x, policy, seed=seed)

    cfg = {"seed": seed, "dataset": "pathmnist", "model": "small_cnn_batchnorm"}
    cfg_hash = config_hash(cfg)

    key = CacheKey(
        checkpoint_hash="dummy_checkpoint_hash",
        dataset_version="pathmnist-28-v1",
        split="val",
        policy="mixed",
        seed=seed,
        preprocessing_config_hash=cfg_hash,
    )
    ck_hash = cache_key_hash(key)

    return {
        "initial_params": initial_params,
        "clean_logits": clean_logits,
        "view": view.detach().clone(),
        "config_hash": cfg_hash,
        "cache_key_hash": ck_hash,
    }


def compare(a: dict, b: dict, device: torch.device) -> list[str]:
    failures = []

    for i, (pa, pb) in enumerate(zip(a["initial_params"], b["initial_params"])):
        if not torch.equal(pa, pb):
            failures.append(f"initial model parameter #{i} differs between runs")

    if not torch.equal(a["clean_logits"], b["clean_logits"]):
        failures.append("clean forward-pass logits (pre-optimization) differ between runs")

    if not torch.equal(a["view"], b["view"]):
        failures.append("seeded augmented TTA view differs between runs")

    if a["config_hash"] != b["config_hash"]:
        failures.append(f"config_hash differs: {a['config_hash']} != {b['config_hash']}")

    if a["cache_key_hash"] != b["cache_key_hash"]:
        failures.append(f"cache_key_hash differs: {a['cache_key_hash']} != {b['cache_key_hash']}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Verifying same-device reproducibility on: {device}")

    run_a = run_once(args.seed, device)
    run_b = run_once(args.seed, device)

    failures = compare(run_a, run_b, device)

    if failures:
        print("REPRODUCIBILITY CHECK FAILED. Diverging operations:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("REPRODUCIBILITY CHECK PASSED (same-device, same-seed):")
    print(f"  - {len(run_a['initial_params'])} initial parameter tensors: identical")
    print("  - clean forward-pass logits: identical")
    print("  - seeded augmented TTA view: identical")
    print(f"  - config_hash: identical ({run_a['config_hash'][:12]}...)")
    print(f"  - cache_key_hash: identical ({run_a['cache_key_hash'][:12]}...)")
    print("Note: MPS-vs-CPU bitwise equivalence is NOT checked or guaranteed (see reproducibility.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
