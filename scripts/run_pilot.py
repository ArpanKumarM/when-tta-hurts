#!/usr/bin/env python3
"""Phase 2A pilot runner: PathMNIST, 28px, SmallCNN/BatchNorm, seed 314159,
validation-only. See docs/pilot_protocol.md for the frozen specification
this script implements -- it must not deviate from that document.

This script has NO test-split access mechanism of any kind: it uses
load_pilot_split() exclusively, which only accepts split in ('train','val').

Usage:
    uv run python scripts/run_pilot.py [--device auto] [--batch-size 256]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from when_tta_hurts.artifacts import (
    atomic_write_json,
    atomic_write_npz,
    hash_state_dict,
    save_checkpoint,
    write_environment_manifest,
)
from when_tta_hurts.config import config_hash, load_config
from when_tta_hurts.data import get_dataset_metadata, load_pilot_split, verify_split_counts
from when_tta_hurts.devices import capture_environment, select_device
from when_tta_hurts.evaluation.tta import aggregate_mean_prefix, compute_ordered_view_logits
from when_tta_hurts.ledger import append_pilot_entry
from when_tta_hurts.metrics import compute_all_metrics, harm_rescue_rates
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything, seeded_generator, worker_init_fn
from when_tta_hurts.training import EarlyStoppingConfig, train_model
from when_tta_hurts.transforms import build_policy

PILOT_CONFIG_PATH = "configs/pilot_pathmnist_28_bn.yaml"
BENCHMARK_ARTIFACT_PATH = Path("artifacts/benchmarks/runtime_benchmark.json")


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _resolve_batch_size(cli_batch_size: int | None, target: int) -> tuple[int, str]:
    """Resolve the batch size to use, per the runtime-benchmark gate. Returns
    (batch_size, note-about-how-it-was-chosen)."""
    if cli_batch_size is not None:
        return cli_batch_size, "explicit --batch-size override"

    if BENCHMARK_ARTIFACT_PATH.exists():
        import json

        with BENCHMARK_ARTIFACT_PATH.open() as f:
            bench = json.load(f)
        chosen = bench.get("chosen_batch_size_per_resolution", {}).get("28")
        if chosen:
            return chosen, f"from runtime benchmark gate ({BENCHMARK_ARTIFACT_PATH})"

    return target, "benchmark artifact unavailable; falling back to config target_batch_size (UNGATED)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(PILOT_CONFIG_PATH)
    cfg_hash = config_hash(cfg)
    seed = cfg["seed"]
    tta_seed = cfg["tta_seed"]

    device = select_device(args.device)
    print(f"=== Phase 2A pilot: PathMNIST/28px/SmallCNN-BatchNorm, seed={seed}, device={device} ===")
    print(f"Config: {PILOT_CONFIG_PATH} (hash={cfg_hash[:12]}...)")

    batch_size, batch_size_note = _resolve_batch_size(args.batch_size, cfg["training"]["target_batch_size"])
    print(f"Batch size: {batch_size} ({batch_size_note})")

    run_id = f"pilot-pathmnist-28-bn-{uuid.uuid4().hex[:8]}"
    artifact_dir = Path("artifacts/pilots") / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifact directory: {artifact_dir}")

    warnings: list[str] = []
    if "UNGATED" in batch_size_note:
        warnings.append(batch_size_note)

    # --- Dataset: split-count verification (metadata + empirical if artifact present) ---
    split_check = verify_split_counts("pathmnist")
    if not split_check["matches"]:
        raise RuntimeError(f"PathMNIST split-count mismatch: {split_check['mismatches']}")
    meta = get_dataset_metadata("pathmnist")

    seed_everything(seed)

    # --- Data loaders: TRAIN and VAL only, via load_pilot_split (no test access mechanism exists) ---
    train_ds = load_pilot_split("pathmnist", split="train", size=cfg["dataset"]["resolution"])
    val_ds = load_pilot_split("pathmnist", split="val", size=cfg["dataset"]["resolution"])

    train_gen = seeded_generator(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=train_gen,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # --- Model ---
    model = build_small_cnn(
        num_classes=cfg["model"]["num_classes"], normalization=cfg["model"]["normalization"]
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != cfg["model"]["expected_param_count"]:
        raise RuntimeError(
            f"SmallCNN param count {n_params} != frozen expected "
            f"{cfg['model']['expected_param_count']} -- architecture drift, stopping."
        )
    initial_state_hash = hash_state_dict(model.state_dict())

    # --- Train ---
    print("Training...")
    t_train_start = time.perf_counter()
    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        max_epochs=cfg["training"]["max_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        early_stopping=EarlyStoppingConfig(
            patience=cfg["training"]["early_stopping"]["patience"],
            min_delta=cfg["training"]["early_stopping"]["min_delta"],
        ),
    )
    training_time = time.perf_counter() - t_train_start
    print(
        f"Training done: {result.epochs_completed} epochs, best_epoch={result.best_epoch}, "
        f"early_stopped={result.early_stopped}, time={training_time:.1f}s"
    )

    best_ckpt_path = artifact_dir / "best_checkpoint.pt"
    best_ckpt_hash = save_checkpoint(result.best_state_dict, best_ckpt_path)

    # --- Clean validation evaluation ---
    print("Evaluating clean validation predictions...")
    model.eval()
    all_clean_logits = []
    all_labels = []
    t_inf_start = time.perf_counter()
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            all_clean_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy().reshape(-1))
    clean_logits = np.concatenate(all_clean_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # --- TTA: deterministic 50-view ordered sequence, computed once, nested prefixes ---
    print("Computing 50-view deterministic TTA sequence on validation set...")
    # Policy stays on CPU: compute_ordered_view_logits runs augmentation on
    # CPU unconditionally due to a measured MPS performance issue (see
    # evaluation/tta.py docstring) -- only the model forward pass uses `device`.
    policy = build_policy(cfg["pilot_tta"]["policy"])
    max_views = max(cfg["pilot_tta"]["view_counts"])

    all_ordered_view_logits = []  # list of [max_views, batch_n, C] arrays, concatenated over batches
    for x, _y in val_loader:
        x = x.to(device)
        view_logits = compute_ordered_view_logits(
            model, x, policy, tta_seed=tta_seed, max_views=max_views, device=device
        )
        all_ordered_view_logits.append(view_logits)
    ordered_view_logits = np.concatenate(all_ordered_view_logits, axis=1)  # [max_views, N, C]
    inference_time = time.perf_counter() - t_inf_start

    peak_mem = {
        "current_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
        "driver_allocated_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
    }

    # --- Metrics ---
    print("Computing metrics...")
    clean_metrics = compute_all_metrics(clean_logits, labels)

    tta_metrics_per_view = {}
    for n in cfg["pilot_tta"]["view_counts"]:
        agg_log_probs = aggregate_mean_prefix(ordered_view_logits, n_views=n)
        m = compute_all_metrics(agg_log_probs, labels)
        hr = harm_rescue_rates(clean_logits, agg_log_probs, labels)
        m["delta_accuracy"] = m["accuracy"] - clean_metrics["accuracy"]
        m.update(hr)
        tta_metrics_per_view[str(n)] = m

    # --- Save artifacts (atomic) ---
    manifest = capture_environment(device)
    write_environment_manifest(manifest, artifact_dir / "env_manifest.json")

    atomic_write_json(cfg, artifact_dir / "resolved_config.json")
    atomic_write_npz(
        {
            "clean_val_logits": clean_logits,
            "val_labels": labels,
            "ordered_tta_val_logits": ordered_view_logits,
        },
        artifact_dir / "predictions.npz",
    )

    metrics_out = {
        "clean": clean_metrics,
        "tta_by_view_count": tta_metrics_per_view,
    }
    atomic_write_json(metrics_out, artifact_dir / "metrics.json")

    run_summary = {
        "run_id": run_id,
        "phase": "pilot",
        "confirmatory": False,
        "split": "validation",
        "seed": seed,
        "tta_seed": tta_seed,
        "config_hash": cfg_hash,
        "git_commit": _git_commit_hash(),
        "dataset": "pathmnist",
        "resolution": cfg["dataset"]["resolution"],
        "dataset_split_check": split_check,
        "dataset_md5_28px": meta.md5_by_resolution.get(28),
        "model": "small_cnn",
        "normalization": "batchnorm",
        "param_count": n_params,
        "initial_state_hash": initial_state_hash,
        "best_checkpoint_hash": best_ckpt_hash,
        "best_checkpoint_path": str(best_ckpt_path),
        "training_history": result.history,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "early_stopped": result.early_stopped,
        "batch_size": batch_size,
        "batch_size_note": batch_size_note,
        "training_time_seconds": training_time,
        "inference_time_seconds": inference_time,
        "peak_mps_memory": peak_mem,
        "warnings": warnings,
    }
    atomic_write_json(run_summary, artifact_dir / "run_summary.json")

    # --- Ledger (append-only) ---
    append_pilot_entry(
        run_id=run_id,
        dataset="pathmnist",
        resolution=cfg["dataset"]["resolution"],
        model="small_cnn",
        normalization="batchnorm",
        seed=seed,
        tta_seed=tta_seed,
        config_hash=cfg_hash,
        git_commit=_git_commit_hash(),
        best_epoch=result.best_epoch,
        epochs_completed=result.epochs_completed,
        early_stopped=result.early_stopped,
        clean_val_accuracy=clean_metrics["accuracy"],
        status="completed",
        artifact_dir=str(artifact_dir),
    )

    print(f"=== Pilot complete. Artifacts: {artifact_dir} ===")
    print(f"Clean val accuracy: {clean_metrics['accuracy']:.4f}")
    for n in cfg["pilot_tta"]["view_counts"]:
        m = tta_metrics_per_view[str(n)]
        print(
            f"  TTA@{n}: acc={m['accuracy']:.4f} delta={m['delta_accuracy']:+.4f} "
            f"harm={m['harm_rate']:.4f} rescue={m['rescue_rate']:.4f}"
        )
    if warnings:
        print(f"WARNINGS: {warnings}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
