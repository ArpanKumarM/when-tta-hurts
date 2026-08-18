"""Native-128px Block D runtime-gate benchmark, per
docs/phase2b_block_d_benchmark_spec.md (frozen operationalization of
docs/phase2b_protocol.md sec.6 / block_d_gate.py's numeric trigger).

This module NEVER trains a real Block D matrix cell, NEVER allocates a
confirmatory attempt, NEVER writes a confirmatory-ledger row, and NEVER
creates an artifacts/confirmatory/D directory -- it only measures
runtime/memory on real native 128px data and feeds the result into
block_d_gate.evaluate_block_d_gate(). Persistence-overhead timing uses a
disposable temporary directory only, never a real attempt directory.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader

from when_tta_hurts.artifacts import atomic_write_json, hash_state_dict, save_checkpoint
from when_tta_hurts.block_d_gate import (
    MAX_END_TO_END_MINUTES_PER_CELL,
    MAX_PESSIMISTIC_TOTAL_HOURS,
    MAX_TRAINING_MINUTES_PER_RUN,
    DatasetBenchmarkRecord,
    evaluate_block_d_gate,
)
from when_tta_hurts.data import get_dataset_metadata, load_pilot_split
from when_tta_hurts.dataset_verification import (
    DEFAULT_DATA_ROOT,
    verify_official_dataset_artifact,
)
from when_tta_hurts.devices import select_device
from when_tta_hurts.matrix import FROZEN_TRAINING_SETTINGS
from when_tta_hurts.models.small_cnn import build_small_cnn
from when_tta_hurts.reproducibility import seed_everything
from when_tta_hurts.result_artifacts import persist_and_verify_completion

SCHEMA_VERSION = "1.0"
BLOCK_D_DATASETS: tuple[str, ...] = ("pathmnist", "bloodmnist")
BLOCK_D_RESOLUTION = 128
BATCH_CANDIDATES: tuple[int, ...] = (64, 128, 256)  # deterministic ascending order
WARMUP_STEPS = 10
MEASURED_STEPS = 30
SAFE_MEMORY_FRACTION = 0.7
BENCHMARK_SEED = 0  # engineering-only, not a confirmatory seed
DEFAULT_OUTPUT_PATH = Path("artifacts/benchmarks/block_d_native_128_benchmark.json")
# Canonical, tracked, permanent decision artifact -- the ONLY thing a future
# Block D training entry point may trust. DEFAULT_OUTPUT_PATH above (the raw,
# detailed timing dump) stays gitignored; this path is the narrow tracked
# exception (see .gitignore).
DEFAULT_DECISION_PATH = Path("artifacts/block_d_gate_decision.json")
FROZEN_PROTOCOL_COMMIT = "ce4c962"
SPEC_COMMIT = "3189580733581e41555b2cccda80366f58e22383"  # docs: freeze Block D benchmark operationalization


class BlockDBenchmarkError(RuntimeError):
    """Base class for all Block D benchmark hard failures -- fail closed,
    never silently degrade to a proxy/synthetic path in production mode."""


class SyntheticInputRejectedError(BlockDBenchmarkError):
    """Raised if production benchmark mode is ever given a non-native,
    resized, or synthetic data source -- there is no configuration flag
    that allows this in production; only test code may inject synthetic
    loaders, via explicit dependency injection of a different function."""


class NonNative128Error(BlockDBenchmarkError):
    """Raised if the verified artifact is not genuinely native 128px."""


class AllBatchCandidatesFailedError(BlockDBenchmarkError):
    """Raised when no batch candidate is safe (OOM, non-finite loss, or
    over the memory fraction) for a dataset -- the entire Block D gate
    fails per the frozen protocol (docs/phase2b_protocol.md sec.6)."""


class IncompleteBenchmarkResultError(BlockDBenchmarkError):
    """Raised when a benchmark result/schema is missing required fields
    before being fed into evaluate_block_d_gate()."""


class BlockDDecisionError(RuntimeError):
    """Raised by the future-training-path decision loader when the
    committed Block D gate decision is absent, untracked, dirty, malformed,
    OMITTED, hash-mismatched, inconsistent with the matrix/specification, or
    the requested batch size differs from the selected one."""


class DecisionAlreadyExistsError(BlockDBenchmarkError):
    """Raised by write_block_d_gate_decision when a decision artifact
    already exists at the target path -- the gate decision is permanent
    once written; overwriting it silently is never allowed."""


class DataLoaderFactory(Protocol):
    def __call__(self, dataset: str, resolution: int, batch_size: int, root: str | Path) -> BlockDLoaders: ...


@dataclass(frozen=True)
class BlockDLoaders:
    """Structurally two loaders only (train/validation) -- no test-loader
    field exists on this type, mirroring TrainValidationLoaders."""

    train_loader: DataLoader
    val_loader: DataLoader
    train_split_size: int
    val_split_size: int


def default_block_d_loader_factory(
    dataset: str, resolution: int, batch_size: int, root: str | Path = DEFAULT_DATA_ROOT
) -> BlockDLoaders:
    """PRODUCTION loader: verifies the official native checksum BEFORE
    constructing any DataLoader, then loads train/val splits ONLY via
    load_pilot_split() (which has no test-split access mechanism of any
    kind). Never accepts a resized proxy -- the artifact filename is
    determined solely by (dataset, resolution)."""
    verification = verify_official_dataset_artifact(dataset, resolution, root=root)
    if verification.resized:
        raise SyntheticInputRejectedError(
            f"{dataset}: verify_official_dataset_artifact reported resized=True -- "
            f"refusing to benchmark a resized proxy in production mode."
        )
    if verification.native_resolution != BLOCK_D_RESOLUTION:
        raise NonNative128Error(
            f"{dataset}: verified native resolution is {verification.native_resolution}, "
            f"not {BLOCK_D_RESOLUTION}."
        )
    train_ds = load_pilot_split(dataset, split="train", size=resolution, root=str(root))
    val_ds = load_pilot_split(dataset, split="val", size=resolution, root=str(root))
    return BlockDLoaders(
        train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        train_split_size=len(train_ds),
        val_split_size=len(val_ds),
    )


def prefetch_block_d_artifacts(root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    """EXPLICIT, separate prefetch operation for the later gate-evaluation
    phase. Downloads (only if absent) and checksum-verifies both Block D
    datasets at native 128px via the same approved official-data path used
    elsewhere in this project (data.load_pilot_split's own download
    mechanism + dataset_verification's independent checksum check).
    NEVER accesses test arrays (load_pilot_split has no test-split
    mechanism). This function is NEVER called implicitly by benchmark or
    plan mode -- it must be invoked explicitly and separately.
    """
    report: dict[str, Any] = {}
    for dataset in BLOCK_D_DATASETS:
        load_pilot_split(dataset, split="train", size=BLOCK_D_RESOLUTION, root=str(root))
        verification = verify_official_dataset_artifact(dataset, BLOCK_D_RESOLUTION, root=root)
        report[dataset] = {
            "artifact_path": verification.artifact_path,
            "expected_checksum_md5": verification.expected_checksum_md5,
            "actual_checksum_md5": verification.actual_checksum_md5,
            "checksum_verified": verification.checksum_verified,
            "resized": verification.resized,
        }
    return report


def plan_block_d_benchmark(matrix_path: str = "configs/experiment_matrix.yaml") -> dict[str, Any]:
    """SIDE-EFFECT-FREE plan mode. Never downloads data, never creates a
    directory, never initializes MPS, never writes a file. Reports the
    two datasets, six cells, expected checksums (read from medmnist.INFO
    metadata only -- no artifact file is opened), candidate batches,
    formulas, and output paths."""
    from medmnist import INFO

    from when_tta_hurts.matrix import parse_and_validate_matrix

    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cells = expanded.cells_by_block["D_conditional_128px"]

    return {
        "datasets": list(BLOCK_D_DATASETS),
        "resolution": BLOCK_D_RESOLUTION,
        "cells": [
            {
                "run_id": c.run_id(),
                "dataset": c.dataset,
                "resolution": c.resolution,
                "model": c.model,
                "normalization": c.normalization,
                "seed": c.seed,
                "n_classes": get_dataset_metadata(c.dataset).n_classes,
            }
            for c in cells
        ],
        "expected_checksums": {ds: INFO[ds]["MD5_128"] for ds in BLOCK_D_DATASETS},
        "batch_candidates": list(BATCH_CANDIDATES),
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "safe_memory_fraction": SAFE_MEMORY_FRACTION,
        "formulas": {
            "projected_training_seconds": ("30 * train_batches_per_epoch * mean_training_step_seconds"),
            "projected_validation_seconds": (
                "30 * validation_batches_per_epoch * mean_validation_step_seconds"
            ),
            "projected_end_to_end_seconds": (
                "setup_seconds + projected_training_seconds + projected_validation_seconds "
                "+ measured_persistence_verification_seconds"
            ),
            "training_gate_seconds": 90 * 60,
            "cell_gate_seconds": 120 * 60,
            "pessimistic_total_hours_ceiling": 24.0,
            "frozen_pessimistic_abc_hours": 3.92,
        },
        "output_path": str(DEFAULT_OUTPUT_PATH),
        "matrix_hash": expanded.source_config_hash,
        "spec_commit": SPEC_COMMIT,
        "protocol_commit": FROZEN_PROTOCOL_COMMIT,
    }


def _mps_memory_snapshot() -> dict[str, int | None]:
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


def _benchmark_training_steps(
    model: nn.Module, loader: DataLoader, device: torch.device, batch_size: int
) -> dict[str, Any]:
    seed_everything(BENCHMARK_SEED)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FROZEN_TRAINING_SETTINGS.learning_rate,
        weight_decay=FROZEN_TRAINING_SETTINGS.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    if device.type == "mps":
        torch.mps.empty_cache()

    it_state = {"it": iter(loader)}

    def _next_batch():
        try:
            return next(it_state["it"])
        except StopIteration:
            it_state["it"] = iter(loader)
            return next(it_state["it"])

    result: dict[str, Any] = {"status": "ok", "oom_occurred": False, "non_finite_loss_occurred": False}
    try:
        model.train()
        for _ in range(WARMUP_STEPS):
            x, y = _next_batch()
            x = x.to(device)
            y = y.to(device).long().view(-1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                result["non_finite_loss_occurred"] = True
                raise RuntimeError(f"non-finite loss during warm-up: {loss.item()}")
            loss.backward()
            optimizer.step()
        if device.type == "mps":
            torch.mps.synchronize()

        step_times = []
        for _ in range(MEASURED_STEPS):
            x, y = _next_batch()
            x = x.to(device)
            y = y.to(device).long().view(-1)
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                result["non_finite_loss_occurred"] = True
                raise RuntimeError(f"non-finite loss during measurement: {loss.item()}")
            loss.backward()
            optimizer.step()
            if device.type == "mps":
                torch.mps.synchronize()
            step_times.append(time.perf_counter() - t0)

        peak_memory = _mps_memory_snapshot()
        mean_step = sum(step_times) / len(step_times)
        median_step = sorted(step_times)[len(step_times) // 2]
        result.update(
            {
                "raw_step_times_seconds": step_times,
                "mean_step_seconds": mean_step,
                "median_step_seconds": median_step,
                "peak_memory": peak_memory,
            }
        )
        rec_max = peak_memory.get("recommended_max_bytes")
        driver_alloc = peak_memory.get("driver_allocated_bytes")
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
            result["oom_occurred"] = True
        if device.type == "mps":
            torch.mps.empty_cache()

    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def _benchmark_validation_steps(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Forward-only, no gradient, no accuracy/prediction value computed or
    stored -- only loss-finiteness (pass/fail) and step duration."""
    criterion = nn.CrossEntropyLoss()
    batches = list(loader)
    n_available = len(batches)
    warmup_n = min(WARMUP_STEPS, n_available)
    measured_target = MEASURED_STEPS

    result: dict[str, Any] = {"status": "ok", "non_finite_loss_occurred": False}
    try:
        model.eval()
        idx = 0
        with torch.no_grad():
            for _ in range(warmup_n):
                x, y = batches[idx % n_available]
                idx += 1
                x = x.to(device)
                y = y.to(device).long().view(-1)
                logits = model(x)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    result["non_finite_loss_occurred"] = True
                    raise RuntimeError(f"non-finite validation loss during warm-up: {loss.item()}")
            if device.type == "mps":
                torch.mps.synchronize()

            step_times = []
            measured = 0
            while measured < measured_target:
                x, y = batches[idx % n_available]
                idx += 1
                x = x.to(device)
                y = y.to(device).long().view(-1)
                if device.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                logits = model(x)
                loss = criterion(logits, y)
                if not torch.isfinite(loss):
                    result["non_finite_loss_occurred"] = True
                    raise RuntimeError(f"non-finite validation loss during measurement: {loss.item()}")
                if device.type == "mps":
                    torch.mps.synchronize()
                step_times.append(time.perf_counter() - t0)
                measured += 1

        mean_step = sum(step_times) / len(step_times)
        median_step = sorted(step_times)[len(step_times) // 2]
        result.update(
            {
                "raw_step_times_seconds": step_times,
                "mean_step_seconds": mean_step,
                "median_step_seconds": median_step,
                "validation_batches_available": n_available,
            }
        )
    except RuntimeError as e:
        result["status"] = "failed"
        result["error"] = str(e)
    return result


def _benchmark_one_dataset(
    dataset: str,
    device_resolver,
    loader_factory: DataLoaderFactory,
    root: str | Path,
    benchmark_root: Path,
) -> dict[str, Any]:
    n_classes = get_dataset_metadata(dataset).n_classes

    verification = verify_official_dataset_artifact(dataset, BLOCK_D_RESOLUTION, root=root)
    if verification.resized:
        raise SyntheticInputRejectedError(f"{dataset}: refusing a resized-proxy artifact in production mode.")
    if verification.native_resolution != BLOCK_D_RESOLUTION:
        raise NonNative128Error(f"{dataset}: artifact is not native {BLOCK_D_RESOLUTION}px.")

    device = device_resolver()

    candidate_results = []
    safe_candidates = []
    for batch_size in BATCH_CANDIDATES:
        t_setup0 = time.perf_counter()
        loaders = loader_factory(dataset, BLOCK_D_RESOLUTION, batch_size, root)
        model = build_small_cnn(num_classes=n_classes, normalization="batchnorm").to(device)
        setup_seconds = time.perf_counter() - t_setup0

        train_result = _benchmark_training_steps(model, loaders.train_loader, device, batch_size)
        entry = {"batch_size": batch_size, "setup_seconds": setup_seconds, "train": train_result}

        if train_result["status"] == "ok":
            val_model = build_small_cnn(num_classes=n_classes, normalization="batchnorm").to(device)
            val_result = _benchmark_validation_steps(val_model, loaders.val_loader, device)
            entry["validation"] = val_result
            is_safe = (
                train_result.get("within_safe_memory_fraction") is not False
                and not train_result["oom_occurred"]
                and not train_result["non_finite_loss_occurred"]
                and val_result["status"] == "ok"
                and not val_result["non_finite_loss_occurred"]
            )
            entry["safe"] = is_safe
            if is_safe:
                safe_candidates.append((batch_size, entry, loaders))
        else:
            entry["safe"] = False

        candidate_results.append(entry)

    if not safe_candidates:
        raise AllBatchCandidatesFailedError(
            f"{dataset}: no batch candidate in {BATCH_CANDIDATES} was safe (OOM, non-finite loss, "
            f"or over {SAFE_MEMORY_FRACTION * 100:.0f}% memory fraction)."
        )

    selected_batch, selected_entry, selected_loaders = max(safe_candidates, key=lambda t: t[0])

    persistence_model = build_small_cnn(num_classes=n_classes, normalization="batchnorm")
    benchmark_dir = benchmark_root / dataset
    persistence_seconds = _measure_persistence_overhead(persistence_model, benchmark_dir, n_classes)

    train_batches_per_epoch = -(-selected_loaders.train_split_size // selected_batch)
    val_batches_per_epoch = -(-selected_loaders.val_split_size // selected_batch)
    mean_train_step = selected_entry["train"]["mean_step_seconds"]
    mean_val_step = selected_entry["validation"]["mean_step_seconds"]

    projected_training_seconds = 30 * train_batches_per_epoch * mean_train_step
    projected_validation_seconds = 30 * val_batches_per_epoch * mean_val_step
    projected_end_to_end_seconds = (
        selected_entry["setup_seconds"]
        + projected_training_seconds
        + projected_validation_seconds
        + persistence_seconds
    )

    return {
        "dataset": dataset,
        "n_classes": n_classes,
        "artifact_path": verification.artifact_path,
        "expected_checksum_md5": verification.expected_checksum_md5,
        "actual_checksum_md5": verification.actual_checksum_md5,
        "resized": verification.resized,
        "device": str(device),
        "candidate_results": candidate_results,
        "selected_batch_size": selected_batch,
        "train_batches_per_epoch": train_batches_per_epoch,
        "validation_batches_per_epoch": val_batches_per_epoch,
        "persistence_verification_seconds": persistence_seconds,
        "projected_training_seconds": projected_training_seconds,
        "projected_validation_seconds": projected_validation_seconds,
        "projected_end_to_end_seconds": projected_end_to_end_seconds,
        "oom_occurred": selected_entry["train"]["oom_occurred"],
        "non_finite_loss_occurred": (
            selected_entry["train"]["non_finite_loss_occurred"]
            or selected_entry["validation"]["non_finite_loss_occurred"]
        ),
    }


def _measure_persistence_overhead(model: nn.Module, benchmark_dir: Path, n_classes: int) -> float:
    """Times the real production persistence/verification code path
    (result_artifacts.persist_and_verify_completion) on a throwaway
    state_dict, using a DISPOSABLE temporary directory -- never a real
    attempt directory, never artifacts/confirmatory/."""
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint_hash = hash_state_dict(state_dict)
    t0 = time.perf_counter()
    save_checkpoint(state_dict, benchmark_dir / "best_checkpoint.pt")
    persist_and_verify_completion(
        benchmark_dir,
        history=[
            {
                "epoch": 1,
                "learning_rate": FROZEN_TRAINING_SETTINGS.learning_rate,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "val_accuracy": 0.0,
                "epoch_runtime_seconds": 0.0,
            }
        ],
        result_fields={
            "run_id": "block_d_benchmark_disposable",
            "attempt_id": 0,
            "best_epoch": 1,
            "best_val_accuracy": 0.0,
            "best_val_loss": 0.0,
            "epochs_completed": 1,
            "early_stopped": False,
            "early_stopping_reason": "",
            "total_runtime_seconds": 0.0,
            "peak_mps_memory": None,
            "checkpoint_hash": checkpoint_hash,
            "config_hash": "disposable",
            "matrix_hash": "disposable",
            "protocol_commit": FROZEN_PROTOCOL_COMMIT,
            "source_commit": "disposable",
            "dataset_artifact_filename": "disposable",
            "dataset_expected_checksum_md5": "disposable",
            "dataset_actual_checksum_md5": "disposable",
            "device": "cpu",
            "dependency_versions": {},
        },
        metadata_fields={
            "run_id": "block_d_benchmark_disposable",
            "attempt_id": 0,
            "block": "D_conditional_128px",
            "dataset": "disposable",
            "resolution": BLOCK_D_RESOLUTION,
            "model": "small_cnn",
            "normalization": "batchnorm",
            "training_policy": "none",
            "seed": BENCHMARK_SEED,
            "frozen_training_settings": {},
            "matrix_hash": "disposable",
            "protocol_commit": FROZEN_PROTOCOL_COMMIT,
            "source_commit": "disposable",
        },
        best_state_dict=state_dict,
        model_factory=lambda: build_small_cnn(num_classes=n_classes, normalization="batchnorm"),
    )
    return time.perf_counter() - t0


def _device_name(entry: dict[str, Any]) -> str:
    raw_device = entry["device"]
    return raw_device.split(":")[0] if ":" in raw_device else raw_device


def run_block_d_benchmark(
    device_resolver=lambda: select_device("mps"),
    loader_factory: DataLoaderFactory = default_block_d_loader_factory,
    root: str | Path = DEFAULT_DATA_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    decision_path: str | Path = DEFAULT_DECISION_PATH,
    matrix_path: str = "configs/experiment_matrix.yaml",
    frozen_pessimistic_abc_hours: float = 3.92,
) -> dict[str, Any]:
    """REAL execution mode. Fails closed (before or during measurement,
    never silently) on: MPS unavailable, artifact absent/checksum-
    mismatched, non-native-128 resolution, non-finite loss, all batch
    candidates unsafe, or an incomplete result schema. NEVER allocates a
    confirmatory attempt, writes a confirmatory-ledger row, or creates
    artifacts/confirmatory/D -- temporary persistence timing uses only a
    disposable directory under output_path's parent.

    Writes TWO artifacts: the detailed raw output at `output_path`
    (gitignored -- large, full of per-step timing), and the canonical,
    tracked, permanent decision at `decision_path` (see
    write_block_d_gate_decision -- refuses to overwrite an existing one).
    """
    from when_tta_hurts.matrix import parse_and_validate_matrix

    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)

    # Fail closed on MPS unavailability BEFORE any dataset access, by
    # resolving the device once up front (device_resolver defaults to
    # select_device("mps"), which raises DeviceUnavailableError -- no
    # CPU-fallback path exists).
    device_resolver()

    benchmark_root = Path(output_path).parent / "_block_d_disposable_persistence"

    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset in BLOCK_D_DATASETS:
        per_dataset[dataset] = _benchmark_one_dataset(
            dataset, device_resolver, loader_factory, root, benchmark_root
        )

    records = {
        ds: DatasetBenchmarkRecord(
            dataset=ds,
            artifact_is_native_128px=(not per_dataset[ds]["resized"]),
            checksum_expected=per_dataset[ds]["expected_checksum_md5"],
            checksum_actual=per_dataset[ds]["actual_checksum_md5"],
            device=_device_name(per_dataset[ds]),
            oom_occurred=per_dataset[ds]["oom_occurred"],
            non_finite_loss_occurred=per_dataset[ds]["non_finite_loss_occurred"],
            projected_training_minutes_per_run=per_dataset[ds]["projected_training_seconds"] / 60,
            projected_end_to_end_minutes_per_cell=per_dataset[ds]["projected_end_to_end_seconds"] / 60,
        )
        for ds in BLOCK_D_DATASETS
    }

    block_d_contribution_seconds = 3 * sum(
        per_dataset[ds]["projected_end_to_end_seconds"] for ds in BLOCK_D_DATASETS
    )
    binding_total_hours = frozen_pessimistic_abc_hours + block_d_contribution_seconds / 3600

    gate_result = evaluate_block_d_gate(
        pathmnist_record=records["pathmnist"],
        bloodmnist_record=records["bloodmnist"],
        pessimistic_total_hours_estimate=binding_total_hours,
    )

    from when_tta_hurts.devices import capture_environment

    manifest = capture_environment(device_resolver())

    output = {
        "schema_version": SCHEMA_VERSION,
        "source_commit": _git_commit_hash(),
        "protocol_commit": FROZEN_PROTOCOL_COMMIT,
        "spec_commit": SPEC_COMMIT,
        "matrix_hash": expanded.source_config_hash,
        "environment": manifest.to_dict(),
        "per_dataset": per_dataset,
        "frozen_pessimistic_abc_hours": frozen_pessimistic_abc_hours,
        "block_d_contribution_seconds": block_d_contribution_seconds,
        "binding_total_hours": binding_total_hours,
        "gate_conditions": list(gate_result.reasons),
        "per_dataset_pass": gate_result.per_dataset_pass,
        "activated": gate_result.activated,
        "final_decision": "INCLUDED" if gate_result.activated else "OMITTED",
    }

    _validate_output_schema(output)
    atomic_write_json(output, output_path)
    raw_output_sha256 = _sha256_file(Path(output_path))

    decision = build_gate_decision(output, raw_output_sha256=raw_output_sha256, raw_output_path=output_path)
    write_block_d_gate_decision(decision, decision_path=decision_path)

    output["decision"] = decision
    output["decision_path"] = str(decision_path)
    return output


_FORBIDDEN_SCIENTIFIC_TERMS = (
    "accuracy",
    "f1",
    "nll",
    "ece",
    "brier",
    "tta_delta",
    "prediction",
    "test_metric",
)


def _find_forbidden_scientific_term(serializable: Any) -> str | None:
    """Word-boundary search for forbidden scientific-metric terms.
    Deliberately NOT a raw substring search: hex checksums/commit hashes/
    SHA-256 digests can contain incidental substrings like 'f1' or 'ece'
    purely by chance, which a raw substring check would misfire on. A
    word-boundary match still catches any real field/value genuinely named
    or containing one of these terms as a standalone token."""
    serialized = json.dumps(serializable).lower()
    for term in _FORBIDDEN_SCIENTIFIC_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", serialized):
            return term
    return None


# Exact (normalized: lowercased, non-alphanumeric stripped) forbidden FIELD
# NAMES -- checked recursively over every dict key in a decision/output,
# independent of _find_forbidden_scientific_term's whole-document value/key
# word-boundary scan. This catches plural/compound key-name variants (e.g.
# 'predictions', 'f1_score') that a word-boundary match on 'prediction'/'f1'
# alone would miss, without flagging unrelated legitimate field names that
# merely contain a similar-looking substring (e.g. 'test_metrics_observed',
# an existing, legitimate boolean ledger field unrelated to any metric
# value) since this checks NORMALIZED EXACT membership, not substrings.
_FORBIDDEN_FIELD_NAME_TOKENS = frozenset(
    {
        "accuracy",
        "accuracies",
        "prediction",
        "predictions",
        "f1",
        "f1score",
        "nll",
        "ece",
        "brier",
        "brierscore",
        "ttadelta",
        "testmetric",
        "testmetrics",
    }
)


def _normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _find_forbidden_field_name(obj: Any) -> str | None:
    """Recursively walk `obj` (dicts and lists only -- JSON has no other
    container types) and return the first dict key whose normalized form
    exactly matches a forbidden field name, or None if none is found."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and _normalize_field_name(key) in _FORBIDDEN_FIELD_NAME_TOKENS:
                return key
            found = _find_forbidden_field_name(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_forbidden_field_name(item)
            if found is not None:
                return found
    return None


def _validate_output_schema(output: dict[str, Any]) -> None:
    required_top = {
        "schema_version",
        "source_commit",
        "protocol_commit",
        "matrix_hash",
        "per_dataset",
        "gate_conditions",
        "per_dataset_pass",
        "activated",
        "final_decision",
    }
    missing = required_top - set(output.keys())
    if missing:
        raise IncompleteBenchmarkResultError(f"Benchmark output missing required top-level fields: {missing}")
    term = _find_forbidden_scientific_term(output)
    if term is not None:
        raise IncompleteBenchmarkResultError(
            f"Benchmark output contains forbidden scientific-metric term '{term}' -- "
            f"this schema must never carry accuracy/prediction/TTA fields."
        )
    field = _find_forbidden_field_name(output)
    if field is not None:
        raise IncompleteBenchmarkResultError(
            f"Benchmark output contains forbidden scientific-metric field name '{field}'."
        )


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


_DECISION_REQUIRED_FIELDS = {
    "schema_version",
    "final_decision",
    "activated",
    "source_commit",
    "protocol_commit",
    "spec_commit",
    "matrix_hash",
    "raw_output_sha256",
    "raw_output_path",
    "per_dataset",
    "gate_condition_booleans",
    "gate_conditions",
    "per_dataset_pass",
    "frozen_pessimistic_abc_hours",
    "block_d_contribution_seconds",
    "binding_total_hours",
    "no_scientific_metric_informed_decision",
    "scientific_metric_confirmation",
}


def build_gate_decision(
    output: dict[str, Any], raw_output_sha256: str, raw_output_path: str | Path
) -> dict[str, Any]:
    """Build the CANONICAL decision from a completed raw benchmark `output`
    (as produced by run_block_d_benchmark before writing). Uses runtime/
    memory/checksum evidence only -- no accuracy/prediction/test-metric
    field is ever read or included."""
    per_dataset = {
        ds: {
            "artifact_path": output["per_dataset"][ds]["artifact_path"],
            "checksum_expected": output["per_dataset"][ds]["expected_checksum_md5"],
            "checksum_actual": output["per_dataset"][ds]["actual_checksum_md5"],
            "resized": output["per_dataset"][ds]["resized"],
            "selected_batch_size": output["per_dataset"][ds]["selected_batch_size"],
            "projected_training_minutes_per_run": (
                output["per_dataset"][ds]["projected_training_seconds"] / 60
            ),
            "projected_end_to_end_minutes_per_cell": (
                output["per_dataset"][ds]["projected_end_to_end_seconds"] / 60
            ),
        }
        for ds in BLOCK_D_DATASETS
    }

    gate_condition_booleans = {
        ds: {
            "native_128px": not output["per_dataset"][ds]["resized"],
            "checksum_match": (
                output["per_dataset"][ds]["expected_checksum_md5"]
                == output["per_dataset"][ds]["actual_checksum_md5"]
            ),
            "device_mps": _device_name(output["per_dataset"][ds]) == "mps",
            "no_oom": not output["per_dataset"][ds]["oom_occurred"],
            "finite_loss": not output["per_dataset"][ds]["non_finite_loss_occurred"],
            "training_within_90_minutes": (
                output["per_dataset"][ds]["projected_training_seconds"] / 60 <= MAX_TRAINING_MINUTES_PER_RUN
            ),
            "end_to_end_within_120_minutes": (
                output["per_dataset"][ds]["projected_end_to_end_seconds"] / 60
                <= MAX_END_TO_END_MINUTES_PER_CELL
            ),
        }
        for ds in BLOCK_D_DATASETS
    }
    gate_condition_booleans["binding_total_within_24_hours"] = (
        output["binding_total_hours"] < MAX_PESSIMISTIC_TOTAL_HOURS
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "final_decision": output["final_decision"],
        "activated": output["activated"],
        "source_commit": output["source_commit"],
        "protocol_commit": output["protocol_commit"],
        "spec_commit": output["spec_commit"],
        "matrix_hash": output["matrix_hash"],
        "raw_output_sha256": raw_output_sha256,
        "raw_output_path": str(raw_output_path),
        "per_dataset": per_dataset,
        "gate_condition_booleans": gate_condition_booleans,
        "gate_conditions": output["gate_conditions"],
        "per_dataset_pass": output["per_dataset_pass"],
        "frozen_pessimistic_abc_hours": output["frozen_pessimistic_abc_hours"],
        "block_d_contribution_seconds": output["block_d_contribution_seconds"],
        "binding_total_hours": output["binding_total_hours"],
        "no_scientific_metric_informed_decision": True,
        "scientific_metric_confirmation": (
            "Decision derived solely from runtime, memory-fraction, checksum, and loss-finiteness "
            "evidence; the benchmark record schema carries no performance-quality field of any kind, "
            "and no held-out split was read at any point in this pipeline."
        ),
    }


def _validate_decision_schema(decision: dict[str, Any]) -> None:
    missing = _DECISION_REQUIRED_FIELDS - set(decision.keys())
    if missing:
        raise IncompleteBenchmarkResultError(f"Block D gate decision missing required fields: {missing}")
    term = _find_forbidden_scientific_term(decision)
    if term is not None:
        raise IncompleteBenchmarkResultError(
            f"Block D gate decision contains forbidden scientific-metric term '{term}'."
        )
    field = _find_forbidden_field_name(decision)
    if field is not None:
        raise IncompleteBenchmarkResultError(
            f"Block D gate decision contains forbidden scientific-metric field name '{field}'."
        )


def write_block_d_gate_decision(
    decision: dict[str, Any], decision_path: str | Path = DEFAULT_DECISION_PATH
) -> None:
    """Write the CANONICAL, tracked, PERMANENT Block D gate decision.
    Refuses to overwrite an existing decision at `decision_path` -- once
    written, a new decision requires an explicit, separately-reviewed
    removal of the old one, never a silent rerun."""
    decision_path = Path(decision_path)
    if decision_path.exists():
        raise DecisionAlreadyExistsError(
            f"{decision_path} already exists -- the Block D gate decision is permanent once written. "
            f"Refusing to overwrite it."
        )
    _validate_decision_schema(decision)
    atomic_write_json(decision, decision_path)


def _default_git_tracked_and_clean(path: Path) -> bool:
    """True iff `path` is tracked by git and has no uncommitted diff
    against HEAD (a clean working tree at exactly that path)."""
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    status = subprocess.run(
        ["git", "status", "--porcelain", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return status.stdout.strip() == ""


def load_and_verify_block_d_decision(
    decision_path: str | Path,
    expected_matrix_hash: str,
    expected_protocol_commit: str = FROZEN_PROTOCOL_COMMIT,
    expected_spec_commit: str = SPEC_COMMIT,
    requested_batch_size: int | None = None,
    dataset: str | None = None,
    require_git_tracked: bool = True,
    git_tracked_and_clean=_default_git_tracked_and_clean,
    raw_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """FUTURE Block D training path consumption point -- not wired into
    any training entry point in this task (Block D training remains
    entirely unlocked/unreachable elsewhere in the codebase). Hard-fails
    if: the decision file is absent, untracked/dirty (unless explicitly
    disabled for testing), malformed, says anything but 'INCLUDED', the
    matrix/protocol/spec hashes don't match, its raw-output hash doesn't
    match the (optionally supplied) raw file, or a requested batch size
    differs from the recorded selected batch for `dataset`.
    """
    path = Path(decision_path)
    if not path.exists():
        raise BlockDDecisionError(f"No Block D gate decision found at {path} -- Block D training is locked.")
    if require_git_tracked and not git_tracked_and_clean(path):
        raise BlockDDecisionError(
            f"{path} is not a clean, git-tracked artifact -- refusing to trust an "
            f"untracked or locally-modified decision."
        )

    try:
        decision = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise BlockDDecisionError(f"{path} is not valid JSON -- malformed decision artifact.") from e

    try:
        _validate_decision_schema(decision)
    except IncompleteBenchmarkResultError as e:
        raise BlockDDecisionError(f"{path} is malformed: {e}") from e

    if raw_output_path is not None:
        raw_path = Path(raw_output_path)
        if not raw_path.exists() or _sha256_file(raw_path) != decision.get("raw_output_sha256"):
            raise BlockDDecisionError(
                "Raw benchmark output is missing or its SHA-256 does not match the decision's "
                "recorded raw_output_sha256 -- refusing to trust a decision without verifiable evidence."
            )

    if decision.get("spec_commit") != expected_spec_commit:
        raise BlockDDecisionError(
            f"Block D gate decision spec_commit {decision.get('spec_commit')} does not match "
            f"expected {expected_spec_commit} -- refusing to trust a stale decision."
        )
    if decision.get("final_decision") != "INCLUDED":
        raise BlockDDecisionError(
            f"Block D gate decision is '{decision.get('final_decision')}', not 'INCLUDED' -- "
            f"Block D training is locked."
        )
    if decision.get("matrix_hash") != expected_matrix_hash:
        raise BlockDDecisionError(
            f"Block D gate decision matrix_hash {decision.get('matrix_hash')} does not match "
            f"expected {expected_matrix_hash} -- refusing to trust a stale decision."
        )
    if decision.get("protocol_commit") != expected_protocol_commit:
        raise BlockDDecisionError(
            f"Block D gate decision protocol_commit {decision.get('protocol_commit')} does not "
            f"match expected {expected_protocol_commit} -- refusing to trust a stale decision."
        )
    if dataset is not None and requested_batch_size is not None:
        selected = decision.get("per_dataset", {}).get(dataset, {}).get("selected_batch_size")
        if selected != requested_batch_size:
            raise BlockDDecisionError(
                f"Requested batch size {requested_batch_size} for {dataset} differs from the "
                f"gate-selected batch size {selected} -- refusing to deviate from the frozen decision."
            )
    return decision
