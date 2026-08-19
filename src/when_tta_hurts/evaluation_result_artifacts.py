"""Phase 2B.4A: required, atomically-persisted, hash-verified completed
validation-evaluation-attempt artifacts. Mirrors result_artifacts.py's
training-attempt pattern, but for a READ-ONLY evaluation of an existing
checkpoint (no checkpoint is written or restored here) -- see
validation_evaluation.py for the orchestration this module is used by.

A validation-evaluation attempt may be marked status="completed" ONLY
after persist_and_verify_evaluation_completion() has:
1. Atomically written predictions.npz, metrics.json, metadata.json,
   view_manifest.json.
2. Validated each JSON file's schema (required keys present).
3. Confirmed every probability array is finite and rows sum to 1 within
   tolerance (a genuine probability distribution, not raw logits/NaNs).
4. Confirmed labels/sample_indices/probability-array lengths agree (no
   silent misalignment).
5. Built + verified artifact_manifest.json (path/size/sha256 for the 4
   content artifacts).
6. Independently recomputed every metric in metrics.json from the stored
   probabilities and confirmed it matches metrics.json's own value (a
   metrics.json written by a different, buggy code path can never be
   marked completed even if internally self-consistent looking).

artifact_manifest.json necessarily excludes itself and status.json, for
the same reason documented in result_artifacts.py.

Never writes or touches images -- predictions.npz contains only labels,
sample indices, and probability arrays.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts.artifacts import atomic_write_json, atomic_write_npz, hash_file

REQUIRED_EVALUATION_ARTIFACTS = (
    "predictions.npz",
    "metrics.json",
    "metadata.json",
    "view_manifest.json",
)
ALL_REQUIRED_EVALUATION_ARTIFACT_FILENAMES = (
    *REQUIRED_EVALUATION_ARTIFACTS,
    "status.json",
    "artifact_manifest.json",
)

_METADATA_REQUIRED_KEYS = {
    "evaluation_id",
    "training_run_id",
    "training_attempt",
    "checkpoint_hash",
    "dataset",
    "resolution",
    "model",
    "normalization",
    "training_policy",
    "seed",
    "tta_seed",
    "tta_seed_config_sha256",
    "tta_seed_freeze_commit",
    "tta_seed_derivation_sha256",
    "prefix_sequence",
    "aggregators",
    "secondary_analyses",
    "protocol_commit",
    "matrix_hash",
    "source_commit",
    "evaluator_fingerprint",
    "evaluator_fingerprint_manifest",
    "evaluation_config_hash",
    "split",
    "n_validation_samples",
}

_VIEW_MANIFEST_REQUIRED_KEYS = {
    "dataset",
    "resolution",
    "tta_seed",
    "tta_seed_config_sha256",
    "tta_seed_freeze_commit",
    "tta_seed_derivation_sha256",
    "n_views",
    "seed_formula",
    "sample_indices",
    "seed_manifest_sha256",
}

_METRICS_REQUIRED_KEYS = {
    "training_run_id",
    "evaluation_config_hash",
    "clean",
    "conditions",
    "latency",
}


class EvaluationPersistenceError(RuntimeError):
    """Raised on ANY failure while persisting/verifying a validation-
    evaluation attempt's required artifacts. Callers must treat this as a
    hard evaluation-attempt failure (status=failed), never completed."""


class EvaluationSchemaValidationError(EvaluationPersistenceError):
    """Raised when a written evaluation artifact is missing required keys."""


def _validate_metadata_schema(metadata: dict) -> None:
    missing = _METADATA_REQUIRED_KEYS - set(metadata.keys())
    if missing:
        raise EvaluationSchemaValidationError(f"metadata.json missing required keys: {sorted(missing)}")
    if metadata["split"] != "validation":
        raise EvaluationSchemaValidationError(
            f"metadata.json split must be 'validation', got {metadata['split']!r}."
        )


def _validate_view_manifest_schema(view_manifest: dict) -> None:
    missing = _VIEW_MANIFEST_REQUIRED_KEYS - set(view_manifest.keys())
    if missing:
        raise EvaluationSchemaValidationError(f"view_manifest.json missing required keys: {sorted(missing)}")


def _validate_metrics_schema(metrics: dict) -> None:
    missing = _METRICS_REQUIRED_KEYS - set(metrics.keys())
    if missing:
        raise EvaluationSchemaValidationError(f"metrics.json missing required keys: {sorted(missing)}")


def _validate_probability_array(name: str, probs: np.ndarray, n_expected: int, n_classes: int) -> None:
    if probs.shape[-2] != n_expected:
        raise EvaluationPersistenceError(
            f"{name}: sample-count mismatch, expected {n_expected}, got {probs.shape[-2]}."
        )
    if probs.shape[-1] != n_classes:
        raise EvaluationPersistenceError(
            f"{name}: class-count mismatch, expected {n_classes}, got {probs.shape[-1]}."
        )
    if not np.all(np.isfinite(probs)):
        raise EvaluationPersistenceError(f"{name}: contains non-finite values (NaN/Inf).")
    if np.any(probs < -1e-6) or np.any(probs > 1 + 1e-6):
        raise EvaluationPersistenceError(
            f"{name}: contains values outside [0, 1] -- not a probability array."
        )
    row_sums = probs.reshape(-1, n_classes).sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise EvaluationPersistenceError(
            f"{name}: rows do not sum to 1 within tolerance (max deviation "
            f"{np.abs(row_sums - 1.0).max():.6f}) -- not normalized probabilities."
        )


def _validate_latency_schema(
    latency: dict[str, Any], expected_prefix_sequence: tuple[int, ...], expected_n_samples: int
) -> None:
    """Verify a persisted latency report (metrics.json's "latency" section,
    see evaluation/latency.py::LatencyReport / validation_evaluation.py::
    compute_evaluation_latency_report()). Raises EvaluationPersistenceError
    on ANY violation -- a malformed/incomplete/non-finite latency report
    must never be marked completed, per
    docs/phase2b_validation_evaluation_engineering_freeze.md sec.1.7."""
    required_top = {"clean_latency_seconds", "n_samples", "by_n"}
    missing = required_top - set(latency.keys())
    if missing:
        raise EvaluationPersistenceError(f"latency report missing required key(s): {sorted(missing)}")

    clean_latency = latency["clean_latency_seconds"]
    if not isinstance(clean_latency, int | float) or not math.isfinite(clean_latency):
        raise EvaluationPersistenceError(f"latency.clean_latency_seconds is not finite: {clean_latency!r}")
    if clean_latency < 0:
        raise EvaluationPersistenceError(f"latency.clean_latency_seconds is negative: {clean_latency}")

    if latency["n_samples"] != expected_n_samples:
        raise EvaluationPersistenceError(
            f"latency.n_samples={latency['n_samples']} does not match the persisted prediction "
            f"sample count {expected_n_samples}."
        )

    by_n = latency["by_n"]
    expected_keys = {str(n) for n in expected_prefix_sequence}
    actual_keys = set(by_n.keys())
    if actual_keys != expected_keys:
        raise EvaluationPersistenceError(
            f"latency.by_n keys {sorted(actual_keys)} do not exactly match the registered N values "
            f"{sorted(expected_keys)}."
        )

    for n in expected_prefix_sequence:
        entry = by_n[str(n)]
        required_entry_keys = {"tta_latency_seconds", "per_sample_latency_seconds", "compute_multiplier"}
        missing_entry = required_entry_keys - set(entry.keys())
        if missing_entry:
            raise EvaluationPersistenceError(f"latency.by_n[{n}] missing key(s): {sorted(missing_entry)}")

        tta_seconds = entry["tta_latency_seconds"]
        per_sample = entry["per_sample_latency_seconds"]
        multiplier = entry["compute_multiplier"]

        for name, value in (
            ("tta_latency_seconds", tta_seconds),
            ("per_sample_latency_seconds", per_sample),
            ("compute_multiplier", multiplier),
        ):
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise EvaluationPersistenceError(f"latency.by_n[{n}].{name} is not finite: {value!r}")
        if tta_seconds < 0:
            raise EvaluationPersistenceError(
                f"latency.by_n[{n}].tta_latency_seconds is negative: {tta_seconds}"
            )
        if per_sample < 0:
            raise EvaluationPersistenceError(
                f"latency.by_n[{n}].per_sample_latency_seconds is negative: {per_sample}"
            )

        expected_per_sample = tta_seconds / expected_n_samples if expected_n_samples > 0 else 0.0
        if not math.isclose(per_sample, expected_per_sample, rel_tol=1e-9, abs_tol=1e-12):
            raise EvaluationPersistenceError(
                f"latency.by_n[{n}].per_sample_latency_seconds={per_sample} does not match "
                f"tta_latency_seconds/n_samples={expected_per_sample}."
            )

        expected_multiplier = (tta_seconds / clean_latency) if clean_latency > 0 else float("inf")
        if not (
            math.isclose(multiplier, expected_multiplier, rel_tol=1e-9, abs_tol=1e-12)
            or (multiplier == float("inf") and expected_multiplier == float("inf"))
        ):
            raise EvaluationPersistenceError(
                f"latency.by_n[{n}].compute_multiplier={multiplier} does not match the frozen formula "
                f"tta_latency_seconds/clean_latency_seconds={expected_multiplier}."
            )


def validate_predictions_arrays(predictions: dict[str, np.ndarray]) -> None:
    """Verify finiteness, normalization, and sample/label/index alignment
    across every array in `predictions` (as would be loaded back from
    predictions.npz). Raises EvaluationPersistenceError on any violation."""
    required = {"labels", "sample_indices", "clean_probs", "view_probs"}
    missing = required - set(predictions.keys())
    if missing:
        raise EvaluationPersistenceError(f"predictions missing required arrays: {sorted(missing)}")

    labels = predictions["labels"]
    sample_indices = predictions["sample_indices"]
    clean_probs = predictions["clean_probs"]
    view_probs = predictions["view_probs"]

    n = labels.shape[0]
    if sample_indices.shape[0] != n:
        raise EvaluationPersistenceError(
            f"sample_indices length {sample_indices.shape[0]} does not match labels length {n}."
        )
    if len(set(sample_indices.tolist())) != n:
        raise EvaluationPersistenceError("sample_indices contains duplicates -- misaligned identity.")

    n_classes = clean_probs.shape[-1]
    _validate_probability_array("clean_probs", clean_probs, n, n_classes)
    _validate_probability_array("view_probs", view_probs, n, n_classes)

    if "bn_adapted_probs" in predictions:
        _validate_probability_array("bn_adapted_probs", predictions["bn_adapted_probs"], n, n_classes)


def build_evaluation_artifact_manifest(
    attempt_dir: str | Path, filenames=REQUIRED_EVALUATION_ARTIFACTS
) -> dict[str, Any]:
    attempt_dir = Path(attempt_dir)
    entries = []
    for filename in filenames:
        path = attempt_dir / filename
        if not path.exists():
            raise EvaluationPersistenceError(f"Cannot build artifact manifest: {path} does not exist.")
        entries.append({"path": filename, "size_bytes": path.stat().st_size, "sha256": hash_file(path)})
    return {"artifacts": entries}


def verify_evaluation_artifact_manifest(attempt_dir: str | Path, manifest: dict[str, Any]) -> None:
    attempt_dir = Path(attempt_dir)
    for entry in manifest["artifacts"]:
        path = attempt_dir / entry["path"]
        if not path.exists():
            raise EvaluationPersistenceError(f"Manifested artifact missing on disk: {path}")
        actual_size = path.stat().st_size
        actual_hash = hash_file(path)
        if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
            raise EvaluationPersistenceError(
                f"Manifest verification failed for {path}: expected size={entry['size_bytes']} "
                f"sha256={entry['sha256']}, got size={actual_size} sha256={actual_hash}."
            )


def persist_and_verify_evaluation_completion(
    attempt_dir: str | Path,
    *,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    view_manifest: dict[str, Any],
    prefix_sequence: tuple[int, ...],
    metric_recomputers: dict[str, tuple[float, Any]] | None = None,
) -> dict[str, Any]:
    """Atomically write predictions.npz/metrics.json/metadata.json/
    view_manifest.json; validate schemas, probability arrays, and the
    required latency-report section (metrics["latency"], validated against
    `prefix_sequence` and the persisted prediction sample count -- see
    _validate_latency_schema()); build + verify artifact_manifest.json;
    independently recompute every metric listed in `metric_recomputers` (a
    dict of {name: (stored_value, recompute_callable)}, each
    `recompute_callable()` called with no arguments and compared against
    `stored_value`) and confirm agreement within tolerance.

    Returns the artifact_manifest dict on success. Raises
    EvaluationPersistenceError (or the EvaluationSchemaValidationError
    subclass) on ANY failure -- callers must treat the attempt as failed.
    A malformed/incomplete/non-finite latency report is exactly such a
    failure -- it can never produce status="completed".
    """
    attempt_dir = Path(attempt_dir)

    validate_predictions_arrays(predictions)
    _validate_metadata_schema(metadata)
    _validate_view_manifest_schema(view_manifest)
    _validate_metrics_schema(metrics)
    _validate_latency_schema(
        metrics["latency"], prefix_sequence, int(np.asarray(predictions["labels"]).shape[0])
    )

    if metric_recomputers:
        for name, (stored_value, recompute) in metric_recomputers.items():
            recomputed = recompute()
            if not np.isclose(recomputed, stored_value, atol=1e-6):
                raise EvaluationPersistenceError(
                    f"Independent metric recomputation mismatch for {name}: "
                    f"stored={stored_value}, recomputed={recomputed}."
                )

    atomic_write_npz(predictions, attempt_dir / "predictions.npz")
    atomic_write_json(metrics, attempt_dir / "metrics.json")
    atomic_write_json(metadata, attempt_dir / "metadata.json")
    atomic_write_json(view_manifest, attempt_dir / "view_manifest.json")

    manifest = build_evaluation_artifact_manifest(attempt_dir, REQUIRED_EVALUATION_ARTIFACTS)
    verify_evaluation_artifact_manifest(attempt_dir, manifest)

    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
    return manifest


def recompute_clean_accuracy(clean_probs: np.ndarray, labels: np.ndarray) -> float:
    """Independent accuracy recomputation directly from stored
    probabilities (not via metrics.py, so a bug shared between the writer
    and metrics.py cannot silently self-validate)."""
    preds = clean_probs.argmax(axis=-1)
    return float((preds == labels).mean())


def recompute_mean_probability_prefix(view_probs: np.ndarray, n_views: int) -> np.ndarray:
    """Independent mean-probability aggregation directly from stored
    per-view probability arrays (already softmaxed -- view_probs stores
    probabilities, not logits, so no further softmax is applied here)."""
    return view_probs[:n_views].mean(axis=0)
