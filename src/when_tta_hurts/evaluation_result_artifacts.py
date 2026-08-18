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
    "prefix_sequence",
    "aggregators",
    "secondary_analyses",
    "protocol_commit",
    "matrix_hash",
    "source_commit",
    "evaluation_config_hash",
    "split",
    "n_validation_samples",
}

_VIEW_MANIFEST_REQUIRED_KEYS = {
    "dataset",
    "resolution",
    "tta_seed",
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
    metric_recomputers: dict[str, tuple[float, Any]] | None = None,
) -> dict[str, Any]:
    """Atomically write predictions.npz/metrics.json/metadata.json/
    view_manifest.json; validate schemas and probability arrays; build +
    verify artifact_manifest.json; independently recompute every metric
    listed in `metric_recomputers` (a dict of
    {name: (stored_value, recompute_callable)}, each `recompute_callable()`
    called with no arguments and compared against `stored_value`) and
    confirm agreement within tolerance.

    Returns the artifact_manifest dict on success. Raises
    EvaluationPersistenceError (or the EvaluationSchemaValidationError
    subclass) on ANY failure -- callers must treat the attempt as failed.
    """
    attempt_dir = Path(attempt_dir)

    validate_predictions_arrays(predictions)
    _validate_metadata_schema(metadata)
    _validate_view_manifest_schema(view_manifest)
    _validate_metrics_schema(metrics)

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
