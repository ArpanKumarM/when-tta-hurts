"""Phase 2B.6A: required, atomically-persisted, hash-verified completed
final-test-evaluation-attempt artifacts.

Structurally separate from evaluation_result_artifacts.py (validation-
only, requires metadata["split"]=="validation") -- this module requires
metadata["split"]=="test" plus additional authorization/fingerprint
bindings, and writes to a disjoint final-test-only artifact namespace
(artifacts/final_test/, see final_test_evaluation.py). Validation and
test artifacts are never mixed: neither module's schema can accept the
other's metadata.

Reuses evaluation_result_artifacts.py's probability-array, latency,
batching, dataset-verification, and BN-adaptation-consistency validators
UNCHANGED -- the only scientific-data difference between a validation and
a final-test evaluation is split=test vs split=validation, never new
math, so those validators are never forked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts.artifacts import atomic_write_json, atomic_write_npz, hash_file
from when_tta_hurts.evaluation_result_artifacts import (
    EvaluationPersistenceError,
    EvaluationSchemaValidationError,
    _validate_batching_schema,
    _validate_bn_adaptation_applicability_consistency,
    _validate_dataset_verification_schema,
    _validate_latency_schema,
    validate_predictions_arrays,
)

REQUIRED_FINAL_TEST_ARTIFACTS: tuple[str, ...] = (
    "predictions.npz",
    "metrics.json",
    "metadata.json",
    "view_manifest.json",
)
ALL_REQUIRED_FINAL_TEST_ARTIFACT_FILENAMES: tuple[str, ...] = (
    *REQUIRED_FINAL_TEST_ARTIFACTS,
    "status.json",
    "artifact_manifest.json",
)

_METADATA_REQUIRED_KEYS = {
    "final_test_evaluation_id",
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
    "dataset_expected_checksum_md5",
    "dataset_verification",
    "batching",
    "metric_input_contract",
    "evaluation_config_hash",
    "split",
    "n_test_samples",
    "statistical_analysis_fingerprint",
    "cross_condition_analysis_fingerprint",
    "final_test_runner_fingerprint",
    "authorization_artifact_sha256",
    "authorization_commit",
    "test_split_accessed",
    "test_predictions_computed",
    "test_metrics_computed",
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


def _validate_metadata_schema(metadata: dict[str, Any]) -> None:
    missing = _METADATA_REQUIRED_KEYS - set(metadata.keys())
    if missing:
        raise EvaluationSchemaValidationError(f"metadata.json missing required keys: {sorted(missing)}")
    if metadata["split"] != "test":
        raise EvaluationSchemaValidationError(
            f"metadata.json split must be 'test' for a final-test artifact, got {metadata['split']!r}."
        )
    if metadata["test_split_accessed"] is not True:
        raise EvaluationSchemaValidationError(
            "metadata.json test_split_accessed must be True for a completed final-test artifact -- "
            "a completed final-test evaluation necessarily accessed the test split."
        )
    if not isinstance(metadata["metric_input_contract"], str) or not metadata["metric_input_contract"]:
        raise EvaluationSchemaValidationError(
            "metadata.json metric_input_contract must be a non-empty string."
        )
    _validate_dataset_verification_schema(metadata)
    _validate_batching_schema(metadata)


def _validate_view_manifest_schema(view_manifest: dict[str, Any]) -> None:
    missing = _VIEW_MANIFEST_REQUIRED_KEYS - set(view_manifest.keys())
    if missing:
        raise EvaluationSchemaValidationError(f"view_manifest.json missing required keys: {sorted(missing)}")


def _validate_metrics_schema(metrics: dict[str, Any]) -> None:
    missing = _METRICS_REQUIRED_KEYS - set(metrics.keys())
    if missing:
        raise EvaluationSchemaValidationError(f"metrics.json missing required keys: {sorted(missing)}")


def build_final_test_artifact_manifest(
    attempt_dir: str | Path, filenames: tuple[str, ...] = REQUIRED_FINAL_TEST_ARTIFACTS
) -> dict[str, Any]:
    attempt_dir = Path(attempt_dir)
    entries = []
    for filename in filenames:
        path = attempt_dir / filename
        if not path.exists():
            raise EvaluationPersistenceError(f"Cannot build artifact manifest: {path} does not exist.")
        entries.append({"path": filename, "size_bytes": path.stat().st_size, "sha256": hash_file(path)})
    return {"artifacts": entries}


def verify_final_test_artifact_manifest(attempt_dir: str | Path, manifest: dict[str, Any]) -> None:
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


def persist_and_verify_final_test_completion(
    attempt_dir: str | Path,
    *,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    view_manifest: dict[str, Any],
    prefix_sequence: tuple[int, ...],
    metric_recomputers: dict[str, tuple[float, Any]] | None = None,
) -> dict[str, Any]:
    """Validate every schema, probability array, and the required latency-
    report section BEFORE any write; then atomically write predictions.npz/
    metrics.json/metadata.json/view_manifest.json; then build + verify
    artifact_manifest.json; then independently recompute every metric in
    `metric_recomputers` and confirm agreement. Returns the artifact
    manifest on success. Raises EvaluationPersistenceError (or the
    EvaluationSchemaValidationError subclass) on ANY failure -- callers
    must treat the attempt as failed. Labels, sample indices, and
    probability evidence are never written to disk until every check
    above has passed in memory."""
    attempt_dir = Path(attempt_dir)

    validate_predictions_arrays(predictions)
    _validate_metadata_schema(metadata)
    _validate_view_manifest_schema(view_manifest)
    _validate_metrics_schema(metrics)
    _validate_bn_adaptation_applicability_consistency(metadata, predictions, metrics)
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

    manifest = build_final_test_artifact_manifest(attempt_dir, REQUIRED_FINAL_TEST_ARTIFACTS)
    verify_final_test_artifact_manifest(attempt_dir, manifest)

    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
    return manifest
