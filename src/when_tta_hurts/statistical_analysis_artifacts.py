"""Persistence/schema-validation layer for statistical-analysis results,
mirroring evaluation_result_artifacts.py's exact discipline: atomic
writes, schema validation before persistence, artifact-manifest
build/verify, fail-closed on non-finite values or schema mismatch. Never
marks an attempt status=completed until every check below passes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import atomic_write_json, hash_file

REQUIRED_ANALYSIS_ARTIFACTS: tuple[str, ...] = ("analysis_result.json",)

_REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "family",
        "analysis_id",
        "analysis_fingerprint",
        "current_evaluator_fingerprint",
        "cells",
        "per_cell_statistics",
        "multiplicity",
        "status",
        "test_split_accessed",
    }
)

_KNOWN_FAMILIES: frozenset[str] = frozenset({"H1", "H2", "H3", "BLOCK_C"})


class AnalysisPersistenceError(RuntimeError):
    """Raised on any analysis-artifact persistence/verification failure.
    Callers must treat the attempt as failed -- never partially
    'completed'."""


class AnalysisSchemaValidationError(AnalysisPersistenceError):
    """Raised when an analysis result fails schema validation."""


def _assert_finite(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_finite(v, f"{path}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _assert_finite(v, f"{path}[{i}]")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisSchemaValidationError(f"Non-finite value at {path}: {value!r}.")


def validate_analysis_result_schema(result: dict[str, Any]) -> None:
    """Fail-closed schema validation for a persisted analysis result.
    Checks required keys, family membership, test-split isolation
    (test_split_accessed must be exactly False -- this module has no path
    that could ever set it True), status token, and full-tree
    finiteness."""
    missing = _REQUIRED_TOP_LEVEL_KEYS - set(result.keys())
    if missing:
        raise AnalysisSchemaValidationError(f"Analysis result missing required keys: {sorted(missing)}.")

    if result["family"] not in _KNOWN_FAMILIES:
        raise AnalysisSchemaValidationError(
            f"Unknown analysis family: {result['family']!r}. Known: {sorted(_KNOWN_FAMILIES)}."
        )

    if result["test_split_accessed"] is not False:
        raise AnalysisSchemaValidationError(
            "test_split_accessed must be exactly False -- refusing to persist an analysis result "
            "that claims (or fails to disclaim) test-split access."
        )

    if result["status"] != "completed":
        raise AnalysisSchemaValidationError(
            f"Only status='completed' results may be persisted, got {result['status']!r}."
        )

    if not isinstance(result["cells"], list) or len(result["cells"]) == 0:
        raise AnalysisSchemaValidationError("'cells' must be a non-empty list.")

    if not isinstance(result["per_cell_statistics"], dict) or len(result["per_cell_statistics"]) == 0:
        raise AnalysisSchemaValidationError("'per_cell_statistics' must be a non-empty dict.")

    _assert_finite(result["per_cell_statistics"], "per_cell_statistics")
    _assert_finite(result["multiplicity"], "multiplicity")


def build_analysis_artifact_manifest(
    attempt_dir: str | Path, filenames: tuple[str, ...] = REQUIRED_ANALYSIS_ARTIFACTS
) -> dict[str, Any]:
    attempt_dir = Path(attempt_dir)
    entries = []
    for filename in filenames:
        path = attempt_dir / filename
        if not path.exists():
            raise AnalysisPersistenceError(f"Cannot build artifact manifest: {path} does not exist.")
        entries.append({"path": filename, "size_bytes": path.stat().st_size, "sha256": hash_file(path)})
    return {"artifacts": entries}


def verify_analysis_artifact_manifest(attempt_dir: str | Path, manifest: dict[str, Any]) -> None:
    attempt_dir = Path(attempt_dir)
    for entry in manifest["artifacts"]:
        path = attempt_dir / entry["path"]
        if not path.exists():
            raise AnalysisPersistenceError(f"Manifested artifact missing on disk: {path}")
        actual_size = path.stat().st_size
        actual_hash = hash_file(path)
        if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
            raise AnalysisPersistenceError(
                f"Manifest verification failed for {path}: expected size={entry['size_bytes']} "
                f"sha256={entry['sha256']}, got size={actual_size} sha256={actual_hash}."
            )


def persist_and_verify_analysis_completion(
    attempt_dir: str | Path,
    *,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Validate schema, write analysis_result.json atomically, build +
    verify the artifact manifest, and write it. Raises
    AnalysisPersistenceError/AnalysisSchemaValidationError on ANY failure
    -- callers must treat the attempt as failed; status='completed' is
    never reachable through a partial write."""
    attempt_dir = Path(attempt_dir)
    validate_analysis_result_schema(result)

    atomic_write_json(result, attempt_dir / "analysis_result.json")

    manifest = build_analysis_artifact_manifest(attempt_dir, REQUIRED_ANALYSIS_ARTIFACTS)
    verify_analysis_artifact_manifest(attempt_dir, manifest)

    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
    return manifest
