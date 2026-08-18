"""Phase 2B.3A Part 2A: required, atomically-persisted, hash-verified
completed-attempt artifacts.

A confirmatory training attempt may be marked status="completed" (and
therefore canonical-eligible) ONLY after this module's
`persist_and_verify_completion()` has:
1. Atomically written training_history.json, result.json, metadata.json.
2. Validated each file's schema (required keys present).
3. Built artifact_manifest.json (path/size/sha256 for the content
   artifacts) and re-read every covered file to verify its hash matches.
4. Confirmed the best checkpoint on disk restores into a fresh model with
   tensors bit-identical to the in-memory best_state_dict.

artifact_manifest.json necessarily excludes itself (a file cannot contain
its own hash) and status.json (whose final "completed" content is written
by run_identity.finish_attempt() strictly AFTER this function succeeds --
see orchestrator.py's ordering). This is a documented, deliberate
exception, not an oversight.

Any failure at any step raises PersistenceVerificationError -- the caller
must then mark the attempt failed/ineligible, never completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from when_tta_hurts.artifacts import atomic_write_json, hash_file

REQUIRED_COMPLETION_ARTIFACTS = (
    "best_checkpoint.pt",
    "training_history.json",
    "result.json",
    "metadata.json",
)
# status.json and artifact_manifest.json are the two remaining files in the
# six-file completed-attempt set; both are written around (not inside) this
# module -- see module docstring.
ALL_REQUIRED_ARTIFACT_FILENAMES = (
    *REQUIRED_COMPLETION_ARTIFACTS,
    "status.json",
    "artifact_manifest.json",
)

_RESULT_REQUIRED_KEYS = {
    "run_id",
    "attempt_id",
    "best_epoch",
    "best_val_accuracy",
    "best_val_loss",
    "epochs_completed",
    "early_stopped",
    "early_stopping_reason",
    "total_runtime_seconds",
    "peak_mps_memory",
    "checkpoint_hash",
    "config_hash",
    "matrix_hash",
    "protocol_commit",
    "source_commit",
    "dataset_artifact_filename",
    "dataset_expected_checksum_md5",
    "dataset_actual_checksum_md5",
    "device",
    "dependency_versions",
}

_METADATA_REQUIRED_KEYS = {
    "run_id",
    "attempt_id",
    "block",
    "dataset",
    "resolution",
    "model",
    "normalization",
    "training_policy",
    "seed",
    "frozen_training_settings",
    "matrix_hash",
    "protocol_commit",
    "source_commit",
}

_HISTORY_EPOCH_REQUIRED_KEYS = {
    "epoch",
    "learning_rate",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "epoch_runtime_seconds",
}


class PersistenceVerificationError(RuntimeError):
    """Raised on ANY failure while persisting/verifying the required
    completed-attempt artifacts. Callers must treat this as a hard
    training-attempt failure (status=failed), never as completed."""


class SchemaValidationError(PersistenceVerificationError):
    """Raised when a written artifact is missing required keys."""


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    size_bytes: int
    sha256: str


def _validate_history_schema(history: list[dict]) -> None:
    if not isinstance(history, list) or len(history) == 0:
        raise SchemaValidationError("training_history.json must be a non-empty list of per-epoch records.")
    for i, entry in enumerate(history):
        missing = _HISTORY_EPOCH_REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise SchemaValidationError(f"training_history.json entry {i} missing keys: {sorted(missing)}")


def _validate_result_schema(result: dict) -> None:
    missing = _RESULT_REQUIRED_KEYS - set(result.keys())
    if missing:
        raise SchemaValidationError(f"result.json missing required keys: {sorted(missing)}")


def _validate_metadata_schema(metadata: dict) -> None:
    missing = _METADATA_REQUIRED_KEYS - set(metadata.keys())
    if missing:
        raise SchemaValidationError(f"metadata.json missing required keys: {sorted(missing)}")


def build_artifact_manifest(
    attempt_dir: str | Path, filenames=REQUIRED_COMPLETION_ARTIFACTS
) -> dict[str, Any]:
    attempt_dir = Path(attempt_dir)
    entries = []
    for filename in filenames:
        path = attempt_dir / filename
        if not path.exists():
            raise PersistenceVerificationError(f"Cannot build artifact manifest: {path} does not exist.")
        entries.append(
            ManifestEntry(path=filename, size_bytes=path.stat().st_size, sha256=hash_file(path)).__dict__
        )
    return {"artifacts": entries}


def verify_artifact_manifest(attempt_dir: str | Path, manifest: dict[str, Any]) -> None:
    attempt_dir = Path(attempt_dir)
    for entry in manifest["artifacts"]:
        path = attempt_dir / entry["path"]
        if not path.exists():
            raise PersistenceVerificationError(f"Manifested artifact missing on disk: {path}")
        actual_size = path.stat().st_size
        actual_hash = hash_file(path)
        if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
            raise PersistenceVerificationError(
                f"Manifest verification failed for {path}: "
                f"expected size={entry['size_bytes']} sha256={entry['sha256']}, "
                f"got size={actual_size} sha256={actual_hash}."
            )


def persist_and_verify_completion(
    attempt_dir: str | Path,
    *,
    history: list[dict],
    result_fields: dict[str, Any],
    metadata_fields: dict[str, Any],
    best_state_dict: dict,
    model_factory,
) -> dict[str, Any]:
    """Atomically write training_history.json, result.json, metadata.json;
    build+verify artifact_manifest.json (covering the 4 content artifacts
    plus best_checkpoint.pt, which must already exist at attempt_dir);
    confirm the on-disk checkpoint restores to bit-identical weights.

    `model_factory`: zero-arg callable returning a freshly-constructed,
    untrained model of the correct architecture (used only to load and
    compare state_dicts -- never trained, never used for inference).

    Returns the artifact_manifest dict on success. Raises
    PersistenceVerificationError (or the SchemaValidationError subclass)
    on ANY failure -- callers must treat the attempt as failed, not
    completed, in that case.
    """
    attempt_dir = Path(attempt_dir)
    checkpoint_path = attempt_dir / "best_checkpoint.pt"
    if not checkpoint_path.exists():
        raise PersistenceVerificationError(f"{checkpoint_path} does not exist -- cannot persist completion.")

    _validate_history_schema(history)
    _validate_result_schema(result_fields)
    _validate_metadata_schema(metadata_fields)

    atomic_write_json(history, attempt_dir / "training_history.json")
    atomic_write_json(result_fields, attempt_dir / "result.json")
    atomic_write_json(metadata_fields, attempt_dir / "metadata.json")

    manifest = build_artifact_manifest(attempt_dir, REQUIRED_COMPLETION_ARTIFACTS)
    verify_artifact_manifest(attempt_dir, manifest)

    # Confirm the best checkpoint on disk restores bit-identically.
    restored_model = model_factory()
    restored_state_dict = torch.load(checkpoint_path, weights_only=True)
    restored_model.load_state_dict(restored_state_dict)
    reference_keys = set(best_state_dict.keys())
    restored_keys = set(restored_model.state_dict().keys())
    if reference_keys != restored_keys:
        raise PersistenceVerificationError(
            "Restored checkpoint state_dict keys do not match the in-memory best_state_dict keys."
        )
    for key, reference_tensor in best_state_dict.items():
        if not torch.equal(reference_tensor, restored_model.state_dict()[key]):
            raise PersistenceVerificationError(
                f"Restored checkpoint tensor '{key}' does not match the in-memory best checkpoint -- "
                f"refusing to mark this attempt completed."
            )

    atomic_write_json(manifest, attempt_dir / "artifact_manifest.json")
    return manifest
