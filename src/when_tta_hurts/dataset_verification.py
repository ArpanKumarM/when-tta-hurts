"""Reusable official-artifact checksum verification.

Phase 2B.2 audit finding: this logic previously existed ONLY inside
scripts/benchmark_runtime.py (PathMNIST-specific), unavailable to the
production training path. This module generalizes it to any registered
dataset/resolution and is the SINGLE SOURCE OF TRUTH both the benchmark
script and the confirmatory training path use -- fails closed BEFORE any
DataLoader is constructed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from medmnist import INFO

DEFAULT_DATA_ROOT = Path("data/raw")

# {name}.npz / {name}_64.npz / {name}_128.npz / {name}_224.npz, per the
# verified, consistent medmnist.INFO convention across pathmnist,
# bloodmnist, dermamnist (checked directly against the installed package).
_INFO_MD5_KEY_BY_RESOLUTION = {28: "MD5", 64: "MD5_64", 128: "MD5_128", 224: "MD5_224"}


class ArtifactVerificationError(RuntimeError):
    """Raised on any checksum/artifact verification failure -- missing
    file, checksum mismatch, unsupported dataset/resolution, or a resized
    proxy masquerading as native. Always a hard failure, always before any
    DataLoader is constructed."""


@dataclass(frozen=True)
class ArtifactVerification:
    dataset: str
    native_resolution: int
    artifact_path: str
    expected_checksum_md5: str
    actual_checksum_md5: str
    checksum_verified: bool
    resized: bool  # always False for a function that only ever accepts native artifacts


def _artifact_filename(dataset: str, resolution: int) -> str:
    return f"{dataset}.npz" if resolution == 28 else f"{dataset}_{resolution}.npz"


def _md5_of_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def expected_official_checksum(dataset: str, resolution: int) -> str:
    """Metadata-only lookup of the official native-resolution artifact's
    expected MD5, from medmnist.INFO -- no disk I/O, no file read; safe to
    call before any dataset artifact necessarily exists locally (e.g. at
    evaluation-identity-computation time). Raises ArtifactVerificationError
    (fails closed) for an unsupported dataset, an unsupported/unregistered
    resolution, or a missing MD5 key. This is the SINGLE SOURCE OF TRUTH
    for the expected checksum -- verify_official_dataset_artifact() below
    calls this internally rather than re-implementing the lookup, so there
    is exactly one place the (dataset, resolution) -> expected-MD5 mapping
    is defined.
    """
    if dataset not in INFO:
        raise ArtifactVerificationError(f"Unsupported dataset '{dataset}' -- not in medmnist.INFO.")

    md5_key = _INFO_MD5_KEY_BY_RESOLUTION.get(resolution)
    if md5_key is None:
        raise ArtifactVerificationError(
            f"Unsupported/unregistered resolution {resolution} -- no known MD5 key "
            f"convention for it. Registered resolutions: {sorted(_INFO_MD5_KEY_BY_RESOLUTION)}."
        )

    expected_checksum = INFO[dataset].get(md5_key)
    if expected_checksum is None:
        raise ArtifactVerificationError(
            f"medmnist.INFO['{dataset}'] has no '{md5_key}' key -- cannot verify "
            f"resolution={resolution}; failing closed rather than proceeding unverified."
        )
    return expected_checksum


def verify_official_dataset_artifact(
    dataset: str,
    resolution: int,
    root: str | Path = DEFAULT_DATA_ROOT,
) -> ArtifactVerification:
    """Verify the official, NATIVE artifact for (dataset, resolution)
    exists locally and its checksum matches medmnist.INFO exactly. Raises
    ArtifactVerificationError (fails closed) on:
    - unsupported dataset name
    - unsupported/unregistered resolution (no MD5 key in medmnist.INFO)
    - the artifact file not existing locally
    - a checksum mismatch

    This function NEVER accepts a resized-proxy substitute -- there is no
    parameter to pass one, and it only ever checks the file whose name
    corresponds exactly to (dataset, resolution) via the official naming
    convention, so a resized 28px file renamed to look like a 64px file
    would still be caught by the checksum check (resizing changes the
    file's bytes and therefore its MD5, so no resized file can ever match
    an official native-resolution checksum).
    """
    expected_checksum = expected_official_checksum(dataset, resolution)

    filename = _artifact_filename(dataset, resolution)
    path = Path(root) / filename
    if not path.exists():
        raise ArtifactVerificationError(
            f"Expected official artifact {path} does not exist. Refusing to construct "
            f"a training DataLoader without a checksum-verified native artifact."
        )

    actual_checksum = _md5_of_file(path)
    if actual_checksum != expected_checksum:
        raise ArtifactVerificationError(
            f"CHECKSUM MISMATCH for {path}: expected {expected_checksum}, got {actual_checksum}. "
            f"Failing closed -- refusing to train against an unverified/corrupt artifact."
        )

    return ArtifactVerification(
        dataset=dataset,
        native_resolution=resolution,
        artifact_path=str(path),
        expected_checksum_md5=expected_checksum,
        actual_checksum_md5=actual_checksum,
        checksum_verified=True,
        resized=False,
    )
