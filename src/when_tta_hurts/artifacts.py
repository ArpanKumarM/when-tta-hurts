"""Write machine-readable artifacts: environment manifests and (later) run
ledger rows. Phase 1 only writes environment manifests; the append-only
run ledger (results/ledger.csv) is a Phase 2 artifact per
docs/experimental_protocol.md's test firewall -- this module provides the
row-append primitive now so Phase 2 doesn't need to invent a new format.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from when_tta_hurts.devices import EnvironmentManifest


def write_environment_manifest(manifest: EnvironmentManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def atomic_write_json(data: Any, path: str | Path) -> None:
    """Write JSON atomically: write to a temp file, then os.replace() into
    place, so a crash mid-write never leaves a partial/corrupt artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    tmp_path.replace(path)


def atomic_write_npz(arrays: dict[str, np.ndarray], path: str | Path) -> None:
    """Write a .npz atomically (same rationale as atomic_write_json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    np.savez(tmp_path, **arrays)
    # np.savez appends .npz if not already present on the tmp path's base name;
    # guard against that by resolving what it actually wrote.
    actual_tmp = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    actual_tmp.replace(path)


def hash_state_dict(state_dict: dict) -> str:
    """Deterministic SHA-256 hash of a model state_dict's tensor contents
    (not just shapes), so two checkpoints can be compared/identified exactly.
    """
    hasher = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def hash_file(path: str | Path) -> str:
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_checkpoint(state_dict: dict, path: str | Path) -> str:
    """Save a model checkpoint and return its content hash (computed from
    the in-memory state_dict, then verified against the hash of the saved
    file's tensor contents by the caller if desired)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state_dict, tmp_path)
    tmp_path.replace(path)
    return hash_state_dict(state_dict)


def append_ledger_row(row: dict[str, Any], path: str | Path) -> None:
    """Append one row to an append-only CSV ledger, creating it with a header
    if it doesn't exist yet. Never rewrites or deletes existing rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
