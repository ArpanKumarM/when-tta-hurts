"""Load YAML experiment configs and compute stable content hashes for run identification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    path = Path(path)
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Config at {path} must parse to a mapping, got {type(data)}")
    return data


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 hash of a config dict, independent of key order.

    Used to tag runs/caches so results stay traceable to the exact config
    that produced them (see docs/experimental_protocol.md's test firewall
    and docs/compute_budget.md's evaluation-cache design).
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_hash_short(config: dict[str, Any], length: int = 12) -> str:
    return config_hash(config)[:length]
