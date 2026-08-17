"""Write machine-readable artifacts: environment manifests and (later) run
ledger rows. Phase 1 only writes environment manifests; the append-only
run ledger (results/ledger.csv) is a Phase 2 artifact per
docs/experimental_protocol.md's test firewall -- this module provides the
row-append primitive now so Phase 2 doesn't need to invent a new format.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from when_tta_hurts.devices import EnvironmentManifest


def write_environment_manifest(manifest: EnvironmentManifest, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


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
