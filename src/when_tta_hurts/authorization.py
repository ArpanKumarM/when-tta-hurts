"""Final-test-evaluation authorization gate, per docs/phase2b_protocol.md
sec.5 and sec.7. Test-split access is locked behind a COMMITTED artifact
file that does not yet exist -- see AUTHORIZATION_ARTIFACT_PATH below.

A CLI flag or environment variable alone can NEVER unlock the test split:
this module intentionally does not read any CLI arg or env var. The ONLY
input is the committed authorization artifact file's content plus git's
own state (is it tracked, is the tree clean at that path).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from when_tta_hurts.config import load_config

AUTHORIZATION_ARTIFACT_PATH = Path("configs/final_evaluation_authorization.yaml")

_REQUIRED_FIELDS = {
    "status",
    "phase2b_protocol_commit",
    "validation_gated_tta_freeze_commit",
    "matrix_config_hash",
    "approved_test_conditions",
    "approval_timestamp",
}


class AuthorizationError(RuntimeError):
    """Raised for any authorization failure -- missing, untracked, dirty,
    malformed, mismatched, or draft. Always a hard failure, always before
    any dataset construction."""


@dataclass(frozen=True)
class Authorization:
    status: str
    phase2b_protocol_commit: str
    validation_gated_tta_freeze_commit: str
    matrix_config_hash: str
    approved_test_conditions: list
    approval_timestamp: str


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def _is_tracked_and_clean(path: Path) -> bool:
    """True only if `path` is tracked by git AND has no uncommitted
    modifications (working tree matches HEAD for this path exactly)."""
    try:
        tracked = _git("ls-files", "--error-unmatch", str(path))
    except subprocess.CalledProcessError:
        return False
    if not tracked:
        return False
    try:
        diff = _git("diff", "HEAD", "--", str(path))
    except subprocess.CalledProcessError:
        return False
    return diff == ""


def verify_authorization(
    artifact_path: str | Path = AUTHORIZATION_ARTIFACT_PATH,
    expected_matrix_config_hash: str | None = None,
) -> Authorization:
    """Verify the final-evaluation authorization artifact. Raises
    AuthorizationError on ANY problem: missing file, untracked, dirty
    working tree at that path, malformed YAML, missing required fields,
    status != 'approved', or (if expected_matrix_config_hash is given) a
    hash mismatch against the current matrix.

    This function performs NO dataset access of any kind. It must be
    called, and must succeed, strictly before any call to
    `data.py::load_dataset(..., allow_test=True)`.
    """
    path = Path(artifact_path)

    if not path.exists():
        raise AuthorizationError(
            f"Final-evaluation authorization artifact {path} does not exist. "
            f"Test-split access is locked until this is created and committed "
            f"per docs/phase2b_protocol.md sec.5/sec.7."
        )

    if not _is_tracked_and_clean(path):
        raise AuthorizationError(
            f"Final-evaluation authorization artifact {path} exists but is not "
            f"tracked-and-clean in git (untracked, or has uncommitted changes). "
            f"A CLI flag or environment variable cannot substitute for a committed "
            f"authorization artifact."
        )

    try:
        raw = load_config(path)
    except Exception as e:
        raise AuthorizationError(f"Authorization artifact {path} is malformed: {e}") from e

    if not isinstance(raw, dict):
        raise AuthorizationError(f"Authorization artifact {path} must parse to a mapping.")

    missing = _REQUIRED_FIELDS - set(raw.keys())
    if missing:
        raise AuthorizationError(
            f"Authorization artifact {path} missing required field(s): {sorted(missing)}"
        )

    if raw["status"] != "approved":
        raise AuthorizationError(
            f"Authorization artifact status is '{raw['status']}', not 'approved'. Hard failure."
        )

    if expected_matrix_config_hash is not None and raw["matrix_config_hash"] != expected_matrix_config_hash:
        raise AuthorizationError(
            f"Authorization artifact's matrix_config_hash ({raw['matrix_config_hash']}) does not "
            f"match the current matrix's config hash ({expected_matrix_config_hash}). "
            f"The frozen matrix has changed since authorization was granted -- hard failure."
        )

    return Authorization(
        status=raw["status"],
        phase2b_protocol_commit=raw["phase2b_protocol_commit"],
        validation_gated_tta_freeze_commit=raw["validation_gated_tta_freeze_commit"],
        matrix_config_hash=raw["matrix_config_hash"],
        approved_test_conditions=raw["approved_test_conditions"],
        approval_timestamp=raw["approval_timestamp"],
    )
