"""Phase 2B.6A: final-test-evaluation authorization gate for the 39-cell
Phase 2B confirmatory matrix.

Distinct from, and NOT interchangeable with, authorization.py's Phase
2B.2 Validation-Gated TTA (H4) gate -- that governs a different, still-
draft, unrelated future algorithm's own eventual test pass, gated by its
own artifact at a different path (configs/final_evaluation_authorization.yaml).
Each gate is self-contained and guards only its own call site; neither
can be substituted for the other.

There is exactly ONE authorization artifact path for THIS gate
(FINAL_TEST_AUTHORIZATION_PATH). No CLI flag, environment variable, or
alternate file can substitute for it -- this module reads no CLI
argument and no environment variable anywhere; the ONLY input is the
committed artifact file's content plus git's own tracked/clean/ancestor
state.

verify_final_test_authorization() performs NO device, checkpoint, or
dataset-array access of any kind -- only git subprocess calls, file
hashing, and metadata-only ledger/matrix reads via
resolve_canonical_training_completion(). It must be called, and must
succeed, strictly before device initialization, checkpoint loading, or
any test-array access (see final_test_evaluation.py's frozen ordering and
evaluation/test_loader.py, which also calls this function itself as a
structural belt-and-suspenders guarantee).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.cross_condition_addendum import compute_cross_condition_fingerprint
from when_tta_hurts.dataset_verification import expected_official_checksum
from when_tta_hurts.final_test_identity import compute_final_test_runner_fingerprint
from when_tta_hurts.matrix import parse_and_validate_matrix
from when_tta_hurts.statistical_analysis import compute_analysis_fingerprint
from when_tta_hurts.validation_evaluation import (
    compute_evaluator_fingerprint,
    resolve_canonical_training_completion,
)

FINAL_TEST_AUTHORIZATION_PATH = Path("artifacts/final_test_authorization.json")

_REQUIRED_KEYS = {
    "status",
    "approval_timestamp",
    "phase2b_protocol_commit",
    "matrix_commit",
    "cross_condition_addendum_commit",
    "evaluator_fingerprint",
    "statistical_analysis_fingerprint",
    "cross_condition_analysis_fingerprint",
    "final_test_runner_fingerprint",
    "official_dataset_checksums",
    "authorized_cells",
}


class FinalTestAuthorizationError(RuntimeError):
    """Raised for ANY authorization failure -- missing artifact, untracked,
    dirty working tree at that path, malformed JSON, missing required
    field, status != 'approved', a bound commit that is not an ancestor of
    HEAD, any fingerprint/checksum mismatch, or an authorized-cells set
    that does not exactly match the current 39-cell matrix/training
    identity. Always a hard failure, always before any device/checkpoint/
    dataset access."""


@dataclass(frozen=True)
class FinalTestAuthorization:
    status: str
    approval_timestamp: str
    phase2b_protocol_commit: str
    matrix_commit: str
    cross_condition_addendum_commit: str
    evaluator_fingerprint: str
    statistical_analysis_fingerprint: str
    cross_condition_analysis_fingerprint: str
    final_test_runner_fingerprint: str
    authorized_cells_by_run_id: dict[str, dict[str, Any]]
    artifact_sha256: str
    authorization_commit: str


def _git(repo_root: str | Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _is_tracked_and_clean(repo_root: str | Path, rel_path: str) -> bool:
    """True only if `rel_path` is tracked by git AND has no uncommitted
    modifications, relative to `repo_root` -- uses `git -C repo_root`
    rather than a process-wide chdir, so callers (and tests) never need to
    mutate global process state."""
    try:
        tracked = _git(repo_root, "ls-files", "--error-unmatch", rel_path)
    except subprocess.CalledProcessError:
        return False
    if not tracked:
        return False
    try:
        diff = _git(repo_root, "diff", "HEAD", "--", rel_path)
    except subprocess.CalledProcessError:
        return False
    return diff == ""


def _is_ancestor_of_head(repo_root: str | Path, commit: str) -> bool:
    try:
        _git(repo_root, "merge-base", "--is-ancestor", commit, "HEAD")
        return True
    except subprocess.CalledProcessError:
        return False


def verify_final_test_authorization(
    artifact_path: str | Path = FINAL_TEST_AUTHORIZATION_PATH,
    matrix_path: str = "configs/experiment_matrix.yaml",
    repo_root: str | Path = ".",
) -> FinalTestAuthorization:
    """Verify the final-test authorization artifact. Raises
    FinalTestAuthorizationError on ANY problem: missing file, untracked,
    dirty working tree at that path, malformed JSON, missing required
    field(s), status != 'approved', any bound commit that is not an
    ancestor of HEAD, any fingerprint/dataset-checksum mismatch against
    the CURRENT repository state, or an authorized_cells set that does
    not exactly match the current 39-cell matrix and its current
    canonical training attempt/checkpoint identity.

    `artifact_path` is interpreted relative to `repo_root` (both file
    existence and git operations), so synthetic tests can point this at
    an isolated temporary git repository without any process-wide chdir
    or mutation of the real repository. The production CLI always calls
    this with the defaults (repo_root=".").
    """
    repo_root = Path(repo_root)
    rel_path = str(artifact_path)
    full_path = repo_root / rel_path

    if not full_path.exists():
        raise FinalTestAuthorizationError(
            f"Final-test authorization artifact {full_path} does not exist. Test-split access is "
            f"locked until this is created and committed."
        )

    if not _is_tracked_and_clean(repo_root, rel_path):
        raise FinalTestAuthorizationError(
            f"Final-test authorization artifact {full_path} exists but is not tracked-and-clean in "
            f"git (untracked, or has uncommitted changes). A CLI flag or environment variable cannot "
            f"substitute for a committed authorization artifact."
        )

    try:
        raw = json.loads(full_path.read_text())
    except Exception as e:
        raise FinalTestAuthorizationError(f"Authorization artifact {full_path} is malformed JSON: {e}") from e

    if not isinstance(raw, dict):
        raise FinalTestAuthorizationError(f"Authorization artifact {full_path} must parse to a JSON object.")

    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise FinalTestAuthorizationError(
            f"Authorization artifact missing required field(s): {sorted(missing)}"
        )

    if raw["status"] != "approved":
        raise FinalTestAuthorizationError(
            f"Authorization artifact status is {raw['status']!r}, not 'approved'."
        )

    for commit_field in ("phase2b_protocol_commit", "matrix_commit", "cross_condition_addendum_commit"):
        commit = raw[commit_field]
        if not isinstance(commit, str) or not commit:
            raise FinalTestAuthorizationError(f"{commit_field} must be a non-empty commit SHA.")
        if not _is_ancestor_of_head(repo_root, commit):
            raise FinalTestAuthorizationError(
                f"{commit_field}={commit} is not an ancestor of HEAD -- authorization is invalidated "
                f"if any required commit has been rewritten or is otherwise unreachable."
            )

    current_evaluator_fp, _ = compute_evaluator_fingerprint()
    if raw["evaluator_fingerprint"] != current_evaluator_fp:
        raise FinalTestAuthorizationError(
            "Authorization's evaluator_fingerprint does not match the current evaluator fingerprint "
            "-- scientific/evaluator-code drift since approval. Hard failure."
        )

    current_analysis_fp, _ = compute_analysis_fingerprint()
    if raw["statistical_analysis_fingerprint"] != current_analysis_fp:
        raise FinalTestAuthorizationError(
            "Authorization's statistical_analysis_fingerprint does not match the current fingerprint."
        )

    current_cross_fp, _ = compute_cross_condition_fingerprint()
    if raw["cross_condition_analysis_fingerprint"] != current_cross_fp:
        raise FinalTestAuthorizationError(
            "Authorization's cross_condition_analysis_fingerprint does not match the current fingerprint."
        )

    current_runner_fp, _ = compute_final_test_runner_fingerprint()
    if raw["final_test_runner_fingerprint"] != current_runner_fp:
        raise FinalTestAuthorizationError(
            "Authorization's final_test_runner_fingerprint does not match the current fingerprint -- "
            "the final-test runner implementation has changed since approval."
        )

    dataset_checksums = raw["official_dataset_checksums"]
    if not isinstance(dataset_checksums, dict):
        raise FinalTestAuthorizationError("official_dataset_checksums must be a mapping.")
    for key, expected_in_artifact in dataset_checksums.items():
        try:
            dataset, resolution_str = key.rsplit("@", 1)
            resolution = int(resolution_str)
        except (ValueError, AttributeError) as e:
            raise FinalTestAuthorizationError(
                f"official_dataset_checksums key {key!r} must be formatted '<dataset>@<resolution>'."
            ) from e
        actual_expected = expected_official_checksum(dataset, resolution)
        if actual_expected != expected_in_artifact:
            raise FinalTestAuthorizationError(
                f"official_dataset_checksums[{key!r}]={expected_in_artifact!r} does not match the "
                f"current official expected checksum {actual_expected!r} for {dataset}@{resolution}px."
            )

    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cells = list(expanded.cells)

    authorized_cells = raw["authorized_cells"]
    if not isinstance(authorized_cells, list):
        raise FinalTestAuthorizationError("authorized_cells must be a list.")
    if len(authorized_cells) != len(cells):
        raise FinalTestAuthorizationError(
            f"authorized_cells has {len(authorized_cells)} entries, but the frozen matrix currently "
            f"resolves to {len(cells)} eligible cells -- authorization must bind exactly the current "
            f"matrix."
        )

    authorized_by_run_id: dict[str, dict[str, Any]] = {}
    for entry in authorized_cells:
        for field_name in ("run_id", "training_attempt", "checkpoint_hash"):
            if field_name not in entry:
                raise FinalTestAuthorizationError(
                    f"authorized_cells entry missing field {field_name!r}: {entry}"
                )
        authorized_by_run_id[entry["run_id"]] = entry

    matrix_run_ids = {c.run_id() for c in cells}
    if set(authorized_by_run_id.keys()) != matrix_run_ids:
        raise FinalTestAuthorizationError(
            "authorized_cells run_id set does not exactly match the frozen matrix's current cell set."
        )

    for cell in cells:
        run_id = cell.run_id()
        _, training_result = resolve_canonical_training_completion(run_id, matrix_path)
        entry = authorized_by_run_id[run_id]
        if entry["training_attempt"] != training_result.attempt_number:
            raise FinalTestAuthorizationError(
                f"authorized_cells[{run_id!r}].training_attempt={entry['training_attempt']} does not "
                f"match the current canonical training attempt {training_result.attempt_number} -- "
                f"training history has changed since approval."
            )
        if entry["checkpoint_hash"] != training_result.checkpoint_hash:
            raise FinalTestAuthorizationError(
                f"authorized_cells[{run_id!r}].checkpoint_hash does not match the current canonical "
                f"checkpoint hash -- training history has changed since approval."
            )

    authorization_commit = _git(repo_root, "log", "-1", "--format=%H", "--", rel_path)
    if not authorization_commit:
        raise FinalTestAuthorizationError(
            f"Could not resolve a commit for {full_path} -- it must be committed, not merely tracked."
        )

    return FinalTestAuthorization(
        status=raw["status"],
        approval_timestamp=raw["approval_timestamp"],
        phase2b_protocol_commit=raw["phase2b_protocol_commit"],
        matrix_commit=raw["matrix_commit"],
        cross_condition_addendum_commit=raw["cross_condition_addendum_commit"],
        evaluator_fingerprint=raw["evaluator_fingerprint"],
        statistical_analysis_fingerprint=raw["statistical_analysis_fingerprint"],
        cross_condition_analysis_fingerprint=raw["cross_condition_analysis_fingerprint"],
        final_test_runner_fingerprint=raw["final_test_runner_fingerprint"],
        authorized_cells_by_run_id=authorized_by_run_id,
        artifact_sha256=hash_file(full_path),
        authorization_commit=authorization_commit,
    )
