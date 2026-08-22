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

import hashlib
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
    next_evaluation_attempt_number,
    resolve_canonical_training_completion,
)

FINAL_TEST_AUTHORIZATION_PATH = Path("artifacts/final_test_authorization.json")

# Duplicated (not imported) from final_test_evaluation.py to avoid a
# circular import -- final_test_evaluation.py imports FROM this module.
# Both must remain "artifacts/final_test"; a test-only mismatch here
# would be caught immediately by test_final_test_authorization.py's own
# real-repo-state tests.
_DEFAULT_FINAL_TEST_ROOT = Path("artifacts/final_test")

# Phase 2B.6D: schema/version 2 adds exact per-cell final-test attempt
# binding (authorized_cells[i].authorized_final_test_attempt) and an
# optional supersession block, used when a new authorization replaces an
# earlier one (e.g. after a runner-code fix following
# docs/phase2b_final_test_accidental_access_incident.md). Schema v1
# ("phase2b.6b-v1") authorizations are no longer accepted -- they lack
# per-cell attempt binding entirely, so they cannot express "attempt 2
# only for the affected cell, attempt 1 only for every other cell" and
# must never silently authorize execution after supersession.
_SUPPORTED_SCHEMA_VERSIONS = frozenset({"phase2b.6d-v2"})

_REQUIRED_KEYS = {
    "schema_version",
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

_REQUIRED_CELL_KEYS = {
    "run_id",
    "training_attempt",
    "checkpoint_hash",
    "authorized_final_test_attempt",
    "dataset",
    "resolution",
}

# If ANY of these supersession keys is present, ALL of them must be
# present (all-or-nothing) and are fully verified -- a schema-v2
# authorization that supersedes nothing (a hypothetical future first-ever
# v2 authorization with no prior incident) may omit the entire block.
_SUPERSESSION_KEYS = frozenset(
    {
        "supersedes_authorization_sha256",
        "supersedes_authorization_commit",
        "incident_record_commit",
        "recovery_policy_commit",
        "no_further_retry",
    }
)


class FinalTestAuthorizationError(RuntimeError):
    """Raised for ANY authorization failure -- missing artifact, untracked,
    dirty working tree at that path, malformed JSON, missing required
    field, status != 'approved', a bound commit that is not an ancestor of
    HEAD, any fingerprint/checksum mismatch, or an authorized-cells set
    that does not exactly match the current 39-cell matrix/training
    identity. Always a hard failure, always before any device/checkpoint/
    dataset access."""


@dataclass(frozen=True)
class VerifiedFinalTestReceipt:
    """Immutable, single-cell snapshot of an already-successful
    verify_final_test_authorization() call. Constructible ONLY via
    FinalTestAuthorization.receipt_for() -- there is no public
    constructor accepting caller-supplied field values, and no
    from_dict()/deserialization path anywhere in this module. Captures
    the run's authorized attempt number as a SNAPSHOT, taken at
    verification time -- it is never recomputed afterward (see
    docs/phase2b_final_test_authorization_receipt_freeze.md, which this
    type implements)."""

    run_id: str
    authorized_attempt: int
    checkpoint_hash: str
    training_attempt: int
    dataset: str
    resolution: int
    dataset_expected_checksum_md5: str
    evaluator_fingerprint: str
    statistical_analysis_fingerprint: str
    cross_condition_analysis_fingerprint: str
    final_test_runner_fingerprint: str
    authorization_artifact_sha256: str
    authorization_commit: str
    phase2b_protocol_commit: str
    matrix_commit: str
    cross_condition_addendum_commit: str


@dataclass(frozen=True)
class FinalTestAuthorization:
    status: str
    schema_version: str
    approval_timestamp: str
    phase2b_protocol_commit: str
    matrix_commit: str
    cross_condition_addendum_commit: str
    evaluator_fingerprint: str
    statistical_analysis_fingerprint: str
    cross_condition_analysis_fingerprint: str
    final_test_runner_fingerprint: str
    authorized_cells_by_run_id: dict[str, dict[str, Any]]
    official_dataset_checksums: dict[str, str]
    artifact_sha256: str
    authorization_commit: str
    supersedes_authorization_sha256: str | None
    supersedes_authorization_commit: str | None
    incident_record_commit: str | None
    recovery_policy_commit: str | None
    no_further_retry: bool | None

    def receipt_for(self, run_id: str) -> VerifiedFinalTestReceipt:
        """The ONLY way to construct a VerifiedFinalTestReceipt. Reads
        exclusively from THIS already-verified object's own fields --
        never re-invokes next_evaluation_attempt_number() or any other
        dynamic/stateful check. Raises FinalTestAuthorizationError if
        `run_id` was not part of the authorized cell set (should be
        unreachable given verify_final_test_authorization()'s own
        upfront run_id-set check, but never assumed)."""
        entry = self.authorized_cells_by_run_id.get(run_id)
        if entry is None:
            raise FinalTestAuthorizationError(f"No authorized cell for run_id {run_id!r}.")
        dataset = entry["dataset"]
        resolution = entry["resolution"]
        checksum_key = f"{dataset}@{resolution}"
        dataset_expected_checksum_md5 = self.official_dataset_checksums.get(checksum_key)
        if dataset_expected_checksum_md5 is None:
            raise FinalTestAuthorizationError(
                f"No official_dataset_checksums entry for {checksum_key!r} -- cannot issue a receipt."
            )
        return VerifiedFinalTestReceipt(
            run_id=run_id,
            authorized_attempt=entry["authorized_final_test_attempt"],
            checkpoint_hash=entry["checkpoint_hash"],
            training_attempt=entry["training_attempt"],
            dataset=dataset,
            resolution=resolution,
            dataset_expected_checksum_md5=dataset_expected_checksum_md5,
            evaluator_fingerprint=self.evaluator_fingerprint,
            statistical_analysis_fingerprint=self.statistical_analysis_fingerprint,
            cross_condition_analysis_fingerprint=self.cross_condition_analysis_fingerprint,
            final_test_runner_fingerprint=self.final_test_runner_fingerprint,
            authorization_artifact_sha256=self.artifact_sha256,
            authorization_commit=self.authorization_commit,
            phase2b_protocol_commit=self.phase2b_protocol_commit,
            matrix_commit=self.matrix_commit,
            cross_condition_addendum_commit=self.cross_condition_addendum_commit,
        )


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


def _historical_blob_sha256(repo_root: str | Path, commit: str, rel_path: str) -> str:
    """SHA-256 of `rel_path`'s content AT `commit` (via `git show`), so a
    superseding authorization's claimed supersedes_authorization_sha256
    can be checked against the REAL historical content, not a
    caller-asserted value. Raises FinalTestAuthorizationError if the
    commit/path cannot be resolved."""
    try:
        content = subprocess.check_output(
            ["git", "-C", str(repo_root), "show", f"{commit}:{rel_path}"], stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        raise FinalTestAuthorizationError(
            f"Could not read {rel_path} at commit {commit} to verify supersession -- {e}"
        ) from e
    return hashlib.sha256(content).hexdigest()


def verify_final_test_authorization(
    artifact_path: str | Path = FINAL_TEST_AUTHORIZATION_PATH,
    matrix_path: str = "configs/experiment_matrix.yaml",
    repo_root: str | Path = ".",
    final_test_root: str | Path = _DEFAULT_FINAL_TEST_ROOT,
    final_test_ledger_path: str | Path | None = None,
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

    if raw["schema_version"] not in _SUPPORTED_SCHEMA_VERSIONS:
        raise FinalTestAuthorizationError(
            f"Authorization schema_version {raw['schema_version']!r} is not supported "
            f"(supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}). An older schema cannot silently "
            f"authorize execution after supersession -- see "
            f"docs/phase2b_final_test_incident_recovery_freeze.md."
        )

    present_supersession_keys = _SUPERSESSION_KEYS & set(raw.keys())
    if present_supersession_keys and present_supersession_keys != _SUPERSESSION_KEYS:
        raise FinalTestAuthorizationError(
            f"Supersession fields must be all-or-nothing; found only "
            f"{sorted(present_supersession_keys)}, missing "
            f"{sorted(_SUPERSESSION_KEYS - present_supersession_keys)}."
        )
    has_supersession = present_supersession_keys == _SUPERSESSION_KEYS
    if has_supersession:
        old_commit = raw["supersedes_authorization_commit"]
        old_sha256 = raw["supersedes_authorization_sha256"]
        incident_commit = raw["incident_record_commit"]
        recovery_commit = raw["recovery_policy_commit"]
        for commit_field, commit in (
            ("supersedes_authorization_commit", old_commit),
            ("incident_record_commit", incident_commit),
            ("recovery_policy_commit", recovery_commit),
        ):
            if not isinstance(commit, str) or not commit:
                raise FinalTestAuthorizationError(f"{commit_field} must be a non-empty commit SHA.")
            if not _is_ancestor_of_head(repo_root, commit):
                raise FinalTestAuthorizationError(
                    f"{commit_field}={commit} is not an ancestor of HEAD -- supersession provenance "
                    f"is invalid."
                )
        actual_old_sha256 = _historical_blob_sha256(repo_root, old_commit, rel_path)
        if actual_old_sha256 != old_sha256:
            raise FinalTestAuthorizationError(
                f"supersedes_authorization_sha256={old_sha256!r} does not match the actual content of "
                f"{rel_path} at commit {old_commit} (sha256={actual_old_sha256!r}) -- supersession "
                f"provenance is invalid."
            )
        if raw["no_further_retry"] is not True:
            raise FinalTestAuthorizationError("no_further_retry must be exactly True when superseding.")

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

    if final_test_ledger_path is None:
        from when_tta_hurts.ledger import FINAL_TEST_LEDGER_PATH as _default_final_test_ledger_path

        final_test_ledger_path = _default_final_test_ledger_path

    authorized_by_run_id: dict[str, dict[str, Any]] = {}
    for entry in authorized_cells:
        for field_name in _REQUIRED_CELL_KEYS:
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
        # Exact-attempt binding (Phase 2B.6D): the authorized final-test
        # attempt number for this cell must EXACTLY equal what the
        # production runner would allocate next -- never merely "<= N".
        # This is what makes "attempt 2 only for the affected cell,
        # attempt 1 only for every other cell" (and never attempt 3, and
        # never attempt 2 for an unaffected cell) mechanically enforced
        # rather than advisory.
        next_attempt = next_evaluation_attempt_number(run_id, final_test_root, final_test_ledger_path)
        if entry["authorized_final_test_attempt"] != next_attempt:
            raise FinalTestAuthorizationError(
                f"authorized_cells[{run_id!r}].authorized_final_test_attempt="
                f"{entry['authorized_final_test_attempt']!r} does not match the production runner's "
                f"next allocatable attempt ({next_attempt}) -- refusing to authorize an attempt number "
                f"that does not exactly match current final-test ledger/attempt-directory state."
            )

    authorization_commit = _git(repo_root, "log", "-1", "--format=%H", "--", rel_path)
    if not authorization_commit:
        raise FinalTestAuthorizationError(
            f"Could not resolve a commit for {full_path} -- it must be committed, not merely tracked."
        )

    return FinalTestAuthorization(
        status=raw["status"],
        schema_version=raw["schema_version"],
        approval_timestamp=raw["approval_timestamp"],
        phase2b_protocol_commit=raw["phase2b_protocol_commit"],
        matrix_commit=raw["matrix_commit"],
        cross_condition_addendum_commit=raw["cross_condition_addendum_commit"],
        evaluator_fingerprint=raw["evaluator_fingerprint"],
        statistical_analysis_fingerprint=raw["statistical_analysis_fingerprint"],
        cross_condition_analysis_fingerprint=raw["cross_condition_analysis_fingerprint"],
        final_test_runner_fingerprint=raw["final_test_runner_fingerprint"],
        authorized_cells_by_run_id=authorized_by_run_id,
        official_dataset_checksums=dataset_checksums,
        artifact_sha256=hash_file(full_path),
        authorization_commit=authorization_commit,
        supersedes_authorization_sha256=raw.get("supersedes_authorization_sha256"),
        supersedes_authorization_commit=raw.get("supersedes_authorization_commit"),
        incident_record_commit=raw.get("incident_record_commit"),
        recovery_policy_commit=raw.get("recovery_policy_commit"),
        no_further_retry=raw.get("no_further_retry"),
    )


def verify_receipt_still_valid(
    receipt: VerifiedFinalTestReceipt,
    dataset: str,
    resolution: int,
    artifact_path: str | Path = FINAL_TEST_AUTHORIZATION_PATH,
    repo_root: str | Path = ".",
) -> None:
    """Static, comparative recheck performed by the test-only loader
    immediately before test-data access (Phase 2B.6F, item 5 of
    docs/phase2b_final_test_authorization_receipt_freeze.md). Confirms
    the receipt is still trustworthy WITHOUT ever recomputing an attempt
    number or scanning the active run's attempt directory/ledger history
    -- the exact defect this module corrects
    (docs/phase2b_final_test_attempt2_preaccess_failure.md).

    Raises FinalTestAuthorizationError on ANY of:
    - `dataset`/`resolution` (the loader's own call arguments) do not
      match the receipt's bound dataset/resolution -- a receipt for a
      different cell must never be silently reused;
    - the authorization file no longer exists;
    - its current SHA-256 no longer matches the receipt's bound value;
    - it is no longer tracked-and-clean in git;
    - any of the four fingerprints has drifted since the receipt was
      issued;
    - the official expected checksum for (dataset, resolution) has
      drifted since the receipt was issued.
    """
    if receipt.dataset != dataset or receipt.resolution != resolution:
        raise FinalTestAuthorizationError(
            f"Receipt is bound to {receipt.dataset}@{receipt.resolution}px, but this loader call "
            f"requested {dataset}@{resolution}px -- refusing to reuse a receipt for a different cell."
        )

    repo_root = Path(repo_root)
    rel_path = str(artifact_path)
    full_path = repo_root / rel_path

    if not full_path.exists():
        raise FinalTestAuthorizationError(
            f"Authorization artifact {full_path} no longer exists -- receipt is no longer valid."
        )
    current_sha256 = hash_file(full_path)
    if current_sha256 != receipt.authorization_artifact_sha256:
        raise FinalTestAuthorizationError(
            "Authorization artifact bytes have changed since the receipt was issued -- receipt is no "
            "longer valid."
        )
    if not _is_tracked_and_clean(repo_root, rel_path):
        raise FinalTestAuthorizationError(
            f"Authorization artifact {full_path} is no longer tracked-and-clean in git -- receipt is "
            f"no longer valid."
        )

    current_evaluator_fp, _ = compute_evaluator_fingerprint()
    if receipt.evaluator_fingerprint != current_evaluator_fp:
        raise FinalTestAuthorizationError("evaluator_fingerprint has drifted since the receipt was issued.")
    current_analysis_fp, _ = compute_analysis_fingerprint()
    if receipt.statistical_analysis_fingerprint != current_analysis_fp:
        raise FinalTestAuthorizationError(
            "statistical_analysis_fingerprint has drifted since the receipt was issued."
        )
    current_cross_fp, _ = compute_cross_condition_fingerprint()
    if receipt.cross_condition_analysis_fingerprint != current_cross_fp:
        raise FinalTestAuthorizationError(
            "cross_condition_analysis_fingerprint has drifted since the receipt was issued."
        )
    current_runner_fp, _ = compute_final_test_runner_fingerprint()
    if receipt.final_test_runner_fingerprint != current_runner_fp:
        raise FinalTestAuthorizationError(
            "final_test_runner_fingerprint has drifted since the receipt was issued."
        )

    current_expected_checksum = expected_official_checksum(dataset, resolution)
    if receipt.dataset_expected_checksum_md5 != current_expected_checksum:
        raise FinalTestAuthorizationError(
            f"Official expected checksum for {dataset}@{resolution}px has drifted since the receipt "
            f"was issued."
        )
