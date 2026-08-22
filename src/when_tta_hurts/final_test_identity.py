"""Phase 2B.6A: final-test-runner implementation fingerprint and
evaluation-identity binding.

Kept as its own small module (rather than inside final_test_evaluation.py
or final_test_authorization.py) so both of those modules can import
fingerprint/identity primitives from here without a circular import:
final_test_authorization.py verifies bindings computed here, while
final_test_evaluation.py (the orchestrator) computes the same bindings to
build the persisted evaluation identity -- neither needs to import the
other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.cross_condition_addendum import CROSS_CONDITION_ADDENDUM_MANIFEST

# Every file whose content could change a reported final-test scientific
# number, an authorization binding, or the identity of the evaluated test
# data: everything CROSS_CONDITION_ADDENDUM_MANIFEST already tracks
# (which itself transitively includes ANALYSIS_FINGERPRINT_MANIFEST's
# matrix/metrics/artifact/validation-evaluation code and the dependency
# lock), plus every new Phase 2B.6A module and the dataset-loading code
# the final-test path newly reaches. Deliberately excludes docs/ledgers --
# a documentation- or ledger-only commit must never change this identity.
FINAL_TEST_RUNNER_MANIFEST: tuple[str, ...] = CROSS_CONDITION_ADDENDUM_MANIFEST + (
    "src/when_tta_hurts/final_test_identity.py",
    "src/when_tta_hurts/final_test_authorization.py",
    "src/when_tta_hurts/final_test_result_artifacts.py",
    "src/when_tta_hurts/final_test_evaluation.py",
    "src/when_tta_hurts/evaluation/test_loader.py",
    "src/when_tta_hurts/dataset_verification.py",
    "src/when_tta_hurts/data.py",
)


class FinalTestFingerprintError(RuntimeError):
    """Raised when a file listed in FINAL_TEST_RUNNER_MANIFEST is missing.
    Fails closed -- never computes a partial fingerprint."""


def compute_final_test_runner_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = FINAL_TEST_RUNNER_MANIFEST,
) -> tuple[str, dict[str, str]]:
    """Deterministic content fingerprint of every final-test-runner-
    relevant tracked file, mirroring compute_evaluator_fingerprint()'s and
    compute_analysis_fingerprint()'s exact discipline: STABLE (independent
    of docs/ledger-only commits), computed from actual working-tree bytes,
    never partial."""
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise FinalTestFingerprintError(
                f"Final-test-runner fingerprint manifest file missing: {rel_path}. Refusing to "
                f"compute a partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


@dataclass(frozen=True)
class FinalTestEvaluationConfig:
    """Every input that determines a final-test evaluation's scientific
    and authorization identity. `split="test"` is a literal in
    compute_final_test_evaluation_id() below, never a field here -- it can
    never be set to anything else by construction."""

    training_run_id: str
    training_attempt: int
    checkpoint_hash: str
    matrix_hash: str
    protocol_commit: str
    tta_seed_config_sha256: str
    tta_seed_freeze_commit: str
    tta_seed_derivation_sha256: str
    evaluator_fingerprint: str
    statistical_analysis_fingerprint: str
    cross_condition_analysis_fingerprint: str
    final_test_runner_fingerprint: str
    authorization_artifact_sha256: str
    authorization_commit: str
    dataset_expected_checksum_md5: str
    extra: dict[str, Any] = field(default_factory=dict)


def compute_final_test_evaluation_id(cfg: FinalTestEvaluationConfig) -> str:
    """Deterministic, STABLE final-test-evaluation identity. `split='test'`
    is baked in literally so it can never accidentally bind an identity to
    any other split. Any scientific/config/authorization drift in `cfg`
    changes this ID; unrelated documentation/ledger-only commits change
    none of `cfg`'s inputs, so this ID is stable across such commits."""
    payload = {"split": "test", **cfg.__dict__}
    return config_hash(payload)
