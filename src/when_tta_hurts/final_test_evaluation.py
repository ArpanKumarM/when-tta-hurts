"""Phase 2B.6A: final-test-evaluation orchestrator for the Phase 2B
39-cell confirmatory matrix.

`run_final_test_evaluation()` -- the real execution path -- is
IMPLEMENTED here per the frozen order in
docs/phase2b_final_test_runner_engineering_freeze.md sec.3, but is NEVER
INVOKED anywhere in the Phase 2B.6A engineering task: no script, CLI
command, or test in this repository calls it against real data, a real
device, or a real checkpoint. It is exercised only by this module's own
tests, always against synthetic arrays, fake devices, and temporary
ledgers/matrices/authorization artifacts.

`plan_final_test_evaluation()` is side-effect-free and test-data-free: it
resolves training-completion and authorization IDENTITY metadata only
(ledger rows, status.json files, fingerprints) and never touches MPS, a
checkpoint's tensor contents, or any dataset array.

Reuses validation_evaluation.py's scientific-computation and attempt-
lifecycle functions UNCHANGED (compute_validation_evaluation,
compute_evaluation_latency_report, load_and_verify_canonical_checkpoint,
start_evaluation_attempt, finish_evaluation_attempt, check_evaluation_skip,
_verify_metrics_semantically, etc.) -- the only scientific-data difference
between a validation and a final-test evaluation is split=test vs
split=validation; no scientific computation is forked.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.cross_condition_addendum import compute_cross_condition_fingerprint
from when_tta_hurts.dataset_verification import expected_official_checksum, verify_official_dataset_artifact
from when_tta_hurts.devices import select_device
from when_tta_hurts.evaluation.test_loader import load_final_test_split
from when_tta_hurts.evaluation.views import build_view_seed_manifest
from when_tta_hurts.evaluation_result_artifacts import EvaluationPersistenceError
from when_tta_hurts.final_test_authorization import (
    FinalTestAuthorizationError,
    verify_final_test_authorization,
)
from when_tta_hurts.final_test_identity import (
    FinalTestEvaluationConfig,
    compute_final_test_evaluation_id,
    compute_final_test_runner_fingerprint,
)
from when_tta_hurts.final_test_result_artifacts import persist_and_verify_final_test_completion
from when_tta_hurts.ledger import FINAL_TEST_LEDGER_PATH, append_final_test_entry
from when_tta_hurts.matrix import parse_and_validate_matrix
from when_tta_hurts.metrics import accuracy
from when_tta_hurts.orchestrator import require_clean_working_tree
from when_tta_hurts.statistical_analysis import compute_analysis_fingerprint
from when_tta_hurts.validation_evaluation import (
    AGGREGATORS,
    DATASET_VERIFICATION_METHOD,
    DATASET_VERIFICATION_VERSION,
    DEFAULT_TTA_SEED_CONFIG_PATH,
    FROZEN_PROTOCOL_COMMIT,
    MAX_VIEWS,
    PREFIX_SEQUENCE,
    PRIMARY_N,
    SECONDARY_ANALYSES,
    AmbiguousEvaluationCompletionError,
    ConflictingEvaluationImplementationError,
    EvaluationLedgerConflictError,
    EvaluationRunStatus,
    EvaluationStaleAttemptError,
    _git_commit_hash,
    _latency_report_to_dict,
    _verify_metrics_semantically,
    compute_evaluation_latency_report,
    compute_evaluator_fingerprint,
    compute_validation_evaluation,
    evaluation_run_directory,
    finish_evaluation_attempt,
    list_evaluation_attempts,
    load_and_verify_canonical_checkpoint,
    load_frozen_tta_seed_config,
    recompute_clean_accuracy,
    recompute_mean_probability_prefix,
    resolve_canonical_training_completion,
    start_evaluation_attempt,
)

DEFAULT_FINAL_TEST_ROOT = Path("artifacts/final_test")
DEFAULT_FINAL_TEST_AMENDMENTS_LEDGER_PATH = Path("artifacts/ledger_final_test_amendments.csv")

# Terminal statuses for a final-test attempt -- mirrors
# validation_evaluation.py's private _TERMINAL_EVAL_STATUSES exactly, but
# redefined here (rather than imported) so this module's stale/skip logic
# has no dependency on that module's private name, keeping the fix fully
# self-contained to FINAL_TEST_RUNNER_MANIFEST-covered files.
_FINAL_TEST_TERMINAL_STATUSES = frozenset({"completed", "failed", "aborted"})


def check_final_test_evaluation_skip(
    training_run_id: str,
    evaluation_config_hash: str,
    root: str | Path = DEFAULT_FINAL_TEST_ROOT,
    ledger_path: str | Path | None = None,
    amendments_ledger_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Final-test-specific idempotent-skip check, correcting the defect in
    docs/phase2b_final_test_accidental_access_incident.md sec.8:
    validation_evaluation.check_evaluation_skip()'s internal call to
    ledger_module.has_evaluation_row(evaluation_config_hash, attempt) (a)
    reads CSV column 'evaluation_id', which does not exist in
    FINAL_TEST_LEDGER_FIELDNAMES (that schema uses
    'final_test_evaluation_id'/'evaluation_config_hash' instead), and
    (b) never receives the caller's ledger_path, always reading the
    VALIDATION ledger's default path regardless. Both are fixed here by
    reading the final-test ledger directly, by its own correct columns.

    Otherwise mirrors check_evaluation_skip()'s exact algorithm and
    ordering, with one deliberate refinement: the "unledgered" staleness
    check (step 1) asks "does ANY ledger row exist for (training_run_id,
    attempt_number)" -- not "does a row exist under the CURRENT request's
    hash specifically". This is the correct semantic for reconciling an
    externally-terminated attempt whose historical hash legitimately
    differs from a fresh request's hash after a runner-code fix (exactly
    this incident's recovery scenario) -- hash agreement for a GIVEN
    attempt number between directory and ledger remains strictly enforced
    in the consistency-check step (step 2) that follows, unchanged in
    spirit from the original algorithm.

    Metadata-only: does NOT touch MPS, the checkpoint, the dataset, view
    generation, or metric calculation.

    In order:
    1. For every attempt DIRECTORY (any status): if nonterminal AND no
       ledger row exists at all for (training_run_id, attempt_number),
       raise EvaluationStaleAttemptError.
    2. Ledger/directory consistency, per (training_run_id, attempt_number):
       directory+ledger hash mismatch -> EvaluationLedgerConflictError;
       ledger-only row must be terminal (aborted/failed) or hard-fail;
       directory-only terminal status with no ledger row -> hard-fail.
    3. Among completed, artifact-manifest-verified attempts: ambiguous
       (>1 matching current hash) -> AmbiguousEvaluationCompletionError;
       exactly one matching -> return it; none matching but a completed
       attempt exists under a DIFFERENT hash ->
       ConflictingEvaluationImplementationError; otherwise return None.
    """
    from when_tta_hurts import ledger as ledger_module
    from when_tta_hurts.final_test_result_artifacts import verify_final_test_artifact_manifest

    if ledger_path is None:
        ledger_path = ledger_module.FINAL_TEST_LEDGER_PATH
    if amendments_ledger_path is None:
        amendments_ledger_path = DEFAULT_FINAL_TEST_AMENDMENTS_LEDGER_PATH

    run_dir = evaluation_run_directory(training_run_id, root)
    all_attempts = list_evaluation_attempts(training_run_id, root)
    dir_by_number = {s["attempt_number"]: s for s in all_attempts}

    ledger_rows_for_run = [
        row
        for row in ledger_module._read_existing_rows(ledger_path)
        if row.get("training_run_id") == training_run_id
    ]
    ledger_by_number: dict[int, dict[str, Any]] = {}
    for row in ledger_rows_for_run:
        ledger_by_number.setdefault(int(row["evaluation_attempt"]), row)

    for status in all_attempts:
        if status.get("status") not in _FINAL_TEST_TERMINAL_STATUSES:
            if status["attempt_number"] not in ledger_by_number:
                raise EvaluationStaleAttemptError(
                    f"Final-test evaluation of {training_run_id} attempt_{status['attempt_number']:03d} "
                    f"is nonterminal (status='{status.get('status')}') and has no ledger row -- "
                    f"refusing to start a new attempt without explicit reconciliation."
                )

    for number in sorted(set(dir_by_number) | set(ledger_by_number)):
        dir_status = dir_by_number.get(number)
        ledger_row = ledger_by_number.get(number)
        if dir_status is not None and ledger_row is not None:
            if dir_status["evaluation_config_hash"] != ledger_row["evaluation_config_hash"]:
                raise EvaluationLedgerConflictError(
                    f"{training_run_id} attempt_{number:03d}: directory evaluation_config_hash "
                    f"{dir_status['evaluation_config_hash']} does not match ledger row's "
                    f"{ledger_row['evaluation_config_hash']}."
                )
        elif dir_status is None and ledger_row is not None:
            if ledger_row["status"] not in ("aborted", "failed"):
                raise EvaluationLedgerConflictError(
                    f"{training_run_id} attempt_{number:03d}: ledger records status="
                    f"{ledger_row['status']!r} but no attempt directory exists -- only aborted/"
                    f"failed attempts may have a deleted directory."
                )
        elif dir_status is not None and ledger_row is None:
            if dir_status.get("status") in _FINAL_TEST_TERMINAL_STATUSES:
                raise EvaluationLedgerConflictError(
                    f"{training_run_id} attempt_{number:03d}: directory has terminal status "
                    f"{dir_status.get('status')!r} but no ledger row exists -- requires explicit "
                    f"reconciliation, not a silent retry."
                )

    matching_completed = []
    conflicting_completed = []
    for status in all_attempts:
        if status.get("status") != EvaluationRunStatus.COMPLETED.value:
            continue
        if ledger_module.is_evaluation_canonical_ineligible(
            status["evaluation_config_hash"], status["attempt_number"], amendments_ledger_path
        ):
            continue
        if status["evaluation_config_hash"] != evaluation_config_hash:
            conflicting_completed.append(status)
            continue
        attempt_dir = run_dir / f"attempt_{status['attempt_number']:03d}"
        manifest_path = attempt_dir / "artifact_manifest.json"
        if not manifest_path.exists():
            raise EvaluationPersistenceError(
                f"Completed final-test attempt_{status['attempt_number']:03d} for "
                f"{training_run_id} is missing artifact_manifest.json."
            )
        import json as _json

        manifest = _json.loads(manifest_path.read_text())
        verify_final_test_artifact_manifest(attempt_dir, manifest)
        matching_completed.append(status)

    if len(matching_completed) > 1:
        attempt_numbers = sorted(s["attempt_number"] for s in matching_completed)
        raise AmbiguousEvaluationCompletionError(
            f"Multiple completed, artifact-verified final-test attempts for {training_run_id} match "
            f"evaluation_config_hash={evaluation_config_hash}: attempts {attempt_numbers}. Refusing "
            f"to silently choose earliest/latest -- resolve via explicit reconciliation before "
            f"proceeding."
        )
    if matching_completed:
        return matching_completed[0]

    if conflicting_completed:
        attempt_numbers = sorted(s["attempt_number"] for s in conflicting_completed)
        conflicting_hashes = sorted({s["evaluation_config_hash"] for s in conflicting_completed})
        raise ConflictingEvaluationImplementationError(
            f"{training_run_id} already has a COMPLETED final-test evaluation (attempt(s) "
            f"{attempt_numbers}, evaluation_config_hash(es) {conflicting_hashes}) that does NOT match "
            f"the current request's evaluation_config_hash={evaluation_config_hash}. Hard failure -- "
            f"will not silently create a second, different-identity canonical completion for the "
            f"same training run. Requires explicit reconciliation before proceeding."
        )
    return None


class FinalTestAuthorizationRequiredError(RuntimeError):
    """Raised when a resolved run_id is not present in the current
    authorization's authorized_cells_by_run_id. Structurally should be
    unreachable once verify_final_test_authorization() has already
    confirmed the authorized_cells run_id set exactly matches the current
    matrix -- checked explicitly and separately anyway (frozen order step
    3), never assumed to follow automatically from step 2."""


# ---------------------------------------------------------------------------
# Plan mode -- side-effect-free, test-data-free, identity/manifest only.
# ---------------------------------------------------------------------------


def plan_final_test_evaluation(
    matrix_path: str = "configs/experiment_matrix.yaml",
    authorization_artifact_path: str | Path | None = None,
    root: str | Path = DEFAULT_FINAL_TEST_ROOT,
) -> dict[str, Any]:
    """SIDE-EFFECT-FREE and TEST-DATA-FREE: never opens predictions.npz,
    metrics.json, a checkpoint's tensor contents, or any dataset array;
    never initializes a device. Reports, for every cell in the frozen
    matrix (in frozen matrix order): its canonical-training-completion
    identity (or the error blocking it), and how many final-test attempts
    already exist on disk (metadata-only, via list_evaluation_attempts()).
    Separately reports whether a committed, current final-test
    authorization exists -- execution is reported "locked" whenever it
    does not, exactly mirroring the real orchestrator's own gate."""
    evaluator_fp, _ = compute_evaluator_fingerprint()
    analysis_fp, _ = compute_analysis_fingerprint()
    cross_fp, _ = compute_cross_condition_fingerprint()
    runner_fp, _ = compute_final_test_runner_fingerprint()

    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    cells = list(expanded.cells)

    cell_reports: list[dict[str, Any]] = []
    n_training_eligible = 0
    for cell in cells:
        run_id = cell.run_id()
        entry: dict[str, Any] = {
            "run_id": run_id,
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "normalization": cell.normalization,
            "training_policy": cell.training_policy,
            "seed": cell.seed,
        }
        try:
            _, training_result = resolve_canonical_training_completion(run_id, matrix_path)
            entry["training_attempt"] = training_result.attempt_number
            entry["checkpoint_hash"] = training_result.checkpoint_hash
            entry["training_eligible"] = True
            n_training_eligible += 1
        except Exception as e:  # noqa: BLE001 -- plan mode reports, never raises
            entry["training_eligible"] = False
            entry["training_error"] = f"{type(e).__name__}: {e}"

        entry["existing_final_test_attempts"] = len(list_evaluation_attempts(run_id, root))
        cell_reports.append(entry)

    auth_kwargs: dict[str, Any] = {}
    if authorization_artifact_path is not None:
        auth_kwargs["artifact_path"] = authorization_artifact_path
    try:
        verify_final_test_authorization(matrix_path=matrix_path, **auth_kwargs)
        authorization_status = "approved"
        authorization_error = None
    except FinalTestAuthorizationError as e:
        authorization_status = "missing_or_not_approved"
        authorization_error = str(e)

    return {
        "evaluator_fingerprint": evaluator_fp,
        "statistical_analysis_fingerprint": analysis_fp,
        "cross_condition_analysis_fingerprint": cross_fp,
        "final_test_runner_fingerprint": runner_fp,
        "n_cells_total": len(cells),
        "n_cells_training_eligible": n_training_eligible,
        "cells": cell_reports,
        "authorization_status": authorization_status,
        "authorization_error": authorization_error,
        "execution_locked": authorization_status != "approved",
        "test_split_accessed": False,
    }


# ---------------------------------------------------------------------------
# Real execution path -- implemented per the frozen order, but NEVER
# invoked anywhere in the Phase 2B.6A engineering task.
# ---------------------------------------------------------------------------


def run_final_test_evaluation(
    run_id: str,
    matrix_path: str = "configs/experiment_matrix.yaml",
    device_resolver=None,
    root: str | Path = DEFAULT_FINAL_TEST_ROOT,
    training_root: str | Path = "artifacts/confirmatory",
    data_root: str | Path = "data/raw",
    final_test_ledger_path: str | Path | None = None,
    final_test_amendments_ledger_path: str | Path = DEFAULT_FINAL_TEST_AMENDMENTS_LEDGER_PATH,
    tta_seed_config_path: str | Path = DEFAULT_TTA_SEED_CONFIG_PATH,
    authorization_artifact_path: str | Path | None = None,
    require_clean_tree: bool = True,
) -> dict[str, Any]:
    """Single entry point for final-test TTA evaluation of one confirmatory
    cell. NEVER invoked anywhere in the Phase 2B.6A engineering task.

    Enforces, IN ORDER (docs/phase2b_final_test_runner_engineering_freeze.md
    sec.3): (1) resolve exact run ID -- the `run_id` parameter itself;
    (2) verify committed authorization; (3) verify `run_id` is in the
    authorization's authorized cell set; (4) resolve canonical training
    completion; (5) recompute and bind every fingerprint/hash into a
    FinalTestEvaluationConfig and its final_test_evaluation_id; (6) check
    existing final-test attempt/ledger state (idempotent skip -- metadata
    only, strictly before device/checkpoint/dataset access); (7) enforce
    clean-tree policy; (8) allocate the attempt; (9) initialize the
    requested device; (10) restore the checkpoint; (11) verify the
    official dataset artifact checksum from raw file bytes; (12) load
    only test_images/test_labels; (13) execute frozen inference;
    (14) compute latency; (15) validate and atomically persist artifacts;
    (16) mark status=completed; (17) append the final-test ledger row.
    Steps (2)-(6) (every identity/authorization check) complete strictly
    before (9)-(12) (device/checkpoint/dataset access) -- a failure in
    (1)-(7) creates ZERO files and ZERO ledger rows, exactly like
    run_validation_evaluation()'s equivalent guarantee.

    `device_resolver` defaults to `select_device('mps')` -- there is no
    CPU fallback and no injectable synthetic-backend flag on the
    production CLI; the parameter exists only so this module's own tests
    can inject a fake device without ever touching MPS.
    """
    if device_resolver is None:
        device_resolver = lambda: select_device("mps")  # noqa: E731
    if final_test_ledger_path is None:
        final_test_ledger_path = FINAL_TEST_LEDGER_PATH

    # Step 2: verify committed authorization.
    auth_kwargs: dict[str, Any] = {}
    if authorization_artifact_path is not None:
        auth_kwargs["artifact_path"] = authorization_artifact_path
    authorization = verify_final_test_authorization(matrix_path=matrix_path, **auth_kwargs)

    # Step 3: verify the cell is included in the authorization manifest.
    if run_id not in authorization.authorized_cells_by_run_id:
        raise FinalTestAuthorizationRequiredError(
            f"'{run_id}' is not present in the current final-test authorization's authorized cell "
            f"set -- refusing to proceed."
        )

    # Immutable, single-cell receipt derived ONCE from the already-
    # verified authorization (Phase 2B.6F,
    # docs/phase2b_final_test_authorization_receipt_freeze.md). Passed
    # through to the test loader instead of letting it re-invoke the
    # full, dynamic verifier a second time after attempt allocation --
    # see docs/phase2b_final_test_attempt2_preaccess_failure.md for why
    # a second dynamic call is unsafe.
    receipt = authorization.receipt_for(run_id)

    # Step 4: resolve canonical training completion.
    cell, training_result = resolve_canonical_training_completion(run_id, matrix_path)

    # Step 5: recompute and bind every fingerprint/hash.
    seed_cfg = load_frozen_tta_seed_config(tta_seed_config_path)
    resolved_tta_seed = seed_cfg.confirmatory_tta_seed
    expanded = parse_and_validate_matrix(matrix_path, block_d_gate_passed=True)
    evaluator_fp, evaluator_manifest = compute_evaluator_fingerprint()
    analysis_fp, _ = compute_analysis_fingerprint()
    cross_fp, _ = compute_cross_condition_fingerprint()
    runner_fp, _ = compute_final_test_runner_fingerprint()
    dataset_expected_checksum = expected_official_checksum(cell.dataset, cell.resolution)

    cfg = FinalTestEvaluationConfig(
        training_run_id=run_id,
        training_attempt=training_result.attempt_number,
        checkpoint_hash=training_result.checkpoint_hash,
        matrix_hash=expanded.source_config_hash,
        protocol_commit=FROZEN_PROTOCOL_COMMIT,
        tta_seed_config_sha256=seed_cfg.config_file_sha256,
        tta_seed_freeze_commit=seed_cfg.freeze_commit,
        tta_seed_derivation_sha256=seed_cfg.derivation_sha256,
        evaluator_fingerprint=evaluator_fp,
        statistical_analysis_fingerprint=analysis_fp,
        cross_condition_analysis_fingerprint=cross_fp,
        final_test_runner_fingerprint=runner_fp,
        authorization_artifact_sha256=authorization.artifact_sha256,
        authorization_commit=authorization.authorization_commit,
        dataset_expected_checksum_md5=dataset_expected_checksum,
    )
    final_test_evaluation_id = compute_final_test_evaluation_id(cfg)
    source_commit = _git_commit_hash()

    # Step 6: check existing final-test attempt/ledger state (metadata only).
    skip = check_final_test_evaluation_skip(
        run_id,
        final_test_evaluation_id,
        root,
        final_test_ledger_path,
        final_test_amendments_ledger_path,
    )
    if skip is not None:
        return {
            "training_run_id": run_id,
            "final_test_evaluation_id": final_test_evaluation_id,
            **skip,
            "status": "skipped_completed",
        }

    # Step 7: enforce clean-tree policy.
    if require_clean_tree:
        require_clean_working_tree()

    # Step 8: allocate the attempt.
    attempt_dir, status = start_evaluation_attempt(
        run_id, final_test_evaluation_id, root, final_test_ledger_path
    )

    test_split_accessed = False
    test_predictions_computed = False
    test_metrics_computed = False
    test_metrics_persisted = False
    failure_stage = "device_initialization"
    try:
        # Step 9: initialize the requested device.
        device = device_resolver()
        failure_stage = "checkpoint_restore"

        # Step 10: restore checkpoint.
        model = load_and_verify_canonical_checkpoint(cell, training_result, training_root)
        failure_stage = "dataset_checksum_verification"

        # Step 11: verify official dataset artifact checksum from raw file bytes.
        dataset_verification = verify_official_dataset_artifact(cell.dataset, cell.resolution, root=data_root)
        if dataset_verification.expected_checksum_md5 != dataset_expected_checksum:
            raise EvaluationPersistenceError(
                "Expected checksum resolved at identity-computation time does not match the "
                "expected checksum resolved at verification time -- refusing an inconsistent binding."
            )
        failure_stage = "test_array_load"

        # Step 12: load only test_images/test_labels.
        try:
            split = load_final_test_split(cell.dataset, cell.resolution, data_root, receipt=receipt)
        except FinalTestAuthorizationError:
            # The loader's OWN static receipt recheck failed -- distinct
            # from a genuine test-array-load failure (corrupt file, shape
            # mismatch, etc), per
            # docs/phase2b_final_test_authorization_receipt_freeze.md
            # item 10.
            failure_stage = "test_loader_authorization_receipt"
            raise
        test_split_accessed = True
        failure_stage = "inference"

        # Step 13: execute frozen inference.
        outcome = compute_validation_evaluation(model, split, resolved_tta_seed, device)
        test_predictions_computed = True
        test_metrics_computed = True
        failure_stage = "latency"

        # Step 14: compute latency.
        latency_report = compute_evaluation_latency_report(model, split, resolved_tta_seed, device)
        failure_stage = "persistence"

        seed_manifest = build_view_seed_manifest(
            resolved_tta_seed, cell.dataset, cell.resolution, split.sample_indices.tolist(), MAX_VIEWS
        )
        seed_manifest_hash = config_hash({"entries": [asdict(e) for e in seed_manifest]})

        metadata = {
            "final_test_evaluation_id": final_test_evaluation_id,
            "training_run_id": run_id,
            "training_attempt": training_result.attempt_number,
            "checkpoint_hash": training_result.checkpoint_hash,
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "model": cell.model,
            "normalization": cell.normalization,
            "training_policy": cell.training_policy,
            "seed": cell.seed,
            "tta_seed": resolved_tta_seed,
            "tta_seed_config_sha256": seed_cfg.config_file_sha256,
            "tta_seed_freeze_commit": seed_cfg.freeze_commit,
            "tta_seed_derivation_sha256": seed_cfg.derivation_sha256,
            "prefix_sequence": list(PREFIX_SEQUENCE),
            "aggregators": list(AGGREGATORS),
            "secondary_analyses": list(SECONDARY_ANALYSES),
            "protocol_commit": cfg.protocol_commit,
            "matrix_hash": cfg.matrix_hash,
            "source_commit": source_commit,
            "evaluator_fingerprint": evaluator_fp,
            "evaluator_fingerprint_manifest": evaluator_manifest,
            "dataset_expected_checksum_md5": dataset_expected_checksum,
            "dataset_verification": {
                "dataset": dataset_verification.dataset,
                "resolution": dataset_verification.native_resolution,
                "expected_checksum_md5": dataset_verification.expected_checksum_md5,
                "actual_checksum_md5": dataset_verification.actual_checksum_md5,
                "checksum_verified": dataset_verification.checksum_verified,
                "resized": dataset_verification.resized,
                "verification_method": DATASET_VERIFICATION_METHOD,
                "verification_version": DATASET_VERIFICATION_VERSION,
                "artifact_path": dataset_verification.artifact_path,
            },
            "batching": outcome["batching"],
            "metric_input_contract": seed_cfg.metric_input_contract,
            "evaluation_config_hash": final_test_evaluation_id,
            "split": "test",
            "n_test_samples": len(split.sample_indices),
            "statistical_analysis_fingerprint": analysis_fp,
            "cross_condition_analysis_fingerprint": cross_fp,
            "final_test_runner_fingerprint": runner_fp,
            "authorization_artifact_sha256": authorization.artifact_sha256,
            "authorization_commit": authorization.authorization_commit,
            "test_split_accessed": True,
            "test_predictions_computed": True,
            "test_metrics_computed": True,
        }
        view_manifest = {
            "dataset": cell.dataset,
            "resolution": cell.resolution,
            "tta_seed": resolved_tta_seed,
            "tta_seed_config_sha256": seed_cfg.config_file_sha256,
            "tta_seed_freeze_commit": seed_cfg.freeze_commit,
            "tta_seed_derivation_sha256": seed_cfg.derivation_sha256,
            "n_views": MAX_VIEWS,
            "seed_formula": (
                "sha256(tta_seed|dataset|resolution|sample_index|view_index)[:8 bytes] % (2**31-1)"
            ),
            "sample_indices": split.sample_indices.tolist(),
            "seed_manifest_sha256": seed_manifest_hash,
        }
        metrics = {
            "training_run_id": run_id,
            "evaluation_config_hash": final_test_evaluation_id,
            **outcome["metrics"],
            "latency": _latency_report_to_dict(latency_report),
        }

        def _recompute_n50_mean_probability_accuracy() -> float:
            n50_probs = recompute_mean_probability_prefix(outcome["predictions"]["view_probs"], PRIMARY_N)
            n50_log_probs = np.log(np.clip(n50_probs, 1e-12, 1.0))
            return accuracy(n50_log_probs, split.labels)

        metric_recomputers = {
            "clean.accuracy": (
                metrics["clean"]["accuracy"],
                lambda: recompute_clean_accuracy(outcome["predictions"]["clean_probs"], split.labels),
            ),
            "naive_tta.mean_probability.N50.accuracy": (
                metrics["conditions"]["naive_tta"]["mean_probability"][PRIMARY_N]["accuracy"],
                _recompute_n50_mean_probability_accuracy,
            ),
        }

        _verify_metrics_semantically(outcome["predictions"], metrics, PREFIX_SEQUENCE)

        manifest = persist_and_verify_final_test_completion(
            attempt_dir,
            predictions=outcome["predictions"],
            metrics=metrics,
            metadata=metadata,
            view_manifest=view_manifest,
            prefix_sequence=PREFIX_SEQUENCE,
            metric_recomputers=metric_recomputers,
        )
        test_metrics_persisted = True

        finish_evaluation_attempt(attempt_dir, status, EvaluationRunStatus.COMPLETED)
        primary_artifact_hash = hash_file(attempt_dir / "predictions.npz")
        append_final_test_entry(
            final_test_evaluation_id=final_test_evaluation_id,
            training_run_id=run_id,
            training_attempt=training_result.attempt_number,
            checkpoint_hash=training_result.checkpoint_hash,
            evaluation_config_hash=final_test_evaluation_id,
            evaluation_attempt=status.attempt_number,
            evaluator_fingerprint=evaluator_fp,
            statistical_analysis_fingerprint=analysis_fp,
            cross_condition_analysis_fingerprint=cross_fp,
            final_test_runner_fingerprint=runner_fp,
            authorization_artifact_sha256=authorization.artifact_sha256,
            authorization_commit=authorization.authorization_commit,
            test_split_accessed=True,
            test_predictions_computed=True,
            test_metrics_computed=True,
            test_metrics_persisted=True,
            test_metrics_observed=True,
            status="completed",
            primary_artifact_hash=primary_artifact_hash,
            started_at=status.started_at,
            ended_at=status.ended_at,
            runtime_seconds=status.ended_at - status.started_at,
            ledger_path=final_test_ledger_path,
        )
        return {
            "status": "completed",
            "training_run_id": run_id,
            "final_test_evaluation_id": final_test_evaluation_id,
            "attempt_number": status.attempt_number,
            "artifact_manifest": manifest,
        }
    except Exception as e:
        finish_evaluation_attempt(attempt_dir, status, EvaluationRunStatus.FAILED, failure_reason=str(e))
        append_final_test_entry(
            final_test_evaluation_id=final_test_evaluation_id,
            training_run_id=run_id,
            training_attempt=training_result.attempt_number,
            checkpoint_hash=training_result.checkpoint_hash,
            evaluation_config_hash=final_test_evaluation_id,
            evaluation_attempt=status.attempt_number,
            evaluator_fingerprint=evaluator_fp,
            statistical_analysis_fingerprint=analysis_fp,
            cross_condition_analysis_fingerprint=cross_fp,
            final_test_runner_fingerprint=runner_fp,
            authorization_artifact_sha256=authorization.artifact_sha256,
            authorization_commit=authorization.authorization_commit,
            test_split_accessed=test_split_accessed,
            test_predictions_computed=test_predictions_computed,
            test_metrics_computed=test_metrics_computed,
            test_metrics_persisted=test_metrics_persisted,
            # A metric computed in-memory is observable (e.g. to a caller
            # inspecting the return value or a debugger) EVEN IF persistence
            # never happens -- "not persisted" must never be reported as
            # "not accessed"/"not observed".
            test_metrics_observed=test_metrics_computed,
            status="failed",
            primary_artifact_hash="",
            started_at=status.started_at,
            ended_at=status.ended_at or time.time(),
            runtime_seconds=(status.ended_at or time.time()) - status.started_at,
            failure_stage=failure_stage,
            failure_reason=str(e),
            ledger_path=final_test_ledger_path,
        )
        raise
