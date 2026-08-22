"""Phase 2B.6A: synthetic tests for the final-test orchestrator
(final_test_evaluation.py), the test-only loader (evaluation/test_loader.py),
the final-test result-artifact schema (final_test_result_artifacts.py),
and the CLI (scripts/run_final_test_evaluation.py).

NONE of these tests touch the real 39-cell matrix's real training/
evaluation ledgers, a real checkpoint, a real dataset array, or MPS.
Every heavy dependency (authorization, canonical-training resolution,
device, checkpoint loading, dataset verification, test-split loading,
inference, latency) is either monkeypatched to a controlled fake or
exercised only through tmp_path-rooted ledgers/attempt directories.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import when_tta_hurts.final_test_evaluation as fte
from when_tta_hurts.evaluation.latency import LatencyReport
from when_tta_hurts.final_test_authorization import FinalTestAuthorizationError
from when_tta_hurts.final_test_evaluation import (
    FinalTestAuthorizationRequiredError,
    plan_final_test_evaluation,
    run_final_test_evaluation,
)
from when_tta_hurts.validation_evaluation import PREFIX_SEQUENCE


class _FakeCell:
    def __init__(self, run_id_value="fake-run-a", dataset="pathmnist", resolution=28):
        self._run_id = run_id_value
        self.dataset = dataset
        self.resolution = resolution
        self.model = "small_cnn"
        self.normalization = "batchnorm"
        self.training_policy = "none"
        self.seed = 0

    def run_id(self):
        return self._run_id


class _FakeTrainingResult:
    def __init__(self, attempt_number=1, checkpoint_hash="chk-a"):
        self.attempt_number = attempt_number
        self.checkpoint_hash = checkpoint_hash


class _FakeExpanded:
    def __init__(self, cells):
        self.cells = cells
        self.source_config_hash = "fake-matrix-hash"


class _FakeAuthorization:
    def __init__(self, run_id="fake-run-a"):
        self.authorized_cells_by_run_id = {run_id: {"training_attempt": 1, "checkpoint_hash": "chk-a"}}
        self.artifact_sha256 = "fake-artifact-sha"
        self.authorization_commit = "fake-authorization-commit"


class _FakeSeedConfig:
    confirmatory_tta_seed = 1306178015
    prefix_sequence = PREFIX_SEQUENCE
    config_file_sha256 = "fake-seed-sha"
    freeze_commit = "fake-freeze-commit"
    derivation_sha256 = "fake-derivation-sha"
    metric_input_contract = "probability_native_v1"


class _CalledUnexpectedly(RuntimeError):
    pass


def _raise_if_called(*_a, **_k):
    raise _CalledUnexpectedly("heavy-dependency function called when it should not have been reached")


def _patch_common(monkeypatch, *, run_id="fake-run-a"):
    """Patch every step up through step 5 (fingerprint/identity binding)
    to fixed, controlled fakes -- steps 6+ are set by each test."""
    cell = _FakeCell(run_id)
    training_result = _FakeTrainingResult()
    authorization = _FakeAuthorization(run_id)

    monkeypatch.setattr(fte, "verify_final_test_authorization", lambda **k: authorization)
    monkeypatch.setattr(fte, "resolve_canonical_training_completion", lambda rid, mp: (cell, training_result))
    monkeypatch.setattr(fte, "load_frozen_tta_seed_config", lambda path: _FakeSeedConfig())
    monkeypatch.setattr(fte, "parse_and_validate_matrix", lambda mp, **k: _FakeExpanded([cell]))
    monkeypatch.setattr(fte, "compute_evaluator_fingerprint", lambda: ("fake-evaluator-fp", {}))
    monkeypatch.setattr(fte, "compute_analysis_fingerprint", lambda: ("fake-analysis-fp", None))
    monkeypatch.setattr(fte, "compute_cross_condition_fingerprint", lambda: ("fake-cross-fp", None))
    monkeypatch.setattr(fte, "compute_final_test_runner_fingerprint", lambda: ("fake-runner-fp", None))
    monkeypatch.setattr(fte, "expected_official_checksum", lambda dataset, resolution: "0" * 32)
    monkeypatch.setattr(fte, "_git_commit_hash", lambda: "fake-source-commit")
    monkeypatch.setattr(fte, "require_clean_working_tree", lambda: None)
    return cell, training_result, authorization


def _read_ledger_rows(ledger_path):
    import csv

    with open(ledger_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Authorization-before-everything-else ordering
# ---------------------------------------------------------------------------


def test_authorization_failure_stops_before_device_or_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        fte,
        "verify_final_test_authorization",
        lambda **k: (_ for _ in ()).throw(FinalTestAuthorizationError("no authorization")),
    )
    monkeypatch.setattr(fte, "resolve_canonical_training_completion", _raise_if_called)

    with pytest.raises(FinalTestAuthorizationError):
        run_final_test_evaluation(
            "fake-run-a",
            device_resolver=_raise_if_called,
            root=tmp_path / "final_test",
            final_test_ledger_path=tmp_path / "ledger.csv",
        )
    assert not (tmp_path / "final_test").exists()
    assert not (tmp_path / "ledger.csv").exists()


def test_run_id_not_in_authorized_cells_stops_before_device(monkeypatch, tmp_path):
    _patch_common(monkeypatch, run_id="fake-run-a")
    # Authorization only knows about "fake-run-a"; request a different ID.
    monkeypatch.setattr(fte, "resolve_canonical_training_completion", _raise_if_called)

    with pytest.raises(FinalTestAuthorizationRequiredError):
        run_final_test_evaluation(
            "some-other-run",
            device_resolver=_raise_if_called,
            root=tmp_path / "final_test",
            final_test_ledger_path=tmp_path / "ledger.csv",
        )
    assert not (tmp_path / "final_test").exists()


def test_idempotent_skip_before_heavy_dependencies(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    skip_payload = {"attempt_number": 1, "status": "completed"}
    monkeypatch.setattr(fte, "check_final_test_evaluation_skip", lambda *a, **k: skip_payload)
    monkeypatch.setattr(fte, "load_and_verify_canonical_checkpoint", _raise_if_called)

    result = run_final_test_evaluation(
        "fake-run-a",
        device_resolver=_raise_if_called,
        root=tmp_path / "final_test",
        final_test_ledger_path=tmp_path / "ledger.csv",
    )
    assert result["status"] == "skipped_completed"
    assert not (tmp_path / "final_test").exists()


# ---------------------------------------------------------------------------
# Truthful failure-stage / lifecycle-flag accounting
# ---------------------------------------------------------------------------


def test_failure_before_test_access_records_truthful_flags(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(fte, "check_final_test_evaluation_skip", lambda *a, **k: None)
    monkeypatch.setattr(fte, "select_device", lambda name: object())

    def _boom(*a, **k):
        raise RuntimeError("checkpoint missing")

    monkeypatch.setattr(fte, "load_and_verify_canonical_checkpoint", _boom)
    monkeypatch.setattr(fte, "load_final_test_split", _raise_if_called)

    ledger_path = tmp_path / "ledger.csv"
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        run_final_test_evaluation(
            "fake-run-a",
            root=tmp_path / "final_test",
            final_test_ledger_path=ledger_path,
        )

    rows = _read_ledger_rows(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "failed"
    assert row["failure_stage"] == "checkpoint_load"
    assert row["test_split_accessed"] == "False"
    assert row["test_predictions_computed"] == "False"
    assert row["test_metrics_observed"] == "False"
    assert "checkpoint missing" in row["failure_reason"]


def test_failure_after_test_access_records_truthful_flags(monkeypatch, tmp_path):
    cell, training_result, authorization = _patch_common(monkeypatch)
    monkeypatch.setattr(fte, "select_device", lambda name: object())
    monkeypatch.setattr(fte, "check_final_test_evaluation_skip", lambda *a, **k: None)
    monkeypatch.setattr(fte, "load_and_verify_canonical_checkpoint", lambda *a, **k: object())

    class _FakeVerification:
        dataset = "pathmnist"
        native_resolution = 28
        expected_checksum_md5 = "0" * 32
        actual_checksum_md5 = "0" * 32
        checksum_verified = True
        resized = False
        artifact_path = "data/raw/pathmnist.npz"

    monkeypatch.setattr(fte, "verify_official_dataset_artifact", lambda *a, **k: _FakeVerification())

    fake_split = type(
        "FakeSplit",
        (),
        {
            "images": None,
            "labels": np.array([0, 1]),
            "sample_indices": np.array([0, 1]),
            "dataset": "pathmnist",
            "resolution": 28,
        },
    )()
    monkeypatch.setattr(fte, "load_final_test_split", lambda *a, **k: fake_split)

    fake_outcome = {
        "predictions": {"labels": np.array([0, 1])},
        "metrics": {"clean": {}, "conditions": {}},
        "batching": {"bn_adaptation_applicable": False},
    }
    monkeypatch.setattr(fte, "compute_validation_evaluation", lambda *a, **k: fake_outcome)

    def _boom_latency(*a, **k):
        raise RuntimeError("latency measurement crashed")

    monkeypatch.setattr(fte, "compute_evaluation_latency_report", _boom_latency)
    monkeypatch.setattr(fte, "persist_and_verify_final_test_completion", _raise_if_called)

    ledger_path = tmp_path / "ledger.csv"
    with pytest.raises(RuntimeError, match="latency measurement crashed"):
        run_final_test_evaluation(
            "fake-run-a",
            root=tmp_path / "final_test",
            final_test_ledger_path=ledger_path,
        )

    rows = _read_ledger_rows(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "failed"
    assert row["failure_stage"] == "latency"
    assert row["test_split_accessed"] == "True"
    assert row["test_predictions_computed"] == "True"
    assert row["test_metrics_computed"] == "True"
    assert row["test_metrics_persisted"] == "False"
    # Computed-in-memory metrics are observable even though never persisted --
    # "not persisted" must never be silently reported as "not observed".
    assert row["test_metrics_observed"] == "True"


def test_completed_happy_path_records_all_flags_true(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(fte, "select_device", lambda name: object())
    monkeypatch.setattr(fte, "check_final_test_evaluation_skip", lambda *a, **k: None)
    monkeypatch.setattr(fte, "load_and_verify_canonical_checkpoint", lambda *a, **k: object())

    class _FakeVerification:
        dataset = "pathmnist"
        native_resolution = 28
        expected_checksum_md5 = "0" * 32
        actual_checksum_md5 = "0" * 32
        checksum_verified = True
        resized = False
        artifact_path = "data/raw/pathmnist.npz"

    monkeypatch.setattr(fte, "verify_official_dataset_artifact", lambda *a, **k: _FakeVerification())

    fake_split = type(
        "FakeSplit",
        (),
        {
            "images": None,
            "labels": np.array([0, 1]),
            "sample_indices": np.array([0, 1]),
            "dataset": "pathmnist",
            "resolution": 28,
        },
    )()
    monkeypatch.setattr(fte, "load_final_test_split", lambda *a, **k: fake_split)

    fake_outcome = {
        "predictions": {"labels": np.array([0, 1]), "clean_probs": np.array([[0.9, 0.1], [0.2, 0.8]])},
        "metrics": {
            "clean": {"accuracy": 1.0},
            "conditions": {"naive_tta": {"mean_probability": {50: {"accuracy": 1.0}}}},
        },
        "batching": {"bn_adaptation_applicable": False},
    }
    monkeypatch.setattr(fte, "compute_validation_evaluation", lambda *a, **k: fake_outcome)
    monkeypatch.setattr(fte, "recompute_clean_accuracy", lambda *a, **k: 1.0)
    monkeypatch.setattr(
        fte, "recompute_mean_probability_prefix", lambda *a, **k: np.array([[0.9, 0.1], [0.2, 0.8]])
    )
    monkeypatch.setattr(fte, "accuracy", lambda *a, **k: 1.0)

    fake_latency = LatencyReport(
        clean_latency_seconds=0.1,
        tta_latency_seconds_by_n={n: 0.1 for n in PREFIX_SEQUENCE},
        per_sample_latency_seconds_by_n={n: 0.05 for n in PREFIX_SEQUENCE},
        compute_multiplier_by_n={n: 1.0 for n in PREFIX_SEQUENCE},
        n_samples=2,
    )
    monkeypatch.setattr(fte, "compute_evaluation_latency_report", lambda *a, **k: fake_latency)
    monkeypatch.setattr(fte, "build_view_seed_manifest", lambda *a, **k: [])
    monkeypatch.setattr(fte, "_verify_metrics_semantically", lambda *a, **k: None)

    fake_manifest = {"artifacts": [{"path": "predictions.npz", "sha256": "deadbeef", "size_bytes": 1}]}

    def _fake_persist(attempt_dir, **kwargs):
        Path(attempt_dir).mkdir(parents=True, exist_ok=True)
        (Path(attempt_dir) / "predictions.npz").write_bytes(b"\x00")
        return fake_manifest

    monkeypatch.setattr(fte, "persist_and_verify_final_test_completion", _fake_persist)

    ledger_path = tmp_path / "ledger.csv"
    result = run_final_test_evaluation(
        "fake-run-a",
        root=tmp_path / "final_test",
        final_test_ledger_path=ledger_path,
    )
    assert result["status"] == "completed"

    rows = _read_ledger_rows(ledger_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["test_split_accessed"] == "True"
    assert row["test_predictions_computed"] == "True"
    assert row["test_metrics_computed"] == "True"
    assert row["test_metrics_persisted"] == "True"
    assert row["test_metrics_observed"] == "True"
    assert row["split"] == "test"
    assert row["confirmatory"] == "True"


# ---------------------------------------------------------------------------
# Plan mode: side-effect-free, test-data-free, real-repo report
# ---------------------------------------------------------------------------


def test_plan_mode_is_side_effect_free(tmp_path):
    before = set(tmp_path.iterdir())
    plan_final_test_evaluation()
    after = set(tmp_path.iterdir())
    assert before == after


def test_plan_mode_never_reads_predictions_checkpoints_or_touches_torch(monkeypatch):
    real_load = np.load

    def guarded_np_load(path, *a, **k):
        if "predictions.npz" in str(path):
            raise AssertionError("plan mode must never read predictions.npz")
        return real_load(path, *a, **k)

    monkeypatch.setattr(np, "load", guarded_np_load)

    import torch

    monkeypatch.setattr(torch, "load", _raise_if_called)
    plan_final_test_evaluation()


def test_plan_mode_reports_real_repo_state():
    """Authorization was absent when this test was first written (Phase
    2B.6A) and is now genuinely approved (Phase 2B.6B,
    docs/phase2b_final_test_authorization_audit.md) -- this assertion is
    updated to the current, legitimate state rather than a hardcoded
    assumption about authorization's presence/absence, exactly the same
    stale-real-state-assertion pattern this project has hit and fixed
    several times before. plan_final_test_evaluation() itself remains
    side-effect-free and test-data-free regardless of authorization
    state, which is what this test actually verifies."""
    report = plan_final_test_evaluation()
    assert report["n_cells_total"] == 39
    assert report["n_cells_training_eligible"] == 39
    assert report["authorization_status"] in ("approved", "missing_or_not_approved")
    assert report["test_split_accessed"] is False
    assert len(report["evaluator_fingerprint"]) > 0
    assert len(report["final_test_runner_fingerprint"]) > 0


def test_plan_mode_matrix_ordering_matches_source():
    from when_tta_hurts.matrix import parse_and_validate_matrix

    expanded = parse_and_validate_matrix("configs/experiment_matrix.yaml", block_d_gate_passed=True)
    expected_order = [c.run_id() for c in expanded.cells]
    report = plan_final_test_evaluation()
    assert [c["run_id"] for c in report["cells"]] == expected_order


# ---------------------------------------------------------------------------
# No retry, no bypass, no scientific-metric printing on the CLI
# ---------------------------------------------------------------------------


def test_run_final_test_evaluation_has_no_retry_or_bypass_parameters():
    sig = inspect.signature(run_final_test_evaluation)
    param_names = set(sig.parameters.keys())
    for forbidden in ("retry", "force", "bypass", "skip_authorization", "unlock"):
        assert forbidden not in param_names


def test_cli_has_no_bypass_flags():
    script = Path("scripts/run_final_test_evaluation.py").read_text()
    import ast

    tree = ast.parse(script)
    add_argument_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]
    flag_strings = set()
    for call in add_argument_calls:
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                flag_strings.add(arg.value)
    forbidden_substrings = (
        "split",
        "force",
        "retry",
        "override",
        "seed",
        "policy",
        "batch",
        "backend",
        "env",
    )
    for flag in flag_strings:
        lowered = flag.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, f"CLI flag {flag!r} looks like a forbidden override."


def _load_cli_module(name):
    """Load scripts/run_final_test_evaluation.py fresh. The freshly loaded
    module still imports run_final_test_evaluation from the ALREADY-
    CACHED when_tta_hurts.final_test_evaluation module (sys.modules), so
    patching attributes on `fte` (imported at the top of this test file)
    correctly affects what the freshly loaded CLI module calls -- this is
    the namespace-correctness property
    docs/phase2b_final_test_accidental_access_incident.md's root-cause
    fix depends on."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, Path("scripts/run_final_test_evaluation.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_evaluate_test_refuses_when_authorization_fails_never_depends_on_real_repo_state(
    monkeypatch,
):
    """Regression test for
    docs/phase2b_final_test_accidental_access_incident.md: the ORIGINAL
    version of this test relied on the real repo's authorization artifact
    being permanently absent, which stopped being true the moment
    Phase 2B.6B's authorization was committed -- and its one intended
    safety net patched the WRONG namespace
    (when_tta_hurts.devices.select_device instead of
    when_tta_hurts.final_test_evaluation.select_device), so real
    execution proceeded, unmocked, inside a pytest process.

    This version NEVER depends on whether the real authorization artifact
    exists: it forces authorization to fail by patching
    verify_final_test_authorization in the EXACT namespace
    run_final_test_evaluation() actually looks it up in
    (when_tta_hurts.final_test_evaluation, i.e. the `fte` module imported
    at the top of this file) -- never the definition module
    (when_tta_hurts.final_test_authorization) and never an unrelated
    module. Device selection and canonical-training resolution are also
    patched, in the SAME correct namespace, as sentinels that hard-fail
    the test if ever reached -- proving authorization-failure gates
    everything before any heavy dependency, deterministically, regardless
    of real repo state, forever."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_test_evaluation.py",
            "evaluate-test",
            "--run-id",
            "A-pathmnist-28px-batchnorm-policy-none-s0",
        ],
    )

    def _raise_auth_error(**k):
        raise FinalTestAuthorizationError("synthetic: authorization intentionally failed for this test")

    monkeypatch.setattr(fte, "verify_final_test_authorization", _raise_auth_error)
    monkeypatch.setattr(fte, "resolve_canonical_training_completion", _raise_if_called)
    monkeypatch.setattr(fte, "select_device", _raise_if_called)

    # Snapshot rather than assume non-existence: the affected cell from
    # docs/phase2b_final_test_accidental_access_incident.md permanently
    # retains its preserved attempt_001 directory -- this test must prove
    # NOTHING NEW was created by THIS invocation, not that the directory
    # tree is empty.
    cell_dir = Path("artifacts/final_test/A-pathmnist-28px-batchnorm-policy-none-s0")
    attempts_before = sorted(p.name for p in cell_dir.iterdir()) if cell_dir.exists() else []

    module = _load_cli_module("run_final_test_evaluation_cli_auth_fail")
    exit_code = module.main()

    assert exit_code == 1
    attempts_after = sorted(p.name for p in cell_dir.iterdir()) if cell_dir.exists() else []
    assert attempts_after == attempts_before, (
        "authorization-failure refusal must never allocate a new attempt"
    )


def test_cli_evaluate_test_refuses_for_unrecognized_run_id_real_repo(monkeypatch):
    """Exercises the REAL, current, valid authorization verifier (safe:
    verify_final_test_authorization() never touches MPS/checkpoint/
    dataset arrays -- confirmed in test_final_test_authorization.py) but
    with a run ID that can never be a real matrix cell, so this can never
    resolve to, allocate an attempt for, or touch any real authorized
    cell -- and therefore never depends on whether authorization exists
    or not. Device selection and canonical-training resolution are
    patched, in the correct namespace, as sentinels proving the
    unauthorized-run-ID refusal happens before either is ever reached."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_test_evaluation.py",
            "evaluate-test",
            "--run-id",
            "not-a-real-matrix-run-id-zzz",
        ],
    )
    monkeypatch.setattr(fte, "resolve_canonical_training_completion", _raise_if_called)
    monkeypatch.setattr(fte, "select_device", _raise_if_called)

    module = _load_cli_module("run_final_test_evaluation_cli_unrecognized_run_id")
    exit_code = module.main()

    assert exit_code == 1
    assert not Path("artifacts/final_test/not-a-real-matrix-run-id-zzz").exists()


def test_regression_wrong_namespace_monkeypatch_does_not_affect_final_test_evaluation_select_device(
    monkeypatch,
):
    """Direct regression test reproducing the ORIGINAL incident's exact
    defect mechanism (never invokes real MPS): final_test_evaluation.py
    does `from when_tta_hurts.devices import select_device`, binding its
    OWN name in its OWN module namespace at import time. Patching
    `when_tta_hurts.devices.select_device` (the definition module) can
    NEVER affect that already-bound reference -- proven here by patching
    both namespaces to distinguishable sentinel values and confirming
    `fte.select_device` still returns ITS OWN sentinel, unaffected by the
    other module's patch."""
    import when_tta_hurts.devices as devices_module

    def _wrong_namespace_sentinel(name):
        return "WRONG_NAMESPACE_SENTINEL"

    def _correct_namespace_sentinel(name):
        return "CORRECT_NAMESPACE_SENTINEL"

    # Patch the DEFINITION module (the historically wrong target)...
    monkeypatch.setattr(devices_module, "select_device", _wrong_namespace_sentinel)
    # ...and independently confirm fte's own already-imported reference
    # is a completely separate binding, unaffected by the patch above,
    # by explicitly setting it to a DIFFERENT sentinel and proving THAT
    # one -- not the devices-module patch -- is what fte actually uses.
    monkeypatch.setattr(fte, "select_device", _correct_namespace_sentinel)

    assert fte.select_device("mps") == "CORRECT_NAMESPACE_SENTINEL"
    assert devices_module.select_device("mps") == "WRONG_NAMESPACE_SENTINEL"
    # The two are proven independent: patching one never touches the other.


# ---------------------------------------------------------------------------
# No test-split reachability
# ---------------------------------------------------------------------------


def test_no_test_split_symbol_reachable_from_orchestrator_module():
    source = inspect.getsource(fte)
    assert "allow_test" not in source
    assert "load_test" not in source


def test_no_scientific_metric_printed_by_cli_redaction_helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_final_test_evaluation_cli2", Path("scripts/run_final_test_evaluation.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fake_result = {
        "status": "completed",
        "training_run_id": "fake-run",
        "final_test_evaluation_id": "fake-id",
        "attempt_number": 1,
        "artifact_manifest": {"artifacts": [{"path": "predictions.npz", "sha256": "abc", "size_bytes": 1}]},
        "metrics": {"clean": {"accuracy": 0.9999}},  # must NEVER leak into the redacted output
    }
    redacted = module._redact_for_print(fake_result)
    assert "metrics" not in redacted
    assert "accuracy" not in json.dumps(redacted)
    assert redacted["status"] == "completed"
    assert redacted["artifact_hashes"] == {"predictions.npz": "abc"}


# ---------------------------------------------------------------------------
# Incident-response structural guards
# (docs/phase2b_final_test_accidental_access_incident.md, Part D)
# ---------------------------------------------------------------------------


def test_static_every_cli_evaluate_test_invocation_patches_a_heavy_dependency():
    """AST-based proof that every test function in this file which drives
    the CLI in evaluate-test mode also monkeypatches at least one heavy-
    dependency symbol on `fte` (the correct namespace) before calling
    module.main() -- so no future CLI test can silently regress to the
    wrong-namespace/no-mock pattern that caused
    docs/phase2b_final_test_accidental_access_incident.md."""
    import ast

    source = Path("tests/test_final_test_evaluation.py").read_text()
    tree = ast.parse(source)

    guarded_symbols = {
        "verify_final_test_authorization",
        "resolve_canonical_training_completion",
        "select_device",
    }

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "test_static_every_cli_evaluate_test_invocation_patches_a_heavy_dependency":
            continue  # this check's own docstring mentions "evaluate-test"; it does not drive the CLI
        source_segment = ast.get_source_segment(source, node) or ""
        if "evaluate-test" not in source_segment:
            continue  # this function does not drive the CLI's evaluate-test mode
        patched_targets = set()
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "setattr"
            ):
                if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                    patched_targets.add(call.args[1].value)
        if not (patched_targets & guarded_symbols):
            violations.append(node.name)

    assert violations == [], (
        f"CLI-evaluate-test test function(s) {violations} do not patch any of "
        f"{sorted(guarded_symbols)} in the correct namespace -- could silently reach "
        f"the real final-test runner."
    )


def test_cli_evaluate_test_never_indexes_real_test_arrays_even_with_valid_authorization(monkeypatch):
    """Even with the real, valid authorization artifact in place, confirms
    no CLI evaluate-test invocation can reach np.load on a
    predictions.npz-shaped path -- exercises the REAL, current
    authorization verifier (safe: proven metadata-only in
    test_final_test_authorization.py) via a run ID that can never be a
    real matrix cell, combined with an active guard on np.load."""
    real_load = np.load

    def _guarded_load(path, *a, **k):
        if str(path).endswith(".npz"):
            raise AssertionError(f"a CLI test attempted to load an .npz array-shaped path: {path}")
        return real_load(path, *a, **k)

    monkeypatch.setattr(np, "load", _guarded_load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_final_test_evaluation.py",
            "evaluate-test",
            "--run-id",
            "not-a-real-matrix-run-id-zzz",
        ],
    )
    monkeypatch.setattr(fte, "resolve_canonical_training_completion", _raise_if_called)
    monkeypatch.setattr(fte, "select_device", _raise_if_called)

    module = _load_cli_module("run_final_test_evaluation_cli_array_guard")
    exit_code = module.main()
    assert exit_code == 1
