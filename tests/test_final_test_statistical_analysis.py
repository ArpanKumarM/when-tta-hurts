"""Phase 2B.7A: tests for the additive, final-test-only statistical-
analysis and cross-condition-addendum runner. Every test uses either
(a) the real, already-authorized repository state in READ-ONLY plan mode
(no prediction array is ever loaded by plan mode, verified explicitly),
or (b) fully synthetic tmp_path fixtures with a directly-constructed
FinalTestAuthorization object (never a real git repo, never real
predictions) for real-analysis-mode tests. NONE invoke evaluate-test,
load real predictions.npz outside synthetic fixtures, or write to any
real production ledger/artifact path.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

import when_tta_hurts.cross_condition_addendum as cca
import when_tta_hurts.final_test_statistical_analysis as ftsa
import when_tta_hurts.statistical_analysis as sa
from when_tta_hurts.final_test_analysis_ledger import append_final_test_analysis_entry as real_append
from when_tta_hurts.final_test_analysis_ledger import (
    ensure_final_test_analysis_ledger_exists as real_ensure,
)
from when_tta_hurts.final_test_analysis_ledger import existing_completed_attempt as real_existing_completed
from when_tta_hurts.final_test_analysis_ledger import (
    next_final_test_analysis_attempt_number as real_next_attempt,
)
from when_tta_hurts.final_test_authorization import FinalTestAuthorization
from when_tta_hurts.final_test_result_artifacts import build_final_test_artifact_manifest

# ---------------------------------------------------------------------------
# Mathematical equivalence: the frozen math must be REUSED, never
# reimplemented.
# ---------------------------------------------------------------------------


def test_reuses_frozen_math_functions_by_identity_not_reimplementation():
    assert ftsa.paired_bootstrap_ci is sa.paired_bootstrap_ci
    assert ftsa.mcnemar_test is sa.mcnemar_test
    assert ftsa.benjamini_hochberg is sa.benjamini_hochberg
    assert ftsa.effect_sizes is sa.effect_sizes
    assert ftsa.derive_family_cells is sa.derive_family_cells
    assert ftsa.compute_analysis_id is sa.compute_analysis_id
    assert ftsa.derive_fixed_pairs is cca.derive_fixed_pairs
    assert ftsa.load_addendum_spec is cca.load_addendum_spec
    assert ftsa.derive_bootstrap_seed is cca.derive_bootstrap_seed
    assert ftsa.did_bootstrap_ci is cca.did_bootstrap_ci


# ---------------------------------------------------------------------------
# No validation fallback, no bypass.
# ---------------------------------------------------------------------------


def test_no_public_function_accepts_a_validation_root_or_split_override():
    forbidden_param_names = {
        "validation_evaluation_root",
        "validation_root",
        "split",
        "official_test_split",
        "test_split_path",
        "env",
        "environ",
    }
    public_functions = [
        ftsa.plan_final_test_statistical_analysis,
        ftsa.plan_final_test_cross_condition_addendum,
        ftsa.compute_final_test_family_analysis,
        ftsa.compute_final_test_hypothesis_did,
        ftsa.compute_final_test_pair_did,
        ftsa.resolve_final_test_canonical_evaluation_identity,
    ]
    for fn in public_functions:
        params = set(inspect.signature(fn).parameters)
        overlap = params & forbidden_param_names
        assert not overlap, f"{fn.__name__} accepts forbidden parameter(s): {overlap}"


def test_source_has_no_environment_variable_or_cli_flag_reads():
    import os

    source = inspect.getsource(ftsa)
    assert "os.environ" not in source
    assert "os.getenv" not in source
    assert "sys.argv" not in source
    assert os  # keep the import meaningful without asserting on it directly


# ---------------------------------------------------------------------------
# Plan mode: side-effect-free, zero prediction loads, against the REAL,
# already-authorized 39-cell repository state.
# ---------------------------------------------------------------------------


def test_plan_family_analysis_resolves_all_39_cells_and_every_family(monkeypatch):
    calls = []
    real_load = np.load

    def _tracking_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(ftsa.np, "load", _tracking_load)

    report = ftsa.plan_final_test_statistical_analysis()

    assert report["authorization_status"] == "approved"
    assert report["n_cells_total"] == 39
    assert report["n_completed_consumed"] == 39
    assert report["n_pending"] == 0
    assert report["n_invalid"] == 0
    assert report["families"]["H1"]["n_cells_required"] == 24
    assert report["families"]["H1"]["complete"] is True
    assert report["families"]["H2"]["n_cells_required"] == 30
    assert report["families"]["H2"]["complete"] is True
    assert report["families"]["H3"]["n_cells_required"] == 12
    assert report["families"]["H3"]["complete"] is True
    assert report["families"]["BLOCK_C"]["n_cells_required"] == 3
    assert report["families"]["BLOCK_C"]["complete"] is True
    assert calls == [], "plan mode must never call numpy.load"


def test_plan_cross_condition_addendum_resolves_expected_pair_counts(monkeypatch):
    calls = []
    real_load = np.load
    monkeypatch.setattr(ftsa.np, "load", lambda *a, **k: (calls.append(a), real_load(*a, **k))[1])

    report = ftsa.plan_final_test_cross_condition_addendum()

    assert report["authorization_status"] == "approved"
    assert report["hypotheses"]["H1"]["n_pairs_required"] == 12
    assert report["hypotheses"]["H1"]["complete"] is True
    assert report["hypotheses"]["H2"]["n_pairs_required"] == 12
    assert report["hypotheses"]["H2"]["complete"] is True
    assert report["hypotheses"]["H3"]["n_pairs_required"] == 6
    assert report["hypotheses"]["H3"]["complete"] is True
    assert calls == [], "plan mode must never call numpy.load"


def test_plan_modes_leave_no_real_repository_side_effects():
    from pathlib import Path

    before_analysis = Path("artifacts/final_test_analysis").exists()
    before_cross = Path("artifacts/final_test_cross_condition").exists()
    before_ledger = Path("artifacts/ledger_final_test_analysis.csv").exists()

    ftsa.plan_final_test_statistical_analysis()
    ftsa.plan_final_test_cross_condition_addendum()

    assert Path("artifacts/final_test_analysis").exists() == before_analysis
    assert Path("artifacts/final_test_cross_condition").exists() == before_cross
    assert Path("artifacts/ledger_final_test_analysis.csv").exists() == before_ledger


def test_plan_reports_not_approved_when_authorization_fails(monkeypatch):
    from when_tta_hurts.final_test_authorization import FinalTestAuthorizationError

    def _raise():
        raise FinalTestAuthorizationError("synthetic failure")

    monkeypatch.setattr(ftsa, "verify_final_test_authorization", _raise)
    report = ftsa.plan_final_test_statistical_analysis()
    assert report["authorization_status"] == "not_approved"
    assert report["families"] == {}

    report2 = ftsa.plan_final_test_cross_condition_addendum()
    assert report2["authorization_status"] == "not_approved"
    assert report2["hypotheses"] == {}


# ---------------------------------------------------------------------------
# Cell 1's carried-forward generation-3 provenance, against the REAL repo.
# ---------------------------------------------------------------------------


def test_cell_1_resolves_eligible_under_its_historical_generation3_sha():
    from when_tta_hurts.final_test_authorization import verify_final_test_authorization

    authorization = verify_final_test_authorization()
    run_id = "A-pathmnist-28px-batchnorm-policy-none-s0"
    identity = ftsa.resolve_final_test_canonical_evaluation_identity(run_id, authorization)
    assert identity["evaluation_status"] == "eligible"
    assert (
        identity["authorization_artifact_sha256"]
        == ftsa.KNOWN_HISTORICAL_AUTHORIZATION_SHA256_BY_RUN_ID[run_id]
    )
    assert identity["authorization_artifact_sha256"] != authorization.artifact_sha256


def test_every_other_completed_cell_binds_to_current_authorization_sha():
    from when_tta_hurts.final_test_authorization import verify_final_test_authorization

    authorization = verify_final_test_authorization()
    other_run_id = "A-pathmnist-28px-batchnorm-policy-none-s1"
    assert other_run_id not in ftsa.KNOWN_HISTORICAL_AUTHORIZATION_SHA256_BY_RUN_ID
    identity = ftsa.resolve_final_test_canonical_evaluation_identity(other_run_id, authorization)
    assert identity["evaluation_status"] == "eligible"
    assert identity["authorization_artifact_sha256"] == authorization.artifact_sha256


# ---------------------------------------------------------------------------
# Synthetic fixtures for real-analysis-mode tests: fully offline, never
# touches git, never touches real production paths.
# ---------------------------------------------------------------------------


def _fake_authorization(run_ids, classifications=None, artifact_sha256="fake-auth-sha"):
    classifications = classifications or {r: "completed_consumed" for r in run_ids}
    return FinalTestAuthorization(
        status="approved",
        schema_version="phase2b.6d-v2",
        approval_timestamp="2026-01-01T00:00:00Z",
        phase2b_protocol_commit="fake-commit",
        matrix_commit="fake-commit",
        cross_condition_addendum_commit="fake-commit",
        evaluator_fingerprint="fake-evaluator-fp",
        statistical_analysis_fingerprint="fake-analysis-fp",
        cross_condition_analysis_fingerprint="fake-cross-fp",
        final_test_runner_fingerprint="fake-runner-fp",
        authorized_cells_by_run_id={
            r: {
                "authorized_final_test_attempt": 1,
                "checkpoint_hash": "chk",
                "training_attempt": 1,
                "dataset": "pathmnist",
                "resolution": 28,
            }
            for r in run_ids
        },
        official_dataset_checksums={"pathmnist@28": "0" * 32},
        artifact_sha256=artifact_sha256,
        authorization_commit="fake-auth-commit",
        supersedes_authorization_sha256=None,
        supersedes_authorization_commit=None,
        incident_record_commit=None,
        recovery_policy_commit=None,
        no_further_retry=None,
        cell_classifications=classifications,
    )


def _write_synthetic_cell(
    tmp_path, run_id, n_samples=20, n_views=3, labels=None, sample_indices=None, seed=0
):
    """Writes a minimal synthetic final-test attempt directory
    (predictions.npz + metadata.json + artifact_manifest.json only --
    the exact files _load_final_test_cell_correctness() reads)."""
    rng = np.random.default_rng(seed)
    attempt_dir = tmp_path / run_id / "attempt_001"
    attempt_dir.mkdir(parents=True)

    if labels is not None:
        n_samples = len(labels)
    if labels is None:
        labels = rng.integers(0, 2, size=n_samples)
    if sample_indices is None:
        sample_indices = np.arange(n_samples)

    clean_logits = rng.normal(size=(n_samples, 2))
    clean_probs = np.exp(clean_logits) / np.exp(clean_logits).sum(axis=-1, keepdims=True)
    view_logits = rng.normal(size=(n_views, n_samples, 2))
    view_probs = np.exp(view_logits) / np.exp(view_logits).sum(axis=-1, keepdims=True)

    np.savez(
        attempt_dir / "predictions.npz",
        labels=labels,
        sample_indices=sample_indices,
        clean_probs=clean_probs,
        view_probs=view_probs,
    )
    (attempt_dir / "metadata.json").write_text(json.dumps({"checkpoint_hash": "chk"}))

    manifest = build_final_test_artifact_manifest(attempt_dir, filenames=("predictions.npz", "metadata.json"))
    (attempt_dir / "artifact_manifest.json").write_text(json.dumps(manifest))
    return attempt_dir


def _write_ledger_row(ledger_path, run_id, authorization_artifact_sha256, evaluation_id=None):
    import csv

    evaluation_id = evaluation_id or f"eval-{run_id}"
    fieldnames = [
        "training_run_id",
        "evaluation_attempt",
        "final_test_evaluation_id",
        "authorization_artifact_sha256",
    ]
    write_header = not ledger_path.exists()
    with ledger_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "training_run_id": run_id,
                "evaluation_attempt": 1,
                "final_test_evaluation_id": evaluation_id,
                "authorization_artifact_sha256": authorization_artifact_sha256,
            }
        )


# ---------------------------------------------------------------------------
# Authorization/lifecycle enforcement and unauthorized-binding rejection.
# ---------------------------------------------------------------------------


def test_resolve_identity_rejects_wrong_authorization_binding(tmp_path):
    run_id = "run-a"
    ledger = tmp_path / "ledger_final_test.csv"
    _write_ledger_row(ledger, run_id, authorization_artifact_sha256="WRONG-SHA")
    authorization = _fake_authorization([run_id], artifact_sha256="expected-sha")

    identity = ftsa.resolve_final_test_canonical_evaluation_identity(run_id, authorization, ledger)
    assert identity["evaluation_status"] == "unauthorized_binding"


@pytest.mark.parametrize("classification", ["pending", "invalid"])
def test_resolve_identity_rejects_non_completed_consumed_cells(tmp_path, classification):
    run_id = "run-a"
    authorization = _fake_authorization([run_id], classifications={run_id: classification})
    identity = ftsa.resolve_final_test_canonical_evaluation_identity(run_id, authorization)
    assert identity["evaluation_status"] == "not_completed_consumed"
    assert identity["classification"] == classification


def test_resolve_identity_unknown_run_id():
    authorization = _fake_authorization(["run-a"])
    identity = ftsa.resolve_final_test_canonical_evaluation_identity("run-does-not-exist", authorization)
    assert identity["evaluation_status"] == "unknown_run_id"


# ---------------------------------------------------------------------------
# Label / sample-index mismatch rejection for paired (cross-condition)
# comparisons.
# ---------------------------------------------------------------------------


def test_compute_pair_did_rejects_label_mismatch(tmp_path):
    from when_tta_hurts.cross_condition_addendum import AddendumInputError, FixedPair

    root = tmp_path / "final_test"
    ledger = tmp_path / "ledger_final_test.csv"
    _write_synthetic_cell(root, "run-a", labels=np.array([0, 1, 0, 1]), sample_indices=np.arange(4))
    _write_synthetic_cell(root, "run-b", labels=np.array([1, 1, 0, 1]), sample_indices=np.arange(4))
    authorization = _fake_authorization(["run-a", "run-b"])
    for r in ("run-a", "run-b"):
        _write_ledger_row(ledger, r, authorization.artifact_sha256)

    pair = FixedPair(
        hypothesis="H1",
        pair_id="H1-fake",
        dataset="pathmnist",
        seed=0,
        condition_a_run_id="run-a",
        condition_b_run_id="run-b",
        condition_a_label="a",
        condition_b_label="b",
        directionality="two_sided",
    )
    with pytest.raises(AddendumInputError, match="label mismatch"):
        ftsa.compute_final_test_pair_did(
            pair, authorization, "fake-fp", n=2, final_test_root=root, final_test_ledger_path=ledger
        )


def test_compute_pair_did_rejects_sample_index_mismatch(tmp_path):
    from when_tta_hurts.cross_condition_addendum import AddendumInputError, FixedPair

    root = tmp_path / "final_test"
    ledger = tmp_path / "ledger_final_test.csv"
    labels = np.array([0, 1, 0, 1])
    _write_synthetic_cell(root, "run-a", labels=labels, sample_indices=np.array([0, 1, 2, 3]))
    _write_synthetic_cell(root, "run-b", labels=labels, sample_indices=np.array([3, 2, 1, 0]))
    authorization = _fake_authorization(["run-a", "run-b"])
    for r in ("run-a", "run-b"):
        _write_ledger_row(ledger, r, authorization.artifact_sha256)

    pair = FixedPair(
        hypothesis="H1",
        pair_id="H1-fake",
        dataset="pathmnist",
        seed=0,
        condition_a_run_id="run-a",
        condition_b_run_id="run-b",
        condition_a_label="a",
        condition_b_label="b",
        directionality="two_sided",
    )
    with pytest.raises(AddendumInputError, match="sample-index mismatch"):
        ftsa.compute_final_test_pair_did(
            pair, authorization, "fake-fp", n=2, final_test_root=root, final_test_ledger_path=ledger
        )


# ---------------------------------------------------------------------------
# Real analysis: atomic persistence, idempotent completion, tamper
# rejection, and "failure never produces a completed result".
# ---------------------------------------------------------------------------


def test_real_family_analysis_persists_atomically_and_is_idempotent(tmp_path, monkeypatch):
    root = tmp_path / "final_test"
    ledger = tmp_path / "ledger_final_test.csv"
    analysis_root = tmp_path / "final_test_analysis"
    analysis_ledger = tmp_path / "ledger_final_test_analysis.csv"

    run_id = "run-a"
    _write_synthetic_cell(root, run_id, n_samples=30, n_views=3)
    authorization = _fake_authorization([run_id])
    _write_ledger_row(ledger, run_id, authorization.artifact_sha256)

    fake_cell = SimpleNamespace(run_id=run_id, role="primary")
    monkeypatch.setattr(ftsa, "derive_family_cells", lambda family, matrix_path: [fake_cell])
    monkeypatch.setattr(ftsa, "FINAL_TEST_ANALYSIS_ROOT", analysis_root)
    monkeypatch.setattr(
        ftsa,
        "ensure_final_test_analysis_ledger_exists",
        lambda: real_ensure(analysis_ledger),
    )
    monkeypatch.setattr(
        ftsa,
        "next_final_test_analysis_attempt_number",
        lambda analysis_id: real_next_attempt(analysis_id, analysis_ledger),
    )
    monkeypatch.setattr(
        ftsa,
        "existing_completed_attempt",
        lambda analysis_id: real_existing_completed(analysis_id, analysis_ledger),
    )
    monkeypatch.setattr(
        ftsa,
        "append_final_test_analysis_entry",
        lambda **kw: real_append(ledger_path=analysis_ledger, **kw),
    )

    result = ftsa.compute_final_test_family_analysis(
        "H1", n=2, final_test_root=root, final_test_ledger_path=ledger, _authorization=authorization
    )
    assert result["status"] == "completed"
    assert result["test_split_accessed"] is False

    attempt_dir = analysis_root / "H1" / "attempt_001"
    assert (attempt_dir / "analysis_result.json").exists()
    assert (attempt_dir / "artifact_manifest.json").exists()
    persisted = json.loads((attempt_dir / "analysis_result.json").read_text())
    assert persisted["analysis_id"] == result["analysis_id"]

    # Idempotent: a second call for the same inputs must NOT create a
    # second attempt directory or recompute -- it reads the persisted
    # result back.
    calls = []
    real_load = np.load
    monkeypatch.setattr(ftsa.np, "load", lambda *a, **k: (calls.append(a), real_load(*a, **k))[1])

    result2 = ftsa.compute_final_test_family_analysis(
        "H1", n=2, final_test_root=root, final_test_ledger_path=ledger, _authorization=authorization
    )
    assert result2 == result
    assert calls == [], "idempotent short-circuit must not reload predictions"
    assert not (analysis_root / "H1" / "attempt_002").exists()


def test_failure_never_produces_a_completed_analysis_result(tmp_path, monkeypatch):
    root = tmp_path / "final_test"
    ledger = tmp_path / "ledger_final_test.csv"
    analysis_root = tmp_path / "final_test_analysis"

    good_run_id, bad_run_id = "run-good", "run-missing"
    _write_synthetic_cell(root, good_run_id)
    authorization = _fake_authorization([good_run_id, bad_run_id])
    _write_ledger_row(ledger, good_run_id, authorization.artifact_sha256)
    # bad_run_id has NO ledger row at all -> not eligible.

    monkeypatch.setattr(
        ftsa,
        "derive_family_cells",
        lambda family, matrix_path: [
            SimpleNamespace(run_id=good_run_id, role="primary"),
            SimpleNamespace(run_id=bad_run_id, role="primary"),
        ],
    )
    monkeypatch.setattr(ftsa, "FINAL_TEST_ANALYSIS_ROOT", analysis_root)

    with pytest.raises(ftsa.FinalTestAnalysisInputError):
        ftsa.compute_final_test_family_analysis(
            "H1", n=2, final_test_root=root, final_test_ledger_path=ledger, _authorization=authorization
        )

    assert not analysis_root.exists(), "no attempt directory may exist after a failed family resolution"


def test_tamper_after_persistence_is_rejected_by_manifest_verification(tmp_path, monkeypatch):
    from when_tta_hurts.statistical_analysis_artifacts import (
        AnalysisPersistenceError,
        verify_analysis_artifact_manifest,
    )

    root = tmp_path / "final_test"
    ledger = tmp_path / "ledger_final_test.csv"
    analysis_root = tmp_path / "final_test_analysis"
    analysis_ledger = tmp_path / "ledger_final_test_analysis.csv"

    run_id = "run-a"
    _write_synthetic_cell(root, run_id, n_samples=20)
    authorization = _fake_authorization([run_id])
    _write_ledger_row(ledger, run_id, authorization.artifact_sha256)

    fake_cell = SimpleNamespace(run_id=run_id, role="primary")
    monkeypatch.setattr(ftsa, "derive_family_cells", lambda family, matrix_path: [fake_cell])
    monkeypatch.setattr(ftsa, "FINAL_TEST_ANALYSIS_ROOT", analysis_root)
    monkeypatch.setattr(
        ftsa,
        "ensure_final_test_analysis_ledger_exists",
        lambda: real_ensure(analysis_ledger),
    )
    monkeypatch.setattr(
        ftsa,
        "next_final_test_analysis_attempt_number",
        lambda analysis_id: real_next_attempt(analysis_id, analysis_ledger),
    )
    monkeypatch.setattr(
        ftsa,
        "existing_completed_attempt",
        lambda analysis_id: real_existing_completed(analysis_id, analysis_ledger),
    )
    monkeypatch.setattr(
        ftsa,
        "append_final_test_analysis_entry",
        lambda **kw: real_append(ledger_path=analysis_ledger, **kw),
    )

    ftsa.compute_final_test_family_analysis(
        "H1", n=2, final_test_root=root, final_test_ledger_path=ledger, _authorization=authorization
    )
    attempt_dir = analysis_root / "H1" / "attempt_001"
    manifest = json.loads((attempt_dir / "artifact_manifest.json").read_text())

    # Tamper with the persisted result after the fact.
    (attempt_dir / "analysis_result.json").write_text('{"tampered": true}')

    with pytest.raises(AnalysisPersistenceError):
        verify_analysis_artifact_manifest(attempt_dir, manifest)


# ---------------------------------------------------------------------------
# Sealed CLI output -- the CLI extension prints only status/IDs/hashes,
# never a scientific value, in plan mode.
# ---------------------------------------------------------------------------


def test_plan_report_json_never_contains_forbidden_scientific_keys():
    forbidden_substrings = (
        "accuracy",
        "delta_accuracy",
        "harm_rate",
        "rescue_rate",
        "p_value",
        "ci_low",
        "ci_high",
        "negative_log_likelihood",
    )
    report = ftsa.plan_final_test_statistical_analysis()
    dumped = json.dumps(report).lower()
    for term in forbidden_substrings:
        assert term not in dumped, f"plan-mode report leaked forbidden term: {term}"

    report2 = ftsa.plan_final_test_cross_condition_addendum()
    dumped2 = json.dumps(report2).lower()
    for term in forbidden_substrings:
        assert term not in dumped2, f"plan-mode report leaked forbidden term: {term}"
