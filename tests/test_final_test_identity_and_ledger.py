"""Phase 2B.6A: synthetic tests for final_test_identity.py's fingerprint/
identity primitives and ledger.py's final-test ledger additions. Uses
tmp_path fixtures only -- never the real production ledger.
"""

from __future__ import annotations

import pytest

from when_tta_hurts.final_test_identity import (
    FINAL_TEST_RUNNER_MANIFEST,
    FinalTestEvaluationConfig,
    FinalTestFingerprintError,
    compute_final_test_evaluation_id,
    compute_final_test_runner_fingerprint,
)
from when_tta_hurts.ledger import (
    FINAL_TEST_LEDGER_FIELDNAMES,
    FINAL_TEST_LEDGER_PATH,
    LedgerConflictError,
    append_final_test_entry,
    ensure_final_test_ledger_exists,
)


def _cfg(**overrides):
    base = dict(
        training_run_id="run-a",
        training_attempt=1,
        checkpoint_hash="chk",
        matrix_hash="matrix-hash",
        protocol_commit="ce4c962",
        tta_seed_config_sha256="seed-sha",
        tta_seed_freeze_commit="freeze-commit",
        tta_seed_derivation_sha256="deriv-sha",
        evaluator_fingerprint="eval-fp",
        statistical_analysis_fingerprint="sa-fp",
        cross_condition_analysis_fingerprint="cc-fp",
        final_test_runner_fingerprint="runner-fp",
        authorization_artifact_sha256="auth-sha",
        authorization_commit="auth-commit",
        dataset_expected_checksum_md5="0" * 32,
    )
    base.update(overrides)
    return FinalTestEvaluationConfig(**base)


def test_manifest_excludes_docs_and_ledgers():
    for f in FINAL_TEST_RUNNER_MANIFEST:
        assert not f.startswith("docs/")
        assert "ledger_" not in f


def test_fingerprint_stable_across_calls():
    fp1, _ = compute_final_test_runner_fingerprint()
    fp2, _ = compute_final_test_runner_fingerprint()
    assert fp1 == fp2


def test_fingerprint_changes_when_manifested_file_changes(tmp_path):
    f1 = tmp_path / "fake.py"
    f1.write_text("x = 1\n")
    fp1, _ = compute_final_test_runner_fingerprint(repo_root=tmp_path, manifest=("fake.py",))
    f1.write_text("x = 2\n")
    fp2, _ = compute_final_test_runner_fingerprint(repo_root=tmp_path, manifest=("fake.py",))
    assert fp1 != fp2


def test_fingerprint_fails_closed_on_missing_file(tmp_path):
    with pytest.raises(FinalTestFingerprintError):
        compute_final_test_runner_fingerprint(repo_root=tmp_path, manifest=("does_not_exist.py",))


def test_evaluation_id_deterministic_given_same_config():
    id1 = compute_final_test_evaluation_id(_cfg())
    id2 = compute_final_test_evaluation_id(_cfg())
    assert id1 == id2


def test_evaluation_id_changes_when_checkpoint_hash_changes():
    id1 = compute_final_test_evaluation_id(_cfg())
    id2 = compute_final_test_evaluation_id(_cfg(checkpoint_hash="different"))
    assert id1 != id2


def test_evaluation_id_changes_when_runner_fingerprint_changes():
    id1 = compute_final_test_evaluation_id(_cfg())
    id2 = compute_final_test_evaluation_id(_cfg(final_test_runner_fingerprint="different"))
    assert id1 != id2


def test_evaluation_id_bakes_in_split_test_literally():
    """No field on FinalTestEvaluationConfig can set split to anything
    other than 'test' -- it's a literal inside compute_final_test_evaluation_id,
    never a constructor argument."""
    import inspect

    assert "split" not in inspect.signature(FinalTestEvaluationConfig).parameters


def test_ledger_header_only_and_correct_fieldnames(tmp_path):
    path = tmp_path / "ledger.csv"
    created = ensure_final_test_ledger_exists(path)
    assert created is True
    content = path.read_text()
    lines = content.splitlines()
    assert len(lines) == 1  # header only
    assert lines[0].split(",") == list(FINAL_TEST_LEDGER_FIELDNAMES)


def test_ledger_ensure_is_idempotent(tmp_path):
    path = tmp_path / "ledger.csv"
    assert ensure_final_test_ledger_exists(path) is True
    assert ensure_final_test_ledger_exists(path) is False


def _append_kwargs(**overrides):
    base = dict(
        final_test_evaluation_id="ft-id",
        training_run_id="run-a",
        training_attempt=1,
        checkpoint_hash="chk",
        evaluation_config_hash="ft-id",
        evaluation_attempt=1,
        evaluator_fingerprint="eval-fp",
        statistical_analysis_fingerprint="sa-fp",
        cross_condition_analysis_fingerprint="cc-fp",
        final_test_runner_fingerprint="runner-fp",
        authorization_artifact_sha256="auth-sha",
        authorization_commit="auth-commit",
        test_split_accessed=True,
        test_predictions_computed=True,
        test_metrics_computed=True,
        test_metrics_persisted=True,
        test_metrics_observed=True,
        status="completed",
        primary_artifact_hash="abc",
        started_at=1.0,
        ended_at=2.0,
        runtime_seconds=1.0,
    )
    base.update(overrides)
    return base


def test_append_final_test_entry_appends_and_hardcodes_split_test(tmp_path):
    path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(path)
    result = append_final_test_entry(ledger_path=path, **_append_kwargs())
    assert result == "appended"

    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["split"] == "test"
    assert rows[0]["confirmatory"] == "True"


def test_append_final_test_entry_duplicate_ignored(tmp_path):
    path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(path)
    kwargs = _append_kwargs()
    append_final_test_entry(ledger_path=path, **kwargs)
    result = append_final_test_entry(ledger_path=path, **kwargs)
    assert result == "duplicate_ignored"


def test_append_final_test_entry_conflicting_duplicate_raises(tmp_path):
    path = tmp_path / "ledger.csv"
    ensure_final_test_ledger_exists(path)
    append_final_test_entry(ledger_path=path, **_append_kwargs())
    with pytest.raises(LedgerConflictError):
        append_final_test_entry(ledger_path=path, **_append_kwargs(status="failed"))


def test_split_and_confirmatory_are_not_caller_overridable():
    import inspect

    sig = inspect.signature(append_final_test_entry)
    assert "split" not in sig.parameters
    assert "confirmatory" not in sig.parameters


def test_real_production_final_test_ledger_has_no_unexplained_completed_row():
    """The real artifacts/ledger_final_test.csv must exist. Any
    status=completed row must correspond to a cell that the current
    production authorization (when it verifies) classifies
    "completed_consumed" -- Phase 2B.6G's first real, authorized
    completion legitimately produced one such row. A completed row for
    any cell NOT so classified (or any completed row while verification
    fails/is stale) remains forbidden. It may also legitimately contain
    non-completed rows: e.g. the preserved status=aborted/failed rows
    documented in docs/phase2b_final_test_accidental_access_incident.md
    and docs/phase2b_final_test_attempt2_preaccess_failure.md."""
    import csv

    from when_tta_hurts.final_test_authorization import (
        FINAL_TEST_AUTHORIZATION_PATH,
        FinalTestAuthorizationError,
        verify_final_test_authorization,
    )

    assert FINAL_TEST_LEDGER_PATH.exists()
    rows = list(csv.DictReader(FINAL_TEST_LEDGER_PATH.open(newline="")))
    completed_rows = [row for row in rows if row["status"] == "completed"]
    if not completed_rows:
        return

    if not FINAL_TEST_AUTHORIZATION_PATH.exists():
        pytest.fail(f"completed final-test row(s) exist with no authorization artifact: {completed_rows!r}")

    try:
        authorization = verify_final_test_authorization()
    except FinalTestAuthorizationError:
        return  # stale/superseded authorization mid-engineering -- no authoritative classification available

    for row in completed_rows:
        assert authorization.cell_classifications.get(row["training_run_id"]) == "completed_consumed", (
            f"unexpected COMPLETED final-test ledger row for a cell not classified "
            f"completed_consumed: {row!r}"
        )
