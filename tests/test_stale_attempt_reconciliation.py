"""Phase 2B.3B Part 2 tests: the runner must never silently start a later
attempt while an earlier attempt is nonterminal and unledgered
(StaleAttemptError), and reconciliation must be explicit, evidence-based,
idempotent, and never touch the attempt directory or mark an interrupted
attempt canonical-eligible.

Synthetic temporary artifacts ONLY -- never touches the real
attempt_001-004 of A-pathmnist-28px-batchnorm-policy-none-s0, the real
attempt_001 of the two reconciled Block A cells, or any real dataset."""

from __future__ import annotations

import pytest
from torch import nn

from when_tta_hurts.artifacts import save_checkpoint
from when_tta_hurts.ledger import LedgerConflictError
from when_tta_hurts.matrix import MatrixCell
from when_tta_hurts.orchestrator import (
    StaleAttemptError,
    check_confirmatory_skip,
    reconcile_stale_attempt,
)
from when_tta_hurts.run_identity import (
    RunStatus,
    finish_attempt,
    next_attempt_number,
    start_attempt,
)

CELL = MatrixCell(
    block="A_core_normalization_resolution",
    dataset="synthetic",
    resolution=4,
    model="small_cnn",
    normalization="batchnorm",
    training_policy="none",
    seed=0,
)


def _start_running_attempt(root, cell=CELL):
    """Start an attempt and leave it in 'running' -- simulates an
    external interruption before finish_attempt() ever runs."""
    attempt_dir, status = start_attempt(cell, root)
    return attempt_dir, status


def _tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 5))


def _complete_attempt_with_partial_artifacts(root, cell=CELL):
    """A 'running' attempt that has SOME artifacts (e.g. a checkpoint was
    saved) but was interrupted before finish_attempt()."""
    attempt_dir, status = start_attempt(cell, root)
    model = _tiny_model()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    save_checkpoint(state_dict, attempt_dir / "best_checkpoint.pt")
    # Deliberately no training_history.json/result.json/metadata.json/
    # artifact_manifest.json/finish_attempt() -- interrupted mid-persistence.
    return attempt_dir, status


# --- StaleAttemptError before any device/data/model activity ---


def test_interruption_before_any_artifacts_blocks_new_attempt(tmp_path):
    root = tmp_path / "confirmatory"
    _start_running_attempt(root)

    with pytest.raises(StaleAttemptError):
        check_confirmatory_skip(
            CELL, root, tmp_path / "ledger_amendments.csv", tmp_path / "ledger_confirmatory.csv"
        )


def test_interruption_with_partial_artifacts_blocks_new_attempt(tmp_path):
    root = tmp_path / "confirmatory"
    _complete_attempt_with_partial_artifacts(root)

    with pytest.raises(StaleAttemptError):
        check_confirmatory_skip(
            CELL, root, tmp_path / "ledger_amendments.csv", tmp_path / "ledger_confirmatory.csv"
        )


def test_stale_error_raised_before_next_attempt_number_would_be_used(tmp_path):
    """Confirms the guard is upstream of anything resembling starting a
    new attempt -- next_attempt_number itself is unaffected (it's a pure
    directory-counting function), but check_confirmatory_skip must refuse
    before any caller would act on that number."""
    root = tmp_path / "confirmatory"
    _start_running_attempt(root)
    # next_attempt_number itself still works (it's not the guard) --
    # the guard lives in check_confirmatory_skip, which every real
    # execution path calls first.
    assert next_attempt_number(CELL, root) == 2
    with pytest.raises(StaleAttemptError):
        check_confirmatory_skip(
            CELL, root, tmp_path / "ledger_amendments.csv", tmp_path / "ledger_confirmatory.csv"
        )


# --- reconciliation: idempotency, conflicts, liveness, evidence ---


def test_reconciliation_refuses_without_confirmed_not_running(tmp_path):
    root = tmp_path / "confirmatory"
    _start_running_attempt(root)
    with pytest.raises(StaleAttemptError):
        reconcile_stale_attempt(
            CELL,
            attempt_number=1,
            reason="evidence: process no longer in ps output",
            confirmed_not_running=False,
            recorded_at="2026-01-01T00:00:00Z",
            root=root,
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )


def test_reconciliation_requires_nonempty_reason(tmp_path):
    root = tmp_path / "confirmatory"
    _start_running_attempt(root)
    with pytest.raises(ValueError):
        reconcile_stale_attempt(
            CELL,
            attempt_number=1,
            reason="   ",
            confirmed_not_running=True,
            recorded_at="2026-01-01T00:00:00Z",
            root=root,
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )


def test_successful_reconciliation_unblocks_next_attempt(tmp_path):
    root = tmp_path / "confirmatory"
    confirmatory_ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _start_running_attempt(root)

    result = reconcile_stale_attempt(
        CELL,
        attempt_number=1,
        reason="evidence: process no longer in ps output, host was rebooted",
        confirmed_not_running=True,
        recorded_at="2026-01-01T00:00:00Z",
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
    )
    assert result == "appended"

    # Now the skip check must NOT raise -- and since no completed
    # attempt exists, it must return None (proceed to a new attempt).
    skip = check_confirmatory_skip(CELL, root, amendments_path, confirmatory_ledger_path)
    assert skip is None
    assert next_attempt_number(CELL, root) == 2


def test_duplicate_reconciliation_is_idempotent(tmp_path):
    root = tmp_path / "confirmatory"
    confirmatory_ledger_path = tmp_path / "ledger_confirmatory.csv"
    _start_running_attempt(root)

    kwargs = dict(
        cell=CELL,
        attempt_number=1,
        reason="evidence: process no longer in ps output",
        confirmed_not_running=True,
        recorded_at="2026-01-01T00:00:00Z",
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
    )
    r1 = reconcile_stale_attempt(**kwargs)
    r2 = reconcile_stale_attempt(**kwargs)
    assert r1 == "appended"
    assert r2 == "duplicate_ignored"

    import csv

    with confirmatory_ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1  # not duplicated


def test_conflicting_reconciliation_hard_fails(tmp_path):
    root = tmp_path / "confirmatory"
    confirmatory_ledger_path = tmp_path / "ledger_confirmatory.csv"
    _start_running_attempt(root)

    reconcile_stale_attempt(
        CELL,
        attempt_number=1,
        reason="evidence: process no longer in ps output",
        confirmed_not_running=True,
        recorded_at="2026-01-01T00:00:00Z",
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
    )
    with pytest.raises(LedgerConflictError):
        reconcile_stale_attempt(
            CELL,
            attempt_number=1,
            reason="a DIFFERENT reason -- conflicting with the first reconciliation",
            confirmed_not_running=True,
            recorded_at="2026-01-01T00:00:00Z",
            root=root,
            confirmatory_ledger_path=confirmatory_ledger_path,
        )


def test_reconciliation_of_a_live_attempt_without_confirmation_is_refused(tmp_path):
    """A 'live' attempt is simulated as still status='running' -- without
    confirmed_not_running=True, reconciliation must refuse regardless of
    whether it's actually live or not (the function cannot know)."""
    root = tmp_path / "confirmatory"
    _start_running_attempt(root)
    with pytest.raises(StaleAttemptError):
        reconcile_stale_attempt(
            CELL,
            attempt_number=1,
            reason="attempting to reconcile a possibly-live attempt",
            confirmed_not_running=False,
            recorded_at="2026-01-01T00:00:00Z",
            root=root,
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )


def test_reconciliation_never_edits_attempt_directory(tmp_path):
    root = tmp_path / "confirmatory"
    attempt_dir, _ = _complete_attempt_with_partial_artifacts(root)
    before_checkpoint = (attempt_dir / "best_checkpoint.pt").read_bytes()
    before_status = (attempt_dir / "status.json").read_text()
    before_files = sorted(p.name for p in attempt_dir.iterdir())

    reconcile_stale_attempt(
        CELL,
        attempt_number=1,
        reason="evidence: process terminated externally, checkpoint partially written",
        confirmed_not_running=True,
        recorded_at="2026-01-01T00:00:00Z",
        root=root,
        confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
    )

    after_checkpoint = (attempt_dir / "best_checkpoint.pt").read_bytes()
    after_status = (attempt_dir / "status.json").read_text()
    after_files = sorted(p.name for p in attempt_dir.iterdir())
    assert before_checkpoint == after_checkpoint
    assert before_status == after_status
    assert before_files == after_files  # no file added/removed


def test_reconciliation_never_marks_canonical_eligible(tmp_path):
    root = tmp_path / "confirmatory"
    confirmatory_ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _start_running_attempt(root)

    reconcile_stale_attempt(
        CELL,
        attempt_number=1,
        reason="evidence: process no longer in ps output",
        confirmed_not_running=True,
        recorded_at="2026-01-01T00:00:00Z",
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
    )
    # Reconciled attempt is aborted, not completed -- never a skip candidate.
    skip = check_confirmatory_skip(CELL, root, amendments_path, confirmatory_ledger_path)
    assert skip is None

    import csv

    with confirmatory_ledger_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["status"] == "aborted"
    assert rows[0]["checkpoint_hash"] == ""


def test_reconciliation_refuses_already_terminal_attempt(tmp_path):
    root = tmp_path / "confirmatory"
    attempt_dir, status = start_attempt(CELL, root)
    finish_attempt(attempt_dir, status, RunStatus.FAILED, failure_reason="normal failure")

    with pytest.raises(StaleAttemptError):
        reconcile_stale_attempt(
            CELL,
            attempt_number=1,
            reason="attempt is already terminal",
            confirmed_not_running=True,
            recorded_at="2026-01-01T00:00:00Z",
            root=root,
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )


def test_reconciliation_refuses_nonexistent_attempt(tmp_path):
    root = tmp_path / "confirmatory"
    with pytest.raises(StaleAttemptError):
        reconcile_stale_attempt(
            CELL,
            attempt_number=1,
            reason="attempt does not exist",
            confirmed_not_running=True,
            recorded_at="2026-01-01T00:00:00Z",
            root=root,
            confirmatory_ledger_path=tmp_path / "ledger_confirmatory.csv",
        )


# --- next-attempt resolution only after reconciliation ---


def test_next_attempt_resolution_only_after_reconciliation(tmp_path):
    root = tmp_path / "confirmatory"
    confirmatory_ledger_path = tmp_path / "ledger_confirmatory.csv"
    amendments_path = tmp_path / "ledger_amendments.csv"
    _start_running_attempt(root)

    # Before reconciliation: blocked.
    with pytest.raises(StaleAttemptError):
        check_confirmatory_skip(CELL, root, amendments_path, confirmatory_ledger_path)

    reconcile_stale_attempt(
        CELL,
        attempt_number=1,
        reason="evidence: process no longer in ps output",
        confirmed_not_running=True,
        recorded_at="2026-01-01T00:00:00Z",
        root=root,
        confirmatory_ledger_path=confirmatory_ledger_path,
    )

    # After reconciliation: unblocked, no skip (nothing eligible yet).
    skip = check_confirmatory_skip(CELL, root, amendments_path, confirmatory_ledger_path)
    assert skip is None
    assert next_attempt_number(CELL, root) == 2


def test_real_block_a_attempts_not_touched_by_this_test_file():
    """Sanity: no OTHER test function body in this file references the
    real artifacts/confirmatory tree as a literal path -- every test above
    uses tmp_path exclusively."""
    import inspect

    marker = "artifacts" + "/confirmatory"  # split to avoid self-matching this line
    for name, obj in list(globals().items()):
        if name.startswith("test_") and name != "test_real_block_a_attempts_not_touched_by_this_test_file":
            if inspect.isfunction(obj):
                assert marker not in inspect.getsource(obj)
