"""Suite-wide safety net.

This guards against a repeat of the Phase 2B.4B incident documented in
docs/phase2b_validation_evaluation_incident.md, where a stale test allowed
a real production validation-evaluation run to start. It snapshots the
real production artifact directory and ledger row count before the whole
test session runs, and asserts nothing changed after -- no test in this
suite is permitted to create real evaluation artifacts or append ledger
rows outside of the one, already-recorded incident row.

It also guards against a repeat of the Phase 2B.6C-Incident documented in
docs/phase2b_final_test_accidental_access_incident.md, where a stale test
plus a wrong-namespace monkeypatch let a real final-test evaluation start
for real inside a pytest process. Two independent mechanisms address this:
(1) a per-test autouse fixture that forces
when_tta_hurts.final_test_evaluation.select_device to a hard-failing
sentinel before every single test, so ANY test that reaches the real
final-test runner's default device-selection path fails loudly and
immediately -- a test that legitimately needs a device must explicitly
re-patch it (or supply its own device_resolver), which simply overrides
this guard, harmlessly; (2) a session-wide snapshot/compare of the real
final-test artifact directory, ledger, and authorization artifact,
mirroring the validation-evaluation guard above.

Both guards share the SAME structural limitation: they can only detect
damage from a test session that reaches its own normal teardown. An
externally terminated process (e.g. `kill -9`, as happened in the
Phase 2B.6C-Incident) bypasses all pytest/fixture teardown code, so
these guards -- like the pre-existing validation-evaluation guard above
-- cannot detect or prevent that specific failure mode. They can only
ever catch a completed session that produced real side effects.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_EVALUATION_ROOT = _REPO_ROOT / "artifacts" / "validation_evaluation"
_PRODUCTION_EVALUATION_LEDGER = _REPO_ROOT / "artifacts" / "ledger_validation_evaluation.csv"
_PRODUCTION_FINAL_TEST_ROOT = _REPO_ROOT / "artifacts" / "final_test"
_PRODUCTION_FINAL_TEST_LEDGER = _REPO_ROOT / "artifacts" / "ledger_final_test.csv"
_PRODUCTION_FINAL_TEST_AUTHORIZATION = _REPO_ROOT / "artifacts" / "final_test_authorization.json"


def _ledger_line_count() -> int:
    if not _PRODUCTION_EVALUATION_LEDGER.exists():
        return 0
    return len(_PRODUCTION_EVALUATION_LEDGER.read_text().splitlines())


def _file_hash_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _no_real_validation_evaluation_side_effects():
    root_existed_before = _PRODUCTION_EVALUATION_ROOT.exists()
    ledger_lines_before = _ledger_line_count()

    yield

    assert _PRODUCTION_EVALUATION_ROOT.exists() == root_existed_before, (
        "A test run created or removed the real production validation-evaluation "
        f"artifact directory ({_PRODUCTION_EVALUATION_ROOT}). No test may touch real "
        "evaluation artifacts -- use tmp_path-based roots and patched runners."
    )
    assert _ledger_line_count() == ledger_lines_before, (
        "A test run appended to the real production validation-evaluation ledger "
        f"({_PRODUCTION_EVALUATION_LEDGER}). No test may write to the real ledger -- "
        "use a tmp_path-based ledger_path."
    )


@pytest.fixture(scope="session", autouse=True)
def _no_real_final_test_side_effects():
    root_existed_before = _PRODUCTION_FINAL_TEST_ROOT.exists()
    ledger_hash_before = _file_hash_or_none(_PRODUCTION_FINAL_TEST_LEDGER)
    authorization_hash_before = _file_hash_or_none(_PRODUCTION_FINAL_TEST_AUTHORIZATION)

    yield

    assert _PRODUCTION_FINAL_TEST_ROOT.exists() == root_existed_before, (
        "A test run created or removed the real production final-test artifact "
        f"directory ({_PRODUCTION_FINAL_TEST_ROOT}). Per "
        "docs/phase2b_final_test_accidental_access_incident.md, no test may touch "
        "real final-test artifacts -- use tmp_path-based roots and patched runners."
    )
    assert _file_hash_or_none(_PRODUCTION_FINAL_TEST_LEDGER) == ledger_hash_before, (
        "A test run modified the real production final-test ledger "
        f"({_PRODUCTION_FINAL_TEST_LEDGER}). No test may write to the real ledger -- "
        "use a tmp_path-based final_test_ledger_path."
    )
    assert _file_hash_or_none(_PRODUCTION_FINAL_TEST_AUTHORIZATION) == authorization_hash_before, (
        "A test run modified the real, committed final-test authorization artifact "
        f"({_PRODUCTION_FINAL_TEST_AUTHORIZATION}). This must never happen."
    )


@pytest.fixture(autouse=True)
def _guard_against_real_final_test_device_selection(monkeypatch):
    """Forces when_tta_hurts.final_test_evaluation.select_device to a
    hard-failing sentinel before EVERY test in the suite. A test that
    legitimately needs `run_final_test_evaluation()` to reach device
    selection must explicitly monkeypatch `fte.select_device` (or supply
    its own `device_resolver=...`) in its own body -- monkeypatch's
    last-write-wins semantics mean that simply overrides this guard for
    that one test, harmlessly. Any test that does NOT do so, and reaches
    this code path anyway, fails immediately and loudly rather than
    silently touching real MPS -- this is the exact choke point the
    Phase 2B.6C-Incident's wrong-namespace bug failed to close."""
    import when_tta_hurts.final_test_evaluation as fte_module

    def _raise_real_device_selection_forbidden(name):
        raise RuntimeError(
            "A test reached the REAL select_device() default in "
            "final_test_evaluation.py. This must never happen -- see "
            "docs/phase2b_final_test_accidental_access_incident.md. If this test "
            "legitimately needs a device, monkeypatch fte.select_device explicitly "
            "or pass device_resolver=... to run_final_test_evaluation()."
        )

    monkeypatch.setattr(fte_module, "select_device", _raise_real_device_selection_forbidden)
    yield
