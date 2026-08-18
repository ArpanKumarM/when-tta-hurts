"""Suite-wide safety net.

This guards against a repeat of the Phase 2B.4B incident documented in
docs/phase2b_validation_evaluation_incident.md, where a stale test allowed
a real production validation-evaluation run to start. It snapshots the
real production artifact directory and ledger row count before the whole
test session runs, and asserts nothing changed after -- no test in this
suite is permitted to create real evaluation artifacts or append ledger
rows outside of the one, already-recorded incident row.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRODUCTION_EVALUATION_ROOT = _REPO_ROOT / "artifacts" / "validation_evaluation"
_PRODUCTION_EVALUATION_LEDGER = _REPO_ROOT / "artifacts" / "ledger_validation_evaluation.csv"


def _ledger_line_count() -> int:
    if not _PRODUCTION_EVALUATION_LEDGER.exists():
        return 0
    return len(_PRODUCTION_EVALUATION_LEDGER.read_text().splitlines())


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
