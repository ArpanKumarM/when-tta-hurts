"""Tests for scripts/benchmark_runtime_real.py's proxy-labeling guarantees
(Phase 2A audit round 2)."""

from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_runtime_real.py"


def test_128px_condition_labeled_resized_proxy_not_official():
    source = SCRIPT_PATH.read_text()
    assert '"resized_proxy_not_official"' in source


def test_native_conditions_labeled_native_official():
    source = SCRIPT_PATH.read_text()
    assert '"native_official_real_data"' in source


def test_64px_resized_condition_removed():
    """The 64px resized-proxy condition was removed in round 2 -- native
    64px now lives only in scripts/benchmark_runtime.py."""
    source = SCRIPT_PATH.read_text()
    assert "small_cnn_64px_resized_proxy" not in source


def test_128px_warning_does_not_claim_official_equivalence():
    source = SCRIPT_PATH.read_text()
    warning_start = source.index('r["WARNING"]')
    warning_end = source.index(")", source.index("accuracy claim", warning_start))
    warning_text = source[warning_start:warning_end]
    assert "NOT official 128px" in warning_text
    assert "NOT a native-resolution measurement" in warning_text
    assert "NOT equivalent to measuring an official" in warning_text


def test_no_test_split_requested():
    source = SCRIPT_PATH.read_text()
    assert 'split="test"' not in source
    assert "split='test'" not in source
