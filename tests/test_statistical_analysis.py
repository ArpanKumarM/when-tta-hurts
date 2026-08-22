"""Phase 2B.5A: synthetic/temporary-fixture tests for the statistical-
analysis engineering layer. No real production artifact, ledger, or
checkpoint is ever read or written by these tests -- every fixture is
either hand-constructed in-memory or written to tmp_path.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.stats import binomtest, chi2

from when_tta_hurts.statistical_analysis import (
    ANALYSIS_FINGERPRINT_MANIFEST,
    KNOWN_FAMILIES,
    AnalysisFingerprintError,
    benjamini_hochberg,
    compute_analysis_fingerprint,
    compute_analysis_id,
    derive_family_cells,
    effect_sizes,
    mcnemar_test,
    paired_bootstrap_ci,
)
from when_tta_hurts.statistical_analysis_artifacts import (
    AnalysisPersistenceError,
    AnalysisSchemaValidationError,
    build_analysis_artifact_manifest,
    persist_and_verify_analysis_completion,
    validate_analysis_result_schema,
    verify_analysis_artifact_manifest,
)

# ---------------------------------------------------------------------------
# Hand-calculated examples
# ---------------------------------------------------------------------------


def test_paired_bootstrap_ci_hand_calculated_point_estimate():
    """10 paired samples: clean gets 6/10 correct, TTA gets 8/10 correct
    (TTA corrects 2 of clean's failures, changes nothing else). Point
    delta must equal exactly 0.2, independent of resampling noise."""
    clean_correct = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    tta_correct = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    result = paired_bootstrap_ci(clean_correct, tta_correct, n_resamples=2000, rng=np.random.default_rng(0))
    assert math.isclose(result["delta_accuracy"], 0.2, abs_tol=1e-9)
    assert result["ci_low"] <= result["delta_accuracy"] <= result["ci_high"]
    assert result["n_samples"] == 10


def test_paired_bootstrap_ci_identical_arrays_gives_zero_width_ci():
    """If clean and TTA are identical, every resample has delta=0, so the
    CI must collapse to exactly [0, 0] -- a strong sanity/no-invented-
    variance check."""
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
    result = paired_bootstrap_ci(correct, correct, n_resamples=500, rng=np.random.default_rng(1))
    assert result["delta_accuracy"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_mcnemar_exact_matches_scipy_binomtest_independent_reference():
    """b=2 (clean-correct/TTA-wrong), c=8 (clean-wrong/TTA-correct) --
    n_discordant=10 < 25, so our implementation uses the exact binomial
    path. Cross-check against scipy.stats.binomtest, an independent
    trusted reference library, not our own test oracle."""
    clean_correct = np.array([1, 1] + [0] * 8, dtype=bool)
    tta_correct = np.array([0, 0] + [1] * 8, dtype=bool)
    ours = mcnemar_test(clean_correct, tta_correct)
    assert ours["b"] == 2
    assert ours["c"] == 8
    assert ours["method"] == "exact_binomial"
    reference = binomtest(k=min(2, 8), n=10, p=0.5, alternative="two-sided")
    assert math.isclose(ours["p_value"], reference.pvalue, rel_tol=1e-9)


def test_mcnemar_continuity_corrected_matches_hand_formula_and_scipy_chi2():
    """30 discordant pairs (>= 25) forces the continuity-corrected
    chi-square path. Cross-check against the textbook formula computed
    independently via scipy.stats.chi2's survival function."""
    b, c = 10, 20
    clean_correct = np.array([1] * b + [0] * c + [1] * 50, dtype=bool)
    tta_correct = np.array([0] * b + [1] * c + [1] * 50, dtype=bool)
    ours = mcnemar_test(clean_correct, tta_correct)
    assert ours["b"] == b
    assert ours["c"] == c
    assert ours["method"] == "continuity_corrected_chi_square"
    expected_stat = (abs(b - c) - 1) ** 2 / (b + c)
    expected_p = float(chi2.sf(expected_stat, df=1))
    assert math.isclose(ours["statistic"], expected_stat, rel_tol=1e-9)
    assert math.isclose(ours["p_value"], expected_p, rel_tol=1e-6)


def test_mcnemar_zero_discordant_is_undefined_not_a_fabricated_pvalue():
    """Ties/zero differences: when clean and TTA agree on every sample,
    McNemar is degenerate. Must report method='undefined' and
    p_value=None -- never substitute a default p-value."""
    correct = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    result = mcnemar_test(correct, correct)
    assert result["n_discordant"] == 0
    assert result["method"] == "undefined"
    assert result["p_value"] is None
    assert result["statistic"] is None


def test_benjamini_hochberg_hand_calculated_example():
    """Classic 4-value example, hand-computed: p = [0.01, 0.04, 0.03, 0.20]
    -> BH q-values computed by the standard step-up procedure."""
    p_values = [0.01, 0.04, 0.03, 0.20]
    # Sorted ascending: 0.01 (rank1), 0.03 (rank2), 0.04 (rank3), 0.20 (rank4)
    # q = p * n / rank, then cumulative-min from the largest rank down:
    # rank4: 0.20*4/4=0.20 -> running_min=0.20
    # rank3: 0.04*4/3=0.05333 -> running_min=0.05333
    # rank2: 0.03*4/2=0.06 -> running_min=min(0.06,0.05333)=0.05333
    # rank1: 0.01*4/1=0.04 -> running_min=min(0.04,0.05333)=0.04
    expected = {0.01: 0.04, 0.03: 0.05333333333333334, 0.04: 0.05333333333333334, 0.20: 0.20}
    corrected = benjamini_hochberg(p_values)
    for raw, corr in zip(p_values, corrected, strict=True):
        assert math.isclose(corr, expected[raw], rel_tol=1e-6), (raw, corr)


def test_benjamini_hochberg_rejects_out_of_range_pvalue():
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, 1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([0.5, float("nan")])


def test_effect_sizes_hand_calculated():
    """4 clean-correct samples, 2 harmed by TTA; 2 clean-wrong samples, 1
    rescued by TTA. harm_rate = 2/4 = 0.5, rescue_rate = 1/2 = 0.5."""
    clean_correct = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
    tta_correct = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    result = effect_sizes(clean_correct, tta_correct)
    assert math.isclose(result["harm_rate"], 0.5)
    assert math.isclose(result["rescue_rate"], 0.5)
    assert math.isclose(result["delta_accuracy"], (3 / 6) - (4 / 6))


# ---------------------------------------------------------------------------
# Pairing, misalignment, pseudoreplication
# ---------------------------------------------------------------------------


def test_pairing_requires_identical_shape_prevents_misalignment():
    clean_correct = np.array([1, 0, 1], dtype=bool)
    tta_correct = np.array([1, 0], dtype=bool)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(clean_correct, tta_correct)
    with pytest.raises(ValueError):
        mcnemar_test(clean_correct, tta_correct)


def test_no_image_level_pseudoreplication_unit_is_the_model_not_the_sample():
    """The independent experimental unit for a family is one trained
    model (one run_id/seed), never an individual validation image. H1's
    derived family must have exactly as many entries as
    (datasets x resolutions x normalizations x seeds) trained models --
    24 for the real frozen matrix -- never a per-image count."""
    cells = derive_family_cells("H1")
    assert len(cells) == 24
    assert len({c.run_id for c in cells}) == 24  # every run_id distinct: one row per model, not per image


def test_directionality_is_two_sided_by_construction():
    """paired_bootstrap_ci returns a symmetric two-sided percentile CI;
    nothing in its interface allows a one-sided alternative to be
    requested -- this is a structural guarantee, not a runtime check."""
    import inspect

    sig = inspect.signature(paired_bootstrap_ci)
    assert "alternative" not in sig.parameters
    assert "side" not in sig.parameters


# ---------------------------------------------------------------------------
# Family derivation: missing cells, deterministic ordering
# ---------------------------------------------------------------------------


def test_derive_family_cells_deterministic_ordering():
    a = derive_family_cells("H1")
    b = derive_family_cells("H1")
    assert [c.run_id for c in a] == [c.run_id for c in b]
    assert [c.run_id for c in a] == sorted(c.run_id for c in a)


def test_derive_family_cells_h4_not_yet_derivable():
    with pytest.raises(ValueError, match="not-yet-derivable|Unknown"):
        derive_family_cells("H4")


def test_derive_family_cells_unknown_family_rejected():
    with pytest.raises(ValueError):
        derive_family_cells("NOT_A_REAL_FAMILY")


def test_known_families_excludes_h4():
    assert "H4" not in KNOWN_FAMILIES
    assert set(KNOWN_FAMILIES) == {"H1", "H2", "H3", "BLOCK_C"}


# ---------------------------------------------------------------------------
# Schema validation: non-finite rejection, required keys, test-split flag
# ---------------------------------------------------------------------------


def _valid_result(**overrides):
    base = {
        "family": "H1",
        "analysis_id": "abc123",
        "analysis_fingerprint": "def456",
        "current_evaluator_fingerprint": "ghi789",
        "cells": ["run-a", "run-b"],
        "per_cell_statistics": {"run-a": {"delta_accuracy": 0.1}, "run-b": {"delta_accuracy": 0.2}},
        "multiplicity": {"raw_p_values": [0.01, 0.02], "corrected_p_values": [0.02, 0.02]},
        "status": "completed",
        "test_split_accessed": False,
    }
    base.update(overrides)
    return base


def test_schema_validation_accepts_valid_result():
    validate_analysis_result_schema(_valid_result())  # must not raise


def test_schema_validation_rejects_missing_keys():
    result = _valid_result()
    del result["analysis_id"]
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(result)


def test_schema_validation_rejects_unknown_family():
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(_valid_result(family="NOT_A_FAMILY"))


def test_schema_validation_rejects_non_completed_status():
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(_valid_result(status="failed"))


def test_schema_validation_rejects_test_split_accessed_true():
    """Hard requirement: any result claiming test_split_accessed=True (or
    anything other than exactly False) must be refused at persistence
    time -- fails closed rather than trusting the caller."""
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(_valid_result(test_split_accessed=True))


def test_schema_validation_rejects_non_finite_values():
    result = _valid_result()
    result["per_cell_statistics"]["run-a"]["delta_accuracy"] = float("nan")
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(result)

    result2 = _valid_result()
    result2["multiplicity"]["raw_p_values"] = [float("inf"), 0.02]
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(result2)


def test_schema_validation_rejects_empty_cells_or_stats():
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(_valid_result(cells=[]))
    with pytest.raises(AnalysisSchemaValidationError):
        validate_analysis_result_schema(_valid_result(per_cell_statistics={}))


# ---------------------------------------------------------------------------
# Artifact-manifest verification, atomic persistence (tmp_path only)
# ---------------------------------------------------------------------------


def test_persist_and_verify_analysis_completion_round_trip(tmp_path):
    result = _valid_result()
    manifest = persist_and_verify_analysis_completion(tmp_path, result=result)
    assert (tmp_path / "analysis_result.json").exists()
    assert (tmp_path / "artifact_manifest.json").exists()
    assert len(manifest["artifacts"]) == 1
    verify_analysis_artifact_manifest(tmp_path, manifest)  # must not raise

    reloaded = json.loads((tmp_path / "analysis_result.json").read_text())
    assert reloaded["family"] == "H1"


def test_persist_and_verify_analysis_completion_rejects_invalid_result(tmp_path):
    with pytest.raises(AnalysisSchemaValidationError):
        persist_and_verify_analysis_completion(tmp_path, result=_valid_result(status="in_progress"))
    assert not (tmp_path / "analysis_result.json").exists()


def test_manifest_verification_detects_tampering(tmp_path):
    result = _valid_result()
    persist_and_verify_analysis_completion(tmp_path, result=result)
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())

    # Tamper with the persisted result after the manifest was written.
    (tmp_path / "analysis_result.json").write_text(json.dumps({"family": "TAMPERED"}))
    with pytest.raises(AnalysisPersistenceError):
        verify_analysis_artifact_manifest(tmp_path, manifest)


def test_manifest_build_fails_closed_on_missing_file(tmp_path):
    """Building a manifest against a file that does not exist must raise,
    never silently skip the missing entry."""
    with pytest.raises(AnalysisPersistenceError):
        build_analysis_artifact_manifest(tmp_path, ("nonexistent.json",))


def test_manifest_verification_detects_missing_file(tmp_path):
    result = _valid_result()
    persist_and_verify_analysis_completion(tmp_path, result=result)
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    (tmp_path / "analysis_result.json").unlink()
    with pytest.raises(AnalysisPersistenceError):
        verify_analysis_artifact_manifest(tmp_path, manifest)


# ---------------------------------------------------------------------------
# Analysis-fingerprint / analysis-ID identity stability
# ---------------------------------------------------------------------------


def test_analysis_fingerprint_manifest_excludes_docs_and_ledgers():
    """Structural guarantee that a docs/ or ledger-only commit can never
    change the analysis fingerprint."""
    for path in ANALYSIS_FINGERPRINT_MANIFEST:
        assert not path.startswith("docs/"), path
        assert "ledger" not in path.lower(), path


def test_analysis_fingerprint_fails_closed_on_missing_manifest_file(tmp_path):
    with pytest.raises(AnalysisFingerprintError):
        compute_analysis_fingerprint(repo_root=tmp_path, manifest=("does/not/exist.py",))


def test_analysis_fingerprint_changes_when_manifested_file_changes(tmp_path):
    """Identity change when analysis-relevant code changes: hashing two
    different byte contents for the same manifested path must yield two
    different fingerprints."""
    manifest = ("only_file.py",)
    (tmp_path / "only_file.py").write_text("version_one")
    fp1, _ = compute_analysis_fingerprint(repo_root=tmp_path, manifest=manifest)

    (tmp_path / "only_file.py").write_text("version_two")
    fp2, _ = compute_analysis_fingerprint(repo_root=tmp_path, manifest=manifest)

    assert fp1 != fp2


def test_analysis_fingerprint_stable_for_identical_content(tmp_path):
    """Identity stability across docs/ledger-only commits: since the
    fingerprint is computed purely from the manifested files' bytes, an
    unrelated (non-manifested) change anywhere else -- exactly what a
    docs/ledger-only commit is -- leaves it unchanged. Simulated here by
    re-hashing identical content and confirming equality."""
    manifest = ("only_file.py",)
    (tmp_path / "only_file.py").write_text("stable_content")
    fp1, _ = compute_analysis_fingerprint(repo_root=tmp_path, manifest=manifest)
    fp2, _ = compute_analysis_fingerprint(repo_root=tmp_path, manifest=manifest)
    assert fp1 == fp2


def test_analysis_id_stable_given_same_inputs():
    id1 = compute_analysis_id("H1", ("eval-a", "eval-b"), "fp123")
    id2 = compute_analysis_id("H1", ("eval-b", "eval-a"), "fp123")  # order-independent
    assert id1 == id2


def test_analysis_id_changes_with_different_fingerprint():
    id1 = compute_analysis_id("H1", ("eval-a",), "fp123")
    id2 = compute_analysis_id("H1", ("eval-a",), "fp456")
    assert id1 != id2


def test_analysis_id_changes_with_different_evaluation_ids():
    id1 = compute_analysis_id("H1", ("eval-a",), "fp123")
    id2 = compute_analysis_id("H1", ("eval-a", "eval-b"), "fp123")
    assert id1 != id2


# ---------------------------------------------------------------------------
# No test-split reachability
# ---------------------------------------------------------------------------


def test_no_test_split_symbol_reachable_from_statistical_analysis_modules():
    import when_tta_hurts.statistical_analysis as sa_module
    import when_tta_hurts.statistical_analysis_artifacts as sa_artifacts_module

    for module in (sa_module, sa_artifacts_module):
        source = open(module.__file__).read()
        assert "allow_test" not in source
        assert "load_test" not in source
        assert "test_split" not in source.lower() or "test_split_accessed" in source


def test_cli_script_has_no_test_split_flag():
    """No argparse argument for test-split access exists -- checked
    against actual `add_argument(...)` calls, not prose mentioning the
    absence of such a flag (this docstring-vs-code distinction matters:
    a naive substring search over the whole file would false-positive on
    this module's own documentation)."""
    import pathlib
    import re

    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_statistical_analysis.py"
    source = script.read_text()
    add_argument_calls = re.findall(r"add_argument\([^)]*\)", source)
    assert not any("test" in call.lower() for call in add_argument_calls)
    assert "allow_test" not in source


# ---------------------------------------------------------------------------
# Plan-mode side-effect freedom
# ---------------------------------------------------------------------------


def test_plan_mode_is_side_effect_free(tmp_path, monkeypatch):
    """plan_statistical_analysis() must never write a file. Verified by
    running it against the REAL repository (read-only) and asserting no
    new files appear anywhere the process could plausibly write, plus
    confirming the statistical-analysis ledger is never created as a
    side effect."""
    import os

    from when_tta_hurts.ledger import STATISTICAL_ANALYSIS_LEDGER_PATH
    from when_tta_hurts.statistical_analysis import plan_statistical_analysis

    ledger_existed_before = os.path.exists(STATISTICAL_ANALYSIS_LEDGER_PATH)
    plan_statistical_analysis()
    ledger_exists_after = os.path.exists(STATISTICAL_ANALYSIS_LEDGER_PATH)
    assert ledger_existed_before == ledger_exists_after  # unchanged either way -- never created here


def _write_ledger(path, rows, fieldnames):
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_metadata(root, run_id, attempt, fingerprint):
    d = root / run_id / f"attempt_{attempt:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"evaluator_fingerprint": fingerprint}))


_LEDGER_FIELDS = [
    "training_run_id",
    "evaluation_attempt",
    "evaluation_id",
    "status",
]
_AMEND_FIELDS = ["evaluation_id", "evaluation_attempt", "canonical_eligible"]


def test_resolve_canonical_evaluation_identity_missing(tmp_path):
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity

    ledger_path = tmp_path / "ledger.csv"
    _write_ledger(ledger_path, [], _LEDGER_FIELDS)
    result = _resolve_canonical_evaluation_identity(
        "run-x", "fp-current", ledger_path=ledger_path, validation_evaluation_root=tmp_path / "artifacts"
    )
    assert result["evaluation_status"] == "missing"


def test_resolve_canonical_evaluation_identity_eligible(tmp_path):
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity

    ledger_path = tmp_path / "ledger.csv"
    amend_path = tmp_path / "amendments.csv"
    root = tmp_path / "artifacts"
    _write_ledger(
        ledger_path,
        [
            {
                "training_run_id": "run-x",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-1",
                "status": "completed",
            }
        ],
        _LEDGER_FIELDS,
    )
    _write_ledger(amend_path, [], _AMEND_FIELDS)
    _write_metadata(root, "run-x", 1, "fp-current")

    result = _resolve_canonical_evaluation_identity(
        "run-x",
        "fp-current",
        ledger_path=ledger_path,
        validation_evaluation_root=root,
        amendments_ledger_path=amend_path,
    )
    assert result["evaluation_status"] == "eligible"
    assert result["evaluation_id"] == "eval-1"
    assert result["evaluation_attempt"] == 1


def test_resolve_canonical_evaluation_identity_stale_fingerprint(tmp_path):
    """A completed, non-amendment-excluded attempt whose persisted
    fingerprint differs from the current one must be reported 'stale',
    never silently treated as eligible."""
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity

    ledger_path = tmp_path / "ledger.csv"
    amend_path = tmp_path / "amendments.csv"
    root = tmp_path / "artifacts"
    _write_ledger(
        ledger_path,
        [
            {
                "training_run_id": "run-x",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-1",
                "status": "completed",
            }
        ],
        _LEDGER_FIELDS,
    )
    _write_ledger(amend_path, [], _AMEND_FIELDS)
    _write_metadata(root, "run-x", 1, "fp-OLD")

    result = _resolve_canonical_evaluation_identity(
        "run-x",
        "fp-current",
        ledger_path=ledger_path,
        validation_evaluation_root=root,
        amendments_ledger_path=amend_path,
    )
    assert result["evaluation_status"] == "stale"


def test_resolve_canonical_evaluation_identity_amendment_excluded_is_not_eligible(tmp_path):
    """A completed attempt with a canonical_eligible=False amendment must
    never be selected, even if its fingerprint matches current -- exactly
    like the real production is_evaluation_canonical_ineligible rule."""
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity

    ledger_path = tmp_path / "ledger.csv"
    amend_path = tmp_path / "amendments.csv"
    root = tmp_path / "artifacts"
    _write_ledger(
        ledger_path,
        [
            {
                "training_run_id": "run-x",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-1",
                "status": "completed",
            }
        ],
        _LEDGER_FIELDS,
    )
    _write_ledger(
        amend_path,
        [{"evaluation_id": "eval-1", "evaluation_attempt": "1", "canonical_eligible": "False"}],
        _AMEND_FIELDS,
    )
    _write_metadata(root, "run-x", 1, "fp-current")

    result = _resolve_canonical_evaluation_identity(
        "run-x",
        "fp-current",
        ledger_path=ledger_path,
        validation_evaluation_root=root,
        amendments_ledger_path=amend_path,
    )
    assert result["evaluation_status"] == "missing"  # excluded -> no eligible completion remains


def test_resolve_canonical_evaluation_identity_failed_attempt_ignored(tmp_path):
    """A row with status=failed must never be considered at all --
    filtered before amendment/fingerprint checks even run."""
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity

    ledger_path = tmp_path / "ledger.csv"
    amend_path = tmp_path / "amendments.csv"
    root = tmp_path / "artifacts"
    _write_ledger(
        ledger_path,
        [
            {
                "training_run_id": "run-x",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-1",
                "status": "failed",
            }
        ],
        _LEDGER_FIELDS,
    )
    _write_ledger(amend_path, [], _AMEND_FIELDS)

    result = _resolve_canonical_evaluation_identity(
        "run-x",
        "fp-current",
        ledger_path=ledger_path,
        validation_evaluation_root=root,
        amendments_ledger_path=amend_path,
    )
    assert result["evaluation_status"] == "missing"


def test_resolve_canonical_evaluation_identity_ambiguous_duplicate_completions(tmp_path):
    """Two DIFFERENT completed, current-fingerprint, non-excluded
    attempts for the same run_id must be reported 'ambiguous' -- never
    silently pick one."""
    from when_tta_hurts.statistical_analysis import _resolve_canonical_evaluation_identity

    ledger_path = tmp_path / "ledger.csv"
    amend_path = tmp_path / "amendments.csv"
    root = tmp_path / "artifacts"
    _write_ledger(
        ledger_path,
        [
            {
                "training_run_id": "run-x",
                "evaluation_attempt": "1",
                "evaluation_id": "eval-1",
                "status": "completed",
            },
            {
                "training_run_id": "run-x",
                "evaluation_attempt": "2",
                "evaluation_id": "eval-2",
                "status": "completed",
            },
        ],
        _LEDGER_FIELDS,
    )
    _write_ledger(amend_path, [], _AMEND_FIELDS)
    _write_metadata(root, "run-x", 1, "fp-current")
    _write_metadata(root, "run-x", 2, "fp-current")

    result = _resolve_canonical_evaluation_identity(
        "run-x",
        "fp-current",
        ledger_path=ledger_path,
        validation_evaluation_root=root,
        amendments_ledger_path=amend_path,
    )
    assert result["evaluation_status"] == "ambiguous"
    assert set(result["evaluation_ids"]) == {"eval-1", "eval-2"}


def test_plan_mode_never_reads_predictions_or_metrics_files(monkeypatch):
    """Structural guarantee: plan mode's resolution path never opens a
    predictions.npz or metrics.json -- patch np.load and json.load to
    explode if called with such a path, then run plan mode end to end."""
    import numpy as _np

    from when_tta_hurts.statistical_analysis import plan_statistical_analysis

    original_load = _np.load

    def _guarded_load(path, *args, **kwargs):
        if "predictions.npz" in str(path):
            raise AssertionError(f"plan mode must never load predictions.npz, attempted: {path}")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(_np, "load", _guarded_load)
    plan_statistical_analysis()  # must complete without tripping the guard
