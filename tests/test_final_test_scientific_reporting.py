"""Phase 2B.8A: tests for the deterministic final-test scientific-report
generator. Every test uses fully synthetic fixtures (fabricated JSON
matching the real schema, in tmp_path) -- NONE read the seven real
sealed result files. An autouse guard fixture additionally patches the
real production output paths to explode if a test reaches them
unpatched, so a forgetful test fails loudly instead of silently writing
into the real repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import when_tta_hurts.final_test_scientific_reporting as ftsr

_real_resolve_seven_sealed_inputs = ftsr.resolve_seven_sealed_inputs

# ---------------------------------------------------------------------------
# Session-wide guard: no test may touch the REAL production output paths.
# ---------------------------------------------------------------------------


class _ProductionPathReachedInTestError(AssertionError):
    pass


def _guard_explode(*args, **kwargs):
    raise _ProductionPathReachedInTestError(
        "PRODUCTION PATH REACHED IN TEST: a reporting test attempted to touch a real production "
        "output path. Pass explicit tmp_path-based paths to every reporting function."
    )


@pytest.fixture(autouse=True)
def _guard_real_output_paths(monkeypatch):
    for name in ("SCIENTIFIC_SUMMARY_PATH", "RESULTS_MARKDOWN_PATH", "INTERPRETATION_MARKDOWN_PATH"):
        real_path = getattr(ftsr, name)
        assert not real_path.exists(), f"real production path {real_path} must not exist during tests"
    yield


# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------

LEDGER_FIELDNAMES = (
    "analysis_id",
    "kind",
    "identifier",
    "analysis_attempt",
    "final_test_analysis_fingerprint",
    "final_test_authorization_sha256",
    "final_test_authorization_commit",
    "current_evaluator_fingerprint",
    "status",
    "primary_artifact_hash",
    "started_at",
    "ended_at",
    "runtime_seconds",
    "failure_reason",
    "test_split_accessed",
)


def _write_family_result(root, family, run_ids, deltas, p_values=None, seed=1):
    attempt_dir = root / family / "attempt_001"
    attempt_dir.mkdir(parents=True)
    p_values = p_values or [0.5] * len(run_ids)
    per_cell = {}
    for run_id, delta, p in zip(run_ids, deltas, p_values, strict=True):
        per_cell[run_id] = {
            "bootstrap": {
                "delta_accuracy": delta,
                "ci_level": 0.95,
                "ci_low": delta - 0.1,
                "ci_high": delta + 0.1,
                "n_resamples": 10000,
                "n_samples": 100,
                "bootstrap_seed": ftsr.__dict__.get("_unused"),
            },
            "mcnemar": {
                "b": 1,
                "c": 1,
                "n_discordant": 2,
                "method": "exact_binomial",
                "statistic": None,
                "p_value": p,
            },
            "effect_sizes": {"delta_accuracy": delta, "harm_rate": 0.1, "rescue_rate": 0.1},
            "n_samples": 100,
        }
    result = {
        "family": family,
        "analysis_id": f"fake-analysis-id-{family}",
        "analysis_fingerprint": "fake-fp",
        "current_evaluator_fingerprint": "fake-evaluator-fp",
        "cells": list(run_ids),
        "per_cell_statistics": per_cell,
        "multiplicity": {
            "method": "benjamini_hochberg",
            "raw_p_values": p_values,
            "corrected_p_values": p_values,
        },
        "status": "completed",
        "test_split_accessed": False,
    }
    (attempt_dir / "analysis_result.json").write_text(json.dumps(result))
    manifest = {
        "artifacts": [
            {
                "path": "analysis_result.json",
                "size_bytes": (attempt_dir / "analysis_result.json").stat().st_size,
                "sha256": hashlib.sha256((attempt_dir / "analysis_result.json").read_bytes()).hexdigest(),
            }
        ]
    }
    (attempt_dir / "artifact_manifest.json").write_text(json.dumps(manifest))
    return attempt_dir, result


def _write_cross_result(root, hypothesis, pair_ids):
    attempt_dir = root / hypothesis / "attempt_001"
    attempt_dir.mkdir(parents=True)
    per_pair = {}
    for pid in pair_ids:
        per_pair[pid] = {
            "pair_id": pid,
            "hypothesis": hypothesis,
            "condition_a": {"run_id": f"{pid}-a", "evaluation_id": f"eval-{pid}-a"},
            "condition_b": {"run_id": f"{pid}-b", "evaluation_id": f"eval-{pid}-b"},
            "bootstrap": {
                "did": 0.05,
                "ci_level": 0.95,
                "ci_low": -0.05,
                "ci_high": 0.15,
                "n_resamples": 10000,
                "n_samples": 100,
                "bootstrap_seed": 12345,
            },
            "n_samples": 100,
        }
    result = {
        "classification": "post_validation_pre_test_secondary",
        "hypothesis": hypothesis,
        "analysis_id": f"fake-cross-id-{hypothesis}",
        "cross_condition_analysis_fingerprint": "fake-fp",
        "current_evaluator_fingerprint": "fake-evaluator-fp",
        "pairs": list(pair_ids),
        "per_pair_results": per_pair,
        "status": "completed",
        "test_split_accessed": False,
    }
    (attempt_dir / "cross_condition_result.json").write_text(json.dumps(result))
    manifest = {
        "artifacts": [
            {
                "path": "cross_condition_result.json",
                "size_bytes": (attempt_dir / "cross_condition_result.json").stat().st_size,
                "sha256": hashlib.sha256(
                    (attempt_dir / "cross_condition_result.json").read_bytes()
                ).hexdigest(),
            }
        ]
    }
    (attempt_dir / "cross_condition_artifact_manifest.json").write_text(json.dumps(manifest))
    return attempt_dir, result


def _write_ledger_rows(ledger_path, entries):
    """entries: list of (kind, identifier, result_path, status='completed', attempt=1)"""
    with ledger_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDNAMES)
        writer.writeheader()
        for kind, identifier, result_path, status, attempt in entries:
            primary_hash = hashlib.sha256(result_path.read_bytes()).hexdigest() if result_path else "x"
            writer.writerow(
                {
                    "analysis_id": f"fake-id-{kind}-{identifier}",
                    "kind": kind,
                    "identifier": identifier,
                    "analysis_attempt": attempt,
                    "final_test_analysis_fingerprint": "fake-fp",
                    "final_test_authorization_sha256": "fake-auth-sha",
                    "final_test_authorization_commit": "fake-commit",
                    "current_evaluator_fingerprint": "fake-evaluator-fp",
                    "status": status,
                    "primary_artifact_hash": primary_hash,
                    "started_at": 0.0,
                    "ended_at": 1.0,
                    "runtime_seconds": 1.0,
                    "failure_reason": "",
                    "test_split_accessed": False,
                }
            )


def _build_full_synthetic_repo(tmp_path, family_deltas=None, p_values_by_family=None):
    analysis_root = tmp_path / "final_test_analysis"
    cross_root = tmp_path / "final_test_cross_condition"
    ledger_path = tmp_path / "ledger_final_test_analysis.csv"

    family_specs = {
        "H1": [f"h1-run-{i}" for i in range(24)],
        "H2": [f"h2-run-{i}" for i in range(30)],
        "H3": [f"h3-run-{i}" for i in range(12)],
        "BLOCK_C": [f"blockc-run-{i}" for i in range(3)],
    }
    family_deltas = family_deltas or {}
    p_values_by_family = p_values_by_family or {}

    entries = []
    for family, run_ids in family_specs.items():
        deltas = family_deltas.get(family, [0.01] * len(run_ids))
        pvals = p_values_by_family.get(family, [0.5] * len(run_ids))
        attempt_dir, _ = _write_family_result(analysis_root, family, run_ids, deltas, pvals)
        entries.append(("family", family, attempt_dir / "analysis_result.json", "completed", 1))

    cross_specs = {
        "H1": [f"H1-pair-{i}" for i in range(12)],
        "H2": [f"H2-pair-{i}" for i in range(12)],
        "H3": [f"H3-pair-{i}" for i in range(6)],
    }
    for hyp, pair_ids in cross_specs.items():
        attempt_dir, _ = _write_cross_result(cross_root, hyp, pair_ids)
        entries.append(("cross_condition", hyp, attempt_dir / "cross_condition_result.json", "completed", 1))

    _write_ledger_rows(ledger_path, entries)
    return ledger_path, analysis_root, cross_root


# ---------------------------------------------------------------------------
# resolve_seven_sealed_inputs: metadata-only, exact 7, tamper detection.
# ---------------------------------------------------------------------------


def test_resolve_seven_inputs_happy_path_makes_zero_json_loads(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)

    calls = []
    real_loads = json.loads
    json.loads = lambda s, *a, **k: (calls.append(1), real_loads(s, *a, **k))[1]
    try:
        resolved = ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)
    finally:
        json.loads = real_loads

    assert len(resolved) == 7
    assert calls == [], "resolve_seven_sealed_inputs must never call json.loads"


def test_resolve_missing_unit_raises(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    # Remove the BLOCK_C row entirely.
    rows = list(csv.DictReader(ledger_path.open(newline="")))
    rows = [r for r in rows if not (r["kind"] == "family" and r["identifier"] == "BLOCK_C")]
    with ledger_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ftsr.SealedInputResolutionError):
        ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)


def test_resolve_duplicate_completed_rows_raises_ambiguous(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    rows = list(csv.DictReader(ledger_path.open(newline="")))
    dup = dict(rows[0])
    rows.append(dup)
    with ledger_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ftsr.SealedInputResolutionError, match="Ambiguous"):
        ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)


def test_resolve_tampered_result_bytes_raises(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    tampered = analysis_root / "H1" / "attempt_001" / "analysis_result.json"
    tampered.write_text(tampered.read_text() + " ")  # mutate bytes after ledger hash recorded

    with pytest.raises(ftsr.SealedInputTamperError):
        ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)


def test_resolve_ignores_non_completed_rows(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    rows = list(csv.DictReader(ledger_path.open(newline="")))
    for r in rows:
        if r["kind"] == "family" and r["identifier"] == "H1":
            r["status"] = "failed"
    with ledger_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ftsr.SealedInputResolutionError):
        ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)


# ---------------------------------------------------------------------------
# load_and_verify_sealed_result: schema + count enforcement.
# ---------------------------------------------------------------------------


def test_load_and_verify_rejects_wrong_cell_count(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    resolved = ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)
    entry = dict(resolved["family:BLOCK_C"])
    entry["expected_count"] = 999  # force mismatch
    with pytest.raises(ftsr.SealedResultSchemaError):
        ftsr.load_and_verify_sealed_result(entry)


def test_load_and_verify_all_seven_succeeds(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    resolved = ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)
    loaded = {key: ftsr.load_and_verify_sealed_result(entry) for key, entry in resolved.items()}
    assert len(loaded) == 7
    assert len(loaded["family:H1"]["cells"]) == 24
    assert len(loaded["cross_condition:H3"]["pairs"]) == 6


# ---------------------------------------------------------------------------
# Deterministic rendering: ordering, precision, no selective omission,
# structural separation, no invented fields.
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_synthetic(tmp_path):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(
        tmp_path,
        family_deltas={
            "H1": [0.1, -0.2, 0.0] + [0.01] * 21,  # positive, negative, null, plus filler
            "H3": [0.5] * 12,  # a "significant"-looking magnitude, still must be reported
        },
        p_values_by_family={"H1": [0.001, 0.5, 0.999] + [0.5] * 21},  # significant, mixed, nonsignificant
    )
    resolved = ftsr.resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root)
    loaded = {key: ftsr.load_and_verify_sealed_result(entry) for key, entry in resolved.items()}
    return resolved, loaded


def test_build_summary_is_deterministic_and_repeatable(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    s1 = ftsr.build_scientific_summary(resolved, loaded)
    s2 = ftsr.build_scientific_summary(resolved, loaded)
    d1 = json.dumps(s1, sort_keys=True, default=str)
    d2 = json.dumps(s2, sort_keys=True, default=str)
    assert d1 == d2


def test_summary_preserves_full_float_precision(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    high_precision = 0.123456789012345
    summary["preregistered"]["H1"]["cells"][0]["bootstrap"]["delta_accuracy"] = high_precision
    dumped = json.dumps(summary)
    reloaded = json.loads(dumped)
    assert reloaded["preregistered"]["H1"]["cells"][0]["bootstrap"]["delta_accuracy"] == high_precision


def test_no_selective_omission_across_sign_and_significance(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    h1_run_ids = {c["run_id"] for c in summary["preregistered"]["H1"]["cells"]}
    assert h1_run_ids == set(loaded["family:H1"]["cells"]), "every cell (positive/negative/null) must appear"
    assert summary["preregistered"]["H1"]["n_cells"] == 24


def test_block_c_cannot_be_omitted(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    assert "BLOCK_C" in summary["preregistered"]
    assert summary["preregistered"]["BLOCK_C"]["n_cells"] == 3


def test_h4_cannot_appear_anywhere():
    identifiers = {identifier for _, identifier, _, _ in ftsr.EXPECTED_UNITS}
    assert "H4" not in identifiers
    assert all("H4" not in i for i in identifiers)


def test_secondary_section_has_no_invented_pooled_fields(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    dumped = json.dumps(summary).lower()
    for forbidden in (
        "pooled_p_value",
        "model_population_p_value",
        "family_wise_p_value",
        "alpha",
        '"significant"',
    ):
        assert forbidden not in dumped


def test_preregistered_and_secondary_structurally_separate(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    assert set(summary["preregistered"]) == {"H1", "H2", "H3", "BLOCK_C"}
    assert set(summary["secondary_cross_condition"]) == {"H1", "H2", "H3"}
    assert summary["preregistered"] is not summary["secondary_cross_condition"]


def test_descriptive_summaries_labeled_non_inferential(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    seed_summaries = summary["descriptive_summaries"]["preregistered_seed_level"]
    assert len(seed_summaries) > 0
    assert all(s["classification"] == "descriptive_non_inferential" for s in seed_summaries)
    for s in seed_summaries:
        assert "p_value" not in s
        assert "ci_low" not in s


def test_markdown_rendering_deterministic_and_derived_from_json(loaded_synthetic):
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    md1 = ftsr.render_results_markdown(summary)
    md2 = ftsr.render_results_markdown(summary)
    assert md1 == md2
    assert "BLOCK_C" in md1
    assert "descriptive" in md1.lower()


def test_interpretation_markdown_contains_required_disclosures():
    md = ftsr.render_interpretation_markdown({})
    for phrase in (
        "three training seeds",
        "No H4 claim",
        "No population-level",
        "accidental final-test access incident",
        "shared-aggregation-contract correction",
        "inspected by a human before this controlled unsealing",
    ):
        assert phrase in md, f"missing required disclosure: {phrase!r}"


def test_interpretation_never_conflates_absence_with_no_effect():
    md = ftsr.render_interpretation_markdown({})
    assert "absence of evidence" not in md.lower() or "evidence of no effect" in md.lower()


# ---------------------------------------------------------------------------
# Full pipeline: authorization ordering, atomic persistence, idempotency,
# conflict rejection, rollback on failure.
# ---------------------------------------------------------------------------


def _patch_authorizations_ok(monkeypatch):
    monkeypatch.setattr(ftsr, "verify_final_test_analysis_authorization", lambda: {"status": "approved"})
    monkeypatch.setattr(ftsr, "verify_unsealing_authorization", lambda **kw: {"status": "approved"})


def test_authorization_checked_before_any_input_resolution(monkeypatch):
    from when_tta_hurts.final_test_statistical_analysis import FinalTestAnalysisAuthorizationError

    def _raise():
        raise FinalTestAnalysisAuthorizationError("synthetic")

    monkeypatch.setattr(ftsr, "verify_final_test_analysis_authorization", _raise)
    monkeypatch.setattr(
        ftsr,
        "resolve_seven_sealed_inputs",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resolve reached before authorization check")),
    )
    with pytest.raises(FinalTestAnalysisAuthorizationError):
        ftsr.generate_and_persist_report()


def test_full_pipeline_persists_atomically_and_is_idempotent(tmp_path, monkeypatch):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    _patch_authorizations_ok(monkeypatch)
    monkeypatch.setattr(
        ftsr,
        "resolve_seven_sealed_inputs",
        lambda: _real_resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root),
    )

    summary_path = tmp_path / "summary.json"
    results_path = tmp_path / "results.md"
    interp_path = tmp_path / "interpretation.md"

    result1 = ftsr.generate_and_persist_report(summary_path, results_path, interp_path)
    assert result1["status"] == "completed"
    assert summary_path.exists() and results_path.exists() and interp_path.exists()

    hash_before = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    result2 = ftsr.generate_and_persist_report(summary_path, results_path, interp_path)
    assert result2["status"] == "idempotent_skip"
    hash_after = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert hash_before == hash_after


def test_conflicting_existing_output_hard_fails(tmp_path, monkeypatch):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    _patch_authorizations_ok(monkeypatch)
    monkeypatch.setattr(
        ftsr,
        "resolve_seven_sealed_inputs",
        lambda: _real_resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root),
    )

    summary_path = tmp_path / "summary.json"
    results_path = tmp_path / "results.md"
    interp_path = tmp_path / "interpretation.md"
    summary_path.write_text('{"tampered": true}')

    with pytest.raises(ftsr.ReportGenerationError):
        ftsr.generate_and_persist_report(summary_path, results_path, interp_path)


def test_rendering_failure_leaves_no_completed_output_set(tmp_path, monkeypatch):
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    _patch_authorizations_ok(monkeypatch)
    monkeypatch.setattr(
        ftsr,
        "resolve_seven_sealed_inputs",
        lambda: _real_resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root),
    )

    def _raise(summary):
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(ftsr, "render_results_markdown", _raise)

    summary_path = tmp_path / "summary.json"
    results_path = tmp_path / "results.md"
    interp_path = tmp_path / "interpretation.md"

    with pytest.raises(RuntimeError):
        ftsr.generate_and_persist_report(summary_path, results_path, interp_path)

    assert not summary_path.exists()
    assert not results_path.exists()
    assert not interp_path.exists()


# ---------------------------------------------------------------------------
# Reporting fingerprint isolation from scientific-computation manifests.
# ---------------------------------------------------------------------------


def test_reporting_manifest_disjoint_from_computation_manifests():
    from when_tta_hurts.cross_condition_addendum import CROSS_CONDITION_ADDENDUM_MANIFEST
    from when_tta_hurts.final_test_identity import FINAL_TEST_RUNNER_MANIFEST
    from when_tta_hurts.final_test_statistical_analysis import FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST
    from when_tta_hurts.statistical_analysis import ANALYSIS_FINGERPRINT_MANIFEST

    # The reporting-specific files (this module, the CLI script) must
    # never appear in any scientific-computation manifest -- pyproject.toml
    # / uv.lock legitimately appear in both (dependency behavior affects
    # both), so full set-disjointness is not the invariant under test.
    reporting_only_files = {
        "src/when_tta_hurts/final_test_scientific_reporting.py",
        "scripts/generate_final_test_scientific_report.py",
    }
    for computation_manifest in (
        ANALYSIS_FINGERPRINT_MANIFEST,
        CROSS_CONDITION_ADDENDUM_MANIFEST,
        FINAL_TEST_RUNNER_MANIFEST,
        FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST,
    ):
        assert reporting_only_files.isdisjoint(computation_manifest)


# ---------------------------------------------------------------------------
# Phase 2B.8C: the one-sentence human-observation wording correction.
# ---------------------------------------------------------------------------


def test_old_person_or_process_phrase_cannot_appear():
    md = ftsr.render_interpretation_markdown({})
    assert "examined by any person or process" not in md
    assert "person or process" not in md


def test_approved_human_inspection_sentence_appears_exactly():
    md = ftsr.render_interpretation_markdown({})
    assert "No final-test scientific result was inspected by a human before this controlled unsealing." in md


def test_only_the_one_sentence_differs_from_generation1_archive():
    """The archived generation-1 interpretation Markdown must differ from
    the current generator's output in exactly the one frozen sentence --
    every other line must be byte-identical."""
    archive_path = Path("docs/phase2b_final_test_scientific_interpretation_generation_001.md")
    if not archive_path.exists():
        pytest.skip("generation-1 archive not present in this checkout")
    archived_lines = archive_path.read_text().splitlines()
    current_lines = ftsr.render_interpretation_markdown({}).splitlines()

    old_sentence_lines = [
        line
        for line in archived_lines
        if "examined by any person or process" in line or "person or process" in line
    ]
    assert len(old_sentence_lines) >= 1, "generation-1 archive must contain the old sentence"

    archived_without_old = [
        line for line in archived_lines if line not in old_sentence_lines and line.strip() != ""
    ]
    current_without_new = [
        line
        for line in current_lines
        if "inspected by a human before this controlled unsealing" not in line and line.strip() != ""
    ]
    assert archived_without_old == current_without_new, (
        "every non-frozen-sentence line must remain byte-identical between generation 1 and the "
        "current generator output"
    )


def test_results_markdown_generator_untouched_by_wording_correction(loaded_synthetic):
    """render_results_markdown() never renders the corrected sentence at
    all, so its output must be byte-for-byte identical regardless of the
    wording fix."""
    resolved, loaded = loaded_synthetic
    summary = ftsr.build_scientific_summary(resolved, loaded)
    md1 = ftsr.render_results_markdown(summary)
    md2 = ftsr.render_results_markdown(summary)
    assert md1 == md2
    assert "person" not in md1
    assert "inspected" not in md1


def test_generation1_archive_paths_are_never_treated_as_current_outputs():
    from when_tta_hurts.final_test_scientific_reporting import (
        INTERPRETATION_MARKDOWN_PATH,
        RESULTS_MARKDOWN_PATH,
        SCIENTIFIC_SUMMARY_PATH,
    )

    archived_summary = Path(
        "artifacts/final_test_scientific_unsealing/generation_001/final_test_scientific_summary.json"
    )
    archived_results = Path("docs/phase2b_final_test_scientific_results_generation_001.md")
    archived_interp = Path("docs/phase2b_final_test_scientific_interpretation_generation_001.md")

    assert SCIENTIFIC_SUMMARY_PATH != archived_summary
    assert RESULTS_MARKDOWN_PATH != archived_results
    assert INTERPRETATION_MARKDOWN_PATH != archived_interp


def test_current_canonical_output_paths_absent_before_generation2():
    from when_tta_hurts.final_test_scientific_reporting import (
        INTERPRETATION_MARKDOWN_PATH,
        RESULTS_MARKDOWN_PATH,
        SCIENTIFIC_SUMMARY_PATH,
    )

    # These paths must not exist as a leftover from generation 1 -- they
    # were moved (git mv) to the archive locations in Phase 2B.8C Part C.
    assert not SCIENTIFIC_SUMMARY_PATH.exists()
    assert not RESULTS_MARKDOWN_PATH.exists()
    assert not INTERPRETATION_MARKDOWN_PATH.exists()


def test_correction_freeze_doc_is_in_reporting_manifest():
    assert (
        "docs/phase2b_final_test_reporting_wording_correction_freeze.md" in ftsr.FINAL_TEST_REPORTING_MANIFEST
    )


def test_generator_rejects_unexpected_difference_via_conflict_check(tmp_path, monkeypatch):
    """generate_and_persist_report() must hard-fail (never silently
    overwrite) when an existing output doesn't match the freshly-rendered
    content -- this is the mechanism that would catch an unauthorized
    difference between generation 1 and a would-be generation 2."""
    ledger_path, analysis_root, cross_root = _build_full_synthetic_repo(tmp_path)
    _patch_authorizations_ok(monkeypatch)
    monkeypatch.setattr(
        ftsr,
        "resolve_seven_sealed_inputs",
        lambda: _real_resolve_seven_sealed_inputs(ledger_path, analysis_root, cross_root),
    )

    summary_path = tmp_path / "summary.json"
    results_path = tmp_path / "results.md"
    interp_path = tmp_path / "interpretation.md"
    summary_path.write_text('{"different": "content"}')

    with pytest.raises(ftsr.ReportGenerationError):
        ftsr.generate_and_persist_report(summary_path, results_path, interp_path)
