"""Phase 2B.9A: tests for the downstream, presentation-only paper-
evidence generator. Every test uses fully synthetic fixtures (fabricated
dicts matching the real `final_test_scientific_summary.json` schema) --
NONE read the real canonical summary or any real sealed artifact. This
module (and this test file) must run in the ROOT environment WITHOUT
matplotlib installed: no test here may import matplotlib, directly or
transitively, at collection or execution time.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import when_tta_hurts.paper_evidence as pe

REPO_ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Static guard: matplotlib must never be importable from module import time,
# and must never be imported by any scientific-computation/evaluation
# module in the repository (only paper_evidence.py's figure-rendering
# functions may reference it, lazily, inside function bodies).
# ---------------------------------------------------------------------------


def test_matplotlib_not_imported_at_module_load_time():
    assert "matplotlib" not in sys.modules, (
        "matplotlib must not be imported merely by importing when_tta_hurts.paper_evidence"
    )


def test_no_scientific_module_imports_matplotlib():
    src_root = REPO_ROOT / "src" / "when_tta_hurts"
    offenders = []
    for path in sorted(src_root.glob("*.py")):
        if path.name == "paper_evidence.py":
            continue
        text = path.read_text()
        if "import matplotlib" in text or "from matplotlib" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"matplotlib imported by non-presentation module(s): {offenders}"


def test_paper_evidence_module_level_matplotlib_import_absent():
    text = (REPO_ROOT / "src" / "when_tta_hurts" / "paper_evidence.py").read_text()
    lines = text.splitlines()
    module_level_import_lines = [
        line
        for line in lines[:40]
        if line.strip().startswith("import matplotlib") or line.strip().startswith("from matplotlib")
    ]
    assert module_level_import_lines == [], "matplotlib must only be imported inside rendering functions"


def test_extraction_and_table_functions_do_not_import_matplotlib(monkeypatch):
    """Exercising every extraction/table/plan function must not pull in
    matplotlib as a side effect."""
    summary = _synthetic_summary()
    pe.build_evidence_plan(summary)
    unmatched = pe.extract_unmatched_cells(summary)
    matched = pe.extract_matched_within_cell(summary)
    h1 = pe.extract_cross_condition_pairs(summary, "H1")
    h2 = pe.extract_cross_condition_pairs(summary, "H2")
    h3 = pe.extract_cross_condition_pairs(summary, "H3")
    block_c = pe.extract_block_c(summary)
    pe.render_design_classification_table()
    pe.render_unmatched_table(unmatched)
    pe.render_matched_table(matched, h3)
    pe.render_cross_condition_table("Table 4", h1)
    pe.render_cross_condition_table("Table 5", h2)
    pe.render_block_c_table(block_c)
    pe.render_claim_adjudication_table()
    assert "matplotlib" not in sys.modules


# ---------------------------------------------------------------------------
# Synthetic fixture builder: matches the real schema (phase2b.8a-v1)
# exactly, but with small, fabricated cardinalities so tests exercise the
# *shape* of the logic without relying on real scientific values. Because
# `load_and_verify_canonical_summary` enforces the real frozen
# cardinalities (24/30/12/3, 12/12/6) as a fail-closed check, tests of the
# pure extraction functions call them directly on this fixture rather
# than going through the full verification gate (that gate is tested
# separately below with monkeypatched verification hooks).
# ---------------------------------------------------------------------------


def _bootstrap(delta: float) -> dict:
    return {
        "delta_accuracy": delta,
        "ci_low": delta - 0.02,
        "ci_high": delta + 0.02,
        "ci_level": 0.95,
        "n_resamples": 10000,
        "n_samples": 100,
        "bootstrap_seed": 1,
    }


def _cell(run_id: str, delta: float) -> dict:
    return {
        "run_id": run_id,
        "bootstrap": _bootstrap(delta),
        "mcnemar": {
            "b": 1,
            "c": 5,
            "n_discordant": 6,
            "method": "exact_binomial",
            "statistic": None,
            "p_value": 1e-10,
        },
        "effect_sizes": {"delta_accuracy": delta, "harm_rate": 0.1, "rescue_rate": 0.01},
        "n_samples": 100,
    }


def _pair(pair_id: str, run_a: str, run_b: str, did: float) -> dict:
    return {
        "pair_id": pair_id,
        "hypothesis": pair_id.split("-")[0],
        "condition_a": {"run_id": run_a, "evaluation_id": f"eval-{run_a}"},
        "condition_b": {"run_id": run_b, "evaluation_id": f"eval-{run_b}"},
        "bootstrap": {
            "did": did,
            "ci_low": did - 0.02,
            "ci_high": did + 0.02,
            "ci_level": 0.95,
            "n_resamples": 10000,
            "n_samples": 100,
            "bootstrap_seed": 2,
        },
        "n_samples": 100,
    }


_H1_UNMATCHED_RUN_IDS = [
    "A-bloodmnist-28px-batchnorm-policy-none-s0",
    "A-bloodmnist-28px-groupnorm-policy-none-s0",
]
_H2_ONLY_RUN_IDS = [
    "B-pathmnist-64px-batchnorm-policy-none-s0",
]
_H3_UNMATCHED_RUN_IDS = [
    "A-bloodmnist-28px-batchnorm-policy-none-s0",  # shared with H1
]
_H3_MATCHED_RUN_IDS = [
    "D-bloodmnist-28px-batchnorm-policy-matched_mixed-s0",
]


def _synthetic_summary() -> dict:
    h1_cells = [_cell(r, -0.03) for r in _H1_UNMATCHED_RUN_IDS]
    h2_cells = [_cell(r, -0.03) for r in _H1_UNMATCHED_RUN_IDS] + [_cell(r, -0.02) for r in _H2_ONLY_RUN_IDS]
    h3_cells = [_cell(r, -0.03) for r in _H3_UNMATCHED_RUN_IDS] + [
        _cell(r, -0.001) for r in _H3_MATCHED_RUN_IDS
    ]
    block_c_cells = [_cell(f"C-dermamnist-28px-resnet18-batchnorm-policy-none-s{s}", 0.001) for s in range(3)]

    def _mult(cells):
        p = [c["mcnemar"]["p_value"] for c in cells]
        return {"method": "benjamini_hochberg", "raw_p_values": p, "corrected_p_values": [x * 2 for x in p]}

    h1_pairs = [
        _pair(
            "H1-pair-0",
            "A-bloodmnist-28px-batchnorm-policy-none-s0",
            "A-bloodmnist-28px-groupnorm-policy-none-s0",
            0.01,
        )
    ]
    h2_pairs = [
        _pair(
            "H2-pair-0",
            "A-bloodmnist-28px-batchnorm-policy-none-s0",
            "B-pathmnist-64px-batchnorm-policy-none-s0",
            0.02,
        )
    ]
    h3_pairs = [
        _pair(
            "H3-pair-0",
            "A-bloodmnist-28px-batchnorm-policy-none-s0",
            "D-bloodmnist-28px-batchnorm-policy-matched_mixed-s0",
            0.03,
        )
    ]

    return {
        "schema_version": pe.EXPECTED_SCHEMA_VERSION,
        "reporting_fingerprint": "fake-fp",
        "provenance": {},
        "inputs": {},
        "preregistered": {
            "H1": {
                "analysis_id": "a1",
                "attempt": 1,
                "n_cells": len(h1_cells),
                "cells": h1_cells,
                "multiplicity": _mult(h1_cells),
            },
            "H2": {
                "analysis_id": "a2",
                "attempt": 1,
                "n_cells": len(h2_cells),
                "cells": h2_cells,
                "multiplicity": _mult(h2_cells),
            },
            "H3": {
                "analysis_id": "a3",
                "attempt": 1,
                "n_cells": len(h3_cells),
                "cells": h3_cells,
                "multiplicity": _mult(h3_cells),
            },
            "BLOCK_C": {
                "analysis_id": "a4",
                "attempt": 1,
                "n_cells": len(block_c_cells),
                "cells": block_c_cells,
                "multiplicity": _mult(block_c_cells),
            },
        },
        "secondary_cross_condition": {
            "H1": {
                "analysis_id": "c1",
                "attempt": 1,
                "classification": "differs",
                "n_pairs": len(h1_pairs),
                "pairs": h1_pairs,
            },
            "H2": {
                "analysis_id": "c2",
                "attempt": 1,
                "classification": "differs",
                "n_pairs": len(h2_pairs),
                "pairs": h2_pairs,
            },
            "H3": {
                "analysis_id": "c3",
                "attempt": 1,
                "classification": "differs",
                "n_pairs": len(h3_pairs),
                "pairs": h3_pairs,
            },
        },
        "descriptive_summaries": {"preregistered_seed_level": []},
    }


# ---------------------------------------------------------------------------
# _parse_run_id
# ---------------------------------------------------------------------------


def test_parse_run_id_valid():
    identity = pe._parse_run_id("C-dermamnist-28px-resnet18-batchnorm-policy-none-s0")
    assert identity == {
        "block": "C",
        "dataset": "dermamnist",
        "resolution": "28",
        "normalization": "resnet18-batchnorm",
        "policy": "none",
        "seed": "0",
    }


def test_parse_run_id_without_model_segment():
    identity = pe._parse_run_id("A-bloodmnist-28px-batchnorm-policy-none-s3")
    assert identity["normalization"] == "batchnorm"
    assert identity["seed"] == "3"


@pytest.mark.parametrize(
    "bad_run_id",
    ["missing-required-token-s0", "A-bloodmnist-28px-batchnorm-policy-none-sX", "no-seed-here-policy-none"],
)
def test_parse_run_id_rejects_malformed_ids(bad_run_id):
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe._parse_run_id(bad_run_id)


# ---------------------------------------------------------------------------
# extract_unmatched_cells: dedup, member_families, per-family BH-p, raw-p
# invariance assertion.
# ---------------------------------------------------------------------------


def test_extract_unmatched_cells_deduplicates_and_tracks_member_families():
    summary = _synthetic_summary()
    rows = pe.extract_unmatched_cells(summary)
    run_ids = [r["run_id"] for r in rows]
    assert len(run_ids) == len(set(run_ids)), "no unmatched cell may appear more than once"
    assert len(rows) == 3  # 2 H1 cells + 1 H2-only cell, H3's cell is a dup of an H1 cell

    shared = next(r for r in rows if r["run_id"] == "A-bloodmnist-28px-batchnorm-policy-none-s0")
    assert set(shared["member_families"]) == {"H1", "H2", "H3"}
    assert set(shared["bh_adjusted_p_by_family"].keys()) == {"H1", "H2", "H3"}

    unique_to_h2 = next(r for r in rows if r["run_id"] == "B-pathmnist-64px-batchnorm-policy-none-s0")
    assert unique_to_h2["member_families"] == ["H2"]


def test_extract_unmatched_cells_excludes_matched_policy_rows():
    summary = _synthetic_summary()
    rows = pe.extract_unmatched_cells(summary)
    assert all(r["policy"] == "none" for r in rows)
    matched_run_ids = {r["run_id"] for r in rows if "matched" in r["run_id"]}
    assert matched_run_ids == set()


def test_extract_unmatched_cells_is_sorted_deterministically():
    summary = _synthetic_summary()
    rows_a = pe.extract_unmatched_cells(summary)
    rows_b = pe.extract_unmatched_cells(summary)
    assert [r["run_id"] for r in rows_a] == [r["run_id"] for r in rows_b]
    sort_keys = [(r["dataset"], int(r["resolution"]), r["normalization"], int(r["seed"])) for r in rows_a]
    assert sort_keys == sorted(sort_keys)


# ---------------------------------------------------------------------------
# extract_matched_within_cell
# ---------------------------------------------------------------------------


def test_extract_matched_within_cell_only_matched_policy():
    summary = _synthetic_summary()
    rows = pe.extract_matched_within_cell(summary)
    assert len(rows) == 1
    assert rows[0]["policy"] == "matched_mixed"
    assert rows[0]["member_families"] == ["H3"]


# ---------------------------------------------------------------------------
# extract_cross_condition_pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hypothesis,expected_count", [("H1", 1), ("H2", 1), ("H3", 1)])
def test_extract_cross_condition_pairs_returns_all_pairs(hypothesis, expected_count):
    summary = _synthetic_summary()
    pairs = pe.extract_cross_condition_pairs(summary, hypothesis)
    assert len(pairs) == expected_count
    for pair in pairs:
        assert "pair_id" in pair
        assert "condition_a" in pair and "condition_b" in pair
        assert "bootstrap" in pair


def test_extract_cross_condition_pairs_sorted_by_pair_id():
    summary = _synthetic_summary()
    summary["secondary_cross_condition"]["H1"]["pairs"] = [
        _pair("H1-pair-2", "x", "y", 0.0),
        _pair("H1-pair-0", "x", "y", 0.0),
        _pair("H1-pair-1", "x", "y", 0.0),
    ]
    pairs = pe.extract_cross_condition_pairs(summary, "H1")
    assert [p["pair_id"] for p in pairs] == ["H1-pair-0", "H1-pair-1", "H1-pair-2"]


# ---------------------------------------------------------------------------
# extract_block_c
# ---------------------------------------------------------------------------


def test_extract_block_c_returns_all_seeds_sorted():
    summary = _synthetic_summary()
    rows = pe.extract_block_c(summary)
    assert len(rows) == 3
    assert [r["seed"] for r in rows] == ["0", "1", "2"]


# ---------------------------------------------------------------------------
# Table rendering: every planned row appears exactly once; no significance
# language in secondary captions; correct BLOCK_C descriptive framing.
# ---------------------------------------------------------------------------


def test_render_unmatched_table_includes_every_row_exactly_once():
    summary = _synthetic_summary()
    rows = pe.extract_unmatched_cells(summary)
    table = pe.render_unmatched_table(rows)
    for row in rows:
        assert table.count(row["run_id"]) == 1


def test_render_matched_table_shows_within_cell_and_did_pairs():
    summary = _synthetic_summary()
    matched = pe.extract_matched_within_cell(summary)
    h3_pairs = pe.extract_cross_condition_pairs(summary, "H3")
    table = pe.render_matched_table(matched, h3_pairs)
    assert matched[0]["run_id"] in table
    assert h3_pairs[0]["pair_id"] in table


@pytest.mark.parametrize("forbidden_word", ["significant", "Significant", "SIGNIFICANT"])
def test_cross_condition_table_never_uses_significance_language(forbidden_word):
    summary = _synthetic_summary()
    h1_pairs = pe.extract_cross_condition_pairs(summary, "H1")
    table = pe.render_cross_condition_table("Table 4", h1_pairs)
    assert forbidden_word not in table


def test_block_c_table_frames_positive_reference_descriptively():
    summary = _synthetic_summary()
    block_c = pe.extract_block_c(summary)
    table = pe.render_block_c_table(block_c)
    assert "descriptive" in table
    assert "not" in table.lower()
    assert "+1.6" in table


def test_claim_adjudication_table_never_asserts_h4():
    table = pe.render_claim_adjudication_table()
    assert "H4" in table  # named only to state it was never made
    assert "Not made, anywhere" in table


# ---------------------------------------------------------------------------
# compute_paper_evidence_fingerprint: fails closed on missing manifest file.
# ---------------------------------------------------------------------------


def test_fingerprint_fails_closed_on_missing_manifest_file(tmp_path):
    (tmp_path / "docs").mkdir()
    with pytest.raises(pe.PaperEvidenceFingerprintError):
        pe.compute_paper_evidence_fingerprint(repo_root=tmp_path)


def test_fingerprint_is_deterministic_across_repeated_calls():
    fp1, _ = pe.compute_paper_evidence_fingerprint()
    fp2, _ = pe.compute_paper_evidence_fingerprint()
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# load_and_verify_canonical_summary: fails closed on every tamper vector.
# Uses monkeypatched verification hooks (never the real production
# summary/authorization) so these tests are fully synthetic.
# ---------------------------------------------------------------------------


@pytest.fixture
def _patched_verification(monkeypatch):
    monkeypatch.setattr(pe, "compute_final_test_reporting_fingerprint", lambda: ("expected-fp", {}))
    monkeypatch.setattr(
        pe,
        "verify_unsealing_authorization",
        lambda: {"status": "approved", "final_test_reporting_fingerprint": "expected-fp"},
    )
    return None


def _full_cardinality_summary() -> dict:
    summary = _synthetic_summary()
    summary["reporting_fingerprint"] = "expected-fp"

    def _fill(family, n):
        base_cell = summary["preregistered"]["H1"]["cells"][0]
        cells = []
        for i in range(n):
            c = json.loads(json.dumps(base_cell))
            c["run_id"] = f"Z-fake-28px-batchnorm-policy-none-s{i}"
            cells.append(c)
        summary["preregistered"][family]["cells"] = cells
        summary["preregistered"][family]["n_cells"] = n
        p = [1e-10] * n
        summary["preregistered"][family]["multiplicity"] = {
            "method": "benjamini_hochberg",
            "raw_p_values": p,
            "corrected_p_values": p,
        }

    def _fill_pairs(hypothesis, n):
        base_pair = summary["secondary_cross_condition"]["H1"]["pairs"][0]
        pairs = []
        for i in range(n):
            p = json.loads(json.dumps(base_pair))
            p["pair_id"] = f"{hypothesis}-pair-{i}"
            pairs.append(p)
        summary["secondary_cross_condition"][hypothesis]["pairs"] = pairs
        summary["secondary_cross_condition"][hypothesis]["n_pairs"] = n

    _fill("H1", 24)
    _fill("H2", 30)
    _fill("H3", 12)
    _fill("BLOCK_C", 3)
    _fill_pairs("H1", 12)
    _fill_pairs("H2", 12)
    _fill_pairs("H3", 6)
    return summary


def test_load_and_verify_canonical_summary_accepts_correctly_shaped_fixture(tmp_path, _patched_verification):
    summary = _full_cardinality_summary()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    loaded = pe.load_and_verify_canonical_summary(path)
    assert loaded["reporting_fingerprint"] == "expected-fp"


def test_load_and_verify_canonical_summary_rejects_missing_file(tmp_path):
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(tmp_path / "does_not_exist.json")


def test_load_and_verify_canonical_summary_rejects_malformed_json(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text("{not valid json")
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(path)


def test_load_and_verify_canonical_summary_rejects_wrong_schema_version(tmp_path, _patched_verification):
    summary = _full_cardinality_summary()
    summary["schema_version"] = "wrong-version"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(path)


def test_load_and_verify_canonical_summary_rejects_tampered_reporting_fingerprint(
    tmp_path, _patched_verification
):
    summary = _full_cardinality_summary()
    summary["reporting_fingerprint"] = "tampered-fp"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(path)


def test_load_and_verify_canonical_summary_rejects_wrong_cell_cardinality(tmp_path, _patched_verification):
    summary = _full_cardinality_summary()
    summary["preregistered"]["H1"]["cells"] = summary["preregistered"]["H1"]["cells"][:1]
    summary["preregistered"]["H1"]["n_cells"] = 1
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(path)


def test_load_and_verify_canonical_summary_rejects_wrong_pair_cardinality(tmp_path, _patched_verification):
    summary = _full_cardinality_summary()
    summary["secondary_cross_condition"]["H1"]["pairs"] = summary["secondary_cross_condition"]["H1"]["pairs"][
        :1
    ]
    summary["secondary_cross_condition"]["H1"]["n_pairs"] = 1
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(path)


def test_load_and_verify_canonical_summary_propagates_authorization_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(pe, "compute_final_test_reporting_fingerprint", lambda: ("expected-fp", {}))

    from when_tta_hurts.final_test_scientific_reporting import UnsealingAuthorizationError

    def _explode():
        raise UnsealingAuthorizationError("stale authorization")

    monkeypatch.setattr(pe, "verify_unsealing_authorization", _explode)
    summary = _full_cardinality_summary()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    with pytest.raises(pe.CanonicalSummaryVerificationError):
        pe.load_and_verify_canonical_summary(path)


# ---------------------------------------------------------------------------
# build_evidence_plan: zero writes, reports exact expected counts.
# ---------------------------------------------------------------------------


def test_build_evidence_plan_reports_expected_counts_and_makes_zero_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = _synthetic_summary()
    plan = pe.build_evidence_plan(summary)
    assert plan["n_unmatched_cells"] == 3
    assert plan["n_matched_within_cell"] == 1
    assert plan["n_h1_pairs"] == 1
    assert plan["n_h2_pairs"] == 1
    assert plan["n_h3_pairs"] == 1
    assert plan["n_block_c_cells"] == 3
    assert len(plan["expected_figures"]) == 5
    assert len(plan["expected_tables"]) == 7
    assert list(tmp_path.rglob("*")) == [], "build_evidence_plan must perform zero filesystem writes"


# ---------------------------------------------------------------------------
# Isolation proof: exercising this module's extraction/table/plan code path
# must never modify the root pyproject.toml or root uv.lock.
# ---------------------------------------------------------------------------


def test_exercising_paper_evidence_never_touches_root_dependency_files():
    pyproject_before = _sha256(REPO_ROOT / "pyproject.toml")
    lock_before = _sha256(REPO_ROOT / "uv.lock")

    summary = _synthetic_summary()
    pe.build_evidence_plan(summary)
    pe.extract_unmatched_cells(summary)
    pe.render_design_classification_table()

    assert _sha256(REPO_ROOT / "pyproject.toml") == pyproject_before
    assert _sha256(REPO_ROOT / "uv.lock") == lock_before


def test_isolated_toolchain_project_files_exist_and_are_distinct_from_root():
    isolated_pyproject = REPO_ROOT / "tools" / "paper_evidence" / "pyproject.toml"
    isolated_lock = REPO_ROOT / "tools" / "paper_evidence" / "uv.lock"
    assert isolated_pyproject.exists()
    assert isolated_lock.exists()
    assert isolated_pyproject != REPO_ROOT / "pyproject.toml"
    assert _sha256(isolated_pyproject) != _sha256(REPO_ROOT / "pyproject.toml")


def test_isolated_pyproject_pins_matplotlib_exactly():
    text = (REPO_ROOT / "tools" / "paper_evidence" / "pyproject.toml").read_text()
    assert "matplotlib==" in text, "matplotlib must be exactly version-pinned, not floor-only"
