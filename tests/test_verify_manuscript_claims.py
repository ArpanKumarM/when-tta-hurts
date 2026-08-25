"""Phase 2B.9B Part D: tests for the read-only manuscript-claim
verification script (`paper/verify_manuscript_claims.py`). The real
manuscript/references/citation-audit/evidence-package files are used
for the "the real draft passes" test; every negative-control test
below uses synthetic tmp_path-based fixtures with the script's module-
level path constants monkeypatched, so no test ever depends on
mutating the real committed manuscript to prove a check works.

None of these tests read raw predictions, datasets, checkpoints, or
sealed per-family analysis results -- the module under test cannot
reach them either (it only reads the canonical summary and the
committed paper-evidence tables/manifest via already-verified
extraction helpers).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "paper" / "verify_manuscript_claims.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_manuscript_claims", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vmc = _load_module()


# ---------------------------------------------------------------------------
# The real, committed manuscript must pass every check.
# ---------------------------------------------------------------------------


def test_real_manuscript_passes_all_checks():
    report = vmc.run_all_checks()
    assert report["status"] == "pass", report["checks"]
    assert report["total_violations"] == 0


def test_script_source_never_references_raw_data_paths():
    text = SCRIPT_PATH.read_text()
    for forbidden in ("predictions.npz", "checkpoints/", "validation_evaluation/", "final_test/"):
        assert forbidden not in text, f"verification script must never reference {forbidden!r}"


# ---------------------------------------------------------------------------
# Synthetic fixtures + monkeypatched paths for negative-control tests.
# ---------------------------------------------------------------------------


_RESULTS_SENTENCE = (
    "All 30 distinct unmatched-policy cells showed harm, ranging from -18.76 to -66.09 percentage "
    "points (Figure 1; Figure 2; Figure 3; Figure 4; Figure 5; Table 1; Table 2; Table 3; Table 4; "
    "Table 5; Table 6; Table 7)."
)

_MINIMAL_VALID_MANUSCRIPT = f"""# Title

## Abstract

Placeholder.

## Introduction

Placeholder.

## Related Work

Placeholder [@fakekey2020].

## Methods

Placeholder.

## Experimental Design

Placeholder.

## Statistical Analysis

Placeholder.

## Results

{_RESULTS_SENTENCE}

## Discussion

Placeholder.

## Limitations

Placeholder mentioning three training seeds, an accidental final-test
access incident, a shared-aggregation-contract correction, and stating
no final-test scientific result was inspected by a human before
unsealing.

## Reproducibility and Audit Trail

Placeholder. No H4 claim is made anywhere. No population-level
inference is made anywhere.

## Conclusion

Placeholder.

## References

Placeholder.
"""

_MINIMAL_BIB = """@article{fakekey2020,
  title = {Fake Title},
  author = {Fake, Author},
  journal = {arXiv preprint arXiv:0000.00000},
  year = {2020},
}
"""

_MINIMAL_AUDIT = """| BibTeX key | Source | What was verified | Status |
|---|---|---|---|
| `fakekey2020` | https://arxiv.org/abs/0000.00000 | Everything | Verified |
"""


@pytest.fixture
def patched_paths(tmp_path, monkeypatch):
    manuscript_path = tmp_path / "manuscript.md"
    references_path = tmp_path / "references.bib"
    audit_path = tmp_path / "citation_audit.md"
    manuscript_path.write_text(_MINIMAL_VALID_MANUSCRIPT)
    references_path.write_text(_MINIMAL_BIB)
    audit_path.write_text(_MINIMAL_AUDIT)

    monkeypatch.setattr(vmc, "MANUSCRIPT_PATH", manuscript_path)
    monkeypatch.setattr(vmc, "REFERENCES_PATH", references_path)
    monkeypatch.setattr(vmc, "CITATION_AUDIT_PATH", audit_path)
    return manuscript_path, references_path, audit_path


def _run_with_manuscript_text(patched_paths, text: str) -> dict:
    manuscript_path, _, _ = patched_paths
    manuscript_path.write_text(text)
    return vmc.run_all_checks()


def _patch_canonical_extraction(monkeypatch):
    """Synthetic canonical-evidence stand-in so numeric-claim checks can
    be exercised without touching the real canonical summary."""
    monkeypatch.setattr(vmc, "build_known_good_decimal_numbers", lambda: {"18.76", "66.09", "1.6"})


def test_missing_required_wording_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace("All 30 distinct unmatched-policy cells", "All 30 base cells")
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("30 distinct unmatched-policy" in v for v in report["checks"]["required_wording"])


def test_double_counted_cell_figure_without_negation_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(
        "## Discussion\n\nPlaceholder.",
        "## Discussion\n\nThis study found 54 distinct cells showing harm.",
    )
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("54 distinct" in v for v in report["checks"]["forbidden_phrases"])


def test_double_counted_cell_figure_with_negation_is_not_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(
        "## Discussion\n\nPlaceholder.",
        "## Discussion\n\nThis study never reports the double-counted, forbidden '54 distinct cells' figure.",
    )
    report = _run_with_manuscript_text(patched_paths, text)
    assert not any("54 distinct" in v for v in report["checks"]["forbidden_phrases"])


def test_unhedged_secondary_significance_claim_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(
        "## Discussion\n\nPlaceholder.",
        "## Discussion\n\nThe secondary normalization comparison was statistically significant.",
    )
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("significan" in v.lower() for v in report["checks"]["forbidden_phrases"])


def test_h4_result_claim_without_negation_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(
        "## Discussion\n\nPlaceholder.",
        "## Discussion\n\nH4 showed a strong mitigation effect in this study.",
    )
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("h4" in v.lower() for v in report["checks"]["forbidden_phrases"])


def test_clinical_claim_without_negation_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(
        "## Discussion\n\nPlaceholder.",
        "## Discussion\n\nThese models are ready for clinical use in practice.",
    )
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("clinical use" in v for v in report["checks"]["forbidden_phrases"])


def test_missing_figure_reference_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace("Figure 4; ", "")
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("Figure 4" in v for v in report["checks"]["figures_and_tables_referenced"])


def test_missing_table_reference_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace("Table 7)", ")")
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("Table 7" in v for v in report["checks"]["figures_and_tables_referenced"])


def test_citation_key_missing_from_bib_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace("[@fakekey2020]", "[@fakekey2020; @undefinedkey1999]")
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("undefinedkey1999" in v for v in report["checks"]["citations"])


def test_citation_key_missing_from_audit_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    _, references_path, _ = patched_paths
    references_path.write_text(
        _MINIMAL_BIB
        + "\n@article{unauditedkey2021,\n  title={X},\n  author={Y},\n  journal={Z},\n  year={2021},\n}\n"
    )
    text = _MINIMAL_VALID_MANUSCRIPT.replace("[@fakekey2020]", "[@fakekey2020; @unauditedkey2021]")
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("unauditedkey2021" in v for v in report["checks"]["citations"])


def test_missing_required_disclosure_is_flagged(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(
        "Placeholder mentioning three training seeds, an accidental final-test\n"
        "access incident, a shared-aggregation-contract correction, and stating\n"
        "no final-test scientific result was inspected by a human before\n"
        "unsealing.",
        "Placeholder.",
    )
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert len(report["checks"]["required_disclosures"]) >= 3


def test_numeric_claim_not_in_evidence_package_is_flagged(patched_paths, monkeypatch):
    monkeypatch.setattr(vmc, "build_known_good_decimal_numbers", lambda: {"1.6"})
    text = _MINIMAL_VALID_MANUSCRIPT.replace("-18.76", "-99.99")
    report = _run_with_manuscript_text(patched_paths, text)
    assert report["status"] == "fail"
    assert any("99.99" in v for v in report["checks"]["numeric_claims_in_results"])


def test_missing_manuscript_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vmc, "MANUSCRIPT_PATH", tmp_path / "does_not_exist.md")
    with pytest.raises(vmc.ManuscriptVerificationError):
        vmc.run_all_checks()


def test_empty_results_section_raises(patched_paths, monkeypatch):
    _patch_canonical_extraction(monkeypatch)
    text = _MINIMAL_VALID_MANUSCRIPT.replace(f"## Results\n\n{_RESULTS_SENTENCE}", "## Results\n")
    manuscript_path, _, _ = patched_paths
    manuscript_path.write_text(text)
    with pytest.raises(vmc.ManuscriptVerificationError):
        vmc.run_all_checks()


def test_whitespace_normalization_reunites_line_wrapped_phrase():
    wrapped = "This sentence mentions Figure\n4 across a line wrap.\n\nSecond paragraph."
    normalized = vmc.normalize_whitespace(wrapped)
    assert "Figure 4" in normalized
