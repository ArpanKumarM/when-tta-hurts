"""Phase 2B.9A: rendering-determinism tests for the isolated paper-
evidence plotting toolchain. These tests exercise ONLY the matplotlib-
dependent figure-rendering functions in `when_tta_hurts.paper_evidence`,
using fully synthetic fixtures, and run exclusively inside the isolated
`tools/paper_evidence` environment (never in the root environment, which
has no matplotlib installed).
"""

from __future__ import annotations

import hashlib

import when_tta_hurts.paper_evidence as pe


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_unmatched_rows():
    return [
        {
            "run_id": f"A-bloodmnist-28px-batchnorm-policy-none-s{i}",
            "dataset": "bloodmnist",
            "resolution": "28",
            "normalization": "batchnorm",
            "seed": str(i),
            "delta_accuracy": -0.03 - 0.001 * i,
            "ci_low": -0.05 - 0.001 * i,
            "ci_high": -0.01 - 0.001 * i,
        }
        for i in range(3)
    ]


def _synthetic_pairs(hypothesis: str, dataset: str = "bloodmnist"):
    return [
        {
            "pair_id": f"{hypothesis}-pair-{i}",
            "condition_a": {"run_id": f"A-{dataset}-28px-batchnorm-policy-none-s{i}"},
            "condition_b": {"run_id": f"A-{dataset}-28px-groupnorm-policy-none-s{i}"},
            "bootstrap": {"did": 0.01 * i, "ci_low": 0.01 * i - 0.02, "ci_high": 0.01 * i + 0.02},
        }
        for i in range(2)
    ]


def _synthetic_block_c_rows():
    return [
        {
            "run_id": f"C-dermamnist-28px-resnet18-batchnorm-policy-none-s{i}",
            "seed": str(i),
            "delta_accuracy": -0.005 + 0.001 * i,
            "ci_low": -0.02 + 0.001 * i,
            "ci_high": 0.01 + 0.001 * i,
        }
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# Matplotlib imports and renders successfully inside this environment.
# ---------------------------------------------------------------------------


def test_matplotlib_imports_and_uses_agg_backend():
    import matplotlib

    matplotlib.use("Agg")
    assert matplotlib.get_backend().lower() == "agg"


def test_okabe_ito_palette_is_frozen_and_colorblind_safe():
    assert pe.OKABE_ITO == {
        "black": "#000000",
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
    }


# ---------------------------------------------------------------------------
# Each figure renders successfully to both PDF and PNG.
# ---------------------------------------------------------------------------


def test_figure_1_renders_pdf_and_png(tmp_path):
    pdf_path, png_path = pe.render_figure_1(_synthetic_unmatched_rows(), tmp_path)
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.read_bytes()[:4] == b"%PDF"
    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_figure_2_renders_pdf_and_png(tmp_path):
    pdf_path, png_path = pe.render_figure_2(_synthetic_pairs("H3"), tmp_path)
    assert pdf_path.exists() and png_path.exists()


def test_figure_3_renders_pdf_and_png(tmp_path):
    pairs = _synthetic_pairs("H1", "bloodmnist") + _synthetic_pairs("H1", "pathmnist")
    pdf_path, png_path = pe.render_figure_3(pairs, tmp_path)
    assert pdf_path.exists() and png_path.exists()


def test_figure_4_renders_pdf_and_png(tmp_path):
    pdf_path, png_path = pe.render_figure_4(_synthetic_pairs("H2"), tmp_path)
    assert pdf_path.exists() and png_path.exists()


def test_figure_5_renders_pdf_and_png(tmp_path):
    pdf_path, png_path = pe.render_figure_5(_synthetic_block_c_rows(), tmp_path)
    assert pdf_path.exists() and png_path.exists()


# ---------------------------------------------------------------------------
# Determinism: two independent renders of the same synthetic input must be
# byte-identical (PDF metadata normalization + PNG default determinism).
# ---------------------------------------------------------------------------


def test_figure_1_pdf_is_byte_identical_across_repeated_renders(tmp_path):
    rows = _synthetic_unmatched_rows()
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    pdf_a, png_a = pe.render_figure_1(rows, tmp_path / "run_a")
    pdf_b, png_b = pe.render_figure_1(rows, tmp_path / "run_b")
    assert _sha256_bytes(pdf_a.read_bytes()) == _sha256_bytes(pdf_b.read_bytes())


def test_figure_1_png_is_byte_identical_across_repeated_renders(tmp_path):
    rows = _synthetic_unmatched_rows()
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    pdf_a, png_a = pe.render_figure_1(rows, tmp_path / "run_a")
    pdf_b, png_b = pe.render_figure_1(rows, tmp_path / "run_b")
    assert _sha256_bytes(png_a.read_bytes()) == _sha256_bytes(png_b.read_bytes())


def test_figure_5_pdf_and_png_byte_identical_across_repeated_renders(tmp_path):
    rows = _synthetic_block_c_rows()
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    pdf_a, png_a = pe.render_figure_5(rows, tmp_path / "run_a")
    pdf_b, png_b = pe.render_figure_5(rows, tmp_path / "run_b")
    assert _sha256_bytes(pdf_a.read_bytes()) == _sha256_bytes(pdf_b.read_bytes())
    assert _sha256_bytes(png_a.read_bytes()) == _sha256_bytes(png_b.read_bytes())


# ---------------------------------------------------------------------------
# generate_all_evidence: end-to-end synthetic run produces exactly the
# expected 10 figure files + 7 table files + one manifest, and reruns are
# idempotent (byte-identical outputs, same manifest content modulo
# nothing -- every field is a pure function of the same input).
# ---------------------------------------------------------------------------


def _full_synthetic_summary():
    """A minimal but schema-complete synthetic summary, distinct from and
    much smaller than the real 39-cell canonical summary -- this test
    only proves generate_all_evidence's plumbing (correct output count,
    idempotent manifest), never a real scientific value."""

    def _cell(run_id, delta):
        return {
            "run_id": run_id,
            "bootstrap": {
                "delta_accuracy": delta,
                "ci_low": delta - 0.02,
                "ci_high": delta + 0.02,
                "ci_level": 0.95,
            },
            "mcnemar": {"p_value": 1e-10},
            "n_samples": 100,
        }

    def _pair(pair_id, run_a, run_b, did):
        return {
            "pair_id": pair_id,
            "condition_a": {"run_id": run_a},
            "condition_b": {"run_id": run_b},
            "bootstrap": {"did": did, "ci_low": did - 0.02, "ci_high": did + 0.02},
        }

    h1_cells = [_cell(f"A-bloodmnist-28px-batchnorm-policy-none-s{i}", -0.03) for i in range(2)]
    h3_cells = h1_cells[:1] + [_cell("D-bloodmnist-28px-batchnorm-policy-matched_mixed-s0", -0.001)]
    block_c_cells = [_cell(f"C-dermamnist-28px-resnet18-batchnorm-policy-none-s{i}", 0.001) for i in range(3)]

    def _mult(cells):
        p = [c["mcnemar"]["p_value"] for c in cells]
        return {"raw_p_values": p, "corrected_p_values": p}

    return {
        "reporting_fingerprint": "fake-fp",
        "preregistered": {
            "H1": {"analysis_id": "a1", "attempt": 1, "cells": h1_cells, "multiplicity": _mult(h1_cells)},
            "H2": {"analysis_id": "a2", "attempt": 1, "cells": h1_cells, "multiplicity": _mult(h1_cells)},
            "H3": {"analysis_id": "a3", "attempt": 1, "cells": h3_cells, "multiplicity": _mult(h3_cells)},
            "BLOCK_C": {
                "analysis_id": "a4",
                "attempt": 1,
                "cells": block_c_cells,
                "multiplicity": _mult(block_c_cells),
            },
        },
        "secondary_cross_condition": {
            "H1": {
                "analysis_id": "c1",
                "attempt": 1,
                "pairs": [
                    _pair(
                        "H1-pair-0",
                        "A-bloodmnist-28px-batchnorm-policy-none-s0",
                        "A-bloodmnist-28px-groupnorm-policy-none-s0",
                        0.01,
                    )
                ],
            },
            "H2": {
                "analysis_id": "c2",
                "attempt": 1,
                "pairs": [
                    _pair(
                        "H2-pair-0",
                        "A-bloodmnist-28px-batchnorm-policy-none-s0",
                        "B-pathmnist-64px-batchnorm-policy-none-s0",
                        0.02,
                    )
                ],
            },
            "H3": {
                "analysis_id": "c3",
                "attempt": 1,
                "pairs": [
                    _pair(
                        "H3-pair-0",
                        "A-bloodmnist-28px-batchnorm-policy-none-s0",
                        "D-bloodmnist-28px-batchnorm-policy-matched_mixed-s0",
                        0.03,
                    )
                ],
            },
        },
    }


def _patched_canonical_summary(monkeypatch, summary):
    monkeypatch.setattr(pe, "load_and_verify_canonical_summary", lambda *_a, **_kw: summary)
    monkeypatch.setattr(pe, "verify_unsealing_authorization", lambda *_a, **_kw: {"status": "approved"})


def test_generate_all_evidence_produces_full_expected_output_set(tmp_path, monkeypatch):
    import json

    summary = _full_synthetic_summary()
    _patched_canonical_summary(monkeypatch, summary)
    monkeypatch.setattr(pe, "hash_file", lambda path: "fake-hash")

    fake_summary_path = tmp_path / "summary.json"
    fake_summary_path.write_text(json.dumps(summary))
    fake_auth_path = tmp_path / "auth.json"
    fake_auth_path.write_text("{}")
    monkeypatch.setattr(pe, "FINAL_TEST_UNSEALING_AUTHORIZATION_PATH", fake_auth_path)

    output_root = tmp_path / "artifacts" / "paper_evidence"
    manifest = pe.generate_all_evidence(summary_path=fake_summary_path, output_root=output_root)

    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    pdf_files = sorted(figures_dir.glob("*.pdf"))
    png_files = sorted(figures_dir.glob("*.png"))
    table_files = sorted(tables_dir.glob("*.md"))
    assert len(pdf_files) == 5
    assert len(png_files) == 5
    assert len(table_files) == 7
    assert (output_root / "paper_evidence_manifest.json").exists()
    assert len(manifest["outputs"]) == 5 * 2 + 7


def test_generate_all_evidence_is_idempotent_byte_identical_rerun(tmp_path, monkeypatch):
    import json

    summary = _full_synthetic_summary()
    _patched_canonical_summary(monkeypatch, summary)
    monkeypatch.setattr(pe, "hash_file", lambda path: "fake-hash")

    fake_summary_path = tmp_path / "summary.json"
    fake_summary_path.write_text(json.dumps(summary))
    fake_auth_path = tmp_path / "auth.json"
    fake_auth_path.write_text("{}")
    monkeypatch.setattr(pe, "FINAL_TEST_UNSEALING_AUTHORIZATION_PATH", fake_auth_path)

    output_root_a = tmp_path / "run_a" / "artifacts" / "paper_evidence"
    output_root_b = tmp_path / "run_b" / "artifacts" / "paper_evidence"
    pe.generate_all_evidence(summary_path=fake_summary_path, output_root=output_root_a)
    pe.generate_all_evidence(summary_path=fake_summary_path, output_root=output_root_b)

    for name in ("figure_1_unmatched_policy_forest.pdf", "figure_5_block_c_positive_control.png"):
        a_bytes = (output_root_a / "figures" / name).read_bytes()
        b_bytes = (output_root_b / "figures" / name).read_bytes()
        assert _sha256_bytes(a_bytes) == _sha256_bytes(b_bytes)

    for name in ("table_2_unmatched_policy.md", "table_6_block_c.md"):
        a_bytes = (output_root_a / "tables" / name).read_bytes()
        b_bytes = (output_root_b / "tables" / name).read_bytes()
        assert a_bytes == b_bytes
