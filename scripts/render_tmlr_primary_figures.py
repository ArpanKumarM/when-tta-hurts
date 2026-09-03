#!/usr/bin/env python3
"""Re-render the 5 primary evidence figures WITHOUT their baked-in title
bar and bottom caption band (both redundant with the LaTeX \\caption) into
paper/tmlr/figures/. Reuses the sealed evidence-package plotting code in
when_tta_hurts.paper_evidence unchanged -- it only suppresses
Axes.set_title / Figure.text / Figure.suptitle for the duration of the
render calls. Does NOT touch artifacts/paper_evidence/.

Usage:
    uv run --with matplotlib --with numpy python scripts/render_tmlr_primary_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes  # noqa: E402
import matplotlib.figure  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import when_tta_hurts.paper_evidence as pe  # noqa: E402
from when_tta_hurts.paper_evidence import (  # noqa: E402
    extract_block_c,
    extract_cross_condition_pairs,
    extract_unmatched_cells,
    load_and_verify_canonical_summary,
)

OUT = REPO / "paper" / "tmlr" / "figures"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_and_verify_canonical_summary()
    jobs = [
        ("figure_1_unmatched_policy_forest", pe.render_figure_1, extract_unmatched_cells(summary)),
        ("figure_2_matched_policy_mitigation", pe.render_figure_2, extract_cross_condition_pairs(summary, "H3")),
        ("figure_3_normalization_heterogeneity", pe.render_figure_3, extract_cross_condition_pairs(summary, "H1")),
        ("figure_4_resolution_comparison", pe.render_figure_4, extract_cross_condition_pairs(summary, "H2")),
        ("figure_5_block_c_positive_control", pe.render_figure_5, extract_block_c(summary)),
    ]

    def _noop(*_a, **_k):
        return None

    saved = (
        matplotlib.axes.Axes.set_title,
        matplotlib.figure.Figure.text,
        matplotlib.figure.Figure.suptitle,
    )
    matplotlib.axes.Axes.set_title = _noop
    matplotlib.figure.Figure.text = _noop
    matplotlib.figure.Figure.suptitle = _noop
    try:
        for name, fn, data in jobs:
            fn(data, OUT)
            print(f"rendered {name} (title/caption suppressed)")
    finally:
        (
            matplotlib.axes.Axes.set_title,
            matplotlib.figure.Figure.text,
            matplotlib.figure.Figure.suptitle,
        ) = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
