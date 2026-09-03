#!/usr/bin/env python3
"""Generate the three post-review "Extended Analyses" figures for the
TMLR manuscript, from the read-only Phase 2C / 2C.2 summary artifacts.

Outputs (PDF + PNG) to paper/tmlr/figures/:
    fig6_scaling_curve
    fig7_component_decomposition
    fig8_label_preservation

Usage:
    uv run python scripts/generate_extension_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "paper" / "tmlr" / "figures"
SEC = json.loads((REPO / "artifacts" / "secondary_analysis_expansion" / "summary.json").read_text())
COMP = json.loads((REPO / "artifacts" / "component_ablation" / "summary.json").read_text())
LAB = json.loads((REPO / "artifacts" / "label_preservation_audit" / "summary.json").read_text())

OI = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "bluish_green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}
PDF_META = {"CreationDate": None, "Creator": "", "Producer": "", "Author": ""}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spelling": False} if False else {"font.size": 9})


def save(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", format="pdf", bbox_inches="tight", metadata=PDF_META)
    fig.savefig(FIG_DIR / f"{stem}.png", format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


# ---------------------------------------------------------------------------
def fig_scaling_curve() -> None:
    Ns = SEC["view_counts"]
    rows = SEC["scaling_curve_rows"]

    def series(pred):
        means, los, his = [], [], []
        for n in Ns:
            d = np.array([r["delta_accuracy"] * 100 for r in rows if r["n_views"] == n and pred(r)])
            means.append(d.mean()); los.append(d.min()); his.append(d.max())
        return np.array(means), np.array(los), np.array(his)

    h_mean, h_lo, h_hi = series(lambda r: r["policy"] == "none" and "BLOCK_C" not in r["families"])
    p_mean, _, _ = series(lambda r: r["policy"] == "none" and "BLOCK_C" not in r["families"] and r["dataset"] == "pathmnist")
    b_mean, _, _ = series(lambda r: r["policy"] == "none" and "BLOCK_C" not in r["families"] and r["dataset"] == "bloodmnist")
    c_mean, _, _ = series(lambda r: "BLOCK_C" in r["families"])
    m_mean, _, _ = series(lambda r: r["policy"] == "matched_mixed")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.fill_between(Ns, h_lo, h_hi, color=OI["blue"], alpha=0.15,
                    label="30 unmatched cells (min–max)")
    ax.plot(Ns, h_mean, "o-", color=OI["blue"], lw=2.2, label="30 unmatched cells (mean)")
    ax.plot(Ns, p_mean, "s--", color=OI["sky_blue"], lw=1.3, ms=4, label="PathMNIST (15)")
    ax.plot(Ns, b_mean, "^--", color=OI["vermillion"], lw=1.3, ms=4, label="BloodMNIST (15)")
    ax.plot(Ns, c_mean, "D-", color=OI["orange"], lw=1.3, ms=4, label="BLOCK_C DermaMNIST/ResNet-18 (3)")
    ax.plot(Ns, m_mean, "v-", color=OI["bluish_green"], lw=1.3, ms=4, label="matched-policy (6)")
    ax.axhline(0, color=OI["black"], lw=0.8)
    ax.axvline(50, color=OI["black"], lw=0.8, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("TTA view count $N$ (log scale)")
    ax.set_ylabel("mean $\\Delta$ accuracy, TTA $-$ clean (pp)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    save(fig, "fig6_scaling_curve")


# ---------------------------------------------------------------------------
def fig_component_decomposition() -> None:
    rows = [r for r in COMP["per_cell_rows"] if r["n_views"] == 50]
    groups = [("pathmnist", "batchnorm", "Path/BN"), ("pathmnist", "groupnorm", "Path/GN"),
              ("bloodmnist", "batchnorm", "Blood/BN"), ("bloodmnist", "groupnorm", "Blood/GN")]
    fams = [("geometric", OI["blue"], "geometric-only"),
            ("intensity", OI["vermillion"], "intensity-only")]
    x = np.arange(len(groups) + 1)
    w = 0.26

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for k, (fam, col, lab) in enumerate(fams):
        vals, errs = [], []
        for ds, nm, _ in groups:
            sub = [r for r in rows if r["policy"] == fam and r["dataset"] == ds and r["norm"] == nm]
            d = np.array([r["delta_accuracy"] * 100 for r in sub])
            vals.append(d.mean()); errs.append(d.std())
        # pooled
        allsub = [r for r in rows if r["policy"] == fam]
        dall = np.array([r["delta_accuracy"] * 100 for r in allsub])
        vals.append(dall.mean()); errs.append(dall.std())
        ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=2, color=col, label=lab)
    # mixed as a marker
    mix = []
    for ds, nm, _ in groups:
        sub = [r for r in rows if r["policy"] == "geometric" and r["dataset"] == ds and r["norm"] == nm]
        mix.append(np.mean([r["mixed_delta_accuracy"] * 100 for r in sub]))
    mix.append(np.mean([r["mixed_delta_accuracy"] * 100 for r in rows if r["policy"] == "geometric"]))
    ax.plot(x, mix, "k_", ms=18, mew=2.2, label="mixed policy")

    ax.axhline(0, color=OI["black"], lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups] + ["pooled"])
    ax.set_ylabel("$\\Delta$ accuracy, TTA $-$ clean (pp), $N{=}50$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    save(fig, "fig7_component_decomposition")


# ---------------------------------------------------------------------------
def fig_label_preservation() -> None:
    order = ["bloodmnist", "dermamnist", "pathmnist"]
    labels = {"bloodmnist": "BloodMNIST", "dermamnist": "DermaMNIST", "pathmnist": "PathMNIST"}
    s2 = [LAB["human"][d]["score2_clear"] * 100 for d in order]
    s1 = [LAB["human"][d]["score1_partial"] * 100 for d in order]
    s0 = [LAB["human"][d]["score0_content_gone"] * 100 for d in order]
    auto = [LAB["automated"][d]["a_not_preserved"] * 100 for d in order]
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.barh(y, s2, color=OI["bluish_green"], label="2 = clearly present")
    ax.barh(y, s1, left=s2, color=OI["yellow"], label="1 = degraded, recognisable")
    ax.barh(y, s0, left=np.array(s2) + np.array(s1), color=OI["vermillion"], label="0 = content gone")
    ax.plot(auto, y, "kD", ms=6, label="automated not-preserved (upper bound)")
    for i, a in enumerate(auto):
        ax.annotate(f"{a:.1f}%", (a, y[i]), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([labels[d] for d in order])
    ax.set_xlabel("% of augmented views (human content-presence score, $n{=}50$/dataset)")
    ax.set_xlim(0, 108)
    ax.legend(fontsize=6.5, loc="lower left", ncol=2)
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "fig8_label_preservation")


if __name__ == "__main__":
    fig_scaling_curve()
    fig_component_decomposition()
    fig_label_preservation()
