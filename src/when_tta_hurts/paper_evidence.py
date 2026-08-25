"""Phase 2B.9A: deterministic paper-evidence generation from the sealed,
canonical generation-2 final-test scientific summary.

This module is downstream PRESENTATION only: it computes zero new
statistics, resamples nothing, and reads exactly one input file --
`artifacts/final_test_scientific_summary.json` -- as already-verified,
already-sealed fact. It never reads raw predictions, validation
artifacts, sealed per-family result JSONs, test datasets, or checkpoints.

Plotting (matplotlib) is imported only inside functions that actually
render a figure, and only when this module is invoked from the isolated
`tools/paper_evidence` environment -- `plan` mode and all extraction/
table functions never import matplotlib and run fine in the root
environment (which does not have matplotlib installed).

Every extraction function is pure (same input dict -> same output
list), sorted deterministically, and never omits a planned cell/pair.
Overlapping family membership (H1's 24 cells are a subset of H2's 30;
H3's 6-cell unmatched arm is a subset of H1/H2) is never allowed to
inflate a unique-cell count -- `extract_unmatched_cells` deduplicates by
run_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.final_test_scientific_reporting import (
    FINAL_TEST_UNSEALING_AUTHORIZATION_PATH,
    SCIENTIFIC_SUMMARY_PATH,
    compute_final_test_reporting_fingerprint,
    verify_unsealing_authorization,
)

PAPER_EVIDENCE_ROOT = Path("artifacts/paper_evidence")
FIGURES_DIR = PAPER_EVIDENCE_ROOT / "figures"
TABLES_DIR = PAPER_EVIDENCE_ROOT / "tables"
MANIFEST_PATH = PAPER_EVIDENCE_ROOT / "paper_evidence_manifest.json"

EXPECTED_SCHEMA_VERSION = "phase2b.8a-v1"

# Every file whose content could change a rendered figure/table's layout,
# extraction logic, or verification -- deliberately disjoint from every
# existing scientific-computation/reporting fingerprint manifest. Never
# added to, and never includes, ANALYSIS_FINGERPRINT_MANIFEST,
# CROSS_CONDITION_ADDENDUM_MANIFEST, FINAL_TEST_RUNNER_MANIFEST,
# FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST, or FINAL_TEST_REPORTING_MANIFEST.
PAPER_EVIDENCE_FINGERPRINT_MANIFEST: tuple[str, ...] = (
    "tools/paper_evidence/pyproject.toml",
    "tools/paper_evidence/uv.lock",
    "src/when_tta_hurts/paper_evidence.py",
    "scripts/generate_paper_evidence.py",
    "docs/phase2b_paper_evidence_toolchain_freeze.md",
    "docs/phase2b_paper_evidence_package_freeze.md",
)


class PaperEvidenceFingerprintError(RuntimeError):
    """Raised when a file listed in PAPER_EVIDENCE_FINGERPRINT_MANIFEST is
    missing. Fails closed -- never computes a partial fingerprint."""


def compute_paper_evidence_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = PAPER_EVIDENCE_FINGERPRINT_MANIFEST,
) -> tuple[str, dict[str, str]]:
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise PaperEvidenceFingerprintError(
                f"Paper-evidence fingerprint manifest file missing: {rel_path}. Refusing to compute a "
                f"partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


class CanonicalSummaryVerificationError(RuntimeError):
    """Raised when the canonical summary is missing, malformed, hash-
    tampered, schema-incompatible, or bound to a stale reporting
    fingerprint/authorization. Fails closed before any extraction."""


def load_and_verify_canonical_summary(
    summary_path: str | Path = SCIENTIFIC_SUMMARY_PATH,
) -> dict[str, Any]:
    """Loads and verifies the canonical generation-2 summary. Checks (in
    order): file exists; valid JSON; schema_version matches; the
    embedded reporting_fingerprint matches a FRESH recomputation; the
    generation-2 unsealing authorization is approved and current; and
    the frozen family/pair cardinalities (H1=24, H2=30, H3=12, BLOCK_C=3,
    cross-H1=12, cross-H2=12, cross-H3=6) are exactly present. Raises
    CanonicalSummaryVerificationError on any failure -- never proceeds
    with a partial or unverified summary."""
    path = Path(summary_path)
    if not path.exists():
        raise CanonicalSummaryVerificationError(f"Canonical summary {path} does not exist.")
    try:
        summary = json.loads(path.read_text())
    except Exception as e:
        raise CanonicalSummaryVerificationError(f"Canonical summary {path} is malformed JSON.") from e

    if summary.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise CanonicalSummaryVerificationError(
            f"Canonical summary schema_version {summary.get('schema_version')!r} does not match "
            f"expected {EXPECTED_SCHEMA_VERSION!r}."
        )

    current_reporting_fp, _ = compute_final_test_reporting_fingerprint()
    if summary.get("reporting_fingerprint") != current_reporting_fp:
        raise CanonicalSummaryVerificationError(
            "Canonical summary's reporting_fingerprint does not match the current reporting "
            "fingerprint -- summary is stale or tampered."
        )

    try:
        auth = verify_unsealing_authorization()
    except Exception as e:
        raise CanonicalSummaryVerificationError(f"Unsealing authorization did not verify: {e}") from e
    if auth.get("final_test_reporting_fingerprint") != current_reporting_fp:
        raise CanonicalSummaryVerificationError(
            "Unsealing authorization's reporting fingerprint does not match the current summary."
        )

    expected_family_counts = {"H1": 24, "H2": 30, "H3": 12, "BLOCK_C": 3}
    for family, expected in expected_family_counts.items():
        entry = summary.get("preregistered", {}).get(family)
        if entry is None or entry.get("n_cells") != expected or len(entry.get("cells", [])) != expected:
            raise CanonicalSummaryVerificationError(
                f"Canonical summary's preregistered {family!r} does not have exactly {expected} cells."
            )

    expected_pair_counts = {"H1": 12, "H2": 12, "H3": 6}
    for hyp, expected in expected_pair_counts.items():
        entry = summary.get("secondary_cross_condition", {}).get(hyp)
        if entry is None or entry.get("n_pairs") != expected or len(entry.get("pairs", [])) != expected:
            raise CanonicalSummaryVerificationError(
                f"Canonical summary's secondary_cross_condition {hyp!r} does not have exactly "
                f"{expected} pairs."
            )

    return summary


# ---------------------------------------------------------------------------
# Deterministic, pure extraction. Every function returns a list already
# sorted in a stable, reproducible order.
# ---------------------------------------------------------------------------


def _parse_run_id(run_id: str) -> dict[str, str]:
    """Mechanically parses the frozen run_id naming convention:
    '<block>-<dataset>-<resolution>px-[<model>-]<normalization>-policy-<policy>-s<seed>'.
    Never guesses a field -- any run_id not matching the frozen shape
    raises, rather than silently returning a partial parse."""
    parts = run_id.split("-")
    if "policy" not in parts or not run_id.split("-s")[-1].isdigit():
        raise CanonicalSummaryVerificationError(f"run_id {run_id!r} does not match the frozen naming shape.")
    block = parts[0]
    dataset = parts[1]
    resolution = parts[2].removesuffix("px")
    policy_idx = parts.index("policy")
    normalization_parts = parts[3:policy_idx]
    normalization = "-".join(normalization_parts)
    policy = parts[policy_idx + 1]
    seed = run_id.rsplit("-s", 1)[-1]
    return {
        "block": block,
        "dataset": dataset,
        "resolution": resolution,
        "normalization": normalization,
        "policy": policy,
        "seed": seed,
    }


def _cell_row(family: str, cell: dict[str, Any], index: int, multiplicity: dict[str, Any]) -> dict[str, Any]:
    """Builds one flat row from a preregistered cell dict plus its
    family's multiplicity arrays (aligned by index, exactly as persisted
    -- reused verbatim, never recomputed)."""
    identity = _parse_run_id(cell["run_id"])
    return {
        "run_id": cell["run_id"],
        "source_family": family,
        **identity,
        "delta_accuracy": cell["bootstrap"]["delta_accuracy"],
        "ci_low": cell["bootstrap"]["ci_low"],
        "ci_high": cell["bootstrap"]["ci_high"],
        "ci_level": cell["bootstrap"]["ci_level"],
        "n_samples": cell["n_samples"],
        "mcnemar_p": multiplicity["raw_p_values"][index],
        "bh_adjusted_p": multiplicity["corrected_p_values"][index],
    }


def extract_unmatched_cells(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The 30 distinct unmatched-policy (policy=none) base cells: H1's 24
    cells UNION H2's 6 Block-D-only cells UNION H3's 6-cell unmatched arm
    -- deduplicated by run_id so overlapping family membership (H1 subset
    of H2; H3's unmatched arm subset of H1/H2) can never inflate the
    count.

    Each row carries its parsed identity, bootstrap delta_accuracy/CI,
    and the single raw McNemar p-value (verified family-invariant for a
    shared cell, since it is the same underlying computation). The
    Benjamini-Hochberg-ADJUSTED p-value is NOT family-invariant -- a cell
    that belongs to multiple hypothesis families (e.g. H1 and H2) has a
    DIFFERENT adjusted value in each family's own correction set, per
    the frozen SAP's per-family BH-FDR design
    (docs/statistical_analysis_plan.md sec.'Multiple comparisons'). This
    function therefore exposes `bh_adjusted_p_by_family`, a dict keyed by
    EVERY family the cell is a member of, rather than silently collapsing
    to one value -- never implying a single canonical adjusted p-value
    for an overlapping cell."""
    by_run_id: dict[str, dict[str, Any]] = {}
    for family in ("H1", "H2", "H3"):
        cells = summary["preregistered"][family]["cells"]
        multiplicity = summary["preregistered"][family]["multiplicity"]
        for index, cell in enumerate(cells):
            run_id = cell["run_id"]
            identity = _parse_run_id(run_id)
            if identity["policy"] != "none":
                continue
            if run_id not in by_run_id:
                row = _cell_row(family, cell, index, multiplicity)
                row["member_families"] = [family]
                row["bh_adjusted_p_by_family"] = {family: row.pop("bh_adjusted_p")}
                by_run_id[run_id] = row
            else:
                existing = by_run_id[run_id]
                existing["member_families"].append(family)
                existing["bh_adjusted_p_by_family"][family] = multiplicity["corrected_p_values"][index]
                assert existing["mcnemar_p"] == multiplicity["raw_p_values"][index], (
                    f"raw McNemar p-value for {run_id!r} differs between families -- this should be "
                    f"structurally impossible for the same underlying computation."
                )
    rows = sorted(
        by_run_id.values(),
        key=lambda r: (r["dataset"], int(r["resolution"]), r["normalization"], int(r["seed"])),
    )
    return rows


def extract_matched_within_cell(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The 6 H3 matched-policy (policy=matched_mixed) within-cell rows.
    Each matched cell is a member of H3 only (never H1/H2), so its
    BH-adjusted p-value is unambiguous."""
    cells = summary["preregistered"]["H3"]["cells"]
    multiplicity = summary["preregistered"]["H3"]["multiplicity"]
    rows = []
    for index, cell in enumerate(cells):
        identity = _parse_run_id(cell["run_id"])
        if identity["policy"] != "matched_mixed":
            continue
        row = _cell_row("H3", cell, index, multiplicity)
        row["member_families"] = ["H3"]
        rows.append(row)
    return sorted(rows, key=lambda r: (r["dataset"], int(r["seed"])))


def extract_cross_condition_pairs(summary: dict[str, Any], hypothesis: str) -> list[dict[str, Any]]:
    """All pairs for one secondary cross-condition hypothesis (H1, H2, or
    H3), sorted deterministically by pair_id. `entry["pairs"]` is already
    the list of full per-pair detail dicts (pair_id, condition_a,
    condition_b, bootstrap, n_samples) -- reused verbatim, never
    recomputed."""
    entry = summary["secondary_cross_condition"][hypothesis]
    return sorted(entry["pairs"], key=lambda p: p["pair_id"])


def extract_block_c(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """The 3 BLOCK_C (DermaMNIST/ResNet-18) seed rows."""
    cells = summary["preregistered"]["BLOCK_C"]["cells"]
    multiplicity = summary["preregistered"]["BLOCK_C"]["multiplicity"]
    rows = [_cell_row("BLOCK_C", cell, index, multiplicity) for index, cell in enumerate(cells)]
    return sorted(rows, key=lambda r: int(r["seed"]))


# ---------------------------------------------------------------------------
# Deterministic Markdown table rendering. Pure functions: same input ->
# same output string. No new statistic is computed anywhere below --
# every number is copied verbatim from the already-verified canonical
# summary, with only a pure x100 percentage-point display conversion.
# ---------------------------------------------------------------------------


def _pp(value: float) -> str:
    """Fractional delta -> percentage-point display string. Pure unit
    conversion for display only, never a new statistic."""
    return f"{value * 100:.2f}"


def render_design_classification_table() -> str:
    lines = [
        "# Table 1 — Experimental-Design and Evidence-Classification",
        "",
        "| Evidence tier | Source | Cells/pairs | Confirmatory? |",
        "|---|---|---|---|",
        "| Preregistered within-cell | H1/H2/H3/BLOCK_C (`preregistered.*`) "  # noqa: E501
        "| 39 unique cells | Yes -- clean-vs-TTA, within one trained model |",
        "| Secondary fixed-model comparison | Cross-condition H1/H2/H3 (`secondary_cross_condition.*`) "  # noqa: E501
        "| 30 pairs (12+12+6) | No -- post-validation/pre-test-specified, never preregistered |",
        "| Descriptive summary | `descriptive_summaries.preregistered_seed_level` "
        "| 13 dataset/resolution/normalization groups "  # noqa: E501
        "| No -- non-inferential, carries no p-value/CI of its own |",
        "| Unsupported/forbidden | H4; pooled/model-population verdicts; secondary significance labels "  # noqa: E501
        "| N/A | Never permitted anywhere in this package |",
        "",
    ]
    return "\n".join(lines)


def render_unmatched_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 2 — Complete 30-Cell Unmatched-Policy Table",
        "",
        "Preregistered within-cell evidence. Every distinct unmatched-policy "
        "cell appears exactly once, regardless of how many hypothesis "
        "families (listed in `member_families`) it belongs to.",
        "",
        "| run_id | dataset | resolution | normalization | seed | Δ accuracy (pp) | 95% CI (pp) "  # noqa: E501
        "| McNemar p | member families | BH-adjusted p (per family) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]"
        bh = "; ".join(f"{fam}={p:.3g}" for fam, p in sorted(r["bh_adjusted_p_by_family"].items()))
        lines.append(
            f"| {r['run_id']} | {r['dataset']} | {r['resolution']}px | {r['normalization']} | "
            f"{r['seed']} | {_pp(r['delta_accuracy'])} | {ci} | {r['mcnemar_p']:.3g} | "
            f"{', '.join(r['member_families'])} | {bh} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_matched_table(within_cell: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 3 — Matched-Policy Within-Cell and Secondary DiD Table",
        "",
        "## Within-cell (preregistered, H3 matched arm)",
        "",
        "| run_id | dataset | seed | Δ accuracy (pp) | 95% CI (pp) | McNemar p | BH-adjusted p |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in within_cell:
        ci = f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]"
        lines.append(
            f"| {r['run_id']} | {r['dataset']} | {r['seed']} | {_pp(r['delta_accuracy'])} | {ci} | "
            f"{r['mcnemar_p']:.3g} | {r['bh_adjusted_p']:.3g} |"
        )
    lines.append("")
    lines.append(
        "## Secondary (post-validation/pre-test-specified, fixed-model DiD -- not a preregistered "
        "cross-condition test)"
    )
    lines.append("")
    lines.append("| pair_id | condition A | condition B | DiD (pp) | 95% CI (pp) |")
    lines.append("|---|---|---|---|---|")
    for p in pairs:
        ci = f"[{_pp(p['bootstrap']['ci_low'])}, {_pp(p['bootstrap']['ci_high'])}]"
        lines.append(
            f"| {p['pair_id']} | {p['condition_a']['run_id']} | {p['condition_b']['run_id']} | "
            f"{_pp(p['bootstrap']['did'])} | {ci} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_cross_condition_table(title: str, pairs: list[dict[str, Any]]) -> str:
    lines = [
        f"# {title}",
        "",
        "Secondary, fixed-model, post-validation/pre-test-specified "
        "difference-in-differences estimates. No pooled p-value, alpha "
        "threshold, or significance decision is computed or implied.",
        "",
        "| pair_id | condition A | condition B | DiD (pp) | 95% CI (pp) |",
        "|---|---|---|---|---|",
    ]
    for p in pairs:
        ci = f"[{_pp(p['bootstrap']['ci_low'])}, {_pp(p['bootstrap']['ci_high'])}]"
        lines.append(
            f"| {p['pair_id']} | {p['condition_a']['run_id']} | {p['condition_b']['run_id']} | "
            f"{_pp(p['bootstrap']['did'])} | {ci} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_block_c_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Table 6 — Complete Three-Seed BLOCK_C Table",
        "",
        "| run_id | seed | Δ accuracy (pp) | 95% CI (pp) | McNemar p | BH-adjusted p |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ci = f"[{_pp(r['ci_low'])}, {_pp(r['ci_high'])}]"
        lines.append(
            f"| {r['run_id']} | {r['seed']} | {_pp(r['delta_accuracy'])} | {ci} | "
            f"{r['mcnemar_p']:.3g} | {r['bh_adjusted_p']:.3g} |"
        )
    lines.append("")
    lines.append(
        "External reference (descriptive only, not an acceptance threshold): the source paper's own "
        "reported TTA improvement at N=50 views was approximately +1.6 percentage points "
        "(docs/phase2b_validation_evaluation_block_c_audit.md sec.7). This project's frozen "
        "operationalization did not reproduce that expected positive improvement in any of the three "
        "seeds above."
    )
    lines.append("")
    return "\n".join(lines)


def render_claim_adjudication_table() -> str:
    lines = [
        "# Table 7 — Claim Adjudication",
        "",
        "| Claim | Evidence tier | Status |",
        "|---|---|---|",
        "| Naive TTA harmed all 30 distinct unmatched-policy base cells "  # noqa: E501
        "| Preregistered within-cell | Supported |",
        "| Matched-policy training mitigates TTA harm "
        "| Secondary fixed-model DiD, descriptively corroborated by separate within-cell patterns "
        "| Supported only secondarily/descriptively -- not a preregistered cross-condition test |",
        "| Normalization changes the magnitude of harm | Secondary fixed-model DiD only "
        "| Supported only secondarily; direction is dataset-dependent |",
        "| Higher resolution reduces TTA harm | Secondary fixed-model DiD only "
        "| Contradicted for BloodMNIST; mixed/near-null for PathMNIST |",
        "| BLOCK_C reproduces the source paper's positive TTA improvement "
        "| Preregistered within-cell (positive control) "
        "| Contradicted -- expected positive improvement not reproduced in any seed |",
        "| Any H4 (Validation-Gated TTA) verdict | None -- no derivable family exists | Not made, anywhere |",
        "| Any model-population or general medical-imaging generalization "
        "| None -- three seeds, fixed policy/budget | Not permitted, anywhere |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic figure rendering. Matplotlib is imported ONLY inside these
# functions -- never at module import time -- so `plan` mode and every
# extraction/table function above run fine with no matplotlib installed
# (as in the root environment). Colors are the frozen Okabe-Ito palette;
# every figure carries a visible zero-reference line and an embedded
# caption stating its evidence tier explicitly.
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

_DETERMINISTIC_PDF_METADATA = {"CreationDate": None, "Creator": "", "Producer": "", "Author": ""}
_DETERMINISTIC_PNG_METADATA = {"Software": ""}


def _save_deterministic(fig: Any, path_no_ext: Path) -> tuple[Path, Path]:
    """Saves one figure as both PDF and PNG (>=150 DPI) with normalized
    metadata for byte-for-byte reproducibility across independent runs."""
    pdf_path = path_no_ext.with_suffix(".pdf")
    png_path = path_no_ext.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", metadata=_DETERMINISTIC_PDF_METADATA)
    fig.savefig(png_path, format="png", dpi=150, metadata=_DETERMINISTIC_PNG_METADATA)
    return pdf_path, png_path


def _forest_plot(
    fig_ax: tuple[Any, Any],
    rows: list[dict[str, Any]],
    labels: list[str],
    values_key: str,
    ci_low_key: str,
    ci_high_key: str,
    row_colors: list[str],
    xlabel: str,
    caption: str,
    title: str,
) -> None:
    fig, ax = fig_ax
    n = len(rows)
    y_positions = list(range(n))
    values = [r[values_key] * 100 for r in rows]
    ci_low = [r[ci_low_key] * 100 for r in rows]
    ci_high = [r[ci_high_key] * 100 for r in rows]
    err_low = [v - lo for v, lo in zip(values, ci_low)]
    err_high = [hi - v for v, hi in zip(values, ci_high)]
    for i, color in enumerate(row_colors):
        ax.errorbar(
            [values[i]],
            [y_positions[i]],
            xerr=[[err_low[i]], [err_high[i]]],
            fmt="o",
            markersize=4,
            capsize=3,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor=OKABE_ITO["black"],
            linewidth=1.2,
        )
    ax.axvline(0.0, color=OKABE_ITO["black"], linewidth=1.0, linestyle="-")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.tick_params(labelsize=9)
    if caption:
        fig.text(0.02, 0.01, caption, fontsize=7, wrap=True, ha="left", va="bottom")


def render_figure_1(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color_by_norm = {"batchnorm": OKABE_ITO["blue"], "groupnorm": OKABE_ITO["vermillion"]}
    labels = [f"{r['dataset']} {r['resolution']}px {r['normalization']} s{r['seed']}" for r in rows]
    colors = [color_by_norm.get(r["normalization"], OKABE_ITO["black"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.28 * len(rows) + 1.0)))
    _forest_plot(
        (fig, ax),
        rows,
        labels,
        "delta_accuracy",
        "ci_low",
        "ci_high",
        colors,
        "Δ accuracy, TTA − clean (pp)",
        "Preregistered within-cell clean-versus-TTA evidence. Each row is one trained model "
        "(dataset x resolution x normalization x seed); no cross-condition comparison is made or "
        "implied here.",
        "Figure 1 — Unmatched-policy TTA effects (30 cells, preregistered)",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_1_unmatched_policy_forest")
    plt.close(fig)
    return paths


def render_figure_2(pairs: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {
            "did": p["bootstrap"]["did"],
            "ci_low": p["bootstrap"]["ci_low"],
            "ci_high": p["bootstrap"]["ci_high"],
        }
        for p in pairs
    ]
    labels = [p["pair_id"] for p in pairs]
    colors = [OKABE_ITO["bluish_green"]] * len(pairs)
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.35 * len(pairs) + 1.0)))
    _forest_plot(
        (fig, ax),
        rows,
        labels,
        "did",
        "ci_low",
        "ci_high",
        colors,
        "DiD, matched − unmatched policy (pp)",
        "Secondary, fixed-model, post-validation/pre-test-specified difference-in-differences "
        "comparison -- not a preregistered cross-condition inference. No significance decision is "
        "made for these estimates.",
        "Figure 2 — Matched-policy mitigation (6 secondary DiD pairs)",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_2_matched_policy_mitigation")
    plt.close(fig)
    return paths


def render_figure_3(pairs: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def dataset_of(p: dict[str, Any]) -> str:
        return _parse_run_id(p["condition_a"]["run_id"])["dataset"]

    datasets = sorted({dataset_of(p) for p in pairs})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 5.0), sharex=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        subset = sorted((p for p in pairs if dataset_of(p) == dataset), key=lambda p: p["pair_id"])
        rows = [
            {
                "did": p["bootstrap"]["did"],
                "ci_low": p["bootstrap"]["ci_low"],
                "ci_high": p["bootstrap"]["ci_high"],
            }
            for p in subset
        ]
        labels = [p["pair_id"] for p in subset]
        colors = [OKABE_ITO["sky_blue"]] * len(subset)
        _forest_plot(
            (fig, ax),
            rows,
            labels,
            "did",
            "ci_low",
            "ci_high",
            colors,
            "DiD, GroupNorm − BatchNorm (pp)",
            "",
            dataset,
        )
    fig.suptitle("Figure 3 — Normalization heterogeneity (12 secondary DiD pairs)", fontsize=10)
    fig.text(
        0.02,
        0.01,
        "The direction of this secondary estimate is dataset-dependent (see panels) and must not be "
        "read as a general BatchNorm-vs-GroupNorm verdict.",
        fontsize=7,
        wrap=True,
        ha="left",
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    paths = _save_deterministic(fig, output_dir / "figure_3_normalization_heterogeneity")
    plt.close(fig)
    return paths


def render_figure_4(pairs: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def group_of(p: dict[str, Any]) -> str:
        identity = _parse_run_id(p["condition_a"]["run_id"])
        return f"{identity['dataset']}/{identity['normalization']}"

    sorted_pairs = sorted(pairs, key=lambda p: (group_of(p), p["pair_id"]))
    labels = [f"{group_of(p)} {p['pair_id']}" for p in sorted_pairs]
    rows = [
        {
            "did": p["bootstrap"]["did"],
            "ci_low": p["bootstrap"]["ci_low"],
            "ci_high": p["bootstrap"]["ci_high"],
        }
        for p in sorted_pairs
    ]
    colors = [OKABE_ITO["orange"]] * len(sorted_pairs)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * len(sorted_pairs) + 1.0)))
    _forest_plot(
        (fig, ax),
        rows,
        labels,
        "did",
        "ci_low",
        "ci_high",
        colors,
        "DiD, high-res − low-res (pp)",
        "BloodMNIST pairs trend contrary to the hypothesized direction; PathMNIST pairs are "
        "mixed/near-null. Neither pattern is a preregistered or confirmatory test of H2. The dashed "
        "vertical line marks the hypothesized positive direction (reference only, not a confirmation "
        "marker).",
        "Figure 4 — Resolution comparison (12 secondary DiD pairs)",
    )
    ax.axvline(
        0.0,
        color=OKABE_ITO["black"],
        linewidth=1.0,
    )
    ax.axvline(
        1.0,
        color=OKABE_ITO["reddish_purple"],
        linewidth=1.0,
        linestyle="--",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_4_resolution_comparison")
    plt.close(fig)
    return paths


def render_figure_5(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"seed {r['seed']}" for r in rows]
    colors = [OKABE_ITO["blue"]] * len(rows)
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    _forest_plot(
        (fig, ax),
        rows,
        labels,
        "delta_accuracy",
        "ci_low",
        "ci_high",
        colors,
        "Δ accuracy, TTA − clean (pp)",
        "The expected positive TTA improvement (~+1.6pp) was not reproduced in this project's "
        "frozen operationalization. The dashed vertical line is the external reference (source "
        "paper), descriptive only -- not an acceptance threshold.",
        "Figure 5 — BLOCK_C positive control (3 seeds, preregistered)",
    )
    ax.axvline(
        1.6,
        color=OKABE_ITO["reddish_purple"],
        linewidth=1.0,
        linestyle="--",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    paths = _save_deterministic(fig, output_dir / "figure_5_block_c_positive_control")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Manifest and orchestration.
# ---------------------------------------------------------------------------


def build_evidence_plan(summary: dict[str, Any]) -> dict[str, Any]:
    """Zero-write, zero-matplotlib-import readiness plan: extracts every
    quantity and reports exact expected output counts/paths. Used by the
    CLI's `plan` subcommand."""
    unmatched = extract_unmatched_cells(summary)
    matched = extract_matched_within_cell(summary)
    h1_pairs = extract_cross_condition_pairs(summary, "H1")
    h2_pairs = extract_cross_condition_pairs(summary, "H2")
    h3_pairs = extract_cross_condition_pairs(summary, "H3")
    block_c = extract_block_c(summary)

    expected_figures = [
        "figure_1_unmatched_policy_forest",
        "figure_2_matched_policy_mitigation",
        "figure_3_normalization_heterogeneity",
        "figure_4_resolution_comparison",
        "figure_5_block_c_positive_control",
    ]
    expected_tables = [
        "table_1_design_classification",
        "table_2_unmatched_policy",
        "table_3_matched_policy",
        "table_4_normalization",
        "table_5_resolution",
        "table_6_block_c",
        "table_7_claim_adjudication",
    ]

    return {
        "n_unmatched_cells": len(unmatched),
        "n_matched_within_cell": len(matched),
        "n_h1_pairs": len(h1_pairs),
        "n_h2_pairs": len(h2_pairs),
        "n_h3_pairs": len(h3_pairs),
        "n_block_c_cells": len(block_c),
        "expected_figures": expected_figures,
        "expected_tables": expected_tables,
        "figures_dir": str(FIGURES_DIR),
        "tables_dir": str(TABLES_DIR),
        "manifest_path": str(MANIFEST_PATH),
    }


def generate_all_evidence(
    summary_path: str | Path = SCIENTIFIC_SUMMARY_PATH,
    output_root: Path = PAPER_EVIDENCE_ROOT,
) -> dict[str, Any]:
    """Real generation: verifies the canonical summary, extracts every
    quantity, renders all 5 figures (PDF+PNG) and 7 tables (Markdown),
    writes the binding manifest, and returns it. Idempotent: reruns
    produce byte-identical output (verified by the caller via hash
    comparison), since every rendering step is a pure function of the
    already-hash-verified canonical summary."""
    summary = load_and_verify_canonical_summary(summary_path)
    paper_evidence_fingerprint, fp_files = compute_paper_evidence_fingerprint()

    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    unmatched = extract_unmatched_cells(summary)
    matched = extract_matched_within_cell(summary)
    h1_pairs = extract_cross_condition_pairs(summary, "H1")
    h2_pairs = extract_cross_condition_pairs(summary, "H2")
    h3_pairs = extract_cross_condition_pairs(summary, "H3")
    block_c = extract_block_c(summary)

    figure_outputs: dict[str, tuple[Path, Path]] = {
        "figure_1_unmatched_policy_forest": render_figure_1(unmatched, figures_dir),
        "figure_2_matched_policy_mitigation": render_figure_2(h3_pairs, figures_dir),
        "figure_3_normalization_heterogeneity": render_figure_3(h1_pairs, figures_dir),
        "figure_4_resolution_comparison": render_figure_4(h2_pairs, figures_dir),
        "figure_5_block_c_positive_control": render_figure_5(block_c, figures_dir),
    }

    table_outputs: dict[str, Path] = {}
    table_contents = {
        "table_1_design_classification": render_design_classification_table(),
        "table_2_unmatched_policy": render_unmatched_table(unmatched),
        "table_3_matched_policy": render_matched_table(matched, h3_pairs),
        "table_4_normalization": render_cross_condition_table(
            "Table 4 — Complete 12-Pair Normalization Table", h1_pairs
        ),
        "table_5_resolution": render_cross_condition_table(
            "Table 5 — Complete 12-Pair Resolution Table", h2_pairs
        ),
        "table_6_block_c": render_block_c_table(block_c),
        "table_7_claim_adjudication": render_claim_adjudication_table(),
    }
    for name, content in table_contents.items():
        path = tables_dir / f"{name}.md"
        path.write_text(content)
        table_outputs[name] = path

    manifest_outputs: dict[str, dict[str, Any]] = {}
    for name, (pdf_path, png_path) in figure_outputs.items():
        for path in (pdf_path, png_path):
            manifest_outputs[str(path.relative_to(output_root.parent))] = {
                "sha256": hash_file(path),
                "size_bytes": path.stat().st_size,
            }
    for name, path in table_outputs.items():
        manifest_outputs[str(path.relative_to(output_root.parent))] = {
            "sha256": hash_file(path),
            "size_bytes": path.stat().st_size,
        }

    auth = verify_unsealing_authorization()
    manifest = {
        "schema_version": "phase2b.9a-v1",
        "paper_evidence_fingerprint": paper_evidence_fingerprint,
        "canonical_summary_sha256": hash_file(Path(summary_path)),
        "canonical_summary_reporting_fingerprint": summary["reporting_fingerprint"],
        "unsealing_authorization_sha256": hash_file(Path(FINAL_TEST_UNSEALING_AUTHORIZATION_PATH)),
        "unsealing_authorization_status": auth.get("status"),
        "outputs": manifest_outputs,
    }
    manifest_path = output_root / "paper_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
