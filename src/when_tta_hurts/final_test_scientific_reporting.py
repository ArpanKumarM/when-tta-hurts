"""Phase 2B.8A: deterministic scientific-report generator for the seven
sealed final-test analysis results (four preregistered families, three
secondary cross-condition hypotheses), per
docs/phase2b_final_test_unsealing_freeze.md.

This module is the ONLY place in the repository permitted to parse the
scientific contents of a final-test analysis result -- every other
module (plan modes, CLI, ledger) treats those files as opaque bytes.
Resolution of WHICH seven inputs are eligible (`resolve_seven_sealed_inputs`)
is metadata-only (ledger rows + manifest hashes, never `json.loads` of a
result file) and is what `plan` mode in the CLI uses. Actual parsing
(`load_and_verify_sealed_result`) happens only inside `unseal`, and only
after both the generation-3 analysis authorization and a dedicated
unsealing authorization have been verified.

Never reads `artifacts/validation_evaluation/` as a scientific input.
Never invents H4, a pooled/model-population p-value, a new hypothesis,
a new endpoint, a new exclusion, or a significance flag for the
secondary addendum (see freeze doc sec.2/6/11).
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from when_tta_hurts.artifacts import hash_file
from when_tta_hurts.config import config_hash
from when_tta_hurts.final_test_statistical_analysis import (
    FINAL_TEST_ANALYSIS_ROOT,
    FINAL_TEST_CROSS_CONDITION_ROOT,
    verify_final_test_analysis_authorization,
)
from when_tta_hurts.statistical_analysis_artifacts import (
    AnalysisPersistenceError,
    validate_analysis_result_schema,
    validate_cross_condition_result_schema,
    verify_analysis_artifact_manifest,
)

FINAL_TEST_ANALYSIS_LEDGER_PATH = Path("artifacts/ledger_final_test_analysis.csv")
FINAL_TEST_UNSEALING_AUTHORIZATION_PATH = Path("artifacts/final_test_unsealing_authorization.json")

SCIENTIFIC_SUMMARY_PATH = Path("artifacts/final_test_scientific_summary.json")
RESULTS_MARKDOWN_PATH = Path("docs/phase2b_final_test_scientific_results.md")
INTERPRETATION_MARKDOWN_PATH = Path("docs/phase2b_final_test_scientific_interpretation.md")

# (kind, identifier, expected_count, count_field_name) -- the exact
# seven inputs and their frozen membership counts (freeze doc sec.1/3).
EXPECTED_UNITS: tuple[tuple[str, str, int, str], ...] = (
    ("family", "H1", 24, "cells"),
    ("family", "H2", 30, "cells"),
    ("family", "H3", 12, "cells"),
    ("family", "BLOCK_C", 3, "cells"),
    ("cross_condition", "H1", 12, "pairs"),
    ("cross_condition", "H2", 12, "pairs"),
    ("cross_condition", "H3", 6, "pairs"),
)

# Every file whose content could change a rendered reporting number,
# ordering, or formatting decision -- deliberately DISJOINT from
# ANALYSIS_FINGERPRINT_MANIFEST / CROSS_CONDITION_ADDENDUM_MANIFEST /
# FINAL_TEST_RUNNER_MANIFEST / FINAL_TEST_STATISTICAL_ANALYSIS_MANIFEST.
# Reporting is a read-only, downstream consumer of those already-sealed
# computations; it must never be added to (or itself include) any of
# their manifests, since that would make already-completed, already-
# authorized scientific computations identity-unstable for a reason
# that has nothing to do with how they were computed.
FINAL_TEST_REPORTING_MANIFEST: tuple[str, ...] = (
    "src/when_tta_hurts/final_test_scientific_reporting.py",
    "scripts/generate_final_test_scientific_report.py",
    "docs/phase2b_final_test_unsealing_freeze.md",
    "docs/phase2b_final_test_reporting_wording_correction_freeze.md",
    "src/when_tta_hurts/statistical_analysis_artifacts.py",
    "pyproject.toml",
    "uv.lock",
)


class FinalTestReportingFingerprintError(RuntimeError):
    """Raised when a file listed in FINAL_TEST_REPORTING_MANIFEST is
    missing. Fails closed -- never computes a partial fingerprint."""


def compute_final_test_reporting_fingerprint(
    repo_root: str | Path = ".",
    manifest: tuple[str, ...] = FINAL_TEST_REPORTING_MANIFEST,
) -> tuple[str, dict[str, str]]:
    repo_root = Path(repo_root)
    file_hashes: dict[str, str] = {}
    for rel_path in manifest:
        path = repo_root / rel_path
        if not path.exists():
            raise FinalTestReportingFingerprintError(
                f"Reporting fingerprint manifest file missing: {rel_path}. Refusing to compute a "
                f"partial fingerprint."
            )
        file_hashes[rel_path] = hash_file(path)
    fingerprint = config_hash({"manifest_version": 1, "files": file_hashes})
    return fingerprint, file_hashes


class SealedInputResolutionError(RuntimeError):
    """Raised when the seven sealed inputs cannot be unambiguously
    resolved: missing, duplicate, ambiguous, non-completed, or a
    ledger/manifest-hash mismatch. Fails closed -- never guesses which
    row is canonical."""


class SealedInputTamperError(RuntimeError):
    """Raised when a resolved input's on-disk bytes do not match its
    ledger-recorded hash, or its manifest fails verification. Fails
    closed before any scientific parsing occurs."""


def _read_ledger_rows(ledger_path: str | Path) -> list[dict[str, Any]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_seven_sealed_inputs(
    ledger_path: str | Path = FINAL_TEST_ANALYSIS_LEDGER_PATH,
    analysis_root: str | Path = FINAL_TEST_ANALYSIS_ROOT,
    cross_root: str | Path = FINAL_TEST_CROSS_CONDITION_ROOT,
) -> dict[str, dict[str, Any]]:
    """METADATA-ONLY resolution of the seven sealed inputs: ledger rows
    plus manifest-file BYTE hashes (never `json.loads` of a result file,
    never a scientific field). Used by both plan mode and as the first
    stage of real unsealing. Raises SealedInputResolutionError if any of
    the seven expected (kind, identifier) units does not resolve to
    EXACTLY one completed row, or SealedInputTamperError if the on-disk
    manifest bytes don't match the ledger's recorded hash."""
    rows = _read_ledger_rows(ledger_path)
    resolved: dict[str, dict[str, Any]] = {}

    for kind, identifier, expected_count, count_field in EXPECTED_UNITS:
        matches = [
            r
            for r in rows
            if r.get("kind") == kind and r.get("identifier") == identifier and r.get("status") == "completed"
        ]
        if len(matches) == 0:
            raise SealedInputResolutionError(f"No completed row for ({kind}, {identifier}).")
        if len(matches) > 1:
            raise SealedInputResolutionError(
                f"Ambiguous: {len(matches)} completed rows for ({kind}, {identifier})."
            )
        row = matches[0]
        attempt = int(row["analysis_attempt"])
        root = analysis_root if kind == "family" else cross_root
        attempt_dir = Path(root) / identifier / f"attempt_{attempt:03d}"
        manifest_filename = (
            "artifact_manifest.json" if kind == "family" else "cross_condition_artifact_manifest.json"
        )
        result_filename = "analysis_result.json" if kind == "family" else "cross_condition_result.json"
        manifest_path = attempt_dir / manifest_filename
        result_path = attempt_dir / result_filename
        if not manifest_path.exists() or not result_path.exists():
            raise SealedInputResolutionError(
                f"Missing artifact file(s) for ({kind}, {identifier}) attempt {attempt}."
            )

        actual_result_hash = hashlib.sha256(result_path.read_bytes()).hexdigest()
        if actual_result_hash != row.get("primary_artifact_hash"):
            raise SealedInputTamperError(
                f"({kind}, {identifier}): on-disk result hash does not match the ledger's recorded "
                f"primary_artifact_hash."
            )

        key = f"{kind}:{identifier}"
        resolved[key] = {
            "kind": kind,
            "identifier": identifier,
            "expected_count": expected_count,
            "count_field": count_field,
            "analysis_id": row["analysis_id"],
            "attempt": attempt,
            "attempt_dir": attempt_dir,
            "manifest_path": manifest_path,
            "result_path": result_path,
            "primary_artifact_hash": row["primary_artifact_hash"],
            "final_test_authorization_sha256": row["final_test_authorization_sha256"],
            "final_test_authorization_commit": row["final_test_authorization_commit"],
            "final_test_analysis_fingerprint": row["final_test_analysis_fingerprint"],
            "current_evaluator_fingerprint": row["current_evaluator_fingerprint"],
        }

    return resolved


class UnsealingAuthorizationError(RuntimeError):
    """Raised when the dedicated unsealing authorization artifact is
    missing, malformed, not approved, or bound to stale fingerprints/
    hashes. Checked BEFORE any scientific-result parsing."""


def verify_unsealing_authorization(
    authorization_path: str | Path = FINAL_TEST_UNSEALING_AUTHORIZATION_PATH,
) -> dict[str, Any]:
    """Fast, content-only gate for the dedicated unsealing authorization
    (distinct from, and layered on top of, the generation-3 final-test-
    analysis authorization). Never opens a result JSON. Raises
    UnsealingAuthorizationError on any of: missing file, malformed JSON,
    status != 'approved', or a recorded reporting fingerprint that does
    not match the CURRENT compute_final_test_reporting_fingerprint()."""
    path = Path(authorization_path)
    if not path.exists():
        raise UnsealingAuthorizationError(f"Unsealing authorization artifact {path} does not exist.")
    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        raise UnsealingAuthorizationError(
            f"Unsealing authorization artifact {path} is malformed JSON."
        ) from e
    if not isinstance(raw, dict) or raw.get("status") != "approved":
        raise UnsealingAuthorizationError(
            f"Unsealing authorization artifact {path} is not status='approved'."
        )
    current_fp, _ = compute_final_test_reporting_fingerprint()
    if raw.get("final_test_reporting_fingerprint") != current_fp:
        raise UnsealingAuthorizationError(
            "Unsealing authorization's final_test_reporting_fingerprint does not match the current "
            "reporting fingerprint -- authorization is stale."
        )
    return raw


class SealedResultSchemaError(RuntimeError):
    """Raised when a resolved input's parsed JSON fails schema
    validation or its declared unit count does not match
    EXPECTED_UNITS. Fails closed -- never proceeds with a partial or
    malformed input."""


def load_and_verify_sealed_result(entry: dict[str, Any]) -> dict[str, Any]:
    """Real parsing entry point -- called ONLY from `unseal`, ONLY after
    both authorizations have been verified. Re-verifies the artifact
    manifest, parses the result JSON, validates its schema, and checks
    its declared unit count against EXPECTED_UNITS before returning it."""
    manifest = json.loads(entry["manifest_path"].read_text())
    try:
        verify_analysis_artifact_manifest(entry["attempt_dir"], manifest)
    except AnalysisPersistenceError as e:
        raise SealedInputTamperError(f"Manifest verification failed for {entry['identifier']}: {e}") from e

    result = json.loads(entry["result_path"].read_text())
    kind = entry["kind"]
    if kind == "family":
        validate_analysis_result_schema(result)
        actual_count = len(result["cells"])
    else:
        validate_cross_condition_result_schema(result)
        actual_count = len(result["pairs"])

    if actual_count != entry["expected_count"]:
        raise SealedResultSchemaError(
            f"{entry['identifier']}: expected {entry['expected_count']} {entry['count_field']}, "
            f"found {actual_count}."
        )
    return result


# ---------------------------------------------------------------------------
# Deterministic, pure rendering. No RNG, no wall-clock timestamp in any
# rendered content, no hidden state -- identical inputs always produce
# byte-identical output.
# ---------------------------------------------------------------------------


def _round_stable(value: Any) -> Any:
    """Identity passthrough -- JSON output preserves full machine-
    readable float precision (freeze doc sec.8/10: 'preserve full
    machine-readable precision'). No rounding is applied anywhere in
    this module."""
    return value


def _extract_preregistered_rows(identifier: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run_id in sorted(result["cells"]):
        stats = result["per_cell_statistics"][run_id]
        rows.append(
            {
                "family": identifier,
                "run_id": run_id,
                "bootstrap": stats["bootstrap"],
                "mcnemar": stats["mcnemar"],
                "effect_sizes": stats["effect_sizes"],
                "n_samples": stats["n_samples"],
            }
        )
    return rows


def _extract_secondary_rows(identifier: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pair_id in sorted(result["pairs"]):
        pr = result["per_pair_results"][pair_id]
        rows.append(
            {
                "hypothesis": identifier,
                "pair_id": pair_id,
                "condition_a": pr["condition_a"],
                "condition_b": pr["condition_b"],
                "bootstrap": pr["bootstrap"],
                "n_samples": pr["n_samples"],
            }
        )
    return rows


def _descriptive_seed_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Groups preregistered rows by (dataset, resolution, normalization)
    -- i.e. by everything in run_id except the seed -- and computes
    simple, auditable mean/sample-stddev/min/max of each group's paired
    effect estimate (bootstrap.delta_accuracy) across its seeds. Labeled
    'descriptive' throughout (freeze doc sec.7) -- never a new
    statistical test, never assigned a p-value or CI of its own."""
    import statistics
    from collections import defaultdict

    groups: dict[str, list[float]] = defaultdict(list)
    group_seeds: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        run_id = row["run_id"]
        group_key = run_id.rsplit("-s", 1)[0] if "-s" in run_id else run_id
        groups[group_key].append(row["bootstrap"]["delta_accuracy"])
        group_seeds[group_key].append(run_id)

    summaries = []
    for group_key in sorted(groups):
        values = groups[group_key]
        summaries.append(
            {
                "group": group_key,
                "classification": "descriptive_non_inferential",
                "n_seeds": len(values),
                "mean": statistics.fmean(values),
                "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
                "min": min(values),
                "max": max(values),
                "seed_run_ids": sorted(group_seeds[group_key]),
            }
        )
    return summaries


def build_scientific_summary(
    resolved: dict[str, dict[str, Any]], loaded: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Pure function: builds the complete machine-readable JSON summary
    from already-loaded, already-validated results. No manuscript prose.
    No H4, no pooled/model-population p-value, no new significance flag
    -- only what EXPECTED_UNITS and the persisted schemas already
    contain."""
    reporting_fp, _ = compute_final_test_reporting_fingerprint()

    preregistered: dict[str, Any] = {}
    all_preregistered_rows: list[dict[str, Any]] = []
    for identifier in ("H1", "H2", "H3", "BLOCK_C"):
        entry = resolved[f"family:{identifier}"]
        result = loaded[f"family:{identifier}"]
        rows = _extract_preregistered_rows(identifier, result)
        all_preregistered_rows.extend(rows)
        preregistered[identifier] = {
            "analysis_id": entry["analysis_id"],
            "attempt": entry["attempt"],
            "n_cells": len(rows),
            "cells": rows,
            "multiplicity": result["multiplicity"],
        }

    secondary: dict[str, Any] = {}
    for identifier in ("H1", "H2", "H3"):
        entry = resolved[f"cross_condition:{identifier}"]
        result = loaded[f"cross_condition:{identifier}"]
        rows = _extract_secondary_rows(identifier, result)
        secondary[identifier] = {
            "analysis_id": entry["analysis_id"],
            "attempt": entry["attempt"],
            "classification": result["classification"],
            "n_pairs": len(rows),
            "pairs": rows,
        }

    return {
        "schema_version": "phase2b.8a-v1",
        "reporting_fingerprint": reporting_fp,
        "provenance": {
            "final_test_closure_commit": "581143e6d1c080c3bfaad941514a356089313926",
            "preregistered_results_commit": "4426bf55476d1761afe15ded4d56d48ded0fee51",
            "cross_condition_results_commit": "29a3bfe",
            "unsealing_freeze_commit": "486028c3b2e40014a38b3bfc818e113a95d39f9c",
        },
        "inputs": {
            key: {
                "kind": v["kind"],
                "identifier": v["identifier"],
                "analysis_id": v["analysis_id"],
                "attempt": v["attempt"],
                "primary_artifact_hash": v["primary_artifact_hash"],
                "final_test_authorization_sha256": v["final_test_authorization_sha256"],
                "final_test_analysis_fingerprint": v["final_test_analysis_fingerprint"],
            }
            for key, v in sorted(resolved.items())
        },
        "preregistered": preregistered,
        "secondary_cross_condition": secondary,
        "descriptive_summaries": {
            "preregistered_seed_level": _descriptive_seed_summary(all_preregistered_rows),
        },
    }


def render_results_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2B Final-Test Scientific Results",
        "",
        "Mechanically generated from the seven sealed, committed final-test "
        "analysis artifacts. Every planned family, cell, and pair is reported "
        "below with no selective omission by magnitude, direction, confidence "
        "interval, or p-value.",
        "",
        "## Preregistered within-cell results (confirmatory)",
        "",
    ]
    for family, data in summary["preregistered"].items():
        lines.append(f"### {family} (n_cells={data['n_cells']})")
        lines.append("")
        lines.append("| run_id | delta_accuracy | ci_low | ci_high | mcnemar_p | bh_adjusted_p |")
        lines.append("|---|---|---|---|---|---|")
        raw_p = data["multiplicity"]["raw_p_values"]
        adj_p = data["multiplicity"]["corrected_p_values"]
        cells_sorted = sorted(data["cells"], key=lambda r: r["run_id"])
        cell_index = {c["run_id"]: i for i, c in enumerate(data["cells"])}
        for row in cells_sorted:
            i = cell_index[row["run_id"]]
            b = row["bootstrap"]
            lines.append(
                f"| {row['run_id']} | {b['delta_accuracy']} | {b['ci_low']} | {b['ci_high']} | "
                f"{raw_p[i]} | {adj_p[i]} |"
            )
        lines.append("")
    lines.append(
        "## Secondary cross-condition results (post-validation, pre-test-specified, fixed-model-only)"
    )
    lines.append("")
    lines.append("**Never pooled into a single p-value or model-population verdict.**")
    lines.append("")
    for hyp, data in summary["secondary_cross_condition"].items():
        lines.append(f"### Cross-condition {hyp} (n_pairs={data['n_pairs']})")
        lines.append("")
        lines.append("| pair_id | did | ci_low | ci_high |")
        lines.append("|---|---|---|---|")
        for row in sorted(data["pairs"], key=lambda r: r["pair_id"]):
            b = row["bootstrap"]
            lines.append(f"| {row['pair_id']} | {b['did']} | {b['ci_low']} | {b['ci_high']} |")
        lines.append("")
    lines.append("## Descriptive seed-level summaries (non-inferential)")
    lines.append("")
    lines.append(
        "These are simple descriptive statistics over already-reported seed "
        "values above. They are NOT additional confirmatory tests and carry "
        "no p-value or confidence interval of their own."
    )
    lines.append("")
    for s in summary["descriptive_summaries"]["preregistered_seed_level"]:
        lines.append(
            f"- `{s['group']}` (n_seeds={s['n_seeds']}, classification={s['classification']}): "
            f"mean={s['mean']}, sample_stdev={s['sample_stdev']}, min={s['min']}, max={s['max']}"
        )
    lines.append("")
    return "\n".join(lines)


def render_interpretation_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 2B Final-Test Scientific Interpretation",
        "",
        "Cautious interpretation, limitations, and claim adjudication only. "
        "Governed entirely by the rules frozen in "
        "docs/phase2b_final_test_unsealing_freeze.md sec.8/9 -- no rule is "
        "introduced here that was not already frozen before this result was "
        "unsealed.",
        "",
        "## Scientific classification (reaffirmed)",
        "",
        "- Preregistered final-test analyses (H1/H2/H3/BLOCK_C) are the "
        "confirmatory within-cell analyses specified by the frozen SAP. They "
        "do not, by themselves, establish the cross-condition differences "
        "implied by H1/H2/H3.",
        "- Cross-condition difference-in-differences analyses are explicitly "
        "secondary, post-validation/pre-test specified, fixed-model-only, and "
        "not preregistered.",
        "- BLOCK_C is a positive-control analysis, reported regardless of direction.",
        "- No H4 claim is made anywhere in this document.",
        "- No population-level or model-population inference is made anywhere in this document.",
        "",
        "## Required limitations",
        "",
        "- Only three training seeds per cell; sample-level paired tests do "
        "not substitute for a model-seed population replication study.",
        "- Limited dataset/architecture coverage (the specific MedMNIST "
        "subsets and ResNet variants actually used).",
        "- A fixed augmentation policy and a fixed TTA view budget (N=50).",
        "- The cross-condition addendum was frozen after validation-stage "
        "results were already observed, but before the official test split "
        "was opened.",
        "",
        "## Incident disclosure",
        "",
        "- The accidental final-test access incident for cell 1 (attempt 1, aborted).",
        "- Two failed final-test engineering attempts (cell 1 attempt 2; "
        "cell 2 attempt 1), neither of which persisted any scientific value.",
        "- The shared-aggregation-contract correction and the validation-"
        "metric-reconciliation mechanism it required.",
        "- All 39 canonical final-test results were produced under the "
        "final, corrected evaluator/aggregation pipeline; cell 1's "
        "compatibility under its historical generation-3 binding was "
        "independently established via 56/56 recomputation checks, never "
        "assumed.",
        "- No final-test scientific result was inspected by a human before this controlled unsealing.",
        "",
    ]
    return "\n".join(lines)


class ReportGenerationError(RuntimeError):
    """Raised on any rendering/validation failure. No partial output set
    is ever left on disk when this is raised."""


def generate_and_persist_report(
    summary_path: str | Path = SCIENTIFIC_SUMMARY_PATH,
    results_path: str | Path = RESULTS_MARKDOWN_PATH,
    interpretation_path: str | Path = INTERPRETATION_MARKDOWN_PATH,
) -> dict[str, Any]:
    """Full unsealing pipeline: verify both authorizations, resolve and
    load all seven sealed inputs, render all three outputs in memory,
    then write them atomically as one logical operation (all three
    temp-write-then-rename, rolling back any already-renamed file if a
    later rename fails). Raises without writing anything if any prior
    step fails. Idempotent: if a byte-identical, already-completed
    output set exists, returns its identity without rewriting."""
    verify_final_test_analysis_authorization()
    verify_unsealing_authorization()

    resolved = resolve_seven_sealed_inputs()
    loaded = {key: load_and_verify_sealed_result(entry) for key, entry in resolved.items()}

    summary = build_scientific_summary(resolved, loaded)
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n").encode()
    results_md = render_results_markdown(summary) + "\n"
    interpretation_md = render_interpretation_markdown(summary) + "\n"

    summary_path, results_path, interpretation_path = (
        Path(summary_path),
        Path(results_path),
        Path(interpretation_path),
    )

    existing_hash = hashlib.sha256(summary_bytes).hexdigest()
    if summary_path.exists():
        current_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest()
        if current_hash == existing_hash and results_path.exists() and interpretation_path.exists():
            return {"status": "idempotent_skip", "summary_sha256": existing_hash}
        raise ReportGenerationError(
            "An existing scientific summary is present but does not match the freshly-rendered "
            "content (or the markdown outputs are missing) -- refusing to overwrite a conflicting "
            "output set."
        )

    tmp_paths = []
    written_final_paths = []
    try:
        for path, content in (
            (summary_path, summary_bytes),
            (results_path, results_md.encode()),
            (interpretation_path, interpretation_md.encode()),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(content)
            tmp_paths.append((tmp, path))

        for tmp, final in tmp_paths:
            shutil.move(str(tmp), str(final))
            written_final_paths.append(final)
    except Exception:
        for final in written_final_paths:
            final.unlink(missing_ok=True)
        for tmp, _ in tmp_paths:
            Path(tmp).unlink(missing_ok=True)
        raise

    return {
        "status": "completed",
        "summary_sha256": existing_hash,
        "outputs": [str(p) for p in (summary_path, results_path, interpretation_path)],
    }


__all__ = [
    "EXPECTED_UNITS",
    "FINAL_TEST_ANALYSIS_LEDGER_PATH",
    "FINAL_TEST_REPORTING_MANIFEST",
    "FINAL_TEST_UNSEALING_AUTHORIZATION_PATH",
    "INTERPRETATION_MARKDOWN_PATH",
    "RESULTS_MARKDOWN_PATH",
    "SCIENTIFIC_SUMMARY_PATH",
    "FinalTestReportingFingerprintError",
    "ReportGenerationError",
    "SealedInputResolutionError",
    "SealedInputTamperError",
    "SealedResultSchemaError",
    "UnsealingAuthorizationError",
    "build_scientific_summary",
    "compute_final_test_reporting_fingerprint",
    "generate_and_persist_report",
    "load_and_verify_sealed_result",
    "render_interpretation_markdown",
    "render_results_markdown",
    "resolve_seven_sealed_inputs",
    "verify_unsealing_authorization",
]
