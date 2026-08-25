#!/usr/bin/env python3
"""Phase 2B.9B Part D: read-only manuscript-claim verification.

Reads exactly: `paper/manuscript.md`, `paper/references.bib`,
`paper/citation_audit.md`, the canonical
`artifacts/final_test_scientific_summary.json` (via
`when_tta_hurts.paper_evidence`'s already-verified extraction helpers),
and the committed `artifacts/paper_evidence/` tables/manifest. Never
reads raw predictions, datasets, checkpoints, or sealed per-family
analysis result JSONs -- only the already-sealed, already-committed
summary and evidence-package artifacts.

Performs zero writes. Exits 0 with a JSON readiness report if every
check passes; exits 1 and prints the failing check(s) otherwise.

Usage:
    uv run python3 paper/verify_manuscript_claims.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MANUSCRIPT_PATH = REPO_ROOT / "paper" / "manuscript.md"
REFERENCES_PATH = REPO_ROOT / "paper" / "references.bib"
CITATION_AUDIT_PATH = REPO_ROOT / "paper" / "citation_audit.md"
PAPER_EVIDENCE_TABLES_DIR = REPO_ROOT / "artifacts" / "paper_evidence" / "tables"
PAPER_EVIDENCE_MANIFEST_PATH = REPO_ROOT / "artifacts" / "paper_evidence" / "paper_evidence_manifest.json"

REQUIRED_FIGURES = tuple(f"Figure {i}" for i in range(1, 6))
REQUIRED_TABLES = tuple(f"Table {i}" for i in range(1, 8))

FORBIDDEN_PHRASES = (
    "54 distinct",
    "h4",
    "significant",
    "significance",
    "clinical use",
    "clinically",
    "diagnostic tool",
    "state-of-the-art",
    "state of the art",
    "cvpr",
)

# A forbidden phrase is only a genuine violation if its enclosing
# paragraph contains none of these negation/rejection cues -- i.e. the
# manuscript is disclaiming, rejecting, or explaining why it avoids the
# phrase, not asserting it as a result. This mirrors how a human
# reviewer would read "we do not claim X" versus "X is true."
NEGATION_CUES = (
    "not ",
    "never",
    "no claim",
    "does not",
    "do not",
    "must not",
    "forbidden",
    "reject",
    "excluded",
    "n't",
    "without",
)

REQUIRED_WORDING = ("30 distinct unmatched-policy",)

REQUIRED_DISCLOSURES = (
    "three training seeds",
    "accidental final-test access incident",
    "shared-aggregation-contract correction",
    "inspected by a human before",
    "no h4 claim",
    "population-level",
)


class ManuscriptVerificationError(RuntimeError):
    """Raised when a required input file is missing. Fails closed --
    never proceeds with a partial verification."""


def load_manuscript() -> str:
    if not MANUSCRIPT_PATH.exists():
        raise ManuscriptVerificationError(f"{MANUSCRIPT_PATH} does not exist.")
    return MANUSCRIPT_PATH.read_text()


def normalize_whitespace(text: str) -> str:
    """Collapses soft line-wraps within a paragraph into single spaces
    (while leaving paragraph breaks as a marker) so a phrase check never
    misses a match purely because Markdown line-wrapping split it across
    two lines -- e.g. '(Figure\\n4)' must still match 'Figure 4'."""
    paragraphs = text.split("\n\n")
    return "\n\n".join(" ".join(p.split()) for p in paragraphs)


def extract_section(text: str, heading: str, next_headings: tuple[str, ...]) -> str:
    start = text.find(f"## {heading}\n")
    if start == -1:
        return ""
    start += len(f"## {heading}\n")
    end = len(text)
    for nh in next_headings:
        idx = text.find(f"## {nh}", start)
        if idx != -1:
            end = min(end, idx)
    return text[start:end]


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*\.\d+")


def extract_decimal_numbers(text: str) -> set[str]:
    """Every decimal number (optionally comma-grouped, optionally
    negative) appearing in the given text, normalized to a bare
    float-string with commas removed."""
    return {m.replace(",", "") for m in _NUMBER_RE.findall(text)}


def build_known_good_decimal_numbers() -> set[str]:
    """Every decimal pp-value that could legitimately appear in the
    Results section, derived mechanically from the canonical summary via
    the same extraction helpers used to build the committed evidence
    package -- never a hand-typed number."""
    from when_tta_hurts.paper_evidence import (
        extract_block_c,
        extract_cross_condition_pairs,
        extract_matched_within_cell,
        extract_unmatched_cells,
        load_and_verify_canonical_summary,
    )

    summary = load_and_verify_canonical_summary()
    known: set[str] = set()

    def _add(value: float) -> None:
        known.add(f"{value * 100:.2f}")
        known.add(f"{value * 100:.1f}")
        known.add(f"{abs(value) * 100:.2f}")
        known.add(f"{abs(value) * 100:.1f}")

    for row in extract_unmatched_cells(summary):
        _add(row["delta_accuracy"])
        _add(row["ci_low"])
        _add(row["ci_high"])
    for row in extract_matched_within_cell(summary):
        _add(row["delta_accuracy"])
        _add(row["ci_low"])
        _add(row["ci_high"])
    for row in extract_block_c(summary):
        _add(row["delta_accuracy"])
        _add(row["ci_low"])
        _add(row["ci_high"])
    for hyp in ("H1", "H2", "H3"):
        for pair in extract_cross_condition_pairs(summary, hyp):
            _add(pair["bootstrap"]["did"])
            _add(pair["bootstrap"]["ci_low"])
            _add(pair["bootstrap"]["ci_high"])

    # The external BLOCK_C descriptive reference (source paper's own
    # reported figure, not a value derived from this project's data) is
    # a legitimate Results-section number and is allow-listed here by
    # its already-documented value
    # (docs/phase2b_validation_evaluation_block_c_audit.md sec.7).
    known.add("1.6")

    return known


def check_numeric_claims(results_text: str) -> list[str]:
    violations = []
    manuscript_numbers = extract_decimal_numbers(results_text)
    known_good = build_known_good_decimal_numbers()
    for number in sorted(manuscript_numbers):
        if number.lstrip("-") not in known_good and number not in known_good:
            violations.append(
                f"Results-section number {number!r} does not match any value derivable from the "
                f"canonical evidence package."
            )
    return violations


def check_required_wording(full_text: str) -> list[str]:
    return [f"Required wording missing: {phrase!r}" for phrase in REQUIRED_WORDING if phrase not in full_text]


def _paragraphs(text: str) -> list[str]:
    return [p for p in text.split("\n\n") if p.strip()]


def check_forbidden_phrases(full_text: str) -> list[str]:
    violations = []
    lowered_paragraphs = [(p, p.lower()) for p in _paragraphs(full_text)]
    for phrase in FORBIDDEN_PHRASES:
        for original, lowered in lowered_paragraphs:
            if phrase in lowered:
                if not any(cue in lowered for cue in NEGATION_CUES):
                    violations.append(
                        f"Forbidden phrase {phrase!r} appears in a paragraph with no negation/rejection "
                        f"cue: {original[:120]!r}..."
                    )
    return violations


def check_figures_and_tables_referenced(full_text: str) -> list[str]:
    violations = []
    for fig in REQUIRED_FIGURES:
        if fig not in full_text:
            violations.append(f"{fig} is never referenced in the manuscript.")
    for tab in REQUIRED_TABLES:
        if tab not in full_text and f"Supplementary {tab}" not in full_text:
            violations.append(f"{tab} is never referenced in the manuscript (main text or supplementary).")
    return violations


_BIB_KEY_RE = re.compile(r"@\w+\{\s*([^,\s]+)\s*,")


def load_bib_keys() -> set[str]:
    if not REFERENCES_PATH.exists():
        raise ManuscriptVerificationError(f"{REFERENCES_PATH} does not exist.")
    return set(_BIB_KEY_RE.findall(REFERENCES_PATH.read_text()))


def check_citations(full_text: str) -> list[str]:
    violations = []
    bib_keys = load_bib_keys()
    # Only keys that actually look like citation markers ([@key] or
    # [@key; @key2] form) count as "used" -- a bare '@' elsewhere would
    # not match this pattern.
    cite_pattern = re.compile(r"\[@([A-Za-z0-9_]+)(?:;\s*@([A-Za-z0-9_]+))*\]")
    used_keys = set()
    for match in cite_pattern.finditer(full_text):
        for group in match.groups():
            if group:
                used_keys.add(group)
    if not used_keys:
        violations.append("No citation keys of the form [@key] were found in the manuscript.")
    for key in sorted(used_keys):
        if key not in bib_keys:
            violations.append(f"Citation key {key!r} used in manuscript but absent from references.bib.")

    if not CITATION_AUDIT_PATH.exists():
        raise ManuscriptVerificationError(f"{CITATION_AUDIT_PATH} does not exist.")
    audit_text = CITATION_AUDIT_PATH.read_text()
    for key in sorted(used_keys):
        if key not in audit_text:
            violations.append(f"Citation key {key!r} has no entry in citation_audit.md.")
            continue
        # Find the audit table row for this key and require it to state
        # some verification status (Verified, or the disclosed
        # Partially verified case) -- never silently accepted.
        row_match = re.search(rf"\| `{re.escape(key)}` \|.*\|\s*$", audit_text, re.MULTILINE)
        row_text = row_match.group(0) if row_match else ""
        if "verified" not in row_text.lower():
            violations.append(f"Citation key {key!r}'s audit row does not record a verification status.")
    return violations


def check_required_disclosures(full_text: str) -> list[str]:
    lowered = full_text.lower()
    return [
        f"Required disclosure missing: {phrase!r}" for phrase in REQUIRED_DISCLOSURES if phrase not in lowered
    ]


def check_evidence_package_present() -> list[str]:
    violations = []
    if not PAPER_EVIDENCE_MANIFEST_PATH.exists():
        violations.append(f"{PAPER_EVIDENCE_MANIFEST_PATH} does not exist.")
        return violations
    manifest = json.loads(PAPER_EVIDENCE_MANIFEST_PATH.read_text())
    if len(manifest.get("outputs", {})) != 17:
        violations.append("paper_evidence_manifest.json does not record exactly 17 outputs.")
    if not PAPER_EVIDENCE_TABLES_DIR.exists() or len(list(PAPER_EVIDENCE_TABLES_DIR.glob("*.md"))) != 7:
        violations.append("artifacts/paper_evidence/tables/ does not contain exactly 7 table files.")
    return violations


def run_all_checks() -> dict[str, Any]:
    raw_text = load_manuscript()
    full_text = normalize_whitespace(raw_text)
    results_text = extract_section(raw_text, "Results", ("Discussion",))
    if not results_text.strip():
        raise ManuscriptVerificationError("Manuscript has no non-empty '## Results' section.")

    checks: dict[str, list[str]] = {
        "numeric_claims_in_results": check_numeric_claims(results_text),
        "required_wording": check_required_wording(full_text),
        "forbidden_phrases": check_forbidden_phrases(full_text),
        "figures_and_tables_referenced": check_figures_and_tables_referenced(full_text),
        "citations": check_citations(full_text),
        "required_disclosures": check_required_disclosures(full_text),
        "evidence_package_present": check_evidence_package_present(),
    }
    all_violations = [v for vs in checks.values() for v in vs]
    return {
        "status": "pass" if not all_violations else "fail",
        "checks": checks,
        "total_violations": len(all_violations),
    }


def main() -> int:
    try:
        report = run_all_checks()
    except ManuscriptVerificationError as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
