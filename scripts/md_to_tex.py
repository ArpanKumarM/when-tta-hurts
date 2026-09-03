#!/usr/bin/env python3
"""Convert paper/manuscript.md -> paper/tmlr/body.tex (body only).

Handles exactly the Markdown subset this manuscript uses: ## headings,
**lead-in bold** / **inline bold**, *italic*, `code`, [@key] / [@k1; @k2]
citations, - bullet lists, blank-line paragraphs, one U+2212 minus. No
pipe tables exist in the source. The abstract is wrapped in an abstract
environment; the References section is dropped (handled by \\bibliography).

Usage: uv run python scripts/md_to_tex.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "paper" / "manuscript.md"
OUT = REPO / "paper" / "tmlr" / "body.tex"


def esc_text(s: str) -> str:
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("−", "$-$")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def esc_code(s: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_\allowbreak "), ("%", r"\%"),
                 ("#", r"\#"), ("&", r"\&"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
                 ("$", r"\$"), ("/", r"/\allowbreak ")]:
        s = s.replace(a, b)
    return s


PLACEHOLDERS: dict[str, str] = {}


def stash(tex: str) -> str:
    key = f"\x00{len(PLACEHOLDERS)}\x00"
    PLACEHOLDERS[key] = tex
    return key


# Textual "Author (year)" / "(Author, year)" mentions that also carry a
# redundant [@key] in the same paragraph: collapse to a single \citet /
# \citep so the work is cited once. Only single-key [@key] markers are
# touched; multi-key [@a; @b] markers are left alone.
_TEXTUAL_CITE = [
    (re.compile(r"\bMedeiros \(2026\)"), "medeiros2026tta", "t"),
    (re.compile(r"\bShanmugam et al\.? \(2021\)"), "shanmugam2021better", "t"),
    (re.compile(r"\bLyzhov et al\.? \(2020\)"), "lyzhov2020greedy", "t"),
    (re.compile(r"\bKim et al\.? \(2020\)"), "kim2020learning", "t"),
    (re.compile(r"\bSherkatghanad et al\.? \(2024\)"), "sherkatghanad2024baytta", "t"),
    (re.compile(r"\bDi Salvo et al\.? \(2024\)"), "disalvo2024medmnistc", "t"),
    (re.compile(r"\bSchneider et al\.? \(2020\)"), "schneider2020improving", "t"),
    (re.compile(r"\bAyhan and Berens \(2018\)"), "ayhan2018ttaug", "t"),
    (re.compile(r"\bKimura \(2024\)"), "kimura2024understanding", "t"),
    (re.compile(r"\(Wu (?:&|\\&) He, 2018\)"), "wu2018group", "p"),
    (re.compile(r"\(Wang et al\.?, 2021\)"), "wang2021tent", "p"),
]


def collapse_textual_cites(p: str) -> str:
    present = set(re.findall(r"\[@([A-Za-z0-9_]+)\]", p))  # single-key markers only
    for rx, key, kind in _TEXTUAL_CITE:
        if key not in present:
            continue
        if not rx.search(p):
            continue
        p = rx.sub(stash(rf"\cite{kind}{{{key}}}"), p, count=1)
        p = p.replace(f" [@{key}]", "", 1).replace(f"[@{key}]", "", 1)
        p = re.sub(r" +([.,;])", r"\1", p)   # space left before punctuation
        p = re.sub(r"  +", " ", p)            # doubled space
    return p


def inline(s: str) -> str:
    s = collapse_textual_cites(s)
    # citations first: [@k], [@k1; @k2; @k3]
    def cite(m: re.Match) -> str:
        keys = re.findall(r"@([A-Za-z0-9_]+)", m.group(0))
        return stash(r"\citep{" + ",".join(keys) + "}")

    s = re.sub(r"\[@[A-Za-z0-9_]+(?:;\s*@[A-Za-z0-9_]+)*\]", cite, s)
    # bare URLs -> \url{} (before escaping, so _ and / survive)
    s = re.sub(r"https?://[^\s)]+", lambda m: stash(r"\url{" + m.group(0).rstrip(".,;") + "}"), s)
    # inline code
    s = re.sub(r"`([^`]+)`", lambda m: stash(r"\texttt{" + esc_code(m.group(1)) + "}"), s)
    # bold / italic
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: stash(r"\textbf{" + esc_text(m.group(1)) + "}"), s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", lambda m: stash(r"\emph{" + esc_text(m.group(1)) + "}"), s)
    # escape the remaining plain text
    s = esc_text(s)
    # em dash
    s = s.replace("--", "---")
    # restore placeholders -- repeat to a fixed point so nested markers
    # (e.g. code inside bold) are all expanded, then verify none remain.
    for _ in range(6):
        if "\x00" not in s:
            break
        for k, v in PLACEHOLDERS.items():
            s = s.replace(k, v)
    if "\x00" in s:
        raise RuntimeError(f"unrestored placeholder in: {s[:200]!r}")
    return s


def convert() -> str:
    raw = SRC.read_text()
    # drop title block (everything before first "## ")
    raw = raw[raw.index("\n## "):]
    # drop the standalone "---" horizontal rules
    lines = [ln for ln in raw.splitlines() if ln.strip() != "---"]

    blocks: list[str] = []
    i = 0
    n = len(lines)
    cur: list[str] = []

    def flush_para():
        if cur:
            text = " ".join(x.strip() for x in cur).strip()
            if text:
                blocks.append(inline(text))
            cur.clear()

    in_refs = False
    while i < n:
        ln = lines[i]
        if ln.startswith("## "):
            flush_para()
            title = ln[3:].strip()
            if title.lower() == "references":
                in_refs = True
                i += 1
                continue
            in_refs = False
            if title.lower() == "abstract":
                blocks.append("@@ABSTRACT_BEGIN@@")
            elif title.lower() == "supplementary material outline":
                blocks.append(r"\section*{" + esc_text(title) + "}")
            else:
                blocks.append(r"\section{" + esc_text(title) + "}")
                if blocks[-2] == "@@ABSTRACT_BODY_END@@" if len(blocks) > 1 else False:
                    pass
            i += 1
            continue
        if in_refs:
            i += 1
            continue
        if ln.strip().startswith(("- ", "* ")):
            flush_para()
            items = []
            while i < n and lines[i].strip().startswith(("- ", "* ")):
                item = lines[i].strip()[2:]
                j = i + 1
                while j < n and lines[j].strip() and not lines[j].strip().startswith(("- ", "* ")) and not lines[j].startswith("## "):
                    item += " " + lines[j].strip()
                    j += 1
                items.append(r"  \item " + inline(item.strip()))
                i = j
            blocks.append("\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}")
            continue
        if ln.strip() == "":
            flush_para()
            i += 1
            continue
        cur.append(ln)
        i += 1
    flush_para()

    # insert figure floats just before the block that first mentions each
    def figure(path, num, caption, star=False):
        env = "figure*" if star else "figure"
        return (f"\\begin{{{env}}}[t]\n\\centering\n"
                f"\\includegraphics[width={'0.95\\textwidth' if star else '0.72\\textwidth'}]{{{path}}}\n"
                f"\\caption{{{caption}}}\n\\label{{fig:{num}}}\n\\end{{{env}}}")

    FIGS = [
        ("Figure 1", figure("figure_1_unmatched_policy_forest",
            "unmatched", "\\textbf{Preregistered within-cell evidence.} Per-cell $\\Delta$ accuracy "
            "(TTA $-$ clean) with paired-bootstrap 95\\% CIs for the 30 distinct unmatched-policy "
            "cells at $N{=}50$. Every interval lies below zero.", star=True)),
        ("Figure 2", figure("figure_2_matched_policy_mitigation",
            "matched", "\\textbf{Secondary fixed-model evidence (non-confirmatory).} Difference-in-"
            "differences of the TTA effect, matched- minus unmatched-policy model, same "
            "dataset/seed; all six intervals exclude zero.")),
        ("Figure 3", figure("figure_3_normalization_heterogeneity",
            "norm", "\\textbf{Secondary, non-confirmatory.} GroupNorm-minus-BatchNorm DiD by "
            "dataset; the sign reverses between PathMNIST and BloodMNIST.")),
        ("Figure 4", figure("figure_4_resolution_comparison",
            "res", "\\textbf{Secondary, non-confirmatory.} 64px-minus-28px DiD by dataset; higher "
            "resolution does not consistently reduce harm.", star=True)),
        ("Figure 5", figure("figure_5_block_c_positive_control",
            "blockc", "\\textbf{Preregistered positive control.} DermaMNIST / ResNet-18 per-seed "
            "$\\Delta$ accuracy; the source study's reported $+1.6$pp improvement did not "
            "reproduce.")),
        ("Figure 6", figure("fig6_scaling_curve",
            "scaling", "\\textbf{View-count scaling curve (preregistered secondary).} Mean $\\Delta$ "
            "accuracy vs.\\ TTA view count $N$; the shaded band is the min--max over the 30 "
            "unmatched cells. Harm is largest at $N{=}1$, flattens by $N\\approx25$, and never "
            "crosses zero.")),
        ("Figure 7", figure("fig7_component_decomposition",
            "component", "\\textbf{Per-augmentation-component decomposition (secondary / exploratory).} "
            "Geometric-only and intensity-only $\\Delta$ accuracy at $N{=}50$ on the 12 Block A 28px "
            "cells, with the mixed-policy value marked. Both families harm accuracy on their own in "
            "all 12 cells.")),
        ("Figure 8", figure("fig8_label_preservation",
            "labelpres", "\\textbf{Label-preservation audit (secondary / exploratory).} Human "
            "content-presence scores for 50 augmented views per dataset; the marker is the "
            "automated not-preserved rate, reported as a conservative upper bound the human "
            "check does not corroborate.")),
    ]
    for anchor, block in FIGS:
        for idx, b in enumerate(blocks):
            if anchor in b and not b.startswith("\\begin{figure"):
                blocks.insert(idx, block)
                break

    TABLES = [
        ("reproduced from the committed evidence package", r"""\begin{table}[t]
\centering
\small
\begin{tabular}{@{}p{2.6cm}p{3.4cm}p{2.4cm}p{4.4cm}@{}}
\toprule
\textbf{Evidence tier} & \textbf{Source} & \textbf{Cells / pairs} & \textbf{Confirmatory?} \\
\midrule
Preregistered within-cell & H1/H2/H3/BLOCK\_C & 39 unique cells & Yes --- clean-vs-TTA, within one trained model \\
Secondary fixed-model comparison & Cross-condition H1/H2/H3 & 30 pairs (12+12+6) & No --- post-validation / pre-test-specified, never preregistered \\
Descriptive summary & Preregistered seed-level groupings & 13 dataset $\times$ resolution $\times$ normalization groups & No --- non-inferential, no p-value / CI of its own \\
Unsupported / forbidden & H4; pooled or model-population verdicts; secondary significance labels & N/A & Never permitted anywhere in this package \\
\bottomrule
\end{tabular}
\caption{\textbf{Experimental-design and evidence-classification map}, referenced throughout Results and Discussion. Reproduced from the committed evidence package.}
\label{tab:design}
\end{table}"""),
        ("claim adjudication) summarizes", r"""\setcounter{table}{6}
\begin{table}[t]
\centering
\small
\begin{tabular}{@{}p{4.6cm}p{4.0cm}p{4.4cm}@{}}
\toprule
\textbf{Claim} & \textbf{Evidence tier} & \textbf{Status} \\
\midrule
Naive TTA harmed all 30 distinct unmatched-policy base cells & Preregistered within-cell & Supported \\
Matched-policy training mitigates TTA harm & Secondary fixed-model DiD, descriptively corroborated by separate within-cell patterns & Supported only secondarily / descriptively --- not a preregistered cross-condition test \\
Normalization changes the magnitude of harm & Secondary fixed-model DiD only & Supported only secondarily; direction is dataset-dependent \\
Higher resolution reduces TTA harm & Secondary fixed-model DiD only & Contradicted for BloodMNIST; mixed / near-null for PathMNIST \\
BLOCK\_C reproduces the source paper's positive TTA improvement & Preregistered within-cell (positive control) & Contradicted --- expected improvement not reproduced in any seed \\
Any H4 (Validation-Gated TTA) verdict & None --- no derivable family exists & Not made, anywhere \\
Any model-population or general medical-imaging generalization & None --- three seeds, fixed policy / budget & Not permitted, anywhere \\
\bottomrule
\end{tabular}
\caption{\textbf{Claim adjudication.} Each headline statement, the tier of evidence behind it, and its status. Reproduced from the committed evidence package.}
\label{tab:claims}
\end{table}"""),
    ]
    for anchor, block in TABLES:
        for idx, b in enumerate(blocks):
            if anchor in b and not b.startswith(("\\begin{table", "\\setcounter")):
                blocks.insert(idx, block)
                break

    # stitch abstract: text between @@ABSTRACT_BEGIN@@ and next \section
    out: list[str] = []
    k = 0
    while k < len(blocks):
        b = blocks[k]
        if b == "@@ABSTRACT_BEGIN@@":
            k += 1
            abs_parts = []
            while k < len(blocks) and not blocks[k].startswith("\\section"):
                abs_parts.append(blocks[k])
                k += 1
            out.append("\\begin{abstract}\n" + "\n\n".join(abs_parts) + "\n\\end{abstract}")
            continue
        out.append(b)
        k += 1
    return "\n\n".join(out) + "\n"


_ANON_CODE_PARA = (
    r"\textbf{Code and data.} All code, the sealed evidence package, and "
    r"the complete preregistration and audit trail will be released publicly "
    r"upon acceptance; the URLs are omitted here for double-blind review."
)


def anonymize(body: str) -> str:
    """Replace the Code-and-data paragraph (which carries identifying
    GitHub URLs) with a review-safe placeholder."""
    new, n = re.subn(
        r"\\textbf\{Code and data\.\}.*?(?=\n\n)",
        lambda _m: _ANON_CODE_PARA,
        body,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("anonymize(): Code-and-data paragraph not found / not unique")
    return new


if __name__ == "__main__":
    import sys

    body = convert()
    if "--anon" in sys.argv:
        out = REPO / "paper" / "tmlr" / "body-anon.tex"
        out.write_text(anonymize(body))
    else:
        out = OUT
        out.write_text(body)
    print(f"wrote {out}  ({len(out.read_text().splitlines())} lines)")
