# TMLR / arXiv build for "When Test-Time Augmentation Hurts".
#
#   make            -> figures + both body files + both PDFs
#   make tmlr       -> anonymized main.pdf (TMLR / OpenReview submission)
#   make arxiv      -> named main-preprint.pdf (arXiv)
#   make body       -> regenerate body.tex and body-anon.tex from ../manuscript.md
#   make figures    -> regenerate all 8 figures
#   make tarballs   -> arxiv-submission.tar.gz + tmlr-submission.tar.gz
#   make clean      -> remove LaTeX build artifacts
#
# Requires: TinyTeX/TeX Live on PATH; `uv` for the Python steps.

REPO := ../..
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all tmlr arxiv body figures bib tarballs clean

all: body figures bib tmlr arxiv

body:
	cd $(REPO) && uv run python scripts/md_to_tex.py
	cd $(REPO) && uv run python scripts/md_to_tex.py --anon

figures:
	cd $(REPO) && uv run --with matplotlib --with numpy python scripts/render_tmlr_primary_figures.py
	cd $(REPO) && uv run --with matplotlib --with numpy python scripts/generate_extension_figures.py

bib:
	cd $(REPO) && python3 -c "import re,pathlib; s=pathlib.Path('paper/references.bib').read_text(); pathlib.Path('paper/tmlr/references.bib').write_text(re.sub(r'(?<!\\\\)_', r'\\\\_', s))"

tmlr:
	$(LATEXMK) main.tex

arxiv:
	$(LATEXMK) main-preprint.tex

STYLE := tmlr.sty tmlr.bst fancyhdr.sty math_commands.tex references.bib
FIGS  := figures/figure_1_unmatched_policy_forest.pdf figures/figure_2_matched_policy_mitigation.pdf \
         figures/figure_3_normalization_heterogeneity.pdf figures/figure_4_resolution_comparison.pdf \
         figures/figure_5_block_c_positive_control.pdf figures/fig6_scaling_curve.pdf \
         figures/fig7_component_decomposition.pdf figures/fig8_label_preservation.pdf

tarballs: all
	rm -rf _pkg && mkdir -p _pkg/figures
	# arXiv: named build, includes .bbl (arXiv does run bibtex, but ship it to be safe)
	cp main-preprint.tex body.tex $(STYLE) _pkg/
	cp $(FIGS) _pkg/figures/
	cp main-preprint.bbl _pkg/ 2>/dev/null || true
	cd _pkg && tar czf ../arxiv-submission.tar.gz . && cd ..
	# TMLR / OpenReview: anonymized build
	rm -rf _pkg && mkdir -p _pkg/figures
	cp main.tex body-anon.tex $(STYLE) _pkg/
	cp $(FIGS) _pkg/figures/
	cp main.bbl _pkg/ 2>/dev/null || true
	cd _pkg && tar czf ../tmlr-submission.tar.gz . && cd ..
	rm -rf _pkg
	@echo "wrote arxiv-submission.tar.gz and tmlr-submission.tar.gz"

clean:
	latexmk -C >/dev/null 2>&1 || true
	latexmk -C main-preprint.tex >/dev/null 2>&1 || true
	rm -f *.bbl *.blg *.aux *.fls *.fdb_latexmk *.out *.log *.tmp b.log lm.log build.log f.log g.log h.log final.log
	rm -rf _pkg
