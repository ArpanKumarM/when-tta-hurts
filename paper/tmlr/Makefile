# TMLR build for "When Test-Time Augmentation Hurts".
#
#   make          -> regenerate body.tex + figures + main.pdf
#   make pdf      -> just rebuild main.pdf from existing body.tex/figures
#   make body     -> regenerate body.tex from ../manuscript.md
#   make figures  -> regenerate the 3 post-review figures
#   make clean    -> remove LaTeX build artifacts
#
# Requires: TinyTeX/TeX Live on PATH; `uv` for the Python steps.

REPO := ../..
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all pdf body figures bib clean

all: body figures bib pdf

body:
	cd $(REPO) && uv run python scripts/md_to_tex.py

figures:
	cd $(REPO) && uv run --with matplotlib --with numpy python scripts/render_tmlr_primary_figures.py
	cd $(REPO) && uv run --with matplotlib --with numpy python scripts/generate_extension_figures.py

bib:
	cd $(REPO) && python3 -c "import re,pathlib; s=pathlib.Path('paper/references.bib').read_text(); pathlib.Path('paper/tmlr/references.bib').write_text(re.sub(r'(?<!\\\\)_', r'\\\\_', s))"

pdf:
	$(LATEXMK) main.tex

clean:
	latexmk -C >/dev/null 2>&1 || true
	rm -f main.bbl main.blg *.aux *.fls *.fdb_latexmk *.out *.log b.log lm.log build.log
