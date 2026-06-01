PYTHON ?= ../.venv/bin/python
LATEXMK ?= latexmk

.PHONY: all assets two-component-tables pdf full clean

all: pdf

assets:
	$(PYTHON) ../scripts/build_paper_assets_exact_single_component.py
	$(PYTHON) ../scripts/build_two_component_paper_tables.py

two-component-tables:
	$(PYTHON) ../scripts/build_two_component_paper_tables.py

pdf:
	$(LATEXMK) -pdf -bibtex -outdir=build main.tex

full: assets pdf

clean:
	$(LATEXMK) -C -outdir=build main.tex
	rm -rf build
