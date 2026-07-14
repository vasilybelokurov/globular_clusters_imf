"""main.tex must never cite a macro that no generated file defines.

Regression suite for a bug I introduced and then had to dig out: `paper_numbers.tex` has
FOUR writers emitting two disjoint macro sets, each doing `write_text` -- so whichever ran
last erased the others. Running the paper build in a different order silently dropped the
28 `Exact*`/`Posterior*` macros that `main.tex` actually cites, leaving the manuscript
uncompilable. Exactly the same clobbering bug as `paper_results_summary.json`.

This test is the tripwire: if any writer ever goes back to overwriting, the manuscript's
citations stop resolving and this fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from globular_clusters_imf.paper_summary import (
    check_macros_cover_manuscript,
    merge_latex_macros,
    read_latex_macros,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_TEX = PROJECT_ROOT / "paper" / "main.tex"
MACRO_FILES = [
    PROJECT_ROOT / "paper" / "tables" / "paper_numbers.tex",
    PROJECT_ROOT / "paper" / "tables" / "two_component_numbers.tex",
]


@pytest.mark.skipif(not MAIN_TEX.exists(), reason="manuscript not present")
def test_every_macro_cited_by_the_manuscript_is_defined():
    missing = check_macros_cover_manuscript(MAIN_TEX, MACRO_FILES)
    assert not missing, (
        "main.tex cites macros that no generated file defines, so the paper will not "
        f"compile: {missing}. A paper-asset writer has clobbered paper_numbers.tex "
        "instead of merging into it."
    )


@pytest.mark.skipif(not MACRO_FILES[0].exists(), reason="paper_numbers.tex not present")
def test_the_headline_macros_the_abstract_uses_are_present():
    """The abstract quotes the profiled-MCMC posterior. Those specific macros must exist."""
    macros = read_latex_macros(MACRO_FILES[0])
    for name in (
        "ExactBestEta", "ExactBestAlpha", "ExactBestMc", "ExactCountRatio",
        "PosteriorEtaMed", "PosteriorAlphaMed", "PosteriorMcMed",
        "PosteriorNzeroMed", "PosteriorMassZeroEightMed",
    ):
        assert name in macros, f"the abstract cites \\{name} but it is not defined"


class TestMergeSemantics:
    def test_a_second_writer_cannot_erase_the_first(self, tmp_path):
        path = tmp_path / "paper_numbers.tex"
        merge_latex_macros(path, {"ExactBestAlpha": "-1.169", "PosteriorNzeroMed": "1722"})
        merge_latex_macros(path, {"SingleComponentNzero": "3485.4"})
        macros = read_latex_macros(path)
        assert macros["ExactBestAlpha"] == "-1.169"      # survived the second write
        assert macros["PosteriorNzeroMed"] == "1722"
        assert macros["SingleComponentNzero"] == "3485.4"

    def test_rewriting_a_macro_updates_it(self, tmp_path):
        path = tmp_path / "paper_numbers.tex"
        merge_latex_macros(path, {"ExactBestAlpha": "-1.169"})
        merge_latex_macros(path, {"ExactBestAlpha": "-1.200"})
        assert read_latex_macros(path)["ExactBestAlpha"] == "-1.200"

    def test_round_trip(self, tmp_path):
        path = tmp_path / "paper_numbers.tex"
        payload = {"A": "1.0", "B": "-2.5", "C": "1.14\\times10^{5}"}
        merge_latex_macros(path, payload)
        assert read_latex_macros(path) == payload
