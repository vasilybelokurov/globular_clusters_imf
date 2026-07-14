"""Shared, merge-safe access to ``paper/tables/paper_results_summary.json``.

Three different scripts write this file:

* ``paper_assets.build_paper_assets``            -> ``single_component_best_model``, ...
* ``build_paper_assets_exact_single_component``  -> ``single_component_exact_global_*``, ...
* ``update_split_alpha_two_component_paper_assets``

Each used ``write_text(json.dumps(payload))``, which **clobbers** the file. So whichever
script ran last erased the other two schemas, and readers that wanted a key from a
different writer raised a bare ``KeyError``. The pipeline therefore only worked in one
particular (undeclared) order, and could not be rebuilt from a clean tree at all -- the
paper assets only regenerated because stale output from an earlier code generation
happened to still be on disk.

The fix is that these writers describe *disjoint* parts of one document, so they should
merge into it rather than replace it. ``merge_paper_results_summary`` does a
read-modify-write; ``require_paper_summary_keys`` turns a missing key into an error that
says which script to run instead of a bare KeyError.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

SUMMARY_FILENAME = "paper_results_summary.json"

# Which script is responsible for each top-level key, so a missing one can be
# reported as "run this" rather than a bare KeyError.
KEY_PROVENANCE: dict[str, str] = {
    "single_component_best_model": "scripts/build_paper_assets.py",
    "single_component_posterior_summary": "scripts/build_paper_assets_exact_single_component.py",
    "single_component_exact_global_logpoly3": "scripts/build_paper_assets_exact_single_component.py",
    "single_component_exact_global_step5": "scripts/build_paper_assets_exact_single_component.py",
    "single_component_exact_powerlaw_a": "scripts/build_paper_assets_exact_single_component.py",
    "single_component_exact_cored_powerlaw_a": "scripts/build_paper_assets_exact_single_component.py",
    "split_alpha_two_component": "scripts/update_split_alpha_two_component_paper_assets.py",
}


class PaperSummaryKeyError(KeyError):
    """A required key is absent because its producing script has not been run."""


def paper_summary_path(tables_dir: Path) -> Path:
    return Path(tables_dir) / SUMMARY_FILENAME


def read_paper_results_summary(tables_dir: Path) -> dict[str, object]:
    """Load the summary, or an empty document if it does not exist yet."""
    path = paper_summary_path(tables_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def merge_paper_results_summary(tables_dir: Path, payload: dict[str, object]) -> dict[str, object]:
    """Merge ``payload`` into the summary, preserving keys written by other scripts.

    Use this instead of ``write_text(json.dumps(payload))``: a plain write destroys the
    contributions of the other two writers and silently breaks their consumers.
    """
    merged = read_paper_results_summary(tables_dir)
    merged.update(payload)
    path = paper_summary_path(tables_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2))
    return merged


MACROS_FILENAME = "paper_numbers.tex"

MACRO_PROVENANCE: dict[str, str] = {
    "Exact": "scripts/build_paper_assets_exact_single_component.py",
    "Posterior": "scripts/build_paper_assets_exact_single_component.py",
    "StepFive": "scripts/build_paper_assets_exact_single_component.py",
    "PowerLawA": "scripts/build_paper_assets_exact_single_component.py",
    "CoredPowerLawA": "scripts/build_paper_assets_exact_single_component.py",
}

_MACRO_RE = re.compile(r"\\providecommand\{\\(\w+)\}\{(.*?)\}\s*$")


def read_latex_macros(path: Path) -> dict[str, str]:
    """Parse ``\\providecommand{\\Name}{value}`` lines into a dict."""
    if not path.exists():
        return {}
    macros: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = _MACRO_RE.match(line.strip())
        if match:
            macros[match.group(1)] = match.group(2)
    return macros


def merge_latex_macros(path: Path, macros: dict[str, str]) -> dict[str, str]:
    """Merge macro definitions into ``paper_numbers.tex`` instead of overwriting it.

    Four scripts write this file, and they emit *disjoint* macro sets:
    `build_paper_assets_exact_single_component` produces the `Exact*`/`Posterior*`/
    `StepFiveDeltaBIC`/... macros that `main.tex` actually cites, while
    `paper_assets.write_summary_macros_tex` (used by three other scripts) produces a
    different set entirely.

    Each used `write_text`, so whichever ran last erased the others. Running the paper
    build in the "wrong" order left 28 macros cited by `main.tex` undefined and the
    manuscript would not compile -- the same clobbering bug as
    `paper_results_summary.json`, in a second file.
    """
    merged = read_latex_macros(path)
    merged.update({str(k): str(v) for k, v in macros.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by the paper asset scripts. Do not edit numerical values by hand.",
        "% Written by MERGE: several scripts contribute disjoint macros to this file.",
    ]
    lines += [f"\\providecommand{{\\{name}}}{{{value}}}" for name, value in sorted(merged.items())]
    path.write_text("\n".join(lines) + "\n")
    return merged


def check_macros_cover_manuscript(main_tex: Path, macros_paths: Sequence[Path]) -> list[str]:
    """Return macros cited by main.tex that no generated file defines.

    A non-empty result means the manuscript will not compile.
    """
    cited = set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", main_tex.read_text()))
    defined: set[str] = set()
    for path in macros_paths:
        defined |= set(read_latex_macros(Path(path)))
    prefixes = ("Exact", "Posterior", "TwoComp", "Shared", "SplitAlpha", "StepFive",
                "PowerLawA", "CoredPowerLawA", "Single", "Detectability", "Blind")
    return sorted(
        name for name in cited - defined if name.startswith(prefixes)
    )


def require_paper_summary_keys(summary: dict[str, object], *keys: str) -> None:
    """Raise a message that names the script to run, rather than a bare KeyError."""
    missing = [key for key in keys if key not in summary]
    if not missing:
        return
    lines = [
        "paper_results_summary.json is missing required key(s): "
        + ", ".join(repr(key) for key in missing),
        "",
        "This file is written by several scripts, each contributing different keys.",
        "Run the producer(s) below, then retry:",
    ]
    producers = {KEY_PROVENANCE.get(key, "(unknown producer)") for key in missing}
    lines.extend(f"  python {producer}" for producer in sorted(producers))
    lines.append("")
    lines.append(f"Keys currently present: {sorted(summary)}")
    raise PaperSummaryKeyError("\n".join(lines))
