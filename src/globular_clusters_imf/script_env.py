"""Shared entry-point setup for the analysis scripts.

62 of the 76 scripts opened with a byte-identical copy of the same six-line
matplotlib/cache preamble. One definition instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Preference order for the working catalogue: richest join first.
CATALOG_CANDIDATES = (
    "baumgardt_gc_catalog_with_origin_and_chemistry.csv",
    "baumgardt_gc_catalog_with_origin_flags.csv",
    "baumgardt_gc_catalog.csv",
)


def configure_headless_matplotlib(project_root: Path | None = None) -> Path:
    """Force a non-interactive backend and keep matplotlib's caches inside the repo."""
    root = project_root or PROJECT_ROOT
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache"))
    (root / ".mplconfig").mkdir(parents=True, exist_ok=True)
    (root / ".cache" / "fontconfig").mkdir(parents=True, exist_ok=True)
    return root


def load_working_catalog(project_root: Path | None = None) -> pd.DataFrame:
    """Load the most complete processed catalogue that exists."""
    root = project_root or PROJECT_ROOT
    processed = root / "data" / "processed"
    for name in CATALOG_CANDIDATES:
        path = processed / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"No processed catalogue found in {processed}. "
        "Run scripts/fetch_baumgardt_catalog.py first."
    )


def load_fit_sample(project_root: Path | None = None) -> pd.DataFrame:
    """Working catalogue with survival thresholds attached -- what the fitters want."""
    from .model import fit_catalog_models

    root = project_root or PROJECT_ROOT
    return fit_catalog_models(load_working_catalog(root), root)["catalog"]
