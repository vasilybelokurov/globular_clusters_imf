"""Shared fixtures.

The catalogue-backed fixtures are session-scoped because `fit_catalog_models`
solves a Brent root per cluster and is slow enough to dominate the suite
otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def scratch_root(tmp_path_factory) -> Path:
    """Somewhere for the fitters to dump their CSVs without touching outputs/."""
    return tmp_path_factory.mktemp("gcimf_outputs")


@pytest.fixture(scope="session")
def raw_catalog() -> pd.DataFrame:
    if not CATALOG_PATH.exists():
        pytest.skip(f"Processed catalogue not present at {CATALOG_PATH}")
    return pd.read_csv(CATALOG_PATH)


@pytest.fixture(scope="session")
def fit_sample(raw_catalog: pd.DataFrame, scratch_root: Path) -> pd.DataFrame:
    """Catalogue with survival thresholds attached -- the input every fitter wants."""
    from globular_clusters_imf.model import fit_catalog_models

    return fit_catalog_models(raw_catalog, scratch_root)["catalog"]


@pytest.fixture(scope="session")
def selection_offset(fit_sample: pd.DataFrame) -> float:
    from globular_clusters_imf.joint_model import calibrate_fixed_selection_offset_dex

    return calibrate_fixed_selection_offset_dex(fit_sample)


@pytest.fixture(scope="session")
def survival_grid(fit_sample: pd.DataFrame, selection_offset: float) -> dict:
    from globular_clusters_imf.joint_model import build_fixed_survival_grid

    return build_fixed_survival_grid(fit_sample, selection_offset_dex=selection_offset)


@pytest.fixture(scope="session")
def context(fit_sample: pd.DataFrame, survival_grid: dict):
    from globular_clusters_imf.joint_model import JointLikelihoodContext

    return JointLikelihoodContext.from_catalog_and_survival_grid(fit_sample, survival_grid)
