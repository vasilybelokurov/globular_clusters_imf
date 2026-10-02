"""The smooth survivability surface must be a converged fit, not one stuck on a bound."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from globular_clusters_imf.smooth_survivability import build_smooth_survivability_grid

CATALOG = Path(__file__).resolve().parents[1] / "data" / "processed" / "baumgardt_gc_catalog_with_origin_flags.csv"


@pytest.mark.parametrize("eta_t", [0.8, 1.25, 1.6])
def test_logistic_surface_fit_converges_inside_its_bounds(eta_t):
    grid = build_smooth_survivability_grid(pd.read_csv(CATALOG), eta_t=eta_t, surface_model="logistic")
    summary = grid["summary"]
    assert summary.optimization_success, summary.optimizer_message
    assert not any(grid["parameters_at_bound"]), grid["parameters_at_bound"]
    probability = np.asarray(grid["survival_probability"])
    assert np.all((probability >= 0.0) & (probability <= 1.0))
    # monotone in mass at every radius
    assert np.all(np.diff(probability, axis=0) >= -1.0e-12)
