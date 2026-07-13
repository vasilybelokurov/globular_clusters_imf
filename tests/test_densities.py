"""Densities must integrate to one, in the base measure they claim to be in.

Regression suite for the base-measure bug: `fit_truncated_lognormal` returned a
density in log10(M) and `fit_truncated_powerlaw` one in M, and their
log-likelihoods were written side by side into model_summary.json. The 2337-nat
Jacobian sum(log(M ln10)) made the lognormal look 2323 nats better when in fact
the power law is 13.6 nats better. A change of variables was being read as evidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from globular_clusters_imf.joint_model import (
    evaluate_imf_family,
    evaluate_radial_model,
    normalize_density_on_grid,
    piecewise_constant_density,
)
from globular_clusters_imf.model import (
    LOG_TEN,
    fit_truncated_lognormal,
    fit_truncated_powerlaw,
    powerlaw_integral,
)

IMF_CASES = [
    ("lognormal", np.array([5.85, np.log(0.60)])),
    ("powerlaw", np.array([-2.0])),
    ("powerlaw_m2", np.array([])),
    ("schechter", np.array([-1.675, 6.35])),
    ("logspline6", np.array([0.8, 0.4, 0.0, -0.4, -0.8])),
]


@pytest.mark.parametrize("family,params", IMF_CASES, ids=[c[0] for c in IMF_CASES])
def test_imf_density_integrates_to_one_per_dex(context, family, params):
    density_grid, _, _ = evaluate_imf_family(
        family, params, context.log_mass_grid, context.log_mass_data
    )
    integral = np.trapezoid(density_grid, context.log_mass_grid)
    assert integral == pytest.approx(1.0, rel=1.0e-3), (
        f"{family} IMF integrates to {integral:.6f} per dex, not 1"
    )


@pytest.mark.parametrize("radial_model,params", [
    ("step5", np.zeros(4)),
    ("logpoly3", np.array([-0.4, -0.8, 0.2])),
    ("powerlaw_a", np.array([-1.0])),
    ("cored_powerlaw_a", np.array([-1.0, -0.5])),
])
def test_radial_density_integrates_to_one_per_dex(context, radial_model, params):
    density_grid, _, _ = evaluate_radial_model(radial_model, params, context)
    integral = np.trapezoid(density_grid, context.log_a_grid)
    assert integral == pytest.approx(1.0, rel=2.0e-2), (
        f"{radial_model} radial density integrates to {integral:.6f} per dex, not 1"
    )


def test_piecewise_constant_density_integrates_to_one():
    edges = np.array([-1.0, -0.4, 0.1, 0.5, 1.0, 1.8])
    weights = np.array([0.1, 0.3, 0.2, 0.25, 0.15])
    grid = np.linspace(edges[0], edges[-1], 20001)
    density = piecewise_constant_density(grid, edges=edges, weights=weights)
    assert np.trapezoid(density, grid) == pytest.approx(1.0, rel=1.0e-3)


def test_normalize_density_on_grid_actually_normalises():
    grid = np.linspace(0.0, 4.0, 501)
    density = normalize_density_on_grid(np.exp(-grid), grid)
    assert np.trapezoid(density, grid) == pytest.approx(1.0, rel=1.0e-6)


def test_powerlaw_integral_matches_closed_form():
    # int_1^10 M^-2 dM = [-1/M] = 1 - 0.1 = 0.9
    assert powerlaw_integral(1.0, 10.0, beta=-2.0) == pytest.approx(0.9, rel=1.0e-9)
    # beta = -1 is the log branch
    assert powerlaw_integral(1.0, np.e, beta=-1.0) == pytest.approx(1.0, rel=1.0e-9)
    assert powerlaw_integral(5.0, 5.0, beta=-2.0) == 0.0


def test_lognormal_and_powerlaw_likelihoods_are_in_the_same_base_measure(fit_sample):
    """The two families must be comparable, and the Jacobian must be the whole gap."""
    lognormal = fit_truncated_lognormal(fit_sample)
    powerlaw = fit_truncated_powerlaw(fit_sample, mass_min_msun=1.0e3, mass_max_msun=1.0e8)

    assert lognormal.base_measure == "log10_msun"
    assert powerlaw.base_measure == "msun"

    # The lognormal is natively per-dex, so nothing should have moved.
    assert lognormal.log_likelihood_per_dex == pytest.approx(
        lognormal.log_likelihood_native_measure
    )

    # The power law must have been converted, by exactly the Jacobian.
    masses = fit_sample["initial_mass_msun"].to_numpy()
    jacobian = float(np.sum(np.log(masses * LOG_TEN)))
    assert powerlaw.log_likelihood_per_dex - powerlaw.log_likelihood_native_measure == pytest.approx(
        jacobian, rel=1.0e-9
    )

    # And the conversion must be large enough to have mattered: it is what flipped
    # the sign of the comparison.
    assert jacobian > 1000.0
    native_gap = lognormal.log_likelihood_native_measure - powerlaw.log_likelihood_native_measure
    per_dex_gap = lognormal.log_likelihood_per_dex - powerlaw.log_likelihood_per_dex
    assert native_gap > 0, "raw numbers used to favour the lognormal"
    assert per_dex_gap < 0, "once comparable, the power law should win"
