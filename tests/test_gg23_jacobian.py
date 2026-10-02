"""GG23 backend: mass definition and the change of variables in the likelihood.

When the catalogue initial masses are recomputed from the observed present-day masses at
every eta_t, the likelihood of the data must carry sum_i ln |d ln M_ini / d ln M_now|.
These tests pin the analytic Jacobian and show, on simulated data, that eta_t is biased
without it and recovered with it.
"""

import numpy as np
import pytest
from scipy import optimize

from globular_clusters_imf.gg23_survivability import (
    GG23_MODELS,
    GG23_STELLAR_EVOLUTION_MASS_FRACTION,
    build_raw_gg23_survival_grid_from_catalog,
    gg23_birth_mass_from_present_msun,
    gg23_birth_survival_mass_cut_msun,
    gg23_dlog_initial_dlog_present,
    gg23_initial_mass_from_present_msun,
    gg23_present_mass_msun,
    gg23_survival_mass_cut_msun,
)

AGE_GYR = 12.0


@pytest.mark.parametrize("model_name", sorted(GG23_MODELS))
@pytest.mark.parametrize("eta_t", [0.4, 1.0, 2.5])
def test_analytic_jacobian_matches_finite_difference(model_name, eta_t):
    model = GG23_MODELS[model_name]
    radius = np.array([1.5, 4.0, 9.0, 25.0])
    present = np.array([3.0e4, 1.0e5, 4.0e5, 2.0e4])
    initial = gg23_initial_mass_from_present_msun(
        present, radius, model, gradient_radius_kpc=radius, age_gyr=AGE_GYR, eta_t=eta_t
    )
    analytic = gg23_dlog_initial_dlog_present(
        initial, radius, model, gradient_radius_kpc=radius, age_gyr=AGE_GYR, eta_t=eta_t
    )
    # Differentiate the exact forward map M_now(M_i) rather than the bisection inverse,
    # which is ill-conditioned for clusters close to their disruption threshold.
    step = 1.0e-6
    up = gg23_present_mass_msun(initial * np.exp(step), radius, model, gradient_radius_kpc=radius, age_gyr=AGE_GYR, eta_t=eta_t)
    down = gg23_present_mass_msun(initial * np.exp(-step), radius, model, gradient_radius_kpc=radius, age_gyr=AGE_GYR, eta_t=eta_t)
    numeric = (2.0 * step) / (np.log(up) - np.log(down))
    np.testing.assert_allclose(analytic, numeric, rtol=1.0e-5)
    assert np.all((analytic > 0.0) & (analytic <= 1.0))


def test_birth_mass_is_gg23_mass_over_mu_sev():
    model = GG23_MODELS["gg23_bh"]
    radius = np.array([3.0, 8.0])
    present = np.array([1.0e5, 3.0e5])
    gg23 = gg23_initial_mass_from_present_msun(present, radius, model, gradient_radius_kpc=radius)
    birth = gg23_birth_mass_from_present_msun(present, radius, model, gradient_radius_kpc=radius)
    np.testing.assert_allclose(birth, gg23 / 0.55, rtol=1.0e-12)
    np.testing.assert_allclose(
        gg23_birth_survival_mass_cut_msun(radius, model, gradient_radius_kpc=radius),
        gg23_survival_mass_cut_msun(radius, model, gradient_radius_kpc=radius) / 0.55,
        rtol=1.0e-12,
    )
    assert GG23_STELLAR_EVOLUTION_MASS_FRACTION == 0.55


def test_gg23_survival_grid_is_on_the_birth_mass_axis():
    import pandas as pd

    catalog = pd.DataFrame({"semi_major_axis_kpc": [2.0, 6.0, 20.0], "eccentricity": [0.5, 0.3, 0.6]})
    grid = build_raw_gg23_survival_grid_from_catalog(catalog, GG23_MODELS["gg23_no_bh"], eta_t=1.0)
    radius = grid["catalog"]["gg23_effective_radius_kpc"].to_numpy()
    expected = gg23_survival_mass_cut_msun(
        radius, GG23_MODELS["gg23_no_bh"], gradient_radius_kpc=catalog["semi_major_axis_kpc"].to_numpy()
    ) / 0.55
    np.testing.assert_allclose(grid["catalog"]["gg23_survival_mass_cut_msun"].to_numpy(), expected, rtol=1.0e-12)


def _simulate_survivors(model, eta_true, n_birth, seed):
    """Birth masses from a fixed Schechter in log M_i, one orbit, GG23 evolution to 12 Gyr."""
    rng = np.random.default_rng(seed)
    radius = 4.0
    alpha, log_mc = -2.0, 6.3
    grid = np.linspace(3.0, 7.5, 4000)
    density = np.power(10.0, (alpha + 1.0) * grid) * np.exp(-np.power(10.0, grid - log_mc))
    cdf = np.cumsum(density)
    cdf /= cdf[-1]
    log_initial = np.interp(rng.uniform(size=n_birth), cdf, grid)
    present = gg23_present_mass_msun(10.0**log_initial, radius, model, age_gyr=AGE_GYR, eta_t=eta_true)
    return present[present > 0.0], radius, alpha, log_mc


def _log_likelihood(present, radius, model, eta_t, alpha, log_mc, with_jacobian):
    """Exact likelihood of observed present masses for a known birth IMF (per dex)."""
    initial = gg23_initial_mass_from_present_msun(present, radius, model, age_gyr=AGE_GYR, eta_t=eta_t)
    if not np.all(np.isfinite(initial)):
        return -np.inf
    log_m = np.log10(initial)
    grid = np.linspace(3.0, 7.5, 20000)
    density = np.power(10.0, (alpha + 1.0) * grid) * np.exp(-np.power(10.0, grid - log_mc))
    norm = np.trapezoid(density, grid)
    cut = np.log10(gg23_survival_mass_cut_msun(radius, model, age_gyr=AGE_GYR, eta_t=eta_t))
    surviving = np.trapezoid(np.where(grid >= cut, density, 0.0), grid) / norm
    log_phi = (alpha + 1.0) * log_m * np.log(10.0) - np.power(10.0, log_m - log_mc) - np.log(norm)
    value = float(np.sum(log_phi) - len(present) * np.log(surviving))
    if with_jacobian:
        jac = gg23_dlog_initial_dlog_present(initial, radius, model, age_gyr=AGE_GYR, eta_t=eta_t)
        value += float(np.sum(np.log(jac)))
    return value


@pytest.mark.parametrize("model_name", ["gg23_no_bh", "gg23_bh"])
def test_jacobian_removes_eta_bias_on_simulated_data(model_name):
    model = GG23_MODELS[model_name]
    eta_true = 1.0
    present, radius, alpha, log_mc = _simulate_survivors(model, eta_true, n_birth=60000, seed=7)

    log_eta_grid = np.linspace(np.log(0.4), np.log(2.5), 201)

    def profile(with_jacobian):
        return np.array(
            [_log_likelihood(present, radius, model, np.exp(x), alpha, log_mc, with_jacobian) for x in log_eta_grid]
        )

    with_jac = profile(True)
    without_jac = profile(False)
    best = int(np.argmax(with_jac))
    inside = log_eta_grid[with_jac > with_jac[best] - 0.5]
    sigma_log_eta = 0.5 * (inside.max() - inside.min())
    eta_with = float(np.exp(log_eta_grid[best]))
    eta_without = float(np.exp(log_eta_grid[int(np.argmax(without_jac))]))
    # With the Jacobian the true eta is recovered within 3 sigma of the likelihood width;
    # without it the estimate is many sigma off (it runs to the edge of the scan).
    assert abs(np.log(eta_with / eta_true)) < 3.0 * sigma_log_eta, (eta_with, sigma_log_eta)
    assert abs(np.log(eta_without / eta_true)) > 5.0 * sigma_log_eta, (eta_without, sigma_log_eta)
