"""Simulate from known parameters, then check the fitter gets them back.

This is the test that would have caught the mass-grid truncation on day one: with a
grid that stops inside the survival-positive region, the recovered alpha is biased
because the IMF normalisation has to absorb the missing probability.

The generative model is exactly the one the code claims to fit:

    lambda_obs(log M, log a) = N0 * phi(log M | theta) * A(log a | eta) * S(log M, a)

so a correct fitter must recover (theta, eta, N0) up to Poisson noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from globular_clusters_imf.joint_model import (
    JointLikelihoodContext,
    JointModelSpec,
    evaluate_imf_family,
    evaluate_radial_model,
    fit_single_joint_model,
)
from globular_clusters_imf.mass_grid import build_log_mass_grid

TRUE_ALPHA = -1.30
TRUE_LOG_MC = 6.40
TRUE_N0 = 40_000


def _make_survival_grid(log_mass_grid, log_a_grid, log_mass_cut_at_centre=5.6, slope=-0.9):
    """Smooth, analytic survival map: threshold falls with radius.

    Deliberately not the catalogue's kernel-smoothed staircase -- the point is to
    test the estimator against a known truth, not to re-test the survival model.
    """
    threshold = log_mass_cut_at_centre + slope * (log_a_grid - np.log10(4.0))
    width = 0.25
    return {
        "log_mass_grid": log_mass_grid,
        "log_a_grid": log_a_grid,
        "semi_major_axis_grid_kpc": np.power(10.0, log_a_grid),
        "survival_probability": 1.0 / (
            1.0 + np.exp(-(log_mass_grid[:, None] - threshold[None, :]) / width)
        ),
        "selection_offset_dex": 0.0,
        "bandwidth_log10_a_dex": 0.18,
        "log_mass_min": float(log_mass_grid[0]),
        "log_mass_max": float(log_mass_grid[-1]),
        "n_mass_grid": int(log_mass_grid.size),
        "mass_step_dex": float(log_mass_grid[1] - log_mass_grid[0]),
    }


def _simulate(spec, imf_params, radial_params, n0, survival_grid, seed):
    """Draw an inhomogeneous Poisson sample on the (log M, log a) grid."""
    rng = np.random.default_rng(seed)
    log_mass_grid = survival_grid["log_mass_grid"]
    log_a_grid = survival_grid["log_a_grid"]

    class _Ctx:
        pass

    stub = _Ctx()
    stub.log_a_grid = log_a_grid
    stub.log_a_data = log_a_grid
    stub.log_a_mean = float(np.mean(log_a_grid))
    stub.log_a_std = float(np.std(log_a_grid))
    stub.radial_step_edges = np.quantile(log_a_grid, np.linspace(0, 1, 6))

    imf_grid, _, _ = evaluate_imf_family(
        spec.imf_family, imf_params, log_mass_grid, log_mass_grid
    )
    radial_grid, _, _ = evaluate_radial_model(spec.radial_model, radial_params, stub)

    intensity = (
        n0 * imf_grid[:, None] * radial_grid[None, :] * survival_grid["survival_probability"]
    )
    d_mass = log_mass_grid[1] - log_mass_grid[0]
    d_a = log_a_grid[1] - log_a_grid[0]
    expected = intensity * d_mass * d_a
    counts = rng.poisson(expected)

    mass_idx, a_idx = np.nonzero(counts)
    log_masses, log_as = [], []
    for m, a in zip(mass_idx, a_idx):
        k = counts[m, a]
        log_masses.extend(rng.uniform(log_mass_grid[m] - d_mass / 2, log_mass_grid[m] + d_mass / 2, k))
        log_as.extend(rng.uniform(log_a_grid[a] - d_a / 2, log_a_grid[a] + d_a / 2, k))
    return np.asarray(log_masses), np.asarray(log_as)


def _context_from_sample(log_masses, log_as, survival_grid):
    import pandas as pd

    catalog = pd.DataFrame(
        {
            "log_initial_mass_msun": log_masses,
            "semi_major_axis_kpc": np.power(10.0, log_as),
        }
    )
    return JointLikelihoodContext.from_catalog_and_survival_grid(catalog, survival_grid)


@pytest.fixture(scope="module")
def truth():
    spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
    log_mass_grid = build_log_mass_grid(2.0, 8.0, 561)
    log_a_grid = np.linspace(np.log10(0.5), np.log10(60.0), 160)
    survival_grid = _make_survival_grid(log_mass_grid, log_a_grid)
    radial_params = np.array([-0.5, -0.3, 0.05])
    log_masses, log_as = _simulate(
        spec, np.array([TRUE_ALPHA, TRUE_LOG_MC]), radial_params,
        TRUE_N0, survival_grid, seed=7,
    )
    return spec, survival_grid, log_masses, log_as


def test_simulation_produces_a_sane_sample(truth):
    _, _, log_masses, _ = truth
    assert 100 < len(log_masses) < 5000
    assert np.all(np.isfinite(log_masses))


def test_recovers_the_imf_on_a_grid_that_covers_the_support(truth):
    """The estimator is unbiased when the integration domain is complete."""
    spec, survival_grid, log_masses, log_as = truth
    context = _context_from_sample(log_masses, log_as, survival_grid)
    payload = fit_single_joint_model(context=context, spec=spec)
    imf = payload["model"]["imf_parameters"]

    assert imf["alpha_dndm"] == pytest.approx(TRUE_ALPHA, abs=0.20), (
        f"alpha recovered as {imf['alpha_dndm']:.3f}, truth {TRUE_ALPHA}"
    )
    assert imf["log10_m_c_msun"] == pytest.approx(TRUE_LOG_MC, abs=0.25), (
        f"log Mc recovered as {imf['log10_m_c_msun']:.3f}, truth {TRUE_LOG_MC}"
    )


def test_recovers_n0_within_poisson_expectations(truth):
    spec, survival_grid, log_masses, log_as = truth
    context = _context_from_sample(log_masses, log_as, survival_grid)
    payload = fit_single_joint_model(context=context, spec=spec)
    n0 = payload["model"]["total_initial_count"]
    assert n0 == pytest.approx(TRUE_N0, rel=0.35), f"N0 recovered as {n0:.0f}, truth {TRUE_N0}"


def test_truncating_the_grid_inside_the_support_biases_the_imf(truth):
    """The bug, demonstrated on data whose truth we know.

    Rebuild the identical problem with the historical floor of log10(M)=3.5, which
    lies above part of the survival-positive region, and the recovered alpha must
    move away from the truth -- the IMF normalisation absorbs the missing
    probability. This is why the published alpha and N0 were what they were.
    """
    spec, survival_grid, log_masses, log_as = truth

    full_context = _context_from_sample(log_masses, log_as, survival_grid)
    full_alpha = fit_single_joint_model(context=full_context, spec=spec)["model"][
        "imf_parameters"
    ]["alpha_dndm"]

    floor = 3.5
    keep = survival_grid["log_mass_grid"] >= floor
    truncated = dict(survival_grid)
    truncated["log_mass_grid"] = survival_grid["log_mass_grid"][keep]
    truncated["survival_probability"] = survival_grid["survival_probability"][keep]
    # the survival map is still clearly non-zero at the new floor: that is the bug
    assert truncated["survival_probability"][0].max() > 0.01

    inside = log_masses >= floor
    trunc_context = _context_from_sample(log_masses[inside], log_as[inside], truncated)
    trunc_alpha = fit_single_joint_model(context=trunc_context, spec=spec)["model"][
        "imf_parameters"
    ]["alpha_dndm"]

    full_error = abs(full_alpha - TRUE_ALPHA)
    trunc_error = abs(trunc_alpha - TRUE_ALPHA)
    assert trunc_error > full_error, (
        "truncating the grid inside the survival-positive region should bias alpha; "
        f"got full={full_alpha:.3f} (err {full_error:.3f}), "
        f"truncated={trunc_alpha:.3f} (err {trunc_error:.3f}), truth={TRUE_ALPHA}"
    )
