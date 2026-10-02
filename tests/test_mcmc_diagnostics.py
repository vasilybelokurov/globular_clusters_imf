"""ESS and split-R-hat must recover known values on synthetic chains."""

import numpy as np
import pytest

from globular_clusters_imf.mcmc_diagnostics import compute_split_rhat, effective_sample_size


def _ar1(rho, n_chains, n_samples, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros((n_chains, n_samples))
    x[:, 0] = rng.normal(size=n_chains) / np.sqrt(1 - rho**2)
    for t in range(1, n_samples):
        x[:, t] = rho * x[:, t - 1] + rng.normal(size=n_chains)
    return x


def test_ess_of_independent_draws_is_close_to_n():
    chains = np.random.default_rng(1).normal(size=(6, 2000))
    assert effective_sample_size(chains) == pytest.approx(12000, rel=0.1)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_ess_of_ar1_matches_analytic_value(rho):
    chains = _ar1(rho, 6, 4000, seed=2)
    expected = chains.size * (1 - rho) / (1 + rho)
    assert effective_sample_size(chains) == pytest.approx(expected, rel=0.2)


def test_split_rhat_flags_a_drifting_chain():
    rng = np.random.default_rng(3)
    good = rng.normal(size=(6, 1000))
    assert compute_split_rhat(good) < 1.01
    drifting = good + np.linspace(0.0, 3.0, 1000)[None, :]
    assert compute_split_rhat(drifting) > 1.1
