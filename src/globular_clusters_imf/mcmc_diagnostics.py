"""MCMC convergence diagnostics.

`_compute_rhat` previously existed as three independent copies across the MCMC
driver scripts. They happened to agree, so nothing was broken -- but R-hat is the
gate that decides whether a chain is trustworthy enough to quote, and having the
gate defined in triplicate means it can drift without anyone noticing. One
definition, one test.
"""

from __future__ import annotations

import numpy as np


def compute_rhat(chains: np.ndarray) -> float:
    """Gelman-Rubin potential scale reduction factor.

    Parameters
    ----------
    chains
        Array of shape ``(n_chains, n_samples)`` holding one scalar parameter.

    Returns
    -------
    float
        R-hat, or NaN when it is undefined (fewer than 2 chains or 2 samples, or a
        zero within-chain variance). NaN is deliberate: silently returning 1.0 for a
        degenerate chain would report "converged" for a chain that never moved.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D (n_chains, n_samples), got shape {chains.shape}")

    n_chains, n_samples = chains.shape
    if n_chains < 2 or n_samples < 2:
        return float("nan")

    chain_means = np.mean(chains, axis=1)
    chain_variances = np.var(chains, axis=1, ddof=1)

    within = float(np.mean(chain_variances))
    between = float(n_samples * np.var(chain_means, ddof=1))
    if within <= 0.0:
        return float("nan")

    posterior_variance = ((n_samples - 1) / n_samples) * within + between / n_samples
    return float(np.sqrt(posterior_variance / within))


def rhat_by_parameter(samples: np.ndarray) -> np.ndarray:
    """R-hat for every parameter of a ``(n_chains, n_samples, n_parameters)`` array."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 3:
        raise ValueError(
            f"samples must be 3-D (n_chains, n_samples, n_parameters), got {samples.shape}"
        )
    return np.array(
        [compute_rhat(samples[:, :, index]) for index in range(samples.shape[2])],
        dtype=float,
    )
