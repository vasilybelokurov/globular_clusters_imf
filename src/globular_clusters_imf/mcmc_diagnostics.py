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


def compute_split_rhat(chains: np.ndarray) -> float:
    """Split-R-hat: each chain is halved before R-hat, so within-chain drift is caught.

    Gelman et al. (2013), Bayesian Data Analysis, 3rd ed., sec. 11.4.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D (n_chains, n_samples), got shape {chains.shape}")
    half = chains.shape[1] // 2
    if half < 2:
        return float("nan")
    return compute_rhat(np.vstack([chains[:, :half], chains[:, half : 2 * half]]))


def effective_sample_size(chains: np.ndarray) -> float:
    """Multi-chain effective sample size (Geyer initial monotone sequence).

    Autocorrelations are combined across chains as in Gelman et al. (2013, sec. 11.5):
    rho_t = 1 - (W - mean_c acov_c(t)) / var_plus. Pairs of lags are summed until the
    first non-positive pair sum, and the pair sums are forced to be non-increasing.

    Parameters
    ----------
    chains
        Array of shape ``(n_chains, n_samples)`` holding one scalar parameter.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError(f"chains must be 2-D (n_chains, n_samples), got shape {chains.shape}")
    n_chains, n_samples = chains.shape
    if n_samples < 4:
        return float("nan")
    centred = chains - chains.mean(axis=1, keepdims=True)
    n_fft = 1 << int(np.ceil(np.log2(2 * n_samples)))
    spectrum = np.fft.rfft(centred, n=n_fft, axis=1)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), n=n_fft, axis=1)[:, :n_samples] / n_samples
    within = float(np.mean(chains.var(axis=1, ddof=1)))
    if within <= 0.0:
        return float("nan")
    chain_means = chains.mean(axis=1)
    between_over_n = float(np.var(chain_means, ddof=1)) if n_chains > 1 else 0.0
    var_plus = (n_samples - 1) / n_samples * within + between_over_n
    rho = 1.0 - (within - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0

    pair_sums = []
    for t in range(0, n_samples - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0.0:
            break
        pair_sums.append(pair)
    pair_sums = np.minimum.accumulate(np.asarray(pair_sums, dtype=float))
    tau = -1.0 + 2.0 * float(np.sum(pair_sums))
    return float(n_chains * n_samples / max(tau, 1.0 / np.log10(max(n_chains * n_samples, 10))))
