"""Polish grid maxima into real maximum-likelihood estimates, with real errors.

The single-component IMF parameters were being read straight off a coarse
profile-likelihood grid: the paper quoted alpha = -1.675 and log10(Mc) = 6.350,
which are grid *nodes*, not maxima, and carried no uncertainties at all. Two
separate problems:

* the reported value is quantised to the grid step, so it is wrong by up to half
  a step for no reason -- the grid is a search device, not an estimator;
* with no interval attached, a reader cannot tell whether a difference between
  two families is meaningful.

This module fixes both. ``refine_from_grid_maximum`` restarts a continuous
optimiser from the best grid point, and ``profile_likelihood_interval`` brackets
where the profile deviance rises by the requested amount, which is the right
interval for a likelihood with nuisance parameters profiled out.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize

# Delta in the NEGATIVE log-likelihood defining an interval, for one interest
# parameter with the rest profiled out (Wilks): 0.5 -> 68.3%, 1.92 -> 95%.
DELTA_NLL_1SIGMA = 0.5
DELTA_NLL_95 = 1.9207


@dataclass
class RefinedParameter:
    name: str
    value: float
    lower: float
    upper: float
    minus: float
    plus: float
    delta_nll: float
    lower_is_bound: bool
    upper_is_bound: bool

    @property
    def is_railed(self) -> bool:
        """True if the interval ran into a parameter bound rather than closing."""
        return self.lower_is_bound or self.upper_is_bound


@dataclass
class RefinementResult:
    parameters: list[RefinedParameter]
    best_params: np.ndarray
    best_nll: float
    grid_params: np.ndarray
    grid_nll: float

    @property
    def nll_improvement_over_grid(self) -> float:
        """How much log-likelihood the grid was leaving on the table."""
        return float(self.grid_nll - self.best_nll)

    def as_rows(self) -> list[dict[str, object]]:
        return [
            {
                "parameter": p.name,
                "value": p.value,
                "minus": p.minus,
                "plus": p.plus,
                "lower": p.lower,
                "upper": p.upper,
                "delta_nll": p.delta_nll,
                "interval_railed_on_bound": p.is_railed,
            }
            for p in self.parameters
        ]


def refine_from_grid_maximum(
    negative_log_likelihood: Callable[[np.ndarray], float],
    grid_params: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    parameter_names: Sequence[str],
    delta_nll: float = DELTA_NLL_1SIGMA,
    compute_intervals: bool = True,
) -> RefinementResult:
    """Continuously optimise from a grid maximum and attach profile intervals.

    ``negative_log_likelihood`` must already have the nuisance parameters (radial
    shape, detectability) profiled out, so that varying one interest parameter and
    re-minimising gives a genuine profile.

    Nelder-Mead is used rather than a gradient method because the nested
    detectability iteration makes the objective only piecewise smooth: finite
    differences across it are unreliable.
    """
    grid_params = np.asarray(grid_params, dtype=float)
    bounds = [(float(lo), float(hi)) for lo, hi in bounds]
    grid_nll = float(negative_log_likelihood(grid_params))

    def bounded(params: np.ndarray) -> float:
        params = np.asarray(params, dtype=float)
        for value, (low, high) in zip(params, bounds, strict=True):
            if not (low <= value <= high):
                return 1.0e30
        return float(negative_log_likelihood(params))

    result = optimize.minimize(
        bounded,
        x0=grid_params,
        method="Nelder-Mead",
        options={"xatol": 1.0e-4, "fatol": 1.0e-6, "maxiter": 2000},
    )
    best_params = np.asarray(result.x, dtype=float)
    best_nll = float(result.fun)
    if best_nll > grid_nll:
        # The polish must never be worse than the point it started from.
        best_params, best_nll = grid_params, grid_nll

    parameters: list[RefinedParameter] = []
    if compute_intervals:
        for index, name in enumerate(parameter_names):
            parameters.append(
                profile_likelihood_interval(
                    negative_log_likelihood=negative_log_likelihood,
                    best_params=best_params,
                    best_nll=best_nll,
                    bounds=bounds,
                    index=index,
                    name=name,
                    delta_nll=delta_nll,
                )
            )

    return RefinementResult(
        parameters=parameters,
        best_params=best_params,
        best_nll=best_nll,
        grid_params=grid_params,
        grid_nll=grid_nll,
    )


def profile_likelihood_interval(
    negative_log_likelihood: Callable[[np.ndarray], float],
    best_params: np.ndarray,
    best_nll: float,
    bounds: Sequence[tuple[float, float]],
    index: int,
    name: str,
    delta_nll: float = DELTA_NLL_1SIGMA,
    max_expansion_steps: int = 40,
) -> RefinedParameter:
    """Interval where the profile NLL rises by ``delta_nll`` above its minimum.

    The interest parameter is held fixed and every other parameter is re-minimised
    at each trial value, which is what makes this a profile rather than a slice.
    """
    best_params = np.asarray(best_params, dtype=float)
    low_bound, high_bound = bounds[index]
    centre = float(best_params[index])
    other = [i for i in range(len(best_params)) if i != index]

    def profile_nll(value: float) -> float:
        if not (low_bound <= value <= high_bound):
            return np.inf
        if not other:
            params = best_params.copy()
            params[index] = value
            return float(negative_log_likelihood(params))

        def objective(free: np.ndarray) -> float:
            params = best_params.copy()
            params[index] = value
            params[other] = free
            for i, p in enumerate(params):
                lo, hi = bounds[i]
                if not (lo <= p <= hi):
                    return 1.0e30
            return float(negative_log_likelihood(params))

        result = optimize.minimize(
            objective,
            x0=best_params[other],
            method="Nelder-Mead",
            options={"xatol": 1.0e-4, "fatol": 1.0e-6, "maxiter": 1000},
        )
        return float(result.fun)

    def excess(value: float) -> float:
        return profile_nll(value) - best_nll - delta_nll

    scale = max(abs(centre) * 0.05, 1.0e-2)

    def find_edge(direction: int) -> tuple[float, bool]:
        limit = high_bound if direction > 0 else low_bound
        step = scale
        previous = centre
        for _ in range(max_expansion_steps):
            candidate = centre + direction * step
            if (direction > 0 and candidate >= limit) or (direction < 0 and candidate <= limit):
                # Ran out of parameter space before the likelihood dropped enough:
                # the interval is open, and saying so matters more than a number.
                if excess(limit) < 0.0:
                    return float(limit), True
                candidate = limit
            if excess(candidate) >= 0.0:
                lo, hi = (previous, candidate) if direction > 0 else (candidate, previous)
                edge = optimize.brentq(excess, lo, hi, xtol=1.0e-5)
                return float(edge), False
            previous = candidate
            step *= 1.6
        return float(previous), True

    lower, lower_railed = find_edge(-1)
    upper, upper_railed = find_edge(+1)
    return RefinedParameter(
        name=name,
        value=centre,
        lower=lower,
        upper=upper,
        minus=float(centre - lower),
        plus=float(upper - centre),
        delta_nll=float(delta_nll),
        lower_is_bound=lower_railed,
        upper_is_bound=upper_railed,
    )


def format_value_with_interval(parameter: RefinedParameter, precision: int = 3) -> str:
    """LaTeX-ready ``value^{+plus}_{-minus}``, flagged if it railed on a bound."""
    text = (
        f"{parameter.value:.{precision}f}"
        f"^{{+{parameter.plus:.{precision}f}}}"
        f"_{{-{parameter.minus:.{precision}f}}}"
    )
    if parameter.is_railed:
        text += r"\,\mathrm{(bound)}"
    return text
