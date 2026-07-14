"""Standalone reproduction of the non-converging fixed-point iteration.

No project imports. numpy + scipy only. Runs in a couple of seconds.

It strips the real code down to the structure that matters:

    1. given a completeness C, the selection map is   S_sel = S_survival * C
    2. the population normalisation is DEFINED as     N0 = N_obs / <S_sel>
    3. the predicted "complete" (pre-detection) counts are  P = N0 * shape
    4. the completeness is fitted by Poisson likelihood on  mu = P * C
    5. goto 1

Step 2 is the problem. Because N0 is *defined* as N_obs / <S_sel>, and <S_sel> is
proportional to the scale of C, the product mu = P * C in step 4 is invariant under
C -> k*C. So step 4 carries no information about the overall scale of C, and the
iteration has no fixed point in that direction. It drifts.

The only thing that stops it drifting instantly is that C = expit(...) saturates at 1
and therefore cannot be rescaled with perfect freedom. That weak restoring force turns
an instant runaway into a slow one, which is why 12 iterations looked fine.

Run:  python 02_minimal_reproduction.py
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------------
# A deliberately tiny stand-in for the real problem.
#   - one "observable" axis x (stands for present mass / distance / |b|)
#   - a fixed survival probability S(x)
#   - a logistic completeness C(x) = expit(intercept + slope * x)
# ----------------------------------------------------------------------------------
N_BINS = 12
X = np.linspace(-2.0, 2.0, N_BINS)
SURVIVAL = np.full(N_BINS, 0.25)          # fixed, known, not fitted
SHAPE = np.exp(-0.5 * X**2)               # population shape over the observable
SHAPE /= SHAPE.sum()

TRUE_INTERCEPT, TRUE_SLOPE = 1.0, 0.8
N_OBSERVED = 165                          # the real catalogue size


def completeness(params: np.ndarray) -> np.ndarray:
    intercept, slope = params
    return np.clip(special.expit(intercept + slope * X), 1e-6, 1.0)


def synthesise_observed_counts() -> np.ndarray:
    """One fixed dataset, generated from the truth."""
    truth = np.array([TRUE_INTERCEPT, TRUE_SLOPE])
    mu = N_OBSERVED * SHAPE * completeness(truth)
    mu *= N_OBSERVED / mu.sum()
    return mu  # use expected counts; Poisson noise is irrelevant to the point


OBSERVED = synthesise_observed_counts()


def population_step(current_completeness: np.ndarray) -> tuple[float, np.ndarray]:
    """Steps 1-3: THIS is where the degeneracy is injected.

    N0 is not a free parameter. It is *defined* so that the model always reproduces
    the observed number of objects, whatever the completeness happens to be.
    """
    selection = SURVIVAL * current_completeness
    selection_fraction = float(np.sum(SHAPE * selection))       # <S_sel>
    n0 = N_OBSERVED / selection_fraction                         # <-- the definition
    predicted_complete = n0 * SHAPE * SURVIVAL
    return n0, predicted_complete


def completeness_step(predicted_complete: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Step 4: fit C by Poisson likelihood against mu = predicted_complete * C."""

    def nll(params: np.ndarray) -> float:
        mu = np.clip(predicted_complete * completeness(params), 1e-12, None)
        return float(-np.sum(OBSERVED * np.log(mu) - mu))

    result = optimize.minimize(
        nll, start, method="L-BFGS-B", bounds=[(-8.0, 8.0), (-8.0, 4.0)]
    )
    return np.asarray(result.x, dtype=float)


def show_the_invariance() -> None:
    """The algebra, checked numerically: mu is invariant under C -> k*C."""
    print("=" * 78)
    print("1. THE FLAT DIRECTION: mu = predicted_complete * C is invariant under C -> k*C")
    print("=" * 78)
    print(f"{'k':>6} {'selection_fraction':>20} {'N0':>14} {'sum(mu)':>12} {'mu[3]':>12}")
    base = completeness(np.array([TRUE_INTERCEPT, TRUE_SLOPE]))
    for k in (1.0, 0.5, 0.25, 0.1):
        scaled = k * base
        selection_fraction = float(np.sum(SHAPE * SURVIVAL * scaled))
        n0 = N_OBSERVED / selection_fraction
        mu = (n0 * SHAPE * SURVIVAL) * scaled
        print(f"{k:>6.2f} {selection_fraction:>20.6f} {n0:>14.1f} {mu.sum():>12.4f} {mu[3]:>12.6f}")
    print()
    print("  N0 moves by 10x. selection_fraction moves by 10x. mu does not move at all.")
    print("  => the completeness likelihood cannot see the scale of C. It is unidentified.")
    print()


def run_the_iteration(n_iterations: int = 200, relaxation: float = 0.7) -> None:
    print("=" * 78)
    print("2. THE CONSEQUENCE: the alternating iteration never settles")
    print("=" * 78)
    params = np.array([0.0, 0.25])
    print(f"{'iter':>5} {'N0':>12} {'mean C':>9} {'N0 x meanC':>12} {'intercept':>10} {'nll':>10}")
    for iteration in range(1, n_iterations + 1):
        current = completeness(params)
        n0, predicted_complete = population_step(current)
        target = completeness_step(predicted_complete, params)
        params = (1.0 - relaxation) * params + relaxation * target

        mu = np.clip(predicted_complete * completeness(params), 1e-12, None)
        nll = float(-np.sum(OBSERVED * np.log(mu) - mu))
        mean_c = float(np.average(completeness(params), weights=predicted_complete))
        if iteration in (1, 2, 5, 12, 25, 50, 100, 150, 200):
            marker = "   <-- the code stopped here (n_iterations = 12)" if iteration == 12 else ""
            print(f"{iteration:>5} {n0:>12.4g} {mean_c:>9.4f} {n0*mean_c:>12.4g} "
                  f"{params[0]:>10.3f} {nll:>10.3f}{marker}")
    print()
    print("  N0 and mean C drift steadily in OPPOSITE directions and never settle, while")
    print("  their PRODUCT stays pinned (660.8 throughout) -- that product is the only")
    print("  thing the data actually constrain. Which way the pair drifts depends only on")
    print("  the starting point, not on the data: in the real code N0 rises 1.8e5 -> 5.7e5,")
    print("  here it falls; either way it is the flat direction being wandered along.")
    print()
    print("  The negative log-likelihood is flat to ~0.1 over the whole run: the iteration")
    print("  is not improving the fit at all, it is sliding sideways.")
    print()
    print("  Whatever N0 you report is decided by where you stopped the loop.")


if __name__ == "__main__":
    show_the_invariance()
    run_the_iteration()
