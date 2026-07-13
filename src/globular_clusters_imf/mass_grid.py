"""Canonical initial-mass integration grid.

Every population number this project reports (in particular the total initial
cluster count ``N0`` and any total initial stellar mass derived from it) is a
statement about clusters *above some lower formation mass* ``M_min``.  The data
carry no information about clusters below the survival threshold, so ``M_min``
is a modelling assumption, never an inference.  It therefore has to be declared
explicitly rather than emerging from a hard-coded grid edge.

Historically the grid was built as ``min(3.5, floor(data_min))`` .. ``max(7.3,
ceil(data_max))``.  That had two separate problems:

1. ``M_min`` was implicit, undocumented and *data dependent* (the lowest-mass
   cluster in the catalogue silently moved it).
2. The floor of 3.5 dex sits **inside** the survival-positive region: the
   modelled survival probability at ``log10 M = 3.5`` reaches 0.74 and averages
   0.18 across radii.  The normalising integral was being truncated within the
   support of its own integrand.

Both are fixed here by making ``log_mass_min`` an explicit parameter and by
refusing to build a grid that clips a survival-positive region it was not told
to clip.

`N0` scales steeply with this choice -- roughly a factor 10 between
``log_mass_min`` of 3.5 and 2.0 for the fiducial Schechter IMF -- so
``describe_mass_grid`` attaches the value to every grid it builds, and callers
are expected to propagate it into their outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fiducial lower cluster-formation mass. 10^2 Msun is below the lowest survival
# threshold implied by the Baumgardt catalogue (min effective cut ~ 10^2.6 Msun),
# so the integration domain fully contains the survival-positive region and the
# selection integral is complete. It is an assumption, not a measurement.
DEFAULT_LOG_MASS_MIN = 2.0

# Upper edge. Well above the most massive initial mass in the catalogue (~10^7.0),
# and far enough into the Schechter cutoff that the integral has converged.
DEFAULT_LOG_MASS_MAX = 8.0

# Number of grid points. Chosen so the selection integral is converged; see
# tests/test_mass_grid.py::test_selection_fraction_converged_at_default_resolution.
DEFAULT_N_MASS_GRID = 561


@dataclass(frozen=True)
class MassGridSpec:
    """Declared integration domain for the initial-mass coordinate."""

    log_mass_min: float = DEFAULT_LOG_MASS_MIN
    log_mass_max: float = DEFAULT_LOG_MASS_MAX
    n_mass_grid: int = DEFAULT_N_MASS_GRID

    def grid(self) -> np.ndarray:
        return np.linspace(self.log_mass_min, self.log_mass_max, self.n_mass_grid)

    @property
    def step_dex(self) -> float:
        return (self.log_mass_max - self.log_mass_min) / (self.n_mass_grid - 1)


def build_log_mass_grid(
    log_mass_min: float = DEFAULT_LOG_MASS_MIN,
    log_mass_max: float = DEFAULT_LOG_MASS_MAX,
    n_mass_grid: int = DEFAULT_N_MASS_GRID,
) -> np.ndarray:
    """Return the declared initial-mass integration grid, in log10(M/Msun)."""
    if not log_mass_max > log_mass_min:
        raise ValueError(
            f"log_mass_max ({log_mass_max}) must exceed log_mass_min ({log_mass_min})."
        )
    if n_mass_grid < 2:
        raise ValueError(f"n_mass_grid must be at least 2, got {n_mass_grid}.")
    return np.linspace(float(log_mass_min), float(log_mass_max), int(n_mass_grid))


def check_grid_covers_survival_support(
    log_mass_grid: np.ndarray,
    effective_log_mass_cuts: np.ndarray,
    log_mass_min_is_physical: bool = False,
) -> dict[str, float]:
    """Verify the grid is not truncating a survival-positive region by accident.

    ``effective_log_mass_cuts`` are the per-cluster survival thresholds already
    shifted by the calibrated selection offset.  A cluster whose threshold lies
    below ``log_mass_grid[0]`` contributes survival probability at the very first
    grid row, which means the integral ``int phi * A * S`` is being cut off while
    its integrand is still non-zero.

    Set ``log_mass_min_is_physical=True`` to declare that the truncation is
    intentional -- i.e. the model genuinely asserts no clusters form below
    ``log_mass_min``, so ``N0`` is by definition a count above that mass.  In
    that case the check only reports, it does not raise.
    """
    log_mass_grid = np.asarray(log_mass_grid, dtype=float)
    cuts = np.asarray(effective_log_mass_cuts, dtype=float)
    cuts = cuts[np.isfinite(cuts)]

    floor = float(log_mass_grid[0])
    n_below = int(np.sum(cuts < floor))
    diagnostics = {
        "log_mass_grid_min": floor,
        "log_mass_grid_max": float(log_mass_grid[-1]),
        "min_effective_log_cut": float(cuts.min()) if cuts.size else float("nan"),
        "n_cuts_below_grid_floor": float(n_below),
    }
    if n_below > 0 and not log_mass_min_is_physical:
        raise ValueError(
            f"The mass grid floor log10(M)={floor:.3f} lies above the survival "
            f"threshold of {n_below} cluster(s) (lowest threshold "
            f"{cuts.min():.3f}). The selection integral would be truncated inside "
            "its own support, which biases N0 by a factor that depends only on "
            "the grid. Either lower log_mass_min, or pass "
            "log_mass_min_is_physical=True to declare the truncation as a "
            "deliberate lower cluster-formation mass."
        )
    return diagnostics


def describe_mass_grid(log_mass_grid: np.ndarray) -> dict[str, float]:
    """Metadata to attach to any N0 so the reader knows what it is a count of."""
    log_mass_grid = np.asarray(log_mass_grid, dtype=float)
    return {
        "log_mass_min": float(log_mass_grid[0]),
        "log_mass_max": float(log_mass_grid[-1]),
        "n_mass_grid": int(log_mass_grid.size),
        "mass_step_dex": float(log_mass_grid[1] - log_mass_grid[0]),
    }
