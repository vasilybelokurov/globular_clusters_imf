"""The integration domain must contain the thing being integrated.

This is the regression suite for the bug that dominated everything else: the mass
grid floor was hard-coded to log10(M)=3.5, which sits *inside* the survival-positive
region, so the selection integral was truncated within its own support and N0 moved
by a factor of ~10 when the floor moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from globular_clusters_imf.joint_model import (
    fit_single_joint_model,
    JointLikelihoodContext,
    JointModelSpec,
    build_fixed_survival_grid,
    fit_single_joint_model_with_fixed_imf_params,
    integrate_survival_fraction,
)
from globular_clusters_imf.mass_grid import (
    DEFAULT_LOG_MASS_MAX,
    DEFAULT_LOG_MASS_MIN,
    build_log_mass_grid,
    check_grid_covers_survival_support,
)

SPEC = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
FIDUCIAL_IMF = np.array([-1.675, 6.35])  # the values the paper published


def test_default_grid_covers_the_whole_survival_positive_region(fit_sample, selection_offset):
    """No cluster's survival threshold may fall below the grid floor."""
    effective_cuts = (
        fit_sample["log_survival_mass_cut_msun"].to_numpy() + selection_offset
    )
    diagnostics = check_grid_covers_survival_support(
        log_mass_grid=build_log_mass_grid(),
        effective_log_mass_cuts=effective_cuts,
    )
    assert diagnostics["n_cuts_below_grid_floor"] == 0
    assert diagnostics["min_effective_log_cut"] > DEFAULT_LOG_MASS_MIN


def test_the_old_hard_coded_floor_is_now_rejected(fit_sample, selection_offset):
    """The historical grid (floor 3.5) truncated 8 clusters' survival thresholds.

    Rebuilding it must now fail loudly instead of silently biasing N0.
    """
    effective_cuts = (
        fit_sample["log_survival_mass_cut_msun"].to_numpy() + selection_offset
    )
    n_truncated = int(np.sum(effective_cuts < 3.5))
    assert n_truncated > 0, "fixture no longer reproduces the historical truncation"

    with pytest.raises(ValueError, match="truncated inside its own support"):
        check_grid_covers_survival_support(
            log_mass_grid=build_log_mass_grid(log_mass_min=3.5, log_mass_max=7.3, n_mass_grid=180),
            effective_log_mass_cuts=effective_cuts,
        )


def test_truncation_can_be_declared_deliberately(fit_sample, selection_offset):
    """A high M_min is legitimate if you *mean* it -- N0 is then a count above it."""
    effective_cuts = (
        fit_sample["log_survival_mass_cut_msun"].to_numpy() + selection_offset
    )
    diagnostics = check_grid_covers_survival_support(
        log_mass_grid=build_log_mass_grid(log_mass_min=3.5, log_mass_max=7.3, n_mass_grid=180),
        effective_log_mass_cuts=effective_cuts,
        log_mass_min_is_physical=True,
    )
    assert diagnostics["n_cuts_below_grid_floor"] > 0


def test_survival_probability_is_a_probability(survival_grid):
    survival = survival_grid["survival_probability"]
    assert np.all(survival >= 0.0)
    assert np.all(survival <= 1.0)


def test_survival_is_monotone_increasing_in_mass(survival_grid):
    """A more massive cluster can never be less likely to survive."""
    survival = survival_grid["survival_probability"]  # (mass, radius)
    assert np.all(np.diff(survival, axis=0) >= -1.0e-12)


def test_selection_fraction_is_converged_at_the_default_resolution(fit_sample, selection_offset):
    """Doubling the mass resolution must not move the selection fraction.

    The survival map is a staircase in mass (one step per cluster), so an
    under-resolved trapezoid integration would quietly bias N0.
    """
    from globular_clusters_imf.joint_model import evaluate_imf_family, evaluate_radial_model

    fractions = []
    for n_mass in (561, 1121):
        grid = build_fixed_survival_grid(
            fit_sample, selection_offset_dex=selection_offset, n_mass_grid=n_mass
        )
        ctx = JointLikelihoodContext.from_catalog_and_survival_grid(fit_sample, grid)
        imf_grid, _, _ = evaluate_imf_family(
            SPEC.imf_family, FIDUCIAL_IMF, ctx.log_mass_grid, ctx.log_mass_data
        )
        radial_grid, _, _ = evaluate_radial_model(
            SPEC.radial_model, np.zeros(3), ctx
        )
        fractions.append(
            integrate_survival_fraction(
                imf_grid, radial_grid, ctx.log_mass_grid, ctx.log_a_grid,
                ctx.survival_probability_grid,
            )
        )
    coarse, fine = fractions
    relative_change = abs(fine - coarse) / fine
    assert relative_change < 0.01, (
        f"selection fraction moved by {relative_change:.2%} when the mass grid was "
        "doubled: the integral is not resolved, so N0 depends on the grid"
    )


def _n0_versus_m_min(fit_sample, selection_offset, fixed_imf: bool) -> dict[float, float]:
    counts = {}
    for log_mass_min in (1.5, 2.0, 2.5):
        n_mass = int(round((DEFAULT_LOG_MASS_MAX - log_mass_min) / 0.0107)) + 1
        grid = build_fixed_survival_grid(
            fit_sample,
            selection_offset_dex=selection_offset,
            log_mass_min=log_mass_min,
            n_mass_grid=n_mass,
        )
        ctx = JointLikelihoodContext.from_catalog_and_survival_grid(fit_sample, grid)
        if fixed_imf:
            payload = fit_single_joint_model_with_fixed_imf_params(
                context=ctx, spec=SPEC, fixed_imf_params=FIDUCIAL_IMF
            )
        else:
            payload = fit_single_joint_model(context=ctx, spec=SPEC)
        counts[log_mass_min] = payload["model"]["total_initial_count"]
    return counts


def test_n0_is_stable_against_m_min_once_the_imf_can_adapt(fit_sample, selection_offset):
    """The headline regression test.

    On the truncated grid, N0 moved by a factor of 10.7 between M_min = 3.5 and 2.0.
    With a grid that covers the whole survival-positive region and an IMF free to
    re-fit, the residual dependence must be mild. Some dependence is *correct* --
    N0 is by definition a count above M_min -- but not an order of magnitude.
    """
    counts = _n0_versus_m_min(fit_sample, selection_offset, fixed_imf=False)
    spread = max(counts.values()) / min(counts.values())
    assert spread < 1.5, f"N0 still swings by {spread:.1f}x with M_min: {counts}"


def test_holding_the_imf_fixed_is_what_makes_n0_grid_dependent(fit_sample, selection_offset):
    """Pins down *why* the fix works, so it cannot be undone by accident.

    Freeze the IMF at the published (steep) alpha = -1.675 and N0 becomes strongly
    M_min-dependent again: a steep IMF genuinely puts most of its clusters at low
    mass, so the count above M_min really does depend on M_min. The stability in the
    test above comes from alpha being free to adapt to the complete domain -- which
    is precisely what the truncated grid prevented.
    """
    counts = _n0_versus_m_min(fit_sample, selection_offset, fixed_imf=True)
    spread = max(counts.values()) / min(counts.values())
    assert spread > 3.0, (
        "expected the fixed steep IMF to remain strongly M_min-dependent, "
        f"got only {spread:.1f}x: {counts}"
    )


def test_n0_always_carries_the_mass_floor_it_is_conditional_on(context):
    """N0 without its M_min is meaningless; the model must never hand one over alone."""
    payload = fit_single_joint_model_with_fixed_imf_params(
        context=context, spec=SPEC, fixed_imf_params=FIDUCIAL_IMF
    )
    assert payload["model"]["n0_log_mass_min"] == pytest.approx(DEFAULT_LOG_MASS_MIN)
    assert payload["summary"].n0_log_mass_min == pytest.approx(DEFAULT_LOG_MASS_MIN)


def test_grid_builder_rejects_nonsense():
    with pytest.raises(ValueError):
        build_log_mass_grid(log_mass_min=7.0, log_mass_max=3.0)
    with pytest.raises(ValueError):
        build_log_mass_grid(n_mass_grid=1)
