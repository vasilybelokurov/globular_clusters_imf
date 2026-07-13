"""The detectability iteration must reach a fixed point, and say so if it doesn't.

Regression suite for: a fixed `n_iterations=6` with no convergence test, which
published an actively diverging iterate (completeness_mean 0.90 -> 0.28, N0 rising,
log-likelihood *falling*).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from globular_clusters_imf.detectability_model import (
    DetectabilityConvergenceError,
    assert_em_converged,
    fit_single_component_detectability_em,
    scan_n0_versus_assumed_mean_completeness,
    solve_intercept_for_mean_completeness,
)
from globular_clusters_imf.joint_model import JointModelSpec

SPEC = JointModelSpec(imf_family="schechter", radial_model="logpoly3")


@pytest.fixture(scope="module")
def em_result(fit_sample, scratch_root):
    return fit_single_component_detectability_em(fit_sample, scratch_root, spec=SPEC)


def test_em_reaches_a_fixed_point(em_result):
    convergence = em_result["convergence"]
    assert convergence.converged, convergence.message
    assert convergence.parameter_residual < convergence.tolerance


def test_em_log_likelihood_never_decreases(em_result):
    """An EM that maximises cannot go downhill. The old one did."""
    assert em_result["convergence"].log_likelihood_monotone
    history = em_result["iteration_history_table"]["log_likelihood"].to_numpy()
    assert np.all(np.diff(history) > -1.0e-6), (
        f"log-likelihood decreased along the iteration: {history}"
    )


def test_em_reports_the_mass_floor_its_n0_is_conditional_on(em_result):
    assert "n0_log_mass_min" in em_result["final_payload"]["model"]


def test_fixed_iteration_count_is_refused(fit_sample, scratch_root):
    """Running N steps and publishing the last one is the bug; the API now blocks it."""
    with pytest.raises(TypeError, match="n_iterations"):
        fit_single_component_detectability_em(
            fit_sample, scratch_root, spec=SPEC, n_iterations=6
        )


def test_non_convergence_raises_rather_than_returning_a_number(fit_sample, scratch_root):
    """Starve it of iterations: it must fail, not hand back the last iterate."""
    with pytest.raises(DetectabilityConvergenceError, match="did NOT converge"):
        fit_single_component_detectability_em(
            fit_sample, scratch_root, spec=SPEC, max_iterations=2, tolerance=1.0e-12
        )


def test_assert_em_converged_is_loud_by_default():
    assert assert_em_converged(1.0e-9, n_iterations_run=3, label="x") is True
    with pytest.raises(DetectabilityConvergenceError):
        assert_em_converged(1.0, n_iterations_run=3, label="x")
    with pytest.warns(RuntimeWarning, match="did NOT converge"):
        assert (
            assert_em_converged(
                1.0, n_iterations_run=3, label="x", raise_on_non_convergence=False
            )
            is False
        )


class _FakeObservableContext:
    """Minimal stand-in exercising the intercept solver's arithmetic."""

    def __init__(self, shape_logits: np.ndarray):
        self._shape = shape_logits


def test_mean_completeness_anchor_hits_its_target(monkeypatch, fit_sample, scratch_root):
    """Solving the intercept must actually produce the declared mean completeness."""
    import globular_clusters_imf.detectability_model as dm

    shape = np.array([[[-2.0, 0.0]], [[1.0, 3.0]]])
    weights = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    monkeypatch.setattr(dm, "completeness_shape_logits", lambda p, c: shape)

    for target in (0.3, 0.5, 0.85, 0.95):
        intercept = solve_intercept_for_mean_completeness(
            raw_params=np.zeros(4),
            observable_context=None,
            weights=weights,
            assumed_mean_completeness=target,
        )
        achieved = float(
            np.sum(weights * special.expit(intercept + shape)) / np.sum(weights)
        )
        assert achieved == pytest.approx(target, abs=1.0e-8)


def test_anchor_brackets_even_with_extreme_slopes(monkeypatch):
    """The bracket must dominate the shape logits, which can exceed 100 in magnitude.

    A fixed +/-40 bracket silently failed to contain the root here.
    """
    import globular_clusters_imf.detectability_model as dm

    shape = np.array([-120.0, 0.0, 120.0])
    weights = np.ones(3)
    monkeypatch.setattr(dm, "completeness_shape_logits", lambda p, c: shape)
    intercept = solve_intercept_for_mean_completeness(
        raw_params=np.zeros(4),
        observable_context=None,
        weights=weights,
        assumed_mean_completeness=0.5,
    )
    achieved = float(np.mean(special.expit(intercept + shape)))
    assert achieved == pytest.approx(0.5, abs=1.0e-6)


@pytest.mark.slow
def test_n0_scales_inversely_with_the_assumed_completeness(fit_sample, scratch_root):
    """Quantifies how much of N0 is assumption rather than measurement.

    N0 = N_obs / (survival x completeness), so if the scan comes back close to
    proportional to 1/completeness, the amplitude of the selection function -- not
    the data -- is setting the answer. This test pins that behaviour so it cannot
    be quietly forgotten when N0 is quoted.
    """
    table = scan_n0_versus_assumed_mean_completeness(
        fit_sample, scratch_root, [0.9, 0.6, 0.3], spec=SPEC,
        raise_on_non_convergence=False,
    )
    assert table["achieved_mean_completeness"].to_numpy() == pytest.approx(
        table["assumed_mean_completeness"].to_numpy(), abs=0.05
    )
    product = table["total_initial_count"] * table["assumed_mean_completeness"]
    spread = float(product.max() / product.min())
    assert spread < 2.0, (
        "N0 x completeness should be roughly invariant; if it is, N0 is being set "
        f"by the completeness assumption. Got spread {spread:.2f}. Table:\n{table}"
    )
