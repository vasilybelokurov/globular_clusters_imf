"""A stalled iteration must be abandoned, not ground out to the cap.

The convergence guard originally ran every non-converging fit all the way to
`max_iterations = 200`. Measured behaviour of the configurations that never converge:
the residual falls for ~30 iterations, flattens at ~3e-3, and sits there unchanged to
200. The remaining ~170 iterations discover nothing and cost ~16x a converging fit --
which made the 135-point grid scan take hours instead of minutes.

Stall detection must reach the *same verdict* (non-convergent), just sooner.
"""

from __future__ import annotations

import numpy as np
import pytest

from globular_clusters_imf.detectability_model import (
    DEFAULT_EM_PATIENCE,
    DetectabilityConvergenceError,
    StallDetector,
    assert_em_converged,
)


class TestStallDetector:
    def test_steadily_improving_residual_never_stalls(self):
        detector = StallDetector()
        residual = 1.0
        for _ in range(200):
            residual *= 0.7  # a healthy, converging iteration
            assert detector.update(residual) is False

    def test_flat_residual_stalls_after_patience(self):
        """The real failure mode: residual pinned at ~3e-3 forever."""
        detector = StallDetector(patience=15)
        assert detector.update(3.0e-3) is False  # first sighting is an improvement
        stalled_at = None
        for step in range(2, 60):
            if detector.update(3.0e-3):
                stalled_at = step
                break
        assert stalled_at == 16, f"expected to stall one patience window in, got {stalled_at}"

    def test_negligible_improvement_does_not_reset_patience(self):
        """A residual creeping down by 0.01% per step is stalled, not converging.

        Requiring a *relative* improvement is what stops numerical jitter from
        resetting the counter forever and defeating the whole mechanism.
        """
        detector = StallDetector(patience=10, min_relative_improvement=0.01)
        residual = 3.0e-3
        stalled = False
        for _ in range(60):
            residual *= 0.9999  # 0.01% -- far below the 1% threshold
            if detector.update(residual):
                stalled = True
                break
        assert stalled

    def test_a_late_genuine_improvement_resets_the_counter(self):
        detector = StallDetector(patience=5)
        for _ in range(4):
            detector.update(1.0e-2)
        assert detector.update(1.0e-3) is False  # real progress: keep going
        assert detector.iterations_since_improvement == 0

    def test_noisy_residual_that_never_beats_its_best_still_stalls(self):
        rng = np.random.default_rng(0)
        detector = StallDetector(patience=DEFAULT_EM_PATIENCE)
        detector.update(1.0e-3)
        stalled = False
        for _ in range(100):
            # jitters around, never establishes a new best
            if detector.update(1.0e-3 * (1.0 + abs(rng.normal(0, 0.1)))):
                stalled = True
                break
        assert stalled


class TestGuardReportsStallDistinctly:
    def test_stalled_message_says_more_iterations_will_not_help(self):
        with pytest.warns(RuntimeWarning, match="stopped improving"):
            assert_em_converged(3.0e-3, n_iterations_run=45, label="x", stalled=True)

    def test_plain_non_convergence_suggests_a_higher_cap(self):
        with pytest.warns(RuntimeWarning, match="higher iteration cap"):
            assert_em_converged(3.0e-3, n_iterations_run=200, label="x", stalled=False)

    def test_hard_failure_is_still_available_on_request(self):
        with pytest.raises(DetectabilityConvergenceError, match="stopped improving"):
            assert_em_converged(3.0e-3, n_iterations_run=45, label="x", stalled=True,
                                raise_on_non_convergence=True)

    def test_a_converged_fit_is_unaffected_by_the_stall_flag(self):
        assert assert_em_converged(1.0e-9, n_iterations_run=30, label="x", stalled=True) is True
