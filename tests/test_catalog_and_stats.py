"""Catalogue parsing, the covariance guard, and R-hat."""

from __future__ import annotations

import numpy as np
import pytest

from globular_clusters_imf.catalog import (
    canonical_cluster_name,
    normalize_gc_name_key,
    parse_power_of_ten_suffix,
    parse_value,
)
from globular_clusters_imf.joint_model import (
    NonPositiveDefiniteHessianError,
    JointModelSpec,
    estimate_parameter_covariance,
    fit_single_joint_model,
)
from globular_clusters_imf.mcmc_diagnostics import compute_rhat


class TestParseValue:
    def test_plain_numbers(self):
        assert parse_value("3.5") == pytest.approx(3.5)
        assert parse_value("-2") == pytest.approx(-2.0)
        assert parse_value(7.25) == pytest.approx(7.25)

    def test_unicode_minus_is_handled(self):
        """The Baumgardt tables use U+2212, not ASCII hyphen."""
        assert parse_value("−3.5") == pytest.approx(-3.5)
        assert parse_value("–4.25") == pytest.approx(-4.25)

    def test_plus_minus_uncertainty_is_dropped_not_parsed_as_the_value(self):
        assert parse_value("5.2 ± 0.3") == pytest.approx(5.2)

    def test_scientific_notation_with_the_middot_convention(self):
        # "1.5 · 10^5" is rendered with the exponent glued to the 10
        assert parse_value("1.5 · 105") == pytest.approx(1.5e5)
        assert parse_value("2.0 ± 0.1 · 106") == pytest.approx(2.0e6)

    def test_unparseable_returns_nan_rather_than_guessing(self):
        assert np.isnan(parse_value(None))
        assert np.isnan(parse_value(float("nan")))

    def test_power_of_ten_suffix(self):
        assert parse_power_of_ten_suffix("105") == pytest.approx(1.0e5)
        assert parse_power_of_ten_suffix("1012") == pytest.approx(1.0e12)


class TestNameNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("NGC 104", "NGC 104"),
        ("NGC104", "NGC104"),          # no space: left alone rather than mangled
        ("Ter 5", "TERZAN 5"),
        ("Terzan 5", "TERZAN 5"),
        ("Djor 1", "DJORG 1"),
        ("Pal 1", "PAL 1"),
        ("NGC 6121 (M 4)", "NGC 6121"),
    ])
    def test_keys_collapse_aliases(self, raw, expected):
        assert normalize_gc_name_key(raw) == expected

    def test_terzan_and_ter_map_together(self):
        assert normalize_gc_name_key("Ter 5") == normalize_gc_name_key("Terzan 5")

    def test_canonical_name_strips_the_alias(self):
        assert canonical_cluster_name("NGC 6121 M4") == "NGC 6121"
        assert canonical_cluster_name("Pal 1") == "Pal 1"


class TestCovarianceGuard:
    def test_covariance_is_positive_definite_at_a_real_optimum(self, context):
        spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
        payload = fit_single_joint_model(context=context, spec=spec)
        covariance = estimate_parameter_covariance(
            payload["raw_parameters"], context=context, spec=spec, bounds=payload["bounds"]
        )
        assert np.all(np.linalg.eigvalsh(covariance) > 0)
        # symmetric
        assert covariance == pytest.approx(covariance.T)

    def test_non_minimum_raises_instead_of_faking_tight_errors(self, context):
        """The old code clipped indefinite Hessians into arbitrarily small variances.

        A failed fit then presented itself as a precise measurement. Evaluate the
        covariance far from the optimum, where the Hessian is not positive definite,
        and it must refuse.
        """
        spec = JointModelSpec(imf_family="schechter", radial_model="logpoly3")
        bounds = [(-4.0, -0.2), (4.5, 7.5), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
        # a point well away from any minimum
        bad = np.array([-3.9, 7.4, 4.5, -4.5, 4.5])
        with pytest.raises(NonPositiveDefiniteHessianError):
            estimate_parameter_covariance(bad, context=context, spec=spec, bounds=bounds)


class TestRhat:
    def test_identical_chains_give_the_gelman_rubin_floor(self):
        """Identical chains give sqrt((n-1)/n), not exactly 1.

        With zero between-chain variance, var_hat = ((n-1)/n) * W, so
        R-hat = sqrt((n-1)/n) -- slightly below 1. That is the correct classic
        Gelman-Rubin value, not a bug; asserting exactly 1.0 here would be wrong.
        """
        rng = np.random.default_rng(0)
        chain = rng.normal(size=500)
        chains = np.vstack([chain, chain, chain])
        expected = np.sqrt((500 - 1) / 500)
        assert compute_rhat(chains) == pytest.approx(expected, abs=1.0e-9)
        assert compute_rhat(chains) <= 1.0

    def test_well_mixed_chains_are_near_one(self):
        rng = np.random.default_rng(1)
        chains = rng.normal(size=(4, 4000))
        assert compute_rhat(chains) == pytest.approx(1.0, abs=0.02)

    def test_badly_separated_chains_are_flagged(self):
        rng = np.random.default_rng(2)
        chains = rng.normal(size=(4, 500)) + np.array([[0.0], [10.0], [20.0], [30.0]])
        assert compute_rhat(chains) > 1.5

    def test_degenerate_input_is_nan_not_a_crash(self):
        assert np.isnan(compute_rhat(np.zeros((1, 10))))
        assert np.isnan(compute_rhat(np.zeros((4, 1))))
        assert np.isnan(compute_rhat(np.ones((4, 10))))  # zero within-chain variance
