"""Unit tests for population utilities."""

import numpy as np
import pytest

from healthcare_sim_sdk.population.risk_distributions import beta_distributed_risks
from healthcare_sim_sdk.population.temporal_dynamics import (
    AR1Process,
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
)


class TestBetaDistributedRisks:
    def test_mean_matches_target(self):
        rng = np.random.default_rng(42)
        risks = beta_distributed_risks(10_000, 0.05, 0.5, rng)
        assert abs(risks.mean() - 0.05) < 0.005

    def test_right_skewed(self):
        rng = np.random.default_rng(42)
        risks = beta_distributed_risks(10_000, 0.05, 0.5, rng)
        assert np.median(risks) < risks.mean()

    def test_bounded(self):
        rng = np.random.default_rng(42)
        risks = beta_distributed_risks(10_000, 0.05, 0.5, rng)
        assert risks.min() >= 0
        assert risks.max() <= 0.99

    def test_different_seeds(self):
        r1 = beta_distributed_risks(100, 0.05, 0.5, np.random.default_rng(1))
        r2 = beta_distributed_risks(100, 0.05, 0.5, np.random.default_rng(2))
        assert not np.array_equal(r1, r2)

    def test_reproducible(self):
        r1 = beta_distributed_risks(100, 0.05, 0.5, np.random.default_rng(42))
        r2 = beta_distributed_risks(100, 0.05, 0.5, np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_rng_is_required(self):
        """Calling without rng must raise — no silent unseeded generator.

        Guards Invariant #3 (RNG partitioning): a missing rng used to
        fall back to ``np.random.default_rng()``, silently breaking
        reproducibility inside a supposedly seeded scenario.
        """
        with pytest.raises(TypeError):
            beta_distributed_risks(100, 0.05, 0.5)  # type: ignore[call-arg]


class TestHazardConversion:
    def test_low_risk_approximately_linear(self):
        """For small risks, hazard ≈ risk."""
        risk = np.array([0.01, 0.02])
        h = annual_risk_to_hazard(risk)
        np.testing.assert_allclose(h, risk, rtol=0.02)

    def test_roundtrip(self):
        """annual -> hazard -> annual should be identity."""
        risk = np.array([0.05, 0.15, 0.50])
        h = annual_risk_to_hazard(risk)
        recovered = 1 - np.exp(-h)
        np.testing.assert_allclose(recovered, risk, rtol=1e-6)

    def test_timestep_probability_less_than_annual(self):
        risk = np.array([0.10])
        h = annual_risk_to_hazard(risk)
        p_weekly = hazard_to_timestep_probability(h, 1 / 52)
        assert p_weekly[0] < risk[0]

    def test_full_year_recovers_annual(self):
        risk = np.array([0.05, 0.20])
        h = annual_risk_to_hazard(risk)
        p_annual = hazard_to_timestep_probability(h, 1.0)
        np.testing.assert_allclose(p_annual, risk, rtol=1e-6)


class TestAR1Process:
    def test_output_shape(self):
        ar1 = AR1Process(100)
        rng = np.random.default_rng(42)
        mods = ar1.step(rng)
        assert mods.shape == (100,)

    def test_bounded(self):
        ar1 = AR1Process(1000, bounds=(0.5, 2.0))
        rng = np.random.default_rng(42)
        for _ in range(100):
            mods = ar1.step(rng)
        assert mods.min() >= 0.5
        assert mods.max() <= 2.0

    def test_mean_reverts_to_mu(self):
        ar1 = AR1Process(5000, rho=0.9, sigma=0.1, mu=1.0)
        rng = np.random.default_rng(42)
        for _ in range(200):
            mods = ar1.step(rng)
        assert abs(mods.mean() - 1.0) < 0.1

    def test_seasonal_step(self):
        ar1 = AR1Process(100)
        rng = np.random.default_rng(42)
        mods = ar1.step_with_season(rng, t=0, seasonal_amplitude=0.2)
        assert mods.shape == (100,)

    def test_autocorrelation(self):
        """Sequential steps should be correlated."""
        ar1 = AR1Process(10_000, rho=0.95, sigma=0.05)
        rng = np.random.default_rng(42)
        # Warm up the process to steady state
        for _ in range(20):
            ar1.step(rng)
        prev = ar1.step(rng)
        curr = ar1.step(rng)
        corr = np.corrcoef(prev, curr)[0, 1]
        assert corr > 0.7
