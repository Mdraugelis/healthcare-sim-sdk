"""Unit tests for Tier 3 rolling CITS."""

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
    Tier3CITS,
)


class TestTier3Initialization:
    """The CITS estimator requires a valid mode."""

    def test_invalid_mode_rejected(self):
        import pytest
        with pytest.raises(ValueError, match="mode must"):
            Tier3CITS(mode="bayesian")

    def test_cits_with_cf_valid(self):
        cits = Tier3CITS(mode="cits_with_cf")
        assert cits.mode == "cits_with_cf"

    def test_its_only_valid(self):
        cits = Tier3CITS(mode="its_only")
        assert cits.mode == "its_only"


class TestNoFitBeforeMinObservations:
    """Tier 3 should not fit until the minimum observations are hit."""

    def test_no_fit_with_few_observations(self):
        cits = Tier3CITS(mode="cits_with_cf", min_observations=13)
        for week in range(10):
            events = cits.update(
                week=week,
                factual_turnover=0.01,
                counterfactual_turnover=0.012,
            )
            assert events == []
        assert len(cits.estimates) == 0

    def test_first_fit_at_min_observations(self):
        cits = Tier3CITS(
            mode="cits_with_cf", min_observations=13,
        )
        for week in range(13):
            cits.update(
                week=week,
                factual_turnover=0.01,
                counterfactual_turnover=0.012,
            )
        assert len(cits.estimates) == 1


class TestCITSWithCFMode:
    """CITS with counterfactual as control recovers known effect."""

    def test_recovers_known_level_difference(self):
        """Known constant effect: factual is consistently 0.002
        below CF (the intervention lowered turnover). CITS should
        recover a negative effect estimate.
        """
        cits = Tier3CITS(
            mode="cits_with_cf",
            refit_interval_weeks=13,
            min_observations=13,
        )
        rng = np.random.default_rng(42)

        # Run for a full year (52 weeks)
        for week in range(52):
            cf_rate = 0.015 + rng.normal(0, 0.001)
            f_rate = cf_rate - 0.002 + rng.normal(0, 0.001)
            cits.update(
                week=week,
                factual_turnover=f_rate,
                counterfactual_turnover=cf_rate,
            )

        # Should have fit 4 quarters
        assert len(cits.estimates) >= 3
        # Final estimate should recover the -0.002 effect direction
        final = cits.estimates[-1]
        assert final.effect_estimate < 0
        # And the 95% CI should cover the truth
        assert final.ci_lower <= -0.002 <= final.ci_upper

    def test_no_effect_in_null_regime(self):
        """When factual and CF are identical (null effect), the
        point estimate should be near zero.
        """
        cits = Tier3CITS(
            mode="cits_with_cf",
            refit_interval_weeks=13,
            min_observations=13,
        )
        rng = np.random.default_rng(42)
        for week in range(52):
            # Same underlying rate with independent noise
            rate = 0.01
            f_rate = rate + rng.normal(0, 0.001)
            cf_rate = rate + rng.normal(0, 0.001)
            cits.update(
                week=week,
                factual_turnover=f_rate,
                counterfactual_turnover=cf_rate,
            )
        # Final estimate should be within ~0.0005 of zero
        final = cits.estimates[-1]
        assert abs(final.effect_estimate) < 0.002
        # CI should cover zero
        assert final.ci_lower <= 0 <= final.ci_upper

    def test_significance_triggers_detection(self):
        """A large effect should fire a DetectionEvent at the first
        quarterly refit.
        """
        cits = Tier3CITS(
            mode="cits_with_cf",
            refit_interval_weeks=13,
            min_observations=13,
        )
        rng = np.random.default_rng(42)
        all_events = []
        for week in range(26):
            # Huge effect: factual is 0.005 below CF
            cf = 0.02 + rng.normal(0, 0.0005)
            f = cf - 0.01 + rng.normal(0, 0.0005)
            events = cits.update(
                week=week,
                factual_turnover=f,
                counterfactual_turnover=cf,
            )
            all_events.extend(events)
        # Should fire a Tier 3 detection
        assert any(e.tier == 3 for e in all_events)


class TestITSOnlyMode:
    """Pure ITS on factual series alone estimates a time trend."""

    def test_detects_trend_in_factual_series(self):
        cits = Tier3CITS(mode="its_only", min_observations=13)
        rng = np.random.default_rng(42)
        # Linear upward drift in factual turnover (decay regime)
        for week in range(52):
            f_rate = 0.01 + week * 0.0002 + rng.normal(0, 0.0005)
            cits.update(
                week=week,
                factual_turnover=f_rate,
            )
        # Should have estimates
        assert len(cits.estimates) >= 3
        final = cits.estimates[-1]
        # Slope should be positive and close to 0.0002
        assert final.effect_estimate > 0
        assert abs(final.effect_estimate - 0.0002) < 0.0002

    def test_stable_series_slope_near_zero(self):
        cits = Tier3CITS(mode="its_only", min_observations=13)
        rng = np.random.default_rng(42)
        for week in range(52):
            f_rate = 0.01 + rng.normal(0, 0.0005)
            cits.update(week=week, factual_turnover=f_rate)
        final = cits.estimates[-1]
        assert abs(final.effect_estimate) < 1e-4
