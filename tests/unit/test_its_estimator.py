"""Unit tests for ``experiments/analysis/its.py``.

These tests exercise the three estimators independently: the new
pre/post segmented regression, the lifted CITS-with-control mode,
and the lifted slope-only mode. Behavioural equivalence between the
lifted functions and the original ``tier3_cits.py`` internals is
covered by the existing ``test_tier3_cits.py`` suite plus the
regression check in the final test class.
"""

import numpy as np
import pytest

from healthcare_sim_sdk.experiments.analysis.its import (
    CITSWithControlResult,
    ITSResult,
    SlopeOnlyResult,
    cits_with_control,
    its_slope_only,
    power_across_seeds,
    segmented_regression,
)


class TestSegmentedRegressionPlantedEffects:
    """Recover known level and slope changes from a clean series."""

    def test_planted_step_change_recovered(self):
        rng = np.random.default_rng(42)
        n_pre, n_post = 36, 24
        noise = 0.001
        pre = 0.020 + rng.normal(0, noise, n_pre)
        post = 0.015 + rng.normal(0, noise, n_post)
        series = np.concatenate([pre, post])

        result = segmented_regression(series, break_index=n_pre)

        assert isinstance(result, ITSResult)
        assert result.level_change < 0
        assert result.level_change_pvalue < 0.05
        assert result.direction == "decrease"
        # Point estimate within 0.002 of the planted -0.005 step
        assert abs(result.level_change - (-0.005)) < 0.002
        assert result.n_pre == n_pre
        assert result.n_post == n_post

    def test_slope_change_detected(self):
        """A post-break downward slope produces a significant
        ``slope_change`` coefficient."""
        rng = np.random.default_rng(0)
        n_pre, n_post = 36, 36
        noise = 0.0002
        pre = 0.02 + rng.normal(0, noise, n_pre)
        post = (
            0.02
            - 0.0003 * np.arange(n_post)
            + rng.normal(0, noise, n_post)
        )
        series = np.concatenate([pre, post])

        result = segmented_regression(
            series, break_index=n_pre, hac_maxlags=0,
        )

        assert result.slope_change < 0
        assert result.slope_change_pvalue < 0.05

    def test_pre_trend_picked_up_by_beta1(self):
        """A real pre-period trend should be flagged in ``pre_slope``."""
        rng = np.random.default_rng(7)
        n_pre, n_post = 36, 24
        noise = 0.0005
        trend = 0.0005
        pre = (
            0.02
            + trend * np.arange(n_pre)
            + rng.normal(0, noise, n_pre)
        )
        post = (
            0.02
            + trend * n_pre
            + rng.normal(0, noise, n_post)
        )
        series = np.concatenate([pre, post])

        result = segmented_regression(series, break_index=n_pre)

        assert result.pre_slope > 0
        assert result.pre_slope_pvalue < 0.05


class TestSegmentedRegressionNullBehavior:
    """A flat noisy series must not produce false significance."""

    def test_null_level_change_not_significant(self):
        rng = np.random.default_rng(42)
        n = 60
        series = 0.015 + rng.normal(0, 0.001, n)
        result = segmented_regression(series, break_index=36)
        assert result.level_change_pvalue > 0.05
        assert result.direction == "none"

    def test_null_pre_slope_not_significant(self):
        rng = np.random.default_rng(42)
        series = 0.02 + rng.normal(0, 0.0005, 60)
        result = segmented_regression(series, break_index=36)
        assert result.pre_slope_pvalue > 0.05


class TestSegmentedRegressionHAC:
    """Newey-West HAC errors inflate against OLS on AR(1) residuals."""

    def test_hac_se_not_smaller_than_ols(self):
        n = 80
        rng = np.random.default_rng(7)
        eps = np.zeros(n)
        eps[0] = rng.normal(0, 0.001)
        rho = 0.7
        for i in range(1, n):
            eps[i] = rho * eps[i - 1] + rng.normal(0, 0.001)
        series = 0.02 + eps
        series[40:] -= 0.003  # planted step

        ols = segmented_regression(series, break_index=40, hac_maxlags=0)
        hac = segmented_regression(series, break_index=40, hac_maxlags=4)

        assert hac.level_change_se >= ols.level_change_se
        # Both should still agree on the point estimate to high precision
        assert abs(hac.level_change - ols.level_change) < 1e-12


class TestSegmentedRegressionInputValidation:
    """Bad inputs should raise ValueError with informative messages."""

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 4"):
            segmented_regression(
                np.array([0.01, 0.02, 0.03]), break_index=1,
            )

    def test_break_index_too_small(self):
        with pytest.raises(ValueError, match="break_index"):
            segmented_regression(np.zeros(10), break_index=1)

    def test_break_index_too_large(self):
        with pytest.raises(ValueError, match="break_index"):
            segmented_regression(np.zeros(10), break_index=9)

    def test_negative_hac_maxlags_rejected(self):
        with pytest.raises(ValueError, match="hac_maxlags"):
            segmented_regression(
                np.zeros(10), break_index=5, hac_maxlags=-1,
            )


class TestPowerAcrossSeeds:
    """``power_across_seeds`` must aggregate per-seed fits correctly."""

    def test_power_near_one_for_large_effect(self):
        rng = np.random.default_rng(42)
        n_pre, n_post = 36, 24
        series_list = []
        for _ in range(30):
            pre = 0.025 + rng.normal(0, 0.001, n_pre)
            post = 0.010 + rng.normal(0, 0.001, n_post)
            series_list.append(np.concatenate([pre, post]))
        power = power_across_seeds(
            series_list,
            break_index=n_pre,
            expected_direction="decrease",
        )
        assert power > 0.9

    def test_power_low_for_null(self):
        """Null effect → power near alpha (permitting generous noise)."""
        rng = np.random.default_rng(42)
        n = 60
        series_list = [
            0.02 + rng.normal(0, 0.001, n) for _ in range(100)
        ]
        power = power_across_seeds(
            series_list,
            break_index=36,
            expected_direction="decrease",
            alpha=0.05,
        )
        assert power < 0.15

    def test_wrong_direction_not_counted(self):
        """A significant increase must not count toward 'decrease' power."""
        rng = np.random.default_rng(42)
        pre = 0.010 + rng.normal(0, 0.0005, 30)
        post = 0.025 + rng.normal(0, 0.0005, 30)
        series = np.concatenate([pre, post])
        power = power_across_seeds(
            [series], break_index=30, expected_direction="decrease",
        )
        assert power == 0.0

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            power_across_seeds([], break_index=5)

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="expected_direction"):
            power_across_seeds(
                [np.zeros(10)],
                break_index=5,
                expected_direction="down",
            )


class TestCITSWithControl:
    """Lifted CITS-with-control mode matches the tier3 behaviour."""

    def test_recovers_known_level_difference(self):
        rng = np.random.default_rng(42)
        n = 52
        cf = 0.015 + rng.normal(0, 0.001, n)
        factual = cf - 0.002 + rng.normal(0, 0.001, n)
        result = cits_with_control(factual, cf)

        assert isinstance(result, CITSWithControlResult)
        assert result.effect_estimate < 0
        assert result.ci_lower <= -0.002 <= result.ci_upper
        assert result.n_observations == 2 * n

    def test_null_effect_ci_covers_zero(self):
        rng = np.random.default_rng(42)
        n = 52
        cf = 0.01 + rng.normal(0, 0.001, n)
        factual = 0.01 + rng.normal(0, 0.001, n)
        result = cits_with_control(factual, cf)
        assert abs(result.effect_estimate) < 0.002
        assert result.ci_lower <= 0 <= result.ci_upper

    def test_explicit_effect_timepoint(self):
        rng = np.random.default_rng(42)
        n = 20
        cf = 0.015 + rng.normal(0, 0.0005, n)
        factual = cf - 0.002 + rng.normal(0, 0.0005, n)
        mid = cits_with_control(factual, cf, effect_timepoint=10)
        last = cits_with_control(factual, cf)  # default n-1
        # Both estimates should be negative; different timepoints
        # generally produce different numerical values
        assert mid.effect_estimate < 0
        assert last.effect_estimate < 0

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            cits_with_control(np.zeros(10), np.zeros(12))

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 3"):
            cits_with_control(np.zeros(2), np.zeros(2))


class TestSlopeOnly:
    """Lifted slope-only mode matches the tier3 behaviour."""

    def test_detects_upward_trend(self):
        rng = np.random.default_rng(42)
        n = 52
        series = (
            0.01
            + 0.0002 * np.arange(n)
            + rng.normal(0, 0.0005, n)
        )
        result = its_slope_only(series)
        assert isinstance(result, SlopeOnlyResult)
        assert result.slope > 0
        assert abs(result.slope - 0.0002) < 0.0002
        assert result.p_value < 0.05

    def test_stable_series_slope_near_zero(self):
        rng = np.random.default_rng(42)
        series = 0.01 + rng.normal(0, 0.0005, 52)
        result = its_slope_only(series)
        assert abs(result.slope) < 1e-4
        assert result.p_value > 0.01

    def test_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 3"):
            its_slope_only(np.array([0.01, 0.02]))


class TestLiftedFunctionsMatchTier3:
    """Regression check: the lifted functions must produce the same
    numeric results as the original ``tier3_cits.py`` internals.

    If this test breaks and the tier3 version was not also updated,
    the lift has silently changed behaviour and any downstream
    monitoring validation could shift.
    """

    def test_cits_with_control_matches_tier3_fit(self):
        from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
            Tier3CITS,
        )

        rng = np.random.default_rng(42)
        n = 30
        cf = 0.015 + rng.normal(0, 0.001, n)
        factual = cf - 0.002 + rng.normal(0, 0.001, n)

        tier3 = Tier3CITS(
            mode="cits_with_cf",
            refit_interval_weeks=13,
            min_observations=n,
        )
        for week in range(n):
            tier3.update(
                week=week,
                factual_turnover=float(factual[week]),
                counterfactual_turnover=float(cf[week]),
            )
        tier3_est = tier3.estimates[-1]

        lifted = cits_with_control(factual, cf)

        assert abs(tier3_est.effect_estimate - lifted.effect_estimate) < 1e-12
        assert abs(tier3_est.ci_lower - lifted.ci_lower) < 1e-12
        assert abs(tier3_est.ci_upper - lifted.ci_upper) < 1e-12
        assert abs(tier3_est.p_value - lifted.p_value) < 1e-12

    def test_slope_only_matches_tier3_fit(self):
        from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
            Tier3CITS,
        )

        rng = np.random.default_rng(42)
        n = 30
        series = (
            0.01
            + 0.0002 * np.arange(n)
            + rng.normal(0, 0.0005, n)
        )

        tier3 = Tier3CITS(mode="its_only", min_observations=n)
        for week in range(n):
            tier3.update(
                week=week, factual_turnover=float(series[week]),
            )
        tier3_est = tier3.estimates[-1]

        lifted = its_slope_only(series)

        assert abs(tier3_est.effect_estimate - lifted.slope) < 1e-12
        assert abs(tier3_est.ci_lower - lifted.ci_lower) < 1e-12
        assert abs(tier3_est.ci_upper - lifted.ci_upper) < 1e-12
        assert abs(tier3_est.p_value - lifted.p_value) < 1e-12
