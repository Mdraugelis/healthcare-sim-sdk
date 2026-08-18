"""Unit tests for Tier 1 Shewhart XmR control chart."""

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
    ShewhartChart,
    Tier1Shewhart,
)


class TestShewhartBaseline:
    """Before the baseline is complete, no events fire."""

    def test_no_events_during_baseline(self):
        chart = ShewhartChart(metric="adherence", baseline_weeks=8)
        for week in range(7):
            events = chart.update(week, 1.0)
            assert events == []

    def test_limits_set_after_baseline(self):
        chart = ShewhartChart(metric="adherence", baseline_weeks=8)
        for week in range(8):
            chart.update(week, 1.0)
        # Constant baseline: xbar=1.0, sigma_hat=0 (no MR)
        assert chart.xbar == 1.0
        assert chart.sigma_hat == 0.0
        assert chart.centerline == 1.0

    def test_sigma_hat_nonzero_with_variation(self):
        chart = ShewhartChart(metric="x", baseline_weeks=8)
        # Baseline with some variation
        rng = np.random.default_rng(42)
        for week in range(8):
            chart.update(week, 1.0 + rng.normal(0, 0.1))
        assert chart.sigma_hat is not None
        assert chart.sigma_hat > 0
        assert chart.ucl > chart.centerline
        assert chart.lcl < chart.centerline


class TestWesternElectricRule1:
    """WE1: one point outside 3-sigma limits fires a critical event."""

    def test_we1_detects_upper_outlier(self):
        chart = ShewhartChart(metric="x", baseline_weeks=8)
        # Stable baseline with small variation
        rng = np.random.default_rng(42)
        for week in range(8):
            chart.update(week, 1.0 + rng.normal(0, 0.01))
        # Clean weeks (values stay within limits)
        for week in range(8, 16):
            events = chart.update(week, 1.0)
            # Might or might not trigger WE2 — expect no WE1
            assert not any(
                e.rule == "WE1_3sigma_upper"
                or e.rule == "WE1_3sigma_lower"
                for e in events
            )
        # Big upward spike far above UCL
        events = chart.update(16, 10.0)
        assert any(e.rule == "WE1_3sigma_upper" for e in events)

    def test_we1_detects_lower_outlier(self):
        chart = ShewhartChart(metric="adherence", baseline_weeks=8)
        rng = np.random.default_rng(42)
        for week in range(8):
            chart.update(week, 1.0 + rng.normal(0, 0.01))
        # Collapse to zero (big drop)
        events = chart.update(8, 0.0)
        assert any(e.rule == "WE1_3sigma_lower" for e in events)
        # The event should be critical severity and direction down
        we1 = next(e for e in events if e.rule == "WE1_3sigma_lower")
        assert we1.severity == "critical"
        assert we1.direction == "down"
        assert we1.week == 8


class TestWesternElectricRule2:
    """WE2: 8 consecutive points on one side of centerline."""

    def test_we2_detects_8_consecutive_above(self):
        chart = ShewhartChart(
            metric="x",
            baseline_weeks=8,
            we_rules=("WE2",),  # WE1 disabled to isolate
        )
        # Baseline clustered near 1.0 (will have small nonzero sigma)
        for week in range(8):
            chart.update(week, 1.0 + (week % 2) * 0.01)
        # Now 8 consecutive points slightly above centerline but
        # within 3-sigma limits
        cp = chart.centerline
        for week in range(8, 15):
            events = chart.update(week, cp + 0.005)
            # Shouldn't fire yet (need 8 in a row; this is #1..7)
            assert not any(
                e.rule == "WE2_8consecutive_above" for e in events
            )
        # 8th consecutive point above centerline
        events = chart.update(15, cp + 0.005)
        assert any(
            e.rule == "WE2_8consecutive_above" for e in events
        )
        we2 = next(
            e for e in events if e.rule == "WE2_8consecutive_above"
        )
        assert we2.direction == "up"
        assert we2.severity == "warning"

    def test_we2_resets_after_firing(self):
        chart = ShewhartChart(
            metric="x",
            baseline_weeks=8,
            we_rules=("WE2",),
        )
        for week in range(8):
            chart.update(week, 1.0 + (week % 2) * 0.01)
        cp = chart.centerline
        # 8 above → fire
        for week in range(8, 16):
            chart.update(week, cp + 0.005)
        # After firing, counter resets; next week above shouldn't fire
        events = chart.update(16, cp + 0.005)
        assert not any(
            e.rule == "WE2_8consecutive_above" for e in events
        )


class TestTier1Orchestrator:
    """The Tier1Shewhart wrapper manages multiple named charts."""

    def test_add_and_update_multiple_metrics(self):
        tier = Tier1Shewhart()
        tier.add_metric("adherence", baseline_weeks=8)
        tier.add_metric("adoption", baseline_weeks=8)
        # Walk 8 weeks of baseline
        for week in range(8):
            tier.update(
                week,
                {
                    ("adherence", None): 1.0,
                    ("adoption", None): 1.0,
                },
            )
        # Now adherence drops sharply (regime D surrogate)
        events = tier.update(
            8,
            {
                ("adherence", None): 0.33,
                ("adoption", None): 1.0,
            },
        )
        # At zero sigma (constant baseline), any deviation triggers
        # WE1 because ucl == lcl == centerline
        assert len(events) >= 1
        assert all(e.metric == "adherence" for e in events)

    def test_per_manager_charts(self):
        tier = Tier1Shewhart()
        tier.add_metric("check_ins", baseline_weeks=8, unit_id=0)
        tier.add_metric("check_ins", baseline_weeks=8, unit_id=1)
        # Manager 0 stable, manager 1 collapses
        rng = np.random.default_rng(42)
        for week in range(8):
            tier.update(
                week,
                {
                    ("check_ins", 0): 4.0 + rng.normal(0, 0.05),
                    ("check_ins", 1): 4.0 + rng.normal(0, 0.05),
                },
            )
        # Manager 1 goes to 0
        events = tier.update(
            8,
            {
                ("check_ins", 0): 4.0,
                ("check_ins", 1): 0.0,
            },
        )
        # Only manager 1 should have fired
        assert any(
            e.unit_id == 1 and e.metric == "check_ins"
            for e in events
        )
        assert not any(
            e.unit_id == 0 for e in events
        )
