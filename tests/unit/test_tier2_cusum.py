"""Unit tests for Tier 2 CUSUM chart."""

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
    Tier2CUSUM,
)


class TestCUSUMBaseline:
    """Baseline initialization: mu0, sigma, k, h are set correctly."""

    def test_baseline_initialization(self):
        cusum = Tier2CUSUM(
            metric="turnover",
            baseline_weeks=8,
            k_multiplier=0.5,
            h_multiplier=4.0,
        )
        rng = np.random.default_rng(42)
        for week in range(8):
            cusum.update(week, 0.01 + rng.normal(0, 0.002))
        # After baseline, mu0 and sigma should be set
        assert cusum.mu0 is not None
        assert abs(cusum.mu0 - 0.01) < 0.01
        assert cusum.sigma is not None
        assert cusum.sigma > 0
        assert cusum.k == 0.5 * cusum.sigma
        assert cusum.h == 4.0 * cusum.sigma

    def test_no_events_during_baseline(self):
        cusum = Tier2CUSUM(metric="x", baseline_weeks=8)
        for week in range(8):
            # Large value doesn't fire during baseline
            events = cusum.update(week, 0.5)
            assert events == []


class TestCUSUMDriftDetection:
    """CUSUM should catch gradual drift before Shewhart can."""

    def test_upward_drift_detection(self):
        cusum = Tier2CUSUM(
            metric="turnover",
            baseline_weeks=8,
            k_multiplier=0.5,
            h_multiplier=4.0,
        )
        rng = np.random.default_rng(42)

        # Baseline: stable around 0.01 with small noise
        for week in range(8):
            cusum.update(week, 0.01 + rng.normal(0, 0.0005))

        # Drift upward at 0.0002 per week over 20 weeks
        detected_week = None
        for week in range(8, 28):
            drift = 0.01 + (week - 8) * 0.0005
            events = cusum.update(
                week, drift + rng.normal(0, 0.0005),
            )
            if any(e.rule == "cusum_upper" for e in events):
                detected_week = week
                break
        # CUSUM should catch this within 15-20 weeks of drift start
        assert detected_week is not None
        assert detected_week <= 25

    def test_step_change_detection(self):
        cusum = Tier2CUSUM(
            metric="turnover",
            baseline_weeks=8,
            k_multiplier=0.5,
            h_multiplier=4.0,
        )
        rng = np.random.default_rng(42)
        for week in range(8):
            cusum.update(week, 0.01 + rng.normal(0, 0.001))
        # Big step change
        detected = False
        for week in range(8, 20):
            events = cusum.update(
                week, 0.05 + rng.normal(0, 0.001),
            )
            if any(e.rule == "cusum_upper" for e in events):
                detected = True
                break
        assert detected

    def test_stable_baseline_no_detection(self):
        cusum = Tier2CUSUM(
            metric="x",
            baseline_weeks=8,
        )
        rng = np.random.default_rng(42)
        all_events = []
        for week in range(60):
            events = cusum.update(
                week, 0.01 + rng.normal(0, 0.0005),
            )
            all_events.extend(events)
        # Some may fire due to random noise; at 0.5σ k and 4σ h,
        # ARL at 0 shift is very long. With only 60 weeks and a
        # stable process, we expect zero or few false alarms.
        assert len(all_events) <= 1


class TestCUSUMReset:
    """After a detection, the CUSUM arm should reset to zero."""

    def test_upper_resets_after_detection(self):
        cusum = Tier2CUSUM(
            metric="x",
            baseline_weeks=8,
        )
        rng = np.random.default_rng(42)
        for week in range(8):
            cusum.update(week, 0.01 + rng.normal(0, 0.001))
        # Push upper above h
        for week in range(8, 25):
            cusum.update(week, 0.05)
        # Once firing and reset, if we go back to baseline, the
        # c_upper should be at or near 0
        for week in range(25, 35):
            cusum.update(week, 0.01)
        assert cusum.c_upper < cusum.h


class TestCUSUMDirection:
    """Two-sided CUSUM tracks both directions independently."""

    def test_downward_drift_fires_lower(self):
        cusum = Tier2CUSUM(
            metric="x",
            baseline_weeks=8,
        )
        rng = np.random.default_rng(42)
        for week in range(8):
            cusum.update(week, 0.05 + rng.normal(0, 0.001))
        # Downward step (good for turnover but tests lower arm)
        detected = False
        for week in range(8, 20):
            events = cusum.update(
                week, 0.01 + rng.normal(0, 0.001),
            )
            if any(e.rule == "cusum_lower" for e in events):
                detected = True
                break
        assert detected

    def test_upper_lower_independent(self):
        cusum = Tier2CUSUM(metric="x", baseline_weeks=8)
        rng = np.random.default_rng(42)
        for week in range(8):
            cusum.update(week, 0.01 + rng.normal(0, 0.001))
        # Oscillate — neither arm should accumulate much
        for week in range(8, 30):
            val = 0.01 + (0.003 if week % 2 == 0 else -0.003)
            cusum.update(week, val)
        # Neither arm should cross the threshold
        assert cusum.c_upper < cusum.h
        assert cusum.c_lower < cusum.h
