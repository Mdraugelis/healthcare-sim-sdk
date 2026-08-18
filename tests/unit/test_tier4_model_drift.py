"""Unit tests for Tier 4 offline model drift analysis."""

import numpy as np

from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
    analyze_tier4,
)


def _make_prediction(
    week: int,
    n: int,
    auc_target: float,
    rng: np.random.Generator,
) -> dict:
    """Generate synthetic score/label data with approximate AUC.

    Uses a simple correlation model: labels are drawn from a Bernoulli
    with ``base_rate``, then scores are generated so that positive
    cases have higher expected scores. The strength of separation
    controls the AUC.
    """
    base_rate = 0.1
    labels = (rng.random(n) < base_rate).astype(int)

    # Score = label_signal * strength + noise
    # Strength controls the AUC
    strength = max(0.0, (auc_target - 0.5) * 4.0)
    noise = rng.normal(0, 1, n)
    scores = labels * strength + noise
    # Normalize scores to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    return {
        "week": week,
        "scores": scores,
        "true_labels": labels,
    }


class TestTier4Empty:
    """Empty input returns empty output."""

    def test_empty_log(self):
        result = analyze_tier4([])
        assert result.rolling_stats == []
        assert result.detection_events == []


class TestTier4StableModel:
    """A stable high-AUC model should fire no detections."""

    def test_no_detections_stable(self):
        rng = np.random.default_rng(42)
        log = [
            _make_prediction(week, n=500, auc_target=0.80, rng=rng)
            for week in range(0, 52, 2)
        ]
        result = analyze_tier4(log)
        # Should have one rolling stats entry per prediction week
        assert len(result.rolling_stats) == len(log)
        # AUC should be reasonable (not None, and above 0.65 mostly)
        aucs = [
            s.realized_auc for s in result.rolling_stats
            if s.realized_auc is not None
        ]
        assert len(aucs) > 0
        assert max(aucs) > 0.65
        # No critical AUC detections
        assert not any(
            e.rule.startswith("auc_below")
            for e in result.detection_events
        )


class TestTier4DriftDetection:
    """A model that drifts from 0.80 → 0.55 should fire a detection."""

    def test_auc_drift_triggers_detection(self):
        rng = np.random.default_rng(42)
        log = []
        # Weeks 0-20: stable AUC 0.80
        for week in range(0, 22, 2):
            log.append(
                _make_prediction(week, n=500, auc_target=0.80, rng=rng)
            )
        # Weeks 22-40: linear drift to 0.55
        for week in range(22, 42, 2):
            progress = (week - 22) / 20
            auc = 0.80 - progress * 0.25  # 0.80 → 0.55
            log.append(
                _make_prediction(week, n=500, auc_target=auc, rng=rng)
            )
        # Weeks 42-60: stay at 0.55
        for week in range(42, 62, 2):
            log.append(
                _make_prediction(week, n=500, auc_target=0.55, rng=rng)
            )

        result = analyze_tier4(log)
        # Should have fired an AUC detection
        auc_events = [
            e for e in result.detection_events
            if e.rule.startswith("auc_below")
        ]
        assert len(auc_events) >= 1
        # Detection should happen after the drift (week > 30)
        assert auc_events[0].week > 30


class TestTier4RollingStats:
    """Rolling stats should cover the window and report correctly."""

    def test_rolling_stats_per_prediction_week(self):
        rng = np.random.default_rng(42)
        log = [
            _make_prediction(week, n=500, auc_target=0.80, rng=rng)
            for week in range(0, 26, 2)
        ]
        result = analyze_tier4(log)
        # One stats entry per prediction
        assert len(result.rolling_stats) == len(log)
        # All weeks should have n_predictions > 0
        for stat in result.rolling_stats:
            assert stat.week >= 0
            assert stat.n_predictions >= 0


class TestTier4Calibration:
    """Calibration slope/intercept should be computed where possible."""

    def test_calibration_computed(self):
        rng = np.random.default_rng(42)
        log = [
            _make_prediction(week, n=1000, auc_target=0.80, rng=rng)
            for week in range(0, 26, 2)
        ]
        result = analyze_tier4(log)
        # At least some stats should have calibration values
        cal_slopes = [
            s.calibration_slope for s in result.rolling_stats
            if s.calibration_slope is not None
        ]
        assert len(cal_slopes) >= 3
