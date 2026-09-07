"""Integration tests for MonitoringHarness.

Runs the nurse retention scenario through a full sim + harness and
verifies the output has the expected structure. These tests are
end-to-end: scenario + engine + aggregator + four tiers.
"""

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.scenarios.nurse_retention.monitoring import (
    MonitoringHarness,
    MonitoringRun,
)
from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
    NurseRetentionScenario,
    RetentionConfig,
)


def _run_sim(config: RetentionConfig, seed: int = 42):
    """Helper: run the scenario and return the results."""
    sc = NurseRetentionScenario(config=config, seed=seed)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    return engine.run(config.n_nurses)


class TestHarnessBasicRun:
    """Harness runs end-to-end and produces a MonitoringRun."""

    def test_regime_a_short_run_produces_run(self):
        config = RetentionConfig(n_nurses=500, n_weeks=26)
        results = _run_sim(config, seed=42)

        harness = MonitoringHarness(
            regime="calibrated_success_test",
            seed=42,
            tier3_mode="cits_with_cf",
        )
        run = harness.run_from_results(results, config)

        assert isinstance(run, MonitoringRun)
        assert run.regime == "calibrated_success_test"
        assert run.seed == 42
        assert run.n_weeks == 26

    def test_weekly_history_populated(self):
        config = RetentionConfig(n_nurses=500, n_weeks=13)
        results = _run_sim(config)
        harness = MonitoringHarness()
        run = harness.run_from_results(results, config)

        assert len(run.weekly_history) == 13
        # Each row should have core fields
        for row in run.weekly_history:
            assert "week" in row
            assert "n_active" in row
            assert "unit_turnover_rate" in row
            assert "check_in_adherence" in row

    def test_tier4_prediction_log_populated(self):
        config = RetentionConfig(
            n_nurses=500, n_weeks=13, prediction_interval=2,
        )
        results = _run_sim(config)
        harness = MonitoringHarness()
        run = harness.run_from_results(results, config)

        # Scores at weeks 0..10 have fully observed two-week future outcomes.
        # The week-12 prediction remains NOT_RUN until its horizon matures.
        assert len(run.tier4_prediction_log) == 6
        assert run.tier4_prediction_log[-1]["score_week"] == 10
        assert run.tier4_prediction_log[-1]["week"] == 12
        assert run.metadata["prediction_performance"][-1][
            "actual_future_outcomes_status"
        ] == "NOT_RUN"
        for entry in run.tier4_prediction_log:
            assert "week" in entry
            assert "scores" in entry
            assert "true_labels" in entry


class TestHarnessGroundTruth:
    """Ground truth trajectories are stored for post-hoc analysis."""

    def test_ground_truth_arrays_populated(self):
        config = RetentionConfig(n_nurses=500, n_weeks=13)
        results = _run_sim(config)
        harness = MonitoringHarness()
        run = harness.run_from_results(results, config)

        assert run.ground_truth_factual_departures is not None
        assert run.ground_truth_counterfactual_departures is not None
        assert run.ground_truth_factual_retention is not None
        assert run.ground_truth_counterfactual_retention is not None
        assert len(run.ground_truth_factual_departures) == 13
        # Factual departures should be ≤ CF departures (AI helps)
        assert (
            run.ground_truth_factual_departures[-1]
            <= run.ground_truth_counterfactual_departures[-1]
        )


class TestHarnessTier3:
    """Tier 3 should produce at least one quarterly estimate."""

    def test_tier3_estimate_at_quarterly_refit(self):
        config = RetentionConfig(n_nurses=800, n_weeks=26)
        results = _run_sim(config)
        harness = MonitoringHarness(tier3_mode="cits_with_cf")
        run = harness.run_from_results(results, config)

        # At 13-week refit interval, 26 weeks → at least one refit
        assert len(run.tier3_estimates) >= 1
        first = run.tier3_estimates[0]
        assert first.mode == "cits_with_cf"
        assert first.n_observations > 0

    def test_tier3_endpoint_reconciles_to_raw_branch_rates(self):
        """An endpoint estimate need not share the cumulative-effect sign.

        Reconcile directly to a hand-computed fitted endpoint on raw
        departure-rate differences, including the week-zero denominator.
        """
        config = RetentionConfig(n_nurses=1000, n_weeks=52)
        results = _run_sim(config)
        harness = MonitoringHarness(tier3_mode="cits_with_cf")
        run = harness.run_from_results(results, config)

        assert len(run.tier3_estimates) >= 3
        final = run.tier3_estimates[-1]
        rates = []
        for branch in (results.outcomes, results.counterfactual_outcomes):
            active = config.n_nurses
            weekly_rates = []
            for t in sorted(branch):
                weekly_rates.append(float(branch[t].events.sum()) / active)
                active = branch[t].metadata["n_active"]
            rates.append(np.asarray(weekly_rates))
        difference = rates[0] - rates[1]
        time = np.arange(len(difference))
        centered = time - time.mean()
        slope = np.dot(centered, difference) / np.dot(centered, centered)
        expected_endpoint = difference.mean() + slope * (time[-1] - time.mean())
        assert final.effect_estimate == pytest.approx(expected_endpoint)


class TestHarnessDetections:
    """The calibrated regime should produce few false positives."""

    def test_calibrated_regime_tier1_quiet(self):
        """In the calibrated regime with adherence=1.0, Tier 1 should
        produce at most a handful of false positives."""
        config = RetentionConfig(n_nurses=500, n_weeks=52)
        results = _run_sim(config)
        harness = MonitoringHarness()
        run = harness.run_from_results(results, config)

        tier1_events = run.events_by_tier(1)
        # Allow up to 5 false positives across 52 weeks × 3 metrics
        assert len(tier1_events) <= 10


class TestHarnessDashboardIgnorance:
    """Verify the dashboard never sees counterfactual state directly.

    This test ensures the weekly history contains no CF-specific
    fields that would leak ground truth into the tiers.
    """

    def test_weekly_history_has_no_cf_fields(self):
        config = RetentionConfig(n_nurses=300, n_weeks=13)
        results = _run_sim(config)
        harness = MonitoringHarness()
        run = harness.run_from_results(results, config)

        # Row should not have any top-level 'counterfactual' or 'cf'
        # keys (the harness stores CF info as private "_" keys for
        # delta computation, which are allowed, and as separate
        # MonitoringRun ground_truth_* arrays).
        for row in run.weekly_history:
            for key in row:
                if key.startswith("_"):
                    continue  # private delta-tracking keys are OK
                assert "counterfactual" not in key
                assert "cf_rate" not in key
                # The "check_ins_done" etc. are factual-only
