"""Integration tests for the Stroke Prevention scenario."""

from scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)
from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from tests.integration.test_step_purity import assert_purity


class TestStrokePurity:
    def test_branched_vs_none_identity(self):
        """The critical purity test on the real stroke scenario."""

        class _StrokeForPurity(StrokePreventionScenario):
            """Wrapper to match assert_purity interface."""
            def __init__(self, time_config, seed=None):
                config = StrokeConfig(
                    n_patients=500, n_weeks=time_config.n_timesteps,
                    prediction_interval=4,
                )
                super().__init__(config=config, seed=seed)
                self.time_config = time_config

        assert_purity(
            _StrokeForPurity,
            n_entities=500,
            n_timesteps=20,
            prediction_schedule=list(range(0, 20, 4)),
        )


class TestStrokeIntervention:
    def test_intervention_reduces_incidents(self):
        """Factual should have fewer incidents than counterfactual."""
        config = StrokeConfig(
            n_patients=5000, n_weeks=52, prediction_interval=4,
            intervention_effectiveness=0.50,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED,
        )
        results = engine.run(5000)

        factual = sum(
            results.outcomes[t].events.sum()
            for t in range(52)
        )
        cf = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(52)
        )
        assert factual < cf, (
            f"Intervention should reduce incidents: "
            f"factual={factual}, counterfactual={cf}"
        )

    def test_analysis_exports(self):
        """All 4 AnalysisDataset methods work on stroke results."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12, prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED,
        ).run(200)

        analysis = results.to_analysis()

        ts = analysis.to_time_series()
        assert len(ts["outcomes"]) == 12

        panel = analysis.to_panel()
        assert len(panel["entity_ids"]) == 200 * 12
        assert panel["unit_of_analysis"] == "patient"

        snap = analysis.to_entity_snapshots(t=0)
        assert len(snap["entity_ids"]) == 200
        assert snap["scores"] is not None

    def test_population_rate(self):
        """Population incident rate should be near the configured rate."""
        config = StrokeConfig(
            n_patients=10_000, n_weeks=52,
            annual_incident_rate=0.05,
            prediction_interval=52,  # no intervention effect
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE,
        ).run(10_000)

        total_incidents = sum(
            results.outcomes[t].events.sum()
            for t in range(52)
        )
        # Expected: ~0.05 per patient per year = ~500 for 10k patients
        observed_rate = total_incidents / 10_000
        assert 0.02 < observed_rate < 0.10, (
            f"Observed annual rate {observed_rate:.3f} outside "
            f"expected range for target 0.05"
        )
