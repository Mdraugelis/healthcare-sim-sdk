"""Boundary condition and edge case tests.

Test the simulation at extremes where the answer is known analytically
or where the system is most likely to break. These tests complement
test_sanity.py with deeper parametric sweeps and stress conditions.

Categories:
- Population size extremes (1, 2, very large)
- Time horizon extremes (1 step, 0 predictions)
- Parameter extremes (0%, 100% rates, thresholds at limits)
- Empty/degenerate prediction schedules
- Analysis export edge cases
"""

import numpy as np
import pytest

from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.scenario import TimeConfig
from sdk.core.results import SimulationResults, AnalysisDataset
from sdk.ml.model import ControlledMLModel
from sdk.ml.performance import confusion_matrix_metrics, theoretical_ppv
from scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)
from scenarios.noshow_overbooking.scenario import (
    ClinicConfig,
    NoShowOverbookingScenario,
)

from .conftest import assert_no_nan_inf


# =====================================================================
# 1. POPULATION SIZE EXTREMES
# =====================================================================

class TestPopulationSizeExtremes:
    """Simulation must work correctly at population extremes."""

    def test_n_equals_1(self):
        """Single-entity simulation should complete without error."""
        config = StrokeConfig(
            n_patients=1, n_weeks=8,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(1)
        assert results.n_entities == 1
        for t in range(8):
            assert results.outcomes[t].events.shape == (1,)

    def test_n_equals_2(self):
        """Two-entity simulation: minimal population for statistics."""
        config = StrokeConfig(
            n_patients=2, n_weeks=8,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(2)
        assert results.n_entities == 2

    def test_large_population(self):
        """Large population (20k) should work and take reasonable time."""
        config = StrokeConfig(
            n_patients=20000, n_weeks=4,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(20000)
        assert results.n_entities == 20000
        assert results.outcomes[0].events.shape == (20000,)


# =====================================================================
# 2. TIME HORIZON EXTREMES
# =====================================================================

class TestTimeHorizonExtremes:
    """Simulation must handle extreme time configurations."""

    def test_single_timestep(self):
        """n_timesteps=1 with prediction at t=0."""
        config = StrokeConfig(
            n_patients=100, n_weeks=1,
            prediction_interval=1,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(100)
        assert len(results.outcomes) == 1
        assert 0 in results.predictions

    def test_minimal_prediction_schedule(self):
        """prediction_interval larger than n_weeks: only predict at t=0.

        StrokeConfig builds schedule as range(0, n_weeks, interval).
        With interval=100, n_weeks=10: schedule=[0] (one prediction).
        Verify engine handles this gracefully.
        """
        config = StrokeConfig(
            n_patients=100, n_weeks=10,
            prediction_interval=100,  # effectively once at t=0
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(100)
        # Only 1 prediction (at t=0) since range(0, 10, 100) = [0]
        assert len(results.predictions) == 1
        assert 0 in results.predictions
        # Outcomes should still exist at every timestep
        assert len(results.outcomes) == 10

    def test_prediction_every_timestep(self):
        """Predict at every single timestep (maximum intervention frequency)."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12,
            prediction_interval=1,  # predict every week
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)
        assert len(results.predictions) == 12
        assert len(results.interventions) == 12


# =====================================================================
# 3. PARAMETER EXTREMES
# =====================================================================

class TestParameterExtremes:
    """Test scenarios at parameter boundaries."""

    def test_100pct_effectiveness_maximum_reduction(self):
        """100% effectiveness should produce maximum event reduction."""
        config = StrokeConfig(
            n_patients=5000, n_weeks=26,
            prediction_interval=4,
            intervention_effectiveness=1.0,
            annual_incident_rate=0.10,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(5000)

        f_total = sum(
            results.outcomes[t].events.sum() for t in range(26)
        )
        cf_total = sum(
            results.counterfactual_outcomes[t].events.sum()
            for t in range(26)
        )
        # With 100% effectiveness, treated patients should have zero risk
        # So factual should be strictly less
        assert f_total < cf_total, (
            f"100% effectiveness: factual={f_total} should be < "
            f"counterfactual={cf_total}"
        )

    def test_very_high_incident_rate(self):
        """Very high incidence (50%/year) should produce many events."""
        config = StrokeConfig(
            n_patients=1000, n_weeks=52,
            annual_incident_rate=0.50,
            prediction_interval=52,
            intervention_effectiveness=0.0,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(1000)

        total = sum(
            results.outcomes[t].events.sum() for t in range(52)
        )
        # At 50% annual rate, expect ~500 events
        assert total > 100, (
            f"High incidence should produce many events, got {total}"
        )

    def test_very_low_incident_rate(self):
        """Very low incidence (0.1%/year) should produce very few events."""
        config = StrokeConfig(
            n_patients=1000, n_weeks=12,
            annual_incident_rate=0.001,
            prediction_interval=12,
            intervention_effectiveness=0.0,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(1000)

        total = sum(
            results.outcomes[t].events.sum() for t in range(12)
        )
        assert total < 20, (
            f"Very low incidence should produce few events, got {total}"
        )


# =====================================================================
# 4. ANALYSIS EXPORT EDGE CASES
# =====================================================================

class TestAnalysisExportEdgeCases:
    """Analysis exports must handle edge cases gracefully."""

    def test_panel_requires_entity_ids(self):
        """to_panel() should raise ValueError if entity_ids is None."""
        # Create results with None entity_ids
        from sdk.core.scenario import Outcomes
        results = SimulationResults(
            n_entities=5,
            time_config=TimeConfig(
                n_timesteps=1, timestep_duration=1 / 52,
                prediction_schedule=[],
            ),
            counterfactual_mode="none",
        )
        results.record_outcomes(
            0, Outcomes(events=np.array([0, 1, 0, 1, 0]),
                        entity_ids=None)
        )
        analysis = results.to_analysis()
        with pytest.raises(ValueError, match="entity_ids"):
            analysis.to_panel()

    def test_entity_snapshots_invalid_timestep(self):
        """to_entity_snapshots() at non-existent timestep should raise."""
        config = StrokeConfig(
            n_patients=100, n_weeks=4,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        ).run(100)
        analysis = results.to_analysis()

        with pytest.raises(ValueError, match="No outcomes"):
            analysis.to_entity_snapshots(t=999)

    def test_time_series_correct_length(self):
        """Time series should have exactly n_timesteps entries."""
        config = StrokeConfig(
            n_patients=100, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(100)
        analysis = results.to_analysis()

        ts = analysis.to_time_series()
        assert len(ts["timesteps"]) == 12
        assert len(ts["outcomes"]) == 12
        assert len(ts["treatment_indicator"]) == 12

    def test_panel_data_dimensions(self):
        """Panel data should have n_entities * n_timesteps rows."""
        config = StrokeConfig(
            n_patients=50, n_weeks=8,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(50)
        analysis = results.to_analysis()

        panel = analysis.to_panel()
        expected_rows = 50 * 8
        assert len(panel["entity_ids"]) == expected_rows, (
            f"Panel should have {expected_rows} rows, "
            f"got {len(panel['entity_ids'])}"
        )
        assert len(panel["outcomes"]) == expected_rows
        assert len(panel["treated"]) == expected_rows

    def test_treatment_indicator_binary(self):
        """Treatment indicator should be strictly 0 or 1."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)

        indicator = results.get_treatment_indicator()
        unique = np.unique(indicator)
        assert np.all(np.isin(unique, [0.0, 1.0])), (
            f"Treatment indicator has non-binary values: {unique}"
        )

    def test_counterfactual_time_series_export(self):
        """Counterfactual branch time series should export correctly."""
        config = StrokeConfig(
            n_patients=200, n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(200)
        analysis = results.to_analysis()

        cf_ts = analysis.to_time_series(branch="counterfactual")
        assert len(cf_ts["outcomes"]) == 12
        assert_no_nan_inf(cf_ts["outcomes"], "CF time series")


# =====================================================================
# 5. ML MODEL BOUNDARY CONDITIONS
# =====================================================================

class TestMLModelBoundaries:
    """ML model must handle extreme operating conditions."""

    def test_fit_with_tiny_population(self):
        """Fitting on very small sample (n=50) should not crash."""
        rng = np.random.default_rng(42)
        n = 50
        risks = rng.random(n)
        labels = (rng.random(n) > 0.7).astype(int)
        model = ControlledMLModel(mode="discrimination", target_auc=0.75)
        # Should complete without error
        report = model.fit(labels, risks, rng, n_iterations=2)
        assert "achieved_auc" in report

    def test_fit_with_imbalanced_classes(self):
        """Fitting with very imbalanced classes (1% positive)."""
        rng = np.random.default_rng(42)
        n = 5000
        risks = rng.random(n)
        labels = np.zeros(n, dtype=int)
        labels[:50] = 1  # 1% positive
        model = ControlledMLModel(mode="discrimination", target_auc=0.75)
        report = model.fit(labels, risks, rng, n_iterations=3)
        assert np.isfinite(report["achieved_auc"])

    def test_threshold_sweep_boundary(self):
        """Confusion matrix should handle threshold at exact score values."""
        rng = np.random.default_rng(42)
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2])

        # Test at exact score values
        for t in y_pred:
            m = confusion_matrix_metrics(y_true, y_pred, threshold=float(t))
            total = m["tp"] + m["fp"] + m["tn"] + m["fn"]
            assert total == len(y_true), (
                f"Confusion matrix doesn't sum to n at threshold={t}"
            )

    def test_ppv_extreme_prevalences(self):
        """Theoretical PPV at extreme prevalences should be reasonable."""
        # Very low prevalence
        ppv = theoretical_ppv(0.001, 0.80, 0.99)
        assert 0 <= ppv <= 1, f"PPV out of range: {ppv}"
        assert np.isfinite(ppv)

        # Very high prevalence
        ppv = theoretical_ppv(0.999, 0.80, 0.99)
        assert 0 <= ppv <= 1, f"PPV out of range: {ppv}"
        assert np.isfinite(ppv)
