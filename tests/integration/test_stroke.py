"""Integration tests for the Stroke Prevention scenario."""

import numpy as np
import pytest
from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
    StrokeConfig,
    StrokePreventionScenario,
)
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from tests.integration.test_step_purity import assert_purity


class TestStrokePurity:
    def test_branched_vs_none_identity(self):
        """The critical purity test on the real stroke scenario."""

        class _StrokeForPurity(StrokePreventionScenario):
            """Wrapper to match assert_purity interface."""

            def __init__(self, time_config, seed=None):
                config = StrokeConfig(
                    allow_infeasible_model=True,
                    n_patients=500,
                    n_weeks=time_config.n_timesteps,
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
            allow_infeasible_model=True,
            n_patients=5000,
            n_weeks=52,
            prediction_interval=4,
            intervention_effectiveness=0.50,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        engine = BranchedSimulationEngine(
            sc,
            CounterfactualMode.BRANCHED,
        )
        results = engine.run(5000)

        factual = sum(results.outcomes[t].events.sum() for t in range(52))
        cf = sum(results.counterfactual_outcomes[t].events.sum() for t in range(52))
        assert factual < cf, (
            f"Intervention should reduce incidents: "
            f"factual={factual}, counterfactual={cf}"
        )

    def test_analysis_exports(self):
        """All 4 AnalysisDataset methods work on stroke results."""
        config = StrokeConfig(
            allow_infeasible_model=True,
            n_patients=200,
            n_weeks=12,
            prediction_interval=4,
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc,
            CounterfactualMode.BRANCHED,
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
            allow_infeasible_model=True,
            n_patients=10_000,
            n_weeks=52,
            annual_incident_rate=0.05,
            prediction_interval=52,  # no intervention effect
        )
        sc = StrokePreventionScenario(config=config, seed=42)
        results = BranchedSimulationEngine(
            sc,
            CounterfactualMode.NONE,
        ).run(10_000)

        total_incidents = sum(results.outcomes[t].events.sum() for t in range(52))
        # Expected: ~0.05 per patient per year = ~500 for 10k patients
        observed_rate = total_incidents / 10_000
        assert 0.02 < observed_rate < 0.10, (
            f"Observed annual rate {observed_rate:.3f} outside "
            f"expected range for target 0.05"
        )


class TestStrokeReferenceFixes:
    """Blocker 3 (14 Nov review): runs as shipped, effect once, one stroke."""

    @staticmethod
    def _run(**kw):
        from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
            StrokeConfig, StrokePreventionScenario,
        )
        cfg = StrokeConfig(n_patients=4000, **kw)
        sc = StrokePreventionScenario(cfg, seed=11)
        return sc, BranchedSimulationEngine(sc).run(cfg.n_patients)

    def test_defaults_run_and_hit_auc(self):
        sc, _ = self._run()
        assert sc.model_status == "fitted"
        assert abs(sc._classifier._fit_report["achieved_auc"] - 0.75) < 0.03

    def test_treatment_applied_once_per_patient(self):
        sc, res = self._run()
        starts = np.concatenate(
            [iv.treated_indices for iv in res.interventions.values()])
        assert len(starts) == len(np.unique(starts)), "a patient started twice"
        refl = sum(iv.metadata["n_flagged_already_treated"]
                   for iv in res.interventions.values())
        assert refl > 0, "no re-flagging occurred; the guard was not exercised"

    def test_effect_multiplier_never_compounds(self):
        from healthcare_sim_sdk.scenarios.stroke_prevention.scenario import (
            CURRENT_RISKS, TX_EFFECT, StrokeConfig, StrokePreventionScenario,
        )
        from healthcare_sim_sdk.core.scenario import Predictions
        sc = StrokePreventionScenario(StrokeConfig(n_patients=50), seed=3)
        state = sc.create_population(50)
        flags = Predictions(scores=np.ones(50), labels=np.ones(50, dtype=int))
        state, _ = sc.intervene(state, flags, 0)
        state, again = sc.intervene(state, flags, 4)
        assert np.all(state[TX_EFFECT] == 0.5)
        assert len(again.treated_indices) == 0
        np.testing.assert_allclose(state[CURRENT_RISKS],
                                   state[0] * 0.5, rtol=1e-12)

    def test_stroke_ends_follow_up(self):
        _, res = self._run(annual_incident_rate=0.30)
        per_patient = sum(o.events for o in res.outcomes.values())
        cf_per_patient = sum(o.events for o in res.counterfactual_outcomes.values())
        assert per_patient.max() == 1 and cf_per_patient.max() == 1
        assert per_patient.sum() > 100  # high incidence; the cap is exercised

    def test_zero_effectiveness_branches_identical(self):
        _, res = self._run(intervention_effectiveness=0.0)
        for t in res.outcomes:
            np.testing.assert_array_equal(
                res.outcomes[t].events, res.counterfactual_outcomes[t].events)

    def test_no_events_without_model_when_allowed(self):
        sc, res = self._run(annual_incident_rate=0.0001,
                            allow_infeasible_model=True)
        assert sc.model_status == "not fitted: too few events"
        assert all(len(iv.treated_indices) == 0
                   for iv in res.interventions.values())

    def test_too_few_events_raises_by_default(self):
        with pytest.raises(ValueError, match="too low to fit a model"):
            self._run(annual_incident_rate=0.0001)
