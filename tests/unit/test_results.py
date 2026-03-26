"""Unit tests for SimulationResults and AnalysisDataset."""

import numpy as np
import pytest

from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)


class _PanelScenario(BaseScenario[np.ndarray]):
    """Scenario that sets entity_ids for panel testing."""
    unit_of_analysis = "patient"

    def create_population(self, n):
        return self.rng.population.random(n)

    def step(self, state, t):
        return state + self.rng.temporal.normal(0, 0.01, len(state))

    def predict(self, state, t):
        return Predictions(scores=state.copy())

    def intervene(self, state, predictions, t):
        treated = predictions.scores > 0.5
        state = state.copy()
        state[treated] *= 0.8
        return state, Interventions(
            treated_indices=np.where(treated)[0]
        )

    def measure(self, state, t):
        events = (
            self.rng.outcomes.random(len(state)) < np.abs(state)
        ).astype(float)
        return Outcomes(
            events=events,
            entity_ids=np.arange(len(state)),
        )


def _run_scenario(n=50, n_t=10, preds_at=None, mode=CounterfactualMode.BRANCHED):
    if preds_at is None:
        preds_at = [3, 7]
    tc = TimeConfig(n_timesteps=n_t, timestep_duration=1/52,
                    prediction_schedule=preds_at)
    sc = _PanelScenario(time_config=tc, seed=42)
    return BranchedSimulationEngine(sc, mode).run(n)


class TestAnalysisDatasetTimeSeries:
    def test_shape(self):
        results = _run_scenario()
        ts = results.to_analysis().to_time_series()
        assert len(ts["timesteps"]) == 10
        assert len(ts["outcomes"]) == 10
        assert len(ts["treatment_indicator"]) == 10

    def test_treatment_indicator(self):
        results = _run_scenario(preds_at=[3, 7])
        ts = results.to_analysis().to_time_series()
        assert ts["treatment_indicator"][3] == 1
        assert ts["treatment_indicator"][7] == 1
        assert ts["treatment_indicator"][0] == 0

    def test_counterfactual_branch(self):
        results = _run_scenario()
        ts = results.to_analysis().to_time_series(branch="counterfactual")
        assert len(ts["outcomes"]) == 10


class TestAnalysisDatasetPanel:
    def test_shape(self):
        n, n_t = 50, 10
        results = _run_scenario(n=n, n_t=n_t)
        panel = results.to_analysis().to_panel()
        assert len(panel["entity_ids"]) == n * n_t
        assert len(panel["timesteps"]) == n * n_t
        assert len(panel["outcomes"]) == n * n_t
        assert len(panel["treated"]) == n * n_t

    def test_entity_ids_complete(self):
        n = 50
        results = _run_scenario(n=n)
        panel = results.to_analysis().to_panel()
        assert set(panel["entity_ids"]) == set(range(n))

    def test_unit_of_analysis(self):
        results = _run_scenario()
        panel = results.to_analysis().to_panel()
        assert panel["unit_of_analysis"] == "patient"

    def test_raises_without_entity_ids(self):
        """Should raise when entity_ids is None."""

        class NoIdScenario(_PanelScenario):
            def measure(self, state, t):
                return Outcomes(events=state.copy())  # no entity_ids

        tc = TimeConfig(n_timesteps=5, timestep_duration=1/52,
                        prediction_schedule=[2])
        sc = NoIdScenario(time_config=tc, seed=42)
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(10)
        with pytest.raises(ValueError, match="entity_ids"):
            results.to_analysis().to_panel()


class TestAnalysisDatasetSnapshots:
    def test_at_prediction_time(self):
        results = _run_scenario(n=50, preds_at=[3])
        snap = results.to_analysis().to_entity_snapshots(t=3)
        assert len(snap["entity_ids"]) == 50
        assert len(snap["outcomes"]) == 50
        assert snap["scores"] is not None

    def test_at_non_prediction_time(self):
        results = _run_scenario(n=50, preds_at=[3])
        snap = results.to_analysis().to_entity_snapshots(t=5)
        assert snap["scores"] is None

    def test_invalid_timestep(self):
        results = _run_scenario(n=50, n_t=10)
        with pytest.raises(ValueError):
            results.to_analysis().to_entity_snapshots(t=99)


class TestAnalysisDatasetSubgroup:
    def test_subgroup_panel(self):
        """Subgroup defaults to 'unknown' when not provided."""
        results = _run_scenario(n=50)
        panel = results.to_analysis().to_subgroup_panel()
        assert "subgroup" in panel
        assert len(panel["subgroup"]) == len(panel["entity_ids"])
