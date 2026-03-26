"""Unit tests for BranchedSimulationEngine."""

import numpy as np

from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)


# -- Toy Counter Scenario -------------------------------------------------

class CounterScenario(BaseScenario[np.ndarray]):
    """Trivial scenario for engine testing.

    State is a 1-D array of counters (one per entity).
    - step: increment each counter by 1
    - predict: return counter values as scores
    - intervene: add 10 to entities above threshold (score > 0)
    - measure: record current counter values
    """

    unit_of_analysis = "counter"

    def create_population(self, n_entities: int) -> np.ndarray:
        return np.zeros(n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        return state + 1

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        return Predictions(scores=state.copy())

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        treated = predictions.scores > 0
        state = state.copy()
        state[treated] += 10
        return state, Interventions(
            treated_indices=np.where(treated)[0]
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        return Outcomes(
            events=state.copy(),
            entity_ids=np.arange(len(state)),
        )


def _make_counter_scenario(
    n_timesteps: int = 10,
    prediction_schedule: list | None = None,
    seed: int = 42,
) -> CounterScenario:
    if prediction_schedule is None:
        prediction_schedule = [5]
    tc = TimeConfig(
        n_timesteps=n_timesteps,
        timestep_duration=1 / 52,
        prediction_schedule=prediction_schedule,
    )
    return CounterScenario(time_config=tc, seed=seed)


# -- Hook Ordering Tests ---------------------------------------------------

class LoggingScenario(BaseScenario[np.ndarray]):
    """Scenario that logs method calls for ordering verification."""

    unit_of_analysis = "test"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_log: list[str] = []

    def create_population(self, n_entities: int) -> np.ndarray:
        self.call_log.append("create_population")
        return np.zeros(n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        self.call_log.append(f"step({t})")
        return state + 1

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        self.call_log.append(f"predict({t})")
        return Predictions(scores=state.copy())

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        self.call_log.append(f"intervene({t})")
        return state, Interventions(treated_indices=np.array([]))

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        self.call_log.append(f"measure({t})")
        return Outcomes(events=state.copy())


class TestHookOrdering:
    def test_step_before_predict_before_intervene_before_measure(self):
        tc = TimeConfig(
            n_timesteps=3,
            timestep_duration=1 / 52,
            prediction_schedule=[1],
        )
        sc = LoggingScenario(time_config=tc, seed=42)
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        )
        engine.run(2)

        # At t=1: step, predict, intervene, measure
        t1_start = sc.call_log.index("step(1)")
        t1_predict = sc.call_log.index("predict(1)")
        t1_intervene = sc.call_log.index("intervene(1)")
        t1_measure = sc.call_log.index("measure(1)")

        assert t1_start < t1_predict < t1_intervene < t1_measure

    def test_predict_only_at_scheduled_times(self):
        tc = TimeConfig(
            n_timesteps=5,
            timestep_duration=1 / 52,
            prediction_schedule=[2, 4],
        )
        sc = LoggingScenario(time_config=tc, seed=42)
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        )
        engine.run(2)

        predict_calls = [c for c in sc.call_log if c.startswith("predict")]
        assert predict_calls == ["predict(2)", "predict(4)"]


# -- Clock Tests -----------------------------------------------------------

class TestClock:
    def test_timesteps_advance_correctly(self):
        tc = TimeConfig(
            n_timesteps=5,
            timestep_duration=1 / 52,
            prediction_schedule=[],
        )
        sc = LoggingScenario(time_config=tc, seed=42)
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.NONE
        )
        engine.run(2)

        step_calls = [c for c in sc.call_log if c.startswith("step")]
        assert step_calls == [
            "step(0)", "step(1)", "step(2)", "step(3)", "step(4)"
        ]


# -- CounterfactualMode.NONE Tests ----------------------------------------

class TestNoneMode:
    def test_single_trajectory(self):
        sc = _make_counter_scenario(n_timesteps=5, prediction_schedule=[])
        engine = BranchedSimulationEngine(sc, CounterfactualMode.NONE)
        results = engine.run(3)

        assert results.counterfactual_mode == "none"
        assert len(results.outcomes) == 5
        assert len(results.counterfactual_outcomes) == 0

    def test_counter_values_no_intervention(self):
        sc = _make_counter_scenario(n_timesteps=5, prediction_schedule=[])
        engine = BranchedSimulationEngine(sc, CounterfactualMode.NONE)
        results = engine.run(3)

        # After 5 steps, each counter should be 5
        final = results.outcomes[4].events
        np.testing.assert_array_equal(final, [5, 5, 5])


# -- CounterfactualMode.BRANCHED Tests ------------------------------------

class TestBranchedMode:
    def test_factual_has_intervention_effect(self):
        sc = _make_counter_scenario(
            n_timesteps=10, prediction_schedule=[5]
        )
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        )
        results = engine.run(3)

        # After t=5: step gives 6, then intervene adds 10 = 16
        # After t=9: 16 + 4 more steps = 20
        final_factual = results.outcomes[9].events
        np.testing.assert_array_equal(final_factual, [20, 20, 20])

    def test_counterfactual_has_no_intervention(self):
        sc = _make_counter_scenario(
            n_timesteps=10, prediction_schedule=[5]
        )
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        )
        results = engine.run(3)

        # Counterfactual: just 10 steps, no intervention
        final_cf = results.counterfactual_outcomes[9].events
        np.testing.assert_array_equal(final_cf, [10, 10, 10])

    def test_results_have_both_branches(self):
        sc = _make_counter_scenario(n_timesteps=5, prediction_schedule=[2])
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        )
        results = engine.run(3)

        assert len(results.outcomes) == 5
        assert len(results.counterfactual_outcomes) == 5

    def test_predictions_recorded(self):
        sc = _make_counter_scenario(
            n_timesteps=10, prediction_schedule=[3, 7]
        )
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        )
        results = engine.run(3)

        assert set(results.predictions.keys()) == {3, 7}
        assert set(results.interventions.keys()) == {3, 7}


# -- CounterfactualMode.SNAPSHOT Tests ------------------------------------

class TestSnapshotMode:
    def test_counterfactual_only_at_prediction_times(self):
        sc = _make_counter_scenario(
            n_timesteps=10, prediction_schedule=[3, 7]
        )
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.SNAPSHOT
        )
        results = engine.run(3)

        assert set(results.counterfactual_outcomes.keys()) == {3, 7}

    def test_factual_has_all_timesteps(self):
        sc = _make_counter_scenario(
            n_timesteps=10, prediction_schedule=[3, 7]
        )
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.SNAPSHOT
        )
        results = engine.run(3)

        assert len(results.outcomes) == 10


# -- Metadata Tests --------------------------------------------------------

class TestMetadata:
    def test_unit_of_analysis_propagated(self):
        sc = _make_counter_scenario()
        engine = BranchedSimulationEngine(sc, CounterfactualMode.NONE)
        results = engine.run(3)
        assert results.unit_of_analysis == "counter"

    def test_validation_recorded(self):
        sc = _make_counter_scenario()
        engine = BranchedSimulationEngine(sc, CounterfactualMode.NONE)
        results = engine.run(3)
        assert "population" in results.validations
        assert "final" in results.validations
