"""Unit tests for state management (clone_state)."""

from dataclasses import dataclass

import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario, Interventions, Outcomes, Predictions, TimeConfig,
)


# Concrete scenario for testing clone_state
class _DummyScenario(BaseScenario[np.ndarray]):
    unit_of_analysis = "test_entity"

    def create_population(self, n_entities: int) -> np.ndarray:
        return self.rng.population.random(n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        return state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        return Predictions(scores=state.copy())

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        return state, Interventions(treated_indices=np.array([]))

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        return Outcomes(events=state.copy())


def _make_scenario():
    tc = TimeConfig(n_timesteps=10, timestep_duration=1 / 52)
    return _DummyScenario(time_config=tc, seed=42)


class TestCloneStateArrays:
    def test_mutation_isolation_numpy(self):
        sc = _make_scenario()
        original = np.array([1.0, 2.0, 3.0])
        cloned = sc.clone_state(original)
        cloned[0] = 999.0
        assert original[0] == 1.0

    def test_values_equal_after_clone(self):
        sc = _make_scenario()
        original = np.array([1.0, 2.0, 3.0])
        cloned = sc.clone_state(original)
        np.testing.assert_array_equal(original, cloned)


class TestCloneStateDataclass:
    def test_mutation_isolation_dataclass(self):
        @dataclass
        class State:
            risks: np.ndarray
            flags: np.ndarray

        sc = _make_scenario()
        original = State(
            risks=np.array([0.1, 0.2]),
            flags=np.array([0, 1]),
        )
        cloned = sc.clone_state(original)
        cloned.risks[0] = 999.0
        assert original.risks[0] == 0.1

    def test_values_equal_after_clone(self):
        @dataclass
        class State:
            risks: np.ndarray

        sc = _make_scenario()
        original = State(risks=np.array([0.5, 0.6]))
        cloned = sc.clone_state(original)
        np.testing.assert_array_equal(original.risks, cloned.risks)


class TestCloneStateNestedDict:
    def test_mutation_isolation_nested_dict(self):
        sc = _make_scenario()
        original = {
            "patients": {
                "risks": np.array([0.1, 0.2]),
                "names": ["a", "b"],
            },
            "schedule": np.array([[1, 2], [3, 4]]),
        }
        cloned = sc.clone_state(original)
        cloned["patients"]["risks"][0] = 999.0
        assert original["patients"]["risks"][0] == 0.1

    def test_deep_nested_mutation(self):
        sc = _make_scenario()
        original = {"level1": {"level2": {"data": np.array([1.0])}}}
        cloned = sc.clone_state(original)
        cloned["level1"]["level2"]["data"][0] = 999.0
        assert original["level1"]["level2"]["data"][0] == 1.0


class TestCloneStateOverride:
    def test_custom_clone_state(self):
        """Scenarios can override clone_state for optimized copying."""

        class OptimizedScenario(_DummyScenario):
            def clone_state(self, state: np.ndarray) -> np.ndarray:
                return state.copy()  # faster than deepcopy for arrays

        tc = TimeConfig(n_timesteps=10, timestep_duration=1 / 52)
        sc = OptimizedScenario(time_config=tc, seed=42)
        original = np.array([1.0, 2.0, 3.0])
        cloned = sc.clone_state(original)
        cloned[0] = 999.0
        assert original[0] == 1.0
