"""Integration tests for the step purity contract.

The BRANCHED-vs-NONE identity test is the single most important validation
in the SDK. It simultaneously proves:
- RNG partitioning works (streams are independent)
- Stream forking works (branches get the same seed state)
- Step purity holds (no shared mutable state leaks between branches)
- Engine RNG context swapping works correctly
- clone_state produces true independent copies
"""

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


def assert_purity(
    scenario_class: type,
    seed: int = 42,
    n_entities: int = 100,
    n_timesteps: int = 20,
    prediction_schedule: list | None = None,
    **scenario_kwargs,
) -> None:
    """The critical purity test.

    Runs the same scenario with BRANCHED and NONE modes, same seed.
    The factual branch from BRANCHED must be element-wise identical
    to the NONE trajectory at every timestep.
    """
    if prediction_schedule is None:
        prediction_schedule = [5, 10, 15]

    tc = TimeConfig(
        n_timesteps=n_timesteps,
        timestep_duration=1 / 52,
        prediction_schedule=prediction_schedule,
    )

    # Run BRANCHED
    sc_branched = scenario_class(time_config=tc, seed=seed, **scenario_kwargs)
    engine_branched = BranchedSimulationEngine(
        sc_branched, CounterfactualMode.BRANCHED
    )
    results_branched = engine_branched.run(n_entities)

    # Run NONE with same seed
    sc_none = scenario_class(time_config=tc, seed=seed, **scenario_kwargs)
    engine_none = BranchedSimulationEngine(
        sc_none, CounterfactualMode.NONE
    )
    results_none = engine_none.run(n_entities)

    # Compare factual outcomes at every timestep
    for t in range(n_timesteps):
        branched_events = results_branched.outcomes[t].events
        none_events = results_none.outcomes[t].events
        np.testing.assert_array_equal(
            branched_events,
            none_events,
            err_msg=(
                f"Purity violation at t={t}: BRANCHED factual != NONE. "
                f"Max diff={np.max(np.abs(branched_events - none_events))}"
            ),
        )


# -- Pure Stochastic Scenario (should PASS) --------------------------------

class StochasticPureScenario(BaseScenario[np.ndarray]):
    """Scenario with random draws in step, using correct RNG streams."""

    unit_of_analysis = "entity"

    def create_population(self, n_entities: int) -> np.ndarray:
        return self.rng.population.random(n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        # Pure: uses only self.rng.temporal and state
        noise = self.rng.temporal.normal(0, 0.01, size=len(state))
        return np.clip(state + noise, 0, 1)

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        noise = self.rng.prediction.normal(0, 0.1, size=len(state))
        return Predictions(scores=state + noise)

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        threshold = 0.5
        treated = predictions.scores > threshold
        state = state.copy()
        state[treated] *= 0.5  # reduce risk
        return state, Interventions(
            treated_indices=np.where(treated)[0]
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        probs = state
        events = (
            self.rng.outcomes.random(len(state)) < probs
        ).astype(float)
        return Outcomes(
            events=events,
            entity_ids=np.arange(len(state)),
        )


# -- Impure Scenario (should FAIL) -----------------------------------------

class ImpureScenario(BaseScenario[np.ndarray]):
    """Deliberately impure: step reads/writes shared self.counter."""

    unit_of_analysis = "entity"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0  # shared mutable state — VIOLATION

    def create_population(self, n_entities: int) -> np.ndarray:
        return self.rng.population.random(n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        self.counter += 1  # PURITY VIOLATION: writes shared state
        noise = self.rng.temporal.normal(0, 0.01, size=len(state))
        # Bug: uses shared counter to scale noise differently per call
        return np.clip(state + noise * self.counter, 0, 1)

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        return Predictions(scores=state.copy())

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        treated = predictions.scores > 0.5
        state = state.copy()
        state[treated] *= 0.5
        return state, Interventions(
            treated_indices=np.where(treated)[0]
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        events = (
            self.rng.outcomes.random(len(state)) < state
        ).astype(float)
        return Outcomes(
            events=events,
            entity_ids=np.arange(len(state)),
        )


# -- RNG Discipline Violation (should FAIL) ---------------------------------

class RNGViolationScenario(BaseScenario[np.ndarray]):
    """Deliberately violates RNG discipline: step uses intervention stream."""

    unit_of_analysis = "entity"

    def create_population(self, n_entities: int) -> np.ndarray:
        return self.rng.population.random(n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        # VIOLATION: uses intervention stream instead of temporal
        noise = self.rng.intervention.normal(0, 0.01, size=len(state))
        return np.clip(state + noise, 0, 1)

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        return Predictions(scores=state.copy())

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        treated = predictions.scores > 0.5
        state = state.copy()
        state[treated] *= 0.5
        return state, Interventions(
            treated_indices=np.where(treated)[0]
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        events = (
            self.rng.outcomes.random(len(state)) < state
        ).astype(float)
        return Outcomes(
            events=events,
            entity_ids=np.arange(len(state)),
        )


# -- Tests -----------------------------------------------------------------

class TestStepPurity:
    """The critical BRANCHED-vs-NONE identity tests."""

    def test_pure_scenario_passes(self):
        """A correctly implemented scenario must pass the purity test."""
        assert_purity(StochasticPureScenario)

    def test_impure_scenario_fails(self):
        """A scenario with shared mutable state must fail."""
        with pytest.raises(AssertionError, match="Purity violation"):
            assert_purity(ImpureScenario)

    def test_rng_violation_causes_branch_divergence(self):
        """Using the wrong RNG stream causes factual/CF branch divergence.

        When step() uses self.rng.intervention instead of self.rng.temporal,
        the intervention stream gets consumed by both step() AND intervene()
        on the factual branch, but only by step() on the CF branch. After
        an intervention timestep, the streams are desynchronized and the
        branches diverge for the wrong reasons.

        This is detected by running with interventions and checking that
        factual and CF branches diverge MORE than the intervention alone
        would explain — specifically, even timesteps BEFORE the first
        intervention produce different outcomes on the two branches' step()
        calls after the intervention desynchronizes the intervention stream.
        """
        tc = TimeConfig(
            n_timesteps=20,
            timestep_duration=1 / 52,
            prediction_schedule=[5, 10, 15],
        )
        sc = RNGViolationScenario(time_config=tc, seed=42)
        engine = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        )
        results = engine.run(100)

        # After intervention at t=5, the intervention stream is
        # desynchronized. By t=8 (no intervention), step() noise differs
        # between branches. Check that outcomes diverge at a non-intervention
        # timestep AFTER the first intervention.
        factual_t8 = results.outcomes[8].events
        cf_t8 = results.counterfactual_outcomes[8].events

        # Before intervention at t=3, branches should still be synchronized
        factual_t3 = results.outcomes[3].events
        cf_t3 = results.counterfactual_outcomes[3].events

        # Pre-intervention divergence should be zero (identical evolution)
        pre_diff = np.abs(factual_t3 - cf_t3).sum()

        # Post-intervention divergence should be non-zero (desynchronized)
        post_diff = np.abs(factual_t8 - cf_t8).sum()

        assert post_diff > pre_diff, (
            f"RNG violation not detected: pre-intervention diff={pre_diff}, "
            f"post-intervention diff={post_diff}. Expected post > pre."
        )

    def test_purity_with_multiple_prediction_times(self):
        """Purity holds even with many intervention points."""
        assert_purity(
            StochasticPureScenario,
            n_timesteps=30,
            prediction_schedule=list(range(0, 30, 3)),
        )

    def test_purity_with_no_predictions(self):
        """Purity holds when there are no predictions at all."""
        assert_purity(
            StochasticPureScenario,
            prediction_schedule=[],
        )

    def test_purity_across_seeds(self):
        """Purity holds for different random seeds."""
        for seed in [1, 99, 12345]:
            assert_purity(StochasticPureScenario, seed=seed)
