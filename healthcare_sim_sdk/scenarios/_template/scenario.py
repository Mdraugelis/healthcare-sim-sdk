"""Scenario Template — Coin Flip with Bias.

A minimal scenario demonstrating the 5-method contract.
Use this as a starting point for new scenarios.

Unit of analysis: entity
State: numpy array of flip probabilities

RNG DISCIPLINE CHECKLIST:
- create_population() -> self.rng.population
- step()              -> self.rng.temporal
- predict()           -> self.rng.prediction
- intervene()         -> self.rng.intervention
- measure()           -> self.rng.outcomes

STEP PURITY CHECKLIST:
- step() uses ONLY (state, t, self.rng.temporal)
- step() does NOT read/write self.* attributes
- step() does NOT use self.rng.intervention or self.rng.prediction
- Run assert_purity(MyScenario) to verify
"""

from typing import Optional

import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)


class CoinFlipScenario(BaseScenario[np.ndarray]):
    """Simplest possible scenario: biased coin flips.

    State is a 1-D array of flip probabilities (one per entity).
    Intervention reduces probability for "high-risk" entities.
    """

    unit_of_analysis = "entity"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        intervention_reduction: float = 0.3,
        treatment_threshold: float = 0.5,
    ):
        super().__init__(time_config=time_config, seed=seed)
        self.intervention_reduction = intervention_reduction
        self.treatment_threshold = treatment_threshold

    def create_population(self, n_entities: int) -> np.ndarray:
        """Initialize flip probabilities from uniform distribution."""
        return self.rng.population.uniform(0.1, 0.9, n_entities)

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """Add small random drift to probabilities."""
        noise = self.rng.temporal.normal(0, 0.02, len(state))
        return np.clip(state + noise, 0.01, 0.99)

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """Noisy observation of true probabilities."""
        noise = self.rng.prediction.normal(0, 0.1, len(state))
        scores = np.clip(state + noise, 0, 1)
        return Predictions(scores=scores)

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int,
    ) -> tuple[np.ndarray, Interventions]:
        """Reduce probability for high-risk entities."""
        treated = predictions.scores > self.treatment_threshold
        state = state.copy()
        state[treated] *= (1 - self.intervention_reduction)
        return state, Interventions(
            treated_indices=np.where(treated)[0],
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Flip coins and record results."""
        flips = (
            self.rng.outcomes.random(len(state)) < state
        ).astype(float)
        return Outcomes(
            events=flips,
            entity_ids=np.arange(len(state)),
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        """Optimized clone for array state."""
        return state.copy()
