"""Stroke Prevention Scenario — Reference Implementation #1.

Simulates a population of patients with time-varying stroke risk.
An ML model identifies high-risk patients for anticoagulant treatment,
which reduces their stroke probability. The branched counterfactual
engine tracks what would have happened without the AI system.

Unit of analysis: patient
State: 2D array (4, n_patients) —
  [base_risks, ar1_modifiers, treatment_effect, current_risks]
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.population.risk_distributions import beta_distributed_risks
from healthcare_sim_sdk.population.temporal_dynamics import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
)

# State array row indices
BASE_RISKS = 0
AR1_MODS = 1
TX_EFFECT = 2   # multiplicative treatment effect (1.0 = no treatment)
CURRENT_RISKS = 3


@dataclass
class StrokeConfig:
    """Configuration for the stroke prevention scenario."""
    n_patients: int = 10_000
    n_weeks: int = 52
    annual_incident_rate: float = 0.05
    concentration: float = 0.5
    prediction_interval: int = 4
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.05
    seasonal_amplitude: float = 0.1
    target_sensitivity: float = 0.80
    target_ppv: float = 0.15
    intervention_effectiveness: float = 0.50
    treatment_threshold: float = 0.5


class StrokePreventionScenario(BaseScenario[np.ndarray]):
    """Stroke prevention with temporal risk dynamics and binary classifier.

    State is a (4, n_patients) array:
    - Row 0: base_risks (immutable after creation)
    - Row 1: AR(1) temporal modifiers (evolve each step)
    - Row 2: treatment_effect (1.0 = untreated, <1.0 = on medication)
    - Row 3: current_risks = base_risks * modifiers * treatment_effect

    This structure ensures step() is pure: all state needed for AR(1)
    evolution and treatment persistence is in the state array.
    """

    unit_of_analysis = "patient"

    def __init__(
        self,
        config: Optional[StrokeConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or StrokeConfig()
        c = self.config

        prediction_schedule = list(
            range(0, c.n_weeks, c.prediction_interval)
        )
        time_config = TimeConfig(
            n_timesteps=c.n_weeks,
            timestep_duration=1 / 52,
            timestep_unit="week",
            prediction_schedule=prediction_schedule,
        )
        super().__init__(time_config=time_config, seed=seed)

        self._classifier = ControlledMLModel(
            mode="classification",
            target_sensitivity=c.target_sensitivity,
            target_ppv=c.target_ppv,
        )
        self._classifier_fitted = False

    def create_population(self, n_entities: int) -> np.ndarray:
        c = self.config
        base_risks = beta_distributed_risks(
            n_patients=n_entities,
            annual_incident_rate=c.annual_incident_rate,
            concentration=c.concentration,
            rng=self.rng.population,
        )
        state = np.zeros((4, n_entities))
        state[BASE_RISKS] = base_risks
        state[AR1_MODS] = 1.0
        state[TX_EFFECT] = 1.0  # no treatment initially
        state[CURRENT_RISKS] = base_risks
        return state

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """Evolve risk via AR(1) with seasonal effects.

        PURITY: uses only self.rng.temporal and state. AR(1) modifiers
        are stored in state[AR1_MODS], not in shared scenario attributes.
        """
        c = self.config
        n = state.shape[1]
        new_state = state.copy()

        # AR(1) process: X_t = rho * X_{t-1} + (1-rho) * seasonal + noise
        seasonal = 1.0 + c.seasonal_amplitude * np.sin(
            2 * np.pi * t / 52 + np.pi / 2
        )
        noise = self.rng.temporal.normal(0, c.ar1_sigma, n)
        new_mods = (
            c.ar1_rho * state[AR1_MODS]
            + (1 - c.ar1_rho) * seasonal
            + noise
        )
        new_mods = np.clip(new_mods, 0.5, 2.0)

        new_state[AR1_MODS] = new_mods
        new_state[CURRENT_RISKS] = np.clip(
            state[BASE_RISKS] * new_mods * state[TX_EFFECT], 0, 0.99
        )
        return new_state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """Run binary classifier on current risk state."""
        risks = state[CURRENT_RISKS]
        hazards = annual_risk_to_hazard(risks)
        window_probs = hazard_to_timestep_probability(
            hazards, self.config.prediction_interval / 52
        )
        true_labels = (
            self.rng.prediction.random(len(risks)) < window_probs
        ).astype(int)

        if not self._classifier_fitted:
            self._classifier.fit(
                true_labels, risks, self.rng.prediction,
                n_iterations=3,
            )
            self._classifier_fitted = True

        scores, labels = self._classifier.predict_binary(
            risks, self.rng.prediction, true_labels,
        )
        return Predictions(
            scores=scores,
            labels=labels,
            metadata={"true_labels": true_labels},
        )

    def intervene(
        self,
        state: np.ndarray,
        predictions: Predictions,
        t: int,
    ) -> tuple[np.ndarray, Interventions]:
        """Treat high-risk patients with anticoagulants."""
        c = self.config
        treated = predictions.scores >= c.treatment_threshold
        state = state.copy()
        # Set persistent treatment effect (survives through future steps)
        state[TX_EFFECT, treated] = (1 - c.intervention_effectiveness)
        # Also apply immediately to current risks
        state[CURRENT_RISKS, treated] *= (1 - c.intervention_effectiveness)
        return state, Interventions(
            treated_indices=np.where(treated)[0],
            metadata={
                "n_treated": int(treated.sum()),
                "effectiveness": c.intervention_effectiveness,
            },
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Generate stroke incidents based on current risk."""
        risks = state[CURRENT_RISKS]
        hazards = annual_risk_to_hazard(risks)
        timestep_probs = hazard_to_timestep_probability(hazards, 1 / 52)
        incidents = (
            self.rng.outcomes.random(len(risks)) < timestep_probs
        ).astype(float)
        return Outcomes(
            events=incidents,
            entity_ids=np.arange(len(risks)),
            metadata={"mean_risk": float(risks.mean())},
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        """Optimized clone for array state."""
        return state.copy()
