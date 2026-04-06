"""Paper 06: SHIELD-RT — Hong et al., JCO 2020.

System for High-Intensity Evaluation During Radiation Therapy.
ML model (AUC 0.80-0.82) identifies RT patients at >10% risk for
acute care visits. High-risk patients randomized to twice-weekly
evaluation vs. standard care. Acute care rate: 22.3% -> 12.3%
(RR 0.556, p=0.02).

Unit of analysis: RT course (patient)
State: 2D numpy array [n_patients x 3]
  col 0: true_acute_care_risk (0-1)
  col 1: received_intervention (0/1) -- cumulative flag
  col 2: had_acute_care_event (0/1) -- cumulative flag

RNG DISCIPLINE:
- create_population() -> self.rng.population
- step()              -> self.rng.temporal
- predict()           -> self.rng.prediction
- intervene()         -> self.rng.intervention
- measure()           -> self.rng.outcomes
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
from healthcare_sim_sdk.ml.model import ControlledMLModel

# Column indices for state array
COL_RISK = 0
COL_INTERVENED = 1
COL_EVENT = 2

# Demographics (paper does not report; using plausible cancer RT population)
RACE_DIST = {
    "White": 0.68,
    "Black": 0.22,  # Duke is in the South; higher Black proportion
    "Hispanic": 0.05,
    "Asian": 0.03,
    "Other": 0.02,
}

# Risk multipliers by demographic (assumed; paper reports no subgroup)
RACE_RISK_MULT = {
    "White": 0.95,
    "Black": 1.15,  # Lower SES, higher comorbidity burden assumption
    "Hispanic": 1.10,
    "Asian": 0.90,
    "Other": 1.00,
}


class ShieldRTScenario(BaseScenario[np.ndarray]):
    """SHIELD-RT: ML-directed evaluation during radiation therapy.

    Simulates a single RT treatment course as one timestep.
    Population: RT courses (n_entities). Each course has a true risk
    of requiring acute care. The ML model identifies high-risk courses.
    Intervention: extra twice-weekly evaluation visits which reduce
    acute care probability via early symptom detection and management.

    Design note: RT courses are modeled as single-episode events
    (no meaningful temporal drift during course). The 'step()' adds
    minor stochastic drift to simulate treatment progression effects
    over 6-week typical RT course (6 timesteps = 6 weeks).

    Parameters derived from paper:
      - base_acute_care_rate = 0.223 (control arm rate)
      - model_auc = 0.81 (midpoint of 0.80-0.82)
      - risk_threshold = 0.10 (>10% risk triggers intervention)
      - intervention_effectiveness = 0.452 (reduces rate to 12.3%,
            i.e., 1 - 0.123/0.223 = 0.448, rounded to 0.452 for fit)
    """

    unit_of_analysis = "rt_course"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        base_acute_care_rate: float = 0.223,
        model_auc: float = 0.81,
        risk_threshold: float = 0.10,
        intervention_effectiveness: float = 0.452,
        high_risk_fraction: float = 0.50,  # ~50% flagged at >10% threshold
    ):
        super().__init__(time_config=time_config, seed=seed)
        self.base_acute_care_rate = base_acute_care_rate
        self.model_auc = model_auc
        self.risk_threshold = risk_threshold
        self.intervention_effectiveness = intervention_effectiveness
        self.high_risk_fraction = high_risk_fraction

        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=model_auc,
        )
        self._model_fitted = False
        self._demographics = None

    def create_population(self, n_entities: int) -> np.ndarray:
        """Create RT course population with beta-distributed risk scores.

        Returns array shape (n_entities, 3):
          col 0: true acute care risk
          col 1: intervened flag (0 at start)
          col 2: event flag (0 at start)
        """
        rng = self.rng.population

        # Beta distribution calibrated to hit base_acute_care_rate
        # We want mean ~0.223 and variance reflecting realistic spread
        # beta(alpha, beta) with alpha=2, beta=6.9 -> mean ~0.224
        alpha = 2.0
        beta_param = alpha * (1 / self.base_acute_care_rate - 1)
        raw_risks = rng.beta(alpha, beta_param, n_entities)

        # Rescale to hit target mean
        scaling = self.base_acute_care_rate / np.mean(raw_risks)
        risks = np.clip(raw_risks * scaling, 0.01, 0.95)

        # Assign demographics
        race_names = list(RACE_DIST.keys())
        race_probs = list(RACE_DIST.values())
        races = rng.choice(race_names, n_entities, p=race_probs)

        # Apply demographic risk multipliers
        for i in range(n_entities):
            mult = RACE_RISK_MULT[races[i]]
            risks[i] = float(np.clip(risks[i] * mult, 0.01, 0.95))

        # Rescale again to maintain target mean
        scaling = self.base_acute_care_rate / np.mean(risks)
        risks = np.clip(risks * scaling, 0.01, 0.95)

        # Store demographics for equity analysis
        self._demographics = races

        # State array: [risk, intervened, event]
        state = np.zeros((n_entities, 3))
        state[:, COL_RISK] = risks
        return state

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """Small temporal drift in acute care risk during RT course.

        RT courses develop toxicity over time — slight upward drift
        in risk as cumulative dose increases. Conservative AR(1).
        """
        rng = self.rng.temporal
        n = state.shape[0]

        # Slight upward drift in risk as RT progresses (treatment toxicity)
        drift = rng.normal(0.002, 0.01, n)  # mean +0.2%/week
        new_state = state.copy()
        new_state[:, COL_RISK] = np.clip(
            state[:, COL_RISK] + drift, 0.01, 0.95
        )
        return new_state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """ML model scores each RT course for acute care risk."""
        true_risks = state[:, COL_RISK]
        n = len(true_risks)

        if not self._model_fitted:
            # Fit model using simulated binary labels from true risks
            true_labels = (
                self.rng.prediction.random(n) < true_risks
            ).astype(int)
            self._model.fit(
                true_labels, true_risks,
                self.rng.prediction, n_iterations=5,
            )
            self._model_fitted = True

        scores = self._model.predict(true_risks, self.rng.prediction)
        return Predictions(
            scores=scores,
            metadata={"true_risks": true_risks.copy()},
        )

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        """Flag high-risk patients for twice-weekly evaluation.

        Risk reduction: intervention reduces acute care probability
        by intervention_effectiveness (multiplicative).
        """
        scores = predictions.scores
        high_risk = scores >= self.risk_threshold
        treated_indices = np.where(high_risk)[0]

        new_state = state.copy()
        # Mark as intervened
        new_state[treated_indices, COL_INTERVENED] = 1.0
        # Reduce risk multiplicatively
        new_state[treated_indices, COL_RISK] = np.clip(
            new_state[treated_indices, COL_RISK]
            * (1.0 - self.intervention_effectiveness),
            0.01, 0.95
        )

        return new_state, Interventions(
            treated_indices=treated_indices,
            metadata={
                "n_high_risk": int(high_risk.sum()),
                "threshold_used": self.risk_threshold,
                "pct_flagged": float(high_risk.mean()),
            },
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Sample acute care events from current risk scores."""
        rng = self.rng.outcomes
        n = state.shape[0]
        risks = state[:, COL_RISK]

        # Stochastic event realization
        events = (rng.random(n) < risks).astype(float)

        # Subgroup breakdown
        race = (
            self._demographics if self._demographics is not None
            else np.array(["Unknown"] * n)
        )

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "true_risk": risks.copy(),
                "intervened": state[:, COL_INTERVENED].copy(),
                "race_ethnicity": race,
            },
            metadata={
                "event_rate": float(events.mean()),
                "n_intervened": int(state[:, COL_INTERVENED].sum()),
                "timestep": t,
            },
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        return state.copy()
