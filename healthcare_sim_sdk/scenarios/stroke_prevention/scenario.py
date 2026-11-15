"""Stroke Prevention Scenario — Reference Implementation #1.

Simulates a population of patients with time-varying stroke risk.
An ML model identifies high-risk patients for anticoagulant treatment,
which reduces their stroke probability. The branched counterfactual
engine tracks what would have happened without the AI system.

Unit of analysis: patient
State: 2D array (6, n_patients) —
  [base_risks, ar1_modifiers, treatment_effect, current_risks,
   had_stroke, stroke_week]

Event timing: step(t) resolves week t for every patient still at risk,
using the risk carried into the week (including treatment started at
t-1), then measure(t) reports those strokes without drawing. A stroke
ends follow-up. Treatment is applied once per patient and persists.
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
TX_EFFECT = 2  # multiplicative treatment effect (1.0 = no treatment)
CURRENT_RISKS = 3
HAD_STROKE = 4  # 1.0 once a stroke has occurred; follow-up ends
STROKE_WEEK = 5  # week of the stroke, -1 if none


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
    # Amplitude of the AR(1) TARGET, not realized risk: defaults produce
    # +/-3.9% after smoothing, lagging the configured peak by 9 weeks.
    seasonal_amplitude: float = 0.1
    # Default model: AUC 0.75, threshold set so PPV for a stroke within the
    # prediction window is target_lift times the window's base rate. At the
    # defaults that is 3.7 x 0.41% = 1.5%; on held-out data it flags ~13% of
    # patients with sensitivity ~0.43. A lift, not a fixed PPV, stays valid
    # when the incidence or the prediction window changes.
    target_auc: float = 0.75
    target_lift: float = 3.7
    # Fixed PPV instead of a lift. Must exceed the window's base rate.
    target_ppv: Optional[float] = None
    # Legacy: set target_sensitivity to use classification mode with both
    # sensitivity and PPV (PPV defaults to 0.15). That old pair needs AUC
    # ~0.98 at the default horizon.
    target_sensitivity: Optional[float] = None
    # The model is fitted on a bootstrap reference cohort with at least this
    # many four-week events (~48,600 patients at the defaults), not on the
    # ~40 events a 10,000-patient first step provides. 200 matches the SDK
    # default; more events fit more tightly but the exact-AUC search scales
    # with cohort size (about 15 s at 100,000 patients).
    reference_min_events: int = 200
    allow_infeasible_model: bool = False
    intervention_effectiveness: float = 0.50
    # Who gets treated. None (default) = the model's own operating point,
    # i.e. the threshold fit() chose to hit target_sensitivity/target_ppv.
    # A float overrides that with a raw-score cutoff — note the scores are
    # calibrated prediction-window probabilities (typically <0.05), not classifier
    # probabilities, so a value like 0.5 treats almost nobody.
    treatment_threshold: Optional[float] = None


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

        prediction_schedule = list(range(0, c.n_weeks, c.prediction_interval))
        time_config = TimeConfig(
            n_timesteps=c.n_weeks,
            timestep_duration=1 / 52,
            timestep_unit="week",
            prediction_schedule=prediction_schedule,
        )
        super().__init__(time_config=time_config, seed=seed)

        # The operating-point model is built at the first prediction, when
        # the window's base rate is known (see _build_classifier).
        self._classifier: Optional[ControlledMLModel] = None
        if c.target_sensitivity is not None:
            self._classifier = ControlledMLModel(
                mode="classification",
                target_sensitivity=c.target_sensitivity,
                target_ppv=0.15 if c.target_ppv is None else c.target_ppv,
                allow_infeasible=c.allow_infeasible_model,
            )
        self._classifier_fitted = False
        self.model_status = "not fitted"

    def _build_classifier(self, base_rate: float) -> Optional[ControlledMLModel]:
        c = self.config
        ppv = c.target_ppv
        if ppv is None:
            # Cap so the target stays a valid probability above the base rate.
            ppv = min(c.target_lift * base_rate, (1 + base_rate) / 2)
        return ControlledMLModel(
            mode="operating_point",
            target_auc=c.target_auc,
            binding="ppv",
            binding_value=ppv,
            allow_infeasible=c.allow_infeasible_model,
        )

    def create_population(self, n_entities: int) -> np.ndarray:
        c = self.config
        base_risks = beta_distributed_risks(
            n_patients=n_entities,
            annual_incident_rate=c.annual_incident_rate,
            concentration=c.concentration,
            rng=self.rng.population,
        )
        state = np.zeros((6, n_entities))
        state[BASE_RISKS] = base_risks
        state[AR1_MODS] = 1.0
        state[TX_EFFECT] = 1.0  # no treatment initially
        state[CURRENT_RISKS] = base_risks
        state[STROKE_WEEK] = -1.0
        return state

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """Resolve week t's strokes, then evolve risk via AR(1).

        PURITY: uses only self.rng.temporal and state. One uniform draw per
        patient per week, whatever their status, so both branches consume
        the stream identically (common random numbers).
        """
        c = self.config
        n = state.shape[1]
        new_state = state.copy()

        # Strokes this week, from the risk carried into the week.
        at_risk = state[HAD_STROKE] == 0
        week_probs = hazard_to_timestep_probability(
            annual_risk_to_hazard(state[CURRENT_RISKS]), 1 / 52)
        coin = self.rng.temporal.random(n)
        struck = at_risk & (coin < week_probs)
        new_state[HAD_STROKE, struck] = 1.0
        new_state[STROKE_WEEK, struck] = t

        # AR(1) process: X_t = rho * X_{t-1} + (1-rho) * seasonal + noise
        seasonal = 1.0 + c.seasonal_amplitude * np.sin(2 * np.pi * t / 52 + np.pi / 2)
        noise = self.rng.temporal.normal(0, c.ar1_sigma, n)
        new_mods = c.ar1_rho * state[AR1_MODS] + (1 - c.ar1_rho) * seasonal + noise
        new_mods = np.clip(new_mods, 0.5, 2.0)

        new_state[AR1_MODS] = new_mods
        new_state[CURRENT_RISKS] = np.where(
            new_state[HAD_STROKE] == 1, 0.0,
            np.clip(state[BASE_RISKS] * new_mods * state[TX_EFFECT], 0, 0.99))
        return new_state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """Run binary classifier on current risk state."""
        risks = state[CURRENT_RISKS]
        hazards = annual_risk_to_hazard(risks)
        window_probs = hazard_to_timestep_probability(
            hazards, self.config.prediction_interval / 52
        )
        true_labels = (self.rng.prediction.random(len(risks)) < window_probs).astype(
            int
        )

        if not self._classifier_fitted:
            if self._classifier is None:
                self._classifier = self._build_classifier(float(window_probs.mean()))
                # fit_reference needs enough events in at most 100,000
                # bootstrap patients; a near-zero incidence cannot supply them.
                expected = 100_000 * float(window_probs.mean())
                if expected < 2 * self._classifier.min_positives_for_gate:
                    if not self.config.allow_infeasible_model:
                        raise ValueError(
                            f"Window base rate {window_probs.mean():.2e} is too "
                            "low to fit a model (expected "
                            f"{expected:.0f} events in 100,000 patients). Set "
                            "allow_infeasible_model=True to run with no model.")
                    self._classifier = None
                    self.model_status = "not fitted: too few events"
            if self._classifier is None:
                pass
            elif self.config.target_sensitivity is not None:
                self._classifier.fit(
                    true_labels,
                    risks,
                    self.rng.prediction,
                    n_iterations=3,
                )
            else:
                self._classifier.fit_reference(
                    risks, self.rng.prediction,
                    label_probabilities=window_probs,
                    min_events=self.config.reference_min_events,
                )
            self._classifier_fitted = True
            if self._classifier is not None:
                self.model_status = "fitted"

        if self._classifier is None:
            # No model could be fitted: nobody is flagged; recorded, not hidden.
            scores = np.zeros(len(risks))
            labels = np.zeros(len(risks), dtype=int)
        else:
            # Label-free scoring: the model never sees outcomes.
            scores, labels = self._classifier.predict_binary(
                risks, self.rng.prediction)
        return Predictions(
            scores=scores,
            labels=labels,
            metadata={"true_labels": true_labels,
                      "model_status": self.model_status},
        )

    def intervene(
        self,
        state: np.ndarray,
        predictions: Predictions,
        t: int,
    ) -> tuple[np.ndarray, Interventions]:
        """Treat high-risk patients with anticoagulants.

        By default the deployment decision IS the model's flag: labels
        are `scores >= classifier.threshold`, where that threshold was
        fit to achieve target_sensitivity/target_ppv. Re-thresholding the
        raw scores here would silently discard that operating point.
        """
        c = self.config
        if c.treatment_threshold is None:
            flagged = predictions.labels == 1
        else:
            flagged = predictions.scores >= c.treatment_threshold
        already = state[TX_EFFECT] < 1.0
        eligible = state[HAD_STROKE] == 0
        # Treatment starts once per patient and persists; re-flagging a
        # patient already on treatment does not compound the effect.
        start = flagged & eligible & ~already
        state = state.copy()
        state[TX_EFFECT, start] = 1 - c.intervention_effectiveness
        state[CURRENT_RISKS, start] *= 1 - c.intervention_effectiveness
        return state, Interventions(
            treated_indices=np.where(start)[0],
            metadata={
                "n_treated": int(start.sum()),
                "n_flagged": int(flagged.sum()),
                "n_flagged_already_treated": int((flagged & already).sum()),
                "n_on_treatment": int((state[TX_EFFECT] < 1.0).sum()),
                "effectiveness": c.intervention_effectiveness,
            },
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Report week t's strokes (resolved in step); no random draws."""
        incidents = (state[STROKE_WEEK] == t).astype(float)
        at_risk = state[HAD_STROKE] == 0
        return Outcomes(
            events=incidents,
            entity_ids=np.arange(state.shape[1]),
            metadata={
                "mean_risk": float(state[CURRENT_RISKS][at_risk].mean())
                if at_risk.any() else 0.0,
                "n_at_risk": int(at_risk.sum()),
            },
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        """Optimized clone for array state."""
        return state.copy()
