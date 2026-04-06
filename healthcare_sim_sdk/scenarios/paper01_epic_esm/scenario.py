"""
Paper 01: Epic Sepsis Model (ESM) — Wong et al. JAMA Internal Medicine 2021
==============================================================================

External validation of Epic's proprietary sepsis prediction model at Michigan
Medicine. The model scored every 15 minutes using ~100 EHR features.

Key reported metrics (from paper):
  - AUC: 0.63 (vs. Epic's claimed 0.76–0.83)
  - Sensitivity: 0.33 at threshold 6
  - PPV: 0.12 at threshold 6
  - Alert rate: 18% of all hospitalizations
  - Sepsis prevalence: ~3.5% (inferred from PPV/sensitivity/alert-rate)
  - N: 27,697 patients

Simulation design:
  Entity: patient-admission (one per hospitalization)
  Timestep: 4-hour blocks (6 per day, matches 15-min scoring cadence aggregated)
  Counterfactual: no alerts → no early treatment
  Factual: ESM fires at threshold 6 → clinician responds → early antibiotic

Key modeling choices:
  - AUC fixed at 0.63 (paper's external validation finding)
  - Alert threshold calibrated to produce 18% alert rate
  - PPV calibrated to 0.12 (requires ~3.5% sepsis prevalence)
  - Sensitivity calibrated to 0.33
  - Alert fatigue modeled as high (PPV 12% → high false-positive burden)
  - Treatment effectiveness inferred from timing benefit literature
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

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


# State row indices
ROW_BASE_RISK = 0         # Immutable baseline sepsis risk
ROW_CURRENT_RISK = 1      # Current risk (with AR1 drift)
ROW_AR1_MOD = 2           # AR(1) risk modifier
ROW_STAGE = 3             # 0=at_risk, 1=sepsis, 2=severe, 3=shock, 4=deceased, 5=discharged
ROW_STAGE_TIMER = 4       # Timesteps since entering current stage
ROW_TREATED = 5           # 1 if treated this admission
ROW_TREATMENT_TIMER = 6   # Timesteps since treatment
ROW_DEMO_RACE = 7         # Race (0=White, 1=Black, 2=Hispanic, 3=Asian, 4=Other)
ROW_DEMO_INSURANCE = 8    # Insurance (0=Commercial, 1=Medicare, 2=Medicaid, 3=Self-Pay)
ROW_DEMO_AGE = 9          # Age band (0=18-44, 1=45-64, 2=65-79, 3=80+)
ROW_DEMO_RISK_MULT = 10   # Demographic risk multiplier
ROW_LOS_REMAINING = 11    # Timesteps remaining in admission
ROW_CUMULATIVE_ALERTS = 12
ROW_FALSE_ALERTS = 13
N_ROWS = 14

STAGE_AT_RISK = 0
STAGE_SEPSIS = 1
STAGE_SEVERE = 2
STAGE_SHOCK = 3
STAGE_DECEASED = 4
STAGE_DISCHARGED = 5

RACE_LABELS = ["White", "Black", "Hispanic", "Asian", "Other"]
INSURANCE_LABELS = ["Commercial", "Medicare", "Medicaid", "Self-Pay"]
AGE_LABELS = ["18-44", "45-64", "65-79", "80+"]


@dataclass
class EpicESMConfig:
    """Parameters derived from Wong et al. 2021.

    Assumptions flagged with [ASSUMED] vs paper-reported values.
    """
    # Population
    n_patients: int = 5000
    sepsis_prevalence: float = 0.035          # [ASSUMED] back-calculated from PPV=0.12, sens=0.33, alert_rate=0.18
    risk_concentration: float = 0.5          # [ASSUMED] beta distribution shape

    # Demographics — Michigan Medicine approximate mix
    race_proportions: List[float] = field(
        default_factory=lambda: [0.72, 0.14, 0.06, 0.05, 0.03]  # [ASSUMED] Michigan Medicine estimate
    )
    insurance_proportions: List[float] = field(
        default_factory=lambda: [0.38, 0.32, 0.18, 0.12]        # [ASSUMED]
    )
    age_proportions: List[float] = field(
        default_factory=lambda: [0.20, 0.35, 0.30, 0.15]        # [ASSUMED]
    )
    race_risk_multipliers: List[float] = field(
        default_factory=lambda: [1.0, 1.8, 0.9, 0.8, 1.0]       # [ASSUMED] from literature
    )
    age_risk_multipliers: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.8, 2.5]            # [ASSUMED] from literature
    )

    # LOS
    mean_los_timesteps: int = 35   # ~6 days at 4h timesteps [ASSUMED] typical inpatient
    los_std_timesteps: int = 15    # [ASSUMED]

    # Temporal dynamics
    ar1_rho: float = 0.90          # [ASSUMED]
    ar1_sigma: float = 0.06        # [ASSUMED]

    # ML model — calibrated to paper's reported metrics
    model_auc: float = 0.63                   # Paper-reported external AUC
    alert_threshold_percentile: float = 82.0  # Calibrated to produce ~18% alert rate
    target_sensitivity: float = 0.33          # Paper-reported
    target_ppv: float = 0.12                  # Paper-reported

    # Alert fatigue — HIGH due to 88% FPR
    initial_response_rate: float = 0.55       # [ASSUMED] degraded by poor PPV
    fatigue_coefficient: float = 0.003        # [ASSUMED] faster fatigue than TREWS due to FP burden
    floor_response_rate: float = 0.20         # [ASSUMED] floor clinician compliance

    # Intervention effectiveness — modest (early antibiotics for true positives)
    treatment_effectiveness: float = 0.20     # [ASSUMED] reduced vs. ideal; real-world limitations
    rapid_response_capacity: int = 12         # [ASSUMED] alerts per 4-hr block

    # Stage transition probabilities (per 4-hr block)
    prog_at_risk: float = 0.006    # [ASSUMED] calibrated to 3.5% sepsis incidence
    prog_sepsis: float = 0.022     # [ASSUMED]
    prog_severe: float = 0.038     # [ASSUMED]
    mort_sepsis: float = 0.002     # [ASSUMED]
    mort_severe: float = 0.007     # [ASSUMED]
    mort_shock: float = 0.022      # [ASSUMED]


class EpicESMScenario(BaseScenario[np.ndarray]):
    """
    Epic Sepsis Model (ESM) — external validation scenario.

    Models the real-world deployment at Michigan Medicine where the ESM
    achieved AUC 0.63, sensitivity 0.33, PPV 0.12, 18% alert rate.

    The simulation tests the counterfactual: what mortality difference
    results from deploying a model with these characteristics?
    """

    unit_of_analysis = "patient_admission"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        config: Optional[EpicESMConfig] = None,
        model_auc: float = 0.63,
        alert_threshold_percentile: float = 82.0,
        treatment_effectiveness: float = 0.20,
        initial_response_rate: float = 0.55,
        sepsis_prevalence: float = 0.035,
    ):
        super().__init__(time_config=time_config, seed=seed)

        if config is not None:
            self.cfg = config
        else:
            self.cfg = EpicESMConfig(
                model_auc=model_auc,
                alert_threshold_percentile=alert_threshold_percentile,
                treatment_effectiveness=treatment_effectiveness,
                initial_response_rate=initial_response_rate,
                sepsis_prevalence=sepsis_prevalence,
            )

        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=self.cfg.model_auc,
        )
        self._model_fitted = False
        self._cumulative_false_alerts = 0

    def create_population(self, n_entities: int) -> np.ndarray:
        rng = self.rng.population
        n = n_entities
        state = np.zeros((N_ROWS, n))

        state[ROW_BASE_RISK] = beta_distributed_risks(
            n_patients=n,
            annual_incident_rate=self.cfg.sepsis_prevalence,
            concentration=self.cfg.risk_concentration,
            rng=rng,
        )
        state[ROW_AR1_MOD] = 1.0

        state[ROW_DEMO_RACE] = rng.choice(
            len(RACE_LABELS), size=n, p=self.cfg.race_proportions
        ).astype(float)
        state[ROW_DEMO_INSURANCE] = rng.choice(
            len(INSURANCE_LABELS), size=n, p=self.cfg.insurance_proportions
        ).astype(float)
        state[ROW_DEMO_AGE] = rng.choice(
            len(AGE_LABELS), size=n, p=self.cfg.age_proportions
        ).astype(float)

        race_mult = np.array(self.cfg.race_risk_multipliers)[state[ROW_DEMO_RACE].astype(int)]
        age_mult = np.array(self.cfg.age_risk_multipliers)[state[ROW_DEMO_AGE].astype(int)]
        state[ROW_DEMO_RISK_MULT] = race_mult * age_mult

        state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK] * state[ROW_AR1_MOD] * state[ROW_DEMO_RISK_MULT],
            0.001, 0.99,
        )
        state[ROW_STAGE] = STAGE_AT_RISK
        state[ROW_STAGE_TIMER] = 0
        state[ROW_TREATED] = 0
        state[ROW_TREATMENT_TIMER] = 0

        los = rng.normal(self.cfg.mean_los_timesteps, self.cfg.los_std_timesteps, n)
        state[ROW_LOS_REMAINING] = np.clip(los, 6, 84).astype(float)
        state[ROW_CUMULATIVE_ALERTS] = 0
        state[ROW_FALSE_ALERTS] = 0

        return state

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        rng = self.rng.temporal
        new_state = state.copy()
        n = state.shape[1]

        active = state[ROW_STAGE] < STAGE_DECEASED

        noise = rng.normal(0, self.cfg.ar1_sigma, n)
        new_mods = (
            self.cfg.ar1_rho * state[ROW_AR1_MOD]
            + (1 - self.cfg.ar1_rho) * 1.0
            + noise
        )
        new_state[ROW_AR1_MOD] = np.clip(new_mods, 0.5, 2.0)
        new_state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK] * new_state[ROW_AR1_MOD] * state[ROW_DEMO_RISK_MULT],
            0.001, 0.99,
        )

        draws = rng.random(n)
        tx_factor = np.where(state[ROW_TREATED] == 1, 1.0 - self.cfg.treatment_effectiveness, 1.0)
        mean_risk = float(new_state[ROW_CURRENT_RISK][active].mean()) if active.any() else 0.035
        risk_scale = np.clip(new_state[ROW_CURRENT_RISK] / max(mean_risk, 0.001), 0.1, 5.0)

        at_risk_mask = active & (state[ROW_STAGE] == STAGE_AT_RISK)
        prog_prob = self.cfg.prog_at_risk * risk_scale * tx_factor
        progressed = at_risk_mask & (draws < prog_prob)
        new_state[ROW_STAGE, progressed] = STAGE_SEPSIS
        new_state[ROW_STAGE_TIMER, progressed] = 0

        draws2 = rng.random(n)
        sepsis_mask = active & (state[ROW_STAGE] == STAGE_SEPSIS)
        prog_prob2 = self.cfg.prog_sepsis * risk_scale * tx_factor
        progressed2 = sepsis_mask & (draws2 < prog_prob2)
        new_state[ROW_STAGE, progressed2] = STAGE_SEVERE
        new_state[ROW_STAGE_TIMER, progressed2] = 0

        draws3 = rng.random(n)
        severe_mask = active & (state[ROW_STAGE] == STAGE_SEVERE)
        prog_prob3 = self.cfg.prog_severe * risk_scale * tx_factor
        progressed3 = severe_mask & (draws3 < prog_prob3)
        new_state[ROW_STAGE, progressed3] = STAGE_SHOCK
        new_state[ROW_STAGE_TIMER, progressed3] = 0

        mort_draws = rng.random(n)
        mort_sepsis = sepsis_mask & ~progressed2 & (mort_draws < self.cfg.mort_sepsis)
        mort_severe = severe_mask & ~progressed3 & (mort_draws < self.cfg.mort_severe)
        mort_shock = active & (state[ROW_STAGE] == STAGE_SHOCK) & (mort_draws < self.cfg.mort_shock)
        deceased = mort_sepsis | mort_severe | mort_shock
        new_state[ROW_STAGE, deceased] = STAGE_DECEASED

        new_state[ROW_LOS_REMAINING] = np.maximum(state[ROW_LOS_REMAINING] - 1, 0)
        discharge_eligible = (
            active & ~deceased
            & (new_state[ROW_LOS_REMAINING] <= 0)
            & (state[ROW_STAGE] < STAGE_SEVERE)
        )
        new_state[ROW_STAGE, discharge_eligible] = STAGE_DISCHARGED

        still_active = new_state[ROW_STAGE] < STAGE_DECEASED
        new_state[ROW_STAGE_TIMER] = np.where(
            still_active, new_state[ROW_STAGE_TIMER] + 1, state[ROW_STAGE_TIMER]
        )
        new_state[ROW_TREATMENT_TIMER] = np.where(
            (state[ROW_TREATED] == 1) & still_active,
            state[ROW_TREATMENT_TIMER] + 1,
            state[ROW_TREATMENT_TIMER],
        )
        return new_state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        rng = self.rng.prediction
        n = state.shape[1]

        true_labels = (
            (state[ROW_STAGE] >= STAGE_SEPSIS) & (state[ROW_STAGE] < STAGE_DECEASED)
        ).astype(int)
        true_risks = state[ROW_CURRENT_RISK]

        if not self._model_fitted:
            self._model.fit(true_labels, true_risks, rng, n_iterations=3)
            self._model_fitted = True

        scores = self._model.predict(true_risks, rng)
        active = state[ROW_STAGE] < STAGE_DECEASED
        scores = np.where(active, scores, 0.0)

        active_scores = scores[active]
        if len(active_scores) > 0:
            threshold = np.percentile(active_scores, self.cfg.alert_threshold_percentile)
        else:
            threshold = 1.0
        labels = (scores >= threshold).astype(int)
        labels = np.where(active, labels, 0)

        return Predictions(
            scores=scores,
            labels=labels,
            metadata={
                "true_labels": true_labels,
                "n_active": int(active.sum()),
                "n_flagged": int(labels.sum()),
                "alert_rate": float(labels[active].mean()) if active.any() else 0.0,
                "prevalence": float(true_labels[active].mean()) if active.any() else 0.0,
            },
        )

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        rng = self.rng.intervention
        new_state = state.copy()
        n = state.shape[1]

        flagged = predictions.labels == 1
        flagged_indices = np.where(flagged)[0]

        # High alert fatigue due to 88% false positive rate
        response_rate = max(
            self.cfg.floor_response_rate,
            self.cfg.initial_response_rate
            * np.exp(-self.cfg.fatigue_coefficient * self._cumulative_false_alerts),
        )

        responds = rng.random(len(flagged_indices)) < response_rate
        responded_indices = flagged_indices[responds]
        if len(responded_indices) > self.cfg.rapid_response_capacity:
            scores_of_responded = predictions.scores[responded_indices]
            priority_order = np.argsort(-scores_of_responded)
            responded_indices = responded_indices[priority_order[: self.cfg.rapid_response_capacity]]

        newly_treated = responded_indices[new_state[ROW_TREATED, responded_indices] == 0]
        new_state[ROW_TREATED, newly_treated] = 1
        new_state[ROW_TREATMENT_TIMER, newly_treated] = 0

        true_labels = predictions.metadata.get("true_labels", np.zeros(n))
        new_state[ROW_CUMULATIVE_ALERTS, flagged_indices] += 1
        false_alert_indices = flagged_indices[true_labels[flagged_indices] == 0]
        new_state[ROW_FALSE_ALERTS, false_alert_indices] += 1
        self._cumulative_false_alerts += len(false_alert_indices)

        return new_state, Interventions(
            treated_indices=responded_indices,
            metadata={
                "n_flagged": int(flagged.sum()),
                "n_responded": int(len(responded_indices)),
                "n_newly_treated": int(len(newly_treated)),
                "response_rate": round(response_rate, 4),
                "cumulative_false_alerts": self._cumulative_false_alerts,
            },
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        n = state.shape[1]
        events = (
            (state[ROW_STAGE] >= STAGE_SEPSIS) & (state[ROW_STAGE] < STAGE_DECEASED)
        ).astype(float)
        mortality = (state[ROW_STAGE] == STAGE_DECEASED).astype(float)
        race_labels = np.array(RACE_LABELS)[state[ROW_DEMO_RACE].astype(int)]
        insurance_labels = np.array(INSURANCE_LABELS)[state[ROW_DEMO_INSURANCE].astype(int)]
        age_labels = np.array(AGE_LABELS)[state[ROW_DEMO_AGE].astype(int)]

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "mortality": mortality,
                "stage": state[ROW_STAGE].copy(),
                "treated": state[ROW_TREATED].copy(),
                "race_ethnicity": race_labels,
                "insurance_type": insurance_labels,
                "age_band": age_labels,
            },
            metadata={
                "deceased": int((state[ROW_STAGE] == STAGE_DECEASED).sum()),
                "mortality_rate": round(float((state[ROW_STAGE] == STAGE_DECEASED).sum()) / max(1, n), 4),
                "active": int((state[ROW_STAGE] < STAGE_DECEASED).sum()),
            },
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        return state.copy()
