"""
Paper 04: InSight RCT scenario — Shimabukuro et al. 2017.

First published RCT of ML-based sepsis prediction.
  - n=142 ICU patients (UCSF Medical Center)
  - Mortality: 21.3% → 8.96% (58% relative reduction, p=0.018)
  - LOS: 13.0 → 10.3 days (p=0.042)
  - 6-feature gradient-boosted model (vital signs only)

Design notes:
  - Small RCT: n=142, so simulation must reflect high variance
  - Open-label design (no blinding — performance may be overestimated)
  - AUC not reported in paper [ASSUMED: ~0.78 based on 6-feature vital-sign models in lit]
  - ICU population: higher baseline mortality than general ward
  - Randomized design maps perfectly to SDK branched simulation

SDK note: RCT maps directly to branched counterfactual.
  Control = counterfactual branch (standard monitoring)
  Intervention = factual branch (InSight alerts)
"""

import numpy as np

from healthcare_sim_sdk.core.scenario import BaseScenario, Predictions, Interventions, Outcomes
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.population.risk_distributions import beta_distributed_risks

# State rows
ROW_BASE_RISK = 0
ROW_CURRENT_RISK = 1
ROW_AR1_MOD = 2
ROW_STAGE = 3         # 0=stable, 1=deteriorating, 2=severe, 3=deceased, 4=discharged
ROW_STAGE_TIMER = 4
ROW_TREATED = 5
ROW_TREATMENT_TIMER = 6
ROW_DEMO_RACE = 7
ROW_DEMO_AGE = 8
ROW_DEMO_RISK_MULT = 9
ROW_LOS_REMAINING = 10
ROW_RCT_ARM = 11       # 0=control, 1=intervention (randomized)
N_ROWS = 12

STAGE_STABLE = 0
STAGE_DETERIORATING = 1
STAGE_SEVERE = 2
STAGE_DECEASED = 3
STAGE_DISCHARGED = 4

RACE_LABELS = ["White", "Black", "Hispanic", "Asian", "Other"]
AGE_LABELS = ["18-44", "45-64", "65-79", "80+"]


class InSightRCTScenario(BaseScenario):
    """
    InSight RCT scenario: ICU sepsis prediction RCT.

    Critically small N (142): simulation variance will be high.
    Paper's large effect size (58% mortality reduction) is consistent
    with underpowered RCT that happened to achieve significance.

    AUC not reported — this is a KEY scientific reporting gap.
    """

    unit_of_analysis = "patient_admission"

    def __init__(self, time_config, seed=None,
                 model_auc=0.78,              # [ASSUMED: not reported in paper]
                 alert_threshold_percentile=85.0,
                 treatment_effectiveness=0.58,  # calibrated to 58% mortality reduction
                 sepsis_prevalence=0.213,        # paper control arm mortality ~ baseline
                 intervention_ratio=0.50,):      # 50/50 RCT randomization
        super().__init__(time_config=time_config, seed=seed)

        self.model_auc = model_auc
        self.alert_threshold_percentile = alert_threshold_percentile
        self.treatment_effectiveness = treatment_effectiveness
        self.sepsis_prevalence = sepsis_prevalence    # ICU population — high baseline risk
        self.intervention_ratio = intervention_ratio

        # UCSF demographics (approximate)
        self.race_proportions = [0.42, 0.05, 0.15, 0.32, 0.06]  # [ASSUMED]
        self.age_proportions = [0.15, 0.30, 0.35, 0.20]          # [ASSUMED]
        self.race_risk_multipliers = [1.0, 1.8, 0.9, 0.8, 1.0]
        self.age_risk_multipliers = [0.5, 1.0, 1.8, 2.5]

        self.ar1_rho = 0.90
        self.ar1_sigma = 0.06

        # ICU LOS: paper reports 13.0 days control, 10.3 days intervention
        # Timestep = 4h → 13 days = 78 timesteps
        self.mean_los_timesteps = 52   # ~8.7 days mean (mix of intervention/control)
        self.los_std_timesteps = 20

        # ICU stage transitions — higher mortality than ward
        self.prog_stable = 0.020       # faster deterioration in ICU
        self.prog_deteriorating = 0.080
        self.mort_deteriorating = 0.008
        self.mort_severe = 0.045       # high ICU mortality calibrated to 21.3% baseline

        # Good clinician response in ICU context (RCT with dedicated monitoring)
        self.initial_response_rate = 0.85
        self.fatigue_coefficient = 0.0005    # low fatigue in research protocol

        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=self.model_auc,
        )
        self._model_fitted = False
        self._cumulative_false_alerts = 0

    def create_population(self, n_entities):
        rng = self.rng.population
        n = n_entities
        state = np.zeros((N_ROWS, n))

        state[ROW_BASE_RISK] = beta_distributed_risks(
            n_patients=n,
            annual_incident_rate=self.sepsis_prevalence,
            concentration=0.6,
            rng=rng,
        )
        state[ROW_AR1_MOD] = 1.0

        state[ROW_DEMO_RACE] = rng.choice(len(RACE_LABELS), size=n, p=self.race_proportions).astype(float)
        state[ROW_DEMO_AGE] = rng.choice(len(AGE_LABELS), size=n, p=self.age_proportions).astype(float)

        race_mult = np.array(self.race_risk_multipliers)[state[ROW_DEMO_RACE].astype(int)]
        age_mult = np.array(self.age_risk_multipliers)[state[ROW_DEMO_AGE].astype(int)]
        state[ROW_DEMO_RISK_MULT] = race_mult * age_mult

        state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK] * state[ROW_AR1_MOD] * state[ROW_DEMO_RISK_MULT], 0.001, 0.99
        )
        state[ROW_STAGE] = STAGE_STABLE
        state[ROW_STAGE_TIMER] = 0
        state[ROW_TREATED] = 0
        state[ROW_TREATMENT_TIMER] = 0

        # RCT randomization
        state[ROW_RCT_ARM] = rng.binomial(1, self.intervention_ratio, n).astype(float)

        los = rng.normal(self.mean_los_timesteps, self.los_std_timesteps, n)
        state[ROW_LOS_REMAINING] = np.clip(los, 6, 120).astype(float)
        return state

    def step(self, state, t):
        rng = self.rng.temporal
        new_state = state.copy()
        n = state.shape[1]
        active = state[ROW_STAGE] < STAGE_DECEASED

        noise = rng.normal(0, self.ar1_sigma, n)
        new_mods = self.ar1_rho * state[ROW_AR1_MOD] + (1 - self.ar1_rho) * 1.0 + noise
        new_state[ROW_AR1_MOD] = np.clip(new_mods, 0.5, 2.0)
        new_state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK] * new_state[ROW_AR1_MOD] * state[ROW_DEMO_RISK_MULT], 0.001, 0.99
        )

        tx_factor = np.where(state[ROW_TREATED] == 1, 1.0 - self.treatment_effectiveness, 1.0)
        mean_risk = float(new_state[ROW_CURRENT_RISK][active].mean()) if active.any() else self.sepsis_prevalence
        risk_scale = np.clip(new_state[ROW_CURRENT_RISK] / max(mean_risk, 0.001), 0.1, 5.0)

        draws = rng.random(n)
        stable_mask = active & (state[ROW_STAGE] == STAGE_STABLE)
        progressed = stable_mask & (draws < self.prog_stable * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed] = STAGE_DETERIORATING
        new_state[ROW_STAGE_TIMER, progressed] = 0

        draws2 = rng.random(n)
        det_mask = active & (state[ROW_STAGE] == STAGE_DETERIORATING)
        progressed2 = det_mask & (draws2 < self.prog_deteriorating * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed2] = STAGE_SEVERE
        new_state[ROW_STAGE_TIMER, progressed2] = 0

        mort_draws = rng.random(n)
        deceased = (
            (det_mask & ~progressed2 & (mort_draws < self.mort_deteriorating)) |
            (active & (state[ROW_STAGE] == STAGE_SEVERE) & (mort_draws < self.mort_severe))
        )
        new_state[ROW_STAGE, deceased] = STAGE_DECEASED

        new_state[ROW_LOS_REMAINING] = np.maximum(state[ROW_LOS_REMAINING] - 1, 0)
        discharge_eligible = (
            active & ~deceased
            & (new_state[ROW_LOS_REMAINING] <= 0)
            & (state[ROW_STAGE] == STAGE_STABLE)
        )
        new_state[ROW_STAGE, discharge_eligible] = STAGE_DISCHARGED

        still_active = new_state[ROW_STAGE] < STAGE_DECEASED
        new_state[ROW_STAGE_TIMER] = np.where(still_active, new_state[ROW_STAGE_TIMER] + 1, state[ROW_STAGE_TIMER])
        new_state[ROW_TREATMENT_TIMER] = np.where(
            (state[ROW_TREATED] == 1) & still_active, state[ROW_TREATMENT_TIMER] + 1, state[ROW_TREATMENT_TIMER]
        )
        return new_state

    def predict(self, state, t):
        rng = self.rng.prediction
        n = state.shape[1]

        true_labels = (
            (state[ROW_STAGE] >= STAGE_DETERIORATING) & (state[ROW_STAGE] < STAGE_DECEASED)
        ).astype(int)
        true_risks = state[ROW_CURRENT_RISK]

        if not self._model_fitted:
            self._model.fit(true_labels, true_risks, rng, n_iterations=3)
            self._model_fitted = True

        scores = self._model.predict(true_risks, rng)
        # Only score intervention arm patients
        intervention_arm = state[ROW_RCT_ARM] == 1
        active = (state[ROW_STAGE] < STAGE_DECEASED) & intervention_arm
        scores = np.where(active, scores, 0.0)

        active_scores = scores[active]
        if len(active_scores) > 0:
            threshold = np.percentile(active_scores, self.alert_threshold_percentile)
        else:
            threshold = 1.0
        labels = (scores >= threshold).astype(int)
        labels = np.where(active, labels, 0)

        return Predictions(
            scores=scores,
            labels=labels,
            metadata={
                "true_labels": true_labels,
                "n_active_intervention": int(active.sum()),
                "n_flagged": int(labels.sum()),
            },
        )

    def intervene(self, state, predictions, t):
        rng = self.rng.intervention
        new_state = state.copy()
        n = state.shape[1]

        flagged = predictions.labels == 1
        flagged_indices = np.where(flagged)[0]

        response_rate = max(
            0.30,
            self.initial_response_rate * np.exp(-self.fatigue_coefficient * self._cumulative_false_alerts)
        )
        responds = rng.random(len(flagged_indices)) < response_rate
        responded_indices = flagged_indices[responds]

        newly_treated = responded_indices[new_state[ROW_TREATED, responded_indices] == 0]
        new_state[ROW_TREATED, newly_treated] = 1
        new_state[ROW_TREATMENT_TIMER, newly_treated] = 0

        true_labels = predictions.metadata.get("true_labels", np.zeros(n))
        false_alert_indices = flagged_indices[true_labels[flagged_indices] == 0]
        self._cumulative_false_alerts += len(false_alert_indices)

        return new_state, Interventions(
            treated_indices=responded_indices,
            metadata={
                "n_flagged": int(flagged.sum()),
                "n_responded": int(len(responded_indices)),
                "n_newly_treated": int(len(newly_treated)),
            },
        )

    def measure(self, state, t):
        n = state.shape[1]
        events = (
            (state[ROW_STAGE] >= STAGE_DETERIORATING) & (state[ROW_STAGE] < STAGE_DECEASED)
        ).astype(float)
        mortality = (state[ROW_STAGE] == STAGE_DECEASED).astype(float)
        race_labels = np.array(RACE_LABELS)[state[ROW_DEMO_RACE].astype(int)]

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "mortality": mortality,
                "stage": state[ROW_STAGE].copy(),
                "treated": state[ROW_TREATED].copy(),
                "rct_arm": state[ROW_RCT_ARM].copy(),
                "race_ethnicity": race_labels,
                "los_remaining": state[ROW_LOS_REMAINING].copy(),
            },
            metadata={
                "deceased": int((state[ROW_STAGE] == STAGE_DECEASED).sum()),
                "mortality_rate": round(float((state[ROW_STAGE] == STAGE_DECEASED).sum()) / max(1, n), 4),
            },
        )

    def clone_state(self, state):
        return state.copy()
