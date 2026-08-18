"""
Sepsis Early Alert Scenario
============================

Simulates an ML-driven sepsis early warning system in an inpatient setting.

Entity: Patient-admission (one entity per hospital stay).
Timestep: 4 hours (matches the 3-6hr clinical decision window).
ML Task: Discrimination — predict which patients are progressing toward sepsis.
Intervention: Alert fires -> clinician evaluates -> early treatment (antibiotics,
              fluids, cultures) reduces probability of sepsis progression.

Key dynamics:
  - Continuous underlying sepsis risk score with AR(1) drift
  - 4-stage disease progression: at_risk -> sepsis -> severe -> shock
  - Alert fatigue: clinician response rate decays with cumulative false alerts
  - Capacity constraint: rapid response team has limited bandwidth per shift
  - Demographic equity tracking: race, insurance, age subgroups

The counterfactual question: "What would have happened without the early alert?"

All parameters grounded in published literature. See config.yaml and README.md.
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


# ---------------------------------------------------------------------------
# State layout: numpy array with shape (N_ROWS, n_patients)
# ---------------------------------------------------------------------------
# Row indices — constants, never mutated
ROW_BASE_RISK = 0          # Immutable baseline sepsis risk
ROW_AR1_MOD = 1            # AR(1) risk modifier (evolves in step)
ROW_CURRENT_RISK = 2       # base_risk * ar1_mod * demographic_mult
ROW_STAGE = 3              # 0=at_risk, 1=sepsis, 2=severe, 3=shock, 4=deceased, 5=discharged
ROW_STAGE_TIMER = 4        # Timesteps since entering current stage
ROW_TREATED = 5            # 1 if patient has been treated this admission
ROW_TREATMENT_TIMER = 6    # Timesteps since treatment (0 if untreated)
ROW_DEMO_RACE = 7          # Encoded race/ethnicity (0-4)
ROW_DEMO_INSURANCE = 8     # Encoded insurance (0-3)
ROW_DEMO_AGE = 9           # Encoded age band (0-3)
ROW_DEMO_RISK_MULT = 10   # Demographic risk multiplier (computed at creation)
ROW_LOS_REMAINING = 11     # Timesteps remaining in admission (decrements in step)
ROW_CUMULATIVE_ALERTS = 12  # Total alerts fired for this patient
ROW_FALSE_ALERTS = 13       # False alerts (for fatigue tracking)
ROW_ONSET_TIMESTEP = 14     # Timestep of first STAGE_SEPSIS (-1)
ROW_BASELINE_DETECT_DELAY = 15  # Clinical detection delay (ts)
N_STATE_ROWS = 16

# Stage encoding
STAGE_AT_RISK = 0
STAGE_SEPSIS = 1
STAGE_SEVERE = 2
STAGE_SHOCK = 3
STAGE_DECEASED = 4
STAGE_DISCHARGED = 5

# Demographic encoding
RACE_LABELS = ["White", "Black", "Hispanic", "Asian", "Other"]
INSURANCE_LABELS = ["Commercial", "Medicare", "Medicaid", "Self-Pay"]
AGE_LABELS = ["18-44", "45-64", "65-79", "80+"]


@dataclass
class SepsisConfig:
    """All scenario parameters in one place."""

    # Population
    n_patients: int = 2000
    sepsis_incidence: float = 0.04
    risk_concentration: float = 0.5
    mean_los_timesteps: int = 35
    los_std_timesteps: int = 12

    # Demographics
    race_proportions: List[float] = field(
        default_factory=lambda: [0.60, 0.13, 0.18, 0.06, 0.03]
    )
    insurance_proportions: List[float] = field(
        default_factory=lambda: [0.40, 0.25, 0.20, 0.15]
    )
    age_proportions: List[float] = field(
        default_factory=lambda: [0.25, 0.35, 0.28, 0.12]
    )
    race_risk_multipliers: List[float] = field(
        default_factory=lambda: [1.0, 1.8, 0.9, 0.8, 1.0]
    )
    age_risk_multipliers: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.8, 2.5]
    )

    # Temporal dynamics
    ar1_rho: float = 0.90
    ar1_sigma: float = 0.06

    # ML model
    model_auc: float = 0.76
    alert_threshold_percentile: float = 90.0  # Flag top N% of risk scores

    # Alert fatigue
    initial_response_rate: float = 0.65
    fatigue_coefficient: float = 0.002
    floor_response_rate: float = 0.25

    # Intervention
    treatment_effectiveness: float = 0.35
    rapid_response_capacity: int = 8

    # Kumar time-dependent treatment effectiveness
    # kumar_half_life_hours=0 means flat mode (backward compatible)
    # kumar_half_life_hours>0 enables exponential decay: effectiveness
    # halves every kumar_half_life_hours after sepsis onset
    kumar_half_life_hours: float = 0.0
    max_treatment_effectiveness: float = 0.50

    # Baseline clinical detection (standard of care, both branches)
    # Detection delay drawn from Beta(alpha, beta) * max_hours
    # Default Beta(2,5) scaled to 24h: mean ~6.9h, median ~5.8h
    baseline_detection_enabled: bool = True
    baseline_detect_alpha: float = 2.0
    baseline_detect_beta: float = 5.0
    baseline_detect_max_hours: float = 24.0

    # Stage transition probabilities per 4hr block
    prog_at_risk: float = 0.008
    prog_sepsis: float = 0.025
    prog_severe: float = 0.040
    mort_sepsis: float = 0.002
    mort_severe: float = 0.008
    mort_shock: float = 0.025


class SepsisEarlyAlertScenario(BaseScenario[np.ndarray]):
    """
    Simulate a sepsis early warning system in an inpatient population.

    State: np.ndarray of shape (N_STATE_ROWS, n_patients).
    Timestep: 4 hours.
    """

    unit_of_analysis = "patient_admission"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        config: Optional[SepsisConfig] = None,
        # Convenience overrides (take precedence over config)
        n_patients: int = 2000,
        model_auc: float = 0.76,
        alert_threshold_percentile: float = 90.0,
        treatment_effectiveness: float = 0.35,
        kumar_half_life_hours: float = 0.0,
        max_treatment_effectiveness: float = 0.50,
        initial_response_rate: float = 0.65,
        fatigue_coefficient: float = 0.002,
        rapid_response_capacity: int = 8,
        sepsis_incidence: float = 0.04,
        baseline_detection_enabled: bool = True,
    ):
        super().__init__(time_config=time_config, seed=seed)

        if config is not None:
            self.cfg = config
        else:
            self.cfg = SepsisConfig(
                n_patients=n_patients,
                model_auc=model_auc,
                alert_threshold_percentile=alert_threshold_percentile,
                treatment_effectiveness=treatment_effectiveness,
                kumar_half_life_hours=kumar_half_life_hours,
                max_treatment_effectiveness=max_treatment_effectiveness,
                initial_response_rate=initial_response_rate,
                fatigue_coefficient=fatigue_coefficient,
                rapid_response_capacity=rapid_response_capacity,
                sepsis_incidence=sepsis_incidence,
                baseline_detection_enabled=baseline_detection_enabled,
            )

        # ML model — discrimination mode targeting AUC
        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=self.cfg.model_auc,
        )
        self._model_fitted = False

        # Track cumulative false alerts across the population (for fatigue)
        # This is a scenario-level counter that reflects system-wide fatigue.
        # It is read-only in step() (step doesn't use it) and updated only
        # in intervene(), which is factual-branch-only.
        self._cumulative_false_alerts = 0

    # ------------------------------------------------------------------
    # 1. CREATE POPULATION
    # ------------------------------------------------------------------
    def create_population(self, n_entities: int) -> np.ndarray:
        """Generate inpatient admission cohort."""
        rng = self.rng.population
        n = n_entities

        state = np.zeros((N_STATE_ROWS, n))

        # Baseline sepsis risk — beta-distributed, heterogeneous
        state[ROW_BASE_RISK] = beta_distributed_risks(
            n_patients=n,
            annual_incident_rate=self.cfg.sepsis_incidence,
            concentration=self.cfg.risk_concentration,
            rng=rng,
        )

        # AR(1) modifier starts at 1.0 (neutral)
        state[ROW_AR1_MOD] = 1.0

        # Demographics — categorical, drawn from configured proportions
        state[ROW_DEMO_RACE] = rng.choice(
            len(RACE_LABELS), size=n, p=self.cfg.race_proportions
        ).astype(float)
        state[ROW_DEMO_INSURANCE] = rng.choice(
            len(INSURANCE_LABELS), size=n, p=self.cfg.insurance_proportions
        ).astype(float)
        state[ROW_DEMO_AGE] = rng.choice(
            len(AGE_LABELS), size=n, p=self.cfg.age_proportions
        ).astype(float)

        # Compute demographic risk multiplier (race × age)
        race_mult = np.array(self.cfg.race_risk_multipliers)[
            state[ROW_DEMO_RACE].astype(int)
        ]
        age_mult = np.array(self.cfg.age_risk_multipliers)[
            state[ROW_DEMO_AGE].astype(int)
        ]
        state[ROW_DEMO_RISK_MULT] = race_mult * age_mult

        # Current risk = base × modifier × demographic
        state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK] * state[ROW_AR1_MOD] * state[ROW_DEMO_RISK_MULT],
            0.001, 0.99,
        )

        # All patients start at stage 0 (at_risk)
        state[ROW_STAGE] = STAGE_AT_RISK
        state[ROW_STAGE_TIMER] = 0
        state[ROW_TREATED] = 0
        state[ROW_TREATMENT_TIMER] = 0

        # Length of stay — drawn from truncated normal
        los = rng.normal(self.cfg.mean_los_timesteps, self.cfg.los_std_timesteps, n)
        state[ROW_LOS_REMAINING] = np.clip(los, 6, 84).astype(float)  # 1-14 days

        state[ROW_CUMULATIVE_ALERTS] = 0
        state[ROW_FALSE_ALERTS] = 0
        state[ROW_ONSET_TIMESTEP] = -1.0

        # Baseline clinical detection delay (drawn once, stored in state)
        if self.cfg.baseline_detection_enabled:
            raw_delay = rng.beta(
                self.cfg.baseline_detect_alpha,
                self.cfg.baseline_detect_beta,
                n,
            )
            # Convert from [0,1] to timesteps: (delay_hours / 4.0), round up
            state[ROW_BASELINE_DETECT_DELAY] = np.ceil(
                raw_delay * self.cfg.baseline_detect_max_hours / 4.0
            )
        else:
            state[ROW_BASELINE_DETECT_DELAY] = 9999.0  # effectively never

        return state

    # ------------------------------------------------------------------
    # 2. STEP — pure function of (state, t, self.rng.temporal)
    # ------------------------------------------------------------------
    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """
        Evolve patient state by one 4-hour block.

        Disease progression, temporal drift, discharge, and mortality
        are all driven by state + temporal RNG. No self.* mutation.
        """
        rng = self.rng.temporal
        new_state = state.copy()
        n = state.shape[1]

        # Identify active patients (not deceased, not discharged)
        active = (state[ROW_STAGE] < STAGE_DECEASED)

        # --- AR(1) drift on risk modifier ---
        noise = rng.normal(0, self.cfg.ar1_sigma, n)
        new_mods = (
            self.cfg.ar1_rho * state[ROW_AR1_MOD]
            + (1 - self.cfg.ar1_rho) * 1.0
            + noise
        )
        new_state[ROW_AR1_MOD] = np.clip(new_mods, 0.5, 2.0)

        # Recompute current risk
        new_state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK]
            * new_state[ROW_AR1_MOD]
            * state[ROW_DEMO_RISK_MULT],
            0.001, 0.99,
        )

        # --- Disease progression (stochastic, risk-modified) ---
        draws = rng.random(n)

        # Treatment reduces progression probability
        if self.cfg.kumar_half_life_hours > 0:
            # Time-dependent effectiveness (Kumar decay curve):
            # effectiveness halves every kumar_half_life_hours after onset
            # treatment_start_t = t - treatment_timer
            treatment_start_t = t - state[ROW_TREATMENT_TIMER]
            delay_timesteps = np.where(
                (state[ROW_TREATED] == 1) & (state[ROW_ONSET_TIMESTEP] >= 0),
                np.maximum(treatment_start_t - state[ROW_ONSET_TIMESTEP], 0),
                0,
            )
            delay_hours = delay_timesteps * 4.0
            effectiveness = np.where(
                state[ROW_TREATED] == 1,
                self.cfg.max_treatment_effectiveness * np.power(
                    0.5, delay_hours / self.cfg.kumar_half_life_hours,
                ),
                0.0,
            )
            # Preventive treatment (treated before onset): max effectiveness
            no_onset = state[ROW_ONSET_TIMESTEP] < 0
            effectiveness = np.where(
                no_onset & (state[ROW_TREATED] == 1),
                self.cfg.max_treatment_effectiveness,
                effectiveness,
            )
            tx_factor = np.where(
                state[ROW_TREATED] == 1,
                1.0 - effectiveness,
                1.0,
            )
        else:
            # Flat mode (backward compatible)
            tx_factor = np.where(
                state[ROW_TREATED] == 1,
                1.0 - self.cfg.treatment_effectiveness,
                1.0,
            )

        # Individual risk scaling: each patient's current_risk relative to
        # the population mean risk. Patients above the mean progress faster;
        # patients below the mean progress slower. The base transition
        # probabilities are calibrated for a patient at mean risk.
        mean_risk = np.mean(new_state[ROW_CURRENT_RISK][active]) if active.any() else 0.04
        risk_scale = np.clip(
            new_state[ROW_CURRENT_RISK] / max(mean_risk, 0.001),
            0.1, 5.0,
        )

        # At-risk -> sepsis
        at_risk_mask = active & (state[ROW_STAGE] == STAGE_AT_RISK)
        prog_prob = self.cfg.prog_at_risk * risk_scale * tx_factor
        progressed = at_risk_mask & (draws < prog_prob)
        new_state[ROW_STAGE, progressed] = STAGE_SEPSIS
        new_state[ROW_STAGE_TIMER, progressed] = 0

        # Record onset timestep for patients newly entering SEPSIS
        newly_septic = progressed & (state[ROW_ONSET_TIMESTEP] < 0)
        new_state[ROW_ONSET_TIMESTEP] = np.where(
            newly_septic, float(t), state[ROW_ONSET_TIMESTEP],
        )

        # Sepsis -> severe
        draws2 = rng.random(n)
        sepsis_mask = active & (state[ROW_STAGE] == STAGE_SEPSIS)
        prog_prob2 = self.cfg.prog_sepsis * risk_scale * tx_factor
        progressed2 = sepsis_mask & (draws2 < prog_prob2)
        new_state[ROW_STAGE, progressed2] = STAGE_SEVERE
        new_state[ROW_STAGE_TIMER, progressed2] = 0

        # Severe -> shock
        draws3 = rng.random(n)
        severe_mask = active & (state[ROW_STAGE] == STAGE_SEVERE)
        prog_prob3 = self.cfg.prog_severe * risk_scale * tx_factor
        progressed3 = severe_mask & (draws3 < prog_prob3)
        new_state[ROW_STAGE, progressed3] = STAGE_SHOCK
        new_state[ROW_STAGE_TIMER, progressed3] = 0

        # --- Baseline clinical detection (standard of care) ---
        # Clinicians detect sepsis through routine care after a
        # patient-specific delay (drawn from Beta distribution at creation).
        # Runs on BOTH branches — this is standard care, not ML.
        if self.cfg.baseline_detection_enabled:
            eligible = (
                (new_state[ROW_STAGE] >= STAGE_SEPSIS)
                & (new_state[ROW_STAGE] < STAGE_DECEASED)
                & (new_state[ROW_TREATED] == 0)
                & (new_state[ROW_ONSET_TIMESTEP] >= 0)
            )
            time_since_onset = t - new_state[ROW_ONSET_TIMESTEP]
            detected = eligible & (
                time_since_onset >= new_state[ROW_BASELINE_DETECT_DELAY]
            )
            new_state[ROW_TREATED, detected] = 1
            new_state[ROW_TREATMENT_TIMER, detected] = 0

        # --- Mortality (competing with progression) ---
        mort_draws = rng.random(n)
        mort_sepsis = sepsis_mask & ~progressed2 & (mort_draws < self.cfg.mort_sepsis)
        mort_severe = severe_mask & ~progressed3 & (mort_draws < self.cfg.mort_severe)
        mort_shock = active & (state[ROW_STAGE] == STAGE_SHOCK) & (
            mort_draws < self.cfg.mort_shock
        )
        deceased = mort_sepsis | mort_severe | mort_shock
        new_state[ROW_STAGE, deceased] = STAGE_DECEASED

        # --- Discharge (LOS countdown) ---
        new_state[ROW_LOS_REMAINING] = np.maximum(state[ROW_LOS_REMAINING] - 1, 0)
        discharge_eligible = (
            active
            & ~deceased
            & (new_state[ROW_LOS_REMAINING] <= 0)
            & (state[ROW_STAGE] < STAGE_SEVERE)  # Severe/shock patients don't discharge
        )
        new_state[ROW_STAGE, discharge_eligible] = STAGE_DISCHARGED

        # --- Timers ---
        still_active = new_state[ROW_STAGE] < STAGE_DECEASED
        new_state[ROW_STAGE_TIMER] = np.where(
            still_active,
            new_state[ROW_STAGE_TIMER] + 1,
            state[ROW_STAGE_TIMER],
        )
        new_state[ROW_TREATMENT_TIMER] = np.where(
            (state[ROW_TREATED] == 1) & still_active,
            state[ROW_TREATMENT_TIMER] + 1,
            state[ROW_TREATMENT_TIMER],
        )

        return new_state

    # ------------------------------------------------------------------
    # 3. PREDICT — ML model scoring
    # ------------------------------------------------------------------
    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """Score all active patients with the sepsis prediction model."""
        rng = self.rng.prediction
        n = state.shape[1]

        # True sepsis status: 1 if patient is in sepsis/severe/shock, 0 otherwise
        true_labels = (
            (state[ROW_STAGE] >= STAGE_SEPSIS) & (state[ROW_STAGE] < STAGE_DECEASED)
        ).astype(int)

        # True underlying risk (what a perfect model would know)
        true_risks = state[ROW_CURRENT_RISK]

        # Fit model on first call
        if not self._model_fitted:
            self._model.fit(true_labels, true_risks, rng, n_iterations=3)
            self._model_fitted = True

        # Generate predictions with controlled AUC
        scores = self._model.predict(true_risks, rng)

        # Only score active patients; set discharged/deceased to 0
        active = (state[ROW_STAGE] < STAGE_DECEASED)
        scores = np.where(active, scores, 0.0)

        # Percentile-based threshold on active patient scores
        active_scores = scores[active]
        if len(active_scores) > 0:
            threshold = np.percentile(
                active_scores, self.cfg.alert_threshold_percentile
            )
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
                "prevalence": float(true_labels[active].mean()) if active.any() else 0,
            },
        )

    # ------------------------------------------------------------------
    # 4. INTERVENE — alert response with fatigue dynamics
    # ------------------------------------------------------------------
    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        """
        Process alerts: flag patients above threshold, simulate clinician
        response (with fatigue), apply treatment to responded patients.
        """
        rng = self.rng.intervention
        new_state = state.copy()
        n = state.shape[1]

        # Who was flagged by the model?
        flagged = predictions.labels == 1
        flagged_indices = np.where(flagged)[0]

        # --- Alert fatigue: response rate decays with cumulative false alerts ---
        response_rate = max(
            self.cfg.floor_response_rate,
            self.cfg.initial_response_rate
            * np.exp(-self.cfg.fatigue_coefficient * self._cumulative_false_alerts),
        )

        # Clinician decides whether to respond to each alert
        responds = rng.random(len(flagged_indices)) < response_rate

        # Capacity constraint: can only respond to N alerts per timestep
        responded_indices = flagged_indices[responds]
        if len(responded_indices) > self.cfg.rapid_response_capacity:
            # Prioritize by prediction score (highest risk first)
            scores_of_responded = predictions.scores[responded_indices]
            priority_order = np.argsort(-scores_of_responded)
            responded_indices = responded_indices[
                priority_order[: self.cfg.rapid_response_capacity]
            ]

        # Apply treatment to responded patients (who haven't been treated yet)
        newly_treated = responded_indices[new_state[ROW_TREATED, responded_indices] == 0]
        new_state[ROW_TREATED, newly_treated] = 1
        new_state[ROW_TREATMENT_TIMER, newly_treated] = 0

        # Track alerts and false alerts
        true_labels = predictions.metadata.get("true_labels", np.zeros(n))
        new_state[ROW_CUMULATIVE_ALERTS, flagged_indices] += 1

        # Identify false alerts: flagged but not actually septic
        false_alert_indices = flagged_indices[true_labels[flagged_indices] == 0]
        new_state[ROW_FALSE_ALERTS, false_alert_indices] += 1

        # Update system-wide false alert counter (for fatigue model)
        n_false_this_step = len(false_alert_indices)
        self._cumulative_false_alerts += n_false_this_step

        return new_state, Interventions(
            treated_indices=responded_indices,
            metadata={
                "n_flagged": int(flagged.sum()),
                "n_responded": int(len(responded_indices)),
                "n_newly_treated": int(len(newly_treated)),
                "response_rate": round(response_rate, 4),
                "cumulative_false_alerts": self._cumulative_false_alerts,
                "capacity_limited": len(flagged_indices[responds])
                > self.cfg.rapid_response_capacity,
            },
        )

    # ------------------------------------------------------------------
    # 5. MEASURE — outcomes on both branches
    # ------------------------------------------------------------------
    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """
        Observe outcomes at this timestep. Called on BOTH branches.

        Primary outcome: sepsis progression event (entered a worse stage).
        Secondary: mortality, stage distribution, alert burden.
        """
        n = state.shape[1]

        # Primary event: patient is in sepsis or worse (stage >= 1, < deceased)
        events = (
            (state[ROW_STAGE] >= STAGE_SEPSIS) & (state[ROW_STAGE] < STAGE_DECEASED)
        ).astype(float)

        # Stage distribution
        stage_counts = {
            "at_risk": int((state[ROW_STAGE] == STAGE_AT_RISK).sum()),
            "sepsis": int((state[ROW_STAGE] == STAGE_SEPSIS).sum()),
            "severe": int((state[ROW_STAGE] == STAGE_SEVERE).sum()),
            "shock": int((state[ROW_STAGE] == STAGE_SHOCK).sum()),
            "deceased": int((state[ROW_STAGE] == STAGE_DECEASED).sum()),
            "discharged": int((state[ROW_STAGE] == STAGE_DISCHARGED).sum()),
        }

        # Mortality as secondary outcome
        mortality = (state[ROW_STAGE] == STAGE_DECEASED).astype(float)

        # Demographic labels for equity analysis
        race_labels = np.array(RACE_LABELS)[state[ROW_DEMO_RACE].astype(int)]
        insurance_labels = np.array(INSURANCE_LABELS)[
            state[ROW_DEMO_INSURANCE].astype(int)
        ]
        age_labels = np.array(AGE_LABELS)[state[ROW_DEMO_AGE].astype(int)]

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "mortality": mortality,
                "stage": state[ROW_STAGE].copy(),
                "treated": state[ROW_TREATED].copy(),
                "onset_timestep": state[ROW_ONSET_TIMESTEP].copy(),
                "cumulative_alerts": state[ROW_CUMULATIVE_ALERTS].copy(),
                "race_ethnicity": race_labels,
                "insurance_type": insurance_labels,
                "age_band": age_labels,
            },
            metadata={
                "stage_counts": stage_counts,
                "mortality_rate": round(
                    stage_counts["deceased"] / max(1, n), 4
                ),
                "treated_fraction": round(
                    float((state[ROW_TREATED] == 1).sum()) / max(1, n), 4
                ),
                "active_patients": int(
                    (state[ROW_STAGE] < STAGE_DECEASED).sum()
                ),
            },
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def clone_state(self, state: np.ndarray) -> np.ndarray:
        """Optimized copy for numpy state."""
        return state.copy()

    def validate_population(self, state: np.ndarray) -> Dict[str, Any]:
        """Check population after creation."""
        n = state.shape[1]
        return {
            "population_created": True,
            "n_patients": n,
            "mean_base_risk": round(float(state[ROW_BASE_RISK].mean()), 4),
            "risk_range": [
                round(float(state[ROW_BASE_RISK].min()), 4),
                round(float(state[ROW_BASE_RISK].max()), 4),
            ],
            "all_at_risk": bool((state[ROW_STAGE] == STAGE_AT_RISK).all()),
            "mean_los": round(float(state[ROW_LOS_REMAINING].mean()), 1),
            "mean_baseline_detect_delay_hrs": round(
                float(state[ROW_BASELINE_DETECT_DELAY].mean()) * 4.0, 1
            ),
            "demographic_check": {
                "race_white_pct": round(
                    float((state[ROW_DEMO_RACE] == 0).sum()) / n, 2
                ),
                "race_black_pct": round(
                    float((state[ROW_DEMO_RACE] == 1).sum()) / n, 2
                ),
            },
        }
