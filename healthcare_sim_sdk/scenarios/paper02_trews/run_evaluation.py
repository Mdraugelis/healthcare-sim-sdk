"""
Runner script for Paper 02: TREWS — Adams, Henry, Sridharan et al. 2022.

Key design insight: TREWS has a two-stage process:
  1. TREWS fires alert
  2. Clinician CONFIRMS alert within 3 hours (or ignores)
  3. Only confirmed alerts enter the treatment pathway

The paper reports mortality effect only in the confirmed-alert cohort.
Sensitivity=0.80, PPV=0.27, alert_rate=7%

Paper reports: 3.3pp absolute mortality reduction in confirmed-alert cohort.
Baseline mortality in that cohort: ~18.7% → 15.4%

Usage:
    cd /data/.openclaw/workspace/healthcare-sim-sdk
    python3 -m healthcare_sim_sdk.scenarios.paper02_trews.run_evaluation
"""

import sys
import json
import numpy as np

from healthcare_sim_sdk.core.scenario import TimeConfig, BaseScenario, Predictions, Interventions, Outcomes
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.population.risk_distributions import beta_distributed_risks

# State row indices
ROW_BASE_RISK = 0
ROW_CURRENT_RISK = 1
ROW_AR1_MOD = 2
ROW_STAGE = 3          # 0=at_risk, 1=sepsis, 2=severe, 3=shock, 4=deceased, 5=discharged
ROW_STAGE_TIMER = 4
ROW_TREATED = 5
ROW_TREATMENT_TIMER = 6
ROW_DEMO_RACE = 7
ROW_DEMO_INSURANCE = 8
ROW_DEMO_AGE = 9
ROW_DEMO_RISK_MULT = 10
ROW_LOS_REMAINING = 11
ROW_ALERT_CONFIRMED = 12  # TREWS-specific: alert was confirmed by clinician
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


class TREWSScenario(BaseScenario):
    """
    TREWS scenario: two-stage alert with clinician confirmation step.

    Key differences from Epic ESM:
    - AUC not directly stated; modeled via sensitivity=0.80, PPV=0.27 (classification mode)
    - 7% alert rate (much lower than ESM's 18%)
    - Alert confirmation within 3h acts as a human filter
    - Effect size applies only to confirmed-alert subgroup
    - 3.3pp mortality reduction in confirmed cohort (18.7% → 15.4%)
    """

    unit_of_analysis = "patient_admission"

    def __init__(self, time_config, seed=None,
                 model_sensitivity=0.80, model_ppv=0.27,
                 alert_threshold_percentile=93.0,
                 confirmation_rate=0.65,
                 treatment_effectiveness=0.18,
                 sepsis_prevalence=0.038,
                 initial_response_rate=0.80,
                 fatigue_coefficient=0.001,):
        super().__init__(time_config=time_config, seed=seed)

        self.sepsis_prevalence = sepsis_prevalence
        self.alert_threshold_percentile = alert_threshold_percentile
        self.confirmation_rate = confirmation_rate      # P(clinician confirms alert within 3h)
        self.treatment_effectiveness = treatment_effectiveness
        self.initial_response_rate = initial_response_rate
        self.fatigue_coefficient = fatigue_coefficient
        self.floor_response_rate = 0.40

        # Race/demographics for 5 JHH hospitals
        self.race_proportions = [0.48, 0.38, 0.08, 0.04, 0.02]   # [ASSUMED] JHH demographics
        self.insurance_proportions = [0.32, 0.28, 0.22, 0.18]     # [ASSUMED]
        self.age_proportions = [0.18, 0.32, 0.32, 0.18]           # [ASSUMED]
        self.race_risk_multipliers = [1.0, 1.8, 0.9, 0.8, 1.0]
        self.age_risk_multipliers = [0.5, 1.0, 1.8, 2.5]

        self.ar1_rho = 0.90
        self.ar1_sigma = 0.06
        self.mean_los_timesteps = 35
        self.los_std_timesteps = 15

        self.prog_at_risk = 0.007
        self.prog_sepsis = 0.024
        self.prog_severe = 0.040
        self.mort_sepsis = 0.0025
        self.mort_severe = 0.008
        self.mort_shock = 0.025
        self.rapid_response_capacity = 15

        # Use classification mode to target sensitivity + PPV
        self._model = ControlledMLModel(
            mode="classification",
            target_sensitivity=model_sensitivity,
            target_ppv=model_ppv,
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
            concentration=0.5,
            rng=rng,
        )
        state[ROW_AR1_MOD] = 1.0

        state[ROW_DEMO_RACE] = rng.choice(len(RACE_LABELS), size=n, p=self.race_proportions).astype(float)
        state[ROW_DEMO_INSURANCE] = rng.choice(len(INSURANCE_LABELS), size=n, p=self.insurance_proportions).astype(float)
        state[ROW_DEMO_AGE] = rng.choice(len(AGE_LABELS), size=n, p=self.age_proportions).astype(float)

        race_mult = np.array(self.race_risk_multipliers)[state[ROW_DEMO_RACE].astype(int)]
        age_mult = np.array(self.age_risk_multipliers)[state[ROW_DEMO_AGE].astype(int)]
        state[ROW_DEMO_RISK_MULT] = race_mult * age_mult

        state[ROW_CURRENT_RISK] = np.clip(
            state[ROW_BASE_RISK] * state[ROW_AR1_MOD] * state[ROW_DEMO_RISK_MULT], 0.001, 0.99
        )
        state[ROW_STAGE] = STAGE_AT_RISK
        state[ROW_STAGE_TIMER] = 0
        state[ROW_TREATED] = 0
        state[ROW_TREATMENT_TIMER] = 0
        state[ROW_ALERT_CONFIRMED] = 0
        state[ROW_FALSE_ALERTS] = 0

        los = rng.normal(self.mean_los_timesteps, self.los_std_timesteps, n)
        state[ROW_LOS_REMAINING] = np.clip(los, 6, 84).astype(float)
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

        draws = rng.random(n)
        tx_factor = np.where(state[ROW_TREATED] == 1, 1.0 - self.treatment_effectiveness, 1.0)
        mean_risk = float(new_state[ROW_CURRENT_RISK][active].mean()) if active.any() else 0.038
        risk_scale = np.clip(new_state[ROW_CURRENT_RISK] / max(mean_risk, 0.001), 0.1, 5.0)

        at_risk_mask = active & (state[ROW_STAGE] == STAGE_AT_RISK)
        progressed = at_risk_mask & (draws < self.prog_at_risk * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed] = STAGE_SEPSIS
        new_state[ROW_STAGE_TIMER, progressed] = 0

        draws2 = rng.random(n)
        sepsis_mask = active & (state[ROW_STAGE] == STAGE_SEPSIS)
        progressed2 = sepsis_mask & (draws2 < self.prog_sepsis * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed2] = STAGE_SEVERE
        new_state[ROW_STAGE_TIMER, progressed2] = 0

        draws3 = rng.random(n)
        severe_mask = active & (state[ROW_STAGE] == STAGE_SEVERE)
        progressed3 = severe_mask & (draws3 < self.prog_severe * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed3] = STAGE_SHOCK
        new_state[ROW_STAGE_TIMER, progressed3] = 0

        mort_draws = rng.random(n)
        deceased = (
            (sepsis_mask & ~progressed2 & (mort_draws < self.mort_sepsis)) |
            (severe_mask & ~progressed3 & (mort_draws < self.mort_severe)) |
            (active & (state[ROW_STAGE] == STAGE_SHOCK) & (mort_draws < self.mort_shock))
        )
        new_state[ROW_STAGE, deceased] = STAGE_DECEASED

        new_state[ROW_LOS_REMAINING] = np.maximum(state[ROW_LOS_REMAINING] - 1, 0)
        discharge_eligible = (
            active & ~deceased
            & (new_state[ROW_LOS_REMAINING] <= 0)
            & (state[ROW_STAGE] < STAGE_SEVERE)
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

        true_labels = ((state[ROW_STAGE] >= STAGE_SEPSIS) & (state[ROW_STAGE] < STAGE_DECEASED)).astype(int)
        true_risks = state[ROW_CURRENT_RISK]

        if not self._model_fitted:
            self._model.fit(true_labels, true_risks, rng, n_iterations=3)
            self._model_fitted = True

        scores = self._model.predict(true_risks, rng, true_labels=true_labels)
        active = state[ROW_STAGE] < STAGE_DECEASED
        scores = np.where(active, scores, 0.0)

        # TREWS uses 7% alert rate (much lower burden than ESM)
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
                "n_active": int(active.sum()),
                "n_flagged": int(labels.sum()),
                "alert_rate": float(labels[active].mean()) if active.any() else 0.0,
                "prevalence": float(true_labels[active].mean()) if active.any() else 0.0,
            },
        )

    def intervene(self, state, predictions, t):
        """
        TREWS two-stage: alert fires → clinician confirms within 3h → treatment.
        The confirmation step is key: not all alerts lead to treatment.
        """
        rng = self.rng.intervention
        new_state = state.copy()
        n = state.shape[1]

        flagged = predictions.labels == 1
        flagged_indices = np.where(flagged)[0]

        # Stage 1: Alert fires
        # Stage 2: Clinician confirms within 3h (confirmation_rate)
        # Confirmation is higher quality filter than ESM response — less fatigue
        response_rate = max(
            self.floor_response_rate,
            self.initial_response_rate * np.exp(-self.fatigue_coefficient * self._cumulative_false_alerts)
        )
        confirms = rng.random(len(flagged_indices)) < (response_rate * self.confirmation_rate)
        confirmed_indices = flagged_indices[confirms]

        if len(confirmed_indices) > self.rapid_response_capacity:
            scores_confirmed = predictions.scores[confirmed_indices]
            priority_order = np.argsort(-scores_confirmed)
            confirmed_indices = confirmed_indices[priority_order[:self.rapid_response_capacity]]

        # Mark alert as confirmed
        new_state[ROW_ALERT_CONFIRMED, confirmed_indices] = 1

        # Treatment only for confirmed alerts
        newly_treated = confirmed_indices[new_state[ROW_TREATED, confirmed_indices] == 0]
        new_state[ROW_TREATED, newly_treated] = 1
        new_state[ROW_TREATMENT_TIMER, newly_treated] = 0

        true_labels = predictions.metadata.get("true_labels", np.zeros(n))
        false_alert_indices = flagged_indices[true_labels[flagged_indices] == 0]
        new_state[ROW_FALSE_ALERTS, false_alert_indices] += 1
        self._cumulative_false_alerts += len(false_alert_indices)

        return new_state, Interventions(
            treated_indices=confirmed_indices,
            metadata={
                "n_flagged": int(flagged.sum()),
                "n_confirmed": int(len(confirmed_indices)),
                "n_newly_treated": int(len(newly_treated)),
                "response_rate": round(response_rate, 4),
                "confirmation_rate_applied": round(self.confirmation_rate, 4),
            },
        )

    def measure(self, state, t):
        n = state.shape[1]
        events = ((state[ROW_STAGE] >= STAGE_SEPSIS) & (state[ROW_STAGE] < STAGE_DECEASED)).astype(float)
        mortality = (state[ROW_STAGE] == STAGE_DECEASED).astype(float)
        race_labels = np.array(RACE_LABELS)[state[ROW_DEMO_RACE].astype(int)]

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "mortality": mortality,
                "stage": state[ROW_STAGE].copy(),
                "treated": state[ROW_TREATED].copy(),
                "alert_confirmed": state[ROW_ALERT_CONFIRMED].copy(),
                "race_ethnicity": race_labels,
            },
            metadata={
                "deceased": int((state[ROW_STAGE] == STAGE_DECEASED).sum()),
                "mortality_rate": round(float((state[ROW_STAGE] == STAGE_DECEASED).sum()) / max(1, n), 4),
                "confirmed_alert_fraction": round(float((state[ROW_ALERT_CONFIRMED] == 1).sum()) / max(1, n), 4),
            },
        )

    def clone_state(self, state):
        return state.copy()


def run_main():
    print("\n=== Paper 02: TREWS (Adams et al. 2022) ===")
    print("Calibrating to: sensitivity=0.80, PPV=0.27, alert_rate=7%")
    print("Target: 3.3pp mortality reduction in confirmed-alert cohort")
    print("Paper baseline: 18.7% → 15.4% in-hospital mortality")

    tc = TimeConfig(
        n_timesteps=84,
        timestep_duration=4/24/365,
        timestep_unit="4h",
        prediction_schedule=list(range(84)),
    )

    n = 5000
    scenario = TREWSScenario(
        time_config=tc,
        seed=42,
        model_sensitivity=0.80,
        model_ppv=0.27,
        alert_threshold_percentile=93.0,
        confirmation_rate=0.65,
        treatment_effectiveness=0.18,
        sepsis_prevalence=0.038,
        initial_response_rate=0.80,
        fatigue_coefficient=0.001,
    )

    engine = BranchedSimulationEngine(scenario, counterfactual_mode=CounterfactualMode.BRANCHED)
    results = engine.run(n)

    # Aggregate results
    last_t = tc.n_timesteps - 1
    factual_mort_rate = 0.0
    cf_mort_rate = 0.0
    if last_t in results.outcomes:
        factual_mort_rate = float(results.outcomes[last_t].secondary["mortality"].sum()) / n
    if last_t in results.counterfactual_outcomes:
        cf_mort_rate = float(results.counterfactual_outcomes[last_t].secondary["mortality"].sum()) / n

    mortality_delta = cf_mort_rate - factual_mort_rate

    # Alert rate
    n_flagged_total = 0
    n_active_total = 0
    for t in range(tc.n_timesteps):
        if t in results.predictions:
            meta = results.predictions[t].metadata
            n_flagged_total += meta.get("n_flagged", 0)
            n_active_total += meta.get("n_active", 0)
    avg_alert_rate = n_flagged_total / max(n_active_total, 1)

    print(f"\n--- RESULTS ---")
    print(f"N patients: {n}")
    print(f"Factual mortality rate (with TREWS): {factual_mort_rate:.4f} ({factual_mort_rate*100:.2f}%)")
    print(f"Counterfactual mortality rate (no TREWS): {cf_mort_rate:.4f} ({cf_mort_rate*100:.2f}%)")
    print(f"Mortality reduction: {mortality_delta*100:.2f} pp [paper reports 3.3pp in confirmed-alert cohort]")
    print(f"Average alert rate: {avg_alert_rate*100:.2f}% [paper reports 7%]")

    if scenario._model._fit_report:
        r = scenario._model._fit_report
        print(f"\nML Model fit report:")
        print(f"  Achieved AUC: {r.get('achieved_auc', 'N/A'):.3f}")
        print(f"  Achieved sensitivity: {r.get('achieved_sensitivity', 'N/A'):.3f} [target 0.80]")
        print(f"  Achieved PPV: {r.get('achieved_ppv', 'N/A'):.3f} [target 0.27]")

    # Verification
    print(f"\n--- VERIFICATION ---")
    checks = {}
    sample_scores = results.predictions.get(5)
    if sample_scores:
        checks["no_nan_inf"] = not np.any(np.isnan(sample_scores.scores)) and not np.any(np.isinf(sample_scores.scores))
        checks["scores_in_unit_interval"] = bool(np.all(sample_scores.scores >= 0) and np.all(sample_scores.scores <= 1))
    checks["alert_rate_in_range"] = abs(avg_alert_rate - 0.07) < 0.12
    checks["mortality_plausible"] = 0.001 < factual_mort_rate < 0.30
    print(f"  No NaN/Inf: {checks.get('no_nan_inf', 'N/A')}")
    print(f"  Scores in [0,1]: {checks.get('scores_in_unit_interval', 'N/A')}")
    print(f"  Alert rate in range: {checks['alert_rate_in_range']} ({avg_alert_rate*100:.1f}%)")
    print(f"  Mortality plausible: {checks['mortality_plausible']}")
    print(f"  OVERALL: {'PASS' if all(checks.values()) else 'PARTIAL'} ({sum(checks.values())}/{len(checks)})")

    return {
        "factual_mortality_rate": factual_mort_rate,
        "cf_mortality_rate": cf_mort_rate,
        "mortality_delta_pp": mortality_delta * 100,
        "avg_alert_rate": avg_alert_rate,
        "checks": checks,
    }


if __name__ == "__main__":
    r = run_main()
    print(f"\n=== SUMMARY ===")
    print(json.dumps(r, indent=2))
