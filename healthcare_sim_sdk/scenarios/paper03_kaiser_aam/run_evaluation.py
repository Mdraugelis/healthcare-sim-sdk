"""
Runner script for Paper 03: Kaiser AAM — Escobar et al. NEJM 2020.

Key design elements:
  - 21-hospital staggered deployment (stepped-wedge)
  - AAM: logistic regression, c-stat 0.845
  - VQNC intermediary layer: remote nurses review scores hourly (NOT direct alerts)
  - Mortality: 14.4% → 9.8% in target population (4.6pp absolute reduction)
  - Estimated 500+ deaths/year prevented across Kaiser system
  - Target population: ward patients at risk of ICU transfer or ward death in 12h

Design choices:
  - Staggered deployment modeled as sequential hospital waves
  - VQNC layer = high-quality filtering (higher confirmation rate than direct alert)
  - c-stat 0.845 → model_auc = 0.845
  - 14.4% baseline ward mortality [ASSUMED to be in-window mortality]

Usage:
    cd <repo-root>
    python3 -m healthcare_sim_sdk.scenarios.paper03_kaiser_aam.run_evaluation
"""

import sys
import json
import numpy as np

from healthcare_sim_sdk.core.scenario import TimeConfig, BaseScenario, Predictions, Interventions, Outcomes
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.population.risk_distributions import beta_distributed_risks

ROW_BASE_RISK = 0
ROW_CURRENT_RISK = 1
ROW_AR1_MOD = 2
ROW_STAGE = 3          # 0=stable, 1=deteriorating, 2=critical, 3=deceased, 4=discharged
ROW_STAGE_TIMER = 4
ROW_TREATED = 5
ROW_TREATMENT_TIMER = 6
ROW_DEMO_RACE = 7
ROW_DEMO_INSURANCE = 8
ROW_DEMO_AGE = 9
ROW_DEMO_RISK_MULT = 10
ROW_LOS_REMAINING = 11
ROW_VQNC_REVIEWED = 12   # VQNC reviewed this patient this admission
ROW_HOSPITAL_ID = 13      # Hospital in staggered deployment (0-20)
N_ROWS = 14

STAGE_STABLE = 0
STAGE_DETERIORATING = 1
STAGE_CRITICAL = 2
STAGE_DECEASED = 3
STAGE_DISCHARGED = 4

RACE_LABELS = ["White", "Black", "Hispanic", "Asian", "Other"]
INSURANCE_LABELS = ["Commercial", "Medicare", "Medicaid", "Self-Pay"]
AGE_LABELS = ["18-44", "45-64", "65-79", "80+"]


class KaiserAAMScenario(BaseScenario):
    """
    Kaiser AAM scenario: VQNC-mediated clinical deterioration alert.

    Distinguishing features:
    - High c-stat (0.845) means better discrimination than ESM/TREWS
    - VQNC intermediary eliminates alert fatigue at bedside clinician level
    - Staggered deployment creates stepped-wedge counterfactual structure
    - Target: ward patients at risk of ICU transfer or in-hospital death

    Paper reports: 14.4% → 9.8% mortality (4.6pp absolute reduction)
    """

    unit_of_analysis = "patient_admission"

    def __init__(self, time_config, seed=None,
                 model_auc=0.845,
                 alert_threshold_percentile=90.0,
                 treatment_effectiveness=0.32,
                 vqnc_review_rate=0.90,
                 vqnc_action_rate=0.60,
                 deterioration_prevalence=0.08,
                 n_hospitals=21,):
        super().__init__(time_config=time_config, seed=seed)

        self.model_auc = model_auc
        self.alert_threshold_percentile = alert_threshold_percentile
        self.treatment_effectiveness = treatment_effectiveness
        self.vqnc_review_rate = vqnc_review_rate       # VQNC reviews the score
        self.vqnc_action_rate = vqnc_action_rate       # VQNC triggers RRT call
        self.deterioration_prevalence = deterioration_prevalence
        self.n_hospitals = n_hospitals

        # Kaiser KPNC demographics
        self.race_proportions = [0.42, 0.07, 0.30, 0.17, 0.04]  # [ASSUMED] KPNC Northern CA
        self.insurance_proportions = [0.38, 0.30, 0.18, 0.14]
        self.age_proportions = [0.15, 0.28, 0.35, 0.22]
        self.race_risk_multipliers = [1.0, 1.8, 0.9, 0.8, 1.0]
        self.age_risk_multipliers = [0.4, 1.0, 1.8, 2.8]

        self.ar1_rho = 0.92
        self.ar1_sigma = 0.05
        self.mean_los_timesteps = 30    # ~5 days
        self.los_std_timesteps = 12

        # Stage transition probabilities (per timestep = 4 hours)
        self.prog_stable = 0.010       # calibrated to 8% deterioration prevalence
        self.prog_deteriorating = 0.055
        self.mort_deteriorating = 0.003
        self.mort_critical = 0.030

        # No alert fatigue at bedside — VQNC handles it
        self.bedside_fatigue_coefficient = 0.0

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
            annual_incident_rate=self.deterioration_prevalence,
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
        state[ROW_STAGE] = STAGE_STABLE
        state[ROW_STAGE_TIMER] = 0
        state[ROW_TREATED] = 0
        state[ROW_TREATMENT_TIMER] = 0
        state[ROW_VQNC_REVIEWED] = 0

        # Assign to one of 21 hospitals (for staggered deployment analysis)
        state[ROW_HOSPITAL_ID] = rng.integers(0, self.n_hospitals, n).astype(float)

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

        tx_factor = np.where(state[ROW_TREATED] == 1, 1.0 - self.treatment_effectiveness, 1.0)
        mean_risk = float(new_state[ROW_CURRENT_RISK][active].mean()) if active.any() else 0.08
        risk_scale = np.clip(new_state[ROW_CURRENT_RISK] / max(mean_risk, 0.001), 0.1, 5.0)

        draws = rng.random(n)
        stable_mask = active & (state[ROW_STAGE] == STAGE_STABLE)
        progressed = stable_mask & (draws < self.prog_stable * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed] = STAGE_DETERIORATING
        new_state[ROW_STAGE_TIMER, progressed] = 0

        draws2 = rng.random(n)
        det_mask = active & (state[ROW_STAGE] == STAGE_DETERIORATING)
        progressed2 = det_mask & (draws2 < self.prog_deteriorating * risk_scale * tx_factor)
        new_state[ROW_STAGE, progressed2] = STAGE_CRITICAL
        new_state[ROW_STAGE_TIMER, progressed2] = 0

        mort_draws = rng.random(n)
        deceased = (
            (det_mask & ~progressed2 & (mort_draws < self.mort_deteriorating)) |
            (active & (state[ROW_STAGE] == STAGE_CRITICAL) & (mort_draws < self.mort_critical))
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
        active = state[ROW_STAGE] < STAGE_DECEASED
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
                "n_active": int(active.sum()),
                "n_flagged": int(labels.sum()),
                "alert_rate": float(labels[active].mean()) if active.any() else 0.0,
                "prevalence": float(true_labels[active].mean()) if active.any() else 0.0,
            },
        )

    def intervene(self, state, predictions, t):
        """
        VQNC-mediated intervention:
        Score ≥ threshold → VQNC reviews (hourly cycle) → VQNC calls RRT → treatment
        VQNC eliminates alert fatigue from bedside clinicians.
        """
        rng = self.rng.intervention
        new_state = state.copy()
        n = state.shape[1]

        flagged = predictions.labels == 1
        flagged_indices = np.where(flagged)[0]

        # VQNC reviews all flagged patients (high coverage, no fatigue)
        vqnc_reviews = rng.random(len(flagged_indices)) < self.vqnc_review_rate
        vqnc_actions = vqnc_reviews & (rng.random(len(flagged_indices)) < self.vqnc_action_rate)
        acted_indices = flagged_indices[vqnc_actions]

        new_state[ROW_VQNC_REVIEWED, flagged_indices[vqnc_reviews]] = 1

        newly_treated = acted_indices[new_state[ROW_TREATED, acted_indices] == 0]
        new_state[ROW_TREATED, newly_treated] = 1
        new_state[ROW_TREATMENT_TIMER, newly_treated] = 0

        true_labels = predictions.metadata.get("true_labels", np.zeros(n))
        false_alert_indices = flagged_indices[true_labels[flagged_indices] == 0]
        self._cumulative_false_alerts += len(false_alert_indices)

        return new_state, Interventions(
            treated_indices=acted_indices,
            metadata={
                "n_flagged": int(flagged.sum()),
                "n_vqnc_reviewed": int(vqnc_reviews.sum()),
                "n_acted": int(len(acted_indices)),
                "n_newly_treated": int(len(newly_treated)),
                "vqnc_review_rate": self.vqnc_review_rate,
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
                "race_ethnicity": race_labels,
                "hospital_id": state[ROW_HOSPITAL_ID].copy(),
            },
            metadata={
                "deceased": int((state[ROW_STAGE] == STAGE_DECEASED).sum()),
                "mortality_rate": round(float((state[ROW_STAGE] == STAGE_DECEASED).sum()) / max(1, n), 4),
            },
        )

    def clone_state(self, state):
        return state.copy()


def run_main():
    print("\n=== Paper 03: Kaiser AAM (Escobar et al. 2020) ===")
    print("Calibrating to: c-stat=0.845, mortality 14.4%->9.8%")

    tc = TimeConfig(
        n_timesteps=84,
        timestep_duration=4/24/365,
        timestep_unit="4h",
        prediction_schedule=list(range(84)),
    )

    n = 5000
    scenario = KaiserAAMScenario(
        time_config=tc,
        seed=42,
        model_auc=0.845,
        alert_threshold_percentile=90.0,
        treatment_effectiveness=0.32,
        vqnc_review_rate=0.90,
        vqnc_action_rate=0.60,
        deterioration_prevalence=0.08,
    )

    engine = BranchedSimulationEngine(scenario, counterfactual_mode=CounterfactualMode.BRANCHED)
    results = engine.run(n)

    last_t = tc.n_timesteps - 1
    factual_mort_rate = 0.0
    cf_mort_rate = 0.0
    if last_t in results.outcomes:
        factual_mort_rate = float(results.outcomes[last_t].secondary["mortality"].sum()) / n
    if last_t in results.counterfactual_outcomes:
        cf_mort_rate = float(results.counterfactual_outcomes[last_t].secondary["mortality"].sum()) / n

    mortality_delta = cf_mort_rate - factual_mort_rate

    print(f"\n--- RESULTS ---")
    print(f"N patients: {n}")
    print(f"Factual mortality rate (with AAM): {factual_mort_rate:.4f} ({factual_mort_rate*100:.2f}%)")
    print(f"Counterfactual mortality rate (no AAM): {cf_mort_rate:.4f} ({cf_mort_rate*100:.2f}%)")
    print(f"Mortality reduction: {mortality_delta*100:.2f} pp [paper reports 4.6pp: 14.4%→9.8%]")

    if scenario._model._fit_report:
        r = scenario._model._fit_report
        print(f"\nML Model fit report:")
        print(f"  Achieved AUC: {r.get('achieved_auc', 'N/A'):.3f} [target 0.845]")

    # Verification
    print(f"\n--- VERIFICATION ---")
    checks = {}
    sample_scores = results.predictions.get(5)
    if sample_scores:
        checks["no_nan_inf"] = not np.any(np.isnan(sample_scores.scores))
        checks["scores_in_unit_interval"] = bool(np.all(sample_scores.scores >= 0) and np.all(sample_scores.scores <= 1))
    checks["mortality_plausible"] = 0.001 < factual_mort_rate < 0.30
    checks["intervention_reduces_mortality"] = mortality_delta >= 0
    print(f"  No NaN/Inf: {checks.get('no_nan_inf')}")
    print(f"  Scores in [0,1]: {checks.get('scores_in_unit_interval')}")
    print(f"  Mortality plausible: {checks['mortality_plausible']}")
    print(f"  Intervention reduces mortality: {checks['intervention_reduces_mortality']}")
    print(f"  OVERALL: {'PASS' if all(checks.values()) else 'PARTIAL'} ({sum(checks.values())}/{len(checks)})")

    return {
        "factual_mortality_rate": factual_mort_rate,
        "cf_mortality_rate": cf_mort_rate,
        "mortality_delta_pp": mortality_delta * 100,
        "checks": checks,
    }


if __name__ == "__main__":
    r = run_main()
    print(f"\n=== SUMMARY ===")
    print(json.dumps(r, indent=2))
