"""Paper 07: Manz et al. — ML Mortality Prediction + Behavioral Nudges.

JAMA Oncology 2020. Penn Medicine. 78 oncologists, 14,607 patients.
Stepped-wedge cluster RCT. GBM 180-day mortality prediction at >=10% threshold.
Behavioral nudges: weekly emails + peer comparison + opt-out text prompts.

Primary outcome: Serious Illness Conversation (SIC) rate.
  - Overall: ~1% -> ~5% (5x increase)
  - High-risk patients: ~4% -> ~15% (3.75x increase)

Modeling approach:
  - Unit: cancer patient visit (encounter-level)
  - Entities: patients (14,607 patients across 78 oncologists)
  - State: [mortality_risk, sic_completed, high_risk_flag, oncologist_id]
  - Timestep: 1 week
  - Prediction: GBM 180-day mortality (AUC not reported in abstract;
    assume 0.78 based on typical GBM performance at this threshold)
  - Intervention: Nudge delivery changes oncologist's p(SIC|patient)

Key design challenge:
  The causal chain is: ML score -> nudge to oncologist -> behavioral change
  -> SIC documentation. The SDK models the patient-level outcome, but the
  intervention acts via physician behavior change. We model this as:
  p(SIC | high_risk, nudge) = p_sic_high_nudge
  p(SIC | high_risk, no_nudge) = p_sic_high_baseline

  Stepped-wedge: We model intervention activation at t=13 (week 13,
  when ~half of clinic groups have crossed over, based on 4 periods / 8 groups).

RNG DISCIPLINE:
- create_population() -> self.rng.population
- step()              -> self.rng.temporal
- predict()           -> self.rng.prediction
- intervene()         -> self.rng.intervention
- measure()           -> self.rng.outcomes
"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from healthcare_sim_sdk.ml.model import ControlledMLModel


@dataclass
class ManzState:
    """State for the Manz behavioral nudge scenario.

    Attributes:
        mortality_risk: True 180-day mortality risk per patient (n,)
        sic_completed: Cumulative flag for SIC (n,) - once done, done
        oncologist_id: Oncologist assignment per patient (n,)
        nudge_active: Whether oncologist is in nudge condition (n_oncologists,)
        n_patients: Total patients
        n_oncologists: Number of oncologists
    """
    mortality_risk: np.ndarray
    sic_completed: np.ndarray
    oncologist_id: np.ndarray
    nudge_active: np.ndarray
    n_patients: int
    n_oncologists: int


class ManzNudgeScenario(BaseScenario[ManzState]):
    """Manz et al. stepped-wedge behavioral nudge simulation.

    Models the effect of ML-triggered behavioral nudges on serious illness
    conversation rates in oncology care.

    Key parameters (paper-derived):
      - n_oncologists = 78
      - base_mortality_rate = 0.40 (40% 180-day mortality typical in
            advanced cancer; paper reports >=10% threshold but doesn't
            give overall mortality rate -> ASSUMED: HIGH uncertainty)
      - model_auc = 0.78 (ASSUMED: paper does not report AUC for the
            GBM; typical GBM AUC for mortality prediction ~0.75-0.82)
      - mortality_threshold = 0.10 (paper-specified)
      - high_risk_fraction: ~0.30 of patients at >=10% threshold (ASSUMED)
      - base_sic_rate_overall = 0.01 (paper-reported)
      - base_sic_rate_high_risk = 0.04 (paper-reported)
      - intervention_sic_rate_overall = 0.05 (paper-reported)
      - intervention_sic_rate_high_risk = 0.15 (paper-reported)
    """

    unit_of_analysis = "patient_encounter"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        n_oncologists: int = 78,
        base_mortality_prevalence: float = 0.40,  # ASSUMED
        model_auc: float = 0.78,  # ASSUMED
        mortality_threshold: float = 0.10,  # paper-specified
        base_sic_rate_low_risk: float = 0.0025,  # derived (see analysis)
        base_sic_rate_high_risk: float = 0.04,   # paper-reported
        nudge_sic_rate_low_risk: float = 0.025,  # derived from overall 5%
        nudge_sic_rate_high_risk: float = 0.15,  # paper-reported
        stepped_wedge_start: int = 13,  # week 13 (approximate midpoint)
    ):
        super().__init__(time_config=time_config, seed=seed)
        self.n_oncologists = n_oncologists
        self.base_mortality_prevalence = base_mortality_prevalence
        self.model_auc = model_auc
        self.mortality_threshold = mortality_threshold
        self.base_sic_rate_low_risk = base_sic_rate_low_risk
        self.base_sic_rate_high_risk = base_sic_rate_high_risk
        self.nudge_sic_rate_low_risk = nudge_sic_rate_low_risk
        self.nudge_sic_rate_high_risk = nudge_sic_rate_high_risk
        self.stepped_wedge_start = stepped_wedge_start

        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=model_auc,
        )
        self._model_fitted = False

    def create_population(self, n_entities: int) -> ManzState:
        """Initialize cancer patient population with mortality risk.

        Distribution design: ~20% of patients above 10% threshold
        (high-risk), 80% below. This is consistent with:
          overall_SIC_base = 0.01 = 0.20*0.04 + 0.80*0.0025

        Use a bimodal-like distribution:
          - 80% patients: low risk (1-8%), mean ~4%
          - 20% patients: high risk (10-60%), mean ~25%
        This captures the fact that GBM threshold >=10% at Penn Medicine
        selects a minority of advanced cancer patients.
        [ASSUMED: high-risk fraction 20%, paper does not report directly]
        """
        rng = self.rng.population
        n = n_entities

        # Low-risk cohort: beta(0.5, 15) -> highly concentrated near 0
        n_low = int(n * 0.80)
        n_high = n - n_low
        low_risks = rng.beta(0.5, 15.0, n_low)  # mean ~0.032, most <0.08

        # High-risk cohort: beta(2, 5) -> mean ~0.286
        high_risks = rng.beta(2.0, 5.0, n_high)  # mean ~0.286, mostly >0.10

        all_risks = np.concatenate([low_risks, high_risks])
        rng.shuffle(all_risks)
        mortality_risk = np.clip(all_risks, 0.005, 0.99)

        # Assign patients to oncologists (unequal loads, realistic)
        # Patients-per-oncologist: ~14607/78 ≈ 187 per oncologist
        oncologist_id = rng.integers(0, self.n_oncologists, n_entities)

        # SIC not yet completed
        sic_completed = np.zeros(n_entities, dtype=float)

        # Nudge not yet active for any oncologist
        nudge_active = np.zeros(self.n_oncologists, dtype=float)

        return ManzState(
            mortality_risk=mortality_risk,
            sic_completed=sic_completed,
            oncologist_id=oncologist_id,
            nudge_active=nudge_active,
            n_patients=n_entities,
            n_oncologists=self.n_oncologists,
        )

    def step(self, state: ManzState, t: int) -> ManzState:
        """Weekly temporal evolution: patient panel turns over slightly.

        Cancer patients: some die/leave, new patients arrive.
        Model as ~2% weekly turnover. Risks of remaining patients
        drift slightly as disease progresses.
        """
        rng = self.rng.temporal
        n = state.n_patients

        # Slight upward drift in mortality risk (disease progression)
        drift = rng.normal(0.001, 0.005, n)
        new_risk = np.clip(state.mortality_risk + drift, 0.01, 0.99)

        # ~2% patient turnover per week (new referrals replace those lost)
        turnover_mask = rng.random(n) < 0.02
        n_new = int(turnover_mask.sum())
        if n_new > 0:
            alpha = 1.5
            beta_p = alpha * (1 / self.base_mortality_prevalence - 1)
            new_risks = rng.beta(alpha, beta_p, n_new)
            scaling = self.base_mortality_prevalence / np.mean(new_risks)
            new_risk[turnover_mask] = np.clip(
                new_risks * scaling, 0.01, 0.99
            )
            state.sic_completed[turnover_mask] = 0.0  # reset for new patients

        state.mortality_risk = new_risk
        return state

    def predict(self, state: ManzState, t: int) -> Predictions:
        """GBM predicts 180-day mortality for each patient."""
        true_risks = state.mortality_risk
        n = len(true_risks)

        if not self._model_fitted:
            true_labels = (
                self.rng.prediction.random(n) < true_risks
            ).astype(int)
            self._model.fit(
                true_labels, true_risks,
                self.rng.prediction, n_iterations=5,
            )
            self._model_fitted = True

        scores = self._model.predict(true_risks, self.rng.prediction)
        high_risk = (scores >= self.mortality_threshold).astype(float)

        return Predictions(
            scores=scores,
            labels=high_risk.astype(int),
            metadata={
                "n_high_risk": int(high_risk.sum()),
                "pct_high_risk": float(high_risk.mean()),
            },
        )

    def intervene(
        self, state: ManzState, predictions: Predictions, t: int
    ) -> tuple[ManzState, Interventions]:
        """Activate nudge for oncologists in intervention condition.

        Stepped-wedge: all oncologists eventually receive intervention.
        At t=stepped_wedge_start, ~50% of oncologists receive nudge.
        At t=stepped_wedge_start+13, all oncologists receive nudge.

        The nudge doesn't directly change patient state — it changes
        oncologist behavior probability, captured in measure().
        """
        # Determine which oncologists are in nudge condition at time t
        if t >= self.stepped_wedge_start + 13:
            # All oncologists nudged by end of study
            n_nudged = self.n_oncologists
        elif t >= self.stepped_wedge_start:
            # Linear ramp from 50% to 100%
            frac = 0.5 + 0.5 * (t - self.stepped_wedge_start) / 13
            n_nudged = int(frac * self.n_oncologists)
        else:
            n_nudged = 0

        # Activate nudge for first n_nudged oncologists
        state.nudge_active = np.zeros(self.n_oncologists, dtype=float)
        if n_nudged > 0:
            state.nudge_active[:n_nudged] = 1.0

        # Identify which patients have nudged oncologists + high risk flag
        high_risk = predictions.labels if predictions.labels is not None else (
            predictions.scores >= self.mortality_threshold
        ).astype(int)
        patient_nudged = state.nudge_active[state.oncologist_id]
        treated_indices = np.where(
            (patient_nudged == 1.0) & (high_risk == 1)
        )[0]

        return state, Interventions(
            treated_indices=treated_indices,
            metadata={
                "n_oncologists_nudged": n_nudged,
                "n_patients_targeted": len(treated_indices),
                "timestep": t,
            },
        )

    def measure(self, state: ManzState, t: int) -> Outcomes:
        """Measure SIC rate as function of risk status and nudge condition.

        SIC probability:
          - High-risk + nudge: nudge_sic_rate_high_risk
          - High-risk + no nudge: base_sic_rate_high_risk
          - Low-risk + nudge: slightly elevated (spillover)
          - Low-risk + no nudge: base_sic_rate_overall (low-risk component)
        """
        rng = self.rng.outcomes
        n = state.n_patients

        # Determine current high-risk status (using true risk for measurement)
        high_risk = (state.mortality_risk >= self.mortality_threshold).astype(float)
        patient_nudged = state.nudge_active[state.oncologist_id]

        # SIC probabilities by cell
        sic_prob = np.where(
            (high_risk == 1) & (patient_nudged == 1),
            self.nudge_sic_rate_high_risk,
            np.where(
                (high_risk == 1) & (patient_nudged == 0),
                self.base_sic_rate_high_risk,
                np.where(
                    (high_risk == 0) & (patient_nudged == 1),
                    self.nudge_sic_rate_low_risk,  # nudge also helps low-risk
                    self.base_sic_rate_low_risk,   # baseline low-risk
                )
            )
        )

        # Sample SIC events this week (per-encounter rate, not lifetime)
        # Paper reports SIC rate as fraction of eligible encounters having
        # SIC documentation — not whether patient ever had one
        sic_events = (rng.random(n) < sic_prob).astype(float)

        # Update cumulative SIC flag (for tracking, not for suppression)
        state.sic_completed = np.clip(
            state.sic_completed + sic_events, 0, 1
        )

        return Outcomes(
            events=sic_events,
            entity_ids=np.arange(n),
            secondary={
                "high_risk": high_risk,
                "nudge_active": patient_nudged,
                "sic_cumulative": state.sic_completed.copy(),
            },
            metadata={
                "sic_rate_this_week": float(sic_events.mean()),
                "pct_high_risk": float(high_risk.mean()),
                "pct_nudged": float(patient_nudged.mean()),
                "n_oncologists_nudged": int(state.nudge_active.sum()),
            },
        )

    def clone_state(self, state: ManzState) -> ManzState:
        import copy
        return ManzState(
            mortality_risk=state.mortality_risk.copy(),
            sic_completed=state.sic_completed.copy(),
            oncologist_id=state.oncologist_id.copy(),
            nudge_active=state.nudge_active.copy(),
            n_patients=state.n_patients,
            n_oncologists=state.n_oncologists,
        )
