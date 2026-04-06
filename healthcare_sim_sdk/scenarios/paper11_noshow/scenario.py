"""Paper 11: No-Show Prediction Targeted Phone Reminders.

Two papers modeled jointly:
  (A) Chong et al. — AJR 2020, Singapore MRI center
      XGBoost AUC 0.74, top-25% targeting, no-show 19.3%->15.9%
      4.2x efficiency gain over random calling

  (B) Rosen et al. — JGIM 2023, VA primary care, RCT
      Random forest, no-show 36%->33% overall
      Black patients: 42%->36% (p<0.001), equity improvement demonstrated

Combined simulation models MRI/outpatient appointment scheduling with
targeted phone reminders as the intervention. Equity analysis is primary.

Unit of analysis: appointment
State: numpy array [n_appointments x 5]
  col 0: true_noshow_risk
  col 1: race_code (0=White, 1=Black, 2=Hispanic, 3=Asian, 4=Other)
  col 2: received_reminder (0/1)
  col 3: reminder_effective (0/1)
  col 4: showed (0/1)

Key parameters:
  Chong: base_noshow_rate=0.193, model_auc=0.74, threshold=top-25%,
         reminder_effectiveness=0.17 (reduces 19.3->15.9, ratio ~0.83)
  Rosen: base_noshow_rate=0.36, model_auc=~0.75 (ASSUMED),
         overall reduction: 36->33% (8.3%), Black: 42->36% (14.3%)

We simulate the Rosen RCT setting (VA primary care) as primary,
with Chong parameters used for sensitivity analysis.

Equity modeling:
  Black patients: higher baseline no-show (42%), larger reminder benefit
  White patients: lower baseline (30-32%), smaller reminder benefit
  This mirrors Rosen's finding that equity IMPROVED under ML-targeted outreach.

RNG DISCIPLINE:
- create_population() -> self.rng.population
- step()              -> self.rng.temporal
- predict()           -> self.rng.prediction
- intervene()         -> self.rng.intervention
- measure()           -> self.rng.outcomes
"""

from typing import Optional, List
import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from healthcare_sim_sdk.ml.model import ControlledMLModel

# Column indices
COL_RISK = 0
COL_RACE = 1
COL_REMINDED = 2
COL_REMIND_EFF = 3
COL_SHOWED = 4

# Race codes
RACE_WHITE = 0
RACE_BLACK = 1
RACE_HISPANIC = 2
RACE_ASIAN = 3
RACE_OTHER = 4

# Race names for equity analysis
RACE_NAMES = {
    RACE_WHITE: "White",
    RACE_BLACK: "Black",
    RACE_HISPANIC: "Hispanic",
    RACE_ASIAN: "Asian",
    RACE_OTHER: "Other",
}

# VA demographics (approximate, from Rosen study context)
RACE_DIST = {
    RACE_WHITE: 0.55,
    RACE_BLACK: 0.25,    # VA serves ~25% Black patients
    RACE_HISPANIC: 0.12,
    RACE_ASIAN: 0.04,
    RACE_OTHER: 0.04,
}

# Baseline no-show rates by race (Rosen: Black 42%, overall 36%)
RACE_NOSHOW_BASE = {
    RACE_WHITE: 0.32,
    RACE_BLACK: 0.42,   # paper-derived
    RACE_HISPANIC: 0.38,
    RACE_ASIAN: 0.28,
    RACE_OTHER: 0.36,
}

# Reminder effectiveness (relative risk reduction of no-show)
# Rosen: Black patients 42%->36% = 14.3% RRR
# Overall: 36%->33% = 8.3% RRR
# White implied: ~30-32%->~27-29% ≈ 10% RRR
RACE_REMIND_EFFECTIVENESS = {
    RACE_WHITE: 0.10,   # ASSUMED: modest benefit
    RACE_BLACK: 0.143,  # paper-derived: (42-36)/42 = 14.3%
    RACE_HISPANIC: 0.12,  # ASSUMED: intermediate
    RACE_ASIAN: 0.08,   # ASSUMED: lower benefit (already low base rate)
    RACE_OTHER: 0.10,   # ASSUMED
}


class NoShowReminderScenario(BaseScenario[np.ndarray]):
    """No-show targeted phone reminder simulation (Chong + Rosen).

    Models ML-targeted phone reminders for appointment no-show reduction.
    Primary focus: Rosen RCT (VA primary care) with equity analysis.

    Key design decisions:
      - Targeting = top-K% risk (top 25% per Chong; similar for Rosen)
      - Reminder capacity: not explicitly constrained (paper doesn't report)
      - Race-stratified effects modeled from Rosen equity finding
    """

    unit_of_analysis = "appointment"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        # Rosen RCT parameters
        base_noshow_rate: float = 0.36,      # paper-derived
        model_auc: float = 0.75,             # ASSUMED (Rosen doesn't report AUC)
        target_top_k: float = 0.25,          # top 25% (Chong); used for Rosen too
        # Effectiveness
        reminder_reduce_noshow: float = 0.083,  # overall RRR (paper-derived)
        # Equity
        model_race_equalized: bool = True,   # assume ML has similar AUC across race
    ):
        super().__init__(time_config=time_config, seed=seed)
        self.base_noshow_rate = base_noshow_rate
        self.model_auc = model_auc
        self.target_top_k = target_top_k
        self.reminder_reduce_noshow = reminder_reduce_noshow
        self.model_race_equalized = model_race_equalized

        self._model = ControlledMLModel(
            mode="discrimination",
            target_auc=model_auc,
        )
        self._model_fitted = False

    def create_population(self, n_entities: int) -> np.ndarray:
        """Create appointment population with race-stratified no-show risk."""
        rng = self.rng.population

        state = np.zeros((n_entities, 5))

        # Assign race
        race_codes = list(RACE_DIST.keys())
        race_probs = list(RACE_DIST.values())
        races = rng.choice(race_codes, n_entities, p=race_probs)
        state[:, COL_RACE] = races.astype(float)

        # Assign base no-show risk from race-stratified beta distributions
        for race_code in race_codes:
            mask = races == race_code
            n_race = mask.sum()
            if n_race == 0:
                continue
            base_rate = RACE_NOSHOW_BASE[race_code]
            # Beta distribution: mean = base_rate, moderate spread
            alpha = 2.0
            beta_p = alpha * (1 / base_rate - 1)
            risks = rng.beta(alpha, beta_p, n_race)
            scaling = base_rate / np.mean(risks)
            state[mask, COL_RISK] = np.clip(risks * scaling, 0.01, 0.95)

        return state

    def step(self, state: np.ndarray, t: int) -> np.ndarray:
        """Appointments refresh each week: new set of patients scheduled.

        IMPORTANT: Reminder effect is temporary (one week only).
        step() restores the base risk for reminded patients by reversing
        the effectiveness reduction applied in intervene().
        This prevents artificial compounding of reminder effects.
        """
        rng = self.rng.temporal
        n = state.shape[0]

        # Restore risk for patients who were reminded (reverse the reduction)
        # Before adding new drift, undo the reminder reduction
        reminded = state[:, COL_REMINDED] == 1.0
        races = state[:, COL_RACE].astype(int)
        new_state = state.copy()
        if reminded.any():
            for race_code in RACE_NOSHOW_BASE:
                race_mask = (races == race_code) & reminded
                if race_mask.sum() == 0:
                    continue
                eff = RACE_REMIND_EFFECTIVENESS[race_code]
                # Reverse: risk_restored = risk_reduced / (1 - eff)
                new_state[race_mask, COL_RISK] = np.clip(
                    state[race_mask, COL_RISK] / max(1 - eff, 0.01),
                    0.01, 0.95
                )

        # Now add weekly drift to refreshed population
        drift = rng.normal(0, 0.01, n)
        new_state[:, COL_RISK] = np.clip(
            new_state[:, COL_RISK] + drift, 0.01, 0.95
        )
        # Reset reminder flags for new timestep
        new_state[:, COL_REMINDED] = 0.0
        new_state[:, COL_REMIND_EFF] = 0.0
        new_state[:, COL_SHOWED] = 0.0
        return new_state

    def predict(self, state: np.ndarray, t: int) -> Predictions:
        """ML model (XGBoost/random forest) predicts no-show probability."""
        true_risks = state[:, COL_RISK]
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
        return Predictions(
            scores=scores,
            metadata={"true_risks": true_risks.copy()},
        )

    def intervene(
        self, state: np.ndarray, predictions: Predictions, t: int
    ) -> tuple[np.ndarray, Interventions]:
        """Target top-K% risk patients for phone reminders.

        Phone reminder effectiveness varies by race (Rosen finding).
        """
        scores = predictions.scores
        n = len(scores)

        # Top-K% targeting
        k = int(n * self.target_top_k)
        if k == 0:
            return state, Interventions(treated_indices=np.array([]))

        threshold_score = np.partition(scores, n - k)[n - k]
        targeted = scores >= threshold_score

        new_state = state.copy()
        new_state[targeted, COL_REMINDED] = 1.0

        # Reminder reduces no-show probability (race-stratified)
        races = state[:, COL_RACE].astype(int)
        for race_code in RACE_NOSHOW_BASE:
            race_mask = (races == race_code) & targeted
            if race_mask.sum() == 0:
                continue
            effectiveness = RACE_REMIND_EFFECTIVENESS[race_code]
            new_state[race_mask, COL_RISK] = np.clip(
                state[race_mask, COL_RISK] * (1 - effectiveness),
                0.01, 0.95
            )
            new_state[race_mask, COL_REMIND_EFF] = effectiveness

        treated_indices = np.where(targeted)[0]
        return new_state, Interventions(
            treated_indices=treated_indices,
            metadata={
                "n_targeted": int(targeted.sum()),
                "pct_targeted": float(targeted.mean()),
                "threshold_score": float(threshold_score),
            },
        )

    def measure(self, state: np.ndarray, t: int) -> Outcomes:
        """Sample show/no-show outcomes. Event = 1 means no-show."""
        rng = self.rng.outcomes
        n = state.shape[0]
        risks = state[:, COL_RISK]
        races = state[:, COL_RACE].astype(int)

        # Sample no-show outcomes
        noshows = (rng.random(n) < risks).astype(float)

        # Build race string array for equity analysis
        race_labels = np.array([RACE_NAMES[r] for r in races])

        # Compute per-race no-show rates
        race_noshow_rates = {}
        for code, name in RACE_NAMES.items():
            mask = races == code
            if mask.sum() > 0:
                race_noshow_rates[name] = float(noshows[mask].mean())

        return Outcomes(
            events=noshows,
            entity_ids=np.arange(n),
            secondary={
                "true_risk": risks.copy(),
                "reminded": state[:, COL_REMINDED].copy(),
                "race_ethnicity": race_labels,
            },
            metadata={
                "noshow_rate": float(noshows.mean()),
                "race_noshow_rates": race_noshow_rates,
                "n_reminded": int(state[:, COL_REMINDED].sum()),
                "timestep": t,
            },
        )

    def clone_state(self, state: np.ndarray) -> np.ndarray:
        return state.copy()
