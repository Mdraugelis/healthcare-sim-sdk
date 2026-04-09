"""Nurse Retention Scenario — Predict turnover risk, intervene via managers.

Simulates a population of nurses with time-varying turnover risk.
An ML model flags high-risk nurses for targeted manager check-ins.
Manager capacity is finite: each manager can conduct a limited number
of retention interventions per week. The branched counterfactual
engine tracks what would have happened without the AI system.

Unit of analysis: nurse
State: NurseRetentionState dataclass with per-nurse numpy arrays

Key domain assumptions (from Laudio/AONL April 2025):
- Early-tenure nurses (<52 weeks) have 2x baseline turnover risk
- Targeted check-ins reduce turnover risk by an effectiveness factor
- Intervention effect decays with a configurable half-life
- When managers have >90 reports, turnover spikes (~40% vs ~27%)
- The strongest retention driver is manager capacity, not model quality

RNG DISCIPLINE CHECKLIST:
- create_population() -> self.rng.population
- step()              -> self.rng.temporal
- predict()           -> self.rng.prediction
- intervene()         -> self.rng.intervention
- measure()           -> self.rng.outcomes

STEP PURITY CHECKLIST:
- step() uses ONLY (state, t, self.rng.temporal)
- step() does NOT read/write self.* mutable attributes
- step() does NOT use self.rng.intervention or self.rng.prediction
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from healthcare_sim_sdk.ml.model import ControlledMLModel
from healthcare_sim_sdk.population.risk_distributions import (
    beta_distributed_risks,
)
from healthcare_sim_sdk.population.temporal_dynamics import (
    annual_risk_to_hazard,
    hazard_to_timestep_probability,
)


@dataclass
class RetentionConfig:
    """Configuration for the nurse retention scenario."""

    # Population
    n_nurses: int = 1000
    nurses_per_manager: int = 100
    annual_turnover_rate: float = 0.22
    risk_concentration: float = 0.5
    new_hire_fraction: float = 0.15
    new_hire_risk_multiplier: float = 2.0
    high_span_turnover_penalty: float = 0.10

    # Temporal dynamics
    n_weeks: int = 52
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.04
    prediction_interval: int = 2  # predict every 2 weeks

    # Model
    model_auc: float = 0.80
    # Percentile threshold: flag nurses at or above this
    # percentile of scores. 70 = flag top 30% (fewer, higher risk).
    # 30 = flag top 70% (more, lower precision).
    risk_threshold_percentile: float = 70.0

    # Intervention (manager check-in)
    max_interventions_per_manager_per_week: int = 4
    intervention_effectiveness: float = 0.50
    intervention_decay_halflife_weeks: float = 6.0
    cooldown_weeks: int = 4


@dataclass
class NurseRetentionState:
    """Complete state for the nurse retention scenario.

    All per-nurse arrays have length n_nurses. Departed nurses
    remain in the arrays (departed=True) but are excluded from
    prediction, intervention, and outcome measurement.
    """

    n_nurses: int
    n_managers: int

    # Per-nurse arrays (length n_nurses)
    base_risk: np.ndarray        # Annual turnover probability
    ar1_modifier: np.ndarray     # Temporal risk modifier (AR(1))
    tenure_weeks: np.ndarray     # Weeks employed at simulation start + t
    is_new_hire: np.ndarray      # Boolean: tenure < 52 weeks
    manager_id: np.ndarray       # Integer manager assignment
    current_risk: np.ndarray     # Effective annual turnover probability

    # AI-directed intervention tracking (factual branch only)
    intervention_effect: np.ndarray       # Current effect [0, 1]
    weeks_since_intervention: np.ndarray  # Weeks since AI check-in

    # Standard-of-care intervention tracking (both branches)
    soc_intervention_effect: np.ndarray   # Current SOC effect [0, 1]
    soc_weeks_since_intervention: np.ndarray  # Weeks since SOC

    # Absorbing state
    departed: np.ndarray         # Boolean: nurse has left
    departed_this_step: np.ndarray  # Boolean: departed THIS step

    # Cumulative counters (for metadata)
    total_departures: int = 0
    total_interventions: int = 0
    total_soc_interventions: int = 0


class NurseRetentionScenario(BaseScenario[NurseRetentionState]):
    """Nurse retention with capacity-constrained manager intervention.

    The model predicts turnover risk scores. Nurses above the risk
    threshold are flagged for manager check-ins. Each manager can
    conduct at most K check-ins per week, prioritizing the highest
    risk scores. Check-ins reduce a nurse's turnover probability
    by an effectiveness factor that decays over time.

    The key experimental question: given a model of quality X (AUC),
    what risk threshold and manager capacity combination minimizes
    preventable departures?
    """

    unit_of_analysis = "nurse"

    def __init__(
        self,
        config: Optional[RetentionConfig] = None,
        seed: Optional[int] = None,
    ):
        self.config = config or RetentionConfig()
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
            mode="discrimination",
            target_auc=c.model_auc,
        )
        self._classifier_fitted = False

        # Precompute decay factor per week from half-life
        self._decay_per_week = 0.5 ** (1.0 / c.intervention_decay_halflife_weeks)

    # ── Method 1: create_population ──────────────────────────

    def create_population(
        self, n_entities: int,
    ) -> NurseRetentionState:
        """Create nurse population with heterogeneous risk profiles.

        Assigns nurses to managers, sets tenure (mix of new hires
        and established staff), and computes initial risk.
        """
        c = self.config
        rng = self.rng.population
        n = n_entities

        # Base turnover risk (beta-distributed, right-skewed)
        base_risk = beta_distributed_risks(
            n_patients=n,
            annual_incident_rate=c.annual_turnover_rate,
            concentration=c.risk_concentration,
            rng=rng,
        )

        # Tenure: new hires get 0-51 weeks, established get 52-260
        is_new_hire = (
            rng.random(n) < c.new_hire_fraction
        ).astype(bool)
        tenure_weeks = np.where(
            is_new_hire,
            rng.integers(0, 52, n),
            rng.integers(52, 260, n),
        ).astype(float)

        # New hires get elevated base risk
        base_risk[is_new_hire] *= c.new_hire_risk_multiplier
        base_risk = np.clip(base_risk, 0, 0.95)

        # Assign to managers (round-robin)
        n_managers = max(1, n // c.nurses_per_manager)
        manager_id = np.arange(n) % n_managers

        # High span-of-control penalty: if nurses_per_manager > 90,
        # add a baseline risk penalty (Laudio finding)
        if c.nurses_per_manager > 90:
            base_risk += c.high_span_turnover_penalty
            base_risk = np.clip(base_risk, 0, 0.95)

        state = NurseRetentionState(
            n_nurses=n,
            n_managers=n_managers,
            base_risk=base_risk,
            ar1_modifier=np.ones(n),
            tenure_weeks=tenure_weeks,
            is_new_hire=is_new_hire,
            manager_id=manager_id,
            current_risk=base_risk.copy(),
            intervention_effect=np.zeros(n),
            weeks_since_intervention=np.full(n, 999.0),
            soc_intervention_effect=np.zeros(n),
            soc_weeks_since_intervention=np.full(n, 999.0),
            departed=np.zeros(n, dtype=bool),
            departed_this_step=np.zeros(n, dtype=bool),
            total_departures=0,
            total_interventions=0,
            total_soc_interventions=0,
        )
        return state

    # ── Method 2: step (PURE) ────────────────────────────────

    def step(
        self, state: NurseRetentionState, t: int,
    ) -> NurseRetentionState:
        """Advance nurse risk by one week.

        PURITY: uses only (state, t, self.rng.temporal).
        Reads self.config and self._decay_per_week (immutable).

        Evolution:
        1. AR(1) risk drift (job satisfaction fluctuates)
        2. Tenure advances (new hires may cross 52-week mark)
        3. Decay both intervention effects (AI and SOC)
        4. Standard-of-care allocation (both branches)
        5. Recompute current risk using max(AI, SOC) effect
        6. Realize departures (absorbing)
        """
        c = self.config
        rng = self.rng.temporal
        new = copy.deepcopy(state)
        n = new.n_nurses
        active = ~new.departed

        # Clear per-step departure flag
        new.departed_this_step = np.zeros(n, dtype=bool)

        # 1. AR(1) temporal drift
        noise = rng.normal(0, c.ar1_sigma, n)
        new_mods = (
            c.ar1_rho * new.ar1_modifier
            + (1 - c.ar1_rho) * 1.0
            + noise
        )
        new.ar1_modifier = np.clip(new_mods, 0.5, 2.0)

        # 2. Advance tenure, update new-hire status
        new.tenure_weeks[active] += 1
        new.is_new_hire = new.tenure_weeks < 52

        # 3. Decay both intervention effects
        new.intervention_effect *= self._decay_per_week
        new.weeks_since_intervention[active] += 1
        new.soc_intervention_effect *= self._decay_per_week
        new.soc_weeks_since_intervention[active] += 1

        # 4. Standard-of-care allocation (runs on BOTH branches)
        #    New hires first, then random fill, K per manager.
        #    Uses rng (temporal) for random selection.
        k = c.max_interventions_per_manager_per_week
        soc_count = 0
        for mgr_id in range(new.n_managers):
            team = np.where(
                active
                & (new.manager_id == mgr_id)
                & (new.soc_weeks_since_intervention
                   >= c.cooldown_weeks)
            )[0]
            if len(team) == 0:
                # Consume fixed RNG to keep streams aligned
                rng.random(1)
                continue

            # New hires first
            new_hires = team[new.is_new_hire[team]]
            others = team[~new.is_new_hire[team]]

            # Shuffle both groups (fixed RNG consumption)
            if len(new_hires) > 0:
                new_hires = rng.permutation(new_hires)
            if len(others) > 0:
                others = rng.permutation(others)

            # Pick up to K: new hires first, then others
            selected = np.concatenate(
                [new_hires, others],
            )[:k]

            if len(selected) > 0:
                new.soc_intervention_effect[selected] = (
                    c.intervention_effectiveness
                )
                new.soc_weeks_since_intervention[selected] = 0
                soc_count += len(selected)

        new.total_soc_interventions += soc_count

        # 5. Recompute current risk: best of AI and SOC effects
        best_effect = np.maximum(
            new.intervention_effect,
            new.soc_intervention_effect,
        )
        effective = (
            new.base_risk * new.ar1_modifier * (1 - best_effect)
        )
        new.current_risk = np.clip(effective, 0, 0.99)

        # 6. Realize departures (weekly probability from annual)
        hazards = annual_risk_to_hazard(new.current_risk)
        weekly_probs = hazard_to_timestep_probability(
            hazards, 1 / 52,
        )
        weekly_probs[new.departed] = 0.0

        departures = rng.random(n) < weekly_probs
        new.departed |= departures
        new.departed_this_step = departures
        new.total_departures += int(departures.sum())

        return new

    # ── Method 3: predict ────────────────────────────────────

    def predict(
        self, state: NurseRetentionState, t: int,
    ) -> Predictions:
        """Score active nurses for turnover risk."""
        active = ~state.departed
        n = state.n_nurses

        # True risk signal the model sees
        risk_signal = state.current_risk

        # Generate true labels for this prediction window
        # (who would actually leave in the next prediction_interval?)
        hazards = annual_risk_to_hazard(risk_signal)
        window_probs = hazard_to_timestep_probability(
            hazards,
            self.config.prediction_interval / 52,
        )
        true_labels = (
            self.rng.prediction.random(n) < window_probs
        ).astype(int)
        # Departed nurses: no prediction needed
        true_labels[~active] = 0

        # Fit model on first call
        if not self._classifier_fitted:
            active_mask = active & (risk_signal > 0)
            self._classifier.fit(
                true_labels[active_mask],
                risk_signal[active_mask],
                self.rng.prediction,
                n_iterations=3,
            )
            self._classifier_fitted = True

        # Score everyone (departed get score=0)
        scores = np.zeros(n)
        if active.sum() > 0:
            scores[active] = self._classifier.predict(
                risk_signal[active],
                self.rng.prediction,
                true_labels[active],
            )

        # Percentile-based threshold on active nurse scores
        active_scores = scores[active]
        if len(active_scores) > 0:
            threshold = np.percentile(
                active_scores,
                self.config.risk_threshold_percentile,
            )
        else:
            threshold = 1.0
        labels = (scores >= threshold).astype(int)
        labels[~active] = 0

        return Predictions(
            scores=scores,
            labels=labels,
            metadata={
                "true_labels": true_labels,
                "n_active": int(active.sum()),
                "n_flagged": int(labels.sum()),
                "threshold_value": float(threshold),
            },
        )

    # ── Method 4: intervene ──────────────────────────────────

    def intervene(
        self, state: NurseRetentionState, predictions: Predictions, t: int,
    ) -> tuple[NurseRetentionState, Interventions]:
        """Managers check in with flagged nurses, capacity-constrained.

        For each manager:
        1. Identify flagged nurses in their team
        2. Exclude recently checked-in (cooldown)
        3. Sort by risk score descending
        4. Take top K (capacity limit)
        5. Apply intervention effect
        """
        c = self.config
        new = copy.deepcopy(state)

        flagged = predictions.labels == 1
        active = ~new.departed
        eligible = (
            flagged
            & active
            & (new.weeks_since_intervention >= c.cooldown_weeks)
        )

        all_treated = []
        intervention_count = 0

        for mgr_id in range(new.n_managers):
            # Nurses on this manager's team who are eligible
            team_eligible = eligible & (new.manager_id == mgr_id)
            if not team_eligible.any():
                continue

            indices = np.where(team_eligible)[0]
            # Sort by risk score descending, take top K
            scores = predictions.scores[indices]
            sorted_order = np.argsort(-scores)
            selected = indices[
                sorted_order[:c.max_interventions_per_manager_per_week]
            ]

            # Apply intervention
            new.intervention_effect[selected] = c.intervention_effectiveness
            new.weeks_since_intervention[selected] = 0
            all_treated.extend(selected.tolist())
            intervention_count += len(selected)

        new.total_interventions += intervention_count

        # Recompute current risk post-intervention: best of AI, SOC
        best_effect = np.maximum(
            new.intervention_effect,
            new.soc_intervention_effect,
        )
        effective = (
            new.base_risk * new.ar1_modifier * (1 - best_effect)
        )
        new.current_risk = np.clip(effective, 0, 0.99)

        treated_indices = np.array(all_treated, dtype=int)
        return new, Interventions(
            treated_indices=treated_indices,
            metadata={
                "n_treated": intervention_count,
                "n_flagged": int(flagged.sum()),
                "n_eligible": int(eligible.sum()),
                "effectiveness": c.intervention_effectiveness,
            },
        )

    # ── Method 5: measure ────────────────────────────────────

    def measure(
        self, state: NurseRetentionState, t: int,
    ) -> Outcomes:
        """Record departures and retention metrics this week."""
        n = state.n_nurses
        active = ~state.departed

        # Per-step departure events (not cumulative)
        events = state.departed_this_step.astype(float)

        # Fraction still employed
        retention_rate = float(active.sum()) / n

        # Risk distribution among active nurses
        if active.any():
            mean_risk = float(state.current_risk[active].mean())
            median_risk = float(
                np.median(state.current_risk[active]),
            )
            new_hire_active = int(
                (state.is_new_hire & active).sum()
            )
        else:
            mean_risk = 0.0
            median_risk = 0.0
            new_hire_active = 0

        return Outcomes(
            events=events,
            entity_ids=np.arange(n),
            secondary={
                "active_mask": active.astype(float),
                "current_risk": state.current_risk.copy(),
            },
            metadata={
                "t": t,
                "total_departures": state.total_departures,
                "total_interventions": state.total_interventions,
                "total_soc_interventions": (
                    state.total_soc_interventions
                ),
                "retention_rate": retention_rate,
                "n_active": int(active.sum()),
                "n_new_hire_active": new_hire_active,
                "mean_risk": mean_risk,
                "median_risk": median_risk,
            },
        )

    # ── Hooks ────────────────────────────────────────────────

    def clone_state(
        self, state: NurseRetentionState,
    ) -> NurseRetentionState:
        """Deep copy for counterfactual branching."""
        return copy.deepcopy(state)

    def validate_population(
        self, state: NurseRetentionState,
    ) -> Dict[str, Any]:
        """Domain-specific population validation."""
        c = self.config
        n = state.n_nurses
        return {
            "population_created": True,
            "n_nurses": n,
            "n_managers": state.n_managers,
            "nurses_per_manager": n / max(state.n_managers, 1),
            "mean_base_risk": float(state.base_risk.mean()),
            "new_hire_fraction": float(state.is_new_hire.mean()),
            "no_departed_at_start": not state.departed.any(),
        }

    def validate_results(self, results) -> Dict[str, Any]:
        """Post-simulation validation."""
        final_t = self.config.n_weeks - 1

        # Get final outcomes for both branches
        factual_meta = results.outcomes[final_t].metadata
        cf_meta = results.counterfactual_outcomes[final_t].metadata

        factual_departures = factual_meta["total_departures"]
        cf_departures = cf_meta["total_departures"]

        return {
            "results_valid": True,
            "factual_departures": factual_departures,
            "counterfactual_departures": cf_departures,
            "departures_prevented": cf_departures - factual_departures,
            "factual_retention_rate": factual_meta["retention_rate"],
            "counterfactual_retention_rate": cf_meta["retention_rate"],
        }
