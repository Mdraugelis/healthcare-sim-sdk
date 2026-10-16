"""Simulation results storage and analysis-ready data export.

Stores predictions, interventions, and outcomes indexed by timestep.
AnalysisDataset provides 4 export methods for different causal
analysis approaches (ITS, DiD, RDD, equity).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .scenario import TimeConfig, Predictions, Interventions, Outcomes


def outcome_values(outcome: Outcomes, key: str) -> np.ndarray:
    """Resolve a named per-entity outcome; never silently substitute events."""
    if key == "events":
        return outcome.events
    if key not in outcome.secondary:
        raise KeyError(
            f"Unknown outcome {key!r}; available: events, "
            + ", ".join(outcome.secondary)
        )
    values = np.asarray(outcome.secondary[key])
    if values.shape != outcome.events.shape:
        raise ValueError(f"Outcome {key!r} must have the same shape as events")
    return values


@dataclass
class SimulationResults:
    """Standardized results container with unit-of-analysis awareness.

    Stores predictions, interventions, and outcomes indexed by timestep.
    Knows the unit_of_analysis declared by the scenario, enabling
    structured export to AnalysisDataset.
    """

    n_entities: int = 0
    time_config: Optional[TimeConfig] = None
    unit_of_analysis: str = "entity"
    counterfactual_mode: str = "branched"

    predictions: Dict[int, Predictions] = field(default_factory=dict)
    interventions: Dict[int, Interventions] = field(default_factory=dict)
    outcomes: Dict[int, Outcomes] = field(default_factory=dict)
    counterfactual_outcomes: Dict[int, Outcomes] = field(default_factory=dict)

    validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_predictions(self, t: int, predictions: Predictions) -> None:
        self.predictions[t] = predictions

    def record_interventions(self, t: int, interventions: Interventions) -> None:
        self.interventions[t] = interventions

    def record_outcomes(
        self, t: int, outcomes: Outcomes, counterfactual: bool = False
    ) -> None:
        if counterfactual:
            self.counterfactual_outcomes[t] = outcomes
        else:
            self.outcomes[t] = outcomes

    def record_validation(self, phase: str, results: Dict[str, Any]) -> None:
        self.validations[phase] = results

    def get_outcome_series(self, outcome_key: str = "events") -> np.ndarray:
        """Aggregated time series, shape (n_timesteps,)."""
        n_t = self.time_config.n_timesteps if self.time_config else 0
        series = np.zeros(n_t)
        for t in range(n_t):
            if t in self.outcomes:
                events = outcome_values(self.outcomes[t], outcome_key)
                series[t] = events.sum()
        return series

    def get_counterfactual_outcome_series(
        self, outcome_key: str = "events"
    ) -> np.ndarray:
        """Aggregated counterfactual time series, shape (n_timesteps,)."""
        n_t = self.time_config.n_timesteps if self.time_config else 0
        series = np.zeros(n_t)
        for t in range(n_t):
            if t in self.counterfactual_outcomes:
                events = outcome_values(self.counterfactual_outcomes[t], outcome_key)
                series[t] = events.sum()
        return series

    def get_treatment_indicator(self) -> np.ndarray:
        """Binary treatment indicator per timestep, shape (n_timesteps,)."""
        n_t = self.time_config.n_timesteps if self.time_config else 0
        indicator = np.zeros(n_t)
        for t in self.interventions:
            if t < n_t:
                indicator[t] = 1
        return indicator

    def to_analysis(self) -> "AnalysisDataset":
        """Convert to an AnalysisDataset for structured causal analysis."""
        return AnalysisDataset(results=self)


class AnalysisDataset:
    """Analysis-ready data export with unit-of-analysis awareness.

    Provides 4 export methods for different causal analysis approaches:
    - to_time_series(): Aggregated series for ITS
    - to_panel(): Entity-level panel for DiD
    - to_entity_snapshots(): Cross-section for RDD
    - to_subgroup_panel(): Panel with group labels for equity analysis
    """

    def __init__(self, results: SimulationResults):
        self.results = results

    @property
    def unit_of_analysis(self) -> str:
        return self.results.unit_of_analysis

    def _get_outcomes_dict(
        self,
        branch: str = "factual",
    ) -> Dict[int, Outcomes]:
        if branch == "counterfactual":
            return self.results.counterfactual_outcomes
        if branch != "factual":
            raise ValueError(f"Unknown branch: {branch!r}")
        return self.results.outcomes

    def to_time_series(
        self,
        outcome_key: str = "events",
        branch: str = "factual",
    ) -> Dict[str, np.ndarray]:
        """Export aggregated time series for ITS analysis.

        Returns:
            Dict with:
            - timesteps: array of timestep indices
            - outcomes: aggregated outcome per timestep
            - treatment_indicator: 1 if intervention occurred at timestep
        """
        n_t = self.results.time_config.n_timesteps
        outcomes_dict = self._get_outcomes_dict(branch)

        timesteps = np.arange(n_t)
        outcomes = np.zeros(n_t)
        for t in range(n_t):
            if t in outcomes_dict:
                events = outcome_values(outcomes_dict[t], outcome_key)
                outcomes[t] = events.sum()

        return {
            "timesteps": timesteps,
            "outcomes": outcomes,
            "treatment_indicator": self.results.get_treatment_indicator(),
        }

    def to_panel(
        self,
        outcome_key: str = "events",
        branch: str = "factual",
        treatment_definition: str = "since_first",
    ) -> Dict[str, Any]:
        """Export entity-level panel data for DiD analysis.

        treatment_definition='since_first' marks each entity from its own
        first assignment. 'ever' marks that entity at every observed time.
        Treatment is the factual assignment even on counterfactual exports.

        Returns:
            Dict with:
            - entity_ids: array of entity IDs (repeated per timestep)
            - timesteps: array of timestep indices (repeated per entity)
            - outcomes: outcome per entity per timestep
            - treated: binary treated indicator per entity per timestep
            - unit_of_analysis: string label

        Raises:
            ValueError: If entity_ids not set on Outcomes.
        """
        n_t = self.results.time_config.n_timesteps
        outcomes_dict = self._get_outcomes_dict(branch)

        # Validate entity_ids exist
        sample_t = next(iter(outcomes_dict), None)
        if sample_t is not None:
            sample = outcomes_dict[sample_t]
            if sample.entity_ids is None:
                raise ValueError(
                    "entity_ids not set on Outcomes. "
                    "Set entity_ids in measure() for panel export."
                )

        if treatment_definition not in ("since_first", "ever"):
            raise ValueError("treatment_definition must be 'since_first' or 'ever'")
        # Interventions name positions, not IDs. Resolve against the factual
        # population at the assignment time (or explicit IDs for lagged outcomes).
        first_treatment = {}
        for t_int, intv in self.results.interventions.items():
            if "treated_entity_ids" in intv.metadata:
                ids = np.asarray(intv.metadata["treated_entity_ids"])
            else:
                out = self.results.outcomes[t_int]
                if out.entity_ids is None:
                    raise ValueError(
                        "entity_ids required to resolve treatment positions"
                    )
                ids = out.entity_ids[intv.treated_indices]
            for eid in ids:
                first_treatment[eid] = min(t_int, first_treatment.get(eid, t_int))

        all_entity_ids: List[np.ndarray] = []
        all_timesteps: List[np.ndarray] = []
        all_outcomes: List[np.ndarray] = []
        all_treated: List[np.ndarray] = []

        for t in range(n_t):
            if t not in outcomes_dict:
                continue
            out = outcomes_dict[t]
            n_e = len(out.events)
            eids = out.entity_ids if out.entity_ids is not None else np.arange(n_e)
            all_entity_ids.append(eids)
            all_timesteps.append(np.full(n_e, t))
            all_outcomes.append(outcome_values(out, outcome_key))
            treated_arr = np.array(
                [
                    int(
                        eid in first_treatment
                        and (
                            treatment_definition == "ever" or t >= first_treatment[eid]
                        )
                    )
                    for eid in eids
                ]
            )
            all_treated.append(treated_arr)

        return {
            "entity_ids": np.concatenate(all_entity_ids)
            if all_entity_ids
            else np.array([]),
            "timesteps": np.concatenate(all_timesteps)
            if all_timesteps
            else np.array([], dtype=int),
            "outcomes": np.concatenate(all_outcomes) if all_outcomes else np.array([]),
            "treated": np.concatenate(all_treated)
            if all_treated
            else np.array([], dtype=int),
            "unit_of_analysis": self.unit_of_analysis,
        }

    def to_entity_snapshots(
        self,
        t: int,
        outcome_key: str = "events",
        branch: str = "factual",
    ) -> Dict[str, Any]:
        """Export cross-sectional data at a single timestep for RDD.

        Returns:
            Dict with:
            - entity_ids: array of entity IDs
            - outcomes: outcome per entity at timestep t
            - scores: prediction scores if available at t (else None)
        """
        outcomes_dict = self._get_outcomes_dict(branch)
        if t not in outcomes_dict:
            raise ValueError(f"No outcomes at timestep {t}")

        out = outcomes_dict[t]
        eids = (
            out.entity_ids if out.entity_ids is not None else np.arange(len(out.events))
        )

        scores = None
        if t in self.results.predictions:
            scores = self.results.predictions[t].scores

        return {
            "entity_ids": eids,
            "outcomes": outcome_values(out, outcome_key),
            "scores": scores,
        }

    def to_subgroup_panel(
        self,
        outcome_key: str = "events",
        subgroup_key: str = "subgroup",
    ) -> Dict[str, Any]:
        """Export panel data with subgroup labels for equity analysis.

        Extends to_panel() with group labels extracted from outcome
        metadata or secondary outcomes.
        """
        panel = self.to_panel(outcome_key=outcome_key, branch="factual")

        # Extract subgroup labels from outcomes metadata/secondary
        n_t = self.results.time_config.n_timesteps
        all_groups: List[np.ndarray] = []
        for t in range(n_t):
            if t not in self.results.outcomes:
                continue
            out = self.results.outcomes[t]
            n_e = len(out.events)
            if subgroup_key in out.secondary:
                all_groups.append(out.secondary[subgroup_key])
            elif subgroup_key in out.metadata:
                groups = out.metadata[subgroup_key]
                if isinstance(groups, np.ndarray):
                    all_groups.append(groups)
                else:
                    all_groups.append(np.full(n_e, groups))
            else:
                all_groups.append(np.full(n_e, "unknown"))

        panel["subgroup"] = np.concatenate(all_groups) if all_groups else np.array([])
        return panel
