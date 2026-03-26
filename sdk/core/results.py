"""Simulation results storage.

Stores predictions, interventions, and outcomes indexed by timestep.
The AnalysisDataset export methods will be implemented in NB07.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .scenario import TimeConfig, Predictions, Interventions, Outcomes


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
    counterfactual_outcomes: Dict[int, Outcomes] = field(
        default_factory=dict
    )

    validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_predictions(self, t: int, predictions: Predictions) -> None:
        self.predictions[t] = predictions

    def record_interventions(
        self, t: int, interventions: Interventions
    ) -> None:
        self.interventions[t] = interventions

    def record_outcomes(
        self, t: int, outcomes: Outcomes, counterfactual: bool = False
    ) -> None:
        if counterfactual:
            self.counterfactual_outcomes[t] = outcomes
        else:
            self.outcomes[t] = outcomes

    def record_validation(
        self, phase: str, results: Dict[str, Any]
    ) -> None:
        self.validations[phase] = results

    def get_outcome_series(
        self, outcome_key: str = "events"
    ) -> np.ndarray:
        """Aggregated time series, shape (n_timesteps,)."""
        n_t = self.time_config.n_timesteps if self.time_config else 0
        series = np.zeros(n_t)
        for t in range(n_t):
            if t in self.outcomes:
                events = self.outcomes[t].events
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
                events = self.counterfactual_outcomes[t].events
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
