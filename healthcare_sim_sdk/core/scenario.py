"""Base scenario interface and data classes for simulation.

Defines the 5-method contract that scenario teams implement, along with
the data classes used to pass information between the engine and scenarios.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

import numpy as np

from .rng import RNGPartitioner, RNGStreams

S = TypeVar("S")


@dataclass
class TimeConfig:
    """Time configuration for the simulation."""
    n_timesteps: int
    timestep_duration: float  # Fraction of a year (e.g., 1/52 for weekly)
    timestep_unit: str = "week"
    prediction_schedule: List[int] = field(default_factory=list)


@dataclass
class Predictions:
    """Container for ML model predictions at a single time point."""
    scores: np.ndarray
    labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Interventions:
    """Container for intervention assignments at a single time point."""
    treated_indices: np.ndarray
    intervention_type: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcomes:
    """Container for outcomes at a single time point."""
    events: np.ndarray
    entity_ids: Optional[np.ndarray] = None
    secondary: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseScenario(ABC, Generic[S]):
    """Base class for all simulation scenarios.

    Teams inherit from this and implement five methods to define their
    scenario. Everything else -- the time loop, counterfactual generation,
    seed management, results storage -- is handled by the SDK engine.

    Type parameter S is the scenario-specific state representation.
    It can be anything: a numpy array, a dataclass, a dictionary,
    a pandas DataFrame. The engine doesn't inspect it.

    RNG DISCIPLINE:
    Scenarios receive partitioned RNG streams (self.rng) with named
    generators for each process type. Rules:

    1. Use self.rng.temporal in step() for temporal evolution
    2. Use self.rng.prediction in predict() for model noise
    3. Use self.rng.intervention in intervene() for randomized assignment
    4. Use self.rng.outcomes in measure() for outcome generation
    5. Use self.rng.population in create_population() for initial state

    STEP PURITY CONTRACT:
    step() must be a pure function of (state, t, self.rng.temporal).
    It must not read or modify any external mutable state.
    """

    unit_of_analysis: str = "entity"

    def __init__(
        self, time_config: TimeConfig, seed: Optional[int] = None
    ):
        self.time_config = time_config
        self.seed = seed
        self._partitioner = RNGPartitioner(
            seed if seed is not None else 42
        )
        self.rng: RNGStreams = self._partitioner.create_streams()

    # -- The Five Methods --------------------------------------------------

    @abstractmethod
    def create_population(self, n_entities: int) -> S:
        """Create the initial population state.

        RNG: Use self.rng.population for all random draws.
        """
        ...

    @abstractmethod
    def step(self, state: S, t: int) -> S:
        """Advance state by one timestep. This is your domain physics.

        PURITY CONTRACT: Must be a pure function of
        (state, t, self.rng.temporal). No external side effects.

        RNG: Use self.rng.temporal for all random draws.
        """
        ...

    @abstractmethod
    def predict(self, state: S, t: int) -> Predictions:
        """Simulate what the ML model sees and outputs.

        Called on the FACTUAL branch only, at prediction_schedule times.

        RNG: Use self.rng.prediction for all random draws.
        """
        ...

    @abstractmethod
    def intervene(
        self, state: S, predictions: Predictions, t: int
    ) -> tuple[S, Interventions]:
        """Apply the policy. Modify state and return both.

        Called on the FACTUAL branch only.

        RNG: Use self.rng.intervention for all random draws.
        """
        ...

    @abstractmethod
    def measure(self, state: S, t: int) -> Outcomes:
        """Measure what happened this timestep.

        Called on BOTH factual and counterfactual branches.
        Set entity_ids on Outcomes for panel data construction.

        RNG: Use self.rng.outcomes for all random draws.
        """
        ...

    # -- Optional Hooks ----------------------------------------------------

    def validate_population(self, state: S) -> Dict[str, bool]:
        """Optional: Validate generated population."""
        return {"population_created": True}

    def validate_results(
        self, results: Any  # SimulationResults (forward ref)
    ) -> Dict[str, Any]:
        """Optional: Validate simulation results."""
        return {"results_valid": True}

    def clone_state(self, state: S) -> S:
        """Create a deep copy of state for counterfactual branching.

        Default uses copy.deepcopy(). Override for optimized copying
        (e.g., np.copy for array-backed state).
        """
        return copy.deepcopy(state)
