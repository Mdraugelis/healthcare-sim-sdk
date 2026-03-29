# Healthcare Intervention Simulation SDK: Design Document

## Document Purpose

This document specifies the architecture for refactoring `pop-ml-simulator` from a
monolithic healthcare AI simulation framework into an **Intervention Simulation SDK**
— a toolkit that provides the invariants of discrete-time healthcare intervention
simulation while leaving the variants to teams building bespoke scenarios.

**Scope:** Phases 1 and 2 prove a *discrete-time scenario SDK with pluggable domain
logic*. The SDK targets healthcare AI deployment evaluation — testing intervention
policies, measuring causal effects, and comparing analytic methods against known
ground truth. Event-driven simulation (continuous-time, event queues) is a documented
extension point, not a Phase 1 commitment.

Two scenarios are developed end-to-end to validate the design:

1. **Stroke Prevention** — refactoring the existing codebase
2. **No-Show Overbooking** — a fundamentally different problem structure drawn from
   a real Epic Cognitive Computing deployment evaluation

### Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial design |
| 2.0 | Mar 2026 | Branched counterfactual engine, structured results with unit-of-analysis metadata, RNG stream partitioning, scoped claims |

---

## Part 1: The Problem with Configuration-Driven Generality

The current `pop-ml-simulator` is excellent for its specific scenario: individual
patients with beta-distributed risk scores evolving through AR(1) temporal dynamics,
a binary classifier with controlled PPV/sensitivity, and a multiplicative risk
reduction intervention.

But extending it to cover materially different healthcare scenarios through
configuration inevitably produces what we might call "Turing-complete YAML" — config
files that encode logic, branching, and domain physics in a format with no debugger,
no type checker, and no stack traces.

Consider the distance between these two scenarios:

**Stroke Prevention:**
- Entity: Individual patient
- State: Scalar annual risk
- Time grain: Weekly/Monthly
- ML task: Binary classification (will this patient have a stroke in 12 months?)
- Intervention: Prescribe anticoagulants → multiplicative risk reduction
- Outcome: Binary event (stroke or no stroke)
- Counterfactual: What if we hadn't treated?

**No-Show Overbooking:**
- Entity: Appointment slot (patient × provider × time)
- State: Per-appointment no-show probability + clinic schedule state + individual
  patient overbooking history
- Time grain: Daily (appointment-level)
- ML task: Probability estimation (what's the chance this patient won't show?)
- Intervention: Double-book a slot → the intervened patient may face longer waits
- Outcomes: Multiple competing — did original patient show? Did overbooked patient
  show? What was the wait time? How many times has this patient been overbooked?
- Counterfactual: What if we hadn't overbooked? What if we'd overbooked differently?

No configuration schema can meaningfully parameterize both. The atoms are different.

---

## Part 2: Invariants and Variants

### The Invariants (What the SDK Provides)

These are the things that should not be reinvented for each scenario:

1. **The Discrete-Time Engine** — managing timestep iteration, advancing the
   simulation clock, calling scenario hooks in the right order. (Event-driven
   engines are a future extension point.)
2. **The ML Model Interface** — a standardized way to plug in a predictive model
   simulation and have it produce scores at each prediction point, with utilities
   for controlling model performance (AUC, PPV, sensitivity, calibration)
3. **The Branched Counterfactual Engine** — running true parallel state trajectories
   (factual and counterfactual) across the full simulation, with the factual branch
   receiving interventions and the counterfactual branch evolving naturally.
   Mathematically valid via RNG stream partitioning.
4. **State Management** — carrying population/entity data through the simulation
   without memory leaks, supporting efficient storage of trajectories
5. **Reproducibility Infrastructure** — RNG stream partitioning by process type,
   configuration tracking, deterministic execution across branches
6. **Analysis-Ready Data Interface** — structured results with unit-of-analysis
   metadata, exportable as entity-timestep panels, aggregated time series, or
   entity-level snapshots depending on the analytic method's requirements
7. **Experiment Management** — Hydra integration for parameter sweeps, experiment
   tracking, configuration composition

### The Variants (What Scenario Teams Implement)

These are the things that change between scenarios:

1. **Entity model** — what you're simulating (patients, appointments, units, panels)
2. **State representation** — what data describes each entity at each moment
3. **Domain physics** — how state evolves absent intervention
4. **ML model behavior** — what the model sees, predicts, and how performance is
   characterized
5. **Intervention mechanics** — what the policy does to state and who it affects
6. **Resource constraints** — what limits intervention capacity
7. **Outcome measurement** — what you're trying to optimize or understand

---

## Part 3: SDK Architecture

### Directory Structure

```
healthcare-intervention-sim-sdk/
├── sdk/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py              # BranchedSimulationEngine
│   │   ├── scenario.py            # BaseScenario (the main class teams inherit)
│   │   ├── results.py             # SimulationResults, AnalysisDataset
│   │   └── rng.py                 # RNGPartitioner
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── binary_classifier.py   # ControlledBinaryClassifier (from existing)
│   │   ├── probability_model.py   # ControlledProbabilityModel (calibrated scores)
│   │   ├── regression_model.py    # ControlledRegressionModel
│   │   └── performance.py         # Hosmer-Lemeshow, calibration, AUC utilities
│   ├── population/
│   │   ├── __init__.py
│   │   ├── risk_distributions.py  # Beta, mixture, empirical distributions
│   │   └── temporal_dynamics.py   # AR(1), seasonal, shock processes
│   ├── analysis/                  # Renamed from causal/ — honest about scope
│   │   ├── __init__.py
│   │   ├── its.py                 # Interrupted Time Series (works on aggregated series)
│   │   ├── panel_utils.py         # Panel data construction for DiD, RDD
│   │   └── visualization.py       # Comparison plots, equity dashboards
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base_config.py         # BaseScenarioConfig dataclass
│   │   └── hydra_utils.py         # Hydra integration helpers
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Logging infrastructure
│       ├── sparse.py              # Sparse matrix utilities
│       ├── vectorization.py       # Vectorized operation helpers
│       └── visualization.py       # Base plotting utilities
│
├── scenarios/
│   ├── stroke_prevention/         # Scenario 1 (refactored existing code)
│   │   ├── __init__.py
│   │   ├── scenario.py            # StrokePreventionScenario(BaseScenario)
│   │   ├── config.yaml
│   │   ├── README.md
│   │   └── notebooks/
│   │
│   ├── noshow_overbooking/        # Scenario 2 (new)
│   │   ├── __init__.py
│   │   ├── scenario.py            # NoShowOverbookingScenario(BaseScenario)
│   │   ├── config.yaml
│   │   ├── README.md
│   │   └── notebooks/
│   │
│   └── _template/                 # Starter template
│       ├── __init__.py
│       ├── scenario.py            # MyScenario(BaseScenario) with stubs
│       ├── config.yaml
│       └── README.md
│
├── experiments/                   # Experiment runners
├── tests/
├── docs/
├── requirements.txt
├── setup.py
└── README.md
```

### Core Interfaces

#### `RNGPartitioner` — Reproducibility Across Branches

```python
"""sdk/core/rng.py"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RNGStreams:
    """Named RNG streams for different simulation processes.

    Each stream is an independent np.random.Generator, deterministically
    derived from a master seed. This guarantees:
    1. Different processes (population gen, temporal evolution, etc.)
       don't interfere with each other across runs.
    2. The factual and counterfactual branches of a branched simulation
       produce identical stochastic evolution for processes they share
       (e.g., temporal dynamics) because they use the same stream state.
    3. Intervention randomness lives on its own stream, so the
       counterfactual branch (which skips intervention) doesn't
       advance the shared streams out of sync.
    """
    population: np.random.Generator
    temporal: np.random.Generator
    prediction: np.random.Generator
    intervention: np.random.Generator
    outcomes: np.random.Generator


class RNGPartitioner:
    """
    Creates deterministic, independent RNG streams from a master seed.

    Uses numpy's SeedSequence to spawn child streams that are
    statistically independent but fully reproducible.

    Usage:
        partitioner = RNGPartitioner(master_seed=42)
        streams = partitioner.create_streams()
        # streams.population, streams.temporal, etc.
    """

    STREAM_NAMES = ["population", "temporal", "prediction",
                    "intervention", "outcomes"]

    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self._seed_seq = np.random.SeedSequence(master_seed)

    def create_streams(self) -> RNGStreams:
        """Create a fresh set of independent RNG streams."""
        child_seeds = self._seed_seq.spawn(len(self.STREAM_NAMES))
        generators = {
            name: np.random.default_rng(seed)
            for name, seed in zip(self.STREAM_NAMES, child_seeds)
        }
        return RNGStreams(**generators)

    def fork(self) -> 'RNGPartitioner':
        """
        Create a new partitioner for a branched simulation.

        The forked partitioner produces streams with the same
        statistical properties but from a fresh SeedSequence spawn,
        ensuring the counterfactual branch doesn't share mutable
        generator state with the factual branch.

        CRITICAL: This is used by BranchedSimulationEngine to ensure
        the counterfactual branch's temporal/population/outcome streams
        start from the same seed state as the factual branch at the
        point of divergence.
        """
        child_seq = self._seed_seq.spawn(1)[0]
        new_partitioner = RNGPartitioner.__new__(RNGPartitioner)
        new_partitioner.master_seed = self.master_seed
        new_partitioner._seed_seq = child_seq
        return new_partitioner
```

#### `BaseScenario` — The Contract Teams Implement

```python
"""sdk/core/scenario.py"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypeVar, Generic
import numpy as np

from .rng import RNGPartitioner, RNGStreams

# Type variable for scenario-specific state
S = TypeVar('S')  # State type


@dataclass
class TimeConfig:
    """Time configuration for the simulation."""
    n_timesteps: int
    timestep_duration: float        # Fraction of a year (e.g., 1/52 for weekly)
    timestep_unit: str = "week"     # Human-readable label
    prediction_schedule: List[int] = field(default_factory=list)


@dataclass
class Predictions:
    """Container for ML model predictions at a single time point."""
    scores: np.ndarray              # Raw prediction scores
    labels: Optional[np.ndarray] = None  # Binary/categorical predictions
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Interventions:
    """Container for intervention assignments at a single time point."""
    treated_indices: np.ndarray     # Which entities receive intervention
    intervention_type: Optional[np.ndarray] = None  # Type of intervention
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcomes:
    """Container for outcomes at a single time point."""
    events: np.ndarray              # Primary outcome (scenario-defined)
    entity_ids: Optional[np.ndarray] = None  # Which entities these outcomes belong to
    secondary: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseScenario(ABC, Generic[S]):
    """
    Base class for all simulation scenarios.

    Teams inherit from this and implement five methods to define their
    scenario. Everything else — the time loop, counterfactual generation,
    seed management, results storage — is handled by the SDK engine.

    Type parameter S is the scenario-specific state representation.
    It can be anything: a numpy array, a dataclass, a dictionary,
    a pandas DataFrame. The engine doesn't inspect it.

    RNG DISCIPLINE:
    Scenarios receive partitioned RNG streams (self.rng) with named
    generators for each process type. This is critical for branched
    counterfactual simulation. Rules:

    1. Use self.rng.temporal in step() for temporal evolution
    2. Use self.rng.prediction in predict() for model noise
    3. Use self.rng.intervention in intervene() for randomized assignment
    4. Use self.rng.outcomes in measure() for outcome generation
    5. Use self.rng.population in create_population() for initial state

    The engine may call step() and measure() on both the factual and
    counterfactual branches. If you use the correct stream for each
    process, the branches will diverge only where intervention changed
    state — not because of RNG desynchronization.

    STEP PURITY CONTRACT:
    step() must be a pure function of (state, t, self.rng.temporal).
    It must not read or modify any external mutable state. The engine
    calls step() on both factual and counterfactual state objects, and
    both calls must produce equivalent evolution for the same inputs.
    Ongoing intervention effects embedded in the state (e.g., a patient
    whose risk was reduced by treatment) are fine — they're part of the
    state, not external side effects.

    UNIT OF ANALYSIS:
    Scenarios must declare their unit_of_analysis — the entity that
    outcomes are measured on. This informs how the results layer
    structures data for causal analysis.
    """

    # Subclasses should set this to declare their statistical unit
    unit_of_analysis: str = "entity"  # "patient", "appointment", "provider_day", etc.

    def __init__(self, time_config: TimeConfig, seed: Optional[int] = None):
        self.time_config = time_config
        self.seed = seed
        self._partitioner = RNGPartitioner(seed if seed is not None else 42)
        self.rng: RNGStreams = self._partitioner.create_streams()

    # ── The Five Methods ────────────────────────────────────────────

    @abstractmethod
    def create_population(self, n_entities: int) -> S:
        """
        Create the initial population state.

        This defines your entities and their starting conditions.
        Return whatever state representation your scenario needs.

        RNG: Use self.rng.population for all random draws.

        Examples:
        - Stroke: np.ndarray of scalar risk scores
        - No-show: DataFrame of appointment slots with patient IDs,
          times, departments, and base no-show probabilities
        - ED flow: Dict with bed_states, queue, staff arrays
        """
        ...

    @abstractmethod
    def step(self, state: S, t: int) -> S:
        """
        Advance state by one timestep. This is your domain physics.

        PURITY CONTRACT: This method must be a pure function of
        (state, t, self.rng.temporal). It must not read or modify any
        external mutable state beyond what is passed in. The engine
        calls step() on BOTH factual and counterfactual branches.

        Called every timestep. Encodes how the world evolves absent
        any new intervention. Existing interventions that are already
        embedded in the state (e.g., a patient currently on medication,
        a slot that was previously overbooked) are fine — they're part
        of the state object the engine passes in.

        RNG: Use self.rng.temporal for all random draws.

        Examples:
        - Stroke: AR(1) temporal risk modifier + seasonal effects
        - No-show: Appointment resolution, patient history updates,
          new schedule generation
        - ED flow: Patient arrivals (Poisson), triage, bed transitions
        """
        ...

    @abstractmethod
    def predict(self, state: S, t: int) -> Predictions:
        """
        Simulate what the ML model sees and outputs.

        Called at each time in the prediction_schedule — on the
        FACTUAL branch only. The counterfactual branch does not
        receive predictions (see Counterfactual Prediction Policy
        in Part 3).

        RNG: Use self.rng.prediction for all random draws.

        The SDK provides utility classes (ControlledBinaryClassifier,
        ControlledProbabilityModel) that handle noise injection
        and performance targeting — you just wire them up to
        your state.

        Examples:
        - Stroke: Binary classifier on integrated risk window
        - No-show: Probability model on appointment features
        - ED flow: Regression model predicting volume
        """
        ...

    @abstractmethod
    def intervene(self, state: S, predictions: Predictions,
                  t: int) -> tuple[S, Interventions]:
        """
        Apply your policy. Modify state. Return both.

        Called on the FACTUAL branch only. The counterfactual branch
        never receives interventions.

        This is where the causal effect gets injected. The SDK calls
        this with the current state and predictions; you decide who
        gets treated and what treatment does to their state.

        RNG: Use self.rng.intervention for all random draws (e.g.,
        randomized assignment, stochastic treatment effects).

        Examples:
        - Stroke: Patients above threshold get risk × (1 - effectiveness)
        - No-show: High no-show slots get double-booked; overbooked
          patients accumulate overbooking history
        - ED flow: Predicted surge triggers staff callback
        """
        ...

    @abstractmethod
    def measure(self, state: S, t: int) -> Outcomes:
        """
        Measure what happened this timestep.

        Called every timestep on BOTH factual and counterfactual
        branches. Returns per-entity outcomes with entity_ids so
        the results layer can construct panel data.

        RNG: Use self.rng.outcomes for all random draws (e.g.,
        stochastic outcome observation, measurement noise).

        Primary outcome goes in events; anything else in secondary.
        IMPORTANT: Set entity_ids on the Outcomes object so the
        results layer can build entity-level panel data.

        Examples:
        - Stroke: Binary incident (stroke or not), entity_ids = patient_ids
        - No-show: Did patient show? Was slot overbooked? Wait time?
          entity_ids = slot_ids or patient_ids depending on analysis
        - ED flow: Wait times, LWBS rate, utilization
        """
        ...

    # ── Optional Hooks ──────────────────────────────────────────────

    def validate_population(self, state: S) -> Dict[str, bool]:
        """
        Optional: Validate that generated population meets expectations.
        Override to add scenario-specific checks.
        """
        return {"population_created": True}

    def validate_results(self, results: 'SimulationResults') -> Dict[str, Any]:
        """
        Optional: Validate simulation results.
        Override to add scenario-specific validation.
        """
        return {"results_valid": True}

    def clone_state(self, state: S) -> S:
        """
        Optional: Create a deep copy of state for counterfactual branching.

        The default implementation uses copy.deepcopy(), which is correct
        but potentially slow for large or complex state objects. Override
        this to provide optimized copying (e.g., copying only arrays that
        mutate, shallow copying immutable sub-structures).

        This is the .clone() / .snapshot() optimization path.
        """
        import copy
        return copy.deepcopy(state)
```

#### `BranchedSimulationEngine` — True Counterfactual Propagation

```python
"""sdk/core/engine.py"""

from typing import Optional
from enum import Enum
import logging

from .scenario import BaseScenario, TimeConfig
from .results import SimulationResults
from .rng import RNGPartitioner

logger = logging.getLogger(__name__)


class CounterfactualMode(Enum):
    """How the engine handles counterfactual generation."""
    NONE = "none"                    # No counterfactuals
    SNAPSHOT = "snapshot"            # Same-step comparison only (v1 behavior)
    BRANCHED = "branched"            # Full parallel trajectory propagation


class BranchedSimulationEngine:
    """
    The simulation engine with true counterfactual propagation.

    This engine maintains two parallel state trajectories:
    - FACTUAL: step → predict → intervene → measure (the policy world)
    - COUNTERFACTUAL: step → measure (the no-intervention world)

    Both branches call step() with the same temporal RNG stream state
    at each timestep, ensuring they experience identical stochastic
    evolution (same disease progression, same appointment arrivals).
    They diverge ONLY where intervention modified the factual state.

    COUNTERFACTUAL PREDICTION POLICY:
    The counterfactual branch does NOT call predict(). Rationale:
    in most healthcare AI evaluation designs, the question is "what
    would have happened if we hadn't deployed the AI at all?" — not
    "what if the AI predicted but we ignored it?" If a scenario needs
    the latter (intention-to-treat analysis), it should implement that
    logic inside its own step() method by embedding prediction as part
    of the domain physics rather than the policy layer.

    This is a deliberate design choice. The engine's predict() →
    intervene() pipeline represents the causal pathway we want to
    evaluate. The counterfactual branch removes that entire pathway.

    RNG SYNCHRONIZATION:
    The engine creates separate RNG streams for each branch using
    RNGPartitioner.fork(). Both branches get streams derived from
    the same master seed, so temporal evolution and outcome generation
    are statistically equivalent between branches before intervention
    effects propagate through state.

    CRITICAL: For this to work correctly, scenario step() must obey
    the purity contract — it must be a pure function of (state, t,
    self.rng.temporal). If step() reads external mutable state or uses
    non-partitioned random sources, the branches will desynchronize
    for the wrong reasons.

    MEMORY:
    Branched mode approximately doubles memory and CPU. Acceptable for
    medium-scale runs (N ≤ 100k, T ≤ 260). For larger simulations,
    scenario authors should implement clone_state() for optimized
    copying, or use SNAPSHOT mode for same-step comparisons only.
    """

    def __init__(self, scenario: BaseScenario,
                 counterfactual_mode: CounterfactualMode =
                     CounterfactualMode.BRANCHED):
        self.scenario = scenario
        self.counterfactual_mode = counterfactual_mode
        self.results: Optional[SimulationResults] = None

    def run(self, n_entities: int) -> SimulationResults:
        """
        Execute the full simulation pipeline.

        BRANCHED mode (default):
        1. Create population → clone into factual + counterfactual state
        2. For each timestep:
           a. Step BOTH branches (same temporal physics)
           b. Factual: predict → intervene (counterfactual: skip)
           c. Measure BOTH branches
        3. Return results with full parallel trajectories

        SNAPSHOT mode (v1 compatible):
        1. Create population
        2. For each timestep:
           a. Step state
           b. If prediction time: snapshot → predict → intervene
           c. Measure factual; measure snapshot (same-step only)
        3. Return results

        NONE mode:
        1. Single trajectory, no counterfactuals
        """
        if self.counterfactual_mode == CounterfactualMode.BRANCHED:
            return self._run_branched(n_entities)
        elif self.counterfactual_mode == CounterfactualMode.SNAPSHOT:
            return self._run_snapshot(n_entities)
        else:
            return self._run_simple(n_entities)

    def _run_branched(self, n_entities: int) -> SimulationResults:
        """Full parallel trajectory counterfactual simulation."""
        sc = self.scenario
        tc = sc.time_config
        results = SimulationResults(
            n_entities=n_entities,
            time_config=tc,
            unit_of_analysis=sc.unit_of_analysis,
            counterfactual_mode="branched"
        )

        # 1. Create shared initial population
        state_factual = sc.create_population(n_entities)
        validation = sc.validate_population(state_factual)
        results.record_validation("population", validation)

        # Clone for counterfactual branch
        state_counterfactual = sc.clone_state(state_factual)

        # Create separate RNG streams for counterfactual branch.
        # We save and restore the scenario's RNG streams so each
        # branch gets streams from the same starting state.
        factual_rng = sc.rng
        cf_partitioner = sc._partitioner.fork()
        cf_rng = cf_partitioner.create_streams()

        logger.info(f"Population created: {n_entities} entities, "
                    f"mode=branched, validation={validation}")

        # 2. Temporal loop — both branches advance every timestep
        for t in range(tc.n_timesteps):

            # 2a. Evolve FACTUAL state
            sc.rng = factual_rng
            state_factual = sc.step(state_factual, t)

            # 2b. Evolve COUNTERFACTUAL state (same physics, no intervention)
            sc.rng = cf_rng
            state_counterfactual = sc.step(state_counterfactual, t)

            # 2c. Predict and intervene on FACTUAL branch only
            if t in tc.prediction_schedule:
                sc.rng = factual_rng
                predictions = sc.predict(state_factual, t)
                results.record_predictions(t, predictions)

                state_factual, interventions = sc.intervene(
                    state_factual, predictions, t
                )
                results.record_interventions(t, interventions)

            # 2d. Measure BOTH branches
            sc.rng = factual_rng
            outcomes_factual = sc.measure(state_factual, t)
            results.record_outcomes(t, outcomes_factual, counterfactual=False)

            sc.rng = cf_rng
            outcomes_cf = sc.measure(state_counterfactual, t)
            results.record_outcomes(t, outcomes_cf, counterfactual=True)

        # Restore factual RNG as default
        sc.rng = factual_rng

        # 3. Final validation
        final_validation = sc.validate_results(results)
        results.record_validation("final", final_validation)
        logger.info(f"Simulation complete: {tc.n_timesteps} timesteps, "
                    f"mode=branched")

        self.results = results
        return results

    def _run_snapshot(self, n_entities: int) -> SimulationResults:
        """Same-step snapshot counterfactuals (v1 compatible)."""
        sc = self.scenario
        tc = sc.time_config
        results = SimulationResults(
            n_entities=n_entities,
            time_config=tc,
            unit_of_analysis=sc.unit_of_analysis,
            counterfactual_mode="snapshot"
        )

        state = sc.create_population(n_entities)
        validation = sc.validate_population(state)
        results.record_validation("population", validation)

        for t in range(tc.n_timesteps):
            state = sc.step(state, t)

            if t in tc.prediction_schedule:
                predictions = sc.predict(state, t)
                results.record_predictions(t, predictions)

                # Snapshot before intervention
                state_snapshot = sc.clone_state(state)

                state, interventions = sc.intervene(
                    state, predictions, t
                )
                results.record_interventions(t, interventions)

                # Measure counterfactual (same-step only)
                cf_outcomes = sc.measure(state_snapshot, t)
                results.record_outcomes(t, cf_outcomes, counterfactual=True)

            outcomes = sc.measure(state, t)
            results.record_outcomes(t, outcomes, counterfactual=False)

        final_validation = sc.validate_results(results)
        results.record_validation("final", final_validation)
        self.results = results
        return results

    def _run_simple(self, n_entities: int) -> SimulationResults:
        """Single trajectory, no counterfactuals."""
        sc = self.scenario
        tc = sc.time_config
        results = SimulationResults(
            n_entities=n_entities,
            time_config=tc,
            unit_of_analysis=sc.unit_of_analysis,
            counterfactual_mode="none"
        )

        state = sc.create_population(n_entities)
        validation = sc.validate_population(state)
        results.record_validation("population", validation)

        for t in range(tc.n_timesteps):
            state = sc.step(state, t)

            if t in tc.prediction_schedule:
                predictions = sc.predict(state, t)
                results.record_predictions(t, predictions)

                state, interventions = sc.intervene(
                    state, predictions, t
                )
                results.record_interventions(t, interventions)

            outcomes = sc.measure(state, t)
            results.record_outcomes(t, outcomes, counterfactual=False)

        final_validation = sc.validate_results(results)
        results.record_validation("final", final_validation)
        self.results = results
        return results
```

#### `SimulationResults` and `AnalysisDataset` — Structured Results

```python
"""sdk/core/results.py"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .scenario import TimeConfig, Predictions, Interventions, Outcomes


@dataclass
class SimulationResults:
    """
    Standardized results container with unit-of-analysis awareness.

    Stores predictions, interventions, and outcomes indexed by timestep.
    Knows the unit_of_analysis declared by the scenario, enabling
    structured export to AnalysisDataset.
    """
    n_entities: int = 0
    time_config: Optional[TimeConfig] = None
    unit_of_analysis: str = "entity"
    counterfactual_mode: str = "branched"

    # Indexed by timestep
    predictions: Dict[int, Predictions] = field(default_factory=dict)
    interventions: Dict[int, Interventions] = field(default_factory=dict)
    outcomes: Dict[int, Outcomes] = field(default_factory=dict)
    counterfactual_outcomes: Dict[int, Outcomes] = field(default_factory=dict)

    # Validation records
    validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Scenario-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_predictions(self, t: int, predictions: Predictions):
        self.predictions[t] = predictions

    def record_interventions(self, t: int, interventions: Interventions):
        self.interventions[t] = interventions

    def record_outcomes(self, t: int, outcomes: Outcomes,
                        counterfactual: bool = False):
        if counterfactual:
            self.counterfactual_outcomes[t] = outcomes
        else:
            self.outcomes[t] = outcomes

    def record_validation(self, phase: str, results: Dict[str, Any]):
        self.validations[phase] = results

    # ── Export to AnalysisDataset ────────────────────────────────

    def to_analysis(self) -> 'AnalysisDataset':
        """
        Convert to an AnalysisDataset for structured causal analysis.

        Returns an AnalysisDataset that knows the unit_of_analysis and
        can export data in multiple formats depending on what the
        analytic method requires.
        """
        return AnalysisDataset(
            results=self,
            unit_of_analysis=self.unit_of_analysis
        )

    # ── Backward-compatible convenience accessors ────────────────

    def get_outcome_series(self, outcome_key: str = "events"
                           ) -> np.ndarray:
        """
        Extract an aggregated time series of outcomes.
        Shape: (n_timesteps,) — summed across entities per timestep.

        NOTE: This is a convenience method suitable for ITS analysis.
        For entity-level panel data (DiD, RDD), use to_analysis()
        instead.
        """
        series = []
        for t in sorted(self.outcomes.keys()):
            o = self.outcomes[t]
            if outcome_key == "events":
                series.append(np.sum(o.events))
            elif outcome_key in o.secondary:
                series.append(np.sum(o.secondary[outcome_key]))
        return np.array(series)

    def get_treatment_indicator(self) -> np.ndarray:
        """
        Extract binary treatment indicator per timestep.
        Shape: (n_timesteps,) — 1 if any intervention occurred.
        """
        tc = self.time_config
        indicator = np.zeros(tc.n_timesteps if tc else 0)
        for t in self.interventions:
            if len(self.interventions[t].treated_indices) > 0:
                indicator[t] = 1
        return indicator


@dataclass
class AnalysisDataset:
    """
    Analysis-ready data export with unit-of-analysis awareness.

    This is the bridge between SimulationResults (which stores
    everything) and causal inference methods (which need specific
    data shapes). Different methods need different structures:

    - ITS: aggregated time series (timestep → outcome sum)
    - DiD: entity-level panel (entity × timestep → outcome, treatment)
    - RDD: entity-level cross-section (entity → running variable,
      outcome, treatment)
    - Subgroup analysis: entity-level with group membership

    The scenario declares its unit_of_analysis, and these methods
    reshape accordingly.
    """
    results: SimulationResults = None
    unit_of_analysis: str = "entity"

    def to_time_series(self, outcome_key: str = "events",
                       branch: str = "factual") -> Dict[str, np.ndarray]:
        """
        Export as aggregated time series. Suitable for ITS.

        Returns:
            {"timesteps": array, "outcomes": array,
             "treatment_indicator": array}
        """
        source = (self.results.outcomes if branch == "factual"
                  else self.results.counterfactual_outcomes)
        timesteps = sorted(source.keys())
        outcomes = []
        for t in timesteps:
            o = source[t]
            if outcome_key == "events":
                outcomes.append(np.sum(o.events))
            elif outcome_key in o.secondary:
                outcomes.append(np.sum(o.secondary[outcome_key]))
        return {
            "timesteps": np.array(timesteps),
            "outcomes": np.array(outcomes),
            "treatment_indicator": self.results.get_treatment_indicator()
        }

    def to_panel(self, outcome_key: str = "events",
                 branch: str = "factual") -> Dict[str, np.ndarray]:
        """
        Export as entity-level panel data. Suitable for DiD.

        Returns:
            {"entity_ids": array, "timesteps": array,
             "outcomes": array, "treated": array}

        Shape: Each array has length (n_entities × n_timesteps).
        Requires that Outcomes include entity_ids.
        """
        source = (self.results.outcomes if branch == "factual"
                  else self.results.counterfactual_outcomes)
        timesteps = sorted(source.keys())

        all_entity_ids = []
        all_timesteps = []
        all_outcomes = []
        all_treated = []

        for t in timesteps:
            o = source[t]
            if o.entity_ids is None:
                raise ValueError(
                    f"Outcomes at t={t} missing entity_ids. "
                    f"Panel data requires entity-level tracking. "
                    f"Set entity_ids in your measure() method."
                )

            n = len(o.entity_ids)
            all_entity_ids.append(o.entity_ids)
            all_timesteps.append(np.full(n, t))

            if outcome_key == "events":
                all_outcomes.append(o.events)
            elif outcome_key in o.secondary:
                all_outcomes.append(o.secondary[outcome_key])

            # Treatment status per entity at this timestep
            if t in self.results.interventions:
                treated_set = set(
                    self.results.interventions[t].treated_indices
                )
                treated = np.array([
                    1 if eid in treated_set else 0
                    for eid in o.entity_ids
                ])
            else:
                treated = np.zeros(n)
            all_treated.append(treated)

        return {
            "entity_ids": np.concatenate(all_entity_ids),
            "timesteps": np.concatenate(all_timesteps),
            "outcomes": np.concatenate(all_outcomes),
            "treated": np.concatenate(all_treated),
            "unit_of_analysis": self.unit_of_analysis
        }

    def to_entity_snapshots(self, t: int,
                            outcome_key: str = "events",
                            branch: str = "factual"
                            ) -> Dict[str, np.ndarray]:
        """
        Export entity-level cross-section at a single timestep.
        Suitable for RDD analysis.

        Returns:
            {"entity_ids": array, "outcomes": array,
             "scores": array (if predictions available)}
        """
        source = (self.results.outcomes if branch == "factual"
                  else self.results.counterfactual_outcomes)
        if t not in source:
            raise ValueError(f"No outcomes at timestep {t}")

        o = source[t]
        result = {
            "entity_ids": o.entity_ids,
            "outcomes": o.events if outcome_key == "events"
                       else o.secondary.get(outcome_key)
        }

        if t in self.results.predictions:
            result["scores"] = self.results.predictions[t].scores

        return result

    def to_subgroup_panel(self, outcome_key: str = "events",
                          subgroup_key: str = "subgroup"
                          ) -> Dict[str, Any]:
        """
        Export panel data with subgroup membership for equity analysis.
        Extends to_panel() with group labels from outcome metadata.
        """
        panel = self.to_panel(outcome_key)

        # Extract subgroup labels from outcome metadata
        source = self.results.outcomes
        timesteps = sorted(source.keys())
        all_subgroups = []
        for t in timesteps:
            o = source[t]
            if subgroup_key in o.metadata:
                all_subgroups.append(o.metadata[subgroup_key])
            elif subgroup_key in o.secondary:
                all_subgroups.append(o.secondary[subgroup_key])
            else:
                all_subgroups.append(
                    np.full(len(o.entity_ids), "unknown")
                )

        panel["subgroups"] = np.concatenate(all_subgroups)
        return panel
```

### SDK-Provided ML Simulation Utilities

Teams don't need to implement ML noise injection from scratch. The SDK provides
reusable model simulators extracted from the current codebase:

```python
"""sdk/ml/binary_classifier.py — extracted from existing MLPredictionSimulator"""

class ControlledBinaryClassifier:
    """
    Simulates a binary classifier with target PPV and sensitivity.
    Extracted from pop-ml-simulator's MLPredictionSimulator.

    Usage in a scenario's predict() method:
        self.classifier = ControlledBinaryClassifier(
            target_sensitivity=0.75, target_ppv=0.35, seed=42
        )
        scores, labels = self.classifier.predict(true_labels, risk_scores)
    """
    def __init__(self, target_sensitivity, target_ppv, seed=None): ...
    def predict(self, true_labels, risk_scores) -> tuple: ...
    def optimize(self, true_labels, risk_scores, n_iter=20) -> dict: ...


class ControlledProbabilityModel:
    """
    Simulates a probability estimation model with target AUC and calibration.

    Unlike ControlledBinaryClassifier which targets PPV/sensitivity at a
    threshold, this targets overall discrimination (AUC) and calibration
    (Hosmer-Lemeshow, calibration slope).

    Usage:
        self.model = ControlledProbabilityModel(
            target_auc=0.75, target_calibration_slope=1.0, seed=42
        )
        scores = self.model.predict(true_probabilities)
    """
    def __init__(self, target_auc, target_calibration_slope=1.0, seed=None): ...
    def predict(self, true_probabilities) -> np.ndarray: ...
    def calibration_report(self, predictions, actuals) -> dict: ...
```

### SDK-Provided Population Utilities

Reusable building blocks extracted from the current codebase:

```python
"""sdk/population/risk_distributions.py"""

def beta_distributed_risks(n, annual_rate, concentration=0.5, rng=None):
    """Exactly the existing assign_patient_risks function."""
    ...

def mixture_distributed_risks(n, components, weights, rng=None):
    """For scenarios needing multi-modal risk distributions."""
    ...


"""sdk/population/temporal_dynamics.py"""

class AR1Process:
    """Exactly the existing EnhancedTemporalRiskSimulator, extracted."""
    def __init__(self, base_values, rho, sigma, bounds, seasonal=None): ...
    def step(self) -> np.ndarray: ...
    def get_matrix(self) -> np.ndarray: ...
```

---

## Part 4: Scenario 1 — Stroke Prevention

This refactors the existing `pop-ml-simulator` code into the SDK's scenario
interface. The goal is zero behavioral change — identical outputs for identical
inputs — with the domain logic cleanly separated from the engine.

**Realism tier:** This is an *operational-realism* scenario. The domain logic
directly refactors production code that has been validated against known
epidemiological parameters.

### State Representation

```python
@dataclass
class StrokeState:
    """State for stroke prevention scenario."""
    n_patients: int
    patient_ids: np.ndarray           # Shape: (n_patients,)
    base_risks: np.ndarray            # Shape: (n_patients,)
    current_risks: np.ndarray         # Shape: (n_patients,) — after temporal mod
    temporal_modifiers: np.ndarray    # Shape: (n_patients,)
    intervention_active: np.ndarray   # Shape: (n_patients,) — bool
    intervention_end_times: np.ndarray  # Shape: (n_patients,) — int
    incidents_this_step: np.ndarray   # Shape: (n_patients,) — bool
```

### Scenario Implementation

```python
"""scenarios/stroke_prevention/scenario.py"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

from sdk.core.scenario import (
    BaseScenario, TimeConfig, Predictions, Interventions, Outcomes
)
from sdk.population.risk_distributions import beta_distributed_risks
from sdk.population.temporal_dynamics import AR1Process
from sdk.ml.binary_classifier import ControlledBinaryClassifier


class StrokePreventionScenario(BaseScenario['StrokeState']):
    """
    Stroke prevention with AI-guided risk reduction.

    Refactored from the original pop-ml-simulator. All domain logic
    that was in VectorizedTemporalRiskSimulator is now here; all
    infrastructure logic is in the SDK engine.
    """

    unit_of_analysis = "patient"

    def __init__(self, time_config: TimeConfig, seed=None,
                 annual_incident_rate=0.08,
                 intervention_effectiveness=0.25,
                 risk_concentration=0.5,
                 ar1_rho=0.9, ar1_sigma=0.1,
                 temporal_bounds=(0.2, 2.5),
                 seasonal_amplitude=0.2,
                 prediction_threshold=0.5,
                 target_sensitivity=0.75,
                 target_ppv=0.35,
                 prediction_window=12,
                 intervention_duration=-1):
        super().__init__(time_config, seed)

        self.annual_incident_rate = annual_incident_rate
        self.intervention_effectiveness = intervention_effectiveness
        self.risk_concentration = risk_concentration
        self.ar1_rho = ar1_rho
        self.ar1_sigma = ar1_sigma
        self.temporal_bounds = temporal_bounds
        self.seasonal_amplitude = seasonal_amplitude
        self.prediction_threshold = prediction_threshold
        self.prediction_window = prediction_window
        self.intervention_duration = intervention_duration

        # SDK-provided ML simulator — uses self.rng.prediction internally
        self.classifier = ControlledBinaryClassifier(
            target_sensitivity=target_sensitivity,
            target_ppv=target_ppv,
            seed=seed
        )

        self.ar1: Optional[AR1Process] = None

    # ── The Five Methods ────────────────────────────────────────

    def create_population(self, n_entities: int) -> 'StrokeState':
        """Generate patients with beta-distributed risk scores."""
        base_risks = beta_distributed_risks(
            n=n_entities,
            annual_rate=self.annual_incident_rate,
            concentration=self.risk_concentration,
            rng=self.rng.population  # Partitioned RNG
        )

        self.ar1 = AR1Process(
            base_values=base_risks,
            rho=self.ar1_rho,
            sigma=self.ar1_sigma,
            bounds=self.temporal_bounds,
            seasonal_amplitude=self.seasonal_amplitude
        )

        return StrokeState(
            n_patients=n_entities,
            patient_ids=np.arange(n_entities),
            base_risks=base_risks,
            current_risks=base_risks.copy(),
            temporal_modifiers=np.ones(n_entities),
            intervention_active=np.zeros(n_entities, dtype=bool),
            intervention_end_times=np.full(n_entities, -1, dtype=int),
            incidents_this_step=np.zeros(n_entities, dtype=bool)
        )

    def step(self, state: 'StrokeState', t: int) -> 'StrokeState':
        """
        Evolve patient risks through AR(1) temporal dynamics.

        PURITY: Uses only (state, t, self.rng.temporal). The AR(1)
        process is deterministic given its current state + the
        temporal RNG stream.
        """
        # Temporal risk evolution — uses self.rng.temporal
        self.ar1.step(rng=self.rng.temporal)
        state.temporal_modifiers = self.ar1.get_current_modifiers()
        state.current_risks = np.clip(
            state.base_risks * state.temporal_modifiers, 0, 0.99
        )

        # Expire finished interventions
        expired = (state.intervention_end_times >= 0) & \
                  (state.intervention_end_times <= t)
        state.intervention_active[expired] = False

        # Apply intervention effect (risk reduction for active treatments)
        effective_risks = state.current_risks.copy()
        effective_risks[state.intervention_active] *= (
            1 - self.intervention_effectiveness
        )

        # Hazard-based incident generation — uses self.rng.temporal
        dt = self.time_config.timestep_duration
        hazards = -np.log(1 - np.clip(effective_risks, 0, 0.999999))
        timestep_probs = 1 - np.exp(-hazards * dt)
        state.incidents_this_step = (
            self.rng.temporal.random(state.n_patients) < timestep_probs
        )

        return state

    def predict(self, state: 'StrokeState', t: int) -> Predictions:
        """
        Simulate ML binary classifier on current risk state.
        Uses self.rng.prediction for noise injection.
        """
        true_labels = (state.current_risks >
                       np.percentile(state.current_risks, 85)).astype(int)

        scores, labels = self.classifier.predict(
            true_labels, state.current_risks,
            rng=self.rng.prediction
        )
        return Predictions(scores=scores, labels=labels)

    def intervene(self, state: 'StrokeState', predictions: Predictions,
                  t: int) -> tuple['StrokeState', Interventions]:
        """Assign interventions: patients above threshold get treatment."""
        # No need to deepcopy — engine handles branching externally
        eligible = ~state.intervention_active
        above_threshold = predictions.scores >= self.prediction_threshold
        to_treat = eligible & above_threshold
        treated_indices = np.where(to_treat)[0]

        state.intervention_active[treated_indices] = True
        if self.intervention_duration == -1:
            end_time = self.time_config.n_timesteps - 1
        else:
            end_time = t + self.intervention_duration - 1
        state.intervention_end_times[treated_indices] = end_time

        interventions = Interventions(
            treated_indices=treated_indices,
            metadata={"threshold": self.prediction_threshold,
                      "n_treated": len(treated_indices)}
        )
        return state, interventions

    def measure(self, state: 'StrokeState', t: int) -> Outcomes:
        """Binary incident outcome with entity IDs for panel data."""
        return Outcomes(
            events=state.incidents_this_step.copy(),
            entity_ids=state.patient_ids.copy(),
            secondary={
                "intervention_active": state.intervention_active.copy(),
                "current_risks": state.current_risks.copy()
            }
        )
```

### Stroke Scenario Configuration

```yaml
# scenarios/stroke_prevention/config.yaml
scenario:
  _target_: scenarios.stroke_prevention.scenario.StrokePreventionScenario

  time_config:
    n_timesteps: 52
    timestep_duration: 0.01923    # 1/52 (weekly)
    timestep_unit: "week"
    prediction_schedule: [0, 12, 24, 36]

  n_entities: 100000
  annual_incident_rate: 0.08
  risk_concentration: 0.5
  ar1_rho: 0.9
  ar1_sigma: 0.1
  temporal_bounds: [0.2, 2.5]
  seasonal_amplitude: 0.2
  target_sensitivity: 0.75
  target_ppv: 0.35
  prediction_threshold: 0.5
  prediction_window: 12
  intervention_effectiveness: 0.25
  intervention_duration: -1
  seed: 42
```

### Running Stroke with Branched Counterfactuals

```python
from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from scenarios.stroke_prevention.scenario import StrokePreventionScenario
from sdk.core.scenario import TimeConfig

time_config = TimeConfig(
    n_timesteps=52, timestep_duration=1/52, timestep_unit="week",
    prediction_schedule=[0, 12, 24, 36]
)

scenario = StrokePreventionScenario(time_config=time_config, seed=42)

engine = BranchedSimulationEngine(
    scenario, counterfactual_mode=CounterfactualMode.BRANCHED
)
results = engine.run(n_entities=100_000)

# Aggregated ITS-style comparison
factual_incidents = results.get_outcome_series("events")
cf_series = results.to_analysis().to_time_series("events", branch="counterfactual")

print(f"Factual total incidents: {factual_incidents.sum()}")
print(f"Counterfactual total incidents: {cf_series['outcomes'].sum()}")
print(f"Estimated lives saved: "
      f"{cf_series['outcomes'].sum() - factual_incidents.sum()}")

# Entity-level panel for DiD
analysis = results.to_analysis()
panel = analysis.to_panel("events", branch="factual")
print(f"Panel shape: {len(panel['entity_ids'])} rows "
      f"({results.n_entities} entities × {time_config.n_timesteps} timesteps)")
```

---

## Part 5: Scenario 2 — No-Show Overbooking

This scenario is drawn directly from the Epic No-Show Predictor evaluation. It
addresses the core simulation question raised in the evaluation's Section 8:

> If patients with high risk of no-shows have a high chance of getting double-booked,
> how often would high-risk patients get double-booked? Allowing for exploration of
> clinic capacity, model performance, variability of risk score for a patient,
> overbook policy. The balance will be how often the same patients get over-booked vs
> the improved access for patients.
>
> Enable the following analysis:
>
> **Establish the Current Baseline (Retrospective)** 
>
> Clinics currently have access to actual no-show rates and have the option of using that information when deciding whether to overbook. This is the existing "model." Before evaluating the Epic predictor, we need to understand how this current approach performs. 
>
> | Metric                                         | Description                                                  |
> | ---------------------------------------------- | ------------------------------------------------------------ |
> | Accuracy of actual no-show rate as a predictor | How well does a clinic's historical no-show rate predict whether a specific patient will no-show? This gives a baseline "model" performance to compare the Epic predictor against. |
> | Overbooked-patient-show rate                   | When a clinic overbooks a slot, how often does the overbooked patient actually show up? |
> | Collision rate                                 | How often do both the original patient and the overbooked patient show up (the failure case)? |
> | Subgroup breakdown                             | All the above metrics are stratified by targeted subgroups (race/ethnicity, insurance type, campus/geography, age). |
> | Overbooking volume                             | Average number of overbookings per week by clinic.           |
> | Utilization rate                               | Current slot utilization rate by clinic.                     |
> 
>**Evaluate the Prediction Model (Retrospective)** 
>  
>Compare the Epic No-Show Predictor against the current baseline established in Phase 1. 
> 
>| Metric                     | Description                                                  |
> | -------------------------- | ------------------------------------------------------------ |
> | Model discrimination       | C-statistic (AUC) overall and by subgroup. Compare directly against the accuracy of the actual no-show rate as a predictor from Phase 1. |
> | Anticipated collision rate | Given the model's performance characteristics, what collision rate would result at various no-show probability thresholds? Compare against Phase 1 baseline collision rate. |
> | Threshold selection        | Use the model performance data to select the no-show probability threshold that achieves the desired balance of collision rate vs. filled slots. <br /><br />The operational goal is to select the optimal threshold to improve utilization at clinics.  Some clinics are underbooking and have utilization of 80% and some have overbooking issues were they average over 100%.   We want to improve upon baseline and get near 100%. |

**Realism tier:** This is an *architecture-validating* scenario. The domain logic
(waitlist selection, overbooking mechanics) is realistic enough to stress-test
the SDK interfaces and answer the evaluation's core questions, but some operational
details are intentionally simplified. Specifically, `_find_overbook_candidate()`
uses random selection from unscheduled patients rather than a realistic waitlist
priority queue, and `_count_recent_overbooks()` uses cumulative counts rather
than rolling time-window counts. These simplifications do not affect the
architecture validation — the interfaces would be identical with more
sophisticated implementations.

### Key Differences from Stroke Scenario

| Dimension | Stroke | No-Show Overbooking |
|-----------|--------|-------------------|
| Entity | Patient (persistent) | Appointment slot (transient, but patient persists across slots) |
| State | Scalar risk per patient | Schedule grid + patient overbooking history |
| Intervention target | The patient at risk | A *different* patient booked into the slot |
| Intervention effect | Reduces risk (helps target) | May increase wait time (potentially harms target) |
| Outcome | Binary event (stroke) | Multiple: show/no-show, slot utilization, wait time, individual burden |
| Equity dimension | Population-level | Individual-level (same patients overbooked repeatedly) |
| Resource constraint | Implicit (threshold) | Explicit (clinic capacity, provider time) |
| Counterfactual | "What if we hadn't treated?" | "What if we hadn't overbooked?" — compounding across days |
| Unit of analysis | patient | appointment (with patient-level rollup) |

### State Representation

```python
@dataclass
class ClinicConfig:
    """Configuration for a clinic or department."""
    name: str
    n_providers: int
    slots_per_provider_per_day: int
    max_overbooking_per_provider: int
    appointment_duration_minutes: int
    buffer_minutes: int


@dataclass
class Patient:
    """Persistent patient attributes that carry across appointments."""
    patient_id: int
    base_noshow_probability: float
    n_past_appointments: int
    n_past_noshows: int
    n_times_overbooked: int
    n_times_overbooked_and_showed: int
    subgroup: str


@dataclass
class AppointmentSlot:
    """A single appointment slot in the schedule."""
    slot_id: int
    day: int
    provider_id: int
    patient_id: int
    predicted_noshow_prob: float
    true_noshow_prob: float
    is_overbooked: bool
    overbooked_patient_id: Optional[int]
    original_patient_showed: bool
    overbooked_patient_showed: bool
    wait_time_minutes: float


@dataclass
class NoShowState:
    """Complete state for no-show overbooking scenario."""
    day: int
    patients: Dict[int, Patient]
    schedule: List[AppointmentSlot]
    clinic_config: ClinicConfig
    overbooking_budget_remaining: Dict[int, int]
    total_slots: int
    total_overbooked_slots: int
    total_noshows: int
    total_both_showed: int
    patient_overbooking_history: Dict[int, int]
```

### Scenario Implementation

```python
"""scenarios/noshow_overbooking/scenario.py"""

import numpy as np
from typing import Dict, List, Optional
from copy import deepcopy

from sdk.core.scenario import (
    BaseScenario, TimeConfig, Predictions, Interventions, Outcomes
)
from sdk.ml.probability_model import ControlledProbabilityModel


class NoShowOverbookingScenario(BaseScenario['NoShowState']):
    """
    No-show prediction with overbooking policy simulation.

    Uses partitioned RNG streams and obeys the step() purity contract
    for correct branched counterfactual execution.
    """

    unit_of_analysis = "appointment"

    def __init__(self, time_config: TimeConfig, seed=None,
                 n_patients=5000,
                 appointments_per_day=80,
                 base_noshow_rate=0.12,
                 noshow_concentration=0.3,
                 model_auc=0.75,
                 model_calibration_slope=1.0,
                 overbooking_threshold=0.30,
                 max_individual_overbooks=3,
                 individual_cap_window_days=90,
                 n_providers=8,
                 slots_per_provider=12,
                 max_overbook_per_provider=2,
                 appointment_duration_minutes=20,
                 buffer_minutes=10,
                 noshow_variability_sigma=0.05):
        super().__init__(time_config, seed)

        self.n_patients = n_patients
        self.appointments_per_day = appointments_per_day
        self.base_noshow_rate = base_noshow_rate
        self.noshow_concentration = noshow_concentration
        self.overbooking_threshold = overbooking_threshold
        self.max_individual_overbooks = max_individual_overbooks
        self.individual_cap_window_days = individual_cap_window_days
        self.noshow_variability_sigma = noshow_variability_sigma
        self.n_providers = n_providers
        self.slots_per_provider = slots_per_provider
        self.max_overbook_per_provider = max_overbook_per_provider
        self.appointment_duration_minutes = appointment_duration_minutes
        self.buffer_minutes = buffer_minutes

        self.noshow_model = ControlledProbabilityModel(
            target_auc=model_auc,
            target_calibration_slope=model_calibration_slope,
            seed=seed
        )

        self.clinic_config = ClinicConfig(
            name="Primary Care",
            n_providers=n_providers,
            slots_per_provider_per_day=slots_per_provider,
            max_overbooking_per_provider=max_overbook_per_provider,
            appointment_duration_minutes=appointment_duration_minutes,
            buffer_minutes=buffer_minutes
        )

    # ── The Five Methods ────────────────────────────────────────

    def create_population(self, n_entities: int) -> 'NoShowState':
        """
        Create patient panel and initial schedule.
        Uses self.rng.population for all random draws.
        """
        rng = self.rng.population

        alpha = self.noshow_concentration
        beta_param = alpha * (1 / self.base_noshow_rate - 1)
        raw_probs = rng.beta(alpha, beta_param, self.n_patients)
        scaling = self.base_noshow_rate / np.mean(raw_probs)
        base_probs = np.clip(raw_probs * scaling, 0.01, 0.80)

        subgroup_assignments = rng.choice(
            ["group_A", "group_B", "group_C", "group_D"],
            size=self.n_patients,
            p=[0.55, 0.20, 0.15, 0.10]
        )

        subgroup_multipliers = {
            "group_A": 0.85, "group_B": 1.10,
            "group_C": 1.30, "group_D": 1.50,
        }
        for i, sg in enumerate(subgroup_assignments):
            base_probs[i] *= subgroup_multipliers[sg]
        base_probs = np.clip(base_probs, 0.01, 0.80)

        patients = {}
        for i in range(self.n_patients):
            patients[i] = Patient(
                patient_id=i,
                base_noshow_probability=base_probs[i],
                n_past_appointments=int(rng.poisson(6)),
                n_past_noshows=0,
                n_times_overbooked=0,
                n_times_overbooked_and_showed=0,
                subgroup=subgroup_assignments[i]
            )

        for p in patients.values():
            if p.n_past_appointments > 0:
                p.n_past_noshows = int(rng.binomial(
                    p.n_past_appointments, p.base_noshow_probability
                ))

        schedule = self._generate_daily_schedule(
            patients, day=0, rng=self.rng.temporal
        )

        return NoShowState(
            day=0, patients=patients, schedule=schedule,
            clinic_config=self.clinic_config,
            overbooking_budget_remaining={
                pid: self.max_overbook_per_provider
                for pid in range(self.n_providers)
            },
            total_slots=0, total_overbooked_slots=0,
            total_noshows=0, total_both_showed=0,
            patient_overbooking_history={}
        )

    def step(self, state: 'NoShowState', t: int) -> 'NoShowState':
        """
        Advance by one day. Resolves appointments, updates histories,
        generates tomorrow's schedule.

        PURITY: Uses only (state, t, self.rng.temporal). All random
        draws for appointment resolution, schedule generation, and
        no-show probability variability come from self.rng.temporal.
        This ensures the factual and counterfactual branches see
        identical "natural" dynamics.
        """
        rng = self.rng.temporal

        # Resolve previous day's appointments
        for slot in state.schedule:
            true_prob = slot.true_noshow_prob
            slot.original_patient_showed = (rng.random() >= true_prob)

            if slot.is_overbooked and slot.overbooked_patient_id is not None:
                ob_patient = state.patients[slot.overbooked_patient_id]
                ob_true_prob = ob_patient.base_noshow_probability
                ob_true_prob += rng.normal(0, self.noshow_variability_sigma)
                ob_true_prob = np.clip(ob_true_prob, 0.01, 0.80)
                slot.overbooked_patient_showed = (rng.random() >= ob_true_prob)

                if (slot.original_patient_showed and
                        slot.overbooked_patient_showed):
                    slot.wait_time_minutes = (
                        self.appointment_duration_minutes +
                        self.buffer_minutes
                    )
                    state.total_both_showed += 1
                else:
                    slot.wait_time_minutes = 0.0

            patient = state.patients[slot.patient_id]
            patient.n_past_appointments += 1
            if not slot.original_patient_showed:
                patient.n_past_noshows += 1
                state.total_noshows += 1
            state.total_slots += 1

            if slot.is_overbooked and slot.overbooked_patient_id is not None:
                ob_patient = state.patients[slot.overbooked_patient_id]
                ob_patient.n_times_overbooked += 1
                if slot.overbooked_patient_showed:
                    ob_patient.n_times_overbooked_and_showed += 1
                state.total_overbooked_slots += 1
                state.patient_overbooking_history[
                    slot.overbooked_patient_id
                ] = ob_patient.n_times_overbooked

        # Generate new schedule
        state.day = t
        state.schedule = self._generate_daily_schedule(
            state.patients, t, rng=rng
        )

        state.overbooking_budget_remaining = {
            pid: self.max_overbook_per_provider
            for pid in range(self.n_providers)
        }

        return state

    def predict(self, state: 'NoShowState', t: int) -> Predictions:
        """
        Simulate Epic no-show prediction model.
        Uses self.rng.prediction for noise injection.
        Called on FACTUAL branch only.
        """
        true_probs = np.array([
            slot.true_noshow_prob for slot in state.schedule
        ])

        predicted_probs = self.noshow_model.predict(
            true_probs, rng=self.rng.prediction
        )

        for i, slot in enumerate(state.schedule):
            slot.predicted_noshow_prob = predicted_probs[i]

        return Predictions(
            scores=predicted_probs,
            metadata={
                "slot_ids": [s.slot_id for s in state.schedule],
                "patient_ids": [s.patient_id for s in state.schedule],
                "true_probs": true_probs
            }
        )

    def intervene(self, state: 'NoShowState', predictions: Predictions,
                  t: int) -> tuple['NoShowState', Interventions]:
        """
        Overbooking policy with guardrails.
        Uses self.rng.intervention for candidate selection.
        Called on FACTUAL branch only.
        """
        rng = self.rng.intervention
        overbooked_slot_indices = []

        scored_slots = sorted(
            enumerate(state.schedule),
            key=lambda x: x[1].predicted_noshow_prob,
            reverse=True
        )

        for idx, slot in scored_slots:
            if slot.predicted_noshow_prob < self.overbooking_threshold:
                break

            if state.overbooking_budget_remaining[slot.provider_id] <= 0:
                continue

            overbook_patient = self._find_overbook_candidate(
                state, slot, t, rng=rng
            )
            if overbook_patient is None:
                continue

            recent_overbooks = state.patients[
                overbook_patient.patient_id
            ].n_times_overbooked
            if recent_overbooks >= self.max_individual_overbooks:
                continue

            slot.is_overbooked = True
            slot.overbooked_patient_id = overbook_patient.patient_id
            state.overbooking_budget_remaining[slot.provider_id] -= 1
            overbooked_slot_indices.append(idx)

        interventions = Interventions(
            treated_indices=np.array(overbooked_slot_indices),
            metadata={
                "n_overbooked": len(overbooked_slot_indices),
                "overbooking_rate": (
                    len(overbooked_slot_indices) / len(state.schedule)
                    if state.schedule else 0
                ),
                "overbooked_patient_ids": [
                    state.schedule[i].overbooked_patient_id
                    for i in overbooked_slot_indices
                ]
            }
        )
        return state, interventions

    def measure(self, state: 'NoShowState', t: int) -> Outcomes:
        """
        Measure multiple outcomes with entity IDs for panel data.
        Called on BOTH branches.
        """
        slot_ids = np.array([s.slot_id for s in state.schedule])
        noshows = np.array([
            not slot.original_patient_showed for slot in state.schedule
        ])

        utilized = []
        for slot in state.schedule:
            if slot.original_patient_showed:
                utilized.append(1)
            elif slot.is_overbooked and slot.overbooked_patient_showed:
                utilized.append(1)
            else:
                utilized.append(0)

        wait_times = np.array([
            slot.wait_time_minutes for slot in state.schedule
        ])

        overbook_counts = np.array([
            p.n_times_overbooked for p in state.patients.values()
        ])

        # Subgroup data for equity analysis
        patient_subgroups = np.array([
            state.patients[s.patient_id].subgroup for s in state.schedule
        ])

        return Outcomes(
            events=noshows,
            entity_ids=slot_ids,
            secondary={
                "slot_utilization": np.array(utilized),
                "wait_times": wait_times,
                "overbook_counts_per_patient": overbook_counts,
                "noshow_rate": np.mean(noshows) if len(noshows) > 0 else 0,
                "utilization_rate": np.mean(utilized) if utilized else 0,
                "collision_count": state.total_both_showed,
            },
            metadata={
                "subgroup": patient_subgroups,
                "patient_ids": np.array([
                    s.patient_id for s in state.schedule
                ])
            }
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _generate_daily_schedule(self, patients, day, rng):
        """Generate a day's appointment slots. Uses provided RNG."""
        total_slots = self.n_providers * self.slots_per_provider
        patient_ids = rng.choice(
            list(patients.keys()),
            size=min(total_slots, len(patients)),
            replace=False
        )

        schedule = []
        for i, pid in enumerate(patient_ids):
            patient = patients[pid]
            appt_noshow_prob = patient.base_noshow_probability + \
                rng.normal(0, self.noshow_variability_sigma)
            appt_noshow_prob = np.clip(appt_noshow_prob, 0.01, 0.80)

            schedule.append(AppointmentSlot(
                slot_id=day * 1000 + i, day=day,
                provider_id=i % self.n_providers,
                patient_id=pid,
                predicted_noshow_prob=0.0,
                true_noshow_prob=appt_noshow_prob,
                is_overbooked=False,
                overbooked_patient_id=None,
                original_patient_showed=False,
                overbooked_patient_showed=False,
                wait_time_minutes=0.0
            ))
        return schedule

    def _find_overbook_candidate(self, state, slot, t, rng):
        """Find a patient to overbook. Uses provided RNG."""
        scheduled_today = {s.patient_id for s in state.schedule}
        candidates = [
            p for pid, p in state.patients.items()
            if pid not in scheduled_today
        ]
        if not candidates:
            return None
        return rng.choice(candidates)
```

### No-Show Configuration

```yaml
# scenarios/noshow_overbooking/config.yaml
scenario:
  _target_: scenarios.noshow_overbooking.scenario.NoShowOverbookingScenario

  time_config:
    n_timesteps: 90
    timestep_duration: 0.00274    # 1/365
    timestep_unit: "day"
    prediction_schedule: null     # Every day (set programmatically)

  n_patients: 5000
  base_noshow_rate: 0.12
  noshow_concentration: 0.3
  noshow_variability_sigma: 0.05

  n_providers: 8
  slots_per_provider: 12
  max_overbook_per_provider: 2
  appointment_duration_minutes: 20
  buffer_minutes: 10

  model_auc: 0.75
  model_calibration_slope: 1.0

  overbooking_threshold: 0.30
  max_individual_overbooks: 3
  individual_cap_window_days: 90

  seed: 42

# Experiment sweeps
sweeps:
  model_performance:
    model_auc: [0.58, 0.65, 0.70, 0.75, 0.80, 0.85]
  threshold_exploration:
    overbooking_threshold: [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
  individual_cap:
    max_individual_overbooks: [1, 2, 3, 5, 10, 999]
  capacity:
    max_overbook_per_provider: [1, 2, 3, 4]
    slots_per_provider: [8, 10, 12, 15]
```

### Running No-Show with Branched Counterfactuals

```python
from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from scenarios.noshow_overbooking.scenario import NoShowOverbookingScenario
from sdk.core.scenario import TimeConfig

time_config = TimeConfig(
    n_timesteps=90, timestep_duration=1/365, timestep_unit="day",
    prediction_schedule=list(range(90))
)

scenario = NoShowOverbookingScenario(time_config=time_config, seed=42)

engine = BranchedSimulationEngine(
    scenario, counterfactual_mode=CounterfactualMode.BRANCHED
)
results = engine.run(n_entities=5000)

# ── True longitudinal counterfactual analysis ──
# The counterfactual branch evolved for 90 days WITHOUT any overbooking.
# Patient overbooking histories in the CF branch are all zero.
# This answers: "What would have happened to these specific patients
# if we had never deployed the overbooking policy?"

analysis = results.to_analysis()

# Factual vs counterfactual utilization over time
factual_ts = analysis.to_time_series("slot_utilization", branch="factual")
cf_ts = analysis.to_time_series("slot_utilization", branch="counterfactual")
print(f"Factual mean utilization: {factual_ts['outcomes'].mean():.3f}")
print(f"Counterfactual mean utilization: {cf_ts['outcomes'].mean():.3f}")

# Entity-level panel for subgroup equity analysis
panel = analysis.to_subgroup_panel("wait_times", subgroup_key="subgroup")
# Can now run DiD or subgroup-stratified comparisons
```

### Connecting to the Evaluation's Section 8 Conditions

| Evaluation Condition | Simulation Parameter Sweep | Output Metric |
|---------------------|---------------------------|---------------|
| "Cap individual patient exposure" | `max_individual_overbooks: [1, 2, 3, 5, 999]` | Burden distribution, max/mean individual overbooks |
| "Monitor wait-time equity" | Subgroup analysis via `to_subgroup_panel()` | Overbooking rate and wait time by subgroup |
| "Complete equity stratification" | `model_auc` sweep × subgroup analysis | Differential burden by subgroup at each AUC |

The branched counterfactual answers: "How much access did we gain, and at what
cost to which patients?" — not just at a single timestep, but across the full
90-day trajectory where overbooking burden compounds.

---

## Part 6: What the Two Scenarios Prove About the SDK

### The Interfaces Hold

Both scenarios implement the same five methods. The engine runs identically. But:

- **State** is completely different (scalar risk array vs. appointment schedule graph)
- **`step()`** does completely different physics (AR(1) evolution vs. schedule generation and appointment resolution)
- **`predict()`** uses different SDK utilities (ControlledBinaryClassifier vs. ControlledProbabilityModel)
- **`intervene()`** has different targets (the patient at risk vs. a different patient booked into the slot)
- **`measure()`** returns different outcome structures (single binary vs. multi-dimensional)
- **`unit_of_analysis`** differs (patient vs. appointment)

None of this required modifying the engine or the base interfaces.

### The Branched Engine Works for Both

In stroke, the counterfactual branch shows what happens when patients never
receive anticoagulants — their risk trajectories continue evolving naturally
over weeks and months, accumulating incidents that the intervention would have
prevented.

In no-show, the counterfactual branch shows what happens when the clinic never
deploys overbooking — appointment slots go unfilled when patients no-show,
but no patient ever experiences a double-booking wait. The compounding effect
(patients in the factual branch accumulate overbooking history that affects
future interventions) is captured because the branches evolve independently.

### The Analysis Interface Serves Multiple Methods

Both scenarios produce `SimulationResults` that export to multiple formats:
- `to_time_series()` for ITS analysis
- `to_panel()` for DiD (entity-level with treatment assignment)
- `to_entity_snapshots()` for RDD (cross-sectional at a threshold)
- `to_subgroup_panel()` for equity analysis

**Honest scope:** The SDK provides analysis-ready data structures, not
turnkey causal inference implementations. ITS is the simplest to automate
(aggregated time series with treatment indicator). DiD, RDD, and Synthetic
Control require additional assumptions, specifications, and validation that
are better handled by the analyst using these data exports with established
statistical packages (e.g., `statsmodels`, `causalinference`, `rdrobust`).

### The RNG Discipline Is Testable

Both scenarios use partitioned RNG streams. This means we can write a
regression test: run the same scenario with CounterfactualMode.BRANCHED
and CounterfactualMode.NONE, and verify that the factual branch produces
identical results in both modes. If it doesn't, there's a stream
desynchronization bug.

---

## Part 7: Implementation Roadmap

### Phase 1: Extract Core SDK (5 weeks)

**Goal:** Working SDK with `BaseScenario`, `BranchedSimulationEngine`,
`RNGPartitioner`, and `AnalysisDataset` — tested by running the stroke
scenario and getting identical outputs to the current codebase.

1. Create `sdk/` package structure
2. Implement `RNGPartitioner` with `SeedSequence`-based stream spawning
3. Define `BaseScenario` with partitioned RNG and step purity contract
4. Implement `BranchedSimulationEngine` with all three counterfactual modes
5. Implement `SimulationResults` and `AnalysisDataset` with export methods
6. Extract `ControlledBinaryClassifier` from `MLPredictionSimulator`
7. Extract `beta_distributed_risks` and `AR1Process` from existing modules
8. Implement `StrokePreventionScenario` using partitioned RNG streams
9. Verify: same inputs → same outputs as current `VectorizedTemporalRiskSimulator`
10. Regression test: factual branch identical across all three engine modes

### Phase 2: Second Scenario + Interface Hardening (4 weeks)

**Goal:** No-show overbooking scenario running end-to-end with branched
counterfactuals. Interface changes discovered during implementation get
folded back into the SDK.

1. Implement `ControlledProbabilityModel` (AUC-targeted, not PPV/sensitivity)
2. Implement `NoShowOverbookingScenario` with partitioned RNG
3. Verify branched counterfactual produces correct compounding behavior
4. Run the evaluation's key questions as parameter sweeps
5. Identify and resolve interface friction
6. Harden `BaseScenario`, `AnalysisDataset` based on learnings
7. Write the scenario starter template (`_template/`)

### Phase 3: Analysis Tooling + Experiment Management (3 weeks)

**Goal:** ITS implementation works on both scenarios. Panel data exports
integrate with standard statistical packages. Hydra sweeps automate
parameter exploration.

1. Implement ITS on `to_time_series()` export
2. Document DiD/RDD workflows using `to_panel()` with external packages
3. Hydra configuration for parameter sweeps
4. Results comparison and visualization utilities
5. Power analysis calculators
6. *Backlog:* Model drift / temporal degradation parameters for ML utilities

### Phase 4: Documentation + Onboarding (2 weeks)

**Goal:** A new team can build a scenario in a day.

1. "Build Your First Scenario" tutorial (step-by-step)
2. SDK API reference documentation
3. Scenario design guide: RNG discipline, step purity, state representation
   patterns, when to use which ML simulator, common pitfalls
4. The two reference scenarios with notebooks

---

## Appendix A: Design Decisions and Rationale

### Why Branched Counterfactuals, Not Just Snapshots

**Problem:** The v1 design deep-copied state before `intervene()`, measured
the counterfactual at that same timestep, then discarded it. This captures
immediate same-step effects but misses compounding effects.

In stroke: a patient not treated at week 12 has a different risk trajectory
at week 24. In no-show: a patient not overbooked today has a different
overbooking history tomorrow, which changes whether they'd be overbooked
next week.

**Solution:** The `BranchedSimulationEngine` maintains both factual and
counterfactual state objects across the full simulation. Both receive
`step()` calls (same physics). Only the factual branch receives `predict()`
and `intervene()`.

**Cost:** ~2× memory and CPU. Acceptable for medium-scale runs (N ≤ 100k,
T ≤ 260). For larger simulations, use `CounterfactualMode.SNAPSHOT` or
implement optimized `clone_state()`.

**Compatibility:** `CounterfactualMode.SNAPSHOT` preserves v1 behavior.
`CounterfactualMode.NONE` runs a single trajectory for performance.

### Why No Predictions on the Counterfactual Branch

**Decision:** The counterfactual branch does not call `predict()`.

**Rationale:** The default causal question in healthcare AI evaluation is
"what would have happened if we hadn't deployed the AI?" — not "what if
the AI predicted but we ignored it?" The engine's `predict() → intervene()`
pipeline represents the complete causal pathway being evaluated.

**Alternative:** For intention-to-treat analysis (where prediction itself
has effects), the scenario should embed prediction logic in `step()` as
part of domain physics, not as part of the intervention pipeline. This
keeps the engine's causal semantics clean.

### Why RNG Stream Partitioning

**Problem:** With a single RNG, the factual branch's `intervene()` advances
the random state. When the counterfactual branch next draws from the same
RNG (for `step()` or `measure()`), it gets different values — not because
the physics differ, but because the RNG is out of sync.

**Solution:** `RNGPartitioner` uses `numpy.random.SeedSequence.spawn()` to
create statistically independent generators for each process type. The
temporal stream is identical in both branches; the intervention stream is
only consumed by the factual branch.

**Implementation:** Scenario authors use `self.rng.temporal` in `step()`,
`self.rng.prediction` in `predict()`, etc. The engine swaps `self.rng`
between factual and counterfactual stream sets at each step.

### The Step Purity Contract

**Requirement:** `step(state, t)` must be a pure function of `(state, t,
self.rng.temporal)`. It must not read or modify external mutable state.

**Why this matters:** The engine calls `step()` on two different state
objects (factual and counterfactual) at each timestep. If `step()` has
side effects — modifying a shared counter, reading from an external data
source that changes between calls — the two branches will diverge for
reasons unrelated to the intervention.

**What's allowed:** State modifications that reflect ongoing intervention
effects (e.g., a patient on medication has reduced risk) are fine — they're
part of the state object the engine passes in. The factual state has the
medication flag set; the counterfactual state doesn't. `step()` responds
to what's in the state it receives.

**What's not allowed:** Writing to `self.some_counter` in `step()`,
because both branches share `self`. Reading from `self.rng.intervention`
in `step()`, because that stream is only consumed by the factual branch.

### The clone_state() Optimization Path

**Default:** `copy.deepcopy(state)` — correct by construction, potentially
slow for large or complex state objects.

**Override:** Scenario authors can implement `clone_state()` to provide
optimized copying. For array-backed state (stroke), this might be
`np.copy()` on each array. For object-graph state (no-show), this might
be shallow copies of immutable structures plus deep copies of mutable ones.

**Guidance:** Start with the default. Profile. If `clone_state()` shows up
in profiling, optimize. The interface is there; use it when you need it.

### Why Generic[S] Instead of Enforcing numpy Arrays

The stroke scenario could work entirely with numpy arrays. The no-show scenario
cannot — it needs dictionaries, lists of dataclass instances, nested structures.
By making state a generic type parameter, we let scenarios use whatever
representation fits their domain. The engine never inspects state; it just
passes it through.

The cost: no vectorization helpers that assume array layout. The benefit:
scenarios aren't forced into unnatural representations.

**Performance guidance:** For large-scale sweeps (N > 50k), prefer array-backed
or columnar state representations. Object-oriented state (dicts of dataclasses)
is more readable but slower to copy and iterate. The SDK does not enforce a
choice; it documents the tradeoff.

### Why Five Methods and Not Seven (or Three)

Three methods (create, step, measure) are insufficient because they force the
scenario to handle prediction and intervention logic inside `step()`, which
tangles domain physics with policy decisions — and breaks the engine's ability
to selectively apply `predict()` and `intervene()` only on the factual branch.

Seven methods (separating "select who to treat" from "apply treatment effect")
adds a distinction that matters architecturally but confuses scenario authors.
In practice, these two concerns are tightly coupled and easier to reason about
together.

Five methods hit the sweet spot: each has a clear, distinct purpose, and a
team can implement them in order without backtracking.

### Why the ML Simulators Are SDK Utilities, Not Part of the Interface

Scenario authors should not need to implement noise injection and performance
targeting from scratch. That's infrastructure. But they DO need to control
what the model sees (observable state) and when it runs (prediction schedule).

The SDK provides `ControlledBinaryClassifier` and `ControlledProbabilityModel`
as tools the scenario can use inside its `predict()` method. The scenario wires
them to its state; the SDK handles the statistics.

### Analysis-Ready Data, Not Turnkey Causal Inference

**Previous claim (v1):** "The causal inference layer (RDD, DiD, ITS, Synthetic
Control) works on both scenarios."

**Revised position (v2):** The SDK provides *analysis-ready data interfaces*
(`AnalysisDataset`) with export methods (`to_time_series`, `to_panel`,
`to_entity_snapshots`, `to_subgroup_panel`) that produce the data shapes
required by standard causal inference methods. ITS is simple enough to
implement directly. DiD, RDD, and Synthetic Control require assumptions and
specifications that are better handled by analysts using established
statistical packages with these data exports.

This is an honest narrowing. The data structures are the hard part — shaping
simulation output into the right format for each method. The statistical
estimation itself is well-served by existing libraries.

### Future Extension: Event-Driven Engine

The current SDK supports discrete-time simulation only. Event-driven simulation
(continuous time, event queues) is relevant for ED flow, surgical scheduling,
and real-time alerting scenarios. The `BaseScenario` interface is compatible
with an event-driven engine — `step()` could process the next event rather than
a fixed timestep — but the `BranchedSimulationEngine` assumes fixed timesteps.

This is deferred, not dismissed. When a scenario requires it, the extension
path is: add a `DiscreteEventEngine` alongside `BranchedSimulationEngine`,
with `TimeConfig` extended to support event-driven scheduling.

### Future Extension: Model Drift

The SDK's ML utilities (`ControlledBinaryClassifier`, `ControlledProbabilityModel`)
currently produce predictions with fixed performance characteristics. In real
deployments, model performance degrades over time due to data drift, workflow
changes, and population shifts.

Adding temporal degradation parameters (e.g., AUC decays from 0.85 to 0.72
over 24 months) would make multi-year simulations more realistic. This is
backlogged for Phase 3.
