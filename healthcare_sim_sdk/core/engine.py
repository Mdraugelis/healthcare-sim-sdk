"""Branched simulation engine with counterfactual propagation.

Manages the discrete-time simulation loop, calling scenario hooks in the
correct order and maintaining parallel factual/counterfactual trajectories.
"""

import logging
from enum import Enum
from typing import Optional

from .results import SimulationResults
from .scenario import BaseScenario

logger = logging.getLogger(__name__)


class CounterfactualMode(Enum):
    """How the engine handles counterfactual generation."""
    NONE = "none"
    SNAPSHOT = "snapshot"
    BRANCHED = "branched"


class BranchedSimulationEngine:
    """Simulation engine with true counterfactual propagation.

    Maintains two parallel state trajectories:
    - FACTUAL: step -> predict -> intervene -> measure
    - COUNTERFACTUAL: step -> measure (no predict/intervene)

    Both branches call step() with the same temporal RNG stream state
    at each timestep, ensuring identical stochastic evolution. They
    diverge ONLY where intervention modified the factual state.

    The counterfactual branch does NOT call predict(). The default
    causal question is "what if we hadn't deployed the AI at all?"
    """

    def __init__(
        self,
        scenario: BaseScenario,
        counterfactual_mode: CounterfactualMode = CounterfactualMode.BRANCHED,
    ):
        self.scenario = scenario
        self.counterfactual_mode = counterfactual_mode
        self.results: Optional[SimulationResults] = None

    def run(self, n_entities: int) -> SimulationResults:
        """Execute the full simulation pipeline."""
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
            counterfactual_mode="branched",
        )

        # 1. Create shared initial population
        state_factual = sc.create_population(n_entities)
        validation = sc.validate_population(state_factual)
        results.record_validation("population", validation)

        # Clone for counterfactual branch
        state_counterfactual = sc.clone_state(state_factual)

        # Create separate RNG streams for counterfactual branch
        factual_rng = sc.rng
        cf_partitioner = sc._partitioner.fork()
        cf_rng = cf_partitioner.create_streams()

        logger.info(
            "Population created: %d entities, mode=branched", n_entities
        )

        # 2. Temporal loop
        for t in range(tc.n_timesteps):
            # 2a. Evolve FACTUAL state
            sc.rng = factual_rng
            state_factual = sc.step(state_factual, t)

            # 2b. Evolve COUNTERFACTUAL state
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
        logger.info(
            "Simulation complete: %d timesteps, mode=branched",
            tc.n_timesteps,
        )

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
            counterfactual_mode="snapshot",
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
                results.record_outcomes(
                    t, cf_outcomes, counterfactual=True
                )

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
            counterfactual_mode="none",
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
