"""Outcome-sealed, whole-course randomized benchmark adapter.

This models independent, already protocol-eligible treatment courses assigned
to standard evaluations or additional evaluations. It does not reconstruct a
risk model, visit physiology, calendar duration, or repeated-course clustering.
Every numerical mechanism input is supplied by the caller, with provenance.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any

import numpy as np

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.rng import RNGPartitioner
from healthcare_sim_sdk.core.scenario import (
    BaseScenario, Interventions, Outcomes, Predictions, TimeConfig,
)


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True)
class _Distribution:
    low: float
    high: float

    @classmethod
    def parse(cls, value: Any, name: str) -> "_Distribution":
        if not isinstance(value, Mapping):
            raise ValueError(f"{name} requires an explicit distribution")
        if value.get("distribution") == "fixed" and set(value) == {
            "distribution", "value",
        }:
            low = high = _finite(value["value"], name)
        elif value.get("distribution") == "uniform" and set(value) == {
            "distribution", "low", "high",
        }:
            low, high = (_finite(value[k], name) for k in ("low", "high"))
        else:
            raise ValueError(f"{name} requires fixed/value or uniform/low/high")
        if low > high:
            raise ValueError(f"{name} distribution bounds are reversed")
        return cls(low, high)

    def draw(self, rng: np.random.Generator) -> float:
        if self.low == self.high:
            return self.low
        return float(rng.uniform(self.low, self.high))


@dataclass
class CourseState:
    assigned_intervention: np.ndarray
    received_extra_evaluation: np.ndarray
    event_probability: np.ndarray
    outcome_uniform: np.ndarray
    baseline_risk: float
    relative_effect: float
    uptake: float
    intervention_applied: bool = False


class ShieldRTBenchmarkScenario(BaseScenario[CourseState]):
    """One independent binary outcome per eligible course, two randomized arms.

    Uncertain baseline risk, relative effect, and uptake are independent draws
    shared across a replicate; they are not individual heterogeneity. Negative
    relative effects permit harm when the complete joint support is valid.
    """

    unit_of_analysis = "rt_course"

    def __init__(
        self, *, n_control: int, n_intervention: int,
        baseline_risk: Mapping, relative_effect: Mapping, uptake: Mapping, seed: int,
    ):
        self.n_control = _integer(n_control, "n_control")
        self.n_intervention = _integer(n_intervention, "n_intervention")
        seed = _integer(seed, "seed", minimum=0)
        if seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        self.baseline_distribution = _Distribution.parse(baseline_risk, "baseline_risk")
        self.effect_distribution = _Distribution.parse(
            relative_effect, "relative_effect"
        )
        self.uptake_distribution = _Distribution.parse(uptake, "uptake")
        for name, distribution in (
            ("baseline_risk", self.baseline_distribution),
            ("uptake", self.uptake_distribution),
        ):
            if not 0 <= distribution.low <= distribution.high <= 1:
                raise ValueError(f"{name} support must lie in [0, 1]")
        possible_treated_risks = [
            p * (1 - effect)
            for p in (self.baseline_distribution.low, self.baseline_distribution.high)
            for effect in (self.effect_distribution.low, self.effect_distribution.high)
        ]
        if not all(np.isfinite(p) and 0 <= p <= 1 for p in possible_treated_risks):
            raise ValueError("Joint baseline_risk/relative_effect support is invalid")
        # A normalized episode index, not an assumed duration in calendar years.
        super().__init__(TimeConfig(1, 1.0, "course", [0]), seed=seed)

    def create_population(self, n_entities: int) -> CourseState:
        n = self.n_control + self.n_intervention
        if _integer(n_entities, "n_entities") != n:
            raise ValueError("Population size must equal the registered arm sizes")
        p0 = self.baseline_distribution.draw(self.rng.population)
        effect = self.effect_distribution.draw(self.rng.population)
        uptake = self.uptake_distribution.draw(self.rng.population)
        assigned: np.ndarray = np.zeros(n, dtype=bool)
        assigned[self.rng.population.permutation(n)[:self.n_intervention]] = True
        # Preallocate independent outcome innovations once. Keeping them in state
        # makes repeated measurement a pure read instead of rerolling the course.
        return CourseState(
            assigned, np.zeros(n, dtype=bool), np.full(n, p0),
            self.rng.outcomes.random(n), p0, effect, uptake,
        )

    def step(self, state: CourseState, t: int) -> CourseState:
        if t != 0:
            raise ValueError("The benchmark has exactly one whole-course timestep")
        return self.clone_state(state)

    def predict(self, state: CourseState, t: int) -> Predictions:
        return Predictions(
            scores=np.full(len(state.assigned_intervention), state.baseline_risk),
            metadata={"role": "eligible cohort; no rescreening or ML model"},
        )

    def intervene(
        self, state: CourseState, predictions: Predictions, t: int,
    ) -> tuple[CourseState, Interventions]:
        updated = self.clone_state(state)
        if not updated.intervention_applied:
            assigned = np.flatnonzero(updated.assigned_intervention)
            received = assigned[
                self.rng.intervention.random(len(assigned)) < state.uptake
            ]
            updated.received_extra_evaluation[received] = True
            updated.event_probability[received] = state.baseline_risk * (
                1 - state.relative_effect
            )
            updated.intervention_applied = True
        return updated, Interventions(
            treated_indices=np.flatnonzero(updated.received_extra_evaluation),
            metadata={
                "control_policy": "standard evaluations",
                "intervention_policy": "standard plus extra evaluations if taken up",
            },
        )

    def measure(self, state: CourseState, t: int) -> Outcomes:
        return Outcomes(
            events=(state.outcome_uniform < state.event_probability).astype(float),
            entity_ids=np.arange(len(state.event_probability)),
            secondary={
                "assigned_intervention": state.assigned_intervention.copy(),
                "received_extra_evaluation": state.received_extra_evaluation.copy(),
                "event_probability": state.event_probability.copy(),
            },
            metadata={
                "baseline_risk_draw": state.baseline_risk,
                "relative_effect_draw": state.relative_effect,
                "uptake_draw": state.uptake,
            },
        )

    def clone_state(self, state: CourseState) -> CourseState:
        return replace(
            state, assigned_intervention=state.assigned_intervention.copy(),
            received_extra_evaluation=state.received_extra_evaluation.copy(),
            event_probability=state.event_probability.copy(),
            outcome_uniform=state.outcome_uniform.copy(),
        )


def validate_inputs(case: Mapping) -> dict[str, Any]:
    """Validate provenance, scalar types and full support without simulating."""
    names = {
        "n_control", "n_intervention", "baseline_risk", "relative_effect", "uptake",
    }
    inputs = case.get("inputs") if isinstance(case, Mapping) else None
    if not isinstance(inputs, Mapping) or set(inputs) != names:
        raise ValueError("case.inputs must contain exactly the five benchmark inputs")
    values = {}
    for name, entry in inputs.items():
        if (
            not isinstance(entry, Mapping) or "value" not in entry
            or not isinstance(entry.get("source_id"), str)
            or not entry["source_id"].strip()
        ):
            raise ValueError(f"{name} requires value and nonempty source_id")
        values[name] = entry["value"]
    # Construction validates supports but draws no population or outcomes.
    ShieldRTBenchmarkScenario(**values, seed=0)
    return values


def run_case(case: Mapping, seed: int) -> dict[str, Any]:
    """Return a trial realization and independent null reference, never a fit.

    Risk difference is intervention minus control; negative means fewer acute
    care events. PASS checks execution integrity, not agreement with a paper.
    Analytic moments condition on the replicate's shared parameter draws and
    average over individual uptake and event randomness.
    """
    values = validate_inputs(case)
    scenario = ShieldRTBenchmarkScenario(**values, seed=seed)
    result = BranchedSimulationEngine(scenario, CounterfactualMode.NONE).run(
        scenario.n_control + scenario.n_intervention
    )
    outcome = result.outcomes[0]
    assigned = outcome.secondary["assigned_intervention"]
    received = outcome.secondary["received_extra_evaluation"]
    probabilities = outcome.secondary["event_probability"]
    control_events = int(outcome.events[~assigned].sum())
    intervention_events = int(outcome.events[assigned].sum())
    control_risk = control_events / scenario.n_control
    intervention_risk = intervention_events / scenario.n_intervention
    p0, effect, uptake = (
        outcome.metadata[k]
        for k in ("baseline_risk_draw", "relative_effect_draw", "uptake_draw")
    )
    p1 = p0 * (1 - effect)
    q1 = p0 * (1 - uptake * effect)
    # Reference seeds occupy a separate domain from allowed trial seeds.
    # The null shares p0, but neither randomized arm's outcome innovations.
    reference = RNGPartitioner(int(seed) + 2**32).create_streams().outcomes
    null_control = int((reference.random(scenario.n_control) < p0).sum())
    null_intervention = int((reference.random(scenario.n_intervention) < p0).sum())
    checks = [
        ("exact_randomized_arm_sizes", int(assigned.sum()) == scenario.n_intervention
         and int((~assigned).sum()) == scenario.n_control,
         "Exact allocation among already eligible independent courses."),
        ("one_binary_outcome_per_course", len(outcome.events) == len(assigned)
         and bool(np.isin(outcome.events, [0, 1]).all()),
         "One whole-course event per entity, no repeated weekly events."),
        ("valid_probabilities", bool(np.isfinite(probabilities).all()
         and ((probabilities >= 0) & (probabilities <= 1)).all()),
         "Complete joint input support and realized probabilities are valid."),
        ("uptake_only_in_intervention", not bool(received[~assigned].any()),
         "Additional evaluations can only reach assigned intervention courses."),
        ("standard_care_preserved", bool((probabilities[~received] == p0).all()),
         "Courses without extra evaluation retain baseline event probability."),
        ("single_trajectory_randomized_trial",
         not bool(result.counterfactual_outcomes),
         "Independent randomized arms run with CounterfactualMode.NONE."),
    ]
    return {
        "seed": int(seed),
        "status": "PASS" if all(ok for _, ok, _ in checks) else "FAIL",
        "predictions": {
            "risk_difference": intervention_risk - control_risk,
            "control_risk": control_risk, "intervention_risk": intervention_risk,
        },
        "baselines": {
            "risk_difference": null_intervention / scenario.n_intervention
            - null_control / scenario.n_control,
        },
        "metrics": {
            "n_control": scenario.n_control, "n_intervention": scenario.n_intervention,
            "control_events": control_events,
            "intervention_events": intervention_events,
            "intervention_uptake_count": int(received.sum()),
            **outcome.metadata,
            "treated_event_probability": p1,
            "analytic_control_risk": p0, "analytic_intervention_risk": q1,
            "analytic_expected_difference": -p0 * effect * uptake,
            "analytic_sampling_variance": p0 * (1 - p0) / scenario.n_control
            + q1 * (1 - q1) / scenario.n_intervention,
            "null_control_events": null_control,
            "null_intervention_events": null_intervention,
        },
        "checks": [
            {"name": name, "status": "PASS" if ok else "FAIL", "detail": detail}
            for name, ok, detail in checks
        ],
    }
