"""Observation must preserve actual engine execution and paired outcomes."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import BaseScenario, TimeConfig, Predictions, Interventions, Outcomes
from healthcare_sim_sdk.experiments.trace import TracingScenario


class ToyScenario(BaseScenario):
    def __init__(self):
        super().__init__(TimeConfig(3, 1, prediction_schedule=[1]), seed=9)

    def create_population(self, n):
        return self.rng.population.uniform(.2, .8, n)

    def step(self, state, t):
        return state + self.rng.temporal.uniform(0, .01, len(state))

    def predict(self, state, t):
        return Predictions(state + self.rng.prediction.uniform(0, .01, len(state)))

    def intervene(self, state, predictions, t):
        state[0] *= .5
        return state, Interventions(np.array([0]))

    def measure(self, state, t):
        # Measuring mutates state, catching traces that omit measure() calls.
        state += self.rng.outcomes.uniform(0, .01, len(state))
        return Outcomes(state.copy(), np.arange(len(state)))


def test_trace_exactly_reconciles_to_engine_and_preserves_rng():
    baseline = ToyScenario()
    observed = ToyScenario()
    wrapped = TracingScenario(observed, lambda s: {"state": s})
    expected = BranchedSimulationEngine(baseline).run(3)
    actual = BranchedSimulationEngine(wrapped).run(3)
    for name in ("outcomes", "counterfactual_outcomes"):
        for t, out in getattr(actual, name).items():
            np.testing.assert_array_equal(out.events, getattr(expected, name)[t].events)
    for t, pred in actual.predictions.items():
        np.testing.assert_array_equal(pred.scores, expected.predictions[t].scores)
    for stream in ("population", "temporal", "prediction", "intervention", "outcomes"):
        assert getattr(baseline.rng, stream).bit_generator.state == getattr(observed.rng, stream).bit_generator.state
    rows = wrapped.to_rows()
    for row in rows:
        if row["stage"] == "measure_after":
            target = actual.outcomes if row["branch"] == "factual" else actual.counterfactual_outcomes
            np.testing.assert_array_equal(row["facts"]["state"], target[row["timestep"]].events)
    assert len({r["trace_id"] for r in rows}) == len(rows)
    assert all(r["branch"] == "factual" for r in rows if r["stage"] == "prediction")
    before = next(r for r in rows if r["stage"] == "intervene_before")
    after = next(r for r in rows if r["stage"] == "intervene_after")
    assert after["facts"]["state"][0] == before["facts"]["state"][0] * .5


def test_records_immutable_projector_isolated_and_reuse_rejected():
    def project(s):
        s[:] = -99
        return {"values": s}
    wrapped = TracingScenario(ToyScenario(), project)
    results = BranchedSimulationEngine(wrapped).run(2)
    assert (results.outcomes[0].events > 0).all()
    with pytest.raises(FrozenInstanceError):
        wrapped.records[0].stage = "changed"
    rows = wrapped.to_rows()
    rows[0]["facts"]["values"][0] = 100
    assert wrapped.to_rows()[0]["facts"]["values"][0] == -99
    with pytest.raises(ValueError, match="single-use"):
        BranchedSimulationEngine(wrapped).run(2)


def test_cap_never_silently_truncates_and_output_projection_is_opt_in():
    wrapped = TracingScenario(ToyScenario(), lambda s: {}, max_records=1)
    with pytest.raises(ValueError, match="max_records"):
        BranchedSimulationEngine(wrapped).run(2)
    wrapped = TracingScenario(ToyScenario(), lambda s: {}, stages=("outcome",),
                             output_projector=lambda kind, out: {"first": out.events[0]})
    result = BranchedSimulationEngine(wrapped).run(2)
    assert len(wrapped.records) == 6
    assert wrapped.to_rows()[0]["facts"]["first"] == result.outcomes[0].events[0]


def test_snapshot_trace_does_not_claim_ambiguous_measurements_are_factual():
    wrapped = TracingScenario(ToyScenario(), lambda s: {"state": s})
    BranchedSimulationEngine(wrapped, CounterfactualMode.SNAPSHOT).run(2)
    measurements = [r for r in wrapped.records if r.timestep == 1 and r.stage == "outcome"]
    assert [r.branch for r in measurements] == ["snapshot_unresolved", "snapshot_unresolved"]
