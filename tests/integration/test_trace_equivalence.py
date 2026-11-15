"""Tracing must not change outcomes or random-number consumption.

Each scenario runs twice from the same seed: once plain, once wrapped in
TracingScenario with a projector that reads state at every stage. Factual
and counterfactual outcomes must be identical element by element, and each
RNG stream must end at the same position.
"""

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine, CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.experiments.trace import TracingScenario


def _reminders(seed):
    from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
        NoShowTargetedReminderScenario,
    )
    tc = TimeConfig(n_timesteps=15, timestep_duration=1 / 365,
                    timestep_unit="day", prediction_schedule=list(range(15)))
    return NoShowTargetedReminderScenario(
        time_config=tc, seed=seed, n_patients=800), 800


def _retention(seed):
    from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
        NurseRetentionScenario, RetentionConfig,
    )
    return NurseRetentionScenario(
        config=RetentionConfig(n_weeks=20), seed=seed), 1000


def _sepsis(seed):
    from healthcare_sim_sdk.scenarios.sepsis_early_alert.scenario import (
        SepsisEarlyAlertScenario,
    )
    tc = TimeConfig(n_timesteps=24, timestep_duration=1 / 8760,
                    timestep_unit="hour",
                    prediction_schedule=list(range(24)))
    return SepsisEarlyAlertScenario(time_config=tc, seed=seed), 1500


SCENARIOS = {"reminders": _reminders, "retention": _retention,
             "sepsis": _sepsis}


def _state_reader(state):
    # Touch the state the way a real projector would, without copying it out.
    return {"type": type(state).__name__}


def _stream_positions(scenario):
    streams = scenario.rng
    return {name: getattr(streams, name).bit_generator.state["state"]
            for name in ("population", "temporal", "prediction",
                         "intervention", "outcomes")}


def _assert_same(plain, traced):
    for branch in ("outcomes", "counterfactual_outcomes"):
        a, b = getattr(plain, branch), getattr(traced, branch)
        assert sorted(a) == sorted(b)
        for t in a:
            np.testing.assert_array_equal(a[t].events, b[t].events)
            if a[t].entity_ids is not None:
                np.testing.assert_array_equal(a[t].entity_ids, b[t].entity_ids)
            for key, value in a[t].secondary.items():
                np.testing.assert_array_equal(value, b[t].secondary[key])
    assert sorted(plain.predictions) == sorted(traced.predictions)
    for t in plain.predictions:
        np.testing.assert_array_equal(plain.predictions[t].scores,
                                      traced.predictions[t].scores)
    for t in plain.interventions:
        np.testing.assert_array_equal(plain.interventions[t].treated_indices,
                                      traced.interventions[t].treated_indices)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
@pytest.mark.parametrize("seed", [3, 17])
def test_tracing_leaves_outcomes_and_rng_unchanged(name, seed):
    plain_scenario, n = SCENARIOS[name](seed)
    plain = BranchedSimulationEngine(plain_scenario).run(n)

    inner, n = SCENARIOS[name](seed)
    wrapped = TracingScenario(inner, _state_reader)
    traced = BranchedSimulationEngine(wrapped).run(n)

    _assert_same(plain, traced)
    assert _stream_positions(plain_scenario) == _stream_positions(inner)
    stages = {r.stage for r in wrapped.records}
    assert {"population", "step_after", "prediction", "intervention",
            "outcome"} <= stages
    branches = {r.branch for r in wrapped.records if r.stage == "outcome"}
    assert branches == {"factual", "counterfactual"}


def test_tracing_in_snapshot_mode_labels_branch_unresolved():
    inner, n = _retention(5)
    wrapped = TracingScenario(inner, _state_reader)
    BranchedSimulationEngine(
        wrapped, counterfactual_mode=CounterfactualMode.SNAPSHOT).run(n)
    branches = {r.branch for r in wrapped.records if r.stage == "outcome"}
    assert "snapshot_unresolved" in branches
