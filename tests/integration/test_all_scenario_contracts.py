"""Behavioral invariants: non-interference AND stream synchronization.

The two assertions detect different defects. This matrix includes every domain
scenario (the toy template has separate engine/unit and notebook coverage). Untreated identity is valid
only without capacity/SOC spillovers; it is checked separately for stroke and
for original no-show appointments, which overbooking does not change.
"""

import importlib

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig


SCENARIOS = [
    ("stroke_prevention", "StrokePreventionScenario"),
    ("sepsis_early_alert", "SepsisEarlyAlertScenario"),
    ("noshow_overbooking", "NoShowOverbookingScenario"),
    ("noshow_targeted_reminders", "NoShowTargetedReminderScenario"),
    ("nurse_retention", "NurseRetentionScenario"),
    ("malnutrition_prioritization", "MalnutritionPrioritizationScenario"),
    ("paper01_epic_esm", "EpicESMScenario"),
    ("paper03_kaiser_aam", "KaiserAAMScenario"),
    ("paper04_insight_rct", "InSightRCTScenario"),
    ("paper05_composer", "COMPOSERScenario"),
    ("paper06_shield_rt", "ShieldRTScenario"),
    ("paper07_manz_nudges", "ManzNudgeScenario"),
    ("paper09_masai", "MASAIScenario"),
]


def make_scenario(folder, name, seed):
    mod = importlib.import_module(f"healthcare_sim_sdk.scenarios.{folder}.scenario")
    cls = getattr(mod, name)
    tc = TimeConfig(12, 1 / 52, prediction_schedule=[0, 3, 6, 9])
    if folder == "stroke_prevention":
        return cls(
            config=mod.StrokeConfig(
                n_weeks=12, allow_infeasible_model=True, treatment_threshold=0.005
            ),
            seed=seed,
        )
    if folder == "nurse_retention":
        return cls(config=mod.RetentionConfig(n_weeks=12, model_auc=0.70), seed=seed)
    if folder == "sepsis_early_alert":
        return cls(time_config=tc, config=mod.SepsisConfig(model_auc=0.70), seed=seed)
    if folder == "paper01_epic_esm":
        return cls(time_config=tc, config=mod.EpicESMConfig(model_auc=0.65), seed=seed)
    if folder == "paper09_masai":
        return cls(time_config=tc, transpara_auc=0.70, seed=seed)
    if folder in (
        "paper03_kaiser_aam",
        "paper04_insight_rct",
        "paper05_composer",
        "paper06_shield_rt",
        "paper07_manz_nudges",
    ):
        return cls(time_config=tc, model_auc=0.65, seed=seed)
    if folder == "malnutrition_prioritization":
        return cls(
            seed=seed, time_config=tc, config=mod.MalnutritionConfig(ml_enabled=False)
        )
    return cls(time_config=tc, seed=seed, model_auc=0.70)


def audit_coupling(scenario, n):
    state_f = scenario.create_population(n)
    state_c = scenario.clone_state(state_f)
    factual = scenario.rng
    control = scenario._partitioner.fork().create_streams()
    for t in range(scenario.time_config.n_timesteps):
        scenario.rng = factual
        state_f = scenario.step(state_f, t)
        scenario.rng = control
        state_c = scenario.step(state_c, t)
        if t in scenario.time_config.prediction_schedule:
            scenario.rng = factual
            pred = scenario.predict(state_f, t)
            state_f, _ = scenario.intervene(state_f, pred, t)
        scenario.rng = factual
        f = scenario.measure(state_f, t)
        scenario.rng = control
        c = scenario.measure(state_c, t)
        for stream in ("temporal", "outcomes"):
            assert (
                getattr(factual, stream).bit_generator.state
                == getattr(control, stream).bit_generator.state
            ), (stream, t)
        if scenario.unit_of_analysis == "appointment" and hasattr(state_f, "schedule"):
            assert [s.patient_id for s in state_f.schedule] == [
                s.patient_id for s in state_c.schedule
            ]
        if type(scenario).__name__ == "NoShowOverbookingScenario":
            np.testing.assert_array_equal(f.events, c.events)
    scenario.rng = factual


@pytest.mark.parametrize("folder,name", SCENARIOS)
@pytest.mark.parametrize("seed", [7, 42])
def test_every_scenario_noninterference_and_coupling(folder, name, seed):
    sc = make_scenario(folder, name, seed)
    audit_coupling(sc, 1000)
    branched = BranchedSimulationEngine(make_scenario(folder, name, seed)).run(1000)
    single = BranchedSimulationEngine(
        make_scenario(folder, name, seed), CounterfactualMode.NONE
    ).run(1000)
    for t in branched.outcomes:
        np.testing.assert_array_equal(
            branched.outcomes[t].events, single.outcomes[t].events
        )


def test_stroke_untreated_identity():
    sc = make_scenario("stroke_prevention", "StrokePreventionScenario", 7)
    result = BranchedSimulationEngine(sc).run(1000)
    treated = np.zeros(1000, bool)
    for assignment in result.interventions.values():
        treated[assignment.treated_indices] = True
    assert (~treated).any()
    for t in result.outcomes:
        np.testing.assert_array_equal(
            result.outcomes[t].events[~treated],
            result.counterfactual_outcomes[t].events[~treated],
        )


def test_stream_audit_catches_a_defect_identity_test_misses():
    from tests.integration.test_step_purity import StochasticPureScenario, assert_purity

    class VariableDrawScenario(StochasticPureScenario):
        def step(self, state, t):
            if state.mean() < 0.4:
                self.rng.temporal.random(1)
            return super().step(state, t)

    assert_purity(VariableDrawScenario)
    scenario = VariableDrawScenario(
        TimeConfig(5, 1 / 52, prediction_schedule=[0]), seed=42
    )
    with pytest.raises(AssertionError):
        audit_coupling(scenario, 1000)
