"""Resolved appointments are measured exactly once, never before observation."""
import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.noshow_overbooking.scenario import (
    ClinicConfig, NoShowOverbookingScenario,
)
from healthcare_sim_sdk.scenarios.noshow_overbooking.realistic_scenario import (
    ClinicConfig as RealisticClinicConfig, RealisticNoShowScenario,
)
from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
    CallerConfig, NoShowTargetedReminderScenario,
)


@pytest.mark.parametrize('kind', ['simple', 'realistic', 'reminders'])
def test_raw_appointments_reconcile_with_resolution_counters(kind):
    tc = TimeConfig(4, 1 / 365, 'day', prediction_schedule=list(range(4)))
    common = dict(time_config=tc, seed=7, n_patients=100, model_type='baseline')
    if kind == 'simple':
        scenario = NoShowOverbookingScenario(
            **common, clinic_config=ClinicConfig(
                n_providers=2, slots_per_provider_per_day=4))
    elif kind == 'realistic':
        scenario = RealisticNoShowScenario(
            **common, clinic_config=RealisticClinicConfig(
                n_providers=2, slots_per_provider_per_day=4))
    else:
        scenario = NoShowTargetedReminderScenario(
            **common, caller_config=CallerConfig(
                n_providers=2, slots_per_provider_per_day=4))
    result = BranchedSimulationEngine(scenario).run(100)
    for branch, records in [('factual', result.outcomes),
                            ('counterfactual', result.counterfactual_outcomes)]:
        first = records[0]
        assert first.events.size == first.entity_ids.size == 0
        assert first.metadata['outcome_observed'] is False
        assert first.metadata['measurement_status'] == 'no_resolved_slots'
        assert all(len(v) == 0 for v in first.secondary.values())
        ids = np.concatenate([out.entity_ids for out in records.values()])
        assert len(ids) == len(set(ids.tolist()))
        assert len(ids) == records[3].metadata['total_resolved']
        assert sum(out.events.sum() for out in records.values()) == records[3].metadata['total_noshows']
        assert all(out.metadata['n_resolved_slots'] == len(out.events)
                   for out in records.values())
        ts = result.to_analysis().to_time_series(branch=branch)
        assert np.isnan(ts['outcomes'][0])
        np.testing.assert_array_equal(ts['observed'], [False, True, True, True])
        panel = result.to_analysis().to_panel(branch=branch)
        assert len(panel['entity_ids']) == len(ids)
        assert not (panel['timesteps'] == 0).any()
