"""Integration tests: step purity, exports, determinism for the
malnutrition prioritization scenario."""

from pathlib import Path

import numpy as np
import pytest

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.malnutrition_prioritization.scenario import (
    MalnutritionConfig,
    MalnutritionPrioritizationScenario,
)
from .test_step_purity import assert_purity


def _run(seed=42, horizon=28, ml_enabled=False, **overrides):
    defaults = dict(
        initial_census=80,
        admissions_per_day=10,
        rdn_assessments_per_day=8,
    )
    defaults.update(overrides)
    cfg = MalnutritionConfig(ml_enabled=ml_enabled, **defaults)
    tc = TimeConfig(
        n_timesteps=horizon,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(horizon)),
    )
    sc = MalnutritionPrioritizationScenario(
        time_config=tc, seed=seed, config=cfg)
    engine = BranchedSimulationEngine(sc, CounterfactualMode.BRANCHED)
    return sc, engine.run(sc.n_entities_required)


class TestMalnutritionPurity:

    def test_step_purity_baseline_workflow(self):
        """BRANCHED factual must equal NONE trajectory (the SDK's
        critical identity test), with the ML tier inert."""
        assert_purity(
            MalnutritionPrioritizationScenario,
            n_entities=100,
            n_timesteps=20,
            config=MalnutritionConfig(ml_enabled=False),
        )

    def test_step_purity_daily_schedule(self):
        assert_purity(
            MalnutritionPrioritizationScenario,
            n_entities=100,
            n_timesteps=20,
            prediction_schedule=list(range(20)),
            config=MalnutritionConfig(ml_enabled=False),
        )


class TestExports:

    def test_unit_of_analysis(self):
        sc, res = _run()
        assert res.unit_of_analysis == "patient_admission"

    def test_time_series_export(self):
        _, res = _run()
        ts = res.to_analysis().to_time_series()
        assert len(ts["timesteps"]) == 28
        assert len(ts["outcomes"]) == 28

    def test_panel_export(self):
        sc, res = _run()
        panel = res.to_analysis().to_panel()
        n = sc.n_entities_required
        assert len(panel["entity_ids"]) == n * 28
        assert panel["unit_of_analysis"] == "patient_admission"

    def test_subgroup_panel_export(self):
        _, res = _run()
        sub = res.to_analysis().to_subgroup_panel(
            subgroup_key="ethnicity")
        groups = set(np.unique(sub["subgroup"]))
        assert "NotHispanic" in groups
        assert "HispanicLatino" in groups

    def test_counterfactual_secondary_available(self):
        # Equity analysis reads CF subgroup metrics directly from the
        # CF Outcomes (to_subgroup_panel is factual-only).
        _, res = _run()
        cf_last = res.counterfactual_outcomes[27].secondary
        assert "ethnicity" in cf_last
        assert "missed" in cf_last


class TestDeterminism:

    def test_same_seed_same_results(self):
        _, res_a = _run(seed=11)
        _, res_b = _run(seed=11)
        for t in (0, 13, 27):
            np.testing.assert_array_equal(
                res_a.outcomes[t].events, res_b.outcomes[t].events)

    def test_different_seeds_differ(self):
        _, res_a = _run(seed=11)
        _, res_b = _run(seed=12)
        harm_a = res_a.outcomes[27].secondary["harm"]
        harm_b = res_b.outcomes[27].secondary["harm"]
        assert not np.array_equal(harm_a, harm_b)


_CONFIG_DIR = Path(
    "healthcare_sim_sdk/scenarios/malnutrition_prioritization/configs")


@pytest.mark.skipif(
    not ((_CONFIG_DIR / "model_frozen.yaml").exists()
         and (_CONFIG_DIR / "model_targets.yaml").exists()),
    reason="local model_targets.yaml / model_frozen.yaml not present")
class TestMalnutritionPurityMLOn:

    def test_step_purity_with_ml_arm(self):
        """Purity must hold with the full ML pipeline active."""
        assert_purity(
            MalnutritionPrioritizationScenario,
            n_entities=100,
            n_timesteps=20,
            prediction_schedule=list(range(20)),
            config=MalnutritionConfig(ml_enabled=True),
        )

    def test_ml_effect_direction_smoke(self):
        _, res = _run(ml_enabled=True)
        last_f = res.outcomes[27].secondary
        last_c = res.counterfactual_outcomes[27].secondary
        # Assessed true+ get seen no later on the ML arm (median).
        f_tta = last_f["time_to_assessment"][
            (last_f["assessed"] == 1)
            & (last_f["mal_at_discharge"] == 1)]
        c_tta = last_c["time_to_assessment"][
            (last_c["assessed"] == 1)
            & (last_c["mal_at_discharge"] == 1)]
        assert np.median(f_tta) <= np.median(c_tta)
