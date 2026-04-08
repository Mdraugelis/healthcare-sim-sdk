"""Equity proof-point validation tests for targeted reminders.

Validates the Rosen et al. (2023) finding that ML-targeted reminders
reduce racial disparities in no-show rates because Black patients
(higher base rates) are correctly ranked higher by the model and
disproportionately benefit from the multiplicative intervention.
"""

import numpy as np

from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
    CallerConfig,
    NoShowTargetedReminderScenario,
)
from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig


def _make_equity_scenario(seed=42, model_auc=0.74, **kwargs):
    """Scenario tuned for equity analysis with large population."""
    tc = TimeConfig(
        n_timesteps=30,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(30)),
    )
    defaults = dict(
        n_patients=2000,
        base_noshow_rate=0.30,
        model_auc=model_auc,
        caller_config=CallerConfig(
            n_providers=6,
            slots_per_provider_per_day=10,
            call_capacity_per_day=20,
            call_success_rate=0.70,
            reminder_effectiveness=0.40,
        ),
    )
    defaults.update(kwargs)
    return NoShowTargetedReminderScenario(
        tc, seed=seed, **defaults
    )


def _get_race_noshow_rates(results, n_days, branch="factual"):
    """Compute no-show rates by race across all timesteps."""
    race_events: dict = {}
    race_counts: dict = {}

    for t in range(n_days):
        if branch == "factual":
            out = results.outcomes[t]
        else:
            out = results.counterfactual_outcomes[t]

        races = out.secondary["race_ethnicity"]
        events = out.events

        for race in np.unique(races):
            mask = races == race
            if race not in race_events:
                race_events[race] = 0.0
                race_counts[race] = 0
            race_events[race] += events[mask].sum()
            race_counts[race] += mask.sum()

    return {
        race: race_events[race] / max(race_counts[race], 1)
        for race in race_events
    }


def _get_race_call_rates(results, n_days):
    """Compute call rates by race on factual branch."""
    race_called: dict = {}
    race_counts: dict = {}

    for t in range(n_days):
        out = results.outcomes[t]
        races = out.secondary["race_ethnicity"]
        called = out.secondary["was_called"]

        for race in np.unique(races):
            mask = races == race
            if race not in race_called:
                race_called[race] = 0.0
                race_counts[race] = 0
            race_called[race] += called[mask].sum()
            race_counts[race] += mask.sum()

    return {
        race: race_called[race] / max(race_counts[race], 1)
        for race in race_called
    }


class TestBaseRateDisparities:
    def test_black_patients_higher_base_rate(self):
        """Black patients should have higher base no-show probability.

        Uses wider race multipliers to produce a detectable signal
        above the noise from Beta distribution + re-scaling.
        """
        # Wider multipliers to overcome geometric-mean dampening
        wide_race = {
            "White": {"prob": 0.55, "noshow_mult": 0.80},
            "Black": {"prob": 0.30, "noshow_mult": 1.35},
            "Hispanic": {"prob": 0.10, "noshow_mult": 1.05},
            "Asian": {"prob": 0.02, "noshow_mult": 0.80},
            "Other": {"prob": 0.03, "noshow_mult": 1.00},
        }
        sc = _make_equity_scenario(
            n_patients=10000,
            race_ethnicity=wide_race,
        )
        state = sc.create_population(10000)

        black_probs = [
            p.base_noshow_probability
            for p in state.patients.values()
            if p.race_ethnicity == "Black"
        ]
        white_probs = [
            p.base_noshow_probability
            for p in state.patients.values()
            if p.race_ethnicity == "White"
        ]

        assert len(black_probs) > 200, "Need enough Black patients"
        assert len(white_probs) > 200, "Need enough White patients"

        black_mean = np.mean(black_probs)
        white_mean = np.mean(white_probs)
        assert black_mean > white_mean, (
            f"Black base rate ({black_mean:.3f}) should exceed "
            f"White ({white_mean:.3f}) due to multiplier 1.35 vs 0.80"
        )


class TestTargetingEquity:
    def test_black_patients_called_more_with_ml(self):
        """ML targeting should call Black patients at higher rate.

        Uses wider race multipliers so the model can clearly
        distinguish risk by race.
        """
        wide_race = {
            "White": {"prob": 0.55, "noshow_mult": 0.80},
            "Black": {"prob": 0.30, "noshow_mult": 1.35},
            "Hispanic": {"prob": 0.10, "noshow_mult": 1.05},
            "Asian": {"prob": 0.02, "noshow_mult": 0.80},
            "Other": {"prob": 0.03, "noshow_mult": 1.00},
        }
        sc = _make_equity_scenario(
            model_auc=0.74,
            n_patients=5000,
            race_ethnicity=wide_race,
        )
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(5000)

        call_rates = _get_race_call_rates(results, 30)
        if "Black" in call_rates and "White" in call_rates:
            assert call_rates["Black"] >= call_rates["White"], (
                f"ML should target Black patients more: "
                f"Black call rate={call_rates['Black']:.3f}, "
                f"White call rate={call_rates['White']:.3f}"
            )


class TestDisparityReduction:
    def test_disparity_narrower_on_factual(self):
        """No-show gap (Black - White) should narrow with intervention."""
        sc = _make_equity_scenario(
            model_auc=0.74,
            caller_config=CallerConfig(
                n_providers=6,
                slots_per_provider_per_day=10,
                call_capacity_per_day=25,
                call_success_rate=0.75,
                reminder_effectiveness=0.50,
            ),
        )
        results = BranchedSimulationEngine(
            sc, CounterfactualMode.BRANCHED
        ).run(2000)

        f_rates = _get_race_noshow_rates(results, 30, "factual")
        cf_rates = _get_race_noshow_rates(results, 30, "counterfactual")

        if "Black" in f_rates and "White" in f_rates:
            f_gap = f_rates["Black"] - f_rates["White"]
            cf_gap = cf_rates["Black"] - cf_rates["White"]

            assert f_gap <= cf_gap, (
                f"Disparity should narrow: factual gap={f_gap:.3f}, "
                f"CF gap={cf_gap:.3f}"
            )

    def test_ml_targeting_reduces_disparity_more_than_random(self):
        """High-AUC model should reduce disparity more than low-AUC."""
        # High AUC = good targeting
        sc_good = _make_equity_scenario(
            seed=42, model_auc=0.80,
            caller_config=CallerConfig(
                n_providers=6,
                slots_per_provider_per_day=10,
                call_capacity_per_day=25,
                call_success_rate=0.75,
                reminder_effectiveness=0.50,
            ),
        )
        res_good = BranchedSimulationEngine(
            sc_good, CounterfactualMode.BRANCHED
        ).run(2000)

        # Low AUC ~ random targeting
        sc_rand = _make_equity_scenario(
            seed=42, model_auc=0.52,
            caller_config=CallerConfig(
                n_providers=6,
                slots_per_provider_per_day=10,
                call_capacity_per_day=25,
                call_success_rate=0.75,
                reminder_effectiveness=0.50,
            ),
        )
        res_rand = BranchedSimulationEngine(
            sc_rand, CounterfactualMode.BRANCHED
        ).run(2000)

        good_f = _get_race_noshow_rates(res_good, 30, "factual")
        good_cf = _get_race_noshow_rates(res_good, 30, "counterfactual")
        rand_f = _get_race_noshow_rates(res_rand, 30, "factual")
        rand_cf = _get_race_noshow_rates(res_rand, 30, "counterfactual")

        if all(
            "Black" in d and "White" in d
            for d in [good_f, good_cf, rand_f, rand_cf]
        ):
            good_reduction = (
                (good_cf["Black"] - good_cf["White"])
                - (good_f["Black"] - good_f["White"])
            )
            rand_reduction = (
                (rand_cf["Black"] - rand_cf["White"])
                - (rand_f["Black"] - rand_f["White"])
            )

            assert good_reduction >= rand_reduction, (
                f"Good model should reduce disparity more: "
                f"good={good_reduction:.3f}, random={rand_reduction:.3f}"
            )
