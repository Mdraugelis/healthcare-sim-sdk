"""Re-run the non-monitoring reports after the ADD fixes.

Use --multirun and the YAML-backed parameters for seed/capacity/AUC/delay grids.
All outputs use finalize_experiment; no generated raw results are committed.
"""

from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from healthcare_sim_sdk.core.engine import BranchedSimulationEngine
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment


def run_cell(cfg):
    root = Path(__file__).resolve().parents[1]
    seed = int(cfg.seed)
    if cfg.kind == "retention":
        from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
            NurseRetentionScenario,
            RetentionConfig,
        )

        config = RetentionConfig(
            model_auc=float(cfg.auc),
            max_interventions_per_manager_per_week=int(cfg.capacity),
        )
        scenario = NurseRetentionScenario(config, seed=seed)
        results = BranchedSimulationEngine(scenario).run(config.n_nurses)
        t = config.n_weeks - 1
        factual = results.outcomes[t].metadata["total_departures"]
        control = results.counterfactual_outcomes[t].metadata["total_departures"]
        report = scenario._classifier._fit_report
        metrics = {
            "factual_departures": factual,
            "counterfactual_departures": control,
            "prevented": control - factual,
            "achieved_auc": report["achieved_auc"],
            "ceiling": report["population_auc_ceiling"],
            "fit_events": report["n_positives"],
        }
    elif cfg.kind == "trews":
        from healthcare_sim_sdk.scenarios.sepsis_early_alert.run_baseline_sweep import (
            load_config,
            run_single,
        )

        raw = load_config("trews_replication")
        metrics = run_single(raw, float(cfg.max_hours), int(cfg.capacity), seed)
    elif cfg.kind == "reminder":
        from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
            CallerConfig,
            NoShowTargetedReminderScenario,
        )

        source = (
            root
            / "scenarios/noshow_targeted_reminders/configs"
            / f"{cfg.reminder_study}_replication.yaml"
        )
        raw = OmegaConf.load(source)
        clinic, policy, pop = raw.clinic, raw.intervention, raw.population
        caller = CallerConfig(
            n_providers=clinic.n_providers,
            slots_per_provider_per_day=clinic.slots_per_provider_per_day,
            call_capacity_per_day=policy.call_capacity_per_day,
            call_success_rate=policy.call_success_rate,
            reminder_effectiveness=policy.reminder_effectiveness,
        )
        extra = {}
        if "race_distribution" in pop:
            extra["race_ethnicity"] = {
                name: {"prob": value, "noshow_mult": pop.race_noshow_multipliers[name]}
                for name, value in pop.race_distribution.items()
            }
        tc = TimeConfig(
            raw.n_days, 1 / 365, prediction_schedule=list(range(raw.n_days))
        )
        scenario = NoShowTargetedReminderScenario(
            tc,
            seed=seed,
            n_patients=raw.n_patients,
            base_noshow_rate=pop.base_noshow_rate,
            noshow_concentration=pop.noshow_concentration,
            model_auc=raw.model.auc,
            targeting_mode=policy.targeting_mode,
            targeting_fraction=policy.get("targeting_fraction", 0.25),
            caller_config=caller,
            **extra,
        )
        results = BranchedSimulationEngine(scenario).run(raw.n_patients)
        f = np.concatenate([o.events for o in results.outcomes.values()])
        c = np.concatenate([o.events for o in results.counterfactual_outcomes.values()])
        races = np.concatenate(
            [o.secondary["race_ethnicity"] for o in results.outcomes.values()]
        )
        assert np.all(f <= c), "Reminder worsened an untreated appointment"
        metrics = {
            "factual_rate": float(f.mean()),
            "counterfactual_rate": float(c.mean()),
            "reduction_pp": float(100 * (c - f).mean()),
            "achieved_auc": scenario._model._fit_report["achieved_auc"],
        }
        for race in ("Black", "White"):
            mask = races == race
            metrics[f"{race}_reduction_pp"] = float(100 * (c[mask] - f[mask]).mean())
        metrics["gap_narrowing_pp"] = (
            metrics["Black_reduction_pp"] - metrics["White_reduction_pp"]
        )
    else:
        raise ValueError(f"Unknown kind {cfg.kind!r}")
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="issue_register")
def main(cfg):
    metrics = run_cell(cfg)
    finalize_experiment(
        Path(HydraConfig.get().runtime.output_dir),
        OmegaConf.to_container(cfg, resolve=True),
        metrics,
        notes="ADD issues register revalidation; simulation conditional on assumptions",
    )


if __name__ == "__main__":
    main()
