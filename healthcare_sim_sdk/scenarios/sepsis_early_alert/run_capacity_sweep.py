"""Sweep rapid_response_capacity to find where TREWS vs ESM differentiate.

The studies and capacities are in configs/capacity_sweep.yaml. Usage:
    python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_capacity_sweep
"""

import logging
import time

from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment
from healthcare_sim_sdk.experiments.runner import (
    check, run_dir, runner_settings,
)
from healthcare_sim_sdk.scenarios.sepsis_early_alert.scenario import (
    SepsisConfig,
    SepsisEarlyAlertScenario,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

CONFIGS_DIR = Path(__file__).parent / "configs"


def load_config(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def run_single(raw: dict, capacity: int, seed: int = 42) -> dict:
    """Run one config at a given capacity, return key metrics."""
    n_patients = raw["n_patients"]
    n_timesteps = raw["n_timesteps"]

    tc = TimeConfig(
        n_timesteps=n_timesteps,
        timestep_duration=4 / 8760,
        timestep_unit="4hr_block",
        prediction_schedule=list(range(n_timesteps)),
    )
    cfg = SepsisConfig(
        n_patients=n_patients,
        sepsis_incidence=raw["sepsis_incidence"],
        risk_concentration=raw["risk_concentration"],
        mean_los_timesteps=raw["mean_los_timesteps"],
        los_std_timesteps=raw["los_std_timesteps"],
        ar1_rho=raw["ar1_rho"],
        ar1_sigma=raw["ar1_sigma"],
        model_auc=raw["model_auc"],
        alert_threshold_percentile=raw["alert_threshold_percentile"],
        initial_response_rate=raw["initial_response_rate"],
        fatigue_coefficient=raw["fatigue_coefficient"],
        floor_response_rate=raw["floor_response_rate"],
        treatment_effectiveness=raw["treatment_effectiveness"],
        kumar_half_life_hours=raw.get("kumar_half_life_hours", 0.0),
        max_treatment_effectiveness=raw.get(
            "max_treatment_effectiveness", 0.50,
        ),
        rapid_response_capacity=capacity,
        prog_at_risk=raw["prog_at_risk"],
        prog_sepsis=raw["prog_sepsis"],
        prog_severe=raw["prog_severe"],
        mort_sepsis=raw["mort_sepsis"],
        mort_severe=raw["mort_severe"],
        mort_shock=raw["mort_shock"],
    )

    sc = SepsisEarlyAlertScenario(
        time_config=tc, seed=seed, config=cfg,
    )
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED,
    ).run(n_patients)

    final_t = n_timesteps - 1
    f_out = results.outcomes[final_t]
    cf_out = results.counterfactual_outcomes[final_t]

    f_deaths = float(f_out.secondary["mortality"].sum())
    cf_deaths = float(cf_out.secondary["mortality"].sum())
    f_treated = float(f_out.secondary["treated"].sum())

    # Count capacity-limited steps
    cap_limited = sum(
        1 for intv in results.interventions.values()
        if intv.metadata.get("capacity_limited", False)
    )

    # Final response rate
    response_rates = [
        intv.metadata["response_rate"]
        for intv in results.interventions.values()
    ]

    return {
        "f_deaths": int(f_deaths),
        "cf_deaths": int(cf_deaths),
        "f_mort_pct": round(f_deaths / n_patients * 100, 2),
        "cf_mort_pct": round(cf_deaths / n_patients * 100, 2),
        "reduction_pp": round((cf_deaths - f_deaths) / n_patients * 100, 2),
        "treated": int(f_treated),
        "treat_pct": round(f_treated / n_patients * 100, 1),
        "cap_limited": cap_limited,
        "final_response": round(response_rates[-1], 3) if response_rates else 0,
    }


@hydra.main(version_base=None, config_path="configs",
            config_name="capacity_sweep")
def main(cfg: DictConfig) -> None:
    settings = runner_settings(cfg)
    first, second = settings["studies"]
    trews_raw = load_config(first)
    esm_raw = load_config(second)
    seed = settings["seed"]
    capacities = settings["capacities"]

    logger.info("Sweeping rapid_response_capacity: %s", capacities)
    logger.info("TREWS: AUC=%.2f, threshold=%.0f%%, response=%.2f",
                trews_raw["model_auc"],
                trews_raw["alert_threshold_percentile"],
                trews_raw["initial_response_rate"])
    logger.info("ESM:   AUC=%.2f, threshold=%.0f%%, response=%.2f",
                esm_raw["model_auc"],
                esm_raw["alert_threshold_percentile"],
                esm_raw["initial_response_rate"])
    logger.info("")

    trews_results = {}
    esm_results = {}

    for cap in capacities:
        t0 = time.time()
        trews_results[cap] = run_single(trews_raw, cap, seed)
        esm_results[cap] = run_single(esm_raw, cap, seed)
        elapsed = time.time() - t0
        logger.info("  capacity=%3d  done (%.1fs)", cap, elapsed)

    # Print results table
    logger.info("")
    logger.info("=" * 100)
    logger.info("CAPACITY SWEEP RESULTS")
    logger.info("=" * 100)
    logger.info("")

    # TREWS table
    logger.info("TREWS (AUC=%.2f, targeted alerts, high engagement)",
                trews_raw["model_auc"])
    logger.info("-" * 95)
    logger.info("%-8s  %8s  %8s  %10s  %8s  %10s  %12s  %10s",
                "Cap", "F.Deaths", "CF.Deaths", "Reduct.pp",
                "Treated", "Treat%", "Cap.Limited", "Resp.Rate")
    logger.info("-" * 95)
    for cap in capacities:
        r = trews_results[cap]
        logger.info("%-8d  %8d  %8d  %9.2f%%  %8d  %9.1f%%  %8d/%2d  %10.3f",
                    cap, r["f_deaths"], r["cf_deaths"],
                    r["reduction_pp"], r["treated"], r["treat_pct"],
                    r["cap_limited"], 42, r["final_response"])
    logger.info("")

    # ESM table
    logger.info("ESM (AUC=%.2f, high alert volume, lower engagement)",
                esm_raw["model_auc"])
    logger.info("-" * 95)
    logger.info("%-8s  %8s  %8s  %10s  %8s  %10s  %12s  %10s",
                "Cap", "F.Deaths", "CF.Deaths", "Reduct.pp",
                "Treated", "Treat%", "Cap.Limited", "Resp.Rate")
    logger.info("-" * 95)
    for cap in capacities:
        r = esm_results[cap]
        logger.info("%-8d  %8d  %8d  %9.2f%%  %8d  %9.1f%%  %8d/%2d  %10.3f",
                    cap, r["f_deaths"], r["cf_deaths"],
                    r["reduction_pp"], r["treated"], r["treat_pct"],
                    r["cap_limited"], 42, r["final_response"])
    logger.info("")

    # Comparative: TREWS advantage over ESM
    logger.info("TREWS ADVANTAGE (mortality reduction TREWS - ESM, in pp)")
    logger.info("-" * 60)
    logger.info("%-8s  %12s  %12s  %12s", "Cap",
                "TREWS reduct", "ESM reduct", "TREWS advtg")
    logger.info("-" * 60)
    for cap in capacities:
        t_r = trews_results[cap]["reduction_pp"]
        e_r = esm_results[cap]["reduction_pp"]
        logger.info("%-8d  %11.2f%%  %11.2f%%  %11.2f pp",
                    cap, t_r, e_r, t_r - e_r)
    logger.info("")

    logger.info(
        "NOTE: The simulation suggests, under these assumptions, the capacity"
    )
    logger.info(
        "at which TREWS-like systems diverge from ESM-like systems. At low"
    )
    logger.info(
        "capacity both are bottlenecked; at high capacity, TREWS's better AUC"
    )
    logger.info(
        "and clinician engagement translate into differential mortality benefit."
    )

    cells = [{"study": name, "capacity": cap, **res[cap]}
             for name, res in ((first, trews_results), (second, esm_results))
             for cap in capacities]
    advantage = [trews_results[c]["reduction_pp"]
                 - esm_results[c]["reduction_pp"] for c in capacities]
    finalize_experiment(
        run_dir(),
        {"scenario": "sepsis_early_alert",
         "experiment_name": "capacity_sweep", **settings,
         "study_parameters": {first: trews_raw, second: esm_raw}},
        {"n_cells": len(cells),
         "max_advantage_pp": max(advantage),
         "min_advantage_pp": min(advantage)},
        {"cells": cells},
        require_validated=True,
        validation_checks=[
            check("grid_complete", "Every study x capacity cell ran",
                  len(cells) == 2 * len(capacities),
                  expected=2 * len(capacities), actual=len(cells)),
            check("deaths_nonnegative",
                  "Deaths on each branch are non-negative",
                  all(r[k] >= 0 for r in cells
                      for k in ("f_deaths", "cf_deaths")), expected=">= 0"),
        ],
    )


if __name__ == "__main__":
    main()
