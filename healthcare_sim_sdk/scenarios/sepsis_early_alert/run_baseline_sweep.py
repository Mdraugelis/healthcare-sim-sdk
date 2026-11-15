"""Sweep baseline clinical detection delay to calibrate TREWS replication.

The ML system's value-add is detecting sepsis BEFORE routine clinical care.
This sweep varies the standard-of-care detection timeline to find the
delay that reproduces the Adams et al. (2022) finding of ~3pp mortality
reduction from TREWS.

The grid is in configs/baseline_sweep.yaml. Usage:
    python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_baseline_sweep
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


def run_single(
    raw: dict, max_hours: float, capacity: int, seed: int = 42,
) -> dict:
    """Run TREWS config at a given baseline detection max_hours."""
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
        baseline_detection_enabled=True,
        baseline_detect_alpha=raw.get("baseline_detect_alpha", 2.0),
        baseline_detect_beta=raw.get("baseline_detect_beta", 5.0),
        baseline_detect_max_hours=max_hours,
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
    cf_treated = float(cf_out.secondary["treated"].sum())

    # Compute mean detection delay from the distribution
    alpha = cfg.baseline_detect_alpha
    beta_param = cfg.baseline_detect_beta
    mean_delay_hrs = (alpha / (alpha + beta_param)) * max_hours

    return {
        "max_hours": max_hours,
        "capacity": capacity,
        "mean_delay_hrs": round(mean_delay_hrs, 1),
        "f_deaths": int(f_deaths),
        "cf_deaths": int(cf_deaths),
        "f_mort_pct": round(f_deaths / n_patients * 100, 2),
        "cf_mort_pct": round(cf_deaths / n_patients * 100, 2),
        "reduction_pp": round(
            (cf_deaths - f_deaths) / n_patients * 100, 2,
        ),
        "f_treated": int(f_treated),
        "cf_treated": int(cf_treated),
        "f_treat_pct": round(f_treated / n_patients * 100, 1),
        "cf_treat_pct": round(cf_treated / n_patients * 100, 1),
    }


@hydra.main(version_base=None, config_path="configs",
            config_name="baseline_sweep")
def main(cfg: DictConfig) -> None:
    settings = runner_settings(cfg)
    trews_raw = load_config(settings["study"])
    seed = settings["seed"]
    max_hours_values = settings["max_hours_values"]
    capacity_values = settings["capacity_values"]
    low, high = settings["target_reduction_pp"]

    logger.info("TREWS Baseline Detection Delay x Capacity Sweep")
    logger.info("=" * 80)
    logger.info(
        "Model: AUC=%.2f, threshold=%.0f%%, response=%.2f, "
        "Kumar t1/2=%.0fh",
        trews_raw["model_auc"],
        trews_raw["alert_threshold_percentile"],
        trews_raw["initial_response_rate"],
        trews_raw.get("kumar_half_life_hours", 0),
    )
    logger.info(
        "Detection delay: Beta(%.1f, %.1f) * max_hours",
        trews_raw.get("baseline_detect_alpha", 2.0),
        trews_raw.get("baseline_detect_beta", 5.0),
    )
    logger.info(
        "Target: ~3pp mortality reduction (Adams et al. 2022)",
    )
    logger.info("")

    all_results = {}
    for cap in capacity_values:
        logger.info("--- Capacity = %d ---", cap)
        for mh in max_hours_values:
            t0 = time.time()
            r = run_single(trews_raw, mh, cap, seed)
            all_results[(cap, mh)] = r
            elapsed = time.time() - t0
            logger.info(
                "  cap=%3d max_hours=%3d  mean=%.1fh  "
                "reduction=%.2fpp  F.treat=%d  CF.treat=%d  "
                "(%.1fs)",
                cap, mh, r["mean_delay_hrs"],
                r["reduction_pp"],
                r["f_treated"], r["cf_treated"],
                elapsed,
            )
        logger.info("")

    # Print summary: reduction_pp matrix
    logger.info("=" * 80)
    logger.info(
        "MORTALITY REDUCTION (pp) by Capacity x Baseline Delay",
    )
    logger.info("=" * 80)
    header = "%-8s" % "Cap \\ Hrs"
    for mh in max_hours_values:
        header += "  %6d" % mh
    logger.info(header)
    logger.info("-" * (10 + 8 * len(max_hours_values)))
    for cap in capacity_values:
        row = "%-8d" % cap
        for mh in max_hours_values:
            r = all_results[(cap, mh)]
            val = r["reduction_pp"]
            marker = "*" if low <= val <= high else " "
            row += "  %5.2f%s" % (val, marker)
        logger.info(row)

    logger.info("")
    logger.info("* = within target range (%s-%s pp)", low, high)
    logger.info("")

    # Print treated counts matrix
    logger.info(
        "ML VALUE-ADD: (F.Treated - CF.Treated) patients",
    )
    logger.info("-" * (10 + 8 * len(max_hours_values)))
    header = "%-8s" % "Cap \\ Hrs"
    for mh in max_hours_values:
        header += "  %6d" % mh
    logger.info(header)
    logger.info("-" * (10 + 8 * len(max_hours_values)))
    for cap in capacity_values:
        row = "%-8d" % cap
        for mh in max_hours_values:
            r = all_results[(cap, mh)]
            diff = r["f_treated"] - r["cf_treated"]
            row += "  %6d" % diff
        logger.info(row)

    logger.info("")
    logger.info("INTERPRETATION:")
    logger.info(
        "  Rows: how many patients ML can treat per 4hr block",
    )
    logger.info(
        "  Cols: how long (max) standard care takes to detect "
        "sepsis",
    )
    logger.info(
        "  Values: mortality reduction from adding ML over "
        "standard care",
    )
    logger.info("")
    logger.info(
        "  The cell where reduction ~= 3pp tells us the "
        "capacity + baseline delay"
    )
    logger.info(
        "  combination that matches the real TREWS deployment.",
    )
    logger.info("")
    logger.info(
        "  All findings are conditional on the stated assumptions."
    )

    cells = list(all_results.values())
    reductions = [r["reduction_pp"] for r in cells]
    finalize_experiment(
        run_dir(),
        {"scenario": "sepsis_early_alert",
         "experiment_name": "baseline_sweep", **settings,
         "study_parameters": trews_raw},
        {"n_cells": len(cells),
         "min_reduction_pp": min(reductions),
         "max_reduction_pp": max(reductions),
         "cells_in_target_band": sum(low <= v <= high for v in reductions)},
        {"cells": cells},
        require_validated=True,
        validation_checks=[
            check("grid_complete", "Every capacity x delay cell ran",
                  len(cells) == len(max_hours_values) * len(capacity_values),
                  expected=len(max_hours_values) * len(capacity_values),
                  actual=len(cells)),
            check("deaths_within_population",
                  "Deaths on each branch are between 0 and the cohort size",
                  all(0 <= r[k] <= trews_raw["n_patients"] for r in cells
                      for k in ("f_deaths", "cf_deaths")),
                  expected=f"[0, {trews_raw['n_patients']}]"),
        ],
    )


if __name__ == "__main__":
    main()
