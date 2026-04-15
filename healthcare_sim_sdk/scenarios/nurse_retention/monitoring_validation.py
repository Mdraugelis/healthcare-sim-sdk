"""Monitoring validation runner — 6-regime × 30-seed sweep.

Executes each monitoring regime, runs the scenario through the
BranchedSimulationEngine, then replays the results through the
MonitoringHarness to produce a MonitoringRun for each (regime, seed)
cell. Saves outputs via the standard experiment lifecycle.

This is the first Hydra-based runner in the repo. It uses
@hydra.main for config composition with joblib parallelism for
overnight sweeps.

Usage:

    # Single run (default regime + seed)
    python monitoring_validation.py

    # Specific regime and seed
    python monitoring_validation.py regime=calibrated_success seed=1000

    # Full sweep (6 regimes × 30 seeds, parallelized)
    python monitoring_validation.py --multirun \\
        regime=calibrated_success,null_program,gradual_decay,\\
capacity_collapse,model_drift,partial_adoption \\
        seed='range(1000,1030)'

All configs in configs/monitoring/ relative to this file.
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.experiments.lifecycle import finalize_experiment
from healthcare_sim_sdk.scenarios.nurse_retention.monitoring.harness import (
    MonitoringHarness,
)
from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
    NurseRetentionScenario,
    RetentionConfig,
)
from healthcare_sim_sdk.scenarios.nurse_retention.time_varying import (
    TimeVaryingParameter,
)

logger = logging.getLogger(__name__)


def _resolve_tvp(raw) -> Any:
    """Convert a YAML dict to a TimeVaryingParameter or pass float.

    YAML format for time-varying parameters:

        intervention_effectiveness:
          base: 0.50
          change_points:
            - [26, 0.5]
            - [52, 0.0]
          interpolation: linear

    If the value is a plain number, returns that number as a float.
    """
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, DictConfig):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, dict) and "base" in raw:
        cps = raw.get("change_points", [])
        cps_tuple = tuple(tuple(cp) for cp in cps)
        return TimeVaryingParameter(
            base=float(raw["base"]),
            change_points=cps_tuple,
            interpolation=raw.get("interpolation", "step"),
        )
    return float(raw)


def _build_retention_config(cfg: DictConfig) -> RetentionConfig:
    """Construct a RetentionConfig from the Hydra config object."""
    pop = cfg.get("population", {})
    model = cfg.get("model", {})
    policy = cfg.get("policy", {})

    return RetentionConfig(
        n_nurses=int(pop.get("n_nurses", 1000)),
        nurses_per_manager=int(pop.get("nurses_per_manager", 100)),
        annual_turnover_rate=float(
            pop.get("annual_turnover_rate", 0.22),
        ),
        risk_concentration=float(
            pop.get("risk_concentration", 0.5),
        ),
        new_hire_fraction=float(
            pop.get("new_hire_fraction", 0.15),
        ),
        new_hire_risk_multiplier=float(
            pop.get("new_hire_risk_multiplier", 2.0),
        ),
        high_span_turnover_penalty=float(
            pop.get("high_span_turnover_penalty", 0.10),
        ),
        n_weeks=int(cfg.get("n_weeks", 104)),
        ar1_rho=float(cfg.get("ar1_rho", 0.95)),
        ar1_sigma=float(cfg.get("ar1_sigma", 0.04)),
        prediction_interval=int(
            cfg.get("prediction_interval", 2),
        ),
        model_auc=_resolve_tvp(model.get("auc", 0.80)),
        max_interventions_per_manager_per_week=_resolve_tvp(
            policy.get("max_interventions_per_week", 6),
        ),
        intervention_effectiveness=_resolve_tvp(
            policy.get("effectiveness", 0.50),
        ),
        intervention_decay_halflife_weeks=float(
            policy.get("decay_halflife_weeks", 6.0),
        ),
        cooldown_weeks=int(policy.get("cooldown_weeks", 4)),
        manager_adherence_rate=float(
            policy.get("manager_adherence_rate", 1.0),
        ),
    )


def run_one(cfg: DictConfig) -> Dict[str, Any]:
    """Run one (regime, seed) cell and return metrics + output path."""
    regime = str(cfg.get("regime", "unnamed"))
    seed = int(cfg.get("seed", 42))
    tier3_mode = str(cfg.get("tier3_mode", "cits_with_cf"))

    logger.info(
        "Starting regime=%s seed=%d", regime, seed,
    )
    t0 = time.time()

    # Build scenario config from Hydra config
    rc = _build_retention_config(cfg)
    scenario = NurseRetentionScenario(config=rc, seed=seed)

    # Run the simulation
    results = BranchedSimulationEngine(
        scenario, CounterfactualMode.BRANCHED,
    ).run(rc.n_nurses)

    sim_time = time.time() - t0

    # Run the monitoring harness over the completed results
    t1 = time.time()
    harness = MonitoringHarness(
        regime=regime,
        seed=seed,
        tier3_mode=tier3_mode,
    )
    mon_run = harness.run_from_results(results, rc)
    harness_time = time.time() - t1

    # Extract key metrics for the lifecycle pipeline
    final_t = rc.n_weeks - 1
    f_meta = results.outcomes[final_t].metadata
    cf_meta = results.counterfactual_outcomes[final_t].metadata

    f_departures = f_meta["total_departures"]
    cf_departures = cf_meta["total_departures"]

    # Tier summary
    tier1_events = mon_run.events_by_tier(1)
    tier2_events = mon_run.events_by_tier(2)
    tier3_events = mon_run.events_by_tier(3)
    tier4_events = mon_run.events_by_tier(4)

    # Tier 3 latest estimate
    tier3_latest = (
        mon_run.tier3_estimates[-1].to_dict()
        if mon_run.tier3_estimates
        else None
    )

    # First detection per tier (if any)
    first_t1 = mon_run.first_detection(1)
    first_t2 = mon_run.first_detection(2)
    first_t3 = mon_run.first_detection(3)
    first_t4 = mon_run.first_detection(4)

    metrics = {
        "regime": regime,
        "seed": seed,
        "n_weeks": rc.n_weeks,
        "n_nurses": rc.n_nurses,
        "factual_departures": f_departures,
        "counterfactual_departures": cf_departures,
        "departures_prevented": cf_departures - f_departures,
        "factual_retention_rate": f_meta["retention_rate"],
        "cf_retention_rate": cf_meta["retention_rate"],
        "sim_seconds": round(sim_time, 1),
        "harness_seconds": round(harness_time, 1),
        "tier1_n_events": len(tier1_events),
        "tier2_n_events": len(tier2_events),
        "tier3_n_events": len(tier3_events),
        "tier4_n_events": len(tier4_events),
        "tier1_first_detection_week": (
            first_t1.week if first_t1 else None
        ),
        "tier2_first_detection_week": (
            first_t2.week if first_t2 else None
        ),
        "tier3_first_detection_week": (
            first_t3.week if first_t3 else None
        ),
        "tier4_first_detection_week": (
            first_t4.week if first_t4 else None
        ),
        "tier3_latest_estimate": tier3_latest,
    }

    # Save the full MonitoringRun as a separate JSON
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    run_path = output_dir / "monitoring_run.json"
    with open(run_path, "w") as f:
        json.dump(mon_run.to_dict(), f, indent=2, default=str)

    # Save ground truth numpy arrays for analysis
    np.savez_compressed(
        output_dir / "ground_truth.npz",
        factual_departures=mon_run.ground_truth_factual_departures,
        cf_departures=mon_run.ground_truth_counterfactual_departures,
        factual_retention=mon_run.ground_truth_factual_retention,
        cf_retention=mon_run.ground_truth_counterfactual_retention,
    )

    # Lifecycle
    config_dict = {
        "scenario": "nurse_retention_monitoring",
        "experiment_name": f"monitoring_{regime}",
        "regime": regime,
        "seed": seed,
        "n_nurses": rc.n_nurses,
        "n_weeks": rc.n_weeks,
        "tier3_mode": tier3_mode,
    }
    try:
        finalize_experiment(output_dir, config_dict, metrics)
    except Exception as exc:
        logger.warning("finalize_experiment failed: %s", exc)

    total_time = time.time() - t0
    logger.info(
        "Completed regime=%s seed=%d in %.1fs "
        "(sim=%.1fs, harness=%.1fs) "
        "departures: F=%d CF=%d saved=%d "
        "events: T1=%d T2=%d T3=%d T4=%d",
        regime, seed, total_time,
        sim_time, harness_time,
        f_departures, cf_departures,
        cf_departures - f_departures,
        len(tier1_events), len(tier2_events),
        len(tier3_events), len(tier4_events),
    )

    return metrics


@hydra.main(
    version_base=None,
    config_path="configs/monitoring",
    config_name="base",
)
def main(cfg: DictConfig) -> None:
    """Entry point for Hydra-driven monitoring validation runs."""
    logger.info(
        "Config:\n%s", OmegaConf.to_yaml(cfg),
    )
    metrics = run_one(cfg)

    # Print summary to console
    print(
        f"\n{'='*60}\n"
        f"REGIME: {metrics['regime']}  SEED: {metrics['seed']}\n"
        f"{'='*60}\n"
        f"Departures: F={metrics['factual_departures']}  "
        f"CF={metrics['counterfactual_departures']}  "
        f"saved={metrics['departures_prevented']}\n"
        f"Retention:  F={metrics['factual_retention_rate']:.1%}  "
        f"CF={metrics['cf_retention_rate']:.1%}\n"
        f"Tiers:  T1={metrics['tier1_n_events']}  "
        f"T2={metrics['tier2_n_events']}  "
        f"T3={metrics['tier3_n_events']}  "
        f"T4={metrics['tier4_n_events']}\n"
        f"First detect:  T1@wk{metrics['tier1_first_detection_week']}  "
        f"T2@wk{metrics['tier2_first_detection_week']}  "
        f"T3@wk{metrics['tier3_first_detection_week']}  "
        f"T4@wk{metrics['tier4_first_detection_week']}\n"
        f"Time: {metrics['sim_seconds']:.1f}s sim + "
        f"{metrics['harness_seconds']:.1f}s harness\n"
    )


if __name__ == "__main__":
    main()
