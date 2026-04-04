"""Hydra-powered threshold optimizer for Access Operations.

Each invocation runs ONE configuration. Hydra handles the grid sweep
via --multirun. Results are written to Hydra's output directory.

Single run:
    python run_threshold_optimizer.py \
        clinic.noshow_rate=0.13 clinic.utilization=90 \
        model.auc=0.83 model.threshold=0.30

Full grid (198 runs):
    python run_threshold_optimizer.py --multirun \
        clinic.noshow_rate=0.07,0.13,0.20 \
        clinic.utilization=80,90,110 \
        model.type=baseline,predictor \
        model.auc=0.65,0.75,0.83 \
        model.threshold=0.20,0.30,0.40,0.50,0.60,0.70,0.80

Predictor-only grid (189 runs):
    python run_threshold_optimizer.py --multirun \
        clinic.noshow_rate=0.07,0.13,0.20 \
        clinic.utilization=80,90,110 \
        model.auc=0.65,0.75,0.83 \
        model.threshold=0.20,0.30,0.40,0.50,0.60,0.70,0.80

Baseline only (9 runs):
    python run_threshold_optimizer.py --multirun \
        clinic.noshow_rate=0.07,0.13,0.20 \
        clinic.utilization=80,90,110 \
        model.type=baseline model.threshold=0.50
"""

import json
import logging
import time
from pathlib import Path

import hydra.core.hydra_config

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine, CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.performance import (
    auc_score, confusion_matrix_metrics,
)
from healthcare_sim_sdk.scenarios.noshow_overbooking.realistic_scenario import (
    ClinicConfig, RealisticNoShowScenario,
)

logger = logging.getLogger(__name__)

# Utilization -> operational params lookup
UTIL_PARAMS = {
    80: {"ob_cap": 3, "wl_day": 2},
    90: {"ob_cap": 2, "wl_day": 5},
    110: {"ob_cap": 1, "wl_day": 10},
}


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="threshold_optimizer",
)
def main(cfg: DictConfig) -> float:
    """Run one configuration and save results."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Resolve utilization-dependent params
    util = cfg.clinic.utilization
    up = UTIL_PARAMS.get(util, UTIL_PARAMS[90])

    cc = ClinicConfig(
        n_providers=cfg.clinic.n_providers,
        slots_per_provider_per_day=cfg.clinic.slots_per_provider,
        max_overbook_per_provider=up["ob_cap"],
        new_waitlist_requests_per_day=up["wl_day"],
    )
    tc = TimeConfig(
        n_timesteps=cfg.n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(cfg.n_days)),
    )

    # Baseline uses historical rate; threshold is the baseline's
    model_type = cfg.model.type
    threshold = cfg.model.threshold
    model_auc = cfg.model.auc
    if model_type == "baseline":
        threshold = 0.50  # staff uses 50% historical rate
        model_auc = 0.0

    sc = RealisticNoShowScenario(
        time_config=tc,
        seed=cfg.seed,
        n_patients=cfg.n_patients,
        base_noshow_rate=cfg.clinic.noshow_rate,
        model_type=model_type,
        model_auc=model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=(
            cfg.policy.max_individual_overbooks
        ),
        overbooking_policy="threshold",
        clinic_config=cc,
        ar1_rho=cfg.policy.ar1_rho,
        ar1_sigma=cfg.policy.ar1_sigma,
    )

    t0 = time.time()
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.NONE,
    ).run(cfg.n_patients)
    sim_time = time.time() - t0

    meta = results.outcomes[cfg.n_days - 1].metadata
    n_ob = meta["total_overbooked"]
    n_res = meta["total_resolved"]

    # -- Compute metrics --

    # Utilization
    uv = []
    for t in range(1, cfg.n_days):
        u = results.outcomes[t].secondary.get("utilization")
        if u is not None and len(u) > 0:
            uv.append(float(u.mean()))

    # AUC + classification metrics
    all_p, all_a = [], []
    for t in range(cfg.n_days):
        if t in results.predictions and (t + 1) < cfg.n_days:
            p = results.predictions[t].scores
            a = results.outcomes[t + 1].events
            if len(p) == len(a):
                all_p.append(p)
                all_a.append(a)
    pa = np.concatenate(all_p) if all_p else np.array([])
    ya = np.concatenate(all_a) if all_a else np.array([])

    auc = float(auc_score(ya, pa)) if len(pa) > 100 else 0
    cm = (
        confusion_matrix_metrics(ya, pa, threshold)
        if len(pa) > 100 else {}
    )

    # Burden (re-run for patient state)
    sc2 = RealisticNoShowScenario(
        time_config=tc, seed=cfg.seed,
        n_patients=cfg.n_patients,
        base_noshow_rate=cfg.clinic.noshow_rate,
        model_type=model_type, model_auc=model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=(
            cfg.policy.max_individual_overbooks
        ),
        overbooking_policy="threshold", clinic_config=cc,
        ar1_rho=cfg.policy.ar1_rho,
        ar1_sigma=cfg.policy.ar1_sigma,
    )
    st = sc2.create_population(cfg.n_patients)
    for t in range(cfg.n_days):
        st = sc2.step(st, t)
        pr = sc2.predict(st, t)
        st, _ = sc2.intervene(st, pr, t)
    max_burden = max(
        p.n_times_overbooked for p in st.patients.values()
    )

    # -- Build metrics dict --
    metrics = {
        "archetype": (
            f"NS{cfg.clinic.noshow_rate:.0%}"
            f"_Util{cfg.clinic.utilization}%"
        ),
        "noshow_config": cfg.clinic.noshow_rate,
        "util_level": cfg.clinic.utilization,
        "ob_cap": up["ob_cap"],
        "wl_pressure": up["wl_day"],
        "model_type": model_type,
        "auc_target": model_auc,
        "threshold": threshold,
        "auc_achieved": auc,
        "ppv": cm.get("ppv", 0),
        "sensitivity": cm.get("sensitivity", 0),
        "specificity": cm.get("specificity", 0),
        "flag_rate": cm.get("flag_rate", 0),
        "utilization": float(np.mean(uv)) if uv else 0,
        "collision_rate": (
            meta["total_collisions"] / max(n_ob, 1)
            if n_ob > 0 else 0
        ),
        "collisions": meta["total_collisions"],
        "overbooked": n_ob,
        "ob_per_week": n_ob / max(cfg.n_days / 7, 1),
        "waitlist": meta["waitlist_size"],
        "wl_served": meta["total_waitlist_served"],
        "avg_wait_days": meta["avg_wait_days"],
        "max_burden": max_burden,
        "mean_burden": meta["mean_overbooking_burden"],
        "noshow_observed": (
            meta["total_noshows"] / max(n_res, 1)
        ),
        "sim_seconds": sim_time,
    }

    # Save via lifecycle
    from healthcare_sim_sdk.experiments.lifecycle import (
        save_experiment, register_experiment,
    )
    out_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()
        .runtime.output_dir
    )
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict["timestamp"] = out_dir.name
    config_dict["scenario"] = "noshow_overbooking"
    config_dict["experiment_name"] = "threshold_optimizer"

    save_experiment(out_dir, config_dict, metrics)
    try:
        register_experiment(out_dir)
    except Exception as e:
        logger.warning("Catalog registration failed: %s", e)

    logger.info(
        "Result: util=%.1f%% coll=%.1f%% WL=%d burden=%d "
        "(%.1fs)",
        metrics["utilization"] * 100,
        metrics["collision_rate"] * 100,
        metrics["waitlist"],
        metrics["max_burden"],
        sim_time,
    )

    # Return utilization as the optimization target
    # (Hydra can use this for Optuna sweeps)
    return metrics["utilization"]


if __name__ == "__main__":
    main()
