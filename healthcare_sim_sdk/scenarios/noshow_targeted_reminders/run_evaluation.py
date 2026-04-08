"""Targeted reminder evaluation experiment.

Runs ML-targeted vs no-intervention (counterfactual) across
configurations matching Chong (2020) and Rosen (2023) findings.

Usage:
    python scenarios/noshow_targeted_reminders/run_evaluation.py

    # Override parameters:
    python scenarios/noshow_targeted_reminders/run_evaluation.py \
        --n-days 90 --model-auc 0.74 --base-noshow-rate 0.193
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from healthcare_sim_sdk.ml.performance import auc_score

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full experiment configuration (saved to output)."""
    experiment_name: str = "noshow_targeted_reminders_evaluation"
    timestamp: str = ""
    seed: int = 42
    n_patients: int = 5000
    n_days: int = 90
    base_noshow_rate: float = 0.36
    n_providers: int = 8
    slots_per_provider: int = 12
    model_auc: float = 0.72
    targeting_mode: str = "top_k"
    targeting_fraction: float = 0.25
    call_capacity_per_day: int = 24
    call_success_rate: float = 0.65
    reminder_effectiveness: float = 0.35
    effectiveness_sweep: Optional[List[float]] = None
    auc_sweep: Optional[List[float]] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.effectiveness_sweep is None:
            self.effectiveness_sweep = [
                0.20, 0.35, 0.50, 0.65, 0.80,
            ]
        if self.auc_sweep is None:
            self.auc_sweep = [0.55, 0.65, 0.74, 0.85]


def run_single(
    config: ExperimentConfig,
    model_type: str = "predictor",
    model_auc: Optional[float] = None,
    reminder_effectiveness: Optional[float] = None,
) -> Dict[str, Any]:
    """Run one scenario configuration and return metrics."""
    auc = model_auc if model_auc is not None else config.model_auc
    eff = (
        reminder_effectiveness if reminder_effectiveness is not None
        else config.reminder_effectiveness
    )

    cc = CallerConfig(
        n_providers=config.n_providers,
        slots_per_provider_per_day=config.slots_per_provider,
        call_capacity_per_day=config.call_capacity_per_day,
        call_success_rate=config.call_success_rate,
        reminder_effectiveness=eff,
    )
    tc = TimeConfig(
        n_timesteps=config.n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(config.n_days)),
    )
    sc = NoShowTargetedReminderScenario(
        time_config=tc,
        seed=config.seed,
        n_patients=config.n_patients,
        base_noshow_rate=config.base_noshow_rate,
        model_type=model_type,
        model_auc=auc,
        targeting_mode=config.targeting_mode,
        targeting_fraction=config.targeting_fraction,
        caller_config=cc,
    )

    t0 = time.time()
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED,
    ).run(config.n_patients)
    elapsed = time.time() - t0

    meta = results.outcomes[config.n_days - 1].metadata

    # Compute no-show rates on both branches
    f_noshows = sum(
        results.outcomes[t].events.sum()
        for t in range(config.n_days)
    )
    f_total = sum(
        len(results.outcomes[t].events)
        for t in range(config.n_days)
    )
    cf_noshows = sum(
        results.counterfactual_outcomes[t].events.sum()
        for t in range(config.n_days)
    )
    cf_total = sum(
        len(results.counterfactual_outcomes[t].events)
        for t in range(config.n_days)
    )

    f_rate = f_noshows / max(f_total, 1)
    cf_rate = cf_noshows / max(cf_total, 1)

    # Compute no-show rates by race on both branches
    race_stats: Dict[str, Dict[str, float]] = {}
    for t in range(config.n_days):
        for branch, outcomes in [
            ("factual", results.outcomes[t]),
            ("counterfactual", results.counterfactual_outcomes[t]),
        ]:
            races = outcomes.secondary.get("race_ethnicity", np.array([]))
            events = outcomes.events
            for race in np.unique(races):
                mask = races == race
                key = f"{branch}_{race}"
                if key not in race_stats:
                    race_stats[key] = {"noshows": 0.0, "total": 0}
                race_stats[key]["noshows"] += events[mask].sum()
                race_stats[key]["total"] += mask.sum()

    race_rates = {
        k: v["noshows"] / max(v["total"], 1)
        for k, v in race_stats.items()
    }

    # AUC: predictions at t vs outcomes at t+1
    all_pred, all_act = [], []
    for t in range(config.n_days):
        if t in results.predictions and (t + 1) < config.n_days:
            p = results.predictions[t].scores
            a = results.outcomes[t + 1].events
            if len(p) == len(a):
                all_pred.append(p)
                all_act.append(a)

    pred_arr = np.concatenate(all_pred) if all_pred else np.array([])
    act_arr = np.concatenate(all_act) if all_act else np.array([])
    achieved_auc = (
        float(auc_score(act_arr, pred_arr))
        if len(pred_arr) > 0 else 0
    )

    return {
        "model_type": model_type,
        "model_auc_target": auc,
        "model_auc_achieved": achieved_auc,
        "reminder_effectiveness": eff,
        "factual_noshow_rate": float(f_rate),
        "counterfactual_noshow_rate": float(cf_rate),
        "absolute_reduction_pp": float((cf_rate - f_rate) * 100),
        "relative_reduction_pct": float(
            (cf_rate - f_rate) / max(cf_rate, 0.001) * 100
        ),
        "total_calls_made": meta["total_calls_made"],
        "total_calls_reached": meta["total_calls_reached"],
        "reach_rate": (
            meta["total_calls_reached"] / max(meta["total_calls_made"], 1)
        ),
        "total_resolved": meta["total_resolved"],
        "race_noshow_rates": race_rates,
        "elapsed_seconds": elapsed,
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run full experiment: control + effectiveness sweep + AUC sweep."""
    all_results = []

    # Control: no intervention (baseline model, zero effectiveness)
    logger.info("Running control (no intervention)...")
    control = run_single(
        config, model_type="baseline", reminder_effectiveness=0.0,
    )
    control["label"] = "no_intervention"
    all_results.append(control)

    # Effectiveness sweep
    for eff in (config.effectiveness_sweep or []):
        logger.info(
            "Running effectiveness=%.2f (AUC=%.2f)...",
            eff, config.model_auc,
        )
        result = run_single(
            config, reminder_effectiveness=eff,
        )
        result["label"] = f"eff_{eff:.2f}"
        all_results.append(result)

    # AUC sweep
    for auc_val in (config.auc_sweep or []):
        logger.info(
            "Running AUC=%.2f (effectiveness=%.2f)...",
            auc_val, config.reminder_effectiveness,
        )
        result = run_single(config, model_auc=auc_val)
        result["label"] = f"auc_{auc_val:.2f}"
        all_results.append(result)

    return {
        "config": asdict(config),
        "results": all_results,
        "summary": _build_summary(config, all_results),
    }


def _build_summary(
    config: ExperimentConfig, results: List[Dict],
) -> Dict[str, Any]:
    """Build human-readable summary."""
    control = next(
        r for r in results if r["label"] == "no_intervention"
    )
    eff_results = [
        r for r in results if r["label"].startswith("eff_")
    ]
    auc_results = [
        r for r in results if r["label"].startswith("auc_")
    ]

    best_eff = max(
        eff_results,
        key=lambda r: r["absolute_reduction_pp"],
    ) if eff_results else None

    return {
        "control_noshow_rate": control["factual_noshow_rate"],
        "best_effectiveness": (
            best_eff["reminder_effectiveness"] if best_eff else None
        ),
        "best_reduction_pp": (
            best_eff["absolute_reduction_pp"] if best_eff else None
        ),
        "n_effectiveness_configs": len(eff_results),
        "n_auc_configs": len(auc_results),
    }


def save_results(experiment: Dict, output_dir: Path):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(experiment["config"], f, indent=2)

    with open(output_dir / "results.json", "w") as f:
        json.dump(experiment["results"], f, indent=2, default=str)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(experiment["summary"], f, indent=2)

    csv_path = output_dir / "results.csv"
    fieldnames = [
        "label", "model_type", "model_auc_target",
        "model_auc_achieved", "reminder_effectiveness",
        "factual_noshow_rate", "counterfactual_noshow_rate",
        "absolute_reduction_pp", "relative_reduction_pct",
        "total_calls_made", "total_calls_reached",
        "reach_rate", "elapsed_seconds",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, extrasaction="ignore",
        )
        writer.writeheader()
        for r in experiment["results"]:
            writer.writerow(r)

    logger.info("Results saved to %s", output_dir)


def print_report(experiment: Dict):
    """Print formatted report to console."""
    results = experiment["results"]
    config = experiment["config"]

    print()
    print("=" * 80)
    print("NO-SHOW TARGETED REMINDERS EVALUATION")
    print("=" * 80)
    print(f"Date: {config['timestamp']}")
    print(f"Patients: {config['n_patients']}, "
          f"Days: {config['n_days']}, "
          f"Seed: {config['seed']}")
    print(f"Base no-show rate: {config['base_noshow_rate']:.1%}")
    print(f"Call capacity: {config['call_capacity_per_day']}/day, "
          f"success rate: {config['call_success_rate']:.0%}")
    print()

    control = next(
        r for r in results if r["label"] == "no_intervention"
    )
    print(f"CONTROL (no calling): "
          f"noshow rate={control['factual_noshow_rate']:.1%}")
    print()

    print(f"{'Label':>12s} {'AUC':>6s} {'Eff':>5s} "
          f"{'F-rate':>7s} {'CF-rate':>7s} "
          f"{'Abs pp':>7s} {'Rel %':>6s}")
    print("-" * 60)
    for r in results:
        if r["label"] == "no_intervention":
            continue
        print(f"{r['label']:>12s} "
              f"{r['model_auc_achieved']:>6.3f} "
              f"{r['reminder_effectiveness']:>5.2f} "
              f"{r['factual_noshow_rate']:>6.1%} "
              f"{r['counterfactual_noshow_rate']:>6.1%} "
              f"{r['absolute_reduction_pp']:>6.1f} "
              f"{r['relative_reduction_pct']:>5.1f}")

    # Equity summary
    print()
    print("EQUITY SUMMARY (by race, best effectiveness):")
    eff_results = [
        r for r in results if r["label"].startswith("eff_")
    ]
    if eff_results:
        best = max(eff_results, key=lambda r: r["absolute_reduction_pp"])
        rr = best.get("race_noshow_rates", {})
        for race in ["White", "Black", "Hispanic"]:
            f_key = f"factual_{race}"
            cf_key = f"counterfactual_{race}"
            if f_key in rr and cf_key in rr:
                print(f"  {race:>10s}: "
                      f"factual={rr[f_key]:.1%}, "
                      f"CF={rr[cf_key]:.1%}, "
                      f"reduction={((rr[cf_key] - rr[f_key]) * 100):.1f}pp")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run targeted reminder evaluation",
    )
    parser.add_argument("--n-patients", type=int, default=5000)
    parser.add_argument("--n-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-auc", type=float, default=0.72)
    parser.add_argument(
        "--base-noshow-rate", type=float, default=0.36,
    )
    parser.add_argument(
        "--targeting-mode", type=str, default="top_k",
        choices=["top_k", "top_fraction", "threshold"],
    )
    parser.add_argument(
        "--call-capacity", type=int, default=24,
    )
    parser.add_argument(
        "--call-success-rate", type=float, default=0.65,
    )
    parser.add_argument(
        "--reminder-effectiveness", type=float, default=0.35,
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = ExperimentConfig(
        seed=args.seed,
        n_patients=args.n_patients,
        n_days=args.n_days,
        model_auc=args.model_auc,
        base_noshow_rate=args.base_noshow_rate,
        targeting_mode=args.targeting_mode,
        call_capacity_per_day=args.call_capacity,
        call_success_rate=args.call_success_rate,
        reminder_effectiveness=args.reminder_effectiveness,
    )

    output_dir = Path(args.output_dir) / (
        f"{config.experiment_name}_{config.timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("Starting experiment: %s", config.experiment_name)

    t0 = time.time()
    experiment = run_experiment(config)
    total_time = time.time() - t0
    logger.info("Experiment complete in %.1f seconds", total_time)

    save_results(experiment, output_dir)
    print_report(experiment)
    print(f"\nResults saved to: {output_dir}")
    return experiment


if __name__ == "__main__":
    main()
