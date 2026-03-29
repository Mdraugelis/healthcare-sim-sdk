"""Structured no-show overbooking evaluation experiment.

Runs baseline vs ML predictor across threshold sweep, logs all
results to experiments/outputs/ with JSON summaries, CSV data,
and configuration tracking.

Usage:
    python experiments/run_noshow_evaluation.py

    # Override parameters:
    python experiments/run_noshow_evaluation.py --n-days 90 --model-auc 0.87
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.noshow_overbooking.realistic_scenario import (  # noqa: E402
    ClinicConfig,
    RealisticNoShowScenario,
)
from sdk.core.engine import (  # noqa: E402
    BranchedSimulationEngine, CounterfactualMode,
)
from sdk.core.scenario import TimeConfig  # noqa: E402
from sdk.ml.performance import (  # noqa: E402
    auc_score, confusion_matrix_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full experiment configuration (saved to output)."""
    experiment_name: str = "noshow_overbooking_evaluation"
    timestamp: str = ""
    seed: int = 42
    n_patients: int = 2000
    n_days: int = 60
    base_noshow_rate: float = 0.13
    n_providers: int = 6
    slots_per_provider: int = 12
    max_overbook_per_provider: int = 2
    new_waitlist_requests_per_day: int = 5
    model_auc: float = 0.83
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.04
    baseline_threshold: float = 0.50
    predictor_thresholds: List[float] = None
    max_individual_overbooks: int = 10

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.predictor_thresholds is None:
            self.predictor_thresholds = [
                0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
            ]


def run_single(config: ExperimentConfig, model_type: str,
               threshold: float) -> Dict[str, Any]:
    """Run one scenario configuration and return metrics."""
    cc = ClinicConfig(
        n_providers=config.n_providers,
        slots_per_provider_per_day=config.slots_per_provider,
        max_overbook_per_provider=config.max_overbook_per_provider,
        new_waitlist_requests_per_day=(
            config.new_waitlist_requests_per_day
        ),
    )
    tc = TimeConfig(
        n_timesteps=config.n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(config.n_days)),
    )
    sc = RealisticNoShowScenario(
        time_config=tc,
        seed=config.seed,
        n_patients=config.n_patients,
        base_noshow_rate=config.base_noshow_rate,
        model_type=model_type,
        model_auc=config.model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=config.max_individual_overbooks,
        clinic_config=cc,
        ar1_rho=config.ar1_rho,
        ar1_sigma=config.ar1_sigma,
    )

    t0 = time.time()
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED,
    ).run(config.n_patients)
    elapsed = time.time() - t0

    meta = results.outcomes[config.n_days - 1].metadata
    n_resolved = meta["total_resolved"]
    n_ob = meta["total_overbooked"]

    # Utilization from resolved slots
    util_vals = []
    for t in range(1, config.n_days):
        u = results.outcomes[t].secondary["utilization"]
        if len(u) > 0:
            util_vals.append(float(u.mean()))

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
    auc = float(auc_score(act_arr, pred_arr)) if len(pred_arr) > 0 else 0

    # Threshold-level metrics
    threshold_metrics = {}
    if len(pred_arr) > 0:
        for t_check in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            m = confusion_matrix_metrics(act_arr, pred_arr, t_check)
            threshold_metrics[f"at_{t_check:.2f}"] = {
                k: (
                    float(v) if isinstance(
                        v, (int, float, np.integer, np.floating)
                    ) else v
                )
                for k, v in m.items()
            }

    return {
        "model_type": model_type,
        "overbooking_threshold": threshold,
        "auc": auc,
        "noshow_rate": meta["total_noshows"] / max(n_resolved, 1),
        "utilization": float(np.mean(util_vals)) if util_vals else 0,
        "total_overbooked": n_ob,
        "overbookings_per_week": n_ob / max(config.n_days / 7, 1),
        "collision_rate": (
            meta["total_collisions"] / max(n_ob, 1)
            if n_ob > 0 else 0
        ),
        "total_collisions": meta["total_collisions"],
        "overbooked_show_rate": (
            meta["total_overbooked_showed"] / max(n_ob, 1)
            if n_ob > 0 else 0
        ),
        "waitlist_size": meta["waitlist_size"],
        "total_waitlist_served": meta["total_waitlist_served"],
        "avg_wait_days": meta["avg_wait_days"],
        "mean_overbooking_burden": meta["mean_overbooking_burden"],
        "total_resolved": n_resolved,
        "elapsed_seconds": elapsed,
        "threshold_metrics": threshold_metrics,
    }


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """Run full experiment: control + baseline + predictor sweep."""
    all_results = []

    # Control: no overbooking
    logger.info("Running control (no overbooking)...")
    control = run_single(config, "baseline", 1.0)
    control["label"] = "no_overbooking"
    all_results.append(control)

    # Baseline: staff historical rate
    logger.info(
        "Running baseline (historical rate >= %.2f)...",
        config.baseline_threshold,
    )
    baseline = run_single(
        config, "baseline", config.baseline_threshold,
    )
    baseline["label"] = "baseline"
    all_results.append(baseline)

    # Predictor sweep
    for thresh in config.predictor_thresholds:
        logger.info(
            "Running predictor (AUC=%.2f, threshold=%.2f)...",
            config.model_auc, thresh,
        )
        pred = run_single(config, "predictor", thresh)
        pred["label"] = f"predictor_t{thresh:.2f}"
        all_results.append(pred)

    return {
        "config": asdict(config),
        "results": all_results,
        "summary": _build_summary(config, all_results),
    }


def _build_summary(config, results):
    """Build human-readable summary."""
    control = next(r for r in results if r["label"] == "no_overbooking")
    baseline = next(r for r in results if r["label"] == "baseline")
    predictors = [
        r for r in results if r["label"].startswith("predictor")
    ]

    # Find predictor at similar utilization to baseline
    best_match = min(
        predictors,
        key=lambda r: abs(r["utilization"] - baseline["utilization"]),
    ) if predictors else None

    return {
        "control_utilization": control["utilization"],
        "control_waitlist": control["waitlist_size"],
        "baseline_auc": baseline["auc"],
        "baseline_utilization": baseline["utilization"],
        "baseline_collision_rate": baseline["collision_rate"],
        "baseline_waitlist": baseline["waitlist_size"],
        "best_predictor_threshold": (
            best_match["overbooking_threshold"] if best_match else None
        ),
        "best_predictor_auc": (
            best_match["auc"] if best_match else None
        ),
        "best_predictor_collision_rate": (
            best_match["collision_rate"] if best_match else None
        ),
        "collision_rate_reduction": (
            (baseline["collision_rate"] - best_match["collision_rate"])
            / max(baseline["collision_rate"], 0.001)
            if best_match else None
        ),
    }


def save_results(experiment: Dict, output_dir: Path):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with open(output_dir / "config.json", "w") as f:
        json.dump(experiment["config"], f, indent=2)

    # Full results
    with open(output_dir / "results.json", "w") as f:
        json.dump(experiment["results"], f, indent=2, default=str)

    # Summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(experiment["summary"], f, indent=2)

    # CSV for easy analysis
    import csv
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "label", "model_type", "overbooking_threshold", "auc",
        "utilization", "collision_rate", "overbookings_per_week",
        "waitlist_size", "total_waitlist_served", "avg_wait_days",
        "mean_overbooking_burden", "noshow_rate", "total_collisions",
        "total_overbooked", "elapsed_seconds",
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
    summary = experiment["summary"]
    config = experiment["config"]

    print()
    print("=" * 80)
    print("NO-SHOW OVERBOOKING EVALUATION")
    print("=" * 80)
    print(f"Date: {config['timestamp']}")
    print(f"Patients: {config['n_patients']}, "
          f"Days: {config['n_days']}, "
          f"Seed: {config['seed']}")
    print(f"Clinic: {config['n_providers']} providers x "
          f"{config['slots_per_provider']} slots = "
          f"{config['n_providers'] * config['slots_per_provider']} "
          f"daily capacity")
    print(f"AR(1) drift: rho={config['ar1_rho']}, "
          f"sigma={config['ar1_sigma']}")
    print()

    control = next(
        r for r in results if r["label"] == "no_overbooking"
    )
    baseline = next(r for r in results if r["label"] == "baseline")

    print(f"CONTROL (no overbooking): "
          f"util={control['utilization']:.1%}, "
          f"waitlist={control['waitlist_size']}")
    print()

    print(f"BASELINE (hist rate >= "
          f"{config['baseline_threshold']:.0%}): "
          f"AUC={baseline['auc']:.3f}, "
          f"util={baseline['utilization']:.1%}, "
          f"collision={baseline['collision_rate']:.1%}, "
          f"OB/wk={baseline['overbookings_per_week']:.1f}, "
          f"waitlist={baseline['waitlist_size']}")
    print()

    print(f"{'Thresh':>6s} {'AUC':>6s} {'Util':>6s} "
          f"{'Collis%':>8s} {'OB/wk':>6s} {'OBshow%':>8s} "
          f"{'WL':>4s} {'Served':>6s} {'Wait':>5s}")
    print("-" * 65)
    for r in results:
        if not r["label"].startswith("predictor"):
            continue
        print(f"{r['overbooking_threshold']:>6.2f} "
              f"{r['auc']:>6.3f} "
              f"{r['utilization']:>5.1%} "
              f"{r['collision_rate']:>7.1%} "
              f"{r['overbookings_per_week']:>6.1f} "
              f"{r['overbooked_show_rate']:>7.1%} "
              f"{r['waitlist_size']:>4d} "
              f"{r['total_waitlist_served']:>6d} "
              f"{r['avg_wait_days']:>5.1f}")

    print()
    if summary.get("collision_rate_reduction"):
        print(
            f"At similar utilization, predictor reduces "
            f"collision rate by "
            f"{summary['collision_rate_reduction']:.0%} "
            f"(threshold={summary['best_predictor_threshold']:.2f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run no-show overbooking evaluation",
    )
    parser.add_argument("--n-patients", type=int, default=2000)
    parser.add_argument("--n-days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-auc", type=float, default=0.83)
    parser.add_argument("--base-noshow-rate", type=float, default=0.13)
    parser.add_argument("--ar1-rho", type=float, default=0.95)
    parser.add_argument("--ar1-sigma", type=float, default=0.04)
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/outputs",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    config = ExperimentConfig(
        seed=args.seed,
        n_patients=args.n_patients,
        n_days=args.n_days,
        model_auc=args.model_auc,
        base_noshow_rate=args.base_noshow_rate,
        ar1_rho=args.ar1_rho,
        ar1_sigma=args.ar1_sigma,
    )

    output_dir = Path(args.output_dir) / (
        f"{config.experiment_name}_{config.timestamp}"
    )

    # Add file logging
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("Starting experiment: %s", config.experiment_name)
    logger.info("Config: %s", json.dumps(asdict(config), indent=2))

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
