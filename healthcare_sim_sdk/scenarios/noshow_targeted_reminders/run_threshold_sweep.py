"""Threshold sweep for targeted reminders — UCSF benchmark comparison.

Sweeps probability thresholds for the reminder-call decision and
computes classification metrics (flag rate, TPR, PPV) at each
threshold, enabling direct comparison to UCSF Health's reported
operating points (UserWeb, 2018).

UCSF chose a 10% threshold over Epic's 15% "High Risk" cutoff
because several specialties had too few patients above 15%.
At 20% threshold, UCSF reported ~12% flag rate and ~60% TPR.

Usage:
    # Single run
    python run_threshold_sweep.py

    # Full grid (54 cells):
    python run_threshold_sweep.py --multirun \
        population.base_noshow_rate=0.09,0.13,0.18 \
        model.auc=0.72,0.80,0.85 \
        threshold=0.05,0.10,0.15,0.20,0.25,0.30

    # Collect results after sweep:
    python run_threshold_sweep.py --collect-sweep <sweep_dir>
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.performance import (
    auc_score,
    confusion_matrix_metrics,
)
from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
    CallerConfig,
    NoShowTargetedReminderScenario,
)

logger = logging.getLogger(__name__)

# UCSF benchmark reference (at 20% threshold, UserWeb 2018)
UCSF_BENCHMARK = {
    "threshold": 0.20,
    "flag_rate": 0.121,
    "tpr": 0.595,
    "total_appointments": 808_405,
    "implied_base_noshow_rate": 0.089,
    "source": "UCSF Health UserWeb, 2018",
}


def run_threshold_cell(
    seed: int = 42,
    n_patients: int = 5000,
    n_days: int = 90,
    base_noshow_rate: float = 0.09,
    noshow_concentration: float = 0.3,
    model_auc: float = 0.80,
    threshold: float = 0.10,
    call_capacity_per_day: int = 48,
    call_success_rate: float = 0.65,
    reminder_effectiveness: float = 0.35,
    n_providers: int = 8,
    slots_per_provider: int = 12,
) -> Dict[str, Any]:
    """Run one threshold configuration and return metrics.

    Uses targeting_mode='threshold' so the call list is determined
    by the probability cutoff, not by capacity. Capacity is set high
    to avoid being the binding constraint.
    """
    cc = CallerConfig(
        n_providers=n_providers,
        slots_per_provider_per_day=slots_per_provider,
        call_capacity_per_day=call_capacity_per_day,
        call_success_rate=call_success_rate,
        reminder_effectiveness=reminder_effectiveness,
    )
    tc = TimeConfig(
        n_timesteps=n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(n_days)),
    )
    sc = NoShowTargetedReminderScenario(
        time_config=tc,
        seed=seed,
        n_patients=n_patients,
        base_noshow_rate=base_noshow_rate,
        noshow_concentration=noshow_concentration,
        model_type="predictor",
        model_auc=model_auc,
        targeting_mode="threshold",
        targeting_threshold=threshold,
        caller_config=cc,
    )

    t0 = time.time()
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED,
    ).run(n_patients)
    elapsed = time.time() - t0

    meta = results.outcomes[n_days - 1].metadata

    # -- Aggregate predictions vs outcomes for classification metrics --
    all_pred, all_actual = [], []
    for t in range(n_days):
        if t in results.predictions and (t + 1) < n_days:
            p = results.predictions[t].scores
            a = results.outcomes[t + 1].events
            if len(p) == len(a):
                all_pred.append(p)
                all_actual.append(a)

    pred_arr = np.concatenate(all_pred) if all_pred else np.array([])
    actual_arr = np.concatenate(all_actual) if all_actual else np.array([])

    achieved_auc = (
        float(auc_score(actual_arr, pred_arr))
        if len(pred_arr) > 100 else 0.0
    )

    # Classification metrics at this threshold
    cm = (
        confusion_matrix_metrics(actual_arr, pred_arr, threshold)
        if len(pred_arr) > 100 else {}
    )

    # -- No-show rates on both branches --
    f_noshows = sum(
        results.outcomes[t].events.sum() for t in range(n_days)
    )
    f_total = sum(
        len(results.outcomes[t].events) for t in range(n_days)
    )
    cf_noshows = sum(
        results.counterfactual_outcomes[t].events.sum()
        for t in range(n_days)
    )
    cf_total = sum(
        len(results.counterfactual_outcomes[t].events)
        for t in range(n_days)
    )

    f_rate = f_noshows / max(f_total, 1)
    cf_rate = cf_noshows / max(cf_total, 1)

    # -- Calling stats --
    total_called = meta["total_calls_made"]
    total_reached = meta["total_calls_reached"]
    total_resolved = meta["total_resolved"]

    # Calls per day
    calls_per_day = total_called / max(n_days, 1)

    return {
        "threshold": threshold,
        "base_noshow_rate": base_noshow_rate,
        "model_auc_target": model_auc,
        "model_auc_achieved": achieved_auc,
        # Classification metrics (model performance at threshold)
        "flag_rate": cm.get("flag_rate", 0),
        "sensitivity": cm.get("sensitivity", 0),  # TPR
        "specificity": cm.get("specificity", 0),
        "ppv": cm.get("ppv", 0),
        "npv": cm.get("npv", 0),
        "f1": cm.get("f1", 0),
        "tp": cm.get("tp", 0),
        "fp": cm.get("fp", 0),
        "tn": cm.get("tn", 0),
        "fn": cm.get("fn", 0),
        # Intervention outcomes
        "factual_noshow_rate": float(f_rate),
        "counterfactual_noshow_rate": float(cf_rate),
        "absolute_reduction_pp": float((cf_rate - f_rate) * 100),
        "relative_reduction_pct": float(
            (cf_rate - f_rate) / max(cf_rate, 0.001) * 100
        ),
        # Operational
        "total_calls_made": total_called,
        "total_calls_reached": total_reached,
        "calls_per_day": calls_per_day,
        "total_resolved": total_resolved,
        "total_appointments": int(f_total),
        # Config
        "n_patients": n_patients,
        "n_days": n_days,
        "call_capacity_per_day": call_capacity_per_day,
        "call_success_rate": call_success_rate,
        "reminder_effectiveness": reminder_effectiveness,
        "elapsed_seconds": elapsed,
    }


def run_full_sweep(
    base_noshow_rates: List[float],
    auc_targets: List[float],
    thresholds: List[float],
    seed: int = 42,
    n_patients: int = 5000,
    n_days: int = 90,
    call_capacity_per_day: int = 48,
    call_success_rate: float = 0.65,
    reminder_effectiveness: float = 0.35,
) -> List[Dict[str, Any]]:
    """Run the full grid sweep and return all cell results."""
    total = len(base_noshow_rates) * len(auc_targets) * len(thresholds)
    results = []
    i = 0
    for ns_rate in base_noshow_rates:
        for auc in auc_targets:
            for thresh in thresholds:
                i += 1
                logger.info(
                    "[%d/%d] ns=%.2f auc=%.2f threshold=%.2f",
                    i, total, ns_rate, auc, thresh,
                )
                cell = run_threshold_cell(
                    seed=seed,
                    n_patients=n_patients,
                    n_days=n_days,
                    base_noshow_rate=ns_rate,
                    model_auc=auc,
                    threshold=thresh,
                    call_capacity_per_day=call_capacity_per_day,
                    call_success_rate=call_success_rate,
                    reminder_effectiveness=reminder_effectiveness,
                )
                results.append(cell)
                logger.info(
                    "  flag=%.1f%% TPR=%.1f%% PPV=%.1f%% "
                    "reduction=%.1fpp (%.1fs)",
                    cell["flag_rate"] * 100,
                    cell["sensitivity"] * 100,
                    cell["ppv"] * 100,
                    cell["absolute_reduction_pp"],
                    cell["elapsed_seconds"],
                )
    return results


def save_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
    config: Optional[Dict] = None,
):
    """Save sweep results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(
            {"n_cells": len(results), "cells": results},
            f, indent=2,
        )

    if config:
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # CSV for easy analysis
    if results:
        csv_fields = [
            "threshold", "base_noshow_rate", "model_auc_target",
            "model_auc_achieved", "flag_rate", "sensitivity",
            "specificity", "ppv", "tp", "fp", "tn", "fn",
            "factual_noshow_rate", "counterfactual_noshow_rate",
            "absolute_reduction_pp", "relative_reduction_pct",
            "calls_per_day", "total_appointments",
        ]
        with open(output_dir / "sweep_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=csv_fields, extrasaction="ignore",
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    logger.info("Results saved to %s", output_dir)


def print_sweep_table(results: List[Dict[str, Any]]):
    """Print a formatted comparison table."""
    print()
    print("=" * 100)
    print("THRESHOLD SWEEP — TARGETED REMINDERS")
    print("=" * 100)
    print()
    print(
        f"{'NS Rate':>7s} {'AUC_t':>5s} {'AUC_a':>5s} "
        f"{'Thresh':>6s} {'Flag%':>6s} {'TPR':>6s} "
        f"{'PPV':>6s} {'Spec':>6s} "
        f"{'F-NS%':>6s} {'CF-NS%':>7s} {'Red pp':>7s} "
        f"{'Call/d':>6s}"
    )
    print("-" * 100)

    for r in sorted(
        results,
        key=lambda x: (
            x["base_noshow_rate"],
            x["model_auc_target"],
            x["threshold"],
        ),
    ):
        print(
            f"{r['base_noshow_rate']:>6.0%} "
            f"{r['model_auc_target']:>5.2f} "
            f"{r['model_auc_achieved']:>5.3f} "
            f"{r['threshold']:>6.2f} "
            f"{r['flag_rate']*100:>5.1f}% "
            f"{r['sensitivity']*100:>5.1f}% "
            f"{r['ppv']*100:>5.1f}% "
            f"{r['specificity']*100:>5.1f}% "
            f"{r['factual_noshow_rate']*100:>5.1f}% "
            f"{r['counterfactual_noshow_rate']*100:>6.1f}% "
            f"{r['absolute_reduction_pp']:>6.1f} "
            f"{r['calls_per_day']:>6.1f}"
        )

    # UCSF comparison callout
    print()
    print("UCSF BENCHMARK (20% threshold, ~808K appointments):")
    print(
        f"  Flag rate: {UCSF_BENCHMARK['flag_rate']:.1%}  "
        f"TPR: {UCSF_BENCHMARK['tpr']:.1%}  "
        f"Implied base rate: {UCSF_BENCHMARK['implied_base_noshow_rate']:.1%}"
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Threshold sweep for targeted reminders",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-patients", type=int, default=5000)
    parser.add_argument("--n-days", type=int, default=90)
    parser.add_argument(
        "--base-noshow-rates", type=str,
        default="0.09,0.13,0.18",
        help="Comma-separated base no-show rates",
    )
    parser.add_argument(
        "--auc-targets", type=str,
        default="0.72,0.80,0.85",
        help="Comma-separated AUC targets",
    )
    parser.add_argument(
        "--thresholds", type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="Comma-separated thresholds",
    )
    parser.add_argument(
        "--call-capacity", type=int, default=48,
        help="Daily call capacity (set high to avoid binding)",
    )
    parser.add_argument(
        "--call-success-rate", type=float, default=0.65,
    )
    parser.add_argument(
        "--reminder-effectiveness", type=float, default=0.35,
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: outputs/sweep_threshold_<timestamp>)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ns_rates = [float(x) for x in args.base_noshow_rates.split(",")]
    aucs = [float(x) for x in args.auc_targets.split(",")]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    total = len(ns_rates) * len(aucs) * len(thresholds)
    logger.info(
        "Starting sweep: %d noshow rates x %d AUCs x %d thresholds "
        "= %d cells",
        len(ns_rates), len(aucs), len(thresholds), total,
    )

    config = {
        "experiment_name": "threshold_sweep_ucsf_benchmark",
        "seed": args.seed,
        "n_patients": args.n_patients,
        "n_days": args.n_days,
        "base_noshow_rates": ns_rates,
        "auc_targets": aucs,
        "thresholds": thresholds,
        "call_capacity_per_day": args.call_capacity,
        "call_success_rate": args.call_success_rate,
        "reminder_effectiveness": args.reminder_effectiveness,
        "targeting_mode": "threshold",
        "ucsf_benchmark": UCSF_BENCHMARK,
    }

    t0 = time.time()
    results = run_full_sweep(
        base_noshow_rates=ns_rates,
        auc_targets=aucs,
        thresholds=thresholds,
        seed=args.seed,
        n_patients=args.n_patients,
        n_days=args.n_days,
        call_capacity_per_day=args.call_capacity,
        call_success_rate=args.call_success_rate,
        reminder_effectiveness=args.reminder_effectiveness,
    )
    total_time = time.time() - t0
    logger.info("Sweep complete in %.1f seconds", total_time)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(__file__).parent
            / "outputs"
            / f"sweep_threshold_{ts}"
        )

    save_results(results, output_dir, config)
    print_sweep_table(results)

    print(f"Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
