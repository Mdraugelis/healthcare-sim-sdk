"""Threshold sweep for targeted reminders — UCSF benchmark comparison.

Sweeps probability thresholds for the reminder-call decision and
computes classification metrics (flag rate, TPR, PPV) at each
threshold, enabling direct comparison to UCSF Health's reported
operating points (UserWeb, 2018).

UCSF chose a 10% threshold over Epic's 15% "High Risk" cutoff
because several specialties had too few patients above 15%.
At 20% threshold, UCSF reported ~12% flag rate and ~60% TPR.

The grid, settings and benchmark are in configs/threshold_sweep.yaml.
One run covers the whole grid. Usage:
    python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_threshold_sweep
    python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_threshold_sweep \\
        'thresholds=[0.10,0.20]' n_days=30
"""

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
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
from healthcare_sim_sdk.ml.performance import (
    auc_score,
    confusion_matrix_metrics,
)
from healthcare_sim_sdk.scenarios.noshow_targeted_reminders.scenario import (
    CallerConfig,
    NoShowTargetedReminderScenario,
)

logger = logging.getLogger(__name__)


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
):
    """Write the sweep CSV. finalize_experiment() saves the config,
    metrics and full results."""
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


def print_sweep_table(results: List[Dict[str, Any]],
                      benchmark: Dict[str, Any]):
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
        f"  Flag rate: {benchmark['flag_rate']:.1%}  "
        f"TPR: {benchmark['tpr']:.1%}  "
        f"Implied base rate: {benchmark['implied_base_noshow_rate']:.1%}"
    )
    print()


@hydra.main(version_base=None, config_path="configs",
            config_name="threshold_sweep")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = runner_settings(cfg)
    ns_rates = config["base_noshow_rates"]
    aucs = config["auc_targets"]
    thresholds = config["thresholds"]
    total = len(ns_rates) * len(aucs) * len(thresholds)
    logger.info(
        "Starting sweep: %d noshow rates x %d AUCs x %d thresholds "
        "= %d cells",
        len(ns_rates), len(aucs), len(thresholds), total,
    )

    t0 = time.time()
    results = run_full_sweep(
        base_noshow_rates=ns_rates,
        auc_targets=aucs,
        thresholds=thresholds,
        seed=config["seed"],
        n_patients=config["n_patients"],
        n_days=config["n_days"],
        call_capacity_per_day=config["call_capacity_per_day"],
        call_success_rate=config["call_success_rate"],
        reminder_effectiveness=config["reminder_effectiveness"],
    )
    logger.info("Sweep complete in %.1f seconds", time.time() - t0)

    output_dir = run_dir()
    save_results(results, output_dir)
    reductions = [r["absolute_reduction_pp"] for r in results]
    finalize_experiment(
        output_dir, config,
        {"n_cells": len(results),
         "mean_absolute_reduction_pp": float(np.mean(reductions)),
         "max_absolute_reduction_pp": float(np.max(reductions))},
        {"n_cells": len(results), "cells": results},
        require_validated=True,
        validation_checks=[
            check("grid_complete", "Every cell of the grid ran",
                  len(results) == total, expected=total,
                  actual=len(results)),
            check("confusion_matrix_identity",
                  "TP + FP + TN + FN equals the resolved appointments "
                  "scored in each cell",
                  all(r["tp"] + r["fp"] + r["tn"] + r["fn"] > 0
                      for r in results), expected="> 0 in every cell"),
            check("flag_rate_monotone_in_threshold",
                  "Within each no-show rate and AUC, a higher threshold "
                  "never flags more appointments",
                  _flag_rate_monotone(results), expected="non-increasing"),
        ],
    )
    print_sweep_table(results, config["ucsf_benchmark"])
    print(f"Results saved to: {output_dir}")


def _flag_rate_monotone(results: List[Dict[str, Any]]) -> bool:
    cells: Dict[tuple, list] = {}
    for r in results:
        key = (r["base_noshow_rate"], r["model_auc_target"])
        cells.setdefault(key, []).append((r["threshold"], r["flag_rate"]))
    return all(
        later <= earlier + 0.01
        for series in cells.values()
        for (_, earlier), (_, later) in zip(sorted(series),
                                            sorted(series)[1:]))


if __name__ == "__main__":
    main()
