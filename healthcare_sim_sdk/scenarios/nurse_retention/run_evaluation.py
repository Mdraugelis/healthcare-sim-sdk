"""Nurse retention evaluation experiment.

Sweeps across model AUC and manager capacity to find the operating
point that maximizes preventable departures within realistic
staffing constraints.

There is intentionally no threshold dimension. Capacity IS the
threshold — each manager takes the top K nurses by risk score.
Any absolute score threshold would be dominated by capacity.

Usage:
    # Default single run
    python scenarios/nurse_retention/run_evaluation.py

    # Override parameters
    python scenarios/nurse_retention/run_evaluation.py \
        --model-auc 0.75 --capacity 6

    # Full sweep (AUC x capacity), parallel across all cores
    python scenarios/nurse_retention/run_evaluation.py --sweep

    # Force sequential execution (useful for debugging)
    python scenarios/nurse_retention/run_evaluation.py --sweep --workers 1
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from healthcare_sim_sdk.ml.performance import auc_score
from healthcare_sim_sdk.scenarios.nurse_retention.scenario import (
    NurseRetentionScenario,
    RetentionConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Full experiment configuration (saved to output)."""

    experiment_name: str = "nurse_retention_evaluation"
    timestamp: str = ""
    seed: int = 42

    # Population
    n_nurses: int = 1000
    nurses_per_manager: int = 100
    annual_turnover_rate: float = 0.22
    new_hire_fraction: float = 0.15
    new_hire_risk_multiplier: float = 2.0

    # Simulation
    n_weeks: int = 52
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.04
    prediction_interval: int = 2

    # Model
    model_auc: float = 0.80

    # Policy
    max_interventions_per_manager_per_week: int = 4
    intervention_effectiveness: float = 0.50
    decay_halflife_weeks: float = 6.0
    cooldown_weeks: int = 4

    # Sweep grids (used in --sweep mode)
    auc_grid: Optional[List[float]] = None
    capacity_grid: Optional[List[int]] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.auc_grid is None:
            self.auc_grid = [0.60, 0.70, 0.80, 0.85]
        if self.capacity_grid is None:
            self.capacity_grid = [2, 4, 6, 8]


def run_single(
    config: ExperimentConfig,
    model_auc: float,
    capacity: int,
) -> Dict[str, Any]:
    """Run one scenario configuration and return metrics."""
    rc = RetentionConfig(
        n_nurses=config.n_nurses,
        nurses_per_manager=config.nurses_per_manager,
        annual_turnover_rate=config.annual_turnover_rate,
        new_hire_fraction=config.new_hire_fraction,
        new_hire_risk_multiplier=config.new_hire_risk_multiplier,
        n_weeks=config.n_weeks,
        ar1_rho=config.ar1_rho,
        ar1_sigma=config.ar1_sigma,
        prediction_interval=config.prediction_interval,
        model_auc=model_auc,
        max_interventions_per_manager_per_week=capacity,
        intervention_effectiveness=config.intervention_effectiveness,
        intervention_decay_halflife_weeks=config.decay_halflife_weeks,
        cooldown_weeks=config.cooldown_weeks,
    )

    scenario = NurseRetentionScenario(config=rc, seed=config.seed)

    t0 = time.time()
    results = BranchedSimulationEngine(
        scenario, CounterfactualMode.BRANCHED,
    ).run(config.n_nurses)
    elapsed = time.time() - t0

    # Extract final-week metadata from both branches
    final_t = config.n_weeks - 1
    f_meta = results.outcomes[final_t].metadata
    cf_meta = results.counterfactual_outcomes[final_t].metadata

    # Compute realized AUC from predictions
    all_pred, all_true = [], []
    for t_step in results.predictions:
        p = results.predictions[t_step]
        all_pred.append(p.scores)
        all_true.append(p.metadata["true_labels"])
    pred_arr = np.concatenate(all_pred) if all_pred else np.array([])
    true_arr = np.concatenate(all_true) if all_true else np.array([])
    # Filter out zero-score (departed) entries
    mask = pred_arr > 0
    realized_auc = (
        float(auc_score(true_arr[mask], pred_arr[mask]))
        if mask.sum() > 100 else 0.0
    )

    # Intervention metrics
    total_interventions = f_meta["total_interventions"]
    n_prediction_rounds = len(results.predictions)
    avg_active_per_round = np.mean([
        results.predictions[t_step].metadata["n_active"]
        for t_step in results.predictions
    ]) if n_prediction_rounds > 0 else 0

    # Weekly departure trajectory
    factual_departures = [
        results.outcomes[t_step].metadata["total_departures"]
        for t_step in sorted(results.outcomes.keys())
    ]
    cf_departures = [
        results.counterfactual_outcomes[t_step].metadata[
            "total_departures"
        ]
        for t_step in sorted(
            results.counterfactual_outcomes.keys()
        )
    ]

    departures_prevented = (
        cf_meta["total_departures"] - f_meta["total_departures"]
    )

    return {
        "model_auc_target": model_auc,
        "realized_auc": realized_auc,
        "manager_capacity": capacity,
        "factual_departures": f_meta["total_departures"],
        "counterfactual_departures": cf_meta["total_departures"],
        "departures_prevented": departures_prevented,
        "prevention_rate": (
            departures_prevented / max(cf_meta["total_departures"], 1)
        ),
        "factual_retention_rate": f_meta["retention_rate"],
        "counterfactual_retention_rate": cf_meta["retention_rate"],
        "total_interventions": total_interventions,
        "avg_active_per_round": float(avg_active_per_round),
        "interventions_per_prevented": (
            total_interventions / max(departures_prevented, 1)
            if departures_prevented > 0 else float("inf")
        ),
        "n_active_final": f_meta["n_active"],
        "mean_risk_final": f_meta["mean_risk"],
        "elapsed_seconds": elapsed,
        "factual_trajectory": factual_departures,
        "counterfactual_trajectory": cf_departures,
    }


CellTask = Tuple[ExperimentConfig, str, float, int]


def _run_cell(task: CellTask) -> Dict[str, Any]:
    """Pool worker: run one cell and attach its label.

    Module-level so it is picklable under the ``spawn`` start method
    used on macOS. Each call constructs a fresh scenario seeded from
    ``config.seed`` — results are deterministic regardless of the
    execution order the pool happens to produce.
    """
    config, label, model_auc, capacity = task
    result = run_single(config, model_auc, capacity)
    result["label"] = label
    return result


def _build_task_list(config: ExperimentConfig) -> List[CellTask]:
    """Build the (control + sweep) task list in canonical order.

    Raises:
        ValueError: if the configured grids produce duplicate labels.
            Labels are used as unique identifiers downstream (summary,
            CSV, result reordering), so two tasks collapsing to the
            same ``auc{auc:.2f}_cap{cap}`` string — e.g. grids like
            ``[0.801, 0.805]`` that both round to ``"0.80"`` — would
            silently drop one result.
    """
    tasks: List[CellTask] = [
        (config, "no_ai_control", config.model_auc, 0),
    ]
    for auc in config.auc_grid:
        for cap in config.capacity_grid:
            tasks.append(
                (config, f"auc{auc:.2f}_cap{cap}", auc, cap),
            )

    seen: Dict[str, Tuple[float, int]] = {}
    for _, label, auc, cap in tasks:
        if label in seen:
            prev_auc, prev_cap = seen[label]
            raise ValueError(
                f"Duplicate sweep label {label!r}: "
                f"(auc={prev_auc}, cap={prev_cap}) collides with "
                f"(auc={auc}, cap={cap}). Labels use the "
                f"'auc{{:.2f}}_cap{{}}' format — adjust the grids "
                f"so every cell has a unique label."
            )
        seen[label] = (auc, cap)
    return tasks


def _resolve_workers(requested: int, n_tasks: int) -> int:
    """Pick an effective worker count.

    ``requested == 0`` means auto: use all CPUs reported by
    ``os.cpu_count()`` (logical cores on most platforms), capped by
    the number of tasks. ``requested >= 1`` uses exactly that many,
    also capped by ``n_tasks``. Tiny sweeps (≤ 2 tasks) always fall
    back to sequential because spawn overhead dominates any speedup.
    """
    if n_tasks <= 2:
        return 1
    cpu = os.cpu_count() or 1
    if requested <= 0:
        return min(cpu, n_tasks)
    return min(requested, n_tasks)


def run_experiment(
    config: ExperimentConfig,
    workers: int = 0,
) -> Dict[str, Any]:
    """Run full experiment: no-AI baseline + AUC x capacity sweep.

    Args:
        config: Experiment configuration (including sweep grids).
        workers: Parallel workers. ``0`` (default) = auto-size to
            cores; ``1`` = sequential; ``N`` = use exactly N workers.
    """
    tasks = _build_task_list(config)
    total = len(tasks)
    effective_workers = _resolve_workers(workers, total)

    if effective_workers == 1:
        logger.info("Running %d cells sequentially...", total)
        results_unordered = []
        for idx, task in enumerate(tasks, 1):
            _, label, auc, cap = task
            logger.info(
                "[%d/%d] %s (AUC=%.2f, cap=%d)...",
                idx, total, label, auc, cap,
            )
            results_unordered.append(_run_cell(task))
    else:
        logger.info(
            "Running %d cells across %d workers...",
            total, effective_workers,
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(effective_workers) as pool:
            results_unordered = []
            for idx, result in enumerate(
                pool.imap_unordered(_run_cell, tasks), 1,
            ):
                logger.info(
                    "[%d/%d] completed %s",
                    idx, total, result["label"],
                )
                results_unordered.append(result)

    # Restore canonical order (control first, then grid order) so
    # CSV row order is stable across sequential/parallel runs.
    by_label = {r["label"]: r for r in results_unordered}
    ordered = [by_label[label] for _, label, _, _ in tasks]

    return {
        "config": asdict(config),
        "results": ordered,
        "summary": _build_summary(ordered),
    }


def _build_summary(results: List[Dict]) -> Dict[str, Any]:
    """Find the best operating point and summarize."""
    sweep = [r for r in results if r["label"] != "no_ai_control"]
    control = next(
        r for r in results if r["label"] == "no_ai_control"
    )

    if not sweep:
        return {"no_sweep_results": True}

    # Best by departures prevented
    best = max(sweep, key=lambda r: r["departures_prevented"])

    # Best by efficiency (interventions per prevented departure)
    efficient = [
        r for r in sweep if r["departures_prevented"] > 0
    ]
    most_efficient = (
        min(efficient, key=lambda r: r["interventions_per_prevented"])
        if efficient else None
    )

    return {
        "baseline_departures": control["counterfactual_departures"],
        "baseline_retention_rate": control[
            "counterfactual_retention_rate"
        ],
        "best_label": best["label"],
        "best_departures_prevented": best["departures_prevented"],
        "best_prevention_rate": best["prevention_rate"],
        "best_factual_retention": best["factual_retention_rate"],
        "best_total_interventions": best["total_interventions"],
        "most_efficient_label": (
            most_efficient["label"] if most_efficient else None
        ),
        "most_efficient_cost": (
            most_efficient["interventions_per_prevented"]
            if most_efficient else None
        ),
    }


def save_results(experiment: Dict, output_dir: Path):
    """Save experiment results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    with open(output_dir / "config.json", "w") as f:
        json.dump(experiment["config"], f, indent=2)

    # Full results (strip trajectories for JSON readability)
    results_clean = []
    for r in experiment["results"]:
        r_copy = {
            k: v for k, v in r.items()
            if not k.endswith("_trajectory")
        }
        results_clean.append(r_copy)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    # Summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(experiment["summary"], f, indent=2)

    # CSV for easy analysis
    csv_path = output_dir / "results.csv"
    fieldnames = [
        "label", "model_auc_target", "realized_auc",
        "manager_capacity",
        "factual_departures", "counterfactual_departures",
        "departures_prevented", "prevention_rate",
        "factual_retention_rate", "counterfactual_retention_rate",
        "total_interventions", "avg_active_per_round",
        "interventions_per_prevented",
        "n_active_final", "mean_risk_final", "elapsed_seconds",
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
    config = experiment["config"]
    results = experiment["results"]
    summary = experiment["summary"]

    print()
    print("=" * 80)
    print("NURSE RETENTION — MODEL PERFORMANCE x MANAGER CAPACITY")
    print("=" * 80)
    print(f"Date: {config['timestamp']}")
    print(
        f"Nurses: {config['n_nurses']}, "
        f"Per manager: {config['nurses_per_manager']}, "
        f"Weeks: {config['n_weeks']}, "
        f"Seed: {config['seed']}"
    )
    print(
        f"Annual turnover: {config['annual_turnover_rate']:.0%}, "
        f"New hire fraction: {config['new_hire_fraction']:.0%}, "
        f"New hire risk multiplier: "
        f"{config['new_hire_risk_multiplier']:.1f}x"
    )
    print(
        f"Intervention: effectiveness="
        f"{config['intervention_effectiveness']:.0%}, "
        f"decay half-life="
        f"{config['decay_halflife_weeks']:.0f}wk, "
        f"cooldown={config['cooldown_weeks']}wk"
    )
    print()

    control = next(
        r for r in results if r["label"] == "no_ai_control"
    )
    print(
        f"BASELINE (no AI): {control['counterfactual_departures']} "
        f"departures, "
        f"retention={control['counterfactual_retention_rate']:.1%}"
    )
    print()

    # Table header
    print(
        f"{'AUC':>5s} {'Cap':>4s} "
        f"{'Depart':>7s} {'CF Dep':>7s} {'Saved':>6s} "
        f"{'Prev%':>6s} {'Intv':>6s} {'Cost':>6s} "
        f"{'Retain':>7s}"
    )
    print("-" * 68)

    sweep = [
        r for r in results if r["label"] != "no_ai_control"
    ]
    # Sort by departures prevented descending
    for r in sorted(
        sweep, key=lambda x: -x["departures_prevented"],
    ):
        cost = (
            f"{r['interventions_per_prevented']:.1f}"
            if r["departures_prevented"] > 0 else "inf"
        )
        print(
            f"{r['model_auc_target']:>5.2f} "
            f"{r['manager_capacity']:>4d} "
            f"{r['factual_departures']:>7d} "
            f"{r['counterfactual_departures']:>7d} "
            f"{r['departures_prevented']:>6d} "
            f"{r['prevention_rate']:>5.1%} "
            f"{r['total_interventions']:>6d} "
            f"{cost:>6s} "
            f"{r['factual_retention_rate']:>6.1%}"
        )

    print()
    if summary.get("best_label"):
        print(
            f"BEST: {summary['best_label']} — "
            f"{summary['best_departures_prevented']} departures "
            f"prevented "
            f"({summary['best_prevention_rate']:.1%}), "
            f"retention={summary['best_factual_retention']:.1%}"
        )
    if summary.get("most_efficient_label"):
        print(
            f"MOST EFFICIENT: {summary['most_efficient_label']} — "
            f"{summary['most_efficient_cost']:.1f} check-ins per "
            f"prevented departure"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run nurse retention evaluation",
    )
    parser.add_argument("--n-nurses", type=int, default=1000)
    parser.add_argument("--nurses-per-manager", type=int, default=100)
    parser.add_argument("--n-weeks", type=int, default=52)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-auc", type=float, default=0.80)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--effectiveness", type=float, default=0.50)
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run full AUC x capacity sweep",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help=(
            "Parallel workers for the sweep. "
            "0 = auto (os.cpu_count(), capped by #cells); "
            "1 = sequential; "
            "N > 1 = use exactly N workers (also capped by #cells). "
            "Sweeps with <=2 cells always run sequentially."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = ExperimentConfig(
        seed=args.seed,
        n_nurses=args.n_nurses,
        nurses_per_manager=args.nurses_per_manager,
        n_weeks=args.n_weeks,
        model_auc=args.model_auc,
        max_interventions_per_manager_per_week=args.capacity,
        intervention_effectiveness=args.effectiveness,
    )

    if not args.sweep:
        # Single run mode
        config.auc_grid = [args.model_auc]
        config.capacity_grid = [args.capacity]

    output_dir = Path(args.output_dir) / (
        f"{config.experiment_name}_{config.timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(fh)

    logger.info("Starting: %s", config.experiment_name)
    logger.info("Config: %s", json.dumps(asdict(config), indent=2))

    t0 = time.time()
    experiment = run_experiment(config, workers=args.workers)
    total_time = time.time() - t0

    logger.info("Complete in %.1f seconds", total_time)

    save_results(experiment, output_dir)
    print_report(experiment)
    print(f"\nResults saved to: {output_dir}")

    return experiment


if __name__ == "__main__":
    main()
