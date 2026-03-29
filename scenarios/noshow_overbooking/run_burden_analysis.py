"""Overbooking burden analysis for bioethics review.

Runs a 365-day simulation and analyzes how overbooking burden
is distributed across patients and demographics.

Key questions:
- How often is the same patient overbooked?
- Do specific demographics bear disproportionate burden?
- How quickly do high-burden patients hit the individual cap?

Usage:
    python scenarios/noshow_overbooking/run_burden_analysis.py
    python scenarios/noshow_overbooking/run_burden_analysis.py --n-days 365 --cap 5
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scenarios.noshow_overbooking.realistic_scenario import (  # noqa: E402
    ClinicConfig,
    RealisticNoShowScenario,
)
from sdk.core.engine import (  # noqa: E402
    BranchedSimulationEngine, CounterfactualMode,
)
from sdk.core.scenario import TimeConfig  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class BurdenConfig:
    experiment_name: str = "overbooking_burden_analysis"
    timestamp: str = ""
    seed: int = 42
    n_patients: int = 2000
    n_days: int = 365
    base_noshow_rate: float = 0.13
    n_providers: int = 6
    slots_per_provider: int = 12
    max_overbook_per_provider: int = 2
    new_waitlist_requests_per_day: int = 5
    model_type: str = "predictor"
    model_auc: float = 0.83
    overbooking_threshold: float = 0.30
    max_individual_overbooks: int = 10
    overbooking_policy: str = "threshold"
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.04

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def run_burden_analysis(config: BurdenConfig) -> Dict[str, Any]:
    """Run simulation and extract per-patient burden data."""
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
        model_type=config.model_type,
        model_auc=config.model_auc,
        overbooking_threshold=config.overbooking_threshold,
        max_individual_overbooks=config.max_individual_overbooks,
        overbooking_policy=config.overbooking_policy,
        clinic_config=cc,
        ar1_rho=config.ar1_rho,
        ar1_sigma=config.ar1_sigma,
    )

    t0 = time.time()
    # Use NONE mode — we don't need counterfactual for burden analysis
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.NONE,
    ).run(config.n_patients)
    elapsed = time.time() - t0
    logger.info("Simulation complete in %.1f seconds", elapsed)

    # Re-run to access final patient state
    sc2 = RealisticNoShowScenario(
        time_config=tc, seed=config.seed,
        n_patients=config.n_patients,
        base_noshow_rate=config.base_noshow_rate,
        model_type=config.model_type,
        model_auc=config.model_auc,
        overbooking_threshold=config.overbooking_threshold,
        max_individual_overbooks=config.max_individual_overbooks,
        overbooking_policy=config.overbooking_policy,
        clinic_config=cc,
        ar1_rho=config.ar1_rho,
        ar1_sigma=config.ar1_sigma,
    )
    state = sc2.create_population(config.n_patients)
    for t in range(config.n_days):
        state = sc2.step(state, t)
        preds = sc2.predict(state, t)
        state, _ = sc2.intervene(state, preds, t)

    patients = state.patients

    # Build per-patient burden data
    burden_data = []
    for pid, p in patients.items():
        burden_data.append({
            "patient_id": pid,
            "n_times_overbooked": p.n_times_overbooked,
            "n_past_appointments": p.n_past_appointments,
            "n_past_noshows": p.n_past_noshows,
            "historical_noshow_rate": p.historical_noshow_rate,
            "base_noshow_prob": p.base_noshow_prob,
            "visit_type": p.visit_type,
            "race_ethnicity": p.race_ethnicity,
            "insurance_type": p.insurance_type,
            "age_band": p.age_band,
        })

    # Aggregate analysis
    meta = results.outcomes[config.n_days - 1].metadata

    analysis = _analyze_burden(burden_data, config, meta)
    analysis["elapsed_seconds"] = elapsed
    analysis["burden_data"] = burden_data

    return analysis


def _analyze_burden(
    burden_data: List[Dict], config: BurdenConfig, meta: Dict,
) -> Dict[str, Any]:
    """Compute burden distribution and demographic breakdowns."""
    ob_counts = [d["n_times_overbooked"] for d in burden_data]
    n_patients = len(burden_data)

    # Distribution
    hist = Counter(ob_counts)
    max_burden = max(ob_counts)

    # Patients at cap
    at_cap = sum(1 for c in ob_counts if c >= config.max_individual_overbooks)

    # By demographics
    demo_burden = {}
    for dim in ["race_ethnicity", "insurance_type",
                "age_band", "visit_type"]:
        groups = defaultdict(list)
        for d in burden_data:
            groups[d[dim]].append(d["n_times_overbooked"])
        demo_burden[dim] = {
            g: {
                "mean_burden": float(np.mean(vals)),
                "max_burden": int(max(vals)),
                "pct_overbooked": float(
                    sum(1 for v in vals if v > 0) / len(vals)
                ),
                "pct_at_cap": float(
                    sum(
                        1 for v in vals
                        if v >= config.max_individual_overbooks
                    ) / len(vals)
                ),
                "n_patients": len(vals),
            }
            for g, vals in sorted(groups.items())
        }

    # High-burden patients profile
    high_burden = [
        d for d in burden_data if d["n_times_overbooked"] >= 5
    ]
    high_burden_demographics = {}
    if high_burden:
        for dim in ["race_ethnicity", "insurance_type", "age_band"]:
            counts = Counter(d[dim] for d in high_burden)
            total = len(high_burden)
            high_burden_demographics[dim] = {
                k: f"{v/total:.0%}" for k, v in counts.most_common()
            }

    return {
        "config": asdict(config),
        "total_patients": n_patients,
        "total_overbooked_events": sum(ob_counts),
        "patients_never_overbooked": sum(
            1 for c in ob_counts if c == 0
        ),
        "patients_overbooked_once": sum(
            1 for c in ob_counts if c == 1
        ),
        "patients_overbooked_2_4": sum(
            1 for c in ob_counts if 2 <= c <= 4
        ),
        "patients_overbooked_5_plus": sum(
            1 for c in ob_counts if c >= 5
        ),
        "patients_at_cap": at_cap,
        "max_burden_observed": max_burden,
        "mean_burden": float(np.mean(ob_counts)),
        "median_burden": float(np.median(ob_counts)),
        "burden_distribution": {
            str(k): v for k, v in sorted(hist.items())
        },
        "demographic_burden": demo_burden,
        "high_burden_demographics": high_burden_demographics,
        "simulation_meta": {
            "total_collisions": meta.get("total_collisions", 0),
            "total_overbooked": meta.get("total_overbooked", 0),
            "total_noshows": meta.get("total_noshows", 0),
            "total_resolved": meta.get("total_resolved", 0),
        },
    }


def generate_burden_report(analysis: Dict) -> str:
    """Generate markdown report for bioethics review."""
    config = analysis["config"]
    lines = []

    lines.append("# Overbooking Burden Analysis: Bioethics Review")
    lines.append("")
    lines.append(
        f"**Experiment:** {config['experiment_name']}"
    )
    lines.append(f"**Date:** {config['timestamp']}")
    lines.append(
        f"**Duration:** {config['n_days']} days "
        f"({config['n_days']/365:.1f} years)"
    )
    lines.append(
        f"**Policy:** {config['overbooking_policy']} "
        f"(threshold={config['overbooking_threshold']:.2f})"
    )
    lines.append(
        f"**Individual cap:** {config['max_individual_overbooks']} "
        f"overbooks per patient"
    )
    lines.append("")

    # Executive summary
    lines.append("## Summary")
    lines.append("")
    n = analysis["total_patients"]
    never = analysis["patients_never_overbooked"]
    once = analysis["patients_overbooked_once"]
    low = analysis["patients_overbooked_2_4"]
    high = analysis["patients_overbooked_5_plus"]
    cap = analysis["patients_at_cap"]

    lines.append(
        f"Over {config['n_days']} days, "
        f"{analysis['total_overbooked_events']} overbooking events "
        f"were distributed across {n - never} patients "
        f"(out of {n} total)."
    )
    lines.append("")

    # Burden distribution table
    lines.append("## Burden Distribution")
    lines.append("")
    lines.append("| Times Overbooked | Patients | % of Panel |")
    lines.append("|-----------------|----------|-----------|")
    lines.append(
        f"| 0 (never) | {never} | {never/n:.1%} |"
    )
    lines.append(
        f"| 1 | {once} | {once/n:.1%} |"
    )
    lines.append(
        f"| 2-4 | {low} | {low/n:.1%} |"
    )
    lines.append(
        f"| 5+ | {high} | {high/n:.1%} |"
    )
    lines.append(
        f"| At cap ({config['max_individual_overbooks']}) "
        f"| {cap} | {cap/n:.1%} |"
    )
    lines.append("")
    lines.append(
        f"**Mean:** {analysis['mean_burden']:.2f} | "
        f"**Median:** {analysis['median_burden']:.0f} | "
        f"**Max:** {analysis['max_burden_observed']}"
    )
    lines.append("")

    # Detailed histogram
    lines.append("### Detailed Distribution")
    lines.append("")
    lines.append("| Times Overbooked | Count |  |")
    lines.append("|-----------------|-------|--|")
    for k, v in sorted(
        analysis["burden_distribution"].items(),
        key=lambda x: int(x[0]),
    ):
        bar = "#" * min(int(v / 2), 50)
        lines.append(f"| {k} | {v} | {bar} |")
    lines.append("")

    # Demographic breakdown
    lines.append("## Burden by Demographics")
    lines.append("")
    lines.append(
        "Does the overbooking policy disproportionately "
        "burden specific populations?"
    )
    lines.append("")

    for dim, label in [
        ("race_ethnicity", "Race/Ethnicity"),
        ("insurance_type", "Insurance Type"),
        ("age_band", "Age Band"),
        ("visit_type", "Visit Type"),
    ]:
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"| {label} | N | Mean Burden | Max | "
            f"% Ever OB'd | % At Cap |"
        )
        lines.append(
            "|---------|---|-------------|-----|"
            "------------|----------|"
        )
        groups = analysis["demographic_burden"].get(dim, {})
        for g, stats in groups.items():
            lines.append(
                f"| {g} | {stats['n_patients']} "
                f"| {stats['mean_burden']:.2f} "
                f"| {stats['max_burden']} "
                f"| {stats['pct_overbooked']:.1%} "
                f"| {stats['pct_at_cap']:.1%} |"
            )
        lines.append("")

    # High-burden patient profile
    if analysis["high_burden_demographics"]:
        lines.append("## Profile of High-Burden Patients (5+ overbooks)")
        lines.append("")
        lines.append(
            f"{high} patients were overbooked 5 or more times. "
            f"Their demographic composition:"
        )
        lines.append("")
        for dim, label in [
            ("race_ethnicity", "Race/Ethnicity"),
            ("insurance_type", "Insurance"),
            ("age_band", "Age"),
        ]:
            demo = analysis["high_burden_demographics"].get(dim, {})
            if demo:
                parts = [f"{k}: {v}" for k, v in demo.items()]
                lines.append(f"- **{label}:** {', '.join(parts)}")
        lines.append("")

    # Equity flags
    lines.append("## Equity Assessment")
    lines.append("")
    race_burden = analysis["demographic_burden"].get(
        "race_ethnicity", {}
    )
    if race_burden:
        burdens = {
            g: s["mean_burden"] for g, s in race_burden.items()
        }
        max_g = max(burdens, key=burdens.get)
        min_g = min(burdens, key=burdens.get)
        ratio = burdens[max_g] / max(burdens[min_g], 0.001)
        lines.append(
            f"- **Highest burden group:** {max_g} "
            f"(mean {burdens[max_g]:.2f} overbooks)"
        )
        lines.append(
            f"- **Lowest burden group:** {min_g} "
            f"(mean {burdens[min_g]:.2f} overbooks)"
        )
        lines.append(
            f"- **Disparity ratio:** {ratio:.1f}x"
        )
        if ratio > 2.0:
            lines.append(
                f"- **FLAG:** {ratio:.1f}x disparity exceeds "
                f"2.0x threshold — review overbooking criteria"
            )
        else:
            lines.append(
                "- Disparity ratio within acceptable range "
                "(< 2.0x)"
            )
    lines.append("")

    lines.append(
        f"*Generated from experiment {config['timestamp']}*"
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Overbooking burden analysis",
    )
    parser.add_argument("--n-days", type=int, default=365)
    parser.add_argument("--n-patients", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--cap", type=int, default=10)
    parser.add_argument(
        "--policy", type=str, default="threshold",
        choices=["threshold", "urgent_first"],
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = BurdenConfig(
        seed=args.seed,
        n_patients=args.n_patients,
        n_days=args.n_days,
        overbooking_threshold=args.threshold,
        max_individual_overbooks=args.cap,
        overbooking_policy=args.policy,
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

    logger.info("Config: %s", json.dumps(asdict(config), indent=2))

    analysis = run_burden_analysis(config)

    # Save outputs
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Save analysis (without raw burden_data for JSON size)
    analysis_save = {
        k: v for k, v in analysis.items() if k != "burden_data"
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis_save, f, indent=2, default=str)

    # Save per-patient data as CSV
    import csv
    csv_path = output_dir / "patient_burden.csv"
    if analysis["burden_data"]:
        fieldnames = list(analysis["burden_data"][0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(analysis["burden_data"])

    # Generate and save report
    report = generate_burden_report(analysis)
    with open(output_dir / "burden_report.md", "w") as f:
        f.write(report)

    # Register in catalog
    from experiments.catalog import ExperimentCatalog
    catalog = ExperimentCatalog()
    catalog.register(
        output_dir, asdict(config),
        {"type": "burden_analysis",
         "patients_at_cap": analysis["patients_at_cap"],
         "mean_burden": analysis["mean_burden"]},
        notes=f"Burden analysis: {config.n_days} days, "
              f"policy={config.overbooking_policy}",
    )

    print(report)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
