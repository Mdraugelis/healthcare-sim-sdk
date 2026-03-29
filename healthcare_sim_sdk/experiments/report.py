"""Markdown report generator for experiment results.

Generates a structured report from one or more experiment runs,
suitable for sharing with stakeholders or archiving.

Usage:
    from healthcare_sim_sdk.experiments.report import generate_report
    report = generate_report("20260328_220210")
    print(report)

    # Or from CLI:
    python experiments/report.py 20260328_220210
"""

import sys
from pathlib import Path
from typing import List, Optional


from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog  # noqa: E402


def generate_report(
    timestamp: str,
    catalog: Optional[ExperimentCatalog] = None,
) -> str:
    """Generate a markdown report for one experiment."""
    if catalog is None:
        catalog = ExperimentCatalog()
    data = catalog.load(timestamp)
    if not data:
        return f"Experiment {timestamp} not found in catalog."

    config = data["config"]
    results = data["results"]
    entry = data["entry"]

    control = _find(results, "no_overbooking")
    baseline = _find(results, "baseline")
    predictors = [
        r for r in results if r.get("label", "").startswith("predictor")
    ]

    lines = []
    _h1(lines, "No-Show Overbooking Evaluation Report")
    lines.append("")
    lines.append(f"**Experiment:** {config.get('experiment_name')}")
    lines.append(f"**Date:** {config.get('timestamp')}")
    lines.append(f"**Seed:** {config.get('seed')}")
    if entry.get("notes"):
        lines.append(f"**Notes:** {entry['notes']}")
    lines.append("")

    # Configuration
    _h2(lines, "Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Patients | {config.get('n_patients'):,} |")
    lines.append(f"| Simulation days | {config.get('n_days')} |")
    lines.append(
        f"| No-show rate | {config.get('base_noshow_rate'):.0%} |"
    )
    daily_cap = (
        config.get("n_providers", 0)
        * config.get("slots_per_provider", 0)
    )
    lines.append(
        f"| Providers | {config.get('n_providers')} |"
    )
    lines.append(
        f"| Daily capacity | {daily_cap} slots |"
    )
    lines.append(
        f"| Waitlist requests/day | "
        f"{config.get('new_waitlist_requests_per_day')} |"
    )
    lines.append(
        f"| AR(1) drift | rho={config.get('ar1_rho')}, "
        f"sigma={config.get('ar1_sigma')} |"
    )
    lines.append(
        f"| ML model target AUC | {config.get('model_auc')} |"
    )
    lines.append(
        f"| Baseline threshold | "
        f"{config.get('baseline_threshold'):.0%} historical rate |"
    )
    lines.append("")

    # Control
    _h2(lines, "Control: No Overbooking")
    lines.append("")
    if control:
        lines.append(
            f"- **Utilization:** {control['utilization']:.1%}"
        )
        lines.append(
            f"- **Waitlist at day {config.get('n_days')}:** "
            f"{control['waitlist_size']}"
        )
        lines.append(
            f"- **No-show rate:** {control['noshow_rate']:.1%}"
        )
    lines.append("")

    # Baseline
    _h2(lines, "Baseline: Staff Historical Rate")
    lines.append("")
    if baseline:
        lines.append(
            f"Staff overbooks when a patient's historical "
            f"no-show rate exceeds "
            f"{config.get('baseline_threshold'):.0%}."
        )
        lines.append("")
        lines.append(f"- **AUC:** {baseline['auc']:.3f}")
        lines.append(
            f"- **Utilization:** {baseline['utilization']:.1%}"
        )
        lines.append(
            f"- **Collision rate:** "
            f"{baseline['collision_rate']:.1%}"
        )
        lines.append(
            f"- **Overbookings/week:** "
            f"{baseline['overbookings_per_week']:.1f}"
        )
        lines.append(
            f"- **Waitlist remaining:** "
            f"{baseline['waitlist_size']}"
        )
        lines.append(
            f"- **Waitlist patients served:** "
            f"{baseline['total_waitlist_served']}"
        )
    lines.append("")

    # Predictor sweep
    _h2(lines, "ML Predictor: Threshold Sweep")
    lines.append("")
    lines.append(
        f"ML predictor with target AUC "
        f"{config.get('model_auc')} evaluated at "
        f"multiple overbooking thresholds."
    )
    lines.append("")
    lines.append(
        "| Threshold | AUC | Utilization | Collision Rate "
        "| OB/Week | OB Show Rate | Waitlist | Served |"
    )
    lines.append(
        "|-----------|-----|-------------|---------------"
        "|---------|--------------|----------|--------|"
    )
    for r in predictors:
        lines.append(
            f"| {r['overbooking_threshold']:.2f} "
            f"| {r['auc']:.3f} "
            f"| {r['utilization']:.1%} "
            f"| {r['collision_rate']:.1%} "
            f"| {r['overbookings_per_week']:.1f} "
            f"| {r['overbooked_show_rate']:.1%} "
            f"| {r['waitlist_size']} "
            f"| {r['total_waitlist_served']} |"
        )
    lines.append("")

    # Key finding
    _h2(lines, "Key Finding")
    lines.append("")
    if entry.get("collision_rate_reduction"):
        pct = entry["collision_rate_reduction"]
        bt = entry.get("best_predictor_threshold")
        lines.append(
            f"At similar utilization to the baseline, the ML "
            f"predictor (threshold={bt:.2f}) reduces the "
            f"collision rate by **{pct:.0%}**."
        )
    lines.append("")

    # Per-threshold classification metrics (if available)
    _h2(lines, "Model Classification Metrics")
    lines.append("")
    lines.append(
        "Performance of the ML model's predicted no-show "
        "probability at various decision thresholds:"
    )
    lines.append("")
    # Use the first predictor run that has threshold_metrics
    tm_source = next(
        (r for r in predictors if r.get("threshold_metrics")), None
    )
    if tm_source:
        lines.append(
            "| Decision Threshold | Sensitivity | Specificity "
            "| PPV | NPV | Flag Rate |"
        )
        lines.append(
            "|-------------------|-------------|------------"
            "|-----|-----|-----------|"
        )
        for key in sorted(tm_source["threshold_metrics"].keys()):
            m = tm_source["threshold_metrics"][key]
            lines.append(
                f"| {key.replace('at_', '')} "
                f"| {m['sensitivity']:.3f} "
                f"| {m['specificity']:.3f} "
                f"| {m['ppv']:.3f} "
                f"| {m['npv']:.3f} "
                f"| {m['flag_rate']:.1%} |"
            )
        lines.append("")

    # Interpretation
    _h2(lines, "Interpretation")
    lines.append("")
    if control and baseline and predictors:
        util_gap = (
            baseline["utilization"] - control["utilization"]
        )
        lines.append(
            f"1. Without overbooking, utilization is "
            f"{control['utilization']:.1%} with "
            f"{control['waitlist_size']} patients on "
            f"the waitlist."
        )
        lines.append(
            f"2. The baseline (historical rate >= "
            f"{config.get('baseline_threshold'):.0%}) "
            f"improves utilization by "
            f"{util_gap:.1%} but has a "
            f"{baseline['collision_rate']:.0%} "
            f"collision rate."
        )
        lines.append(
            "3. The ML predictor achieves similar "
            "utilization with substantially lower "
            "collision rates at the right threshold."
        )
        lines.append(
            f"4. The predictor's AUC advantage "
            f"({entry.get('best_predictor_auc', 0):.3f} "
            f"vs {entry.get('baseline_auc', 0):.3f}) "
            f"comes from tracking current behavioral "
            f"drift that the historical rate can't see."
        )
    lines.append("")

    return "\n".join(lines)


def generate_comparison_report(
    timestamps: List[str],
    catalog: Optional[ExperimentCatalog] = None,
) -> str:
    """Generate a comparison report across multiple experiments."""
    if catalog is None:
        catalog = ExperimentCatalog()
    entries = catalog.compare(timestamps)
    if not entries:
        return "No experiments found."

    lines = []
    _h1(lines, "Experiment Comparison")
    lines.append("")
    lines.append(
        "| Timestamp | Patients | Days | AUC Target "
        "| Baseline AUC | Predictor AUC | Collision Reduction |"
    )
    lines.append(
        "|-----------|----------|------|------------"
        "|--------------|---------------|---------------------|"
    )
    for e in entries:
        cr = e.get("collision_rate_reduction")
        cr_str = f"{cr:.0%}" if cr else "N/A"
        lines.append(
            f"| {e['timestamp']} "
            f"| {e.get('n_patients', '')} "
            f"| {e.get('n_days', '')} "
            f"| {e.get('model_auc', '')} "
            f"| {e.get('baseline_auc', 0):.3f} "
            f"| {e.get('best_predictor_auc', 0):.3f} "
            f"| {cr_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _h1(lines, text):
    lines.append(f"# {text}")


def _h2(lines, text):
    lines.append(f"## {text}")


def _find(results, label):
    return next((r for r in results if r.get("label") == label), None)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/report.py <timestamp>")
        print("       python experiments/report.py --list")
        print("       python experiments/report.py --compare ts1 ts2")
        sys.exit(1)

    if sys.argv[1] == "--list":
        cat = ExperimentCatalog()
        for e in cat.list_experiments():
            cr = e.get("collision_rate_reduction")
            cr_str = f"{cr:.0%}" if cr else "?"
            print(
                f"  {e['timestamp']}  "
                f"n={e.get('n_patients')}, "
                f"d={e.get('n_days')}, "
                f"auc={e.get('model_auc')}, "
                f"reduction={cr_str}"
            )
    elif sys.argv[1] == "--compare":
        report = generate_comparison_report(sys.argv[2:])
        print(report)
    else:
        report = generate_report(sys.argv[1])
        print(report)
