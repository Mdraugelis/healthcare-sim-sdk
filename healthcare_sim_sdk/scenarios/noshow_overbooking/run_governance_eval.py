"""AI Governance evaluation for no-show overbooking.

Evaluates the ML predictor against the governance framework's
Phase 1 validation metrics, equity audit fairness rules, and
identifies the optimal policy + threshold configuration.

Governance targets:
- AUC >= 0.70 (minimum), >= 0.80 (target)
- PPV >= 30% at intervention threshold
- Calibration slope 0.8-1.2
- Sensitivity >= 50% at threshold
- No subgroup AUC more than 15% worse than overall
- Flagging proportional to actual no-show rate by subgroup
- Overbooking burden not concentrated by demographics

Usage:
    python scenarios/noshow_overbooking/run_governance_eval.py
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


from healthcare_sim_sdk.scenarios.noshow_overbooking.realistic_scenario import (
    ClinicConfig, RealisticNoShowScenario,
)
from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine, CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.performance import (
    auc_score, calibration_slope, confusion_matrix_metrics,
)

logger = logging.getLogger(__name__)


# -- Governance thresholds from the evaluation plan -----------------------

GOVERNANCE = {
    "auc_minimum": 0.70,
    "auc_target": 0.80,
    "ppv_minimum": 0.30,
    "sensitivity_minimum": 0.50,
    "calibration_slope_low": 0.8,
    "calibration_slope_high": 1.2,
    "subgroup_auc_max_gap": 0.15,  # no subgroup > 15% worse
    "burden_disparity_max": 2.0,   # max ratio between groups
}


@dataclass
class GovernanceConfig:
    experiment_name: str = "governance_evaluation"
    timestamp: str = ""
    seed: int = 42
    n_patients: int = 2000
    n_days: int = 90
    base_noshow_rate: float = 0.13
    n_providers: int = 6
    slots_per_provider: int = 12
    max_overbook_per_provider: int = 2
    new_waitlist_requests_per_day: int = 5
    model_aucs: List[float] = None
    thresholds: List[float] = None
    policies: List[str] = None
    max_individual_overbooks: int = 10
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.04

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.model_aucs is None:
            self.model_aucs = [0.70, 0.80, 0.83, 0.87]
        if self.thresholds is None:
            self.thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        if self.policies is None:
            self.policies = ["threshold", "urgent_first"]


def run_single_config(
    config: GovernanceConfig,
    model_auc: float,
    threshold: float,
    policy: str,
) -> Dict[str, Any]:
    """Run one configuration and compute all governance metrics."""
    cc = ClinicConfig(
        n_providers=config.n_providers,
        slots_per_provider_per_day=config.slots_per_provider,
        max_overbook_per_provider=config.max_overbook_per_provider,
        new_waitlist_requests_per_day=(
            config.new_waitlist_requests_per_day
        ),
    )
    tc = TimeConfig(
        n_timesteps=config.n_days, timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(config.n_days)),
    )
    sc = RealisticNoShowScenario(
        time_config=tc, seed=config.seed,
        n_patients=config.n_patients,
        base_noshow_rate=config.base_noshow_rate,
        model_type="predictor", model_auc=model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=config.max_individual_overbooks,
        overbooking_policy=policy, clinic_config=cc,
        ar1_rho=config.ar1_rho, ar1_sigma=config.ar1_sigma,
    )

    results = BranchedSimulationEngine(
        sc, CounterfactualMode.NONE,
    ).run(config.n_patients)

    # Collect predictions vs outcomes
    all_pred, all_act = [], []
    all_race, all_ins, all_age = [], [], []
    for t in range(config.n_days):
        if t in results.predictions and (t + 1) < config.n_days:
            p = results.predictions[t].scores
            a = results.outcomes[t + 1].events
            if len(p) == len(a):
                all_pred.append(p)
                all_act.append(a)
                all_race.append(
                    results.outcomes[t + 1].secondary.get(
                        "race_ethnicity", np.array([])
                    )
                )
                all_ins.append(
                    results.outcomes[t + 1].secondary.get(
                        "insurance_type", np.array([])
                    )
                )
                all_age.append(
                    results.outcomes[t + 1].secondary.get(
                        "age_band", np.array([])
                    )
                )

    pred_arr = np.concatenate(all_pred) if all_pred else np.array([])
    act_arr = np.concatenate(all_act) if all_act else np.array([])
    race_arr = np.concatenate(all_race) if all_race else np.array([])
    ins_arr = np.concatenate(all_ins) if all_ins else np.array([])
    age_arr = np.concatenate(all_age) if all_age else np.array([])

    # Overall metrics
    overall_auc = (
        float(auc_score(act_arr, pred_arr))
        if len(pred_arr) > 100 else 0
    )
    overall_cm = (
        confusion_matrix_metrics(act_arr, pred_arr, threshold)
        if len(pred_arr) > 100 else {}
    )
    cal_slope = 0.0
    if len(pred_arr) > 100:
        cal_slope, _, _ = calibration_slope(act_arr, pred_arr)

    meta = results.outcomes[config.n_days - 1].metadata

    # Utilization
    util_vals = []
    for t in range(1, config.n_days):
        u = results.outcomes[t].secondary.get("utilization")
        if u is not None and len(u) > 0:
            util_vals.append(float(u.mean()))

    # Per-subgroup AUC
    subgroup_metrics = {}
    for dim, arr in [
        ("race_ethnicity", race_arr),
        ("insurance_type", ins_arr),
        ("age_band", age_arr),
    ]:
        if len(arr) == 0 or len(arr) != len(pred_arr):
            continue
        groups = {}
        for g in np.unique(arr):
            mask = arr == g
            n_g = mask.sum()
            if n_g < 50:
                continue
            g_auc = float(auc_score(act_arr[mask], pred_arr[mask]))
            g_cm = confusion_matrix_metrics(
                act_arr[mask], pred_arr[mask], threshold,
            )
            g_noshow = float(act_arr[mask].mean())
            g_flagged = float(
                (pred_arr[mask] >= threshold).mean()
            )
            groups[str(g)] = {
                "n": int(n_g),
                "auc": g_auc,
                "ppv": g_cm.get("ppv", 0),
                "sensitivity": g_cm.get("sensitivity", 0),
                "flag_rate": g_flagged,
                "actual_noshow_rate": g_noshow,
                "flag_to_noshow_ratio": (
                    g_flagged / max(g_noshow, 0.001)
                ),
            }
        subgroup_metrics[dim] = groups

    # Check governance rules
    governance_checks = _check_governance(
        overall_auc, overall_cm, cal_slope, subgroup_metrics,
        meta,
    )

    n_ob = meta.get("total_overbooked", 0)
    return {
        "model_auc_target": model_auc,
        "threshold": threshold,
        "policy": policy,
        "overall_auc": overall_auc,
        "ppv": overall_cm.get("ppv", 0),
        "sensitivity": overall_cm.get("sensitivity", 0),
        "specificity": overall_cm.get("specificity", 0),
        "calibration_slope": float(cal_slope),
        "flag_rate": overall_cm.get("flag_rate", 0),
        "utilization": float(np.mean(util_vals)) if util_vals else 0,
        "collision_rate": (
            meta.get("total_collisions", 0) / max(n_ob, 1)
            if n_ob > 0 else 0
        ),
        "total_collisions": meta.get("total_collisions", 0),
        "total_overbooked": n_ob,
        "waitlist_size": meta.get("waitlist_size", 0),
        "mean_burden": meta.get("mean_overbooking_burden", 0),
        "subgroup_metrics": subgroup_metrics,
        "governance_checks": governance_checks,
        "all_checks_pass": all(
            c["passed"] for c in governance_checks
        ),
    }


def _check_governance(
    auc, cm, cal_slope, subgroup_metrics, meta,
) -> List[Dict]:
    """Check all governance rules."""
    checks = []

    # 1. AUC minimum
    checks.append({
        "rule": "AUC >= 0.70 (minimum)",
        "value": f"{auc:.3f}",
        "target": "0.70",
        "passed": auc >= GOVERNANCE["auc_minimum"],
    })

    # 2. AUC target
    checks.append({
        "rule": "AUC >= 0.80 (target)",
        "value": f"{auc:.3f}",
        "target": "0.80",
        "passed": auc >= GOVERNANCE["auc_target"],
    })

    # 3. PPV
    ppv = cm.get("ppv", 0)
    checks.append({
        "rule": "PPV >= 30% at threshold",
        "value": f"{ppv:.1%}",
        "target": "30%",
        "passed": ppv >= GOVERNANCE["ppv_minimum"],
    })

    # 4. Sensitivity
    sens = cm.get("sensitivity", 0)
    checks.append({
        "rule": "Sensitivity >= 50% at threshold",
        "value": f"{sens:.1%}",
        "target": "50%",
        "passed": sens >= GOVERNANCE["sensitivity_minimum"],
    })

    # 5. Calibration
    checks.append({
        "rule": "Calibration slope 0.8-1.2",
        "value": f"{cal_slope:.3f}",
        "target": "0.8-1.2",
        "passed": (
            GOVERNANCE["calibration_slope_low"]
            <= cal_slope
            <= GOVERNANCE["calibration_slope_high"]
        ),
    })

    # 6. Subgroup AUC fairness
    for dim, groups in subgroup_metrics.items():
        for g, stats in groups.items():
            gap = auc - stats["auc"]
            pct_gap = gap / max(auc, 0.001)
            checks.append({
                "rule": (
                    f"{dim}={g}: AUC within 15% of overall"
                ),
                "value": (
                    f"{stats['auc']:.3f} "
                    f"(gap={pct_gap:.0%})"
                ),
                "target": f">= {auc * 0.85:.3f}",
                "passed": pct_gap <= GOVERNANCE[
                    "subgroup_auc_max_gap"
                ],
            })

    # 7. Proportional flagging
    for dim, groups in subgroup_metrics.items():
        for g, stats in groups.items():
            ratio = stats["flag_to_noshow_ratio"]
            checks.append({
                "rule": (
                    f"{dim}={g}: flag rate proportional "
                    f"to no-show rate"
                ),
                "value": (
                    f"flag={stats['flag_rate']:.1%}, "
                    f"noshow={stats['actual_noshow_rate']:.1%}, "
                    f"ratio={ratio:.2f}"
                ),
                "target": "ratio 0.5-2.0",
                "passed": 0.5 <= ratio <= 2.0,
            })

    return checks


def generate_governance_report(
    config: GovernanceConfig,
    all_results: List[Dict],
) -> str:
    """Generate governance-ready markdown report."""
    lines = []
    lines.append("# AI Governance Evaluation: No-Show Predictor")
    lines.append("")
    lines.append(
        f"*{config.timestamp} | {config.n_patients} patients | "
        f"{config.n_days} days | seed {config.seed}*"
    )
    lines.append("")

    # Phase 1: Find configurations meeting all criteria
    lines.append("## Phase 1: Local Validation Metrics")
    lines.append("")
    lines.append(
        "| AUC Target | Threshold | Policy | AUC | PPV "
        "| Sens | Cal Slope | Util | Collision "
        "| All Pass |"
    )
    lines.append(
        "|------------|-----------|--------|-----|-----"
        "|------|-----------|------|----------"
        "|----------|"
    )
    for r in all_results:
        status = "YES" if r["all_checks_pass"] else "no"
        lines.append(
            f"| {r['model_auc_target']:.2f} "
            f"| {r['threshold']:.2f} "
            f"| {r['policy']} "
            f"| {r['overall_auc']:.3f} "
            f"| {r['ppv']:.1%} "
            f"| {r['sensitivity']:.1%} "
            f"| {r['calibration_slope']:.2f} "
            f"| {r['utilization']:.1%} "
            f"| {r['collision_rate']:.1%} "
            f"| **{status}** |"
        )
    lines.append("")

    # Identify optimal configuration
    passing = [r for r in all_results if r["all_checks_pass"]]
    if passing:
        # Among passing, pick lowest collision rate
        optimal = min(passing, key=lambda r: r["collision_rate"])
        lines.append("### Recommended Configuration")
        lines.append("")
        lines.append(
            f"**Model AUC: {optimal['model_auc_target']:.2f}, "
            f"Threshold: {optimal['threshold']:.2f}, "
            f"Policy: {optimal['policy']}**"
        )
        lines.append("")
        auc_note = (
            "meets 0.80 target"
            if optimal["overall_auc"] >= 0.80
            else "meets 0.70 minimum"
        )
        lines.append(
            f"- AUC: {optimal['overall_auc']:.3f} ({auc_note})"
        )
        lines.append(
            f"- PPV: {optimal['ppv']:.1%} (>= 30% required)"
        )
        lines.append(
            f"- Sensitivity: {optimal['sensitivity']:.1%} "
            f"(>= 50% required)"
        )
        lines.append(
            f"- Calibration: {optimal['calibration_slope']:.2f} "
            f"(0.8-1.2 required)"
        )
        lines.append(
            f"- Utilization: {optimal['utilization']:.1%}"
        )
        lines.append(
            f"- Collision rate: {optimal['collision_rate']:.1%}"
        )
    else:
        lines.append(
            "### No configuration meets all governance criteria."
        )
        # Show closest
        best = min(
            all_results,
            key=lambda r: sum(
                0 if c["passed"] else 1
                for c in r["governance_checks"]
            ),
        )
        n_fail = sum(
            0 if c["passed"] else 1
            for c in best["governance_checks"]
        )
        lines.append(
            f"Closest: AUC={best['model_auc_target']:.2f}, "
            f"thresh={best['threshold']:.2f}, "
            f"policy={best['policy']} "
            f"({n_fail} checks failing)"
        )
    lines.append("")

    # Equity audit
    lines.append("## Equity Audit")
    lines.append("")

    # Use optimal or best config for equity detail
    ref = optimal if passing else best
    for dim, label in [
        ("race_ethnicity", "Race/Ethnicity"),
        ("insurance_type", "Insurance Type"),
        ("age_band", "Age Band"),
    ]:
        groups = ref["subgroup_metrics"].get(dim, {})
        if not groups:
            continue
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            f"| {label} | N | AUC | AUC Gap | PPV "
            f"| Sens | Flag Rate | NoShow Rate | Flag/NS Ratio |"
        )
        lines.append(
            "|---------|---|-----|---------|-----"
            "|------|-----------|-------------|---------------|"
        )
        for g, s in sorted(groups.items()):
            gap = ref["overall_auc"] - s["auc"]
            pct_gap = gap / max(ref["overall_auc"], 0.001)
            flag_icon = ""
            if pct_gap > GOVERNANCE["subgroup_auc_max_gap"]:
                flag_icon = " !!!"
            lines.append(
                f"| {g} | {s['n']} "
                f"| {s['auc']:.3f} "
                f"| {pct_gap:.0%}{flag_icon} "
                f"| {s['ppv']:.1%} "
                f"| {s['sensitivity']:.1%} "
                f"| {s['flag_rate']:.1%} "
                f"| {s['actual_noshow_rate']:.1%} "
                f"| {s['flag_to_noshow_ratio']:.2f} |"
            )
        lines.append("")

    # Fairness summary
    lines.append("### Fairness Rule Checks")
    lines.append("")
    for c in ref["governance_checks"]:
        status = "PASS" if c["passed"] else "**FAIL**"
        lines.append(
            f"- [{status}] {c['rule']}: {c['value']} "
            f"(target: {c['target']})"
        )
    lines.append("")

    n_pass = sum(1 for c in ref["governance_checks"] if c["passed"])
    n_total = len(ref["governance_checks"])
    lines.append(
        f"**{n_pass}/{n_total} governance checks passed.**"
    )
    lines.append("")

    return "\n".join(lines)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = GovernanceConfig()
    output_dir = (
        Path("outputs")
        / f"{config.experiment_name}_{config.timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("Config: %s", json.dumps(asdict(config), indent=2))

    all_results = []
    total = (
        len(config.model_aucs)
        * len(config.thresholds)
        * len(config.policies)
    )
    i = 0
    t0 = time.time()

    for model_auc in config.model_aucs:
        for threshold in config.thresholds:
            for policy in config.policies:
                i += 1
                logger.info(
                    "[%d/%d] AUC=%.2f thresh=%.2f policy=%s",
                    i, total, model_auc, threshold, policy,
                )
                result = run_single_config(
                    config, model_auc, threshold, policy,
                )
                all_results.append(result)
                gc = result["governance_checks"]
                n_p = sum(1 for c in gc if c["passed"])
                status = (
                    "ALL PASS" if result["all_checks_pass"]
                    else f"{n_p}/{len(gc)}"
                )
                logger.info(
                    "  -> AUC=%.3f PPV=%.1%% Sens=%.1%% [%s]",
                    result["overall_auc"],
                    result["ppv"] * 100,
                    result["sensitivity"] * 100,
                    status,
                )

    elapsed = time.time() - t0
    logger.info("Complete in %.1f seconds", elapsed)

    # Save results
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    results_save = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items()}
        results_save.append(r_copy)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_save, f, indent=2, default=str)

    # Generate report
    report = generate_governance_report(config, all_results)
    with open(output_dir / "governance_report.md", "w") as f:
        f.write(report)

    # Register in catalog
    from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog
    n_passing = sum(1 for r in all_results if r["all_checks_pass"])
    catalog = ExperimentCatalog()
    catalog.register(
        output_dir, asdict(config),
        {
            "type": "governance_evaluation",
            "configs_tested": len(all_results),
            "configs_passing": n_passing,
        },
        notes=(
            f"Governance eval: {len(all_results)} configs, "
            f"{n_passing} passing all checks"
        ),
    )

    print(report)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
