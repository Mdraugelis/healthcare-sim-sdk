"""
Runner script for Paper 04: InSight RCT — Shimabukuro et al. 2017.

Scenario class lives in ``scenario.py``. This module is the CLI /
verification harness around it, including the n=142 variance
analysis that mimics the original underpowered RCT.

Usage:
    cd <repo-root>
    python3 -m healthcare_sim_sdk.scenarios.paper04_insight_rct.run_evaluation
"""

import json
import numpy as np

from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.scenarios.paper04_insight_rct.scenario import InSightRCTScenario


def run_main():
    print("\n=== Paper 04: InSight RCT (Shimabukuro et al. 2017) ===")
    print("Calibrating to: mortality 21.3% control → 8.96% intervention (58% reduction)")
    print("NOTE: AUC not reported in paper — assuming 0.78 [HIGH IMPACT ASSUMPTION]")
    print("NOTE: n=142 in paper — simulation uses n=1000 for stability, n=142 for validation")

    tc = TimeConfig(
        n_timesteps=84,
        timestep_duration=4/24/365,
        timestep_unit="4h",
        prediction_schedule=list(range(84)),
    )

    # Run with n=1000 for stable estimates
    n = 1000
    scenario = InSightRCTScenario(
        time_config=tc,
        seed=42,
        model_auc=0.78,
        treatment_effectiveness=0.58,
        sepsis_prevalence=0.213,
    )

    engine = BranchedSimulationEngine(scenario, counterfactual_mode=CounterfactualMode.BRANCHED)
    results = engine.run(n)

    last_t = tc.n_timesteps - 1
    factual_mort_rate = 0.0
    cf_mort_rate = 0.0
    if last_t in results.outcomes:
        mort = results.outcomes[last_t].secondary["mortality"]
        arm = results.outcomes[last_t].secondary["rct_arm"]
        factual_mort_rate = float(mort.sum()) / n
        # Also compute arm-specific rates
        intervention_arm = arm == 1
        control_arm = arm == 0
        intervention_mort = float(mort[intervention_arm].sum()) / max(intervention_arm.sum(), 1)
        control_mort = float(mort[control_arm].sum()) / max(control_arm.sum(), 1)
        print(f"\n  Arm-specific: Control={control_mort*100:.2f}%, Intervention={intervention_mort*100:.2f}%")
        print(f"  Paper reports: Control=21.3%, Intervention=8.96%")

    if last_t in results.counterfactual_outcomes:
        cf_mort_rate = float(results.counterfactual_outcomes[last_t].secondary["mortality"].sum()) / n

    mortality_delta = cf_mort_rate - factual_mort_rate

    print(f"\n--- RESULTS ---")
    print(f"N patients: {n}")
    print(f"Factual mortality rate (with InSight): {factual_mort_rate:.4f} ({factual_mort_rate*100:.2f}%)")
    print(f"Counterfactual mortality rate (no InSight): {cf_mort_rate:.4f} ({cf_mort_rate*100:.2f}%)")
    print(f"Mortality reduction: {mortality_delta*100:.2f} pp [paper reports 12.34pp: 21.3%→8.96%]")

    if scenario._model._fit_report:
        r = scenario._model._fit_report
        print(f"\nML Model fit report:")
        print(f"  Achieved AUC: {r.get('achieved_auc', 'N/A'):.3f} [assumed target 0.78]")

    # Run with n=142 to show variance (mimicking actual RCT)
    print(f"\n--- VARIANCE ANALYSIS (n=142, mimicking actual RCT) ---")
    mort_deltas = []
    for seed in range(20):
        s2 = InSightRCTScenario(time_config=tc, seed=seed, model_auc=0.78, treatment_effectiveness=0.58, sepsis_prevalence=0.213)
        e2 = BranchedSimulationEngine(s2, counterfactual_mode=CounterfactualMode.BRANCHED)
        r2 = e2.run(142)
        f_mort = float(r2.outcomes[last_t].secondary["mortality"].sum()) / 142 if last_t in r2.outcomes else 0
        cf_mort = float(r2.counterfactual_outcomes[last_t].secondary["mortality"].sum()) / 142 if last_t in r2.counterfactual_outcomes else 0
        mort_deltas.append((cf_mort - f_mort) * 100)
    print(f"  n=142 mortality delta (20 seeds): mean={np.mean(mort_deltas):.2f}pp, std={np.std(mort_deltas):.2f}pp")
    print(f"  Range: [{min(mort_deltas):.2f}, {max(mort_deltas):.2f}] pp")

    # Verification
    print(f"\n--- VERIFICATION ---")
    checks = {}
    sample_scores = results.predictions.get(5)
    if sample_scores:
        checks["no_nan_inf"] = not np.any(np.isnan(sample_scores.scores))
        checks["scores_in_unit_interval"] = bool(np.all(sample_scores.scores >= 0) and np.all(sample_scores.scores <= 1))
    checks["mortality_plausible"] = 0.001 < factual_mort_rate < 0.50
    checks["intervention_direction"] = mortality_delta >= 0
    print(f"  No NaN/Inf: {checks.get('no_nan_inf')}")
    print(f"  Scores in [0,1]: {checks.get('scores_in_unit_interval')}")
    print(f"  Mortality plausible: {checks['mortality_plausible']}")
    print(f"  Intervention direction correct: {checks['intervention_direction']}")
    print(f"  OVERALL: {'PASS' if all(checks.values()) else 'PARTIAL'} ({sum(checks.values())}/{len(checks)})")

    return {
        "factual_mortality_rate": factual_mort_rate,
        "cf_mortality_rate": cf_mort_rate,
        "mortality_delta_pp": mortality_delta * 100,
        "rct_variance_std_pp": float(np.std(mort_deltas)),
        "checks": checks,
    }


if __name__ == "__main__":
    r = run_main()
    print(f"\n=== SUMMARY ===")
    print(json.dumps(r, indent=2))
