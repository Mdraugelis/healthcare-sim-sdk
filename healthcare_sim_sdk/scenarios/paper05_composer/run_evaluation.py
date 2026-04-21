"""
Runner script for Paper 05: COMPOSER — Boussina/Wardi et al. npj Digital Medicine 2024.

Scenario class lives in ``scenario.py``. This module is the CLI /
verification harness around it.

Usage:
    cd <repo-root>
    python3 -m healthcare_sim_sdk.scenarios.paper05_composer.run_evaluation
"""

import json
import numpy as np

from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.scenarios.paper05_composer.scenario import COMPOSERScenario


def run_main():
    print("\n=== Paper 05: COMPOSER (Boussina/Wardi et al. 2024) ===")
    print("Calibrating to: 1.9pp mortality reduction, 5.0pp bundle compliance increase")
    print("NOTE: AUC not explicitly reported — assumed 0.81 [HIGH IMPACT ASSUMPTION]")
    print("NOTE: Pre-intervention mortality baseline assumed ~11% [HIGH IMPACT ASSUMPTION]")

    tc = TimeConfig(
        n_timesteps=84,
        timestep_duration=4/24/365,
        timestep_unit="4h",
        prediction_schedule=list(range(84)),
    )

    n = 5000
    scenario = COMPOSERScenario(
        time_config=tc,
        seed=42,
        model_auc=0.81,
        treatment_effectiveness=0.17,
        bundle_compliance_boost=0.05,
        baseline_bundle_compliance=0.50,
        sepsis_prevalence=0.112,
        nurse_response_rate=0.72,
    )

    engine = BranchedSimulationEngine(scenario, counterfactual_mode=CounterfactualMode.BRANCHED)
    results = engine.run(n)

    last_t = tc.n_timesteps - 1
    factual_mort_rate = 0.0
    cf_mort_rate = 0.0
    factual_bundle = 0.0
    cf_bundle = 0.0

    if last_t in results.outcomes:
        factual_mort_rate = float(results.outcomes[last_t].secondary["mortality"].sum()) / n
        factual_bundle = float(results.outcomes[last_t].secondary["bundle_compliant"].sum()) / n
    if last_t in results.counterfactual_outcomes:
        cf_mort_rate = float(results.counterfactual_outcomes[last_t].secondary["mortality"].sum()) / n
        cf_bundle = float(results.counterfactual_outcomes[last_t].secondary["bundle_compliant"].sum()) / n

    mortality_delta = cf_mort_rate - factual_mort_rate
    bundle_delta = factual_bundle - cf_bundle

    # Alert rate
    n_flagged_total = 0
    n_active_total = 0
    for t in range(tc.n_timesteps):
        if t in results.predictions:
            meta = results.predictions[t].metadata
            n_flagged_total += meta.get("n_flagged", 0)
            n_active_total += meta.get("n_active", 0)
    avg_alert_rate = n_flagged_total / max(n_active_total, 1)

    print(f"\n--- RESULTS ---")
    print(f"N patients: {n}")
    print(f"Factual mortality rate (with COMPOSER): {factual_mort_rate:.4f} ({factual_mort_rate*100:.2f}%)")
    print(f"Counterfactual mortality rate (no COMPOSER): {cf_mort_rate:.4f} ({cf_mort_rate*100:.2f}%)")
    print(f"Mortality reduction: {mortality_delta*100:.2f} pp [paper reports 1.9pp]")
    print(f"")
    print(f"Factual bundle compliance: {factual_bundle:.4f} ({factual_bundle*100:.2f}%)")
    print(f"Counterfactual bundle compliance: {cf_bundle:.4f} ({cf_bundle*100:.2f}%)")
    print(f"Bundle compliance increase: {bundle_delta*100:.2f} pp [paper reports 5.0pp]")
    print(f"Average alert rate: {avg_alert_rate*100:.2f}%")

    if scenario._model._fit_report:
        r = scenario._model._fit_report
        print(f"\nML Model fit report:")
        print(f"  Achieved AUC: {r.get('achieved_auc', 'N/A'):.3f} [assumed 0.81]")

    # Verification
    print(f"\n--- VERIFICATION ---")
    checks = {}
    sample_scores = results.predictions.get(5)
    if sample_scores:
        checks["no_nan_inf"] = not np.any(np.isnan(sample_scores.scores))
        checks["scores_in_unit_interval"] = bool(np.all(sample_scores.scores >= 0) and np.all(sample_scores.scores <= 1))
    checks["mortality_plausible"] = 0.001 < factual_mort_rate < 0.30
    checks["intervention_direction"] = mortality_delta >= 0
    checks["bundle_direction"] = bundle_delta >= 0
    print(f"  No NaN/Inf: {checks.get('no_nan_inf')}")
    print(f"  Scores in [0,1]: {checks.get('scores_in_unit_interval')}")
    print(f"  Mortality plausible: {checks['mortality_plausible']}")
    print(f"  Intervention reduces mortality: {checks['intervention_direction']}")
    print(f"  Intervention improves bundle compliance: {checks['bundle_direction']}")
    print(f"  OVERALL: {'PASS' if all(checks.values()) else 'PARTIAL'} ({sum(checks.values())}/{len(checks)})")

    return {
        "factual_mortality_rate": factual_mort_rate,
        "cf_mortality_rate": cf_mort_rate,
        "mortality_delta_pp": mortality_delta * 100,
        "bundle_delta_pp": bundle_delta * 100,
        "avg_alert_rate": avg_alert_rate,
        "checks": checks,
    }


if __name__ == "__main__":
    r = run_main()
    print(f"\n=== SUMMARY ===")
    print(json.dumps(r, indent=2))
