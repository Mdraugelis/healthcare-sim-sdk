"""Verification gates for the malnutrition prioritization scenario.

Stage A (model-only; requires the gitignored configs/model_targets.yaml
-- see model_targets.example.yaml):
  1. Calibrate the subgroup-differential scorer (deterministic) and
     FREEZE the params into configs/model_frozen.yaml (gitignored).
  2. Target-reproduction gate:
     - calibration cohort: |AUROC - target| <= 0.005, lifts +/- 0.05
     - independent cohort: capture@K +/- 0.03, P@K +/- 0.04,
       sens/PPV at the deployed threshold +/- 0.04
     Independent-cohort lift realizations are reported informational
     (prevalence- and margin-size-sensitive).

Stage B (full-sim boundary gate; requires intervene/erosion/coding):
  Boundary-condition mini-sims per the design doc + narrative case
  walkthroughs. Stage B green is THE verification gate -- no
  experiment may be interpreted before it passes.

Usage:
  python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization\
.run_verification --stage A
"""

import argparse
import json
import sys
from pathlib import Path

from .subgroup_performance import (
    CALIBRATION_SEED,
    TOL_AUROC,
    TOL_CAPTURE,
    TOL_LIFT,
    TOL_PRECISION,
    FrozenSubgroupScorer,
    _raw_scores,
    _sigmoid,
    build_calibration_cohort,
    calibrate,
    evaluate,
    freeze,
    load_targets,
)

FROZEN_PATH = Path(__file__).parent / "configs" / "model_frozen.yaml"
REPORT_DIR = Path("outputs") / "malnutrition_verification"


def _check(name, achieved, target, tol):
    ok = abs(achieved - target) <= tol
    return {
        "check": name,
        "achieved": round(float(achieved), 4),
        "target": target,
        "tolerance": tol,
        "pass": bool(ok),
    }


def stage_a(recalibrate: bool = True) -> bool:
    print("=" * 70)
    print("Stage A -- frozen-model calibration + target "
          "reproduction gate")
    print("=" * 70)

    targets = load_targets()
    from .scenario import MalnutritionConfig
    g_default = MalnutritionConfig().severity_score_gradient

    if recalibrate or not FROZEN_PATH.exists():
        print("\nCalibrating (deterministic, ~1-2 min at n=100k)...")
        params, achieved = calibrate(targets)
        freeze(params, achieved, targets, str(FROZEN_PATH))
        print(f"Frozen params written to {FROZEN_PATH}")
    scorer = FrozenSubgroupScorer.load(str(FROZEN_PATH))
    params = scorer.params

    rows = []

    # -- Gate 1: calibration cohort, WITH the default severity
    #    gradient active (the deployed scorer runs with it; gating
    #    without it would certify a configuration nobody runs).
    cal = build_calibration_cohort(
        prevalence=targets.calibration_prevalence)
    m_cal = evaluate(params, cal, targets, gradient=g_default)
    rows.append(_check("cal AUROC", m_cal["auroc"],
                       targets.auroc, TOL_AUROC))
    for k, t in targets.lifts.items():
        rows.append(_check(f"cal lift {k}", m_cal[f"lift_{k}"],
                           t, TOL_LIFT))

    # -- Gate 2: independent cohort (curve fidelity, out-of-sample) ---
    ev = build_calibration_cohort(
        prevalence=targets.calibration_prevalence,
        seed=CALIBRATION_SEED + 1)
    m_ev = evaluate(params, ev, targets, gradient=g_default)
    for kf, tv in targets.capture.items():
        rows.append(_check(f"eval capture@{kf*100:.1f}%",
                           m_ev["capture"][kf], tv, TOL_CAPTURE))
    for kf, tv in targets.precision.items():
        rows.append(_check(f"eval P@{kf*100:.1f}%",
                           m_ev["precision"][kf], tv, TOL_PRECISION))

    # Sens/PPV at the deployed threshold via the probability map.
    z = _raw_scores(params, ev, gradient=g_default)
    scores = _sigmoid(params.map_a * z + params.map_b)
    flagged = scores >= targets.threshold_anchor
    y = ev["y"]
    sens = float(y[flagged].sum() / y.sum())
    ppv = float(y[flagged].mean())
    flag_rate = float(flagged.mean())
    rows.append(_check("eval sens@threshold", sens,
                       targets.sens, TOL_PRECISION))
    rows.append(_check("eval PPV@threshold", ppv,
                       targets.ppv, TOL_PRECISION))
    rows.append(_check("eval flag rate@threshold", flag_rate,
                       targets.flag_rate_anchor, 0.02))

    # -- Report ---------------------------------------------------------
    print(f"\n{'check':<26} {'achieved':>9} {'target':>8} "
          f"{'tol':>6}  result")
    for r in rows:
        print(f"{r['check']:<26} {r['achieved']:>9.4f} "
              f"{r['target']:>8.3f} {r['tolerance']:>6.3f}  "
              f"{'PASS' if r['pass'] else 'FAIL'}")

    info = {
        "eval_auroc": round(m_ev["auroc"], 4),
        "eval_auprc": round(m_ev["auprc"], 4),
        "eval_lift_female": round(m_ev["lift_female"], 3),
        "eval_lift_black": round(m_ev["lift_black"], 3),
        "eval_lift_hl": round(m_ev["lift_hl"], 3),
        "eval_lift_male": round(m_ev["lift_male"], 3),
    }
    print("\nInformational (independent cohort; lifts are margin-"
          "size sensitive):")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("  note: male lift is emergent (not calibrated).")

    # Report-only fidelity at swept gradient values (never gated).
    for g in (0.0, 0.6):
        m_g = evaluate(params, ev, targets, gradient=g)
        print(f"  report-only g={g}: AUROC {m_g['auroc']:.4f} "
              f"AUPRC {m_g['auprc']:.4f} "
              f"capture@10% {m_g['capture'].get(0.10, float('nan')):.3f}")

    all_pass = all(r["pass"] for r in rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage_a_report.json").write_text(json.dumps(
        {"gates": rows, "informational": info,
         "all_pass": all_pass}, indent=2))
    print(f"\nStage A: {'PASS' if all_pass else 'FAIL'} "
          f"(report: {REPORT_DIR / 'stage_a_report.json'})")
    return all_pass


def stage_b() -> bool:
    """Boundary-condition mini-sims (see boundary_checks.py)."""
    from .boundary_checks import run_boundary_suite
    return run_boundary_suite()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "all"],
                    default="A")
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="use existing model_frozen.yaml")
    args = ap.parse_args()

    ok = True
    if args.stage in ("A", "all"):
        ok &= stage_a(recalibrate=not args.no_recalibrate)
    if args.stage in ("B", "all"):
        ok &= stage_b()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
