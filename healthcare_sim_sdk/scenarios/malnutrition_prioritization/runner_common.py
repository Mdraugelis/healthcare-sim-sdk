"""Shared multi-seed runner machinery + Always-On Verification.

Every experiment reduces each seed's SimulationResults to scalars
IMMEDIATELY and releases the full results object before the next seed
runs (R21) -- full per-entity secondary arrays for both branches are
tens of MB per run and must not accumulate across 200 seeds.
"""

from dataclasses import fields
from typing import Any, Dict, List

import numpy as np

from ...core.engine import (
    BranchedSimulationEngine,
    CounterfactualMode,
)
from ...core.scenario import TimeConfig
from .scenario import (
    MalnutritionConfig,
    MalnutritionPrioritizationScenario,
    SOURCE_ML,
    STATUS_DISCHARGED,
)

_CFG_FIELDS = {f.name for f in fields(MalnutritionConfig)}

SUBGROUP_KEYS = ("ethnicity", "race", "sex", "payer")


def config_from_dict(d: Dict[str, Any]) -> MalnutritionConfig:
    """Build a MalnutritionConfig from a (Hydra) dict, ignoring keys
    that are not config fields and tuple-ifying proportions."""
    kwargs = {}
    for k, v in d.items():
        if k not in _CFG_FIELDS:
            continue
        if isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    return MalnutritionConfig(**kwargs)


def run_once(cfg: MalnutritionConfig, horizon: int, seed: int):
    tc = TimeConfig(
        n_timesteps=horizon,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(horizon)),
    )
    sc = MalnutritionPrioritizationScenario(
        time_config=tc, seed=seed, config=cfg)
    engine = BranchedSimulationEngine(
        sc, CounterfactualMode.BRANCHED)
    return sc, engine.run(sc.n_entities_required)


def reduce_results(res) -> Dict[str, Any]:
    """One seed -> scalar metrics for both arms (R21)."""
    H = len(res.outcomes) - 1
    f = res.outcomes[H].secondary
    c = res.counterfactual_outcomes[H].secondary

    out: Dict[str, Any] = {}
    for arm, s in (("f", f), ("c", c)):
        mal = s["mal_at_discharge"] == 1
        n_mal = max(int(mal.sum()), 1)
        assessed_mal = (s["assessed"] == 1) & mal
        tta = s["time_to_assessment"][assessed_mal]
        discharged = s["status"] == STATUS_DISCHARGED
        n_disch = max(int(discharged.sum()), 1)
        out[f"missed_rate_{arm}"] = float(s["missed"].sum() / n_mal)
        out[f"n_mal_at_discharge_{arm}"] = int(mal.sum())
        out[f"median_tta_{arm}"] = (
            float(np.median(tta)) if len(tta) else -1.0)
        out[f"mean_tta_{arm}"] = (
            float(tta.mean()) if len(tta) else -1.0)
        out[f"harm_total_{arm}"] = float(s["harm"].sum())
        out[f"coded_rate_{arm}"] = float(
            s["coded"][discharged].sum() / n_disch)
        out[f"severe_at_assess_frac_{arm}"] = float(
            (s["severity_at_assess"][assessed_mal] == 2).mean()
            if assessed_mal.sum() else 0.0)
        # Subgroup missed rates (per-arm dict of dicts).
        for key in SUBGROUP_KEYS:
            groups = np.unique(s[key])
            for g in groups:
                gmask = mal & (s[key] == g)
                if gmask.sum() >= 5:
                    out[f"missed_{key}_{g}_{arm}"] = float(
                        s["missed"][gmask].sum() / gmask.sum())
                    out[f"n_{key}_{g}_{arm}"] = int(gmask.sum())
    # Derived contrasts.
    out["delta_missed_pp"] = 100 * (
        out["missed_rate_c"] - out["missed_rate_f"])
    out["delta_tta_days"] = (
        out["mean_tta_c"] - out["mean_tta_f"])
    out["delta_harm"] = out["harm_total_c"] - out["harm_total_f"]
    out["delta_coded_pp"] = 100 * (
        out["coded_rate_f"] - out["coded_rate_c"])
    # ML attribution + censoring.
    src = f["assess_source"][f["assessed"] == 1]
    out["ml_sourced_assessments"] = int((src == SOURCE_ML).sum())
    tm = f["true_malnourished"] == 1
    out["censored_truepos_frac"] = float(
        (tm & (f["status"] != STATUS_DISCHARGED)).sum()
        / max(tm.sum(), 1))
    return out


def reduce_patient_arrays(res) -> Dict[str, np.ndarray]:
    """One seed -> compact per-patient arrays for the outcome overlay
    (exp7 queue stage, run at eff=0).

    Saves everything the overlay needs to re-apply the two-channel
    effect functions at arbitrary efficacy without re-simulating the
    queue: shared disease physics once, per-arm process facts per arm.
    Coins and continuous-LOS leftovers stay float64 (they enter
    bit-exact recomputation); day fields are small integers, exact in
    float32.
    """
    H = len(res.outcomes) - 1
    f = res.outcomes[H].secondary
    c = res.counterfactual_outcomes[H].secondary
    out: Dict[str, np.ndarray] = {
        # Shared physics (arm-invariant at eff=0 by construction).
        "onset_day": np.asarray(f["onset_day"], np.float32),
        "admit_day": np.asarray(f["admit_day"], np.float32),
        "severity": np.asarray(f["severity"], np.int8),
        "u_readmit": np.asarray(f["u_readmit"], np.float64),
        "ml_flag_day": np.asarray(f["ml_flag_day"], np.float32),
        "sex": np.asarray(f["sex"]),
        "race": np.asarray(f["race"]),
        "ethnicity": np.asarray(f["ethnicity"]),
        "payer": np.asarray(f["payer"]),
    }
    for arm, s in (("f", f), ("c", c)):
        out[f"assessed_{arm}"] = np.asarray(s["assessed"], bool)
        out[f"assess_day_{arm}"] = np.asarray(
            s["assess_day"], np.float32)
        out[f"assess_source_{arm}"] = np.asarray(
            s["assess_source"], np.int8)
        out[f"diagnosed_{arm}"] = np.asarray(s["diagnosed"], bool)
        out[f"coded_{arm}"] = np.asarray(s["coded"], bool)
        out[f"discharge_day_{arm}"] = np.asarray(
            s["discharge_day"], np.float32)
        out[f"los_{arm}"] = np.asarray(s["los"], np.float32)
        out[f"los_remaining_{arm}"] = np.asarray(
            s["los_remaining"], np.float64)
        out[f"readmit_{arm}"] = np.asarray(s["readmit"], bool)
        out[f"readmit0_{arm}"] = np.asarray(s["readmit0"], np.float64)
        out[f"status_{arm}"] = np.asarray(s["status"], np.int8)
        out[f"mal_at_discharge_{arm}"] = np.asarray(
            s["mal_at_discharge"], bool)
        out[f"missed_{arm}"] = np.asarray(s["missed"], bool)
    return out


def run_cell(cfg: MalnutritionConfig, horizon: int,
             seeds: List[int]) -> List[Dict[str, Any]]:
    rows = []
    for seed in seeds:
        _, res = run_once(cfg, horizon, seed)
        rows.append(reduce_results(res))  # release res immediately
        rows[-1]["seed"] = seed
    return rows


def summarize(rows: List[Dict[str, Any]],
              keys: List[str]) -> Dict[str, Any]:
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=float)
        if len(vals) == 0:
            continue
        out[k] = {
            "mean": float(vals.mean()),
            "ci_lo": float(np.percentile(vals, 2.5)),
            "ci_hi": float(np.percentile(vals, 97.5)),
            "n_seeds": int(len(vals)),
        }
    return out


# -- Always-On Verification (sim-guide Phase 4) ---------------------------

def always_on_verification(cfg: MalnutritionConfig,
                           horizon: int, seed: int) -> Dict[str, Any]:
    """Structural / statistical / conservation checks on one full run.
    Returns {'checks': [...], 'all_pass': bool}."""
    _, res = run_once(cfg, horizon, seed)
    H = len(res.outcomes) - 1
    f = res.outcomes[H].secondary
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok),
                       "detail": str(detail)})

    # Structural: population conservation + status partition.
    n = len(f["status"])
    statuses_ok = all(
        np.isin(res.outcomes[t].secondary["status"],
                [0, 1, 2]).all()
        and len(res.outcomes[t].secondary["status"]) == n
        for t in range(0, H + 1, max(H // 6, 1)))
    add("population conservation / status partition", statuses_ok)

    # Structural: no NaN in outcome arrays.
    nan_free = all(
        not np.isnan(np.asarray(f[k], dtype=float)).any()
        for k in ("missed", "harm", "coded", "los", "readmit"))
    add("no NaN in outcomes", nan_free)

    # Conservation: event semantics (sum daily events == missed).
    total_events = sum(
        res.outcomes[t].events.sum() for t in range(H + 1))
    add("event semantics (sum events == missed count)",
        total_events == f["missed"].sum(),
        f"{total_events:.0f} vs {f['missed'].sum():.0f}")

    # Conservation: capacity respected both branches.
    cap = cfg.rdn_assessments_per_day
    cap_ok = all(
        res.outcomes[t].metadata["assessed_today"] <= cap
        and res.counterfactual_outcomes[t].metadata[
            "assessed_today"] <= cap
        for t in range(H + 1))
    add("capacity respected (both branches)", cap_ok)

    # Statistical: Arm-C coded prevalence within 4-sigma of the
    # campus target (CLT).
    c = res.counterfactual_outcomes[H].secondary
    discharged = c["status"] == STATUS_DISCHARGED
    n_disch = max(int(discharged.sum()), 1)
    rate = float(c["coded"][discharged].sum() / n_disch)
    target = cfg.coded_prevalence
    sigma = float(np.sqrt(target * (1 - target) / n_disch))
    add("Arm-C coded prevalence 4-sigma CLT",
        abs(rate - target) < 4 * sigma + 0.01,
        f"{rate:.4f} vs {target} (4s={4*sigma:.4f})")

    # Accounting: censored true+ fraction bounded.
    tm = f["true_malnourished"] == 1
    cens = float(
        (tm & (f["status"] != STATUS_DISCHARGED)).sum()
        / max(tm.sum(), 1))
    add("censored true+ fraction < 15%", cens < 0.15,
        f"{cens:.3f}")

    return {"checks": checks,
            "all_pass": all(x["pass"] for x in checks)}


def format_verification(v: Dict[str, Any]) -> str:
    lines = ["## Always-On Verification", ""]
    for c in v["checks"]:
        mark = "PASS" if c["pass"] else "**FAIL**"
        detail = f" ({c['detail']})" if c["detail"] else ""
        lines.append(f"- [{mark}] {c['check']}{detail}")
    lines.append("")
    lines.append(
        "Case walkthroughs: see "
        "`outputs/malnutrition_verification/case_walkthroughs.md` "
        "(Stage B). Model fidelity: Stage A gate "
        "(`stage_a_report.json`).")
    if not v["all_pass"]:
        lines.append(
            "\n**VERIFICATION FAILED -- do not interpret results "
            "below until resolved.**")
    return "\n".join(lines)
