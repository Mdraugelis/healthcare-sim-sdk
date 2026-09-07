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
        verification = verify_results(cfg, res)
        if not verification["all_pass"]:
            raise ValueError(f"Run seed {seed} failed verification: {verification}")
        rows.append(reduce_results(res))  # release res immediately
        rows[-1]["seed"] = seed
    return rows


def summarize(rows: List[Dict[str, Any]],
              keys: List[str]) -> Dict[str, Any]:
    from ...experiments.uncertainty import summarize_metrics
    return summarize_metrics(rows, keys)


# -- Always-On Verification (sim-guide Phase 4) ---------------------------

def verify_results(cfg: MalnutritionConfig, res) -> Dict[str, Any]:
    """Check the actual run on every timestep and both branches.

    Calibration/censoring checks are diagnostics because sweeps deliberately
    change these quantities. Structural failures always block interpretation.
    Infinity is permitted only for the documented never-onset sentinel.
    """
    from ...experiments.review import evidence
    checks = []
    horizon = res.time_config.n_timesteps
    branches = (res.outcomes, res.counterfactual_outcomes)
    complete = all(set(b) == set(range(horizon)) for b in branches)
    checks.append(evidence("timesteps_complete", "PASS" if complete else "FAIL"))
    if not complete:
        return {"checks": checks, "all_pass": False}
    n = res.n_entities
    shapes, statuses, finite, capacity, reconciliation = [], [], [], [], []
    binary = ("assessed", "missed", "coded", "readmit", "diagnosed")
    for branch in branches:
        previous_missed = np.zeros(n)
        for t in range(horizon):
            out = branch[t]
            sec = out.secondary
            shapes.append(len(out.events) == n and out.entity_ids is not None
                          and len(out.entity_ids) == n
                          and len(np.unique(out.entity_ids)) == n
                          and np.array_equal(out.entity_ids, np.arange(n))
                          and all(np.asarray(a).shape == (n,) for a in sec.values()))
            statuses.append(bool(np.isin(sec["status"], [0, 1, 2]).all()))
            for key, values in {"events": out.events, **sec}.items():
                arr = np.asarray(values)
                if arr.dtype.kind not in "biufc":
                    continue
                finite.append(bool((~np.isnan(arr) & ~np.isneginf(arr)).all())
                              if key == "onset_day" else bool(np.isfinite(arr).all()))
            finite.extend(bool(np.isin(sec[k], [0, 1]).all()) for k in binary)
            finite.append(bool(np.isin(out.events, [0, 1]).all()))
            reconciliation.append(np.array_equal(
                out.events, sec["missed"] - previous_missed,
            ))
            previous_missed = sec["missed"]
            assessed_today = int(np.sum(sec["assess_day"] == t))
            capacity.append(0 <= assessed_today <= cfg.rdn_assessments_per_day
                            and assessed_today == out.metadata["assessed_today"]
                            and np.array_equal(sec["assessed"] == 1,
                                               sec["assess_day"] >= 0))
            statuses.append(np.array_equal(sec["assessed"] == 1,
                                           sec["assess_source"] > 0))
        reconciliation.append(sum(out.events.sum() for out in branch.values())
                              == branch[horizon - 1].secondary["missed"].sum())
    for name, values in (("population_accounting", shapes + statuses),
                         ("finite_outcomes", finite),
                         ("capacity", capacity),
                         ("event_reconciliation", reconciliation)):
        checks.append(evidence(name, "PASS" if all(values) else "FAIL",
                               actual=sum(values), expected=len(values)))
    prediction_ok = [set(res.predictions) == set(res.time_config.prediction_schedule)]
    for pred in res.predictions.values():
        if np.asarray(pred.scores).shape != (n,):
            prediction_ok.append(False)
            continue
        scoreable = pred.metadata.get(
            "scoreable", np.ones(len(pred.scores), dtype=bool),
        )
        if np.asarray(scoreable).shape != (n,):
            prediction_ok.append(False)
            continue
        scores = pred.scores[scoreable]
        prediction_ok.append(bool(np.isfinite(scores).all()
                                  and ((scores >= 0) & (scores <= 1)).all()))
    checks.append(evidence(
        "prediction_bounds", "PASS" if all(prediction_ok) else "FAIL",
        detail=("Non-scoreable NaN values are allowed; "
                "scoreable scores must be finite in [0,1]."),
    ))
    c = res.counterfactual_outcomes[horizon - 1].secondary
    discharged = c["status"] == STATUS_DISCHARGED
    count = int(discharged.sum())
    rate = float(c["coded"][discharged].mean()) if count else None
    sigma = (np.sqrt(cfg.coded_prevalence * (1 - cfg.coded_prevalence) / count)
             if count else 0)
    checks.append(evidence("baseline_coding_calibration",
                           "PASS" if rate is not None
                           and abs(rate - cfg.coded_prevalence) < 4 * sigma + 0.01
                           else "INCONCLUSIVE",
                           actual=rate, expected=cfg.coded_prevalence,
                           required=False, category="diagnostic",
                           detail=("Conditional baseline diagnostic; capacity/policy "
                                   "sweeps may intentionally move this rate.")))
    f = res.outcomes[horizon - 1].secondary
    true_positive = f["true_malnourished"] == 1
    count = int(true_positive.sum())
    censored = (
        float((true_positive & (f["status"] != STATUS_DISCHARGED)).sum() / count)
        if count else None
    )
    checks.append(evidence(
        "censoring", "PASS" if censored is not None and censored < .15
        else "INCONCLUSIVE",
        actual=censored, expected="< 0.15", required=False,
        category="diagnostic", detail="Report horizon-dependent incomplete follow-up.",
    ))
    for check in checks:
        check["pass"] = check["status"] == "PASS"
        check["check"] = check["name"]
    return {"checks": checks,
            "all_pass": all(c["pass"] for c in checks if c["required"])}


def always_on_verification(cfg: MalnutritionConfig,
                           horizon: int, seed: int) -> Dict[str, Any]:
    """Compatibility helper for a standalone check; runners check actual runs."""
    _, res = run_once(cfg, horizon, seed)
    return verify_results(cfg, res)


def format_verification(v: Dict[str, Any]) -> str:
    lines = ["## Always-On Verification", ""]
    for c in v["checks"]:
        mark = c.get("status", "PASS" if c.get("pass") else "FAIL")
        detail = f" ({c['detail']})" if c["detail"] else ""
        name = c.get("name", c.get("check", "unnamed"))
        lines.append(f"- [{mark}] {name}{detail}")
    lines.append("")
    lines.append(
        "Case and model certification evidence must match this run's "
        "review packet. A separate default-run report is not current-run evidence.")
    if not v["all_pass"]:
        lines.append(
            "\n**VERIFICATION FAILED -- do not interpret results "
            "below until resolved.**")
    return "\n".join(lines)
