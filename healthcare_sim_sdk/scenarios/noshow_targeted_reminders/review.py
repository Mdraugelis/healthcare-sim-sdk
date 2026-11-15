"""Review-packet evidence for one targeted-reminder run.

Every check below is computed from the run being reviewed. Nothing here
changes the scenario's behavior or its random-number use: the reviewed run
is wrapped in ``TracingScenario``, which observes without drawing.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ...core.engine import BranchedSimulationEngine, CounterfactualMode
from ...core.rng import RNGPartitioner
from ...core.scenario import TimeConfig
from ...experiments.review import evidence
from ...experiments.trace import TracingScenario
from .scenario import CallerConfig, NoShowTargetedReminderScenario

REGISTRY = Path(__file__).parent / "configs" / "review_claims.yaml"

# Offset keeps case selection independent of the simulation's own streams.
_CASE_SEED_OFFSET = 7_000_000


def _time_config(horizon: int, with_predictions: bool = True) -> TimeConfig:
    return TimeConfig(
        n_timesteps=horizon, timestep_duration=1 / 365, timestep_unit="day",
        prediction_schedule=list(range(horizon)) if with_predictions else [],
    )


def build_scenario(cell: Dict[str, Any], defaults: Dict[str, Any],
                   n_patients: int, horizon: int, seed: int,
                   with_predictions: bool = True,
                   effectiveness_override=None):
    caller = dict(cell.get("caller", {}))
    if effectiveness_override is not None:
        caller["reminder_effectiveness"] = effectiveness_override
    return NoShowTargetedReminderScenario(
        time_config=_time_config(horizon, with_predictions), seed=seed,
        n_patients=n_patients, caller_config=CallerConfig(**caller),
        **defaults,
    )


def _project(state) -> Dict[str, Any]:
    return {
        "day": state.day,
        "scheduled": len(state.schedule),
        "resolved_total": state.total_slots_resolved,
        "noshows_total": state.total_noshows,
        "calls_made_total": state.total_calls_made,
        "calls_reached_total": state.total_calls_reached,
    }


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _mismatched_steps(first, second) -> Tuple[int, List[str]]:
    bad_steps, bad_keys = 0, set()
    for t in first:
        if t not in second or not _same(first[t].events, second[t].events):
            bad_steps += 1
            continue
        for key, value in first[t].secondary.items():
            if not _same(value, second[t].secondary.get(key)):
                bad_keys.add(key)
    if set(first) != set(second):
        bad_steps += len(set(first) ^ set(second))
    return bad_steps, sorted(bad_keys)


def _final(outcomes) -> Dict[str, Any]:
    return outcomes[max(outcomes)].metadata


def _select_cases(results, seed: int, cell_id: str) -> List[Dict[str, Any]]:
    """One seeded random slot per category, from the recorded branches."""
    rng = RNGPartitioner(seed + _CASE_SEED_OFFSET).create_streams().population
    categories = {
        "helped": [], "reached_no_effect": [], "missed_noshow": [],
    }
    for t in sorted(results.outcomes):
        f, cf = results.outcomes[t], results.counterfactual_outcomes[t]
        reached = f.secondary["was_reached"]
        called = f.secondary["was_called"]
        for i, slot_id in enumerate(f.entity_ids):
            noshow_f, noshow_cf = f.events[i], cf.events[i]
            if reached[i] and not noshow_f and noshow_cf:
                categories["helped"].append((t, i, int(slot_id)))
            elif reached[i] and noshow_f:
                categories["reached_no_effect"].append((t, i, int(slot_id)))
            elif not called[i] and noshow_f:
                categories["missed_noshow"].append((t, i, int(slot_id)))
    stories = {
        "helped": ("was called and reached, kept the appointment, and missed "
                   "it on the no-reminder branch."),
        "reached_no_effect": ("was called and reached, and still missed the "
                              "appointment."),
        "missed_noshow": ("was not called (outside the day's call list) and "
                          "missed the appointment on both branches."),
    }
    cases = []
    for category, candidates in categories.items():
        if not candidates:
            cases.append({
                "category": category, "status": "NOT_RUN",
                "selection": "seeded random draw within category",
                "candidate_count": 0,
                "narrative": "No slot in this run fits the category.",
            })
            continue
        t, i, slot_id = candidates[int(rng.integers(len(candidates)))]
        f = results.outcomes[t]
        cases.append({
            "category": category, "status": "PASS",
            "selection": ("seeded random draw within category "
                          f"(seed {seed + _CASE_SEED_OFFSET})"),
            "candidate_count": len(candidates),
            "narrative": (
                f"Slot {slot_id}, resolved on day {t}, patient group "
                f"{f.secondary['race_ethnicity'][i]} / "
                f"{f.secondary['insurance_type'][i]} / "
                f"{f.secondary['age_band'][i]}, {stories[category]}"
            ),
            "artifact": f"traces/{cell_id}_seed{seed}.jsonl",
        })
    return cases


def run_reviewed(cell_id: str, cell: Dict[str, Any], config: Dict[str, Any],
                 seed: int, output_dir: Path):
    """Run one cell/seed and return (metrics, checks, cases, artifacts)."""
    horizon = config["horizon_days"]
    n = config["n_patients"]
    defaults = config["scenario_defaults"]

    inner = build_scenario(cell, defaults, n, horizon, seed)
    traced = TracingScenario(
        inner, _project,
        stages=("population", "step_after", "intervention", "outcome"),
    )
    results = BranchedSimulationEngine(traced).run(n)

    trace_name = f"traces/{cell_id}_seed{seed}.jsonl"
    trace_path = Path(output_dir) / trace_name
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text("\n".join(
        json.dumps(row, sort_keys=True) for row in traced.to_rows()) + "\n")

    checks = []

    # The no-AI branch must equal a world where the model never runs.
    no_ai = BranchedSimulationEngine(
        build_scenario(cell, defaults, n, horizon, seed,
                       with_predictions=False),
        CounterfactualMode.NONE,
    ).run(n)
    bad, keys = _mismatched_steps(results.counterfactual_outcomes,
                                  no_ai.outcomes)
    checks.append(evidence(
        "counterfactual_is_no_ai_world", "PASS" if not bad and not keys
        else "FAIL", actual={"mismatched_steps": bad, "fields": keys},
        expected="identical slot outcomes on every day",
        detail="Counterfactual branch versus a separate run with an empty "
               "prediction schedule."))

    # Common random numbers: both branches see the same slots and patients.
    shared = all(
        _same(results.outcomes[t].entity_ids,
              results.counterfactual_outcomes[t].entity_ids)
        and all(_same(results.outcomes[t].secondary[k],
                      results.counterfactual_outcomes[t].secondary[k])
                for k in ("race_ethnicity", "insurance_type", "age_band"))
        for t in results.outcomes)
    checks.append(evidence(
        "branches_share_schedule", "PASS" if shared else "FAIL",
        detail="Slot IDs and patient groups match on every day."))

    capacity = cell["caller"]["call_capacity_per_day"]
    daily_calls = [len(results.interventions[t].treated_indices)
                   for t in sorted(results.interventions)]
    cf_final, f_final = (_final(results.counterfactual_outcomes),
                         _final(results.outcomes))
    capacity_ok = (max(daily_calls, default=0) <= capacity
                   and cf_final["total_calls_made"] == 0)
    checks.append(evidence(
        "call_capacity_respected", "PASS" if capacity_ok else "FAIL",
        actual={"max_calls_per_day": max(daily_calls, default=0),
                "no_program_calls": cf_final["total_calls_made"]},
        expected={"max_calls_per_day": f"<= {capacity}",
                  "no_program_calls": 0},
        category="operations"))

    accounting = {}
    for name, outcomes in (("program", results.outcomes),
                           ("no_program", results.counterfactual_outcomes)):
        meta = _final(outcomes)
        accounting[name] = {
            "events_sum": float(sum(o.events.sum() for o in outcomes.values())),
            "slots_sum": int(sum(len(o.events) for o in outcomes.values())),
            "total_noshows": meta["total_noshows"],
            "total_resolved": meta["total_resolved"],
        }
    reached_resolved = (f_final["total_reminded_showed"]
                        + f_final["total_reminded_noshowed"])
    accounting_ok = all(
        a["events_sum"] == a["total_noshows"]
        and a["slots_sum"] == a["total_resolved"]
        for a in accounting.values()
    ) and (accounting["program"]["total_resolved"]
           == accounting["no_program"]["total_resolved"]) and (
        reached_resolved <= f_final["total_calls_reached"]
        <= f_final["total_calls_made"])
    checks.append(evidence(
        "outcome_accounting", "PASS" if accounting_ok else "FAIL",
        actual=accounting,
        detail="Per-day outcomes sum to the running totals on each branch; "
               "both branches resolve the same slots; reached <= made."))

    zero = BranchedSimulationEngine(
        build_scenario(cell, defaults, n, horizon, seed,
                       effectiveness_override=0.0)).run(n)
    bad_zero, _ = _mismatched_steps(zero.outcomes, zero.counterfactual_outcomes)
    checks.append(evidence(
        "zero_effect_identity", "PASS" if bad_zero == 0 else "FAIL",
        actual={"mismatched_steps": bad_zero},
        detail="With reminder_effectiveness 0, no-show outcomes are "
               "identical on both branches."))

    traced_totals = {
        r.timestep: json.loads(r.facts_json)["event_total"]
        for r in traced.records if r.stage == "outcome"
        and r.branch == "factual"}
    trace_ok = traced_totals and all(
        traced_totals.get(t) == float(results.outcomes[t].events.sum())
        for t in results.outcomes)
    checks.append(evidence(
        "trace_reconciles", "PASS" if trace_ok else "FAIL",
        detail="Factual outcome totals in the trace equal the recorded "
               "results on every day.", category="engineering",
        artifacts=[trace_name]))

    resolved = accounting["no_program"]["total_resolved"]
    rate_cf = accounting["no_program"]["total_noshows"] / resolved
    target = defaults["base_noshow_rate"]
    se = float(np.sqrt(target * (1 - target) / resolved))
    z = (rate_cf - target) / se
    checks.append(evidence(
        "no_program_rate_matches_config",
        "PASS" if abs(z) <= 4 else "FAIL",
        actual={"rate": rate_cf, "z": z, "resolved_slots": resolved},
        expected={"rate": target, "tolerance": "4 binomial SE"},
        detail="Binomial SE ignores clustering of slots within patients, "
               "so it is slightly too narrow.", category="statistics"))

    cases = _select_cases(results, seed, cell_id)
    checks.append(evidence(
        "case_traces",
        "PASS" if any(c["status"] == "PASS" for c in cases) else "NOT_RUN",
        detail="One seeded random slot per category; absent categories "
               "are listed.", category="inspectability",
        artifacts=[trace_name]))

    rate_f = accounting["program"]["total_noshows"] / resolved
    metrics = {
        "noshow_rate_program_pct": 100 * rate_f,
        "noshow_rate_no_program_pct": 100 * rate_cf,
        "reduction_pp": 100 * (rate_cf - rate_f),
        "noshows_prevented": (accounting["no_program"]["total_noshows"]
                              - accounting["program"]["total_noshows"]),
        "calls_made": f_final["total_calls_made"],
    }
    return metrics, checks, cases, [trace_name]
