"""Evaluator-only scoring of committed predictions against sealed evidence.

This module has no runner dependency. Empirical distributions describe the
submitted study statistic; no additional observation variance is added here.
"""

from numbers import Integral, Real
from typing import Any

import numpy as np


def _scalar(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Expected a finite numeric scalar")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("Expected a finite numeric scalar")
    return number


def _samples(values: Any) -> np.ndarray:
    numbers: np.ndarray = np.asarray([_scalar(value) for value in values], dtype=float)
    if numbers.ndim != 1 or not numbers.size:
        raise ValueError("Expected nonempty scalar samples")
    return numbers


def empirical_crps(samples: Any, observation: float) -> float:
    """CRPS of the equally weighted empirical CDF, in O(n log n) time.

    This is E|X-y| - E|X-X'|/2 for independent draws from the empirical
    distribution, including its diagonal pairs; it is not fair/adjusted CRPS.
    """
    x = np.sort(_samples(samples))
    y = _scalar(observation)
    n = len(x)
    # Centering avoids cancellation of a large common location in the pair sum.
    with np.errstate(over="ignore", invalid="ignore"):
        centered = x - x[0]
        weights = 2.0 * np.arange(n) - n + 1.0
        spread = float(np.dot(weights, centered) / (float(n) ** 2))
        score = float(np.mean(np.abs(x - y)) - spread)
    if not np.isfinite(score):
        raise ValueError("CRPS arithmetic overflowed")
    return max(0.0, score)  # Roundoff can be slightly negative at zero.


def interval_score(
    lower: float, upper: float, observation: float, alpha: float = 0.1,
) -> float:
    """Central interval score: width plus 2/alpha times distance outside."""
    lo, hi, y, a = map(_scalar, (lower, upper, observation, alpha))
    if not 0 < a < 1 or lo > hi:
        raise ValueError("Require 0 < alpha < 1 and lower <= upper")
    score = hi - lo + 2.0 / a * max(lo - y, y - hi, 0.0)
    if not np.isfinite(score):
        raise ValueError("Interval score arithmetic overflowed")
    return score


def _summary(values: list, target: float, alpha: float) -> dict:
    x = _samples(values)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = _scalar(float(x.mean()))
        sd = _scalar(float(x.std(ddof=1)))
        lo, hi = np.quantile(x, [alpha / 2, 1 - alpha / 2])
        error = _scalar(mean - target)
    return {
        "n_runs": len(x), "mean": mean, "signed_error": error,
        "absolute_error": abs(error), "seed_sd": sd,
        "mcse_mean": _scalar(sd / np.sqrt(len(x))),
        "predictive_interval": {
            "alpha": alpha, "lower": _scalar(lo), "upper": _scalar(hi),
            "contains_reference": bool(lo <= target <= hi),
            "method": "equal-tailed empirical quantiles; linear interpolation",
        },
        "interval_score": interval_score(lo, hi, target, alpha),
        "crps": empirical_crps(x, target),
    }


def _input_adequacy(case: dict, endpoint_ids: set) -> dict:
    """Trace only input-used provenance; unused source records cannot pass/fail."""
    sources, inputs = case.get("sources", {}), case.get("inputs", {})
    reasons: list[str] = []
    visited: set[str] = set()
    uncertain = False
    failed = False

    def visit(source_id: Any, ancestors: set) -> None:
        nonlocal uncertain, failed
        if not isinstance(source_id, str) or source_id not in sources:
            uncertain = True
            reasons.append(f"Missing input source: {source_id!r}")
            return
        if source_id in ancestors:
            failed = True
            reasons.append(f"Cyclic source derivation: {source_id}")
            return
        if source_id in visited:
            return
        visited.add(source_id)
        source = sources[source_id]
        if not isinstance(source, dict):
            uncertain = True
            reasons.append(f"Malformed source: {source_id}")
            return
        kind = source.get("kind")
        if kind in {"assumed", "development"}:
            uncertain = True
            reasons.append(f"{source_id}: {kind} input needs adequacy review")
        elif kind == "target_outcome":
            target_ids = source.get("target_ids", [])
            released = case.get("released_targets", [])
            allowed = (
                case.get("information_track") == "conditional"
                and isinstance(target_ids, list) and bool(target_ids)
                and all(isinstance(t, str) for t in target_ids)
                and set(target_ids).issubset(set(released))
                and not set(target_ids).intersection(endpoint_ids)
            )
            if not allowed:
                failed = True
                reasons.append(f"{source_id}: prohibited target-derived input")
            else:
                reasons.append(f"{source_id}: explicitly released context only")
        elif kind not in {"protocol", "external"}:
            uncertain = True
            reasons.append(f"{source_id}: unknown provenance kind")
        if not source.get("citation") or not source.get("available_at"):
            uncertain = True
            reasons.append(f"{source_id}: missing citation or availability date")
        derived = source.get("derived_from", [])
        if not isinstance(derived, list):
            failed = True
            reasons.append(f"{source_id}: malformed source dependencies")
            return
        for parent in derived:
            visit(parent, ancestors | {source_id})

    if not isinstance(inputs, dict) or not isinstance(sources, dict):
        return {"status": "INCONCLUSIVE", "reasons": ["Missing input registry"]}
    if not inputs:
        uncertain = True
        reasons.append("No sourced inputs supplied")
    for name, entry in inputs.items():
        if not isinstance(entry, dict) or "value" not in entry:
            uncertain = True
            reasons.append(f"{name}: missing input value/source record")
        visit(entry.get("source_id") if isinstance(entry, dict) else None, set())
    return {
        "status": "FAIL" if failed else "INCONCLUSIVE" if uncertain else "PASS",
        "used_sources": sorted(visited), "reasons": reasons,
        "scope": "Provenance admissibility, not proof of clinical adequacy",
    }


def evaluate(case: dict, submission: dict, reference: dict) -> dict:
    """Score all committed runs without dropping failed/missing repetitions.

    Malformed scoring specifications raise ValueError. Incomplete or invalid
    submitted evidence yields explicit FAIL and no numerical scores. Only the
    preregistered primary endpoint determines numerical fit; secondary results
    remain descriptive. Human and independent validation are never automated.
    """
    specs = case.get("endpoints", [])
    if not isinstance(specs, list) or not specs:
        raise ValueError("Case requires endpoint specifications")
    endpoints = {}
    for spec in specs:
        eid = spec.get("id")
        if not isinstance(eid, str) or not eid or eid in endpoints:
            raise ValueError("Endpoint IDs must be unique nonempty strings")
        lo, hi = _scalar(spec["lower_bound"]), _scalar(spec["upper_bound"])
        if lo > hi or not spec.get("units"):
            raise ValueError("Endpoint requires ordered bounds and units")
        endpoints[eid] = (lo, hi, spec["units"])
    ids = set(endpoints)
    scoring = case["scoring"]
    primary = scoring["primary_endpoint"]
    alpha = _scalar(scoring["interval_alpha"])
    tolerance = _scalar(scoring["absolute_error_tolerance"])
    minimum_skill = _scalar(scoring["minimum_skill"])
    if primary not in ids or not 0 < alpha < 1 or tolerance < 0:
        raise ValueError("Invalid preregistered scoring specification")
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL",
                       "detail": detail})

    case_id, study_id = case.get("case_id"), case.get("study", {}).get("id")
    check("reference_identity", bool(case_id and study_id)
          and reference.get("case_id") == case_id
          and reference.get("study_id") == study_id,
          "Reference must match case_id and study.id exactly")
    check("reference_exposure", reference.get("exposure") in {"published", "synthetic"},
          "Reference exposure must be published or synthetic")
    targets = reference.get("targets", {})
    check("reference_endpoints", isinstance(targets, dict) and set(targets) == ids,
          "Reference endpoint set must equal the preregistered endpoint set")
    for eid, (lo, hi, units) in endpoints.items():
        target = targets.get(eid, {}) if isinstance(targets, dict) else {}
        try:
            value = _scalar(target["value"])
            valid = lo <= value <= hi and bool(target.get("source"))
            valid = valid and target.get("units", units) == units
        except (ValueError, TypeError, KeyError):
            valid = False
        check(f"reference:{eid}", valid,
              "Require finite in-range value, source and matching declared units")
    runs = submission.get("runs", [])
    check("run_count", isinstance(runs, list) and len(runs) >= 2,
          "At least two complete independent repetitions are required")
    runs = runs if isinstance(runs, list) else []
    seeds = [r.get("seed") if isinstance(r, dict) else None for r in runs]
    valid_seeds = all(isinstance(s, Integral) and not isinstance(s, bool)
                      and int(s) >= 0 for s in seeds)
    check("unique_seeds", valid_seeds and len(set(seeds)) == len(seeds)
          if valid_seeds else False, "Run seeds must be unique nonnegative integers")
    expected = case.get("run_plan", {}).get("seeds")
    if expected is not None:
        check("planned_seeds", valid_seeds and len(expected) == len(seeds)
              and set(expected) == set(seeds), "Every planned seed must be present")
    execution = submission.get("execution")
    if execution is not None:
        check("execution", isinstance(execution, dict)
              and execution.get("status") == "PASS", "Execution must have passed")
    pending_checks = False
    retained = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            check(f"run:{index}", False, "Run record must be an object")
            retained.append({"index": index, "status": "INVALID"})
            continue
        seed = run.get("seed")
        retained.append({"index": index,
                         "seed": int(seed) if isinstance(seed, Integral) else str(seed),
                         "status": str(run.get("status", "MISSING"))})
        check(f"run:{index}:status", run.get("status") == "PASS",
              "Failed runs are retained and prevent numerical scoring")
        predictions, baselines = run.get("predictions"), run.get("baselines")
        check(f"run:{index}:endpoints", isinstance(predictions, dict)
              and set(predictions) == ids, "Exact prediction endpoint set required")
        check(f"run:{index}:baselines", isinstance(baselines, dict)
              and primary in baselines and set(baselines).issubset(ids),
              "Primary baseline required; secondary baselines optional")
        for label, values in (("prediction", predictions), ("baseline", baselines)):
            if not isinstance(values, dict):
                continue
            for eid, value in values.items():
                try:
                    number = _scalar(value)
                    valid = (eid in ids
                             and endpoints[eid][0] <= number <= endpoints[eid][1])
                except (TypeError, ValueError):
                    valid = False
                check(f"run:{index}:{label}:{eid}", valid,
                      "Require finite scalar inside endpoint bounds")
        run_checks = run.get("checks", [])
        if not isinstance(run_checks, list):
            check(f"run:{index}:checks", False, "Checks must be a list")
            continue
        for item in run_checks:
            status = item.get("status") if isinstance(item, dict) else None
            name = item.get("name") if isinstance(item, dict) else "?"
            detail = (str(item.get("detail", "")) if isinstance(item, dict)
                      else "Invalid check")
            check(f"run:{index}:check:{name}",
                  status in {"PASS", "INCONCLUSIVE", "NOT_RUN", "NOT_APPLICABLE"},
                  detail)
            pending_checks |= status in {"INCONCLUSIVE", "NOT_RUN"}
    adequate = _input_adequacy(case, ids)
    exposure = case.get("exposure")
    blinding = {
        "status": "NOT_APPLICABLE" if exposure == "outcome_exposed_development"
        else "INCONCLUSIVE",
        "declared_exposure": exposure,
        "reason": "Published outcomes already informed development" if exposure
        == "outcome_exposed_development" else
        "A declared unseen label does not establish independent blinding",
    }
    complete = all(c["status"] == "PASS" for c in checks)
    result: dict[str, Any] = {
        "case_id": case_id, "study_id": study_id,
        "information_track": case.get("information_track"),
        "reference_exposure": reference.get("exposure"),
        "completeness": {"status": "PASS" if complete else "FAIL"},
        "numerical_fit": {"status": "NOT_RUN", "primary_endpoint": primary},
        "blinding": blinding, "input_adequacy": adequate,
        "independent_validation": {
            "status": "NOT_APPLICABLE" if exposure == "outcome_exposed_development"
            or reference.get("exposure") == "synthetic" else "INCONCLUSIVE",
            "reason": "Numerical scoring cannot establish external independence "
            "or human approval",
        },
        "human_review": "NOT_RUN", "endpoints": {}, "checks": checks,
        "retained_runs": retained, "n_submitted_runs": len(runs),
        "scoring": dict(scoring),
        "notes": [
            "Scores compare the same submitted and observed statistic; "
            "no sampling variance is added.",
            "Predictive intervals describe empirical run variation conditional "
            "on modeled uncertainty.",
            "MCSE describes precision of the prediction mean, assuming "
            "independent repetitions.",
            "One study cannot establish interval coverage or clinical "
            "generalizability.",
            "Input provenance PASS is an admissibility check, "
            "not proof of clinical adequacy.",
        ],
    }
    if not complete:
        result["status"] = "FAIL"
        return result
    try:
        for eid, (_, _, units) in endpoints.items():
            target = targets[eid]["value"]
            record = _summary([r["predictions"][eid] for r in runs], target, alpha)
            record.update({"units": units, "reference_value": float(target),
                           "baseline_crps": None, "crps_skill": None,
                           "skill_status": "NOT_APPLICABLE"})
            has_baseline = [eid in r["baselines"] for r in runs]
            if any(has_baseline) and not all(has_baseline):
                raise ValueError(f"Inconsistent optional baseline presence for {eid}")
            if all(has_baseline):
                baseline = empirical_crps([r["baselines"][eid] for r in runs], target)
                record["baseline_crps"] = baseline
                if baseline > 0:
                    record["crps_skill"] = _scalar(1 - record["crps"] / baseline)
                    record["skill_status"] = "PASS"
                else:
                    record["skill_status"] = "INCONCLUSIVE"
                    record["skill_reason"] = (
                        "Zero baseline CRPS; relative skill undefined")
            result["endpoints"][eid] = record
    except (ValueError, OverflowError) as error:
        check("score_arithmetic", False, str(error))
        result["endpoints"] = {}
        result["completeness"]["status"] = "FAIL"
        result["status"] = "FAIL"
        return result
    main = result["endpoints"][primary]
    error_pass = main["absolute_error"] <= tolerance
    skill = main["crps_skill"]
    skill_pass = None if skill is None else skill >= minimum_skill
    status = "FAIL" if not error_pass or skill_pass is False else (
        "INCONCLUSIVE" if skill_pass is None or pending_checks else "PASS")
    result["numerical_fit"].update({
        "status": status, "absolute_error_pass": error_pass,
        "minimum_skill_pass": skill_pass,
        "scope": "Preregistered primary-endpoint numerical criteria only",
    })
    result["status"] = (
        "FAIL" if adequate["status"] == "FAIL" or status == "FAIL"
        else "INCONCLUSIVE")
    return result
