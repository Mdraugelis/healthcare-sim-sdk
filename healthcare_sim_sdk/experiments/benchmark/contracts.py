"""Strict, JSON-only benchmark contracts and input provenance checks."""

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(Path(path).read_text(), object_pairs_hook=_pairs,
                      parse_constant=reject)


def _keys(obj, required, optional=()):
    if not isinstance(obj, dict):
        raise ValueError("Expected an object")
    missing = set(required) - obj.keys()
    extra = obj.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"Invalid fields: missing={sorted(missing)}, "
                         f"unknown={sorted(extra)}")


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected nonempty text")


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError("Identifier must contain only letters/digits/_/-")


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite number, not a boolean")
    if not math.isfinite(value):
        raise ValueError("Expected a finite number")
    return value


def validate_case(case):
    """Reject undeclared fields, cyclic provenance and target-derived inputs.

    Source declarations require an independent curator's review. This validates
    their consistency; it cannot discover a dishonest citation or hidden prior
    knowledge in an author or a trained model.
    """
    _keys(case, ("schema_version", "case_id", "study", "information_track",
                 "exposure", "input_cutoff", "sources", "inputs",
                 "released_targets", "endpoints", "scoring", "run_plan"))
    if type(case["schema_version"]) is not int or case["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    _identifier(case["case_id"])
    _keys(case["study"], ("id", "title", "citation", "population",
                          "assignment_unit", "outcome", "denominator",
                          "horizon", "estimand", "analysis", "limitations"))
    for value in case["study"].values():
        _text(value)
    if case["information_track"] not in ("predeployment", "conditional"):
        raise ValueError("Unknown information track")
    if case["exposure"] not in ("outcome_exposed_development", "declared_unseen"):
        raise ValueError("Unknown exposure declaration")
    cutoff = date.fromisoformat(case["input_cutoff"])
    released = case["released_targets"]
    if not isinstance(released, list) or len(set(released)) != len(released):
        raise ValueError("released_targets must be unique")
    for target in released:
        _identifier(target)
    if released and case["information_track"] != "conditional":
        raise ValueError("Predeployment track cannot release target outcomes")
    endpoints = case["endpoints"]
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("At least one endpoint is required")
    endpoint_ids = []
    for endpoint in endpoints:
        _keys(endpoint, ("id", "units", "lower_bound", "upper_bound"))
        _identifier(endpoint["id"])
        _text(endpoint["units"])
        if finite(endpoint["lower_bound"]) >= finite(endpoint["upper_bound"]):
            raise ValueError("Invalid endpoint bounds")
        endpoint_ids.append(endpoint["id"])
    if len(set(endpoint_ids)) != len(endpoint_ids):
        raise ValueError("Duplicate endpoint")
    if set(endpoint_ids) & set(released):
        raise ValueError("Released outcomes cannot also be scored")
    _keys(case["scoring"], ("primary_endpoint", "interval_alpha",
                            "absolute_error_tolerance", "minimum_skill"))
    score = case["scoring"]
    if score["primary_endpoint"] not in endpoint_ids:
        raise ValueError("Primary endpoint is undeclared")
    supported = {"risk_difference": (-1, 1), "control_risk": (0, 1),
                 "intervention_risk": (0, 1)}
    if (set(endpoint_ids) != set(supported)
            or score["primary_endpoint"] != "risk_difference"
            or any((e["lower_bound"], e["upper_bound"]) != supported[e["id"]]
                   for e in endpoints)):
        raise ValueError("Endpoints/support/primary must match shield_rt_v1")
    if not 0 < finite(score["interval_alpha"]) < 1:
        raise ValueError("interval_alpha must be between zero and one")
    if finite(score["absolute_error_tolerance"]) < 0:
        raise ValueError("Negative error tolerance")
    if finite(score["minimum_skill"]) > 1:
        raise ValueError("Skill threshold cannot exceed one")
    _keys(case["run_plan"], ("adapter", "seeds"))
    if case["run_plan"]["adapter"] != "shield_rt_v1":
        raise ValueError("Unsupported adapter")
    seeds = case["run_plan"]["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Require at least two unique uint32 run seeds")
    sources = case["sources"]
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Source ledger is required")
    for sid, source in sources.items():
        _identifier(sid)
        _keys(source, ("kind", "citation", "available_at", "derived_from"),
              ("target_ids",))
        if source["kind"] not in ("protocol", "external", "assumed",
                                  "development", "target_outcome"):
            raise ValueError("Unknown source kind")
        _text(source["citation"])
        if date.fromisoformat(source["available_at"]) > cutoff:
            raise ValueError(f"Source {sid} was unavailable at the cutoff")
        dependencies = source["derived_from"]
        if (not isinstance(dependencies, list)
                or any(dep not in sources for dep in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise ValueError("Invalid source dependencies")
        targets = source.get("target_ids", [])
        if not isinstance(targets, list) or len(set(targets)) != len(targets):
            raise ValueError("Invalid source target_ids")
        if source["kind"] == "target_outcome" and not targets:
            raise ValueError("Target-outcome sources must declare target_ids")
        if source["kind"] != "target_outcome" and targets:
            raise ValueError("Only target-outcome sources may name targets")
    visited, visiting = set(), set()

    def walk(sid):
        if sid in visiting:
            raise ValueError("Cyclic input provenance")
        if sid in visited:
            return
        visiting.add(sid)
        source = sources[sid]
        targets = set(source.get("target_ids", []))
        if targets and not targets.issubset(set(released)):
            raise ValueError(f"Prohibited withheld-outcome dependency: {sid}")
        for dep in source["derived_from"]:
            walk(dep)
        visiting.remove(sid)
        visited.add(sid)

    # The entire delivered ledger is visible to the worker: unused targets
    # must not be smuggled in as unreferenced source records either.
    for sid in sources:
        walk(sid)
    inputs = case["inputs"]
    _keys(inputs, ("n_control", "n_intervention", "baseline_risk",
                   "relative_effect", "uptake"))
    for spec in inputs.values():
        _keys(spec, ("value", "source_id"))
        if spec["source_id"] not in sources:
            raise ValueError("Input source is missing")
    canonical(case)
    # Adapter checks all numerical and joint-support constraints before freeze.
    from .shield_rt import validate_inputs
    validate_inputs(case)
    return case


def validate_reference(reference, case=None):
    _keys(reference, ("case_id", "study_id", "targets", "exposure"))
    _identifier(reference["case_id"])
    _text(reference["study_id"])
    if reference["exposure"] not in ("published", "synthetic"):
        raise ValueError("Unknown reference exposure")
    if not isinstance(reference["targets"], dict) or not reference["targets"]:
        raise ValueError("Reference targets are required")
    for target in reference["targets"].values():
        _keys(target, ("value", "source"), ("note",))
        finite(target["value"])
        _text(target["source"])
    if case is not None:
        if (case["case_id"] != reference["case_id"]
                or case["study"]["id"] != reference["study_id"]):
            raise ValueError("Study/reference identity mismatch")
        endpoint_map = {e["id"]: e for e in case["endpoints"]}
        if set(reference["targets"]) != set(endpoint_map):
            raise ValueError("Reference must match the exact scored endpoints")
        for key, target in reference["targets"].items():
            endpoint = endpoint_map[key]
            value = target["value"]
            if not endpoint["lower_bound"] <= value <= endpoint["upper_bound"]:
                raise ValueError("Reference outside endpoint support")
    return reference
