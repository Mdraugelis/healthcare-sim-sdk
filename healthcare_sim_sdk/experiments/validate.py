"""Run-scoped validation with explicit evidence states.

Generic checks establish artifact integrity, not clinical or analysis validity.
Callers declare decision-specific required checks and metric rules explicitly.
"""

import argparse
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass
class Check:
    """Validation evidence; the original seven positional fields remain valid.

    An explicit status overrides the legacy boolean. NOT_APPLICABLE requires
    an applicability explanation and rationale before it can satisfy a gate.
    """
    name: str
    description: str
    expected: Any
    actual: Any
    tolerance: str
    passed: bool = False
    detail: str = ""
    status: Optional[CheckStatus] = None
    required: bool = True
    category: str = "structural"
    applicability: str = ""
    rationale: str = ""

    def __post_init__(self):
        self.status = (CheckStatus(self.status) if self.status is not None
                       else CheckStatus.PASS if self.passed else CheckStatus.FAIL)
        self.passed = self.status == CheckStatus.PASS


@dataclass(frozen=True)
class MetricRule:
    """Explicit rule for a dot-separated metrics key (no name-based guessing)."""
    path: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = True
    numeric: bool = True
    rationale: str = ""

    def __post_init__(self):
        if not self.path or any(not part for part in self.path.split(".")):
            raise ValueError("MetricRule.path must be a nonempty dot-separated key")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (isinstance(bound, bool)
                                      or not math.isfinite(bound)):
                raise ValueError("Metric rule bounds must be finite numbers")
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise ValueError("Metric rule minimum exceeds maximum")
        if not self.numeric and (self.minimum is not None or self.maximum is not None):
            raise ValueError("Metric bounds require numeric=True")


def _reject_constant(value: str):
    raise ValueError(f"Nonfinite JSON number: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return number


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: Path) -> Dict[str, Any]:
    """Reject nonstandard constants, overflowing numbers and non-object roots."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=_reject_constant,
                          parse_float=_finite_float, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object at the document root")
    return value


def ensure_required_checks(
    checks: Sequence[Check], names: Iterable[str],
) -> List[Check]:
    """Mark declared checks required and retain absent evidence as NOT_RUN."""
    result = list(checks)
    existing = {check.name: check for check in result}
    for name in names:
        if name in existing:
            for check in result:
                if check.name == name:
                    check.required = True
        else:
            check = Check(name, "Declared required evidence was not supplied",
                          "executed applicable check", "missing", "declared criterion",
                          status=CheckStatus.NOT_RUN, category="completeness")
            result.append(check)
            existing[name] = check
    return result


def checks_passed(checks: Sequence[Check]) -> bool:
    """A fail or unresolved required check blocks the gate; empty is not green."""
    if not checks or not any(check.required for check in checks):
        return False
    for check in checks:
        if check.status == CheckStatus.FAIL:
            return False
        if check.required:
            justified_na = (check.status == CheckStatus.NOT_APPLICABLE
                            and bool(check.applicability.strip())
                            and bool(check.rationale.strip()))
            if check.status != CheckStatus.PASS and not justified_na:
                return False
    return True


def _metric_checks(metrics: Mapping, rules: Sequence[MetricRule]) -> List[Check]:
    checks = []
    for rule in rules:
        value: Any = metrics
        present = True
        for part in rule.path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                present = False
                break
            value = value[part]
        expected = (f"numeric value in [{rule.minimum}, {rule.maximum}]"
                    if rule.numeric else "metric present")
        status = CheckStatus.PASS
        detail = ""
        if not present:
            value = "missing"
            status = CheckStatus.NOT_RUN
        elif rule.numeric:
            numeric = (isinstance(value, (int, float)) and not isinstance(value, bool)
                       and (not isinstance(value, float) or math.isfinite(value)))
            if not numeric:
                status = CheckStatus.FAIL
                detail = (
                    "A finite numeric metric is required; booleans are not numbers.")
            elif ((rule.minimum is not None and value < rule.minimum)
                  or (rule.maximum is not None and value > rule.maximum)):
                status = CheckStatus.FAIL
                detail = "Metric is outside the explicitly declared bounds."
        checks.append(Check(f"Metric: {rule.path}", "Explicit metric rule", expected,
                            value, "inclusive declared bounds", detail=detail,
                            status=status, required=rule.required, category="metric",
                            rationale=rule.rationale))
    return checks


def validate_generic(
    output_dir: Path,
    metric_rules: Optional[Sequence[MetricRule]] = None,
    required_checks: Iterable[str] = (),
) -> List[Check]:
    """Validate saved objects and declared rules without inferring clinical bounds."""
    output_dir = Path(output_dir)
    checks = []
    artifacts = {}
    for filename in ("config.json", "metrics.json"):
        path = output_dir / filename
        exists = path.is_file()
        checks.append(Check(f"{filename} exists", "Required experiment artifact",
                            "file exists", str(exists), "exact", exists))
        if not exists:
            checks.append(Check(f"{filename} valid JSON",
                                "Artifact must be a strict JSON object",
                                "finite, unambiguous JSON object", "file unavailable",
                                "exact", status=CheckStatus.NOT_RUN))
            continue
        try:
            artifacts[filename] = load_json_object(path)
            checks.append(Check(f"{filename} valid JSON",
                                "Artifact must be a strict JSON object",
                                "finite, unambiguous JSON object", "valid JSON object",
                                "exact", True))
        except (OSError, ValueError, UnicodeError) as exc:
            checks.append(Check(f"{filename} valid JSON",
                                "Artifact must be a strict JSON object",
                                "finite, unambiguous JSON object", str(exc),
                                "exact", False))
    config = artifacts.get("config.json", {})
    checks.append(Check(
        "Metrics recorded", "An experiment must save at least one metric",
        "nonempty metrics object", str(bool(artifacts.get("metrics.json"))),
        "exact", bool(artifacts.get("metrics.json")),
        status=None if "metrics.json" in artifacts else CheckStatus.NOT_RUN))
    valid_seed = (isinstance(config.get("seed"), int)
                  and not isinstance(config.get("seed"), bool) and config["seed"] >= 0)
    valid_timestamp = (isinstance(config.get("timestamp"), str)
                       and bool(config["timestamp"].strip()))
    for name, description, valid in (
        ("Seed recorded", "Reproducibility requires a nonnegative integer seed",
         valid_seed),
        ("Timestamp recorded", "A run requires a nonempty timestamp string",
         valid_timestamp),
    ):
        checks.append(Check(
            name, description, description, str(valid), "exact", valid,
            status=None if "config.json" in artifacts else CheckStatus.NOT_RUN))
    checks.extend(_metric_checks(artifacts.get("metrics.json", {}), metric_rules or ()))
    return ensure_required_checks(checks, required_checks)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_appendix(config: Dict, checks: List[Check]) -> str:
    """Render missing evidence explicitly; gate success applies to declared scope."""
    counts = {status: sum(check.status == status for check in checks)
              for status in CheckStatus}
    lines = ["# Appendix: Simulation Validation", "",
             "**Declared validation gate: "
             + ("PASS" if checks_passed(checks) else "BLOCKED") + ".**",
             "", "; ".join(f"{status.value}: {counts[status]}"
                           for status in CheckStatus) + ".", "",
             "This gate covers only the checks listed below. "
             "Artifact integrity does not establish "
             "clinical assumptions, model fidelity, or analysis validity.", "",
             f"*Experiment: {_cell(config.get('timestamp', '?'))} | "
             f"Seed: {_cell(config.get('seed', '?'))}*", "",
             "| # | Check | Expected | Actual | Status | Required | Category |",
             "|---|-------|----------|--------|--------|----------|----------|"]
    for index, check in enumerate(checks, 1):
        lines.append(f"| {index} | {_cell(check.name)} | {_cell(check.expected)} | "
                     f"{_cell(check.actual)} | {CheckStatus(check.status).value} | "
                     f"{check.required} | {_cell(check.category)} |")
    lines.extend(["", "## Evidence details", ""])
    for check in checks:
        lines.extend([f"**{check.name} ({CheckStatus(check.status).value}):** "
                      f"{check.description}", "",
                      f"- Acceptance/tolerance: {check.tolerance}"])
        if check.applicability:
            lines.append(f"- Applicability: {check.applicability}")
        if check.rationale:
            lines.append(f"- Rationale: {check.rationale}")
        if check.detail:
            lines.append(f"- Detail: {check.detail}")
        lines.append("")
    return "\n".join(lines)


def collect_validation(
    output_dir: Path,
    scenario_validator: Optional[Callable[[Path, Dict], List[Check]]] = None,
    *, metric_rules: Optional[Sequence[MetricRule]] = None,
    required_checks: Iterable[str] = (),
) -> List[Check]:
    """Collect generic and optional scenario checks, then enforce completeness."""
    output_dir = Path(output_dir)
    checks = validate_generic(output_dir, metric_rules=metric_rules)
    if scenario_validator:
        try:
            config = load_json_object(output_dir / "config.json")
        except (OSError, ValueError, UnicodeError) as exc:
            checks.append(Check(
                "Scenario validator", "Scenario checks require valid configuration",
                "executed", "invalid configuration", "exact", detail=str(exc),
                status=CheckStatus.NOT_RUN, category="scenario"))
        else:
            try:
                scenario_checks = list(scenario_validator(output_dir, config))
                if not all(isinstance(check, Check) for check in scenario_checks):
                    raise TypeError("Scenario validator must return Check records")
                checks.extend(scenario_checks)
                if not scenario_checks:
                    checks.append(Check(
                        "Scenario validator", "Scenario validator returned no evidence",
                        "one or more checks", "empty", "exact",
                        status=CheckStatus.NOT_RUN, category="scenario"))
            except Exception as exc:
                checks.append(Check(
                    "Scenario validator", "Scenario validator must complete",
                    "completed checks", type(exc).__name__, "exact", detail=str(exc),
                    status=CheckStatus.FAIL, category="scenario"))
    return ensure_required_checks(checks, required_checks)


def validate_experiment(
    output_dir: Path,
    scenario_validator: Optional[Callable[[Path, Dict], List[Check]]] = None,
    *, metric_rules: Optional[Sequence[MetricRule]] = None,
    required_checks: Iterable[str] = (),
) -> str:
    """Return the backward-compatible Markdown appendix with honest gate state."""
    checks = collect_validation(
        output_dir, scenario_validator, metric_rules=metric_rules,
        required_checks=required_checks)
    try:
        config = load_json_object(Path(output_dir) / "config.json")
    except (OSError, ValueError, UnicodeError):
        config = {}
    return format_appendix(config, checks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--require-check", action="append", default=[],
                        help="Required evidence check name")
    parser.add_argument("--metric-rules", type=Path,
                        help='JSON object with a "rules" array of MetricRule fields')
    args = parser.parse_args(argv)
    try:
        rules = None
        if args.metric_rules:
            specification = load_json_object(args.metric_rules)
            if not isinstance(specification.get("rules"), list):
                raise ValueError('Metric rules document requires a "rules" array')
            rules = [MetricRule(**rule) for rule in specification["rules"]]
        checks = collect_validation(args.output_dir, metric_rules=rules,
                                    required_checks=args.require_check)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    try:
        config = load_json_object(args.output_dir / "config.json")
    except (OSError, ValueError, UnicodeError):
        config = {}
    print(format_appendix(config, checks))
    return 0 if checks_passed(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
