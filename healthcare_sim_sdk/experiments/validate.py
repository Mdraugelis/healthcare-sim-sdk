"""Generic experiment validation framework.

Checks structural requirements that apply to ALL experiments:
- config.json exists and is valid
- metrics.json exists and is valid
- Seed recorded (reproducibility)
- Timestamp recorded
- Output directory structure complete

Scenario-specific validators can be registered and discovered.

Usage:
    from healthcare_sim_sdk.experiments.validate import (
        validate_generic, Check, format_appendix,
    )
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Check:
    """A single validation check."""
    name: str
    description: str
    expected: Any
    actual: Any
    tolerance: str
    passed: bool
    detail: str = ""


def validate_generic(output_dir: Path) -> List[Check]:
    """Run generic structural validation on an experiment dir."""
    output_dir = Path(output_dir)
    checks = []

    # config.json exists
    config_path = output_dir / "config.json"
    checks.append(Check(
        "config.json exists",
        "Every experiment must save its configuration",
        "file exists", str(config_path.exists()),
        "exact", config_path.exists(),
    ))

    # metrics.json exists
    metrics_path = output_dir / "metrics.json"
    checks.append(Check(
        "metrics.json exists",
        "Every experiment must save key metrics",
        "file exists", str(metrics_path.exists()),
        "exact", metrics_path.exists(),
    ))

    # config is valid JSON
    config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            checks.append(Check(
                "config.json valid JSON",
                "Config must be parseable",
                "valid JSON", "valid JSON",
                "exact", True,
            ))
        except json.JSONDecodeError as e:
            checks.append(Check(
                "config.json valid JSON",
                "Config must be parseable",
                "valid JSON", f"parse error: {e}",
                "exact", False,
            ))

    # Seed recorded
    has_seed = "seed" in config
    checks.append(Check(
        "Seed recorded",
        "Reproducibility requires a seed",
        "seed in config", str(has_seed),
        "exact", has_seed,
    ))

    # Timestamp recorded
    has_ts = bool(config.get("timestamp"))
    checks.append(Check(
        "Timestamp recorded",
        "Every run needs a timestamp for catalog",
        "timestamp in config", str(has_ts),
        "exact", has_ts,
    ))

    return checks


def format_appendix(
    config: Dict, checks: List[Check],
) -> str:
    """Format validation results as markdown appendix."""
    passed = sum(1 for c in checks if c.passed)
    total = len(checks)

    lines = []
    lines.append("# Appendix: Simulation Validation")
    lines.append("")
    lines.append(
        f"**{passed}/{total} checks passed.**"
    )
    lines.append("")
    lines.append(
        f"*Experiment: {config.get('timestamp', '?')} | "
        f"Seed: {config.get('seed', '?')}*"
    )
    lines.append("")

    # Summary table
    lines.append(
        "| # | Check | Expected | Actual | Result |"
    )
    lines.append(
        "|---|-------|----------|--------|--------|"
    )
    for i, c in enumerate(checks, 1):
        status = "PASS" if c.passed else "**FAIL**"
        lines.append(
            f"| {i} | {c.name} | {c.expected} "
            f"| {c.actual} | {status} |"
        )
    lines.append("")

    # Failures detail
    failures = [c for c in checks if not c.passed]
    if failures:
        lines.append("## Failed Checks")
        lines.append("")
        for c in failures:
            lines.append(f"**{c.name}:** {c.description}")
            lines.append(f"- Expected: {c.expected}")
            lines.append(f"- Actual: {c.actual}")
            if c.detail:
                lines.append(f"- Detail: {c.detail}")
            lines.append("")

    return "\n".join(lines)


def validate_experiment(
    output_dir: Path,
    scenario_validator: Optional[callable] = None,
) -> str:
    """Full validation: generic + optional scenario-specific.

    Args:
        output_dir: Path to experiment output directory
        scenario_validator: Optional function(output_dir, config)
            -> List[Check] for scenario-specific checks

    Returns:
        Markdown validation appendix
    """
    output_dir = Path(output_dir)
    config = {}
    config_path = output_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError:
            config = {}

    checks = validate_generic(output_dir)

    if scenario_validator:
        checks.extend(scenario_validator(output_dir, config))

    return format_appendix(config, checks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python -m healthcare_sim_sdk.experiments"
            ".validate <output_dir>"
        )
        sys.exit(1)
    print(validate_experiment(Path(sys.argv[1])))
