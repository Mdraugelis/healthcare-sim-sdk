#!/usr/bin/env python3
"""Pre-commit hook: no bare np.random calls outside rng.py.

Enforces Invariant #3: all randomness must flow through
RNGPartitioner streams. Only core/rng.py may reference
np.random directly.
"""

import re
import sys
from pathlib import Path

ALLOWED_FILES = {
    Path("healthcare_sim_sdk/core/rng.py"),
    # Pre-registration validation script for the ITS estimator. Runs
    # pure statistical Monte Carlo trials (Type I rate calibration,
    # CI coverage, power monotonicity) on experiments/analysis/its.py.
    # No scenario, no engine, no entities — the RNGPartitioner
    # contract does not apply. See test_its_estimator.py
    # TestStatisticalValidation for the locked-in assertions.
    Path("scripts/validate_its_estimator.py"),
}

ALLOWED_DIRS = {
    Path("tests"),
}

# Patterns that indicate USAGE of np.random (not just type hints)
# np.random.Generator in type annotations is allowed
PATTERNS = [
    re.compile(r"np\.random\.(?!Generator)"),
    re.compile(r"numpy\.random\.(?!Generator)"),
    re.compile(r"from\s+numpy\s+import\s+random"),
]


def is_allowed(path: Path) -> bool:
    if path in ALLOWED_FILES:
        return True
    for d in ALLOWED_DIRS:
        try:
            path.relative_to(d)
            return True
        except ValueError:
            continue
    return False


def check_file(path: Path) -> list:
    violations = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        for pattern in PATTERNS:
            if pattern.search(line):
                violations.append(
                    f"  {path}:{i}: {stripped}"
                )
    return violations


if __name__ == "__main__":
    files = [Path(f) for f in sys.argv[1:]]
    violations = []
    for f in files:
        if not f.suffix == ".py":
            continue
        if is_allowed(f):
            continue
        violations.extend(check_file(f))
    if violations:
        print(
            "INVARIANT VIOLATION: bare np.random usage found\n"
            "\n"
            "All randomness must flow through RNGPartitioner.\n"
            "Use self.rng.temporal, self.rng.population, etc.\n"
            "\n"
            + "\n".join(violations),
            file=sys.stderr,
        )
        sys.exit(1)
    sys.exit(0)
