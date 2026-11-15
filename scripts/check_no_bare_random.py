#!/usr/bin/env python3
"""Pre-commit hook: all randomness flows through RNGPartitioner.

Thin wrapper over rule SDK001 in scripts/sdk_lint.py (AST import and name
resolution). Approved locations and the full rule are documented in
docs/lint_rules.md. Usage: check_no_bare_random.py FILE ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sdk_lint import BLOCK, lint_paths  # noqa: E402


def check_file(path):
    """Violations in one file, as printable strings (kept for older callers)."""
    return ["  " + f.render() for f in lint_paths([str(path)], rules={"SDK001"})
            if f.code == "SDK001" and f.severity == BLOCK]


def main(argv):
    files = [f for f in argv if f.endswith(".py")]
    findings = [f for f in lint_paths(files, rules={"SDK001"})
                if f.code == "SDK001" and f.severity == BLOCK]
    if findings:
        print("INVARIANT VIOLATION: randomness outside RNGPartitioner\n\n"
              "Use self.rng.temporal, self.rng.population, etc.\n\n"
              + "\n".join("  " + f.render() for f in findings),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
