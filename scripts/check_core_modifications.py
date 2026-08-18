#!/usr/bin/env python3
"""Pre-commit hook: warn on core/ modifications.

Prints a warning when SDK invariant files are staged for commit.
Does NOT block — branch protection handles the merge gate.
Always exits 0.
"""

import subprocess
import sys

CORE_PREFIX = "healthcare_sim_sdk/core/"


def main():
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True, text=True,
    )
    staged = result.stdout.strip().splitlines()
    core_files = [f for f in staged if f.startswith(CORE_PREFIX)]

    if core_files:
        files_list = "\n".join(f"  - {f}" for f in core_files)
        print(
            "\n"
            "  CORE MODIFICATION DETECTED\n"
            "  ──────────────────────────\n"
            "  You are modifying SDK invariant files:\n"
            f"{files_list}\n"
            "\n"
            "  These files define the SDK architectural contract.\n"
            "  Changes here affect ALL scenarios and ALL users.\n"
            "\n"
            "  Before proceeding, consider:\n"
            "  1. Is this a scenario-level change that belongs\n"
            "     in your scenario directory instead?\n"
            "  2. Should this be proposed as a design change\n"
            "     to the main repo via RFC/discussion?\n"
            "  3. Does this change break any of the 5 invariants\n"
            "     documented in CLAUDE.md?\n"
            "\n"
            "  This commit will proceed, but core/ changes\n"
            "  require PR review from a maintainer.\n",
            file=sys.stderr,
        )

    # Always exit 0: this is a warning, not a gate
    sys.exit(0)


if __name__ == "__main__":
    main()
