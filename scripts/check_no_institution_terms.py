#!/usr/bin/env python3
"""Pre-commit hook: block institution-specific terms from being committed.

Enforces the data boundary (see docs/data_boundary.md). Institution
confidential content — organization names, campus codes, benchmark
figures under confidentiality, staff names, internal work-item numbers —
must never land in tracked files or commit messages. A .gitignore rule
only stops an accidental `git add`; this hook is the convention-independent
backstop that catches the term even when an ignore rule is missing.

The list of terms is deliberately NOT hardcoded here: writing the real
terms into this tracked script would itself leak them. Instead the terms
live in a local, gitignored denylist file. Copy the tracked example to
create yours:

    cp .institution-denylist.example .institution-denylist
    # then edit .institution-denylist with your real terms

Denylist format (one rule per line):
  - Blank lines and lines starting with '#' are ignored.
  - A line beginning with 're:' is treated as a case-insensitive regex.
  - Any other line is matched as a case-insensitive literal substring.

If .institution-denylist is absent, the hook prints a one-time notice and
passes — so external contributors with no confidential data are never
blocked. The person handling institution values keeps the file locally.
"""

import re
import sys
from pathlib import Path

DENYLIST_FILE = Path(".institution-denylist")
EXAMPLE_FILE = Path(".institution-denylist.example")

# Never scan these (the denylist machinery itself, binaries, vendored code).
EXCLUDE_PREFIXES = (
    ".institution-denylist",
    ".venv/",
    ".git/",
)
EXCLUDE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".ipynb_checkpoints",
    ".pyc", ".so", ".woff", ".woff2", ".ttf", ".ico",
)


def load_rules():
    """Return (literals, regexes) parsed from the local denylist.

    Returns (None, None) when the denylist file does not exist so the
    caller can distinguish "no rules configured" from "no rules matched".
    """
    if not DENYLIST_FILE.exists():
        return None, None
    literals = []
    regexes = []
    for raw in DENYLIST_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("re:"):
            pattern = line[3:].strip()
            try:
                regexes.append(re.compile(pattern, re.IGNORECASE))
            except re.error as exc:
                print(
                    f"WARNING: bad regex in {DENYLIST_FILE}: {pattern!r} "
                    f"({exc}) — skipped",
                    file=sys.stderr,
                )
        else:
            literals.append(line.lower())
    return literals, regexes


def is_excluded(path: Path) -> bool:
    p = path.as_posix()
    if p.startswith(EXCLUDE_PREFIXES):
        return True
    if p.endswith(EXCLUDE_SUFFIXES):
        return True
    return False


def scan_file(path: Path, literals, regexes) -> list:
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return violations  # binary or gone — nothing to scan
    for i, line in enumerate(text.splitlines(), 1):
        low = line.lower()
        hit = None
        for lit in literals:
            if lit in low:
                hit = lit
                break
        if hit is None:
            for rx in regexes:
                if rx.search(line):
                    hit = rx.pattern
                    break
        if hit is not None:
            violations.append(f"  {path}:{i}: matched denylist rule [{hit}]")
    return violations


def main(argv) -> int:
    literals, regexes = load_rules()

    if literals is None:
        # No local denylist configured. Do not block — just inform once.
        if EXAMPLE_FILE.exists():
            print(
                f"NOTE: {DENYLIST_FILE} not found — institution-term guard "
                f"is inactive.\n"
                f"      If you handle confidential values, run: "
                f"cp {EXAMPLE_FILE} {DENYLIST_FILE}",
                file=sys.stderr,
            )
        return 0

    if not literals and not regexes:
        return 0  # file present but empty — nothing to enforce

    violations = []
    for arg in argv:
        path = Path(arg)
        if is_excluded(path):
            continue
        violations.extend(scan_file(path, literals, regexes))

    if violations:
        print(
            "DATA BOUNDARY VIOLATION: institution-specific term found in "
            "staged content.\n"
            "\n"
            "This content must not be committed to tracked files or commit\n"
            "messages. Move institution values into a gitignored location\n"
            "(a local/ directory or a *.private.* file) and commit only a\n"
            "sanitized *.example.* template. See docs/data_boundary.md.\n"
            "\n"
            "If this is a false positive, refine the rule in "
            f"{DENYLIST_FILE}.\n"
            "\n"
            + "\n".join(violations),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
