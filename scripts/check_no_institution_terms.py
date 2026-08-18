#!/usr/bin/env python3
"""Pre-commit hook: block institution-specific terms from being committed.

Enforces the data boundary (see docs/data_boundary.md). Institution
confidential content — organization names, campus codes, benchmark
figures under confidentiality, staff names, internal work-item numbers —
must never land in tracked files or commit messages. A .gitignore rule
only stops an accidental `git add`; this hook is the convention-independent
backstop that catches the term even when an ignore rule is missing.

Three surfaces are checked, because a term leaks through any of them:

1. File contents — the obvious case.
2. File and directory NAMES — a sanitized file can still sit at a path
   like ``configs/<org>_projection.yaml``. Content rewrites (including
   ``git filter-repo --replace-text``) do not touch path names, so a
   name-only leak survives history rewrites and stays visible in every
   tree listing and diff header.
3. Commit messages — via the ``commit-msg`` stage, where pre-commit
   passes the message file as ``.git/COMMIT_EDITMSG``.

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

# Never scan at all — the denylist machinery itself and vendored code.
# NOTE: '.git/' is deliberately NOT listed here. pre-commit's commit-msg
# stage passes the message file as '.git/COMMIT_EDITMSG' (relative to the
# repo root), so excluding '.git/' silently disables commit-message
# checking entirely. Git-internal paths are handled by is_git_internal()
# below, which skips their *names* but still scans their *contents*.
SKIP_ENTIRELY_PREFIXES = (
    ".institution-denylist",
    ".venv/",
)

# Contents are unreadable as text; the NAME is still checked.
BINARY_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".pdf",
    ".pyc", ".so", ".woff", ".woff2", ".ttf", ".ico",
)

# Path components that are git-internal rather than tracked content.
GIT_INTERNAL_PREFIXES = (".git/",)

# Path separators and word joiners. Splitting on these lets word-boundary
# rules such as 're:\bSITE_A\b' match inside 'site_a_projection.yaml',
# where '_' is a word character and \b would otherwise never fire.
NAME_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")


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


def match_rule(text, literals, regexes):
    """Return the first rule matching `text`, or None."""
    low = text.lower()
    for lit in literals:
        if lit in low:
            return lit
    for rx in regexes:
        if rx.search(text):
            return rx.pattern
    return None


def skip_entirely(path: Path) -> bool:
    return path.as_posix().startswith(SKIP_ENTIRELY_PREFIXES)


def is_git_internal(path: Path) -> bool:
    """True for git-internal paths (e.g. the commit-msg buffer).

    Their contents are real content to check; their names are not
    author-chosen tracked paths, so name checks are skipped.
    """
    return path.as_posix().startswith(GIT_INTERNAL_PREFIXES)


def is_binary(path: Path) -> bool:
    return path.as_posix().endswith(BINARY_SUFFIXES)


def scan_name(path: Path, literals, regexes) -> list:
    """Check the path itself — every directory component and the filename.

    Both the raw path and a separator-normalized form are tested so that
    word-boundary rules fire on '_'- and '-'-joined name components.
    """
    posix = path.as_posix()
    normalized = NAME_SEPARATORS.sub(" ", posix)
    for candidate in (posix, normalized):
        hit = match_rule(candidate, literals, regexes)
        if hit is not None:
            return [f"  {posix}: PATH NAME matched denylist rule [{hit}]"]
    return []


def scan_file(path: Path, literals, regexes) -> list:
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return violations  # binary or gone — nothing to scan
    for i, line in enumerate(text.splitlines(), 1):
        hit = match_rule(line, literals, regexes)
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
        if skip_entirely(path):
            continue
        if not is_git_internal(path):
            violations.extend(scan_name(path, literals, regexes))
        if not is_binary(path):
            violations.extend(scan_file(path, literals, regexes))

    if violations:
        print(
            "DATA BOUNDARY VIOLATION: institution-specific term found in "
            "staged content.\n"
            "\n"
            "This content must not be committed to tracked files, path\n"
            "names, or commit messages. Move institution values into a\n"
            "gitignored location (a local/ directory or a *.private.*\n"
            "file) and commit only a sanitized *.example.* template.\n"
            "A flagged PATH NAME must be renamed — sanitizing the file\n"
            "contents alone leaves the term in the repository tree.\n"
            "See docs/data_boundary.md.\n"
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
