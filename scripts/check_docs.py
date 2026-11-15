#!/usr/bin/env python3
"""Documentation checks for agent guidance (issue #9).

Standard library only, deterministic and offline. Each finding prints as
``path:line: CODE message``, sorted. Exit status is 1 if any finding is
reported. Rules are described in docs/lint_rules.md.

    python scripts/check_docs.py                 # enabled rules, this checkout
    python scripts/check_docs.py --codes DOC001  # chosen rules only
    python scripts/check_docs.py --root DIR      # another tree (tests)
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# Rules that run by default. Later changes switch more on as their files land.
ENABLED: Tuple[str, ...] = (
    "DOC001", "DOC002", "DOC003", "DOC006", "DOC007", "DOC008", "DOC009",
)
ALL_CODES: Tuple[str, ...] = (
    "DOC001", "DOC002", "DOC003", "DOC004", "DOC005",
    "DOC006", "DOC007", "DOC008", "DOC009",
)

AUTHORITY_MAP = "docs/AUTHORITY.md"
CLASSES = ("current", "rationale", "proposal", "record")

ROOT_GUIDANCE = "AGENTS.md"
ROOT_BUDGET = 80

# link path -> (target as written in the link, import line accepted instead,
#               required)
SYMLINKS: Dict[str, Tuple[str, Optional[str], bool]] = {
    "CLAUDE.md": ("AGENTS.md", "@AGENTS.md", True),
    ".claude/invariants.yaml": ("../config/invariants.yaml", None, False),
}

SCENARIO_ROOT = "healthcare_sim_sdk/scenarios"
SCENARIO_STATUSES = (
    "reference", "operational", "calibration exercise",
    "institution-specific", "template",
)
SCENARIO_HEADINGS = (
    "## What it models", "## Entry points", "## Verify",
    "## Private files", "## Known limits",
)

# retired path -> (replacement, files that may still name it)
RETIRED: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    ".claude/invariants.yaml": (
        "config/invariants.yaml",
        (AUTHORITY_MAP, ".gitignore", "docs/lint_rules.md"),
    ),
    ".claude/agents/sim-guide.md": (
        "docs/simulation_protocol.md",
        (AUTHORITY_MAP, "README.md", "docs/lint_rules.md",
         ".claude/agents/sim-guide.md"),
    ),
}
SELF_FILES = ("scripts/check_docs.py", "tests/unit/test_check_docs.py")

EXCLUDED_PREFIXES = ("tests/lint_fixtures/",)
WALK_SKIP_DIRS = {
    ".git", "outputs", "node_modules", "__pycache__", "build", "dist",
    ".venv", "venv", ".pytest_cache", ".ruff_cache", ".mypy_cache",
}
TEXT_SUFFIXES = {
    ".md", ".py", ".yaml", ".yml", ".toml", ".sh", ".cfg", ".txt", ".json",
}
TEXT_NAMES = {".gitignore", ".flake8", ".pre-commit-config.yaml"}

ROOT_DIRS = (
    "healthcare_sim_sdk", "scripts", "tests", "docs", "config", "examples",
    "reports", "research", "requirements", ".claude", ".github",
)
ROOT_FILES = {
    "AGENTS.md", "CLAUDE.md", "README.md", "CONTRIBUTING.md",
    "pyproject.toml", ".pre-commit-config.yaml", ".flake8",
    ".python-version", ".gitignore",
}
PATH_WORD = re.compile(
    r"^(?:" + "|".join(re.escape(d) for d in ROOT_DIRS) + r")/[\w./\-]*$"
)
PLACEHOLDER = re.compile(r"[<>*{}$]|YYYY|\.\.\.")
FILE_LIKE = re.compile(r"\.(?:py|ya?ml|md|toml|json|sh|ipynb|txt)$")

LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
CODE_SPAN = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^\s*(```|~~~)")
SUPPRESS = re.compile(r"<!--\s*docs-check:\s*allow\s+(DOC\d{3})\s*(?:--\s*(.*?))?\s*-->")
MAP_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([a-z]+)\s*\|")


class Finding(NamedTuple):
    path: str
    line: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


class Repo:
    """The files under check, plus the authority map read from them."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.files = self._list_files()
        self.fileset = set(self.files)
        self._text: Dict[str, List[str]] = {}
        self.map_rows, self.map_findings = self._read_map()

    # -- file discovery ---------------------------------------------------
    def _list_files(self) -> List[str]:
        names = self._git_files()
        if names is None:
            names = self._walk_files()
        return sorted(
            n for n in names if not n.startswith(EXCLUDED_PREFIXES)
        )

    def _git_files(self) -> Optional[List[str]]:
        """Tracked files plus new files not yet added (ignored files are
        left out), or None when root is not the top of a checkout."""
        try:
            top = subprocess.run(
                ["git", "-C", str(self.root), "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            if Path(top).resolve() != self.root:
                return None
            out = subprocess.run(
                ["git", "-C", str(self.root), "ls-files", "-z", "--cached",
                 "--others", "--exclude-standard"],
                capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        # A tracked file deleted in the working tree is not there to check.
        return [n for n in out.split("\0") if n and self._lexists(n)]

    def _walk_files(self) -> List[str]:
        names = []
        for base, dirs, files in os.walk(self.root):
            dirs[:] = sorted(
                d for d in dirs
                if d not in WALK_SKIP_DIRS and not d.endswith(".egg-info")
                and not os.path.islink(os.path.join(base, d))
            )
            for name in files:
                full = Path(base) / name
                names.append(full.relative_to(self.root).as_posix())
        return names

    def _lexists(self, rel: str) -> bool:
        return os.path.lexists(self.root / rel)

    # -- reading ----------------------------------------------------------
    def lines(self, rel: str) -> List[str]:
        if rel not in self._text:
            try:
                text = (self.root / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                text = ""
            self._text[rel] = text.splitlines()
        return self._text[rel]

    def exists(self, rel: str) -> bool:
        """True for a file or directory inside the tree."""
        rel = os.path.normpath(rel).replace(os.sep, "/")
        if rel.startswith("../") or rel == "..":
            return False
        return (self.root / rel).exists()

    def markdown(self) -> List[str]:
        return [f for f in self.files if f.endswith(".md")]

    # -- authority map ----------------------------------------------------
    def _read_map(self):
        rows: List[Tuple[str, str, int]] = []
        findings: List[Finding] = []
        if AUTHORITY_MAP not in self.fileset:
            return rows, findings
        seen: Dict[str, int] = {}
        for number, line in enumerate(self.lines(AUTHORITY_MAP), 1):
            match = MAP_ROW.match(line)
            if not match:
                continue
            pattern, cls = match.group(1), match.group(2)
            if cls not in CLASSES:
                findings.append(Finding(
                    AUTHORITY_MAP, number, "DOC006",
                    f"unknown class '{cls}' for `{pattern}`; "
                    f"use one of: {', '.join(CLASSES)}"))
                continue
            if pattern in seen:
                findings.append(Finding(
                    AUTHORITY_MAP, number, "DOC006",
                    f"`{pattern}` is listed twice (first on line "
                    f"{seen[pattern]})"))
                continue
            seen[pattern] = number
            rows.append((pattern, cls, number))
        return rows, findings

    def class_of(self, rel: str) -> Optional[str]:
        """Class of a file: an exact row, else a glob row, else the
        longest folder row. None when no row covers it."""
        best: Tuple[int, int, Optional[str]] = (-1, -1, None)
        for pattern, cls, _ in self.map_rows:
            if pattern == rel:
                rank = (3, len(pattern))
            elif "*" in pattern and fnmatch.fnmatchcase(rel, pattern):
                rank = (2, len(pattern))
            elif pattern.endswith("/") and rel.startswith(pattern):
                rank = (1, len(pattern))
            else:
                continue
            if rank > best[:2]:
                best = (rank[0], rank[1], cls)
        return best[2]


# -- markdown scanning ----------------------------------------------------
def prose_lines(lines: Sequence[str]) -> Iterable[Tuple[int, str]]:
    """Numbered lines outside fenced code blocks."""
    fence = None
    for number, line in enumerate(lines, 1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            yield number, line


def suppressed(lines: Sequence[str], number: int, code: str) -> bool:
    """A reasoned ``docs-check: allow`` on the line or the line above."""
    for index in (number - 1, number - 2):
        if 0 <= index < len(lines):
            for found, reason in SUPPRESS.findall(lines[index]):
                if found == code and reason.strip():
                    return True
    return False


def path_words(span: str) -> List[str]:
    """Repo paths named inside one code span."""
    words = []
    for word in span.split():
        word = word.strip("\"'(),;")
        if PLACEHOLDER.search(word):
            continue
        word = re.sub(r"(::[\w:\[\]\-]+|:\d+(?:-\d+)?|#L\d+)$", "", word)
        if PATH_WORD.match(word) or word in ROOT_FILES:
            words.append(word)
    return words


# -- rules ----------------------------------------------------------------
def doc001_links(repo: Repo) -> List[Finding]:
    """Relative markdown links resolve."""
    findings = []
    for rel in repo.markdown():
        lines = repo.lines(rel)
        base = os.path.dirname(rel)
        for number, line in prose_lines(lines):
            visible = CODE_SPAN.sub("", line)
            for target in LINK.findall(visible):
                if re.match(r"^(?:[a-z][a-z0-9+.\-]*:|#|//)", target, re.I):
                    continue
                path = target.split("#", 1)[0].split("?", 1)[0]
                if not path or suppressed(lines, number, "DOC001"):
                    continue
                joined = path.lstrip("/") if path.startswith("/") else (
                    os.path.join(base, path))
                if not repo.exists(joined):
                    findings.append(Finding(
                        rel, number, "DOC001",
                        f"link target does not exist: {target}"))
    return findings


def doc002_paths(repo: Repo) -> List[Finding]:
    """Current-rule documents name only paths that exist."""
    findings = []
    for rel in repo.markdown():
        if repo.class_of(rel) != "current":
            continue
        lines = repo.lines(rel)
        for number, line in prose_lines(lines):
            for span in CODE_SPAN.findall(line):
                for word in path_words(span):
                    if repo.exists(word):
                        continue
                    if suppressed(lines, number, "DOC002"):
                        continue
                    findings.append(Finding(
                        rel, number, "DOC002",
                        f"path does not exist: {word}"))
    return findings


def doc003_symlinks(repo: Repo) -> List[Finding]:
    """Adapter symlinks point where they should; none dangle."""
    findings = []
    for link, (target, fallback, required) in sorted(SYMLINKS.items()):
        full = repo.root / link
        if not os.path.lexists(full):
            if required:
                findings.append(Finding(
                    link, 1, "DOC003",
                    f"missing; create it with: ln -s {target} {link}"))
            continue
        if os.path.islink(full):
            actual = os.readlink(full)
            if actual != target:
                findings.append(Finding(
                    link, 1, "DOC003",
                    f"points at '{actual}', expected '{target}'"))
            continue
        content = "\n".join(repo.lines(link)).strip()
        if fallback and content == fallback:
            continue  # the documented import form, for checkouts without symlinks
        if content == target:
            findings.append(Finding(
                link, 1, "DOC003",
                "is a plain file holding the link target: this checkout has "
                "symlinks switched off (git config core.symlinks). Switch "
                "them on and check out again"
                + (f", or replace the file with the line '{fallback}'"
                   if fallback else "")))
            continue
        findings.append(Finding(
            link, 1, "DOC003",
            f"is a regular file; it must be a symlink to '{target}'"
            + (f" or contain only '{fallback}'" if fallback else "")
            + " so the two cannot drift"))
    for rel in repo.files:
        full = repo.root / rel
        if os.path.islink(full) and not full.exists():
            if rel in SYMLINKS:
                continue  # the wrong-target finding above already covers it
            findings.append(Finding(
                rel, 1, "DOC003",
                f"dangling symlink to '{os.readlink(full)}'"))
    return findings


def scenario_dirs(repo: Repo) -> List[str]:
    prefix = SCENARIO_ROOT + "/"
    return sorted({
        os.path.dirname(f) for f in repo.files
        if f.startswith(prefix) and f.count("/") == prefix.count("/") + 1
        and os.path.basename(f) == "scenario.py"
    })


def doc004_scenario_guidance(repo: Repo) -> List[Finding]:
    """Every scenario folder has local guidance in the standard shape."""
    findings = []
    for folder in scenario_dirs(repo):
        rel = f"{folder}/AGENTS.md"
        if rel not in repo.fileset:
            findings.append(Finding(
                f"{folder}/scenario.py", 1, "DOC004",
                f"scenario has no AGENTS.md; copy {SCENARIO_ROOT}"
                "/_template/AGENTS.md and fill it in"))
            continue
        lines = repo.lines(rel)
        status = [
            (n, line) for n, line in enumerate(lines, 1)
            if line.startswith("**Status:**")
        ]
        if not status:
            findings.append(Finding(
                rel, 1, "DOC004", "missing the '**Status:**' line"))
        else:
            number, line = status[0]
            value = line.split("**Status:**", 1)[1].strip().rstrip(".")
            value = value.split(".")[0].split(";")[0].strip().lower()
            if value not in SCENARIO_STATUSES:
                findings.append(Finding(
                    rel, number, "DOC004",
                    f"status '{value}' is not one of: "
                    + ", ".join(SCENARIO_STATUSES)))
        present = {line.strip() for line in lines}
        for heading in SCENARIO_HEADINGS:
            if heading not in present:
                findings.append(Finding(
                    rel, 1, "DOC004", f"missing heading '{heading}'"))
    return findings


def doc005_scenario_commands(repo: Repo) -> List[Finding]:
    """Paths, runner modules and test paths in scenario guidance resolve."""
    findings = []
    for folder in scenario_dirs(repo):
        rel = f"{folder}/AGENTS.md"
        if rel not in repo.fileset:
            continue
        lines = repo.lines(rel)
        for number, line in prose_lines(lines):
            for span in CODE_SPAN.findall(line):
                for problem in _span_problems(repo, folder, span):
                    if not suppressed(lines, number, "DOC005"):
                        findings.append(Finding(rel, number, "DOC005", problem))
    return findings


def _span_problems(repo: Repo, folder: str, span: str) -> List[str]:
    if PLACEHOLDER.search(span):
        return []
    words = span.split()
    problems = []
    if "-m" in words and words[0].startswith("python"):
        module = words[words.index("-m") + 1] if (
            words.index("-m") + 1 < len(words)) else ""
        if module.startswith("healthcare_sim_sdk"):
            base = module.replace(".", "/")
            if not (repo.exists(base + ".py")
                    or repo.exists(base + "/__main__.py")):
                problems.append(f"runner module does not exist: {module}")
        words = words[words.index("-m") + 2:]
    for word in words:
        word = word.strip("\"'(),;")
        word = re.sub(r"(::[\w:\[\]\-]+|:\d+(?:-\d+)?)$", "", word)
        if word.startswith("-") or "=" in word:
            continue
        if not ("/" in word or FILE_LIKE.search(word)):
            continue
        if repo.exists(word) or repo.exists(f"{folder}/{word}"):
            continue
        problems.append(f"path does not exist: {word}")
    return problems


def doc006_authority(repo: Repo) -> List[Finding]:
    """Every markdown file has exactly one class in the authority map."""
    if AUTHORITY_MAP not in repo.fileset:
        return [Finding(AUTHORITY_MAP, 1, "DOC006",
                        "the authority map is missing")]
    findings = list(repo.map_findings)
    for pattern, _, number in repo.map_rows:
        if "*" in pattern:
            hit = any(fnmatch.fnmatchcase(f, pattern) for f in repo.files)
        elif pattern.endswith("/"):
            hit = any(f.startswith(pattern) for f in repo.files)
        else:
            hit = pattern in repo.fileset
        if not hit:
            findings.append(Finding(
                AUTHORITY_MAP, number, "DOC006",
                f"`{pattern}` matches no file; remove or correct the row"))
    for rel in repo.markdown():
        if repo.class_of(rel) is None:
            findings.append(Finding(
                rel, 1, "DOC006",
                f"not classified in {AUTHORITY_MAP}; add a row saying "
                "whether it is current, rationale, proposal or record"))
    return findings


def doc007_root_budget(repo: Repo) -> List[Finding]:
    """Root guidance exists and stays short."""
    if ROOT_GUIDANCE not in repo.fileset:
        return [Finding(ROOT_GUIDANCE, 1, "DOC007",
                        "root guidance file is missing")]
    count = len(repo.lines(ROOT_GUIDANCE))
    if count > ROOT_BUDGET:
        return [Finding(
            ROOT_GUIDANCE, ROOT_BUDGET + 1, "DOC007",
            f"{count} lines; the limit is {ROOT_BUDGET}. It loads in every "
            "session: move detail to the folder it applies to or to docs/")]
    return []


def frontmatter(lines: Sequence[str]) -> Optional[Dict[str, str]]:
    """Top-level keys of a leading ``---`` block; None if there is none."""
    if not lines or lines[0].strip() != "---":
        return None
    keys: Dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return keys
        match = re.match(r"^([A-Za-z][\w\-]*):\s*(.*)$", line)
        if match:
            keys[match.group(1)] = match.group(2).strip()
    return None  # never closed


def doc008_adapters(repo: Repo) -> List[Finding]:
    """Tool adapter files carry the frontmatter their tool requires."""
    findings = []
    for rel in repo.files:
        if not rel.endswith(".md"):
            continue
        if rel.startswith(".claude/agents/"):
            keys = frontmatter(repo.lines(rel))
            if keys is None:
                findings.append(Finding(
                    rel, 1, "DOC008",
                    "no frontmatter; without 'name' and 'description' the "
                    "agent is not registered"))
                continue
            for key in ("name", "description"):
                if not keys.get(key):
                    findings.append(Finding(
                        rel, 1, "DOC008",
                        f"frontmatter has no '{key}'"))
        elif rel.startswith(".claude/rules/"):
            keys = frontmatter(repo.lines(rel))
            if keys is None or "paths" not in keys:
                findings.append(Finding(
                    rel, 1, "DOC008",
                    "no 'paths' frontmatter; an unscoped rule loads in "
                    "every session and belongs in AGENTS.md instead"))
    return findings


def doc009_retired(repo: Repo) -> List[Finding]:
    """Retired paths are named only in records and compatibility notes."""
    findings = []
    for rel in repo.files:
        suffix = os.path.splitext(rel)[1]
        if suffix not in TEXT_SUFFIXES and os.path.basename(rel) not in TEXT_NAMES:
            continue
        if rel in SELF_FILES or os.path.islink(repo.root / rel):
            continue
        if repo.class_of(rel) == "record":
            continue
        lines = repo.lines(rel)
        for old, (new, allowed) in sorted(RETIRED.items()):
            if rel in allowed:
                continue
            for number, line in enumerate(lines, 1):
                if old in line and not suppressed(lines, number, "DOC009"):
                    findings.append(Finding(
                        rel, number, "DOC009",
                        f"names the retired path {old}; use {new}"))
    return findings


RULES = {
    "DOC001": doc001_links,
    "DOC002": doc002_paths,
    "DOC003": doc003_symlinks,
    "DOC004": doc004_scenario_guidance,
    "DOC005": doc005_scenario_commands,
    "DOC006": doc006_authority,
    "DOC007": doc007_root_budget,
    "DOC008": doc008_adapters,
    "DOC009": doc009_retired,
}


def run(root: Path, codes: Sequence[str] = ENABLED) -> List[Finding]:
    repo = Repo(root)
    findings: List[Finding] = []
    for code in codes:
        findings.extend(RULES[code](repo))
    return sorted(set(findings))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--codes", nargs="+", choices=ALL_CODES,
                        default=list(ENABLED), metavar="CODE")
    args = parser.parse_args(argv)
    findings = run(args.root, args.codes)
    for finding in findings:
        print(finding.render())
    print(f"docs-check: {len(findings)} finding(s); "
          f"rules {', '.join(args.codes)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
