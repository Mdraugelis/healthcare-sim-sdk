#!/usr/bin/env python3
"""SDK-specific lint rules (issue #8). Deterministic, AST-based.

Rules (see docs/lint_rules.md):
  SDK001 block  Randomness outside RNGPartitioner (stdlib random, numpy.random
                members other than Generator, getattr(np, "random"), SciPy
                .rvs() without random_state).
  SDK002 block  Vacuous assertion: a constant-truthy test, a tuple test, or a
                comparison of an expression with itself.
  SDK003 block  A broad except that swallows a failure of the experiment
                lifecycle (save, register, finalize).
         warn   Any other broad except (bare, Exception, BaseException) whose
                handler does not re-raise. Not reported in tests.
  SDK004 warn   Test function with no check (no assert, pytest.raises/warns,
                or call to an assert*/check*/verify*/testing helper).
  SDK005 warn   Test whose every assertion sits inside an if-branch, so the
                test can pass having checked nothing.
  SDK006 warn   BaseScenario subclass with an unfinished contract method
                (body is only pass, ..., or raise NotImplementedError).
  SDK000 block  Malformed or stale suppression.

Suppress one finding on its line, or on the line directly above, with a
reason:   # sdk-lint: allow SDK003 -- diagnostic runner, failure re-raised later

Usage: python scripts/sdk_lint.py [--rule SDK001] [--strict] PATH ...
Exit 1 when any block-severity finding remains (or any finding with --strict).
"""

import argparse
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

BLOCK, WARN = "block", "warn"

# Approved locations for direct RNG construction (SDK001).
RNG_APPROVED_FILES = {
    "healthcare_sim_sdk/core/rng.py",
    # Pure statistical Monte Carlo on the ITS estimator; no scenario/engine.
    "scripts/validate_its_estimator.py",
}
RNG_APPROVED_DIRS = ("tests/",)
RNG_ANNOTATION_NAMES = {"Generator"}

LIFECYCLE_CALLS = {
    "finalize_experiment", "save_experiment", "register_experiment",
    "register_sweep", "write_provenance",
}
CONTRACT_METHODS = ("create_population", "step", "predict", "intervene",
                    "measure")
CHECK_CALL = re.compile(r"(^|_)(assert|check|verify|validate)|^testing$|"
                        r"^assert_|^raises$|^warns$|^approx$|^fail$")
SUPPRESS = re.compile(
    r"#\s*sdk-lint:\s*allow\s+(SDK\d{3})(?:\s*--\s*(.*\S))?")
BROAD = {"Exception", "BaseException"}

# Lint fixtures are deliberately bad code; they are linted only by tests.
EXCLUDED_DIRS = ("tests/lint_fixtures/",)


@dataclass(frozen=True, order=True)
class Finding:
    path: str
    line: int
    col: int
    code: str
    severity: str
    message: str

    def render(self) -> str:
        return (f"{self.path}:{self.line}:{self.col}: {self.code} "
                f"[{self.severity}] {self.message}")


def _posix(path: Path, root: Optional[Path]) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve()) if root else path
    except ValueError:
        rel = path
    return rel.as_posix()


def _is_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return (rel.startswith("tests/") or "/tests/" in rel
            or name.startswith("test_") or name == "conftest.py")


def _dotted(node: ast.AST) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


class _Checker(ast.NodeVisitor):
    def __init__(self, rel: str, lines: Sequence[str]):
        self.rel = rel
        self.lines = lines
        self.findings: List[Finding] = []
        self.is_test = _is_test_file(rel)
        self.rng_exempt = (rel in RNG_APPROVED_FILES
                           or rel.startswith(RNG_APPROVED_DIRS))
        self.numpy_names: Set[str] = set()
        self.numpy_random_names: Set[str] = set()
        self.stdlib_random_names: Set[str] = set()
        self.imported_rng_names: Set[str] = set()
        self.scenario_bases = {"BaseScenario"}

    def add(self, node, code, severity, message):
        self.findings.append(Finding(
            self.rel, getattr(node, "lineno", 1),
            getattr(node, "col_offset", 0) + 1, code, severity, message))

    # ---- SDK001: randomness ---------------------------------------------
    def visit_Import(self, node):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            if alias.name == "numpy":
                self.numpy_names.add(bound)
            elif alias.name == "numpy.random":
                if alias.asname:
                    self.numpy_random_names.add(alias.asname)
                else:
                    self.numpy_names.add("numpy")
                self._rng(node, "numpy.random imported as a module")
            elif alias.name == "random":
                self.stdlib_random_names.add(bound)
                self._rng(node, "stdlib 'random' imported")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "random":
            self._rng(node, "import from stdlib 'random'")
        elif node.module == "numpy.random":
            bad = [a for a in node.names if a.name not in RNG_ANNOTATION_NAMES]
            if bad:
                self._rng(node, f"import of numpy.random.{bad[0].name}")
            self.imported_rng_names.update(a.asname or a.name for a in bad)
        elif node.module == "numpy":
            for alias in node.names:
                if alias.name == "random":
                    self.numpy_random_names.add(alias.asname or "random")
                    self._rng(node, "'random' imported from numpy")
        if node.module and node.module.split(".")[-1] == "scenario":
            for alias in node.names:
                if alias.name == "BaseScenario":
                    self.scenario_bases.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "random":
            root = node.value.value
            if (isinstance(root, ast.Name) and root.id in self.numpy_names
                    and node.attr not in RNG_ANNOTATION_NAMES):
                self._rng(node, f"numpy.random.{node.attr} used directly")
        elif (isinstance(node.value, ast.Name)
              and node.value.id in self.numpy_random_names
              and node.attr not in RNG_ANNOTATION_NAMES):
            self._rng(node, f"numpy.random.{node.attr} used directly")
        elif (isinstance(node.value, ast.Name)
              and node.value.id in self.stdlib_random_names):
            self._rng(node, f"random.{node.attr} used directly")
        self.generic_visit(node)

    def visit_Call(self, node):
        name = _call_name(node)
        if isinstance(node.func, ast.Name) and name in self.imported_rng_names:
            self._rng(node, f"call to numpy.random.{name}")
        if (name == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in self.numpy_names
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "random"):
            self._rng(node, "getattr(numpy, 'random')")
        if (name == "rvs" and isinstance(node.func, ast.Attribute)
                and not any(k.arg == "random_state" for k in node.keywords)):
            self._rng(node, "SciPy .rvs() without random_state uses global "
                            "NumPy state")
        self.generic_visit(node)

    def _rng(self, node, what):
        if not self.rng_exempt:
            self.add(node, "SDK001", BLOCK,
                     f"{what}; draw from an RNGPartitioner stream")

    # ---- SDK002: vacuous assertions --------------------------------------
    def visit_Assert(self, node):
        test = node.test
        if isinstance(test, ast.Tuple):
            self.add(node, "SDK002", BLOCK,
                     "assert on a tuple is always true")
        elif isinstance(test, ast.Constant) and bool(test.value):
            self.add(node, "SDK002", BLOCK,
                     f"assert on constant {test.value!r} is always true")
        elif (isinstance(test, ast.Compare) and len(test.ops) == 1
              and isinstance(test.ops[0], (ast.Eq, ast.Is, ast.LtE, ast.GtE))
              and ast.dump(test.left) == ast.dump(test.comparators[0])
              and not isinstance(test.left, ast.Call)):
            self.add(node, "SDK002", BLOCK,
                     "assert compares an expression with itself")
        self.generic_visit(node)

    # ---- SDK003: swallowed failures --------------------------------------
    def visit_Try(self, node):
        if not self.is_test:
            calls = {_call_name(n) for stmt in node.body
                     for n in ast.walk(stmt) if isinstance(n, ast.Call)}
            lifecycle = sorted(calls & LIFECYCLE_CALLS)
            for handler in node.handlers:
                broad = handler.type is None or (
                    isinstance(handler.type, ast.Name)
                    and handler.type.id in BROAD)
                reraises = any(isinstance(n, ast.Raise)
                               for stmt in handler.body for n in ast.walk(stmt))
                if broad and not reraises:
                    if lifecycle:
                        self.add(handler, "SDK003", BLOCK,
                                 f"broad except swallows a failure of "
                                 f"{', '.join(lifecycle)}; let it raise")
                    else:
                        self.add(handler, "SDK003", WARN,
                                 "broad except does not re-raise; record "
                                 "the failure or narrow the exception")
        self.generic_visit(node)

    visit_TryStar = visit_Try

    # ---- SDK004/SDK005: tests that cannot fail ---------------------------
    def visit_FunctionDef(self, node):
        if self.is_test and node.name.startswith("test"):
            self._check_test(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_test(self, node):
        asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
        checks = [n for n in ast.walk(node)
                  if isinstance(n, ast.Call) and CHECK_CALL.search(
                      _call_name(n) or "")]
        if not asserts and not checks:
            self.add(node, "SDK004", WARN,
                     f"test '{node.name}' contains no check")
            return
        if asserts and not checks:
            unconditional = [a for a in asserts
                             if not _inside_if(node, a)]
            if not unconditional:
                self.add(node, "SDK005", WARN,
                         f"every assertion in '{node.name}' is inside an if; "
                         "the test can pass without checking anything")

    # ---- SDK006: unfinished scenario methods -----------------------------
    def visit_ClassDef(self, node):
        bases = {(_dotted(b) or "").split(".")[-1] for b in node.bases}
        bases |= {(_dotted(b.value) or "").split(".")[-1]
                  for b in node.bases if isinstance(b, ast.Subscript)}
        if bases & self.scenario_bases:
            for item in node.body:
                if (isinstance(item, ast.FunctionDef)
                        and item.name in CONTRACT_METHODS
                        and _unfinished(item)):
                    self.add(item, "SDK006", WARN,
                             f"{node.name}.{item.name} is unfinished")
        self.generic_visit(node)


def _inside_if(func: ast.AST, target: ast.AST) -> bool:
    for n in ast.walk(func):
        if isinstance(n, ast.If) and not n.orelse:
            if any(target is m for stmt in n.body for m in ast.walk(stmt)):
                return True
    return False


def _unfinished(fn: ast.FunctionDef) -> bool:
    body = [s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                    and isinstance(s.value.value, str))]
    if not body:
        return True
    if len(body) != 1:
        return False
    stmt = body[0]
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
            and stmt.value.value is Ellipsis:
        return True
    if isinstance(stmt, ast.Raise) and stmt.exc is not None:
        target = stmt.exc.func if isinstance(stmt.exc, ast.Call) else stmt.exc
        return _dotted(target) == "NotImplementedError"
    return False


def _comments(source: str):
    """(line, text) for real comments only; strings are not suppressions."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                yield tok.start[0], tok.string
    except (tokenize.TokenError, IndentationError):
        return


def _apply_suppressions(rel: str, lines: Sequence[str],
                        findings: List[Finding]) -> List[Finding]:
    marks = {}
    out: List[Finding] = []
    for number, text in _comments("\n".join(lines)):
        if "sdk-lint:" not in text:
            continue
        m = SUPPRESS.search(text)
        if not m:
            out.append(Finding(rel, number, 1, "SDK000", BLOCK,
                               "malformed sdk-lint comment; use "
                               "'# sdk-lint: allow SDK00N -- reason'"))
            continue
        if not m.group(2):
            out.append(Finding(rel, number, 1, "SDK000", BLOCK,
                               f"suppression of {m.group(1)} needs a reason "
                               "after '--'"))
            continue
        marks[number] = m.group(1)
    used = set()
    for f in findings:
        hit = next((ln for ln in (f.line, f.line - 1)
                    if marks.get(ln) == f.code), None)
        if hit is None:
            out.append(f)
        else:
            used.add(hit)
    for number, code in marks.items():
        if number not in used:
            out.append(Finding(rel, number, 1, "SDK000", BLOCK,
                               f"stale suppression: no {code} finding here"))
    return out


def lint_source(source: str, rel: str) -> List[Finding]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [Finding(rel, exc.lineno or 1, 1, "SDK000", BLOCK,
                        f"syntax error: {exc.msg}")]
    checker = _Checker(rel, lines)
    checker.visit(tree)
    return sorted(set(_apply_suppressions(rel, lines, checker.findings)))


def iter_files(paths: Iterable[str], root: Path) -> List[Path]:
    files = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            files.append(path)
    return files


def lint_paths(paths: Iterable[str], root: Path = Path("."),
               rules: Optional[Set[str]] = None) -> List[Finding]:
    findings: List[Finding] = []
    for path in iter_files(paths, root):
        rel = _posix(path, root)
        if rel.startswith(EXCLUDED_DIRS) or "/.venv" in "/" + rel:
            continue
        for f in lint_source(path.read_text(encoding="utf-8"), rel):
            if rules is None or f.code in rules or f.code == "SDK000":
                findings.append(f)
    return sorted(findings)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="*",
                        default=["healthcare_sim_sdk", "scripts", "tests"])
    parser.add_argument("--rule", action="append",
                        help="Only report this rule (repeatable)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as blocking")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    findings = lint_paths(args.paths, args.root,
                          set(args.rule) if args.rule else None)
    for f in findings:
        print(f.render())
    blocking = [f for f in findings
                if f.severity == BLOCK or args.strict]
    warns = sum(f.severity == WARN for f in findings)
    print(f"sdk-lint: {len(findings) - warns} blocking, {warns} warnings",
          file=sys.stderr)
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
