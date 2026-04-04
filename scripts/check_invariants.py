#!/usr/bin/env python3
"""Pre-commit hook: verify SDK architectural invariants.

Uses AST parsing — no package imports required. Works in
pre-commit's isolated environment.

Checks:
- BaseScenario has exactly 5 abstract methods
- RNGStreams has exactly 5 fields
- Engine does not inspect state internals
"""

import ast
import sys
from pathlib import Path

EXPECTED_ABSTRACT = {
    "create_population", "step", "predict", "intervene", "measure",
}
EXPECTED_STREAMS = {
    "population", "temporal", "prediction", "intervention", "outcomes",
}

STATE_NAMES = {
    "state_factual", "state_counterfactual",
    "state", "state_snapshot",
}


def find_abstract_methods(filepath: Path) -> set:
    """Find @abstractmethod decorated methods in a file."""
    tree = ast.parse(filepath.read_text())
    methods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Name)
                        and dec.id == "abstractmethod"):
                    methods.add(node.name)
    return methods


def find_dataclass_fields(filepath: Path, class_name: str) -> set:
    """Find annotated fields in a dataclass."""
    tree = ast.parse(filepath.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.ClassDef)
                and node.name == class_name):
            fields = set()
            for item in node.body:
                if isinstance(item, ast.AnnAssign):
                    if isinstance(item.target, ast.Name):
                        fields.add(item.target.id)
            return fields
    return set()


def check_scenario_contract():
    path = Path("healthcare_sim_sdk/core/scenario.py")
    if not path.exists():
        print("SKIP: scenario.py not found (wrong CWD?)")
        return True
    abstract = find_abstract_methods(path)
    if abstract != EXPECTED_ABSTRACT:
        added = abstract - EXPECTED_ABSTRACT
        removed = EXPECTED_ABSTRACT - abstract
        msg = (
            "INVARIANT VIOLATION: 5-method scenario contract\n"
        )
        if added:
            msg += f"  Added: {added}\n"
        if removed:
            msg += f"  Removed: {removed}\n"
        msg += f"  Expected: {EXPECTED_ABSTRACT}\n"
        msg += f"  Found: {abstract}\n"
        print(msg, file=sys.stderr)
        return False
    return True


def check_rng_streams():
    path = Path("healthcare_sim_sdk/core/rng.py")
    if not path.exists():
        print("SKIP: rng.py not found (wrong CWD?)")
        return True
    fields = find_dataclass_fields(path, "RNGStreams")
    if fields != EXPECTED_STREAMS:
        print(
            "INVARIANT VIOLATION: RNGStreams fields changed\n"
            f"  Expected: {EXPECTED_STREAMS}\n"
            f"  Found: {fields}",
            file=sys.stderr,
        )
        return False

    # Also check STREAM_NAMES list
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id == "STREAM_NAMES"
                    for t in (
                        node.targets if isinstance(
                            node.targets, list
                        ) else [node.targets]
                    )
                )):
            if isinstance(node.value, ast.List):
                names = set()
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant):
                        names.add(elt.value)
                if names != EXPECTED_STREAMS:
                    print(
                        "INVARIANT VIOLATION: "
                        "STREAM_NAMES out of sync\n"
                        f"  Expected: {EXPECTED_STREAMS}\n"
                        f"  Found: {names}",
                        file=sys.stderr,
                    )
                    return False
    return True


def check_engine_state_opacity():
    path = Path("healthcare_sim_sdk/core/engine.py")
    if not path.exists():
        print("SKIP: engine.py not found (wrong CWD?)")
        return True
    tree = ast.parse(path.read_text())

    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Name)
                    and fn.id in (
                        "getattr", "isinstance", "hasattr",
                    )
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id in STATE_NAMES):
                violations.append(
                    f"  Line {node.lineno}: "
                    f"{fn.id}({node.args[0].id}, ...)"
                )
    if violations:
        print(
            "INVARIANT VIOLATION: engine inspects state\n"
            + "\n".join(violations),
            file=sys.stderr,
        )
        return False
    return True


if __name__ == "__main__":
    results = [
        check_scenario_contract(),
        check_rng_streams(),
        check_engine_state_opacity(),
    ]
    if all(results):
        print("All invariant checks passed.")
        sys.exit(0)
    else:
        sys.exit(1)
