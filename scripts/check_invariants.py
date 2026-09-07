#!/usr/bin/env python3
"""Verify contracts and engine boundaries against .claude/invariants.yaml.

AST guards catch direct access, simple aliases and reflective access; they are
architectural regression checks, not a proof for arbitrary dynamic Python.
No healthcare_sim_sdk imports are needed in the isolated pre-commit environment.
"""

import argparse
import ast
import sys
from pathlib import Path

import yaml


DEFAULT_DEFINITIONS = Path(".claude/invariants.yaml")
REFLECTION = {"getattr", "setattr", "delattr", "hasattr", "isinstance", "issubclass",
              "vars", "dir", "type", "len", "iter", "list", "tuple", "dict", "set",
              "fields", "asdict", "astuple", "is_dataclass", "__getattribute__",
              "getitem"}
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def load_definitions(root=Path("."), definitions=None):
    path = Path(definitions) if definitions else Path(root) / DEFAULT_DEFINITIONS
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Invariant definitions must be a mapping")
    for key in ("scenario_contract", "rng_contract", "engine_constraints",
                "protected_files"):
        if key not in value:
            raise ValueError(f"Missing invariant definition: {key}")
    return value


def _path(root, module):
    return Path(root) / Path(*module.split(".")).with_suffix(".py")


def _tree(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _method_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if (isinstance(node, ast.Call) and _method_name(node.func) == "getattr"
            and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)):
        return node.args[1].value
    return None


def find_abstract_methods(filepath, class_name="BaseScenario"):
    for node in ast.walk(_tree(filepath)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.name for item in node.body
                    if isinstance(item, FUNCTION_NODES)
                    and any(_method_name(dec) == "abstractmethod"
                            for dec in item.decorator_list)}
    return set()


def find_dataclass_fields(filepath, class_name):
    for node in ast.walk(_tree(filepath)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {item.target.id for item in node.body
                    if isinstance(item, ast.AnnAssign)
                    and isinstance(item.target, ast.Name)}
    return set()


def _violation(message):
    print(f"INVARIANT VIOLATION: {message}", file=sys.stderr)
    return False


def check_scenario_contract(root=Path("."), definitions=None):
    try:
        config = definitions or load_definitions(root)
        contract = config["scenario_contract"]
        expected = set(contract["required_abstract_methods"])
        actual = find_abstract_methods(
            _path(root, contract["module"]), contract["class"])
        if actual != expected:
            return _violation(
                f"scenario contract changed\n  Expected: {sorted(expected)}"
                f"\n  Found: {sorted(actual)}")
        return True
    except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
        return _violation(f"cannot validate scenario contract: {exc}")


def check_rng_streams(root=Path("."), definitions=None):
    try:
        config = definitions or load_definitions(root)
        contract = config["rng_contract"]
        path = _path(root, contract["module"])
        expected = set(contract["required_streams"])
        actual = find_dataclass_fields(path, contract["streams_class"])
        if actual != expected:
            return _violation(
                f"RNGStreams fields changed\n  Expected: {sorted(expected)}"
                f"\n  Found: {sorted(actual)}")
        declarations = []
        for node in ast.walk(_tree(path)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target] if isinstance(node, ast.AnnAssign) else [])
            if any(isinstance(target, ast.Name) and target.id == "STREAM_NAMES"
                   for target in targets):
                declarations.append(ast.literal_eval(node.value))
        if (len(declarations) != 1 or not isinstance(declarations[0], (list, tuple))
                or len(declarations[0]) != len(expected)
                or set(declarations[0]) != expected):
            return _violation(
                "STREAM_NAMES missing, duplicated or out of sync "
                "with YAML required_streams")
        return True
    except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
        return _violation(f"cannot validate RNG contract: {exc}")


def _aliases(tree, names):
    """Propagate simple aliases without declaring derived predictions to be state."""
    aliases = set(names)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                value = node.value
                if isinstance(value, ast.Name) and value.id in aliases:
                    targets = (node.targets if isinstance(node, ast.Assign)
                               else [node.target])
                    for target in targets:
                        if isinstance(target, ast.Name) and target.id not in aliases:
                            aliases.add(target.id)
                            changed = True
    return aliases


def engine_violations(source, constraints):
    """Return line-addressed violations for temporary fixtures or the real engine."""
    tree = ast.parse(source)
    definitions = {constraint["name"]: constraint for constraint in constraints}
    supported = {"state_opacity", "counterfactual_isolation"}
    unknown = set(definitions) - supported
    if unknown:
        raise ValueError(
            f"No guard implemented for engine constraints: {sorted(unknown)}")
    violations = set()
    # Analyze each method independently to avoid unrelated local aliases leaking
    # between factual and counterfactual helper methods.
    scopes = [node for node in ast.walk(tree) if isinstance(node, FUNCTION_NODES)]
    containers = [tree] + [node for node in ast.walk(tree)
                           if isinstance(node, ast.ClassDef)]
    for container in containers:
        scopes.append(ast.Module(
            body=[node for node in container.body
                  if not isinstance(node, (*FUNCTION_NODES, ast.ClassDef))],
            type_ignores=[]))
    for scope in scopes:
        states = _aliases(
            scope, definitions.get("state_opacity", {}).get("state_variable_names", ()))
        cf_definition = definitions.get("counterfactual_isolation", {})
        if ("counterfactual_isolation" in definitions
                and "state_variable_names" not in cf_definition):
            raise ValueError(
                "counterfactual_isolation requires explicit state_variable_names")
        counterfactuals = _aliases(scope, cf_definition.get("state_variable_names", ()))
        reflectors = _aliases(scope, REFLECTION)
        predictors = {"predict"}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign):
                if _method_name(node.value) == "predict":
                    predictors.update(target.id for target in node.targets
                                      if isinstance(target, ast.Name))
                if _method_name(node.value) in REFLECTION:
                    reflectors.update(target.id for target in node.targets
                                      if isinstance(target, ast.Name))
        predictors = _aliases(scope, predictors)
        reflectors = _aliases(scope, reflectors)
        for node in ast.walk(scope):
            if isinstance(node, (ast.Attribute, ast.Subscript)):
                if isinstance(node.value, ast.Name) and node.value.id in states:
                    violations.add((node.lineno, "state_opacity", ast.unparse(node)))
            if not isinstance(node, ast.Call):
                continue
            method = _method_name(node.func)
            arguments = list(node.args) + [keyword.value for keyword in node.keywords]
            if method in reflectors and any(
                isinstance(arg, ast.Name) and arg.id in states for arg in arguments
            ):
                violations.add((node.lineno, "state_opacity", ast.unparse(node)))
            if "counterfactual_isolation" in definitions and method in predictors:
                if any(isinstance(item, ast.Name) and item.id in counterfactuals
                       for arg in arguments for item in ast.walk(arg)):
                    violations.add((node.lineno, "counterfactual_isolation",
                                    ast.unparse(node)))
    return [f"Line {line}: {kind}: {expression}"
            for line, kind, expression in sorted(violations)]


def check_engine_state_opacity(root=Path("."), definitions=None):
    """Check all declared engine constraints, including counterfactual isolation."""
    try:
        config = definitions or load_definitions(root)
        source = (Path(root) / "healthcare_sim_sdk/core/engine.py").read_text(
            encoding="utf-8")
        violations = engine_violations(source, config["engine_constraints"])
        if violations:
            return _violation(
                "engine boundary violations\n  " + "\n  ".join(violations))
        return True
    except (OSError, SyntaxError, ValueError, KeyError, TypeError) as exc:
        return _violation(f"cannot validate engine boundaries: {exc}")


def check_protected_files(root=Path("."), definitions=None):
    config = definitions or load_definitions(root)
    missing = [path for path in config["protected_files"]["paths"]
               if not (Path(root) / path).is_file()]
    return _violation(f"protected files missing: {missing}") if missing else True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--definitions", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_definitions(args.root, args.definitions)
        results = [
            check_protected_files(args.root, config),
            check_scenario_contract(args.root, config),
            check_rng_streams(args.root, config),
            check_engine_state_opacity(args.root, config),
        ]
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        _violation(f"cannot load invariant definitions: {exc}")
        return 1
    if all(results):
        print("All invariant checks passed "
              "(YAML-defined contracts and engine boundaries).")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
