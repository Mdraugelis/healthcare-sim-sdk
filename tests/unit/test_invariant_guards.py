"""Intentionally bad temporary code proves the architectural guard catches faults."""

import copy
import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("invariant_guard", ROOT / "scripts/check_invariants.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


@pytest.fixture
def definitions():
    return guard.load_definitions(ROOT)


def test_real_repository_contracts_pass(definitions):
    assert guard.check_protected_files(ROOT, definitions)
    assert guard.check_scenario_contract(ROOT, definitions)
    assert guard.check_rng_streams(ROOT, definitions)
    assert guard.check_engine_state_opacity(ROOT, definitions)


@pytest.mark.parametrize("body", [
    "value = state.risk", "value = state['risk']", "value = getattr(state, 'risk')",
    "value = hasattr(state, 'risk')", "value = isinstance(state, dict)",
    "value = vars(state)", "value = type(state)", "value = dir(state)",
    "value = len(state)", "state.risk = 3", "state['risk'] = 3",
    "alias = state\nvalue = alias.risk", "alias = state\nvalue = alias['risk']",
    "inspect = getattr\nvalue = inspect(state, 'risk')",
    "value = builtins.getattr(state, 'risk')", "value = dataclasses.asdict(state)",
    "inspect = builtins.getattr\nvalue = inspect(state, 'risk')",
])
def test_direct_and_reflective_state_inspection_is_rejected(definitions, body):
    source = "def advance(state):\n" + "\n".join("    " + line for line in body.splitlines())
    violations = guard.engine_violations(source, definitions["engine_constraints"])
    assert any("state_opacity" in violation for violation in violations)


@pytest.mark.parametrize("body", [
    "sc.predict(state_counterfactual, t)", "sc.predict(state=state_counterfactual, timestep=t)",
    "alias = state_counterfactual\nsc.predict(alias, t)",
    "sc.predict(state_snapshot, t)", "getattr(sc, 'predict')(state_counterfactual, t)",
    "predictor = sc.predict\npredictor(state_counterfactual, t)",
])
def test_counterfactual_predictions_rejected(definitions, body):
    source = "def advance():\n" + "\n".join("    " + line for line in body.splitlines())
    violations = guard.engine_violations(source, definitions["engine_constraints"])
    assert any("counterfactual_isolation" in violation for violation in violations)


def test_forwarding_state_and_factual_predictions_remain_allowed(definitions):
    source = """def advance():
    state_counterfactual = sc.step(state_counterfactual, t)
    outcomes = sc.measure(state_counterfactual, t)
    predictions = sc.predict(state_factual, t)
    state_snapshot = sc.clone_state(state_factual)
"""
    assert not guard.engine_violations(source, definitions["engine_constraints"])


def test_module_inspection_is_checked_alongside_methods(definitions):
    source = "def valid(state):\n    return sc.step(state, 1)\nvalue = state.risk\n"
    assert any("state_opacity" in item for item in guard.engine_violations(
        source, definitions["engine_constraints"]))


def test_yaml_is_the_source_for_names_and_contracts(tmp_path, definitions):
    config = copy.deepcopy(definitions)
    config["scenario_contract"]["required_abstract_methods"] = ["evolve"]
    config["rng_contract"]["required_streams"] = ["events"]
    config["engine_constraints"][0]["state_variable_names"] = ["payload"]
    path = tmp_path / "healthcare_sim_sdk/core"
    path.mkdir(parents=True)
    (path / "scenario.py").write_text(
        "class BaseScenario:\n    @abstractmethod\n    def evolve(self): pass\n")
    (path / "rng.py").write_text("class RNGStreams:\n    events: object\nSTREAM_NAMES = ['events']\n")
    (path / "engine.py").write_text("value = payload['internal']\n")
    definition_file = tmp_path / "invariants.yaml"
    definition_file.write_text(yaml.safe_dump(config))
    loaded = guard.load_definitions(tmp_path, definition_file)
    assert guard.check_scenario_contract(tmp_path, loaded)
    assert guard.check_rng_streams(tmp_path, loaded)
    assert not guard.check_engine_state_opacity(tmp_path, loaded)


def test_missing_or_duplicate_stream_names_fail(tmp_path, definitions):
    path = tmp_path / "healthcare_sim_sdk/core"
    path.mkdir(parents=True)
    fields = "class RNGStreams:\n" + "".join(
        f"    {name}: object\n" for name in definitions['rng_contract']['required_streams'])
    (path / "rng.py").write_text(fields)
    assert not guard.check_rng_streams(tmp_path, definitions)
    names = definitions['rng_contract']['required_streams']
    (path / "rng.py").write_text(fields + f"STREAM_NAMES = {names + [names[0]]!r}\n")
    assert not guard.check_rng_streams(tmp_path, definitions)


def test_unrelated_abstract_method_does_not_satisfy_contract(tmp_path):
    path = tmp_path / "contract.py"
    path.write_text("class BaseScenario: pass\nclass Other:\n"
                    "    @abstractmethod\n    def create_population(self): pass\n")
    assert guard.find_abstract_methods(path) == set()


def test_missing_protected_file_blocks(tmp_path, definitions):
    assert not guard.check_protected_files(tmp_path, definitions)


def test_unknown_engine_constraint_fails_closed(definitions):
    constraints = definitions["engine_constraints"] + [{"name": "new_unimplemented_rule"}]
    with pytest.raises(ValueError, match="No guard implemented"):
        guard.engine_violations("pass", constraints)


def test_cli_rejects_faulty_temporary_code(tmp_path, definitions):
    path = tmp_path / "healthcare_sim_sdk/core"
    path.mkdir(parents=True)
    for filename in ("scenario.py", "rng.py", "results.py"):
        (path / filename).write_text((ROOT / "healthcare_sim_sdk/core" / filename).read_text())
    (path / "engine.py").write_text("def run(state):\n    return state.value\n")
    definition_file = tmp_path / "definitions.yaml"
    definition_file.write_text(yaml.safe_dump(definitions))
    assert guard.main(["--root", str(tmp_path), "--definitions", str(definition_file)]) == 1
