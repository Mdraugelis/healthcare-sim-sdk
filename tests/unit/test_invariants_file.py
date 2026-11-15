"""config/invariants.yaml is the single source for the SDK contract
(issue #9). These tests hold the code, the lint rules and the documents
that restate it to that file, so a second copy cannot drift unnoticed."""

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
INVARIANTS = yaml.safe_load((ROOT / "config" / "invariants.yaml").read_text())
PROTOCOL = (ROOT / "docs" / "simulation_protocol.md").read_text()
ROOT_GUIDANCE = (ROOT / "AGENTS.md").read_text()

sys.path.insert(0, str(ROOT / "scripts"))
import sdk_lint  # noqa: E402


def load_hook():
    spec = importlib.util.spec_from_file_location(
        "invariants_hook", ROOT / "scripts" / "check_invariants.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    return hook


# -- the file against the code --------------------------------------------
def test_abstract_methods_match_base_scenario():
    from healthcare_sim_sdk.core.scenario import BaseScenario
    assert set(INVARIANTS["scenario_contract"]["required_abstract_methods"]) \
        == set(BaseScenario.__abstractmethods__)


def test_streams_match_rng_streams():
    from healthcare_sim_sdk.core import rng
    streams = INVARIANTS["rng_contract"]["required_streams"]
    cls = getattr(rng, INVARIANTS["rng_contract"]["streams_class"])
    assert set(streams) == {f.name for f in dataclasses.fields(cls)}
    partitioner = getattr(rng, INVARIANTS["rng_contract"]["partitioner_class"])
    # Order matters: seeds are spawned in this order, so reordering the
    # streams would change every seeded result.
    assert streams == list(partitioner.STREAM_NAMES)


def test_every_path_the_file_names_exists():
    paths = list(INVARIANTS["protected_files"]["paths"])
    paths += [v for k, v in INVARIANTS["behavioral_enforcement"].items()
              if k != "note"]
    paths += [INVARIANTS["analytical_first_pass"]["reference_protocol"],
              INVARIANTS["analytical_first_pass"]["worked_example"]]
    for pattern in INVARIANTS["banned_patterns"]:
        paths += pattern["allowed_in"]
    missing = sorted({p for p in paths if not (ROOT / p).exists()})
    assert missing == []


# -- the hook reads the file ----------------------------------------------
def test_hook_takes_its_contract_from_the_file():
    hook = load_hook()
    assert hook.EXPECTED_ABSTRACT == set(
        INVARIANTS["scenario_contract"]["required_abstract_methods"])
    assert hook.EXPECTED_STREAMS == set(
        INVARIANTS["rng_contract"]["required_streams"])
    opacity = next(c for c in INVARIANTS["engine_constraints"]
                   if c["name"] == "state_opacity")
    assert hook.STATE_NAMES == set(opacity["state_variable_names"])
    assert hook.PROTECTED_FILES == INVARIANTS["protected_files"]["paths"]


def test_hook_fails_when_the_file_and_code_disagree(monkeypatch, capsys):
    hook = load_hook()
    monkeypatch.chdir(ROOT)
    assert hook.check_scenario_contract()
    assert hook.check_rng_streams()
    monkeypatch.setattr(hook, "EXPECTED_ABSTRACT",
                        hook.EXPECTED_ABSTRACT | {"explain"})
    monkeypatch.setattr(hook, "EXPECTED_STREAMS",
                        hook.EXPECTED_STREAMS - {"outcomes"})
    monkeypatch.setattr(hook, "PROTECTED_FILES",
                        hook.PROTECTED_FILES + ["healthcare_sim_sdk/gone.py"])
    assert not hook.check_scenario_contract()
    assert not hook.check_rng_streams()
    assert not hook.check_protected_files()
    assert "healthcare_sim_sdk/gone.py" in capsys.readouterr().err


@pytest.mark.parametrize("text, expected", [
    (None, "is missing"),
    ("version: 1\nscenario_contract: {}\n", "lacks a required entry"),
    ("version: [1\n", "not valid YAML"),
])
def test_hook_explains_a_broken_file(tmp_path, monkeypatch, text, expected):
    hook = load_hook()
    monkeypatch.setattr(hook, "ROOT", tmp_path)
    path = tmp_path / "config" / "invariants.yaml"
    if text is not None:
        path.parent.mkdir()
        path.write_text(text)
    with pytest.raises(SystemExit, match=expected):
        hook.load_invariants(path)


# -- the file against the lint rule that enforces it ----------------------
def test_rng_exemptions_match_sdk_lint():
    numpy_rule = next(p for p in INVARIANTS["banned_patterns"]
                      if p["pattern"].startswith("np"))
    assert set(numpy_rule["allowed_in"]) == (
        set(sdk_lint.RNG_APPROVED_FILES) | set(sdk_lint.RNG_APPROVED_DIRS))


# -- the documents that restate the contract ------------------------------
NUMBER_WORDS = {5: "five", 6: "six"}


def test_root_guidance_names_every_method_and_stream():
    for name in INVARIANTS["scenario_contract"]["required_abstract_methods"]:
        assert f"`{name}`" in ROOT_GUIDANCE
    for name in INVARIANTS["rng_contract"]["required_streams"]:
        assert re.search(rf"\b{name}\b", ROOT_GUIDANCE)


def test_fitness_criteria_agree_across_file_protocol_and_root():
    criteria = INVARIANTS["fitness_criteria"]["checks"]
    screen = PROTOCOL.split("### Quick Fitness Screen", 1)[1].split("###", 1)[0]
    questions = re.findall(r"^\*\*[^*]+\?\*\*", screen, flags=re.M)
    assert len(questions) == len(criteria)
    assert f"{NUMBER_WORDS[len(criteria)]} fitness criteria" in ROOT_GUIDANCE


def test_protocol_phases_and_levels_match_root_guidance():
    phases = re.findall(r"^## Phase (\d): ", PROTOCOL, flags=re.M)
    levels = re.findall(r"^### Level (\d): ", PROTOCOL, flags=re.M)
    assert phases == [str(n) for n in range(1, 7)]
    assert levels == [str(n) for n in range(1, 6)]
    assert f"The {NUMBER_WORDS[len(levels)]} levels:" in ROOT_GUIDANCE
    mandatory = ROOT_GUIDANCE.split("The five levels:", 1)[1].split("4. ", 1)[0]
    assert len(re.findall(r"^\s+- ", mandatory, flags=re.M)) == len(levels)
