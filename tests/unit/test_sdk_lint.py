"""Positive and negative fixtures for each custom lint rule (issue #8)."""

import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "lint_fixtures"
sys.path.insert(0, str(ROOT / "scripts"))
import sdk_lint  # noqa: E402


def lint(name):
    text = (FIXTURES / name).read_text()
    rel = text.splitlines()[0].split("lint-as:", 1)[1].strip()
    return sdk_lint.lint_source(text, rel)


def codes(findings, severity=None):
    return Counter(f.code for f in findings
                   if severity is None or f.severity == severity)


@pytest.mark.parametrize("name", sorted(
    p.name for p in FIXTURES.glob("*_good.py")))
def test_good_fixtures_are_clean(name):
    assert lint(name) == []


def test_sdk001_finds_every_bypass():
    found = lint("sdk001_bad.py")
    assert codes(found) == {"SDK001": 12}
    lines = {f.line for f in found}
    # imports (2, 4-6), alias call, reference, npr, nr, random, getattr,
    # rvs, and a call to a name imported from numpy.random
    assert lines == {2, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16}
    assert all(f.severity == "block" for f in found)


def test_sdk001_respects_approved_paths():
    text = (FIXTURES / "sdk001_bad.py").read_text()
    for rel in ("healthcare_sim_sdk/core/rng.py", "tests/unit/test_x.py",
                "scripts/validate_its_estimator.py"):
        assert codes(sdk_lint.lint_source(text, rel))["SDK001"] == 0


def test_sdk002_vacuous_asserts():
    found = lint("sdk002_bad.py")
    assert codes(found, "block") == {"SDK002": 4}


def test_sdk003_blocks_lifecycle_and_warns_otherwise():
    found = lint("sdk003_bad.py")
    assert codes(found, "block") == {"SDK003": 1}
    assert codes(found, "warn") == {"SDK003": 1}
    blocked = next(f for f in found if f.severity == "block")
    assert "finalize_experiment" in blocked.message


def test_sdk003_not_reported_in_tests():
    text = (FIXTURES / "sdk003_bad.py").read_text()
    assert sdk_lint.lint_source(text, "tests/unit/test_x.py") == []


def test_sdk004_and_sdk005():
    found = lint("sdk004_005_bad.py")
    assert codes(found, "warn") == {"SDK004": 1, "SDK005": 1}
    assert codes(found, "block") == {}


def test_sdk007_runner_outside_the_contract():
    found = lint("sdk007_bad.py")
    assert codes(found, "block") == {"SDK007": 6}
    messages = " ".join(f.message for f in found)
    for expected in ("imports argparse", "reads sys.argv",
                     "calls save_experiment()", "calls ExperimentCatalog()",
                     "does not start through Hydra",
                     "never calls finalize_experiment()"):
        assert expected in messages


def test_sdk007_applies_only_to_launchable_scenario_scripts():
    text = (FIXTURES / "sdk007_bad.py").read_text()
    # The same code outside scenarios/, or without a __main__ block, is
    # not a runner.
    assert codes(sdk_lint.lint_source(text, "scripts/tool.py"))["SDK007"] == 0
    library = text.split("if __name__")[0]
    assert codes(sdk_lint.lint_source(
        library, "healthcare_sim_sdk/scenarios/example/helpers.py"))[
            "SDK007"] == 0


def test_sdk007_cannot_be_suppressed_with_a_comment():
    text = (FIXTURES / "sdk007_bad.py").read_text().replace(
        "import argparse\n",
        "import argparse  # sdk-lint: allow SDK007 -- just this once\n")
    found = sdk_lint.lint_source(
        text, "healthcare_sim_sdk/scenarios/example/run_old_style.py")
    assert codes(found, "block") == {"SDK007": 6, "SDK000": 1}


def test_sdk007_exemption_needs_a_listed_reason(monkeypatch):
    good = (FIXTURES / "sdk007_good.py").read_text()
    start = good.index("    finalize_experiment(")
    end = good.index("require_validated=True)") + len(
        "require_validated=True)")
    tool = good[:start] + "    print(run_dir(), settings)" + good[end:]
    rel = "healthcare_sim_sdk/scenarios/example/diagnostic.py"
    (finding,) = sdk_lint.lint_source(tool, rel)
    assert "never calls finalize_experiment()" in finding.message
    monkeypatch.setattr(sdk_lint, "_EXEMPTIONS", {rel})
    assert sdk_lint.lint_source(tool, rel) == []
    # An exempt script that does finalize must leave the list.
    (stale,) = sdk_lint.lint_source(good, rel)
    assert "listed as exempt" in stale.message


def test_sdk007_finalize_must_be_strict_and_carry_checks():
    """Three calls: no arguments, strict switched off, and strict passed as
    a variable the rule cannot read. All are blocked."""
    found = lint("sdk007_unverified.py")
    assert codes(found, "block") == {"SDK007": 4}
    messages = [f.message for f in found]
    assert sum("without require_validated=True" in m for m in messages) == 3
    assert sum("without validation_checks=" in m for m in messages) == 1
    assert lint("sdk007_good.py") == []


def test_sdk006_unfinished_contract_methods():
    found = lint("sdk006_bad.py")
    names = sorted(f.message.split(".")[1].split(" ")[0] for f in found)
    assert names == ["intervene", "measure", "predict", "step"]
    assert codes(found, "warn") == {"SDK006": 4}


def test_suppressions_need_reasons_and_must_be_used():
    found = lint("sdk000_bad.py")
    messages = sorted(f.message for f in found if f.code == "SDK000")
    assert len(messages) == 3
    assert any("needs a reason" in m for m in messages)
    assert any("stale suppression" in m for m in messages)
    assert any("malformed" in m for m in messages)
    # The reasonless suppression does not hide the finding it targets.
    assert codes(found)["SDK003"] == 1


def test_strings_are_not_suppressions():
    text = 'DOC = "# sdk-lint: allow SDK003"\n'
    assert sdk_lint.lint_source(text, "healthcare_sim_sdk/x.py") == []


def test_output_is_sorted_and_deterministic():
    first = lint("sdk001_bad.py")
    assert [(f.path, f.line, f.col) for f in first] == sorted(
        (f.path, f.line, f.col) for f in first)
    assert [f.render() for f in first] == [
        f.render() for f in lint("sdk001_bad.py")]


def test_repository_has_no_blocking_findings():
    result = subprocess.run(
        [sys.executable, "scripts/sdk_lint.py"], cwd=ROOT,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fixtures_are_excluded_from_repository_lint():
    findings = sdk_lint.lint_paths(["tests"], ROOT)
    assert not [f for f in findings if "lint_fixtures" in f.path]
