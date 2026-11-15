"""The version is stated in three places. They must agree."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def declared_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    return re.search(r'^version = "([^"]+)"', text, re.M).group(1)


def test_changelog_leads_with_the_declared_version():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    newest = re.search(r"^## (\S+) ", changelog, re.M).group(1)
    assert newest == declared_version()


def test_install_commands_pin_the_declared_tag():
    tag = f"healthcare-sim-sdk.git@v{declared_version()}"
    assert tag in (ROOT / "README.md").read_text()
    assert tag in (ROOT / "CHANGELOG.md").read_text()


def test_design_document_names_the_release():
    header = (ROOT / "docs" / "algorithm_description.md").read_text()[:1500]
    assert f"SDK release {declared_version()}" in header
