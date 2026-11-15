"""Each documentation rule fires on a broken tree and passes on a good one
(issue #9). Trees are built in tmp_path, so no broken link or dangling
symlink is ever tracked."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import check_docs  # noqa: E402

SCENARIO = "healthcare_sim_sdk/scenarios/demo"
GOOD_SCENARIO_GUIDE = """# Demo

**Status:** operational

## What it models
One clinic.

## Entry points
- Config: `configs/base.yaml`
- Runner: `python -m healthcare_sim_sdk.scenarios.demo.run_evaluation`

## Verify
- `pytest tests/integration/test_demo.py::TestDemo`

## Private files
None.

## Known limits
- None recorded.
"""
MAP_HEADER = "| Path | Class | Notes |\n|---|---|---|\n"


def write(root, rel, text=""):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def tree(tmp_path):
    """A small tree that passes every rule."""
    write(tmp_path, "AGENTS.md",
          "# Rules\n\nSee [the map](docs/AUTHORITY.md) and `docs/guide.md`.\n")
    os.symlink("AGENTS.md", tmp_path / "CLAUDE.md")
    write(tmp_path, "config/invariants.yaml", "version: 1\n")
    (tmp_path / ".claude").mkdir()
    os.symlink("../config/invariants.yaml",
               tmp_path / ".claude" / "invariants.yaml")
    write(tmp_path, ".claude/agents/guide.md",
          "---\nname: guide\ndescription: Follows the guide.\n---\nRead it.\n")
    write(tmp_path, ".claude/rules/nested.md",
          "---\npaths:\n  - \"healthcare_sim_sdk/**\"\n---\nRead AGENTS.md.\n")
    write(tmp_path, "docs/guide.md", "# Guide\n\nRun `scripts/run.py`.\n")
    write(tmp_path, "docs/history.md",
          "# History\n\nWe removed `scripts/gone.py`.\n")
    write(tmp_path, "scripts/run.py")
    write(tmp_path, f"{SCENARIO}/scenario.py")
    write(tmp_path, f"{SCENARIO}/run_evaluation.py")
    write(tmp_path, f"{SCENARIO}/configs/base.yaml")
    write(tmp_path, f"{SCENARIO}/AGENTS.md", GOOD_SCENARIO_GUIDE)
    write(tmp_path, "tests/integration/test_demo.py")
    write(tmp_path, "docs/AUTHORITY.md", MAP_HEADER + (
        "| `AGENTS.md` | current | root |\n"
        "| `CLAUDE.md` | current | adapter |\n"
        "| `.claude/` | current | adapters |\n"
        "| `docs/AUTHORITY.md` | current | this |\n"
        "| `docs/guide.md` | current | guide |\n"
        "| `docs/history.md` | record | history |\n"
        "| `healthcare_sim_sdk/` | current | local |\n"))
    return tmp_path


def found(root, code):
    return [f for f in check_docs.run(root, [code])]


def test_good_tree_passes_every_rule(tree):
    assert check_docs.run(tree, check_docs.ALL_CODES) == []


# -- DOC001 ---------------------------------------------------------------
def test_doc001_broken_link(tree):
    write(tree, "docs/guide.md", "See [gone](missing.md) and [ok](guide.md).\n")
    (finding,) = found(tree, "DOC001")
    assert (finding.path, finding.line) == ("docs/guide.md", 1)
    assert "missing.md" in finding.message


def test_doc001_ignores_urls_anchors_and_code(tree):
    write(tree, "docs/guide.md", (
        "[a](https://example.com/x.md) [b](#top) [c](mailto:a@b.c)\n"
        "`[d](not-a-link.md)`\n"
        "```\n[e](in-a-fence.md)\n```\n"
        "[f](guide.md#section)\n"))
    assert found(tree, "DOC001") == []


def test_doc001_link_cannot_leave_the_tree(tree):
    write(tree, "docs/guide.md", "[up](../../outside.md)\n")
    assert len(found(tree, "DOC001")) == 1


# -- DOC002 ---------------------------------------------------------------
def test_doc002_dead_path_in_current_document(tree):
    write(tree, "docs/guide.md", "Run `scripts/gone.py --fast`.\n")
    (finding,) = found(tree, "DOC002")
    assert "scripts/gone.py" in finding.message


def test_doc002_records_may_name_removed_code(tree):
    # docs/history.md names scripts/gone.py and is classed as a record.
    assert found(tree, "DOC002") == []


def test_doc002_skips_placeholders_and_strips_line_numbers(tree):
    write(tree, "docs/guide.md", (
        "`scripts/<name>.py` `tests/**/test_*.py` `scripts/run.py:12` "
        "`tests/integration/test_demo.py::TestDemo::test_x`\n"))
    assert found(tree, "DOC002") == []


def test_doc002_suppression_needs_a_reason(tree):
    body = "Was `scripts/gone.py`. <!-- docs-check: allow DOC002{} -->\n"
    write(tree, "docs/guide.md", body.format(" -- removed in 0.4"))
    assert found(tree, "DOC002") == []
    write(tree, "docs/guide.md", body.format(""))
    assert len(found(tree, "DOC002")) == 1


# -- DOC003 ---------------------------------------------------------------
def test_doc003_copy_instead_of_symlink(tree):
    (tree / "CLAUDE.md").unlink()
    write(tree, "CLAUDE.md", "# Rules\n\nA copy that will drift.\n")
    (finding,) = found(tree, "DOC003")
    assert "must be a symlink" in finding.message


def test_doc003_wrong_target(tree):
    (tree / "CLAUDE.md").unlink()
    os.symlink("README.md", tree / "CLAUDE.md")
    (finding,) = found(tree, "DOC003")
    assert "expected 'AGENTS.md'" in finding.message


def test_doc003_missing_required_link(tree):
    (tree / "CLAUDE.md").unlink()
    (finding,) = found(tree, "DOC003")
    assert "ln -s AGENTS.md CLAUDE.md" in finding.message


def test_doc003_import_line_is_accepted(tree):
    (tree / "CLAUDE.md").unlink()
    write(tree, "CLAUDE.md", "@AGENTS.md\n")
    assert found(tree, "DOC003") == []


def test_doc003_checkout_without_symlink_support(tree):
    # Git writes the target path into a plain file when symlinks are off.
    (tree / "CLAUDE.md").unlink()
    write(tree, "CLAUDE.md", "AGENTS.md")
    (finding,) = found(tree, "DOC003")
    assert "core.symlinks" in finding.message


def test_doc003_optional_link_may_be_absent(tree):
    (tree / ".claude" / "invariants.yaml").unlink()
    assert found(tree, "DOC003") == []


def test_doc003_dangling_symlink(tree):
    os.symlink("nowhere.md", tree / "docs" / "stray.md")
    (finding,) = found(tree, "DOC003")
    assert finding.path == "docs/stray.md"
    assert "dangling" in finding.message


# -- DOC004 / DOC005 ------------------------------------------------------
def test_doc004_scenario_without_guidance(tree):
    (tree / SCENARIO / "AGENTS.md").unlink()
    (finding,) = found(tree, "DOC004")
    assert finding.path == f"{SCENARIO}/scenario.py"
    assert "_template/AGENTS.md" in finding.message


def test_doc004_missing_heading_and_bad_status(tree):
    text = GOOD_SCENARIO_GUIDE.replace("## Verify", "## Checks").replace(
        "operational", "finished")
    write(tree, f"{SCENARIO}/AGENTS.md", text)
    messages = [f.message for f in found(tree, "DOC004")]
    assert any("'## Verify'" in m for m in messages)
    assert any("status 'finished'" in m for m in messages)


def test_doc005_dead_runner_test_and_config(tree):
    text = (GOOD_SCENARIO_GUIDE
            .replace("run_evaluation`", "run_missing`")
            .replace("test_demo.py", "test_absent.py")
            .replace("configs/base.yaml", "configs/absent.yaml"))
    write(tree, f"{SCENARIO}/AGENTS.md", text)
    messages = sorted(f.message for f in found(tree, "DOC005"))
    assert len(messages) == 3
    assert "configs/absent.yaml" in messages[0]
    assert "test_absent.py" in messages[1]
    assert "run_missing" in messages[2]


def test_doc005_template_placeholders_are_skipped(tree):
    write(tree, f"{SCENARIO}/AGENTS.md", GOOD_SCENARIO_GUIDE.replace(
        "scenarios.demo.run_evaluation", "scenarios.<name>.<runner>"))
    assert found(tree, "DOC005") == []


# -- DOC006 ---------------------------------------------------------------
def test_doc006_unclassified_document(tree):
    write(tree, "docs/new.md", "# New\n")
    (finding,) = found(tree, "DOC006")
    assert finding.path == "docs/new.md"


def test_doc006_stale_duplicate_and_unknown_rows(tree):
    path = tree / "docs" / "AUTHORITY.md"
    path.write_text(path.read_text() + (
        "| `docs/removed.md` | record | gone |\n"
        "| `docs/guide.md` | record | twice |\n"
        "| `docs/history.md` | canon | bad class |\n"))
    messages = " ".join(f.message for f in found(tree, "DOC006"))
    assert "matches no file" in messages
    assert "listed twice" in messages
    assert "unknown class 'canon'" in messages


def test_doc006_most_specific_row_wins(tree):
    write(tree, "docs/AUTHORITY.md", MAP_HEADER + (
        "| `docs/` | record | folder |\n"
        "| `docs/*.md` | rationale | pattern |\n"
        "| `docs/guide.md` | current | exact |\n"))
    repo = check_docs.Repo(tree)
    assert repo.class_of("docs/guide.md") == "current"
    assert repo.class_of("docs/history.md") == "rationale"
    assert repo.class_of("AGENTS.md") is None


def test_doc006_missing_map(tree):
    (tree / "docs" / "AUTHORITY.md").unlink()
    (finding,) = found(tree, "DOC006")
    assert "missing" in finding.message


# -- DOC007 / DOC008 / DOC009 ---------------------------------------------
def test_doc007_root_over_budget(tree):
    write(tree, "AGENTS.md", "line\n" * (check_docs.ROOT_BUDGET + 10))
    (finding,) = found(tree, "DOC007")
    assert f"limit is {check_docs.ROOT_BUDGET}" in finding.message
    write(tree, "AGENTS.md", "line\n" * check_docs.ROOT_BUDGET)
    assert found(tree, "DOC007") == []


def test_doc008_agent_without_frontmatter(tree):
    write(tree, ".claude/agents/guide.md", "# Guide\n\nNo frontmatter.\n")
    (finding,) = found(tree, "DOC008")
    assert "not registered" in finding.message


def test_doc008_agent_missing_description_and_unscoped_rule(tree):
    write(tree, ".claude/agents/guide.md", "---\nname: guide\n---\nBody\n")
    write(tree, ".claude/rules/nested.md", "Always do this.\n")
    messages = " ".join(f.message for f in found(tree, "DOC008"))
    assert "no 'description'" in messages
    assert "no 'paths' frontmatter" in messages


def test_doc009_retired_path(tree):
    write(tree, "docs/guide.md", "Read `.claude/invariants.yaml` first.\n")
    write(tree, "docs/history.md", "It was `.claude/invariants.yaml`.\n")
    (finding,) = found(tree, "DOC009")  # the record is exempt
    assert finding.path == "docs/guide.md"
    assert "config/invariants.yaml" in finding.message


# -- the tool itself ------------------------------------------------------
def test_output_is_sorted_and_stable(tree):
    write(tree, "docs/guide.md", "[b](zz.md) [a](aa.md)\n[c](mm.md)\n")
    first = [f.render() for f in check_docs.run(tree, ["DOC001"])]
    assert first == sorted(first)
    assert first == [f.render() for f in check_docs.run(tree, ["DOC001"])]


def test_git_checkout_and_plain_tree_agree(tree):
    """The gate checks an exported snapshot with no .git; a developer
    checks a checkout. Both must see the same files."""
    plain = [f.render() for f in check_docs.run(tree, check_docs.ALL_CODES)]
    write(tree, ".gitignore", "outputs/\n")
    write(tree, "outputs/notes.md", "[x](gone.md)\n")  # ignored, never read
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    write(tree, "docs/untracked.md", "# Not added yet\n")
    in_git = [f.render() for f in check_docs.run(tree, check_docs.ALL_CODES)]
    assert plain == []
    assert in_git == [
        "docs/untracked.md:1: DOC006 not classified in docs/AUTHORITY.md; "
        "add a row saying whether it is current, rationale, proposal or "
        "record"]


def test_cli_exit_status(tree, capsys):
    assert check_docs.main(["--root", str(tree)]) == 0
    write(tree, "docs/guide.md", "[gone](missing.md)\n")
    assert check_docs.main(["--root", str(tree), "--codes", "DOC001"]) == 1
    assert "docs/guide.md:1: DOC001" in capsys.readouterr().out


def test_this_repository_passes_its_enabled_rules():
    assert [f.render() for f in check_docs.run(ROOT)] == []
