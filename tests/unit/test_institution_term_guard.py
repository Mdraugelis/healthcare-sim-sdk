"""Unit tests for the institution-term data boundary guard.

The guard (scripts/check_no_institution_terms.py) is the backstop that
keeps confidential institution values out of tracked content. It checks
three surfaces, each of which has leaked in practice:

  - file contents
  - file and directory NAMES (content rewrites do not touch these)
  - commit messages (pre-commit's commit-msg stage)

These tests use a synthetic denylist in a temp directory; no real
institution terms appear here or are needed to run them.
"""

import importlib.util
from pathlib import Path

import pytest

GUARD_PATH = (
    Path(__file__).resolve().parents[2] / "scripts"
    / "check_no_institution_terms.py"
)


def load_guard():
    """Import the hook as a module (it lives outside the package)."""
    spec = importlib.util.spec_from_file_location("guard", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DENYLIST = "\n".join([
    "# synthetic rules — no real values",
    "AcmeHealth",
    "Firstname Lastname",
    "re:(?<![A-Za-z0-9])SITE[_ -]?A(?![A-Za-z0-9])",
    "",
])


@pytest.fixture
def guard(tmp_path, monkeypatch):
    """Guard module running in a temp repo with a synthetic denylist."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".institution-denylist").write_text(DENYLIST)
    return load_guard()


def test_clean_file_passes(guard, tmp_path):
    (tmp_path / "clean.py").write_text("x = 1\n")
    assert guard.main(["clean.py"]) == 0


def test_term_in_contents_blocks(guard, tmp_path):
    (tmp_path / "config.yaml").write_text("owner: AcmeHealth\n")
    assert guard.main(["config.yaml"]) == 1


def test_term_in_path_name_blocks(guard, tmp_path):
    """A sanitized file can still sit at a leaking path.

    This is the case git filter-repo --replace-text cannot fix: content
    rewrites leave path names untouched.
    """
    target = tmp_path / "configs"
    target.mkdir()
    (target / "acmehealth_projection.yaml").write_text("rate: 0.1\n")
    assert guard.main(["configs/acmehealth_projection.yaml"]) == 1


def test_term_in_directory_name_blocks(guard, tmp_path):
    target = tmp_path / "AcmeHealth" / "sub"
    target.mkdir(parents=True)
    (target / "notes.md").write_text("nothing sensitive\n")
    assert guard.main(["AcmeHealth/sub/notes.md"]) == 1


def test_separator_aware_rule_matches_snake_case_path(guard, tmp_path):
    r"""Campus-code rules must fire inside snake_case path names.

    '_' is a word character, so a '\bSITE_A\b' rule can never match
    'site_a_projection.yaml' — in raw OR normalized form. The denylist
    idiom for this is a separator-aware lookaround, which the example
    file documents. This test pins that idiom working end to end.
    """
    (tmp_path / "site_a_projection.yaml").write_text("rate: 0.1\n")
    assert guard.main(["site_a_projection.yaml"]) == 1


def test_separator_aware_rule_avoids_substring_false_positive(
    guard, tmp_path
):
    """'site_a' inside a longer word must not trip the rule."""
    (tmp_path / "parasite_abdomen.md").write_text("clinical notes\n")
    assert guard.main(["parasite_abdomen.md"]) == 0


def test_multiword_literal_matches_hyphenated_path(guard, tmp_path):
    """A 'Firstname Lastname' rule must catch 'firstname-lastname.md'.

    The raw path has no space, so only the separator-normalized form
    matches. This is what the normalization pass buys.
    """
    (tmp_path / "firstname-lastname.md").write_text("bio\n")
    assert guard.main(["firstname-lastname.md"]) == 1


def test_binary_file_name_still_checked(guard, tmp_path):
    """Contents are unreadable, but the name is not exempt."""
    (tmp_path / "acmehealth_logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    assert guard.main(["acmehealth_logo.png"]) == 1


def test_commit_message_is_scanned(guard, tmp_path):
    """Regression: '.git/' exclusion silently disabled the msg hook.

    pre-commit's commit-msg stage passes the message file as
    '.git/COMMIT_EDITMSG' relative to the repo root. Excluding '.git/'
    wholesale made that stage a no-op.
    """
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "COMMIT_EDITMSG").write_text(
        "feat: rollout for AcmeHealth campus\n"
    )
    assert guard.main([".git/COMMIT_EDITMSG"]) == 1


def test_clean_commit_message_passes(guard, tmp_path):
    """The git-internal path name itself must not trip the name check."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "COMMIT_EDITMSG").write_text("feat: add estimator\n")
    assert guard.main([".git/COMMIT_EDITMSG"]) == 0


def test_denylist_file_itself_is_skipped(guard):
    """The denylist holds the real terms by design."""
    assert guard.main([".institution-denylist"]) == 0


def test_no_denylist_passes(tmp_path, monkeypatch):
    """External contributors without the local denylist are not blocked."""
    monkeypatch.chdir(tmp_path)
    module = load_guard()
    (tmp_path / "config.yaml").write_text("owner: AcmeHealth\n")
    assert module.main(["config.yaml"]) == 0


def test_empty_denylist_passes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".institution-denylist").write_text("# only comments\n")
    module = load_guard()
    (tmp_path / "config.yaml").write_text("owner: AcmeHealth\n")
    assert module.main(["config.yaml"]) == 0


def test_bad_regex_is_skipped_not_fatal(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".institution-denylist").write_text(
        "re:[unclosed\nAcmeHealth\n"
    )
    module = load_guard()
    (tmp_path / "config.yaml").write_text("owner: AcmeHealth\n")
    assert module.main(["config.yaml"]) == 1
    assert "bad regex" in capsys.readouterr().err
