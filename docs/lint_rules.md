# Lint rules and the quality gate

These checks look for concrete quality problems: tests that cannot fail, failures that are hidden, and randomness that bypasses the SDK's RNG contract. They judge the code, not who or what wrote it. Every check is deterministic: the same source gives the same output, in the same order.

Run everything with one command (see [the gate](#the-quality-gate)):

```bash
scripts/quality_gate.sh
```

## Tools

| Tool | Configuration | Scope |
|---|---|---|
| Flake8 | `.flake8` (Flake8 does not read `pyproject.toml`) | `healthcare_sim_sdk/`, `tests/`, `scripts/` |
| Ruff, targeted rules only | `[tool.ruff]` in `pyproject.toml` | Whole repository |
| sdk-lint | `scripts/sdk_lint.py` | `healthcare_sim_sdk/`, `scripts/`, `tests/` |
| Invariant check | `scripts/check_invariants.py`, reading `config/invariants.yaml` | Scenario contract, RNG streams, state opacity, protected files |
| docs-check | `scripts/check_docs.py` | Markdown files, agent guidance, adapter files |

`tests/lint_fixtures/` holds deliberately bad code. It is excluded from every tool and from test collection, and is linted only by `tests/unit/test_sdk_lint.py`.

## Rules

Block means the gate fails. Warn means the finding is printed and the gate passes. `--strict` turns warnings into failures.

| Code | Severity | What it finds | Why |
|---|---|---|---|
| SDK001 | Block | `import random`, `import numpy.random`, any `numpy.random` member other than `Generator` (called or only referenced, through any alias), `from numpy import random`, `getattr(np, "random")`, and SciPy `.rvs()` without `random_state=` | Randomness outside `RNGPartitioner` breaks coupling between branches and reproducibility |
| SDK002 | Block | `assert True`, `assert "text"`, `assert (cond, "msg")`, `assert x == x` | These assertions can never fail |
| SDK003 | Block | A bare, `Exception` or `BaseException` handler that does not re-raise, around a call to `finalize_experiment`, `save_experiment`, `register_experiment`, `register_sweep` or `write_provenance` | A run whose results were not saved must not report success |
| SDK003 | Warn | Any other broad handler that does not re-raise (not reported in tests) | Narrow the exception, or record the failure and suppress with a reason |
| SDK004 | Warn | A test with no `assert`, no `pytest.raises`/`warns`, and no call to an `assert*`, `check*`, `verify*` or `testing` helper | It passes as long as nothing raises |
| SDK005 | Warn | A test whose every assertion is inside an `if` with no `else` | If the condition is never true, the test checks nothing |
| SDK006 | Warn | A `BaseScenario` subclass whose `create_population`, `step`, `predict`, `intervene` or `measure` is only `pass`, `...`, a docstring, or `raise NotImplementedError` | An unfinished scenario method |
| SDK000 | Block | A suppression comment without a reason, a malformed one, or one that suppresses nothing | Suppressions must stay explained and current |
| PT011 | Block (Ruff) | `pytest.raises(ValueError)` without `match=` | The test passes even when a different error was raised |
| PT012 | Block (Ruff) | More than one statement inside `pytest.raises` | Only the statement that should raise belongs inside |
| B017 | Block (Ruff) | `pytest.raises(Exception)` | Catches everything |
| PT015, B011 | Block (Ruff) | `assert False` | Use `pytest.fail()`; `assert` is removed under `python -O` |
| PGH004 | Block (Ruff) | Blanket `# noqa` without codes | Hides every future finding on that line |
| RUF100 | Block (Ruff) | `# noqa` for a Ruff rule that no longer fires | Stale suppression. Flake8's own codes are declared `external`, so their `noqa` comments are left to Flake8 |

`scripts/check_no_bare_random.py` (the pre-commit hook) runs SDK001 only.

Approved SDK001 locations: `healthcare_sim_sdk/core/rng.py` (the partitioner), `scripts/validate_its_estimator.py` (statistical Monte Carlo on the estimator; no scenario or engine), and `tests/` (test fixtures construct their own generators).

## Documentation checks

`scripts/check_docs.py` keeps the guidance that agents and people rely on from going stale. It uses the standard library only, reads files and nothing else, and prints sorted `path:line: CODE message` lines. It checks tracked files plus new files not yet added; ignored files are left out. Every finding blocks.

| Code | What it finds | Why |
|---|---|---|
| DOC001 | A relative markdown link whose target does not exist | A dead link sends the reader nowhere |
| DOC002 | A repo path in backticks that does not exist, in a document classed `current` | A current rule must not point at something that is gone. Records are exempt |
| DOC003 | `CLAUDE.md` that is not a symlink to `AGENTS.md` (or the one-line import `@AGENTS.md`); any dangling symlink | A copy drifts from its source |
| DOC004 | A scenario folder without an `AGENTS.md`, or one missing its status line or a standard heading | Local guidance is part of a scenario |
| DOC005 | A runner module, test path or file named in a scenario `AGENTS.md` that does not exist | Commands in guidance must be real |
| DOC006 | A markdown file with no class in [`AUTHORITY.md`](AUTHORITY.md); a row that matches no file, repeats, or has an unknown class | Every document is a current rule, rationale, a proposal or a record |
| DOC007 | Root `AGENTS.md` longer than 80 lines | It loads in every session |
| DOC008 | A file in `.claude/agents/` without `name` and `description` frontmatter; a file in `.claude/rules/` without `paths` | Without them the tool does not register the adapter, or loads it everywhere |
| DOC009 | A retired path named outside records and compatibility notes | Old locations must not creep back |

`ENABLED` at the top of the script lists the rules that run by default. Run others with `--codes`.

Suppress DOC001, DOC002, DOC005 or DOC009 on one line with a reasoned comment on that line or the line above:

```markdown
The class lived in `scripts/old_tool.py`. <!-- docs-check: allow DOC002 -- removed in 0.4 -->
```

A suppression without a reason is ignored. Tests build their broken trees at run time (`tests/unit/test_check_docs.py`), so no dead link or dangling symlink is tracked.

## Suppressing a finding

Put the comment on the reported line or the line directly above it, and give a reason after `--`:

```python
except Exception as exc:  # sdk-lint: allow SDK003 -- recorded as a FAIL check
```

A suppression without a reason, or one that no longer matches a finding, is itself a blocking SDK000 finding. Use Flake8 and Ruff codes with `# noqa: CODE`; a bare `# noqa` is rejected.

## Test profiles

Some tests need institution-private files that are never in the repository. They carry the `private` marker (`tests/private_profile.py`).

| Profile | Command | Private tests |
|---|---|---|
| Public (CI and the gate) | `pytest -m "not private"` | Deselected and reported as deselected |
| Private | `SIM_SDK_PRIVATE_TESTS=1 pytest -m private` | Run. A missing private file fails the run |
| Plain `pytest` | `pytest` | Skipped when files are absent (previous behavior) |

## The quality gate

`scripts/quality_gate.sh` is the one entry point. CI runs it, and so can anyone locally.

1. **Explicit inputs.** It exports a clean snapshot of a git ref (default `HEAD`) with `git archive`. Untracked and ignored files, including private configs, never enter the snapshot. `--worktree` uses tracked files from the working tree instead, for checking uncommitted work.
2. **Locked environment.** Python 3.12 (`.python-version`) and `requirements/lock.txt`, installed with `uv pip sync --require-hashes`. The same lock constrains build dependencies (`--build-constraints`), so the one source-only package (`antlr4-python3-runtime`) builds with locked tools.
3. **Network off.** Every later step runs with network access removed, when the platform allows it (Linux with `unshare` and root or sudo). Loopback stays up for the notebook kernels. Where isolation is unavailable, the gate prints `network isolation: NOT ENFORCED`; `--require-offline` makes that a failure.
4. **Checks.** Invariants, SDK001 over every tracked file, Flake8, Ruff, sdk-lint, docs-check, the public test profile, and the example notebooks.
5. **Reproducibility.**
   - Lint output from two separate copies of the snapshot must be byte-identical.
   - The wheel is built twice, in two separate copies, with `SOURCE_DATE_EPOCH` set to the commit time. Both wheels must have the same SHA-256.
   - The wheel's YAML files must equal the tracked YAML files (minus the private-campus folder), and it must contain no private file.
6. **Wheel smoke test.** The wheel is installed into a fresh environment from the local cache, with no network. From a directory outside the checkout, the test imports the package, loads a Hydra config shipped in the wheel, runs a small branched simulation, and imports the ITS module.

The gate writes `outputs/quality_gate/summary.txt` with every step's status, the wheel SHA-256 and the lock file's SHA-256.

### Updating the lock

```bash
uv pip compile pyproject.toml --extra dev --extra lint --extra build \
  --universal --python-version 3.12 --generate-hashes --no-header \
  -o requirements/lock.txt
```

Review the diff, then run the gate.
