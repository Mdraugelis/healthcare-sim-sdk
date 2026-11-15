# Contributing

Thanks for your interest in contributing to the Healthcare Intervention Simulation SDK. Bug reports, scenarios, documentation, and tests are all welcome.

## Getting started

```bash
git clone https://github.com/Mdraugelis/healthcare-sim-sdk.git
cd healthcare-sim-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Before you open a PR

Run the quality gate locally. CI runs the same script:

```bash
scripts/quality_gate.sh --worktree
```

It needs `git`, `uv` and `tar`. It installs Python 3.12 and `requirements/lock.txt` (hash-verified), then runs Flake8, Ruff, sdk-lint, docs-check, the invariant checks, the public tests, the notebooks, two wheel builds that must be identical, and a smoke test of the installed wheel, with network off where the platform allows. See `docs/lint_rules.md`.

Tests that need institution-private files are marked `private` and are not part of the gate. Run them where the files exist with `SIM_SDK_PRIVATE_TESTS=1 pytest -m private`.

Pre-commit hooks enforce the architecture invariants described in `AGENTS.md` and defined in `config/invariants.yaml`:

- The 5-method scenario contract (`create_population`, `step`, `predict`, `intervene`, `measure`)
- `step()` purity and RNG partitioning
- No bare `np.random` outside `core/rng.py`

If your change requires modifying an invariant, open a discussion first — see the "Invariants" section of `AGENTS.md`.

## Pull requests

- All changes go through PR review; direct pushes to `main` are not accepted.
- Keep PRs focused. A scenario, a bug fix, or a doc improvement is a good unit.
- Include tests under `tests/unit/`, `tests/integration/`, or `tests/bulletproof/` as appropriate.
- New scenarios should live in their own directory under `healthcare_sim_sdk/scenarios/` and carry their own runners, verification and `AGENTS.md`. Start from `healthcare_sim_sdk/scenarios/_template/`; `scripts/check_docs.py` fails until the guidance file is filled in.

## Licensing of contributions

This project is licensed under the Apache License 2.0. By submitting a contribution, you agree that your contribution is licensed under the same terms (see Section 5 of the LICENSE file). You retain copyright to your contribution.

## Reporting issues

Use GitHub Issues. For bug reports, please include:

- SDK version / commit hash
- Minimal reproduction (scenario config, seed, command)
- Expected vs. actual behavior
- Verification output if relevant
