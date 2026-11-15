# healthcare-sim-sdk

Python SDK for evaluating a healthcare AI deployment before it happens. A discrete-time engine runs one population twice, with the AI and without it, on coupled random streams. The difference between the two runs is the effect of the AI.

This file holds the rules that apply everywhere in the repository. Each scenario folder and `healthcare_sim_sdk/core/` has its own `AGENTS.md`: read it before you work there. [`docs/AUTHORITY.md`](docs/AUTHORITY.md) says which documents are current rules, which are proposals, and which are records.

## Invariants

These hold for every change. `config/invariants.yaml` is the source, and `scripts/check_invariants.py` checks the code against it. Changing one is a design decision: open an issue against `main` first, and never do it inside a scenario change.

- **Five-method scenario contract.** A scenario implements `create_population`, `step`, `predict`, `intervene` and `measure`. Do not add or remove methods.
- **Step purity.** `step()` depends only on `(state, t, self.rng.temporal)`. No side effects and no shared mutable state.
- **Partitioned randomness.** Every draw comes from an `RNGPartitioner` stream: population, temporal, prediction, intervention or outcomes. Never call `np.random` or `random` directly. Both branches must take the same number of draws at every step.
- **Opaque state.** State can be any type. The engine never looks inside it.
- **No predictions on the counterfactual branch.** The question is "what if the AI had not been deployed?"

## Data boundary

This repository is public. Institution-specific values, names, rates and model files never enter tracked files or commit messages. Private files are ignored by git on purpose; do not add them. See [`docs/data_boundary.md`](docs/data_boundary.md).

## Changing code

- Run `scripts/quality_gate.sh --worktree` before opening a pull request. CI runs the same gate. The rules are in [`docs/lint_rules.md`](docs/lint_rules.md).
- Quick loop while editing: `flake8 healthcare_sim_sdk tests scripts`, `python scripts/sdk_lint.py`, `python scripts/check_docs.py`, `pytest -m "not private" tests/`.
- Tests go in `tests/unit/`, `tests/integration/` or `tests/bulletproof/`. A test must be able to fail.
- Every change reaches `main` through a pull request.
- Use NumPy vector operations, not Python loops, over a population.
- Parameter sweeps are Hydra YAML run with `--multirun`, never Python loops.
- Runners live in their scenario folder and end by calling `finalize_experiment()`. See [`docs/experiment_verification.md`](docs/experiment_verification.md).
- `outputs/` is never committed.
- Analysis exports go through `AnalysisDataset`. See [`docs/analysis_exports.md`](docs/analysis_exports.md).
- A new scenario starts as a copy of `healthcare_sim_sdk/scenarios/_template/`, including its `AGENTS.md`.

## Running or interpreting a simulation

Follow [`docs/simulation_protocol.md`](docs/simulation_protocol.md), and read it before the first run of a session. These parts are mandatory:

1. **Fit.** Check the use case against the six fitness criteria. Say so directly when the SDK is the wrong tool.
2. **Analytical first.** If point estimates of sensitivity, PPV and prevalence answer the question, do the closed-form pass and skip the simulator ([`docs/analytical_first_pass.md`](docs/analytical_first_pass.md)). Use the SDK when the answer depends on the score distribution.
3. **Verify every run before interpreting it.** No exceptions. The five levels:
   - structural integrity;
   - statistical sanity;
   - conservation laws and monotonicity;
   - three to five case walkthroughs, told as narratives;
   - boundary conditions, on the first run of a new scenario.
4. **Show the verification summary first.** Resolve any failure before interpreting results.
5. **Report honestly.** Write "the simulation suggests, under these assumptions". List the assumptions and what the simulation cannot tell. Scenarios built on published studies are calibration exercises, not validation.
6. **Never make the deployment decision.** Provide the analysis. The person decides.
