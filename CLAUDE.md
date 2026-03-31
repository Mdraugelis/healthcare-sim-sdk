# Healthcare Intervention Simulation SDK

## Project Overview

Python SDK for healthcare AI deployment evaluation. Discrete-time simulation engine with pluggable scenario logic and branched counterfactual support.

## Architecture Invariants

- **5-method scenario contract**: Scenarios implement `create_population`, `step`, `predict`, `intervene`, `measure`. Do not add or remove methods from this interface.
- **Step purity**: `step()` must be a pure function of `(state, t, self.rng.temporal)`. No external side effects or shared mutable state.
- **RNG partitioning**: All randomness flows through `RNGPartitioner` streams (population, temporal, prediction, intervention, outcomes). Never use bare `np.random` calls.
- **Generic state type**: State can be any type (arrays, dataclasses, dicts). The engine never inspects state internals.
- **No predictions on counterfactual branch**: Default causal question is "what if we hadn't deployed the AI?"

## Code Quality

- Run `flake8 healthcare_sim_sdk/` before committing
- Run `pytest tests/` before committing
- Tests go in `tests/unit/`, `tests/integration/`, or `tests/bulletproof/`

## Key Directories

- `healthcare_sim_sdk/core/` -- Engine, scenario interface, results, RNG (touch with care)
- `healthcare_sim_sdk/ml/` -- ML model simulators
- `healthcare_sim_sdk/scenarios/` -- Each scenario is self-contained
- `healthcare_sim_sdk/experiments/` -- Catalog, report, validation infrastructure

## Conventions

- Use NumPy vectorized operations over Python loops for population-level computation
- **Experiment configuration uses Hydra** (YAML configs, not Python dataclasses). When building parameter sweeps, define the grid in YAML and use `@hydra.main` or `--multirun`. Never hardcode sweep parameters in Python loops.
- Analysis exports use the `AnalysisDataset` interface, not raw dataframes
- Scenario-specific runners live in their scenario directory, not in `experiments/`
- All branches require PR to merge to main
