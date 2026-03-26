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

- Run `flake8 sdk/ scenarios/` before committing
- Run `mypy sdk/ scenarios/` before committing
- Run `pytest tests/` before committing
- Tests go in `tests/unit/` or `tests/integration/`

## Key Directories

- `sdk/core/` -- Engine, scenario interface, results, RNG (touch with care)
- `sdk/ml/` -- ML model simulators (binary classifier, probability model, regression)
- `scenarios/` -- Each scenario is self-contained with its own `scenario.py`, `config.yaml`, and `README.md`
- `scenarios/_template/` -- Starting point for new scenarios

## Conventions

- Use NumPy vectorized operations over Python loops for population-level computation
- Configuration uses Hydra (YAML configs in scenario directories)
- Analysis exports use the `AnalysisDataset` interface, not raw dataframes
