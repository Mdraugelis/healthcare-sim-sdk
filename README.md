# Healthcare Intervention Simulation SDK

A toolkit for healthcare AI deployment evaluation that provides the **invariants** of discrete-time intervention simulation while leaving the **variants** to teams building bespoke scenarios. Test intervention policies, measure causal effects, and compare analytic methods against known ground truth.

Refactored from `pop-ml-simulator` to support materially different healthcare scenarios without "Turing-complete YAML."

## Core Concepts

- **BaseScenario** -- Teams implement exactly 5 methods: `create_population`, `step`, `predict`, `intervene`, `measure`. The SDK handles everything else.
- **BranchedSimulationEngine** -- Runs true parallel state trajectories (factual + counterfactual). Three modes: `BRANCHED` (full parallel trajectories), `SNAPSHOT` (same-step only), `NONE` (single trajectory).
- **RNGPartitioner** -- Uses `numpy.random.SeedSequence.spawn()` for statistically independent streams (population, temporal, prediction, intervention, outcomes). Ensures factual and counterfactual branches diverge only due to intervention, not RNG desynchronization.
- **AnalysisDataset** -- Structured results with unit-of-analysis metadata. Exports to time series (ITS), panel data (DiD), entity snapshots (RDD), and subgroup panels (equity analysis).

## Reference Scenarios

| Scenario | Entity | ML Task | Intervention |
|---|---|---|---|
| **Stroke Prevention** | Patient | Binary classification (stroke risk) | Prescribe anticoagulants (risk reduction) |
| **No-Show Overbooking** | Appointment slot | Probability estimation (no-show) | Double-book high-risk slots |

## Directory Structure

```
sdk/
  core/           Engine, scenario interface, results, RNG partitioning
  ml/             ML simulation utilities (binary classifier, probability model)
  population/     Risk distributions, temporal dynamics
  analysis/       ITS, panel utilities, visualization
  config/         Configuration, Hydra integration
  utils/          Logging, sparse matrix, vectorization helpers
scenarios/
  stroke_prevention/    Reference scenario 1
  noshow_overbooking/   Reference scenario 2
  _template/            Starter template for new scenarios
experiments/            Experiment runners
tests/
  unit/
  integration/
docs/
```

## Quick Start

```python
from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.rng import RNGPartitioner
from scenarios.stroke_prevention.scenario import StrokePreventionScenario

scenario = StrokePreventionScenario(seed=42, n_patients=10_000)
engine = BranchedSimulationEngine(mode=CounterfactualMode.BRANCHED)

results = engine.run(scenario, n_steps=52)

# Export for analysis
ts = results.to_time_series()       # Interrupted time series
panel = results.to_panel()          # Difference-in-differences
snapshots = results.to_entity_snapshots()  # Regression discontinuity
```

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/

# Linting and type checking
flake8 sdk/ scenarios/
mypy sdk/ scenarios/
```

## Implementation Roadmap

| Phase | Duration | Goal |
|---|---|---|
| 1. Extract Core SDK | 5 weeks | Working SDK with stroke scenario producing identical outputs to the current codebase |
| 2. Second Scenario | 4 weeks | No-show overbooking running end-to-end with branched counterfactuals |
| 3. Analysis Tooling | 3 weeks | ITS, panel exports, Hydra experiment sweeps |
| 4. Documentation | 2 weeks | Tutorial, API reference, scenario design guide |

## Design Document

See [`healthcare-sim-sdk-design-v2_1.md`](healthcare-sim-sdk-design-v2_1.md) for the full architecture specification.
