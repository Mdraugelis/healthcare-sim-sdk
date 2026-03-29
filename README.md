# Healthcare Intervention Simulation SDK

A toolkit for healthcare AI deployment evaluation. Test intervention policies, measure causal effects, compare analytic methods against known ground truth, and run governance evaluations — all with simulated populations where the answer is known.

## Install

```bash
pip install git+https://github.com/Mdraugelis/healthcare-sim-sdk.git
```

## What It Does

You define a healthcare scenario by implementing 5 methods. The SDK handles the simulation engine, counterfactual branching, RNG partitioning, and analysis exports.

```python
from healthcare_sim_sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.scenarios.noshow_overbooking.realistic_scenario import (
    RealisticNoShowScenario, ClinicConfig,
)

# Configure a clinic
cc = ClinicConfig(n_providers=6, slots_per_provider_per_day=12)
tc = TimeConfig(n_timesteps=90, timestep_duration=1/365,
                timestep_unit="day", prediction_schedule=list(range(90)))

# Run simulation with ML predictor
scenario = RealisticNoShowScenario(
    time_config=tc, seed=42, n_patients=2000,
    model_type="predictor", model_auc=0.83,
    overbooking_threshold=0.30, clinic_config=cc,
)
engine = BranchedSimulationEngine(scenario, CounterfactualMode.BRANCHED)
results = engine.run(2000)

# Export for analysis
analysis = results.to_analysis()
ts = analysis.to_time_series()           # For ITS
panel = analysis.to_panel()              # For DiD
equity = analysis.to_subgroup_panel()    # For equity analysis
```

## Reference Scenarios

| Scenario | Entity | ML Task | Intervention |
|---|---|---|---|
| **Stroke Prevention** | Patient | Binary classification (stroke risk) | Anticoagulants (risk reduction) |
| **No-Show Overbooking** | Appointment | Probability estimation (no-show) | Double-book high-risk slots |
| **Template (Coin Flip)** | Entity | Any | Reduce event probability |

## Core Concepts

- **5-Method Contract** -- Scenarios implement `create_population`, `step`, `predict`, `intervene`, `measure`. The SDK handles everything else.
- **Branched Counterfactual Engine** -- Parallel factual + counterfactual trajectories. Both branches share the same temporal physics; they diverge only where intervention changes state.
- **RNG Stream Partitioning** -- 5 independent streams (population, temporal, prediction, intervention, outcomes) via `SeedSequence.spawn()`. Ensures branches don't desynchronize.
- **ControlledMLModel** -- Simulates ML models with target AUC, PPV, and sensitivity. 4-component noise injection with prevalence-aware feasibility checking.
- **Experiment Infrastructure** -- Catalog, structured outputs (config/results/report/validation), markdown report generation.

## Package Structure

```
healthcare_sim_sdk/
  core/               Engine, scenario interface, results, RNG
  ml/                 ML model simulator, performance metrics
  population/         Risk distributions, temporal dynamics (AR(1))
  scenarios/
    stroke_prevention/      Reference scenario 1
    noshow_overbooking/     Reference scenario 2 + evaluation runners
    _template/              Starter template
  experiments/        Catalog, report generator, validation

examples/             13 reference notebooks (read-only)
tests/                266 tests (unit, integration, bulletproof)
experiments/outputs/  Archived experiment results and reports
```

## No-Show Overbooking Evaluation

The primary use case: evaluate whether an ML no-show predictor improves overbooking decisions over the current practice of using patient historical rates.

**Run an evaluation:**
```bash
python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_evaluation \
    --n-days 90 --n-patients 2000 --model-auc 0.83 --threshold 0.30
```

**Run a governance evaluation (48 configs, equity audit):**
```bash
python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_governance_eval
```

**Run a 365-day burden analysis for bioethics:**
```bash
python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_burden_analysis \
    --n-days 365 --cap 10
```

Each run produces: `config.json`, `results.json`, `results.csv`, `report.md`, `validation_appendix.md`.

## Key Findings (Simulation)

| Metric | No Overbooking | Current Practice (hist >= 50%) | ML Predictor (thresh=0.30) |
|--------|---------------|-------------------------------|---------------------------|
| Utilization | 85.9% | 89.2% | **90.3%** |
| Collision rate | N/A | 43.4% | **31.6%** |
| Waitlist (day 60) | 302 | 6 | **0** |

Governance: 8/9 criteria met. All equity checks pass (no subgroup AUC gap > 3%, proportional flagging).

See `experiments/outputs/comprehensive_evaluation_report.md` for the full analysis.

## Development

```bash
git clone https://github.com/Mdraugelis/healthcare-sim-sdk.git
cd healthcare-sim-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/                    # 266 tests
flake8 healthcare_sim_sdk/       # Lint
```

## Design Document

See [`docs/healthcare-sim-sdk-design-v2_1.md`](docs/healthcare-sim-sdk-design-v2_1.md) for the full architecture specification.
