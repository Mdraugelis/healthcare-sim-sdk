# Healthcare Intervention Simulation SDK

When a health system deploys a predictive model into clinical operations, we never observe the counterfactual — what would have happened without the AI. Did the intervention actually reduce strokes, or were we going to see that drop anyway? Did the overbooking algorithm improve access, or just shift the burden onto certain patients?

This SDK provides a controlled environment where the ground truth is known, so that policy makers, clinical leaders, researchers, and engineers can explore the boundaries of the problem space and solution space *before* committing real resources and affecting real patients.

## The Questions This Helps You Answer

**How good does the model need to be?** Simulate ML models at any performance level — AUC of 0.70 vs. 0.85, different PPV/sensitivity tradeoffs — and see what actually changes in patient outcomes. Find the minimum viable model performance for your use case before you build or buy anything.

**How many resources should we apply?** Vary staffing levels, alert thresholds, overbooking caps, and intervention capacity. Discover whether adding a nurse navigator moves the needle more than improving the model, or whether doubling telepharm staff matters more than a better algorithm.

**How should we set up real-world evaluation?** The SDK exports analysis-ready datasets for Interrupted Time Series, Difference-in-Differences, Regression Discontinuity, and panel methods. Test these methods against known ground truth in simulation so you can design your quasi-experimental rollout with confidence that your analytic approach will detect the effect if it's there.

**Will this be equitable?** Run governance evaluations with demographic subgroup analysis, burden distribution audits, and fairness checks — before a single real patient is affected. Produce the evidence an IRB or ethics committee needs to see.

**Should we buy this vendor's tool?** When a vendor claims "our AUC is 0.83," simulate that performance in your operational context — your population mix, your staffing, your workflow. See whether their claimed performance actually moves outcomes in your setting.

**Is the program actually working?** Rehearse your monitoring and evaluation plan on synthetic data. If your analysis can't recover the known effect in simulation, it won't find a real one either.

## Who This Is For

- **Clinical leaders** deciding whether to greenlight an AI pilot and how to staff it
- **Policy makers** evaluating intervention strategies and resource tradeoffs
- **Researchers** designing quasi-experimental evaluations and validating causal inference methods
- **Data scientists** testing model performance requirements and calibration
- **Governance teams** producing equity audits and risk assessments before deployment
- **Procurement teams** stress-testing vendor claims against operational reality

## Agent-Assisted Workflow

This SDK is designed for use with an AI assistant (Claude Code, Claude Cowork, or similar) that works *with* the human throughout the simulation process. The agent guides scenario development, runs verification after every simulation — including case-level walkthroughs and conservation law checks — and helps translate findings into language appropriate for clinical leaders, governance committees, or procurement conversations.

The agent protocol lives in [`.claude/agents/sim-guide.md`](.claude/agents/sim-guide.md). It covers use-case fitness screening, the always-on verification checklist, and stakeholder communication guidance. Any Claude Code or Cowork session that opens this repo picks it up automatically.

## Install

```bash
pip install git+https://github.com/Mdraugelis/healthcare-sim-sdk.git
```

## Quick Start

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

## What You Can Explore

The SDK gives you levers across four dimensions:

**Model performance** — Simulate ML models with controlled AUC, PPV, sensitivity, and calibration. Compare an ML predictor against a simple baseline (e.g., historical no-show rate) against no model at all. The `ControlledMLModel` uses 4-component noise injection with prevalence-aware feasibility checking, so you can't accidentally simulate an impossible model.

**Operational configuration** — Number of providers, slots per provider, overbooking caps, alert thresholds, intervention policies (threshold-based vs. priority-based), demand arrival rates. These are the real-world knobs your operations team controls.

**Population dynamics** — Base event rates, population heterogeneity (via beta distribution concentration), temporal drift (AR(1) with seasonal effects), and demographic-specific multipliers for race/ethnicity, insurance type, age, and visit pattern. Stress-test your intervention against realistic population variation.

**Evaluation design** — Run the same scenario under different analytic methods and check which ones recover the true causal effect. Export time-series data for ITS, panel data for DiD, cross-sectional snapshots for RDD, and subgroup panels for equity analysis.

## Reference Scenarios

| Scenario | Entity | ML Task | Intervention |
|---|---|---|---|
| **Stroke Prevention** | Patient | Binary classification (stroke risk) | Anticoagulants (risk reduction) |
| **No-Show Overbooking** | Appointment | Probability estimation (no-show) | Double-book high-risk slots |
| **Template (Coin Flip)** | Entity | Any | Reduce event probability |

## Core Architecture

- **5-Method Contract** -- Scenarios implement `create_population`, `step`, `predict`, `intervene`, `measure`. The SDK handles everything else.
- **Branched Counterfactual Engine** -- Parallel factual + counterfactual trajectories. Both branches share the same temporal physics; they diverge only where intervention changes state.
- **RNG Stream Partitioning** -- 5 independent streams (population, temporal, prediction, intervention, outcomes) via `SeedSequence.spawn()`. Ensures branches don't desynchronize.
- **ControlledMLModel** -- Simulates ML models with target AUC, PPV, and sensitivity. 4-component noise injection with prevalence-aware feasibility checking.
- **Experiment Infrastructure** -- Catalog, structured outputs (config/results/report/validation), markdown report generation.

## No-Show Overbooking Evaluation

The primary reference use case: evaluate whether an ML no-show predictor improves overbooking decisions over the current practice of using patient historical rates.

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

### Key Findings (Simulation)

| Metric | No Overbooking | Current Practice (hist >= 50%) | ML Predictor (thresh=0.30) |
|--------|---------------|-------------------------------|---------------------------|
| Utilization | 85.9% | 89.2% | **90.3%** |
| Collision rate | N/A | 43.4% | **31.6%** |
| Waitlist (day 60) | 302 | 6 | **0** |

Governance: 8/9 criteria met. All equity checks pass (no subgroup AUC gap > 3%, proportional flagging).

See `experiments/outputs/comprehensive_evaluation_report.md` for the full analysis.

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
