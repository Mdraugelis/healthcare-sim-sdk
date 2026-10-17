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

## Do You Actually Need the Full Simulator?

**Not every question needs Monte Carlo.** Before building a scenario, ask whether your decision question can be answered in closed form — a two-proportion power formula with an autocorrelation derate often is enough. Power, minimum detectable effect, Bayes-bounded PPV, and causal-chain feasibility (sensitivity × action_rate × efficacy) all collapse to algebra. If that's your question, **write 200 lines of Python and skip the SDK.** A spreadsheet-grade answer is better than a Monte Carlo answer you can't explain.

You need the full simulator when the decision depends on the **score distribution**, not just point estimates:

| If the question is… | Closed form works | SDK required |
|---|---|---|
| "Do we have the volume to detect a 15% reduction?" | ✅ | |
| "Is the causal chain strong enough to matter?" | ✅ | |
| "How does Group B compare to Group A given a baseline delta?" | ✅ | |
| "What threshold should we pick?" | | ✅ |
| "How does subgroup calibration shift the operating point?" | | ✅ |
| "What if alert fatigue has a 6-hour half-life?" | | ✅ |
| "Does earlier detection produce more lead time, and does that compound?" | | ✅ |
| "Will our planned ITS actually recover the effect on entity-level data?" | | ✅ |

**Quick heuristic:** if the decision rests on point estimates of sensitivity, PPV, and prevalence, closed-form suffices. If it rests on the score *distribution* — because a threshold is being chosen, a ranking is being used, a calibration is being trusted, or a subgroup is being stratified — the SDK's `ControlledMLModel` is required.

Either way, the verification protocol is the same (Bayes constraint, monotonicity, boundaries, CLT sanity, sensitivity sweep, explicit gap list). A closed-form answer is not an excuse to skip rigor. See [`docs/analytical_first_pass.md`](docs/analytical_first_pass.md) for the worked protocol and a reference example from the PeriGen EFM evaluation.

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
| **No-Show Targeted Reminders** | Appointment | Probability estimation (no-show) | Phone reminders to high-risk patients |
| **Sepsis Early Alert** | Patient-admission | Discrimination (sepsis progression) | Early treatment (antibiotics, fluids) |
| **Template (Coin Flip)** | Entity | Any | Reduce event probability |

## Core Architecture

- **5-Method Contract** -- Scenarios implement `create_population`, `step`, `predict`, `intervene`, `measure`. The SDK handles everything else.
- **Branched Counterfactual Engine** -- Parallel factual + counterfactual trajectories. Both branches share the same temporal physics; they diverge only where intervention changes state.
- **RNG Stream Partitioning** -- 5 independent streams (population, temporal, prediction, intervention, outcomes) via `SeedSequence.spawn()`. Ensures branches don't desynchronize.
- **ControlledMLModel** -- Simulates ML models from a target AUC or an operating point (PPV or sensitivity). Scores are label-free noise on true risk, Platt-calibrated, and gated against the population's Bayes ceiling. See `docs/design_principles.md`, "ML Model Simulation".
- **Experiment Infrastructure** -- Catalog, structured outputs (config/results/report/validation), markdown report generation.

## Published-Study Calibration Exercises

The SDK includes exploratory exercises based on published studies. Several used
the reported clinical result to tune effectiveness or capacity. These are
calibration exercises, not independent predictions of those same results.

The historical [30-entry audit](research/synthesis_report.md) mixes intervention,
prediction, methods and review papers. Its reproduction counts and direction-of-effect
labels should not be interpreted as an independent external-validation score.
The [paper-based validation approach](docs/paper_based_validation.md) and
[methodological alignment review](docs/validation_methodology_alignment.md)
define the evidence and study-specific checks needed for new claims.

### Chong et al. (2020): MRI No-Show Reminders

The [Chong configuration](healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/chong_replication.yaml)
uses a target AUC of 0.74 and top-25% targeting. It explicitly labels reminder
effectiveness 0.4437 as calibrated to the reported intervention no-show rate.
Historical simulation fits are preserved in the
[no-show report](reports/noshow_targeted_reminders_proof_points.md).

### Tarabichi et al. (2023): Targeted Reminders and Racial Disparities

The [primary randomized study](https://doi.org/10.1007/s11606-023-08209-0)
was conducted at MetroHealth in Cleveland. Eligible appointments had predicted
no-show risk of at least 15%; both arms received existing automated reminders.
The repository previously misattributed this study to Rosen and VA medical centers.
The [legacy configuration](healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/rosen_replication.yaml)
retains that filename and requires cohort, parameter and design reconstruction
before supporting a paper-reproduction claim. Its historical fitted results are
not a blinded validation of the randomized comparison.

### Adams et al. (2022): TREWS Sepsis Early Warning

The [TREWS study](https://doi.org/10.1038/s41591-022-01894-0) reported an adjusted
association between confirmation within three hours and mortality among its
eligible sepsis-alert cohort. That contrast differs from comparing AI deployment
with no AI across all admissions.

The [legacy configuration](healthcare_sim_sdk/scenarios/sepsis_early_alert/configs/trews_replication.yaml)
and [calibration report](reports/trews_baseline_detection_sweep.md) explore
standard care, treatment timing, alert response and capacity. Their six-hour
effectiveness half-life and high capacity are scenario assumptions, not parameters
identified by this TREWS comparison. Fitting these mechanisms to its mortality
result does not independently establish them or reproduce the paper's estimand.

---

## No-Show Overbooking Evaluation

The original reference use case: evaluate whether an ML no-show predictor improves overbooking decisions over the current practice of using patient historical rates.

**Run an evaluation:**
```bash
python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_evaluation \
    --n-days 90 --n-patients 2000 --model-auc 0.83 --threshold 0.30
```

**Run a governance evaluation (48 configs, equity audit):**
```bash
python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_governance_eval
```

Each run produces: `config.json`, `results.json`, `results.csv`, `report.md`, `validation_appendix.md`.

### Key Findings (Simulation)

| Metric | No Overbooking | Current Practice (hist >= 50%) | ML Predictor (thresh=0.30) |
|--------|---------------|-------------------------------|---------------------------|
| Utilization | 85.9% | 89.2% | **90.3%** |
| Collision rate | N/A | 43.4% | **31.6%** |
| Waitlist (day 60) | 302 | 6 | **0** |

All equity checks pass (no subgroup AUC gap > 3%, proportional flagging).

## Package Structure

```
healthcare_sim_sdk/
  core/               Engine, scenario interface, results, RNG
  ml/                 ML model simulator, performance metrics
  population/         Risk distributions, temporal dynamics (AR(1))
  scenarios/
    stroke_prevention/          Reference scenario
    noshow_overbooking/         Overbooking evaluation + runners
    noshow_targeted_reminders/  Chong & Rosen replications
    sepsis_early_alert/         TREWS replication + capacity sweep
    _template/                  Starter template
  experiments/        Catalog, report generator, validation

examples/             13 reference notebooks (read-only)
reports/              Replication reports and paper drafts
tests/                Unit, integration, and bulletproof tests

local/                Private working space — never tracked (any depth)
*.private.*           Institution-specific values — never tracked
*.example.*           Sanitized templates — tracked deliberately
```

## Data Boundary

The SDK is open-sourceable; the work done *with* it often is not. Campus
rates, benchmark figures under confidentiality, staff names, and raw
institutional actuals must never enter tracked files or git history. The
repo is **default-private** for this material: it stays out of version
control unless someone explicitly promotes a sanitized copy.

Three simple rules — and a backstop so you don't have to remember them
perfectly:

1. **Put confidential work in a private location.** Anything under a
   `local/` directory (any depth) or named `*.private.*` is gitignored
   automatically — so a brand-new experiment folder is covered with no new
   rules.
2. **Promote by sanitizing.** To share an experiment's *structure*, copy
   its config to a `*.example.*` file, swap real values for placeholders,
   and commit that. Promotion is deliberate.
3. **The commit-time guard catches the rest.** A pre-commit hook scans
   staged content and commit messages against a local denylist of
   confidential terms and blocks the commit on a match — even if an ignore
   rule was missing.

If you handle confidential data, set the guard up once per clone:

```bash
cp .institution-denylist.example .institution-denylist   # then edit it
pre-commit install --hook-type commit-msg                # + message scanning
```

Contributors with no confidential data can ignore all of this — the guard
stays silent without a local denylist. Full details, including the
values-vs-structure rationale and what the boundary does *not* cover, are
in [`docs/data_boundary.md`](docs/data_boundary.md).

## Development

```bash
git clone https://github.com/Mdraugelis/healthcare-sim-sdk.git
cd healthcare-sim-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest tests/                    # 266 tests
flake8 healthcare_sim_sdk/       # Lint
```

## Design Principles

See [`docs/design_principles.md`](docs/design_principles.md) for the design principles document — why the SDK is built the way it is, what it enables, what it deliberately excludes, and the lessons learned validating it against 30 published healthcare AI deployment papers.

See [`docs/paper_based_validation.md`](docs/paper_based_validation.md) for the proposed approach to sourced parameters, explicit assumptions, blinded paper-outcome validation, and quasi-experimental tests against hidden simulator effects.

See [`docs/validation_methodology_alignment.md`](docs/validation_methodology_alignment.md) for foundational citations, the current implementation assessment, and requirements for future validation claims.

## Contributing

Contributions are welcome — bug reports, new scenarios, documentation, tests. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, invariant rules, and PR checklist.

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [`NOTICE`](NOTICE) for attribution.

Copyright 2026 Michael Draugelis.
