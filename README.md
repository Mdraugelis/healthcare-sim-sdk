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
| **No-Show Targeted Reminders** | Appointment | Probability estimation (no-show) | Phone reminders to high-risk patients |
| **Sepsis Early Alert** | Patient-admission | Discrimination (sepsis progression) | Early treatment (antibiotics, fluids) |
| **Template (Coin Flip)** | Entity | Any | Reduce event probability |

## Core Architecture

- **5-Method Contract** -- Scenarios implement `create_population`, `step`, `predict`, `intervene`, `measure`. The SDK handles everything else.
- **Branched Counterfactual Engine** -- Parallel factual + counterfactual trajectories. Both branches share the same temporal physics; they diverge only where intervention changes state.
- **RNG Stream Partitioning** -- 5 independent streams (population, temporal, prediction, intervention, outcomes) via `SeedSequence.spawn()`. Ensures branches don't desynchronize.
- **ControlledMLModel** -- Simulates ML models with target AUC, PPV, and sensitivity. 4-component noise injection with prevalence-aware feasibility checking.
- **Experiment Infrastructure** -- Catalog, structured outputs (config/results/report/validation), markdown report generation.

## Validated Replications

The SDK has been validated against published studies. Each replication calibrates the simulation to match a paper's setting, then checks whether key findings are reproduced within tolerance.

A systematic 30-paper audit (April 2026) found **3 fully REPRODUCED** papers (SHIELD-RT, TREWS, Chong/Rosen), **5 PARTIALLY_REPRODUCED**, and **0 NOT_REPRODUCED** (the SDK never contradicted a paper's direction of effect). Seven additional scenarios are available under `healthcare_sim_sdk/scenarios/paperNN_*/`. Full audit results: [`research/synthesis_report.md`](research/synthesis_report.md).

### 1. Chong et al. (2020) -- ML-Targeted MRI No-Show Reminders

**Paper:** Chong LR et al., *American Journal of Roentgenology*, 2020. XGBoost model (AUC 0.74) predicted MRI appointment no-shows at a single center in Singapore. Top 25% of patients by predicted risk received phone reminders, reducing no-show rates from 19.3% to 15.9%.

**Replication config:** [`scenarios/noshow_targeted_reminders/configs/chong_replication.yaml`](healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/chong_replication.yaml)

| Metric | Published | Simulated | Status |
|--------|-----------|-----------|--------|
| Baseline no-show rate | 19.3% | 19.7% | PASS |
| Intervention no-show rate | 15.9% | 15.7% | PASS |
| Absolute reduction | 3.4pp | 3.9pp | PASS |
| Model AUC | 0.74 | 0.733 | PASS |

Over 90 days: 2,160 phone calls made, 340 no-shows averted, ~1 additional patient seen per provider per week.

### 2. Rosen et al. (2023) -- ML Reminders Reduce Racial Disparities

**Paper:** Rosen et al., *JGIM*, 2023. RCT at VA Medical Centers testing ML-driven phone reminders for primary care. Overall no-show reduction from 36% to 33%, with a critical equity finding: Black patients experienced a larger reduction (6pp) than White patients (2pp), narrowing the racial no-show gap.

**Replication config:** [`scenarios/noshow_targeted_reminders/configs/rosen_replication.yaml`](healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/rosen_replication.yaml)

| Metric | Published | Simulated | Status |
|--------|-----------|-----------|--------|
| Control no-show rate | 36% | 35.9% | PASS |
| Intervention no-show rate | 33% | 32.0% | PASS |
| Black absolute reduction | 6pp | 5.3pp | PASS |
| White absolute reduction | ~2pp | 3.2pp | PASS |
| Disparity reduction | Significant | 2.1pp narrowing | PASS |
| Black called more often | Yes | 20% vs 15% | PASS |

The equity mechanism: ML correctly identifies higher-risk patients (who are disproportionately from underserved populations), targets them for intervention, and the multiplicative effectiveness produces larger absolute reductions for higher-probability patients -- narrowing the disparity gap.

### 3. Adams et al. (2022) -- TREWS Sepsis Early Warning

**Paper:** Adams R et al., *Nature Medicine*, 2022. Prospective multi-site study of the Targeted Real-time Early Warning System (TREWS). AUC ~0.82 with 89% clinician engagement, demonstrating ~3.3pp adjusted mortality reduction for sepsis patients through targeted, high-confidence alerting.

**Replication config:** [`scenarios/sepsis_early_alert/configs/trews_replication.yaml`](healthcare_sim_sdk/scenarios/sepsis_early_alert/configs/trews_replication.yaml)

The sepsis scenario models a 6-stage disease progression (at-risk, sepsis, severe, shock, deceased, discharged), AR(1) risk dynamics, alert fatigue, and capacity constraints. Three mechanisms are critical to reproducing the TREWS finding:

- **Baseline clinical detection** -- Standard-of-care sepsis detection on both branches. Clinicians catch sepsis through routine care after a patient-specific delay drawn from a Beta distribution (mean ~6 hours, matching CMS SEP-1 compliance). The ML system only gets credit for improvement *over* standard care.
- **Kumar time-dependent treatment effectiveness** -- Treatment effectiveness decays exponentially after sepsis onset (halves every 6 hours, per Kumar et al. 2006). Earlier detection translates to higher per-patient effectiveness.
- **Decentralized capacity model** -- Adams et al. reported no dedicated staff for alert review; TREWS was a decentralized bedside system where providers evaluated alerts in Epic. The simulation models this with high capacity (200 per 4hr block) and a 60% timely confirmation rate (matching the paper's 61% of target sepsis patients confirmed within 3 hours).

| Metric | Published | Simulated (30 seeds) | Status |
|--------|-----------|---------------------|--------|
| Mortality reduction (septic patients) | 3.3pp (adjusted) | 4.27pp mean (95% CI: 3.30-5.56) | PASS |
| CF mortality rate (septic) | 19.2% (unadjusted comparison arm) | 15.76% +/- 1.37% | Consistent |
| Factual mortality rate (septic) | 14.6% (unadjusted study arm) | 11.49% +/- 1.25% | Consistent |

The published 3.3pp falls within the simulation's 95% confidence interval. All 30 replications produced a reduction above 3pp (range: 3.13-6.24pp).

**How the timing advantage works:** The ML model's stochastic alert firing creates a detection timing race against baseline clinical detection. At the calibrated capacity, 28% of septic patients receive ML treatment before baseline detection fires, and 241 patients are treated *before sepsis onset* (prophylactically flagged as high-risk while still at-risk). For these patients, Kumar decay amplifies the timing advantage: factual mean treatment effectiveness is 28.3% vs 20.1% on the counterfactual branch (8.2pp gain). The remaining 72% of septic patients receive treatment at the same time on both branches.

**Calibration insight:** Initial runs with a hard capacity cap of 10 (modeling a dedicated rapid response team) produced only 0.12pp reduction. Reading the Adams et al. paper revealed that TREWS was decentralized with no dedicated staff -- every bedside provider could evaluate alerts. Setting capacity to 200 (effectively unconstrained) and response rate to 60% (matching the timely confirmation rate) brought the simulation into alignment with the published finding. The key bottleneck was capacity, not model performance or detection timing.

Full calibration report: [`reports/trews_baseline_detection_sweep.md`](reports/trews_baseline_detection_sweep.md) | No-show proof points: [`reports/noshow_targeted_reminders_proof_points.md`](reports/noshow_targeted_reminders_proof_points.md)

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

## Design Principles

See [`docs/design_principles.md`](docs/design_principles.md) for the design principles document — why the SDK is built the way it is, what it enables, what it deliberately excludes, and the lessons learned validating it against 30 published healthcare AI deployment papers.
