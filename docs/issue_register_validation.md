# ADD issues register: validation and reproduction

Reviewed 16 October 2026. The ADD retains its historical baseline investigations;
its resolution column and the revised reports describe the corrected code.

The five-method scenario interface and five RNG streams are unchanged. Core
changes are limited to the single-use lifecycle guard and analysis export fixes.
Scenario objects must be constructed anew for each run. Stroke's unreachable
legacy classification defaults now raise; an exploratory infeasibility override
must be explicit and emits warnings with the original requirements retained.

## Checks

Use Python 3.12 and install the declared development dependencies:

```sh
python -m pip install -e '.[dev]'
python scripts/check_invariants.py
git ls-files '*.py' | xargs python scripts/check_no_bare_random.py
flake8 healthcare_sim_sdk/
pytest tests/ -q
python scripts/execute_notebooks.py
python -m healthcare_sim_sdk.scenarios.paper01_epic_esm.run_evaluation
```

The contract matrix covers all 13 domain scenarios, two seeds, and 12 steps:
BRANCHED-versus-NONE identity and per-step temporal/outcomes stream positions.
It checks original overbooking appointments and schedules remain paired, and
checks never-treated stroke patients. A deliberately state-dependent draw
passes the non-interference test and fails the coupling audit, proving that the
checks detect different defects. Untreated identity is not asserted across
capacity/SOC spillover boundaries where it is not a valid causal assumption.

The regression module covers named outcomes, entity-ID treatment timing,
single-use rejection, mode-unused arguments, prevalence refitting, infeasibility
reports, sufficient reference events, paired CITS, functional AR(1) seasonality,
copy-on-write appointments, percentile routing, score-driven nudges, mature
outcome labels, logistic recalibration and CUSUM calibration. Reference cohorts
bootstrap the empirical risks and independently draw labels for an explicit
horizon, targeting at least 200 expected events. A ceiling rejection on a finite reference triggers one independent fourfold expansion (up to 100,000 entities); a confirmed failure still raises and retains the original requirement. Nurse drift refits preserve the
initial untreated reference rather than fitting again on selected survivors.

The complete test suite retains 11 skips for tests requiring institution-local,
gitignored calibrated malnutrition artifacts. Structural malnutrition tests use
explicit public synthetic parameters and make no calibration-fidelity claim.

All 14 existing notebooks execute. The retired no-show evaluator notebook now
uses the supported scenario API and exports operational outcomes. Executed copies
are saved under gitignored `outputs/notebook_validation/`. The additional teaching
notebooks proposed in ADD Section 15 remain future work, outside I-26's repair of
existing notebooks.

## Report sweeps

Use the module entrypoints from the repository root. These grids are Hydra
configuration overrides, not hardcoded Python sweep loops. Each cell saves its
config and metrics through `finalize_experiment()`. Raw outputs are ephemeral and
must not be committed. The report builder refuses incomplete grids.

```sh
# Six monitoring regimes, 30 seeds each, 1,000 nurses.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  regime=calibrated_success,null_program,gradual_decay,capacity_collapse,model_drift,partial_adoption \
  seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/issue_register_monitoring_verified

# Direct population-size check; independent control site at each size.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  regime=calibrated_success,null_program population.n_nurses=6800 \
  seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/issue_register_monitoring_6800_verified

# Calibrate h on 100 independent null runs. The fixed rule is the smallest
# candidate whose Wilson 95% upper alarm-probability bound is <=10% over 104 weeks.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  regime=null_program seed='range(9000,9100)' \
  hydra.sweep.dir=outputs/issue_register_calibration_100

# Fresh null holdouts, not used to select h.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  regime=null_program seed='range(2000,2030)' \
  hydra.sweep.dir=outputs/issue_register_null_holdout

# Retention quality x capacity, 30 seeds per cell (360 cells).
python -m healthcare_sim_sdk.experiments.issue_register_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  kind=retention auc=0.60,0.70,0.75 capacity=2,4,6,8 \
  seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/issue_register_retention_verified

# Preserve the two reminder study configurations, without re-tuning effects.
python -m healthcare_sim_sdk.experiments.issue_register_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  kind=reminder reminder_study=chong,rosen seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/issue_register_reminder

# Re-run the original 28-cell TREWS delay/capacity sensitivity grid.
python -m healthcare_sim_sdk.experiments.issue_register_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 \
  kind=trews capacity=10,25,50,100 max_hours=6,12,18,24,36,48,72 seed=42 \
  hydra.sweep.dir=outputs/issue_register_trews

python scripts/build_issue_register_reports.py
```

Chart calibration settings are in
`healthcare_sim_sdk/experiments/configs/cusum_calibration.yaml`. The report builder
replays the saved prospective residual series at the selected h, so no simulation
or model outcome is changed by calibrating the chart threshold. The initial
20-run pilot and its first holdouts were exploratory; the expanded 100-run
calibration and fresh 2000–2029 holdouts establish the reported check.

Tier 3 reports distinguish any-alarm fractions over repeated looks from a final
endpoint test. Neither is silently equated with cumulative departures prevented.
Tier 4 uses the same scores and common active risk set for its separate factual
versus oracle AUC comparison; only mature factual outcomes drive alarms.

## Interpretation limits

A fixed training-reference AUC does not guarantee that AUC in a selected,
post-intervention population stays constant. A closed cohort's raw turnover is
not stationary. A model-performance alarm may reflect the intervention's effect
on labels. Independent control sites introduce baseline heterogeneity; a noisy
or biased endpoint test does not become valid merely because an oracle effect is
known. The revised reports state these limits and withdraw their old conclusions
where necessary. All numerical conclusions are conditional on simulation inputs.
