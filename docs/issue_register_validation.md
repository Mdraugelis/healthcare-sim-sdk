# ADD issues register: validation and reproduction

Reviewed 16 October 2026. The ADD retains its historical baseline investigations;
its resolution column and the revised reports describe the corrected code.

The five-method scenario interface and five RNG streams are unchanged. Core
changes are limited to the single-use lifecycle guard and analysis export fixes.
Scenario objects must be constructed anew for each run. Stroke's unreachable
legacy classification defaults now raise; an exploratory infeasibility override
must be explicit and emits warnings with the original requirements retained.

## Completed rerun on the fixed revision

All seven report sweeps were restarted from the clean fixed revision
[`88be0ae`](https://github.com/Mdraugelis/healthcare-sim-sdk/commit/88be0aee41df8edf93ef647b5883f93be475484b),
after I-01–I-29 were implemented. They ran on 16 October 2026 from 22:16:39
to 22:20:21 UTC. The revision and clean working tree were checked before and
after the sweeps. No algorithm changes were made during or after these runs.
The later report-builder change only selects the input folder and records provenance.

| Sweep | Completed and registered cells |
|---|---:|
| Six monitoring regimes, 1,000 nurses | 180 |
| Success/null comparison, 6,800 nurses | 60 |
| Independent null calibration | 100 |
| Fresh null holdouts | 30 |
| Retention AUC × capacity × seed | 360 |
| Two reminder study configurations | 60 |
| TREWS-style delay × capacity | 28 |
| **Total** | **818** |

The four report tables were rebuilt from these outputs and reproduce the
previous revised aggregates. Raw monitoring alarm metadata now also uses the
final configured h=96; it is no longer inherited from provisional chart settings.
The 100 calibration seeds remain separate from the 30 holdout seeds. The
[provenance record](issue_register_rerun_provenance.json) saves exact
commands, revision/tree, environment versions, timestamps, cell counts, and
artifact/report hashes. Raw results stay gitignored under
`outputs/postfix_88be0ae_complete/`.

Parallel workers use `hydra.job.chdir=true`, keeping their catalogs inside their
own output directories. After workers exit, each saved config receives a unique
run key, completion time and source revision, and each sweep is registered serially
into its own catalog. Resolved simulation inputs remain in `.hydra/config.yaml`.
An interrupted attempt using a shared catalog is excluded from every report.

### Verification summary

| Check | Result |
|---|---|
| Complete, unique parameter/seed grids and registered cells | 818/818 |
| Finite saved numeric metrics and monitoring metadata | All cells |
| Weekly population conservation and retention identities | 38,480 records across 370 monitoring cells |
| Exact factual/no-AI departure equality in null programs | 190/190 null cells |
| Retention event/count accounting and sufficient reference events | 360/360 cells |
| Reminder appointment-level non-worsening assertion | 60/60 cells |
| TREWS mortality/count accounting | 28/28 cells |
| Full test suite | 626 passed; 11 private-artifact skips |
| Existing notebooks | 14/14 executed |
| Epic ESM structural/statistical/null checks | 6/6 |
| Lint and architectural/RNG hooks | Passed |

A separate 104-week, 1,000-nurse seed-1000 trajectory replay passed at least 2,396
assertions covering entity IDs/shapes, absorbing departure accounting, finite
risks/scores, score bounds, per-manager AI capacity, reference events and model
quality. Total observed departures differed from the sum of conditional expected
departures by −0.73 standard deviations on the factual branch and −0.34 on the
no-AI branch, within the four-sigma sanity bound. Existing regression tests cover
zero-effect boundaries and branch coupling; the Epic rerun also checks its null
and random-model boundaries.

**Flagged, not silently passed:** 25 of 90 distinct retention AUC–seed reference
fits miss the 0.02 delivered-AUC tolerance (largest gap 0.052). Each fit is reused
in four capacity cells. Warnings remain visible, and the retention report now
states this spread alongside mean delivered AUC. The monitoring report continues
to disclose Tier 3's weak endpoint detection and Tier 4's alarms as cohort
calibration changes. These are limitations of the simulated evaluation design.

### Four representative nurse trajectories

Four nurses were selected without replacement using a separate audit population
stream, seed 314159. Weeks below are zero-based. The audit checked every weekly
observation for these nurses; the narratives show the decisive points.

- Nurse 635 began with annual baseline risk 0.516 and 147 weeks of tenure. Risk
  drift put week-0 risk at 0.499. The nurse received model scores but no AI check-in
  before departing in week 4 on both branches. Their paired histories agree.
- Nurse 695 began with risk 0.349 and 154 weeks of tenure. The first AI check-in
  was in week 34, with score 0.0240 for the two-week prediction horizon. The
  post-check-in annual risk was 0.169, versus 0.334 on the no-AI branch. Five AI
  check-ins occurred; the nurse remained employed on both branches at week 103.
  Later SOC allocations can differ, so branch risks need not stay ordered.
- Nurse 26 began with risk 0.387 and 187 weeks of tenure. The first AI check-in
  occurred in week 30 at score 0.0294, with post-check-in annual risk 0.199 versus
  0.220 without AI. Four AI check-ins occurred. The nurse nevertheless departed
  in week 102 on both branches: a lower departure probability does not guarantee
  prevention for an individual nurse.
- Nurse 177 began with risk 0.162 and 77 weeks of tenure. Week-0 risk was 0.147.
  The nurse received scores but was never selected for an AI check-in, and
  remained employed on both branches through week 103. Both branches ended at
  annual risk 0.095. No trajectory/accounting anomaly was found.

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
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  regime=calibrated_success,null_program,gradual_decay,capacity_collapse,model_drift,partial_adoption \
  seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_monitoring_verified

# Direct population-size check; independent control site at each size.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  regime=calibrated_success,null_program population.n_nurses=6800 \
  seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_monitoring_6800_verified

# Calibrate h on 100 independent null runs. The fixed rule is the smallest
# candidate whose Wilson 95% upper alarm-probability bound is <=10% over 104 weeks.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  regime=null_program seed='range(9000,9100)' \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_calibration_100

# Fresh null holdouts, not used to select h.
python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  regime=null_program seed='range(2000,2030)' \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_null_holdout

# Retention quality x capacity, 30 seeds per cell (360 cells).
python -m healthcare_sim_sdk.experiments.issue_register_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  kind=retention auc=0.60,0.70,0.75 capacity=2,4,6,8 \
  seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_retention_verified

# Preserve the two reminder study configurations, without re-tuning effects.
python -m healthcare_sim_sdk.experiments.issue_register_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  kind=reminder reminder_study=chong,rosen seed='range(1000,1030)' \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_reminder

# Re-run the original 28-cell TREWS delay/capacity sensitivity grid.
python -m healthcare_sim_sdk.experiments.issue_register_validation \
  --multirun hydra/launcher=joblib hydra.launcher.n_jobs=4 hydra.job.chdir=true \
  kind=trews capacity=10,25,50,100 max_hours=6,12,18,24,36,48,72 seed=42 \
  hydra.sweep.dir=outputs/postfix_88be0ae_complete/issue_register_trews

python scripts/build_issue_register_reports.py \
  --output-root outputs/postfix_88be0ae_complete
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
