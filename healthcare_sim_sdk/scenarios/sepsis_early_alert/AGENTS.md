# sepsis_early_alert

**Status:** operational

## What it models
An inpatient sepsis early-warning alert. Patients move through at-risk, sepsis, severe and shock. An alert leads to clinician review and early treatment, limited by alert fatigue and rapid-response capacity. The unit is a patient admission and a timestep is 4 hours.

## Entry points
- Base parameters: `config.yaml`. Study configs: `configs/trews_replication.yaml`, `configs/esm_replication.yaml`.
- `python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_replication`: runs a replication config.
- `python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_trews_replication`: the TREWS exercise.
- `python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_capacity_sweep`: sweeps rapid-response capacity to find where two models of different quality separate.
- `python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_baseline_sweep`: sweeps the usual-care detection delay used to calibrate the TREWS exercise.
- `python -m healthcare_sim_sdk.scenarios.sepsis_early_alert.run_timing_diagnostic`: measures model detection time against usual-care detection.

## Verify
- `pytest tests/integration/test_sepsis_alert.py tests/bulletproof/test_sepsis_conservation.py`
- `pytest tests/integration/test_all_scenario_contracts.py -k sepsis_early_alert`

## Private files
None.

## Known limits
- `events` sums patient-time in sepsis, not new cases. New cases are in `secondary["incident_onset"]`; select them with `outcome_key="incident_onset"`. See `docs/analysis_exports.md`.
- Alert fatigue is a system-wide counter kept on the scenario object and reset in `create_population()`. Only the factual branch uses it. A scenario object runs once.
- The TREWS config is calibrated: baseline detection and treatment timing were tuned until the result bracketed the published value. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- Scores carry no time-to-onset information, so "detects hours before clinicians" is set by configuration, not predicted.
