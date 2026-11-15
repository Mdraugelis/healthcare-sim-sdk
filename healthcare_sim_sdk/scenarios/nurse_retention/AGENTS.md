# nurse_retention

**Status:** operational

## What it models
Nurses with time-varying turnover risk. A model flags high-risk nurses for manager check-ins, and each manager has limited capacity. The unit is a nurse and a timestep is a week. Design reasoning is in `EXPERIMENT_DESIGN.md`.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.nurse_retention.run_evaluation`: the evaluation experiment. `configs/retention_sweep.yaml` holds the Hydra grid for model quality against manager capacity.
- `python -m healthcare_sim_sdk.scenarios.nurse_retention.monitoring_validation`: Hydra runner for the four-tier monitoring harness across six regimes; configs under `configs/monitoring/`.
- `monitoring/`: Tier 1 Shewhart, Tier 2 CUSUM, Tier 3 controlled ITS, Tier 4 model drift.

## Verify
- `pytest tests/integration/test_monitoring_harness.py`
- `pytest tests/unit/test_tier1_shewhart.py tests/unit/test_tier2_cusum.py tests/unit/test_tier3_cits.py tests/unit/test_tier4_model_drift.py`
- `pytest tests/integration/test_all_scenario_contracts.py -k nurse_retention`

## Private files
None.

## Known limits
- The cohort is closed: departures are not replaced and high-risk nurses leave first. Raw turnover falls without any AI, so Tier 2 charts observed minus predicted departures. See `docs/algorithm_description.md` section 12.4.
- Tier 3 with the counterfactual branch as control is an oracle benchmark, not a deployable method. The default is an independent observed control.
- Monitoring is performative: a working program lowers the model's measured AUC. Tier 4 reports that separately.
- Standard-of-care check-ins draw one priority per nurse per step. Keep the draw count fixed when editing `step()`.
