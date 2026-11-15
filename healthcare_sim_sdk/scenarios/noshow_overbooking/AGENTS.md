# noshow_overbooking

**Status:** operational

## What it models
A clinic that double-books slots a model predicts will be no-shows. Utilization rises; so can waits and the overbooking burden on some patient groups. The unit is an appointment. See `README.md` in this folder for the design.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_evaluation`: the structured evaluation experiment.
- `python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_governance_eval`: the AI governance evaluation.
- `python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_burden_analysis`: who carries the overbooking burden, for ethics review.
- `python -m healthcare_sim_sdk.scenarios.noshow_overbooking.run_threshold_optimizer`: Hydra sweep; config `configs/threshold_optimizer.yaml`.
- `realistic_scenario.py` and `evaluation_harness.py` are libraries the runners use.

## Verify
- `pytest tests/integration/test_noshow.py`
- `pytest tests/integration/test_all_scenario_contracts.py -k noshow_overbooking`
- Notebooks `examples/09_noshow_scenario.ipynb` and `examples/noshow_evaluation.ipynb` run in the quality gate.

## Private files
None.

## Known limits
- State is a dataclass of dicts and lists, so `step()` and `intervene()` copy it before changing it. Keep that when editing.
- A show-up coin is drawn for every slot, overbooked or not. This keeps the two branches' random streams aligned; do not make the draw conditional. See `docs/algorithm_description.md` section 6.
- Two model types: `baseline` (the patient's own history) and `predictor` (`ControlledProbabilityModel`).
