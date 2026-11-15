# malnutrition_prioritization

**Status:** institution-specific

## What it models
Dietitians work one daily list: time-bound consults first, then a model's risk ranking, with the nursing screen as a floor. The counterfactual is the same capacity working the pre-model queue. The unit is an inpatient and a timestep is a day. Summed over a run, `events` equals the final count of missed malnutrition cases.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization.run_experiment`: Hydra runner; base `configs/base.yaml`, experiments in `configs/experiment/`.
- `python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization.run_los_readmit_sweep`, then `outcome_overlay.py`, `estimands.py` and `power_map.py`: the length-of-stay and readmission experiment.
- `python -m healthcare_sim_sdk.scenarios.malnutrition_prioritization.run_verification --stage A`: the verification gates. Run them before interpreting any result.

## Verify
- `pytest tests/integration/test_malnutrition_prioritization.py tests/bulletproof/test_malnutrition_conservation.py`
- With the private files present: `SIM_SDK_PRIVATE_TESTS=1 pytest -m private`

## Private files
- `configs/model_frozen.yaml` and `configs/model_targets.yaml` hold the real model's parameters and targets. Start from `configs/model_targets.example.yaml`.
- Everything in `configs/campus/` except `configs/campus/example.yaml`.
- Git ignores all of these. Never add them, and never copy their values into tracked files, tests or commit messages. See `docs/data_boundary.md`.
- Without them, runs with the model on raise `FileNotFoundError`. Tests substitute a public, uncalibrated synthetic scorer; three tests marked `private` need the real files.

## Known limits
- Results from the synthetic scorer say nothing about the real model.
- `intervene()` only writes the next day's worklist. All assessments and the capacity check happen in `step()`, on both branches.
- Every temporal draw in `step()` is a fixed-size array drawn unconditionally, with masks applied afterwards. Never draw inside a data-dependent mask or loop.
- The scorer in `subgroup_performance.py` has subgroup-specific accuracy, which `ControlledMLModel` does not.
