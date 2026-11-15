# paper01_epic_esm

**Status:** calibration exercise

## What it models
Epic Sepsis Model, Wong et al., JAMA Internal Medicine 2021. An external validation of a proprietary sepsis score: AUC 0.63, alerts at threshold 6. The unit is a patient admission and a timestep is 4 hours.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.paper01_epic_esm.run_evaluation`: prints the simulated result beside the published value.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper01_epic_esm`
- `pytest tests/unit/test_issue_register.py -k incident_onset`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- Simulated prevalence is 9.8% against 3.5% in the paper. The runner makes sanity checks only.
- `events` sums patient-time in sepsis; new cases are in `secondary["incident_onset"]`.
