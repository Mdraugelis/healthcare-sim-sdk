# paper07_manz_nudges

**Status:** calibration exercise

## What it models
Manz et al., JAMA Oncology 2020. A mortality model triggers behavioral nudges to oncologists; the outcome is the rate of serious-illness conversations. The unit is a patient and a timestep is a week.

## Entry points
- No runner and no config. Build the scenario class from `scenario.py` in code.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper07_manz_nudges`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- Direction checks only.
- The high-risk nudge goes to the patients the model selects, so outcomes depend on its scores. Keep that link when editing.
- The paper's abstract reports no AUC; 0.78 is assumed.
