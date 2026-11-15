# paper03_kaiser_aam

**Status:** calibration exercise

## What it models
Kaiser Advance Alert Monitor, Escobar et al., NEJM 2020. A deterioration score reviewed by remote nurses before the ward is called, rolled out to 21 hospitals in waves. The unit is a ward patient.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.paper03_kaiser_aam.run_evaluation`: prints the simulated result beside the published value.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper03_kaiser_aam`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- Simulated mortality reduction is 1.32 points against 4.6 published. The runner checks direction and plausibility only.
- Baseline ward mortality of 14.4% is assumed to be in-window mortality.
