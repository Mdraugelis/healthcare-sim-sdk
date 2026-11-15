# paper04_insight_rct

**Status:** calibration exercise

## What it models
InSight sepsis RCT, Shimabukuro et al., 2017. A small ICU trial (n = 142) of a vital-signs sepsis model. The unit is an ICU patient; the control arm is the counterfactual branch.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.paper04_insight_rct.run_evaluation`: prints the simulated result beside the published value.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper04_insight_rct`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- Simulated control mortality is 42.3% against 21.3% published. Direction and plausibility checks only.
- The paper reports no AUC; 0.78 is assumed.
