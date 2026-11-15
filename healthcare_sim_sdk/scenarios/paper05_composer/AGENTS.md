# paper05_composer

**Status:** calibration exercise

## What it models
COMPOSER, Boussina, Wardi et al., npj Digital Medicine 2024. A nurse-facing sepsis advisory in the emergency department, with bundle compliance and mortality as outcomes. The unit is a septic patient.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.paper05_composer.run_evaluation`: prints the simulated result beside the published value.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper05_composer`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- The no-AI branch has 0% bundle compliance, giving a 26-point gain against 5 published. Direction checks only.
- AUC, baseline mortality and baseline compliance are assumed, not reported.
