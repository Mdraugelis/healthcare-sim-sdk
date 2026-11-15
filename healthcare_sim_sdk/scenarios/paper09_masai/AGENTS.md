# paper09_masai

**Status:** calibration exercise

## What it models
MASAI, Lång et al., Lancet Oncology 2023. AI triage of screening mammograms: screens scoring at or above a set percentile get double reading, and the rest get a single AI-assisted read. The unit is a screening mammogram.

## Entry points
- No runner and no config. Build the scenario class from `scenario.py` in code.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper09_masai`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- It points the wrong way: 4.30 cancers per 1,000 in the simulated AI arm against 4.57 for control. The trial found 6.7 and 5.0.
- The no-AI branch is scored as single AI-assisted reading, not double reading. That is the cause, and it is not fixed.
- Routing uses a score percentile, because model scores are calibrated probabilities and a fixed cut written for another scale flags nobody.
