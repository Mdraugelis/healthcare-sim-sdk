# paper06_shield_rt

**Status:** calibration exercise

## What it models
SHIELD-RT, Hong et al., JCO 2020. A model flags radiotherapy patients at high risk of acute care; flagged patients get twice-weekly evaluation. The unit is a radiotherapy course.

## Entry points
- No runner and no config. Build the scenario class from `scenario.py` in code.

## Verify
- `pytest tests/integration/test_all_scenario_contracts.py -k paper06_shield_rt`
- `pytest tests/unit/test_benchmark_shield_rt.py tests/integration/test_blinded_benchmark.py`
- The contract test checks step purity and branch coupling only. Nothing tests agreement with the paper.

## Private files
None.

## Known limits
- This is a calibration exercise. The published result set its inputs, so it cannot confirm that result. Never describe it as validated or reproduced. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- At its documented settings it stops with an error: AUC 0.81 against a population ceiling of about 0.72. The model gate is correct to refuse.
- Effectiveness was fitted to the trial result, and the effect compounds on each re-flag.
- The separate frozen-input benchmark uses `examples/benchmarks/shield_rt_pilot.json`. It is outcome-exposed development work, not a blinded result. See `docs/blinded_benchmark.md`.
