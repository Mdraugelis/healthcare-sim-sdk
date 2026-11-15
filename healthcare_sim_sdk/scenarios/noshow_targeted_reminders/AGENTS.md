# noshow_targeted_reminders

**Status:** operational

## What it models
Phone reminders aimed at the appointments a model scores as most likely to be missed. A reminder lowers that appointment's no-show probability. Caller capacity is set in `CallerConfig`. The unit is an appointment.

## Entry points
- `python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_evaluation`: the main evaluation; `configs/evaluation.yaml`.
- `python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_threshold_sweep`: sweeps the targeting threshold and compares with a published benchmark; grids and the benchmark are in `configs/threshold_sweep.yaml`.
- `python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_review_demo`: writes a public review packet from `configs/review_demo.yaml` and `configs/review_claims.yaml`. Exempt from `finalize_experiment()`: a packet has its own manifest. See `docs/review_packets.md`.
- `python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.calibrate`: fits `reminder_effectiveness` to a published effect (`setting=chong` or `setting=rosen`). Exempt: a tuning tool.
- Study configs: `configs/chong_replication.yaml`, `configs/rosen_replication.yaml`. Local projection: `configs/local_projection.yaml`.

## Verify
- `pytest tests/integration/test_noshow_reminders.py tests/integration/test_reminders_review.py`
- `pytest tests/bulletproof/test_reminder_conservation.py tests/bulletproof/test_reminder_equity.py`
- `pytest tests/integration/test_all_scenario_contracts.py -k noshow_targeted_reminders`

## Private files
None. Put site-specific values in an ignored `*.private.yaml` copy of `configs/local_projection.yaml`, never in the tracked file.

## Known limits
- The Chong and Rosen configs are calibrated to each study: effectiveness was fitted to the published result. They show the mechanism, not a prediction. See [calibration exercises](../../../docs/design_principles.md#published-study-calibration-exercises).
- The tracked report `reports/noshow_targeted_reminders_proof_points.md` is a record dated 16 October 2026.
