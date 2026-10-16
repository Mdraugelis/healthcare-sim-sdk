# TREWS-style baseline detection and capacity sweep — revised

**Re-run: 16 October 2026. Replaces the pre-label-free report.**

This reproduces the original 28-cell grid with label-free scoring and an independent reference fit. Each cell uses seed 42, 5,000 admissions, 42 four-hour steps, model AUC target 0.82, top-7% alerts, and the unchanged TREWS replication assumptions. This is a sensitivity illustration with one seed per cell; it provides no Monte Carlo confidence interval and does not establish a faithful TREWS replication.

Entries are final cumulative whole-admission mortality differences, C−F, in percentage points. They are neither sepsis-patient mortality differences nor a sum of repeated mortality occupancy.

| Capacity per four-hour step | 6h max | 12h max | 18h max | 24h max | 36h max | 48h max | 72h max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| 25 | 0.28 | 0.28 | 0.28 | 0.28 | 0.28 | 0.28 | 0.28 |
| 50 | 0.48 | 0.48 | 0.48 | 0.48 | 0.46 | 0.46 | 0.46 |
| 100 | 0.60 | 0.60 | 0.60 | 0.60 | 0.62 | 0.62 | 0.62 |

The baseline clinical delay is Beta(2,5) × the column maximum; SOC runs on both branches. Kumar decay links treatment timing to efficacy. The model scores risk level and has no feature-level knowledge of proximity to sepsis onset. Therefore any numerical agreement with a published mortality effect would not validate the clinical detection mechanism. The historical replication conclusion and its exact pre-fix numbers are withdrawn. Sepsis `events` are patient-time occupancy; use `incident_onset` for case counts.

See [reproduction commands](../docs/issue_register_validation.md). Raw cells: `outputs/issue_register_trews/`.
