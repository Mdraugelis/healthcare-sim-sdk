# Nurse retention: model quality and manager capacity — revised

**Re-run: 16 October 2026. Replaces the pre-label-free report.**

**Source:** [`88be0ae`](https://github.com/Mdraugelis/healthcare-sim-sdk/commit/88be0aee41df8edf93ef647b5883f93be475484b) on a clean checkout, after all algorithm fixes. See the [rerun provenance](../docs/issue_register_rerun_provenance.json).

The simulation suggests, under these assumptions, the following cumulative departure contrasts. Each cell has 30 seeds (1000–1029), 1,000 nurses, and 52 weekly steps. All model fits use a larger independent bootstrap reference from the initial risk population, with two-week labels. Subsequent refits keep that reference; survivor selection does not redefine the configured training quality.

| Reference AUC target | Capacity per manager | Mean departures prevented, C−F (95% CI of mean) | Mean delivered reference AUC | Mean fit positives |
|---:|---:|---:|---:|---:|
| 0.60 | 2 | 15.90 (14.55, 17.25) | 0.596 | 197 |
| 0.60 | 4 | 19.57 (17.87, 21.27) | 0.596 | 197 |
| 0.60 | 6 | 21.73 (20.49, 22.98) | 0.596 | 197 |
| 0.60 | 8 | 19.10 (17.55, 20.65) | 0.596 | 197 |
| 0.70 | 2 | 23.40 (21.47, 25.33) | 0.698 | 197 |
| 0.70 | 4 | 30.07 (28.05, 32.08) | 0.698 | 197 |
| 0.70 | 6 | 28.37 (26.72, 30.02) | 0.698 | 197 |
| 0.70 | 8 | 24.67 (23.65, 25.69) | 0.698 | 197 |
| 0.75 | 2 | 27.03 (25.35, 28.71) | 0.747 | 197 |
| 0.75 | 4 | 32.93 (31.22, 34.65) | 0.747 | 197 |
| 0.75 | 6 | 30.83 (29.23, 32.44) | 0.747 | 197 |
| 0.75 | 8 | 28.73 (26.83, 30.64) | 0.747 | 197 |

**Fit-quality check:** 25/90 distinct AUC–seed reference fits miss the 0.02 delivered-AUC tolerance; the largest absolute gap is 0.052. Each fit is reused across four capacity cells. These misses emit warnings; the aggregate means above do not certify that every fitted model delivers its nominal AUC.

The old AUC 0.80 and 0.85 cells are withdrawn: this risk distribution does not support those targets reliably. They are replaced by attainable reference targets rather than described as delivered model quality. Configured quality refers to the training reference, not guaranteed discrimination in a selected post-intervention cohort.

**Estimand limit:** both branches receive SOC; the factual branch also allocates AI check-ins. The current scenario does not enforce a single shared SOC-plus-AI time budget. These contrasts therefore measure the configured AI augmentation, not pure targeting benefit at equal total manager hours. The earlier equal-time interpretation is withdrawn. Confidence intervals describe the mean across synthetic sites; they are not prediction intervals for an individual real deployment.

See [reproduction commands](../docs/issue_register_validation.md). Raw cells are under `outputs/postfix_88be0ae_complete/issue_register_retention_verified/`.
