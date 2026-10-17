# Targeted reminders: revised proof-point checks

**Re-run: 16 October 2026. Replaces the pre-label-free report.**

**Source:** [`88be0ae`](https://github.com/Mdraugelis/healthcare-sim-sdk/commit/88be0aee41df8edf93ef647b5883f93be475484b) on a clean checkout, after all algorithm fixes. See the [rerun provenance](../docs/issue_register_rerun_provenance.json).

The original configured study parameters are retained, including their previously calibrated intervention effect sizes. Models now fit on an independent reference sampled from the patient-risk distribution instead of one small daily schedule. Each study uses 30 seeds (1000–1029); no effect parameter is re-tuned to recover a published number. This tests the configured simulation, not the truth of the clinical effect.

| Study configuration | Mean no-AI no-show rate | Mean factual no-show rate | Reduction, pp (95% CI of mean) | Delivered reference AUC | Gap narrowing, Black−White pp (95% CI) |
|---|---:|---:|---:|---:|---:|
| chong | 19.6% | 16.1% | 3.54 (3.47, 3.61) | 0.740 | 0.50 (0.33, 0.67) |
| rosen | 36.4% | 32.7% | 3.75 (3.70, 3.79) | 0.721 | 1.68 (1.55, 1.81) |

Positive gap narrowing means a larger absolute reduction among Black than White patients. Every appointment is paired to its no-AI counterpart, and the runner asserts that reminders do not increase any appointment’s no-show outcome. That monotonicity does not guarantee that a racial gap narrows in each finite seed, nor that increasing AUC improves every equity metric. Previous single-seed assertions of guaranteed disparity reduction are withdrawn. The Rosen demographic multipliers and effect sizes are assumed/calibrated inputs, not independently validated site facts.

Chong runs 3,000 patients over 90 days; Rosen runs 5,000 over 180 days, using the tracked replication YAMLs. Rates pool resolved appointment slots, including the initial untreated schedule. See [reproduction commands](../docs/issue_register_validation.md). Raw cells: `outputs/postfix_88be0ae_complete/issue_register_reminder/`.
