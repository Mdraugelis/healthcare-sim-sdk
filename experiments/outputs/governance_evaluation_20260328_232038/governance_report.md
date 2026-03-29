# AI Governance Evaluation: No-Show Predictor

*20260328_232038 | 2000 patients | 90 days | seed 42*

## Phase 1: Local Validation Metrics

| AUC Target | Threshold | Policy | AUC | PPV | Sens | Cal Slope | Util | Collision | All Pass |
|------------|-----------|--------|-----|-----|------|-----------|------|----------|----------|
| 0.70 | 0.15 | threshold | 0.672 | 20.1% | 65.5% | 0.98 | 88.0% | 59.1% | **no** |
| 0.70 | 0.15 | urgent_first | 0.672 | 20.1% | 65.5% | 0.98 | 88.0% | 59.1% | **no** |
| 0.70 | 0.20 | threshold | 0.679 | 25.9% | 46.1% | 0.99 | 88.4% | 58.6% | **no** |
| 0.70 | 0.20 | urgent_first | 0.672 | 26.4% | 46.7% | 0.98 | 88.0% | 59.1% | **no** |
| 0.70 | 0.25 | threshold | 0.678 | 31.4% | 29.2% | 1.00 | 89.1% | 54.6% | **no** |
| 0.70 | 0.25 | urgent_first | 0.672 | 31.5% | 28.6% | 0.98 | 88.0% | 59.1% | **no** |
| 0.70 | 0.30 | threshold | 0.668 | 39.8% | 17.8% | 0.98 | 88.0% | 50.3% | **no** |
| 0.70 | 0.30 | urgent_first | 0.672 | 36.0% | 16.1% | 0.98 | 88.0% | 59.1% | **no** |
| 0.70 | 0.35 | threshold | 0.693 | 49.5% | 10.6% | 1.17 | 87.3% | 43.0% | **no** |
| 0.70 | 0.35 | urgent_first | 0.672 | 42.2% | 9.8% | 0.98 | 88.0% | 59.1% | **no** |
| 0.70 | 0.40 | threshold | 0.684 | 50.6% | 5.1% | 1.05 | 87.4% | 43.4% | **no** |
| 0.70 | 0.40 | urgent_first | 0.672 | 39.4% | 4.2% | 0.98 | 88.0% | 59.1% | **no** |
| 0.80 | 0.15 | threshold | 0.697 | 19.4% | 73.9% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.15 | urgent_first | 0.697 | 19.4% | 73.9% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.20 | threshold | 0.697 | 23.3% | 58.5% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.20 | urgent_first | 0.697 | 23.3% | 58.5% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.25 | threshold | 0.703 | 30.3% | 47.1% | 0.88 | 88.5% | 50.0% | **no** |
| 0.80 | 0.25 | urgent_first | 0.697 | 28.1% | 45.7% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.30 | threshold | 0.699 | 35.2% | 34.5% | 0.88 | 88.2% | 50.0% | **no** |
| 0.80 | 0.30 | urgent_first | 0.697 | 34.0% | 33.5% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.35 | threshold | 0.710 | 41.4% | 25.1% | 0.90 | 88.2% | 51.4% | **no** |
| 0.80 | 0.35 | urgent_first | 0.697 | 40.1% | 25.0% | 0.82 | 88.8% | 51.4% | **no** |
| 0.80 | 0.40 | threshold | 0.717 | 51.9% | 22.4% | 0.95 | 88.6% | 41.7% | **no** |
| 0.80 | 0.40 | urgent_first | 0.697 | 47.2% | 18.8% | 0.82 | 88.8% | 51.4% | **no** |
| 0.83 | 0.15 | threshold | 0.848 | 28.9% | 86.7% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.15 | urgent_first | 0.848 | 28.9% | 86.7% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.20 | threshold | 0.848 | 39.5% | 70.1% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.20 | urgent_first | 0.848 | 39.5% | 70.1% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.25 | threshold | 0.859 | 49.3% | 59.2% | 1.72 | 89.8% | 30.5% | **no** |
| 0.83 | 0.25 | urgent_first | 0.848 | 47.7% | 57.4% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.30 | threshold | 0.865 | 57.5% | 48.5% | 1.73 | 90.2% | 30.8% | **no** |
| 0.83 | 0.30 | urgent_first | 0.848 | 55.0% | 47.1% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.35 | threshold | 0.850 | 59.1% | 37.7% | 1.64 | 89.9% | 33.4% | **no** |
| 0.83 | 0.35 | urgent_first | 0.848 | 62.2% | 40.5% | 1.65 | 90.0% | 35.6% | **no** |
| 0.83 | 0.40 | threshold | 0.856 | 66.2% | 32.9% | 1.66 | 89.8% | 30.9% | **no** |
| 0.83 | 0.40 | urgent_first | 0.848 | 64.5% | 31.7% | 1.65 | 90.0% | 35.6% | **no** |
| 0.87 | 0.15 | threshold | 0.828 | 23.7% | 88.7% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.15 | urgent_first | 0.828 | 23.7% | 88.7% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.20 | threshold | 0.828 | 35.4% | 75.0% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.20 | urgent_first | 0.828 | 35.4% | 75.0% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.25 | threshold | 0.830 | 43.8% | 62.6% | 1.39 | 89.8% | 37.1% | **no** |
| 0.87 | 0.25 | urgent_first | 0.828 | 43.7% | 63.6% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.30 | threshold | 0.845 | 52.2% | 54.4% | 1.41 | 89.3% | 34.2% | **no** |
| 0.87 | 0.30 | urgent_first | 0.828 | 49.9% | 52.4% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.35 | threshold | 0.829 | 54.5% | 46.8% | 1.31 | 90.3% | 35.7% | **no** |
| 0.87 | 0.35 | urgent_first | 0.828 | 56.5% | 45.7% | 1.37 | 89.4% | 36.9% | **no** |
| 0.87 | 0.40 | threshold | 0.842 | 60.0% | 41.9% | 1.36 | 89.8% | 31.1% | **no** |
| 0.87 | 0.40 | urgent_first | 0.828 | 61.8% | 37.9% | 1.37 | 89.4% | 36.9% | **no** |

### No configuration meets all governance criteria.
Closest: AUC=0.83, thresh=0.25, policy=threshold (1 checks failing)

## Equity Audit

### Race/Ethnicity

| Race/Ethnicity | N | AUC | AUC Gap | PPV | Sens | Flag Rate | NoShow Rate | Flag/NS Ratio |
|---------|---|-----|---------|-----|------|-----------|-------------|---------------|
| Asian | 411 | 0.830 | 3% | 47.1% | 47.1% | 12.4% | 12.4% | 1.00 |
| Black | 870 | 0.877 | -2% | 49.2% | 67.4% | 21.3% | 15.5% | 1.37 |
| Hispanic | 1057 | 0.864 | -1% | 51.8% | 63.9% | 21.0% | 17.0% | 1.23 |
| Other | 367 | 0.839 | 2% | 41.5% | 65.4% | 22.3% | 14.2% | 1.58 |
| White | 3703 | 0.855 | 0% | 49.8% | 55.7% | 14.5% | 12.9% | 1.12 |

### Insurance Type

| Insurance Type | N | AUC | AUC Gap | PPV | Sens | Flag Rate | NoShow Rate | Flag/NS Ratio |
|---------|---|-----|---------|-----|------|-----------|-------------|---------------|
| Commercial | 2959 | 0.844 | 2% | 45.0% | 53.1% | 14.7% | 12.5% | 1.18 |
| Medicaid | 1319 | 0.882 | -3% | 57.5% | 66.7% | 19.3% | 16.6% | 1.16 |
| Medicare | 1607 | 0.857 | 0% | 48.1% | 59.1% | 17.6% | 14.3% | 1.23 |
| Self-Pay | 523 | 0.870 | -1% | 51.5% | 67.1% | 19.7% | 15.1% | 1.30 |

### Fairness Rule Checks

- [PASS] AUC >= 0.70 (minimum): 0.859 (target: 0.70)
- [PASS] AUC >= 0.80 (target): 0.859 (target: 0.80)
- [PASS] PPV >= 30% at threshold: 49.3% (target: 30%)
- [PASS] Sensitivity >= 50% at threshold: 59.2% (target: 50%)
- [**FAIL**] Calibration slope 0.8-1.2: 1.718 (target: 0.8-1.2)
- [PASS] race_ethnicity=Asian: AUC within 15% of overall: 0.830 (gap=3%) (target: >= 0.730)
- [PASS] race_ethnicity=Black: AUC within 15% of overall: 0.877 (gap=-2%) (target: >= 0.730)
- [PASS] race_ethnicity=Hispanic: AUC within 15% of overall: 0.864 (gap=-1%) (target: >= 0.730)
- [PASS] race_ethnicity=Other: AUC within 15% of overall: 0.839 (gap=2%) (target: >= 0.730)
- [PASS] race_ethnicity=White: AUC within 15% of overall: 0.855 (gap=0%) (target: >= 0.730)
- [PASS] insurance_type=Commercial: AUC within 15% of overall: 0.844 (gap=2%) (target: >= 0.730)
- [PASS] insurance_type=Medicaid: AUC within 15% of overall: 0.882 (gap=-3%) (target: >= 0.730)
- [PASS] insurance_type=Medicare: AUC within 15% of overall: 0.857 (gap=0%) (target: >= 0.730)
- [PASS] insurance_type=Self-Pay: AUC within 15% of overall: 0.870 (gap=-1%) (target: >= 0.730)
- [PASS] race_ethnicity=Asian: flag rate proportional to no-show rate: flag=12.4%, noshow=12.4%, ratio=1.00 (target: ratio 0.5-2.0)
- [PASS] race_ethnicity=Black: flag rate proportional to no-show rate: flag=21.3%, noshow=15.5%, ratio=1.37 (target: ratio 0.5-2.0)
- [PASS] race_ethnicity=Hispanic: flag rate proportional to no-show rate: flag=21.0%, noshow=17.0%, ratio=1.23 (target: ratio 0.5-2.0)
- [PASS] race_ethnicity=Other: flag rate proportional to no-show rate: flag=22.3%, noshow=14.2%, ratio=1.58 (target: ratio 0.5-2.0)
- [PASS] race_ethnicity=White: flag rate proportional to no-show rate: flag=14.5%, noshow=12.9%, ratio=1.12 (target: ratio 0.5-2.0)
- [PASS] insurance_type=Commercial: flag rate proportional to no-show rate: flag=14.7%, noshow=12.5%, ratio=1.18 (target: ratio 0.5-2.0)
- [PASS] insurance_type=Medicaid: flag rate proportional to no-show rate: flag=19.3%, noshow=16.6%, ratio=1.16 (target: ratio 0.5-2.0)
- [PASS] insurance_type=Medicare: flag rate proportional to no-show rate: flag=17.6%, noshow=14.3%, ratio=1.23 (target: ratio 0.5-2.0)
- [PASS] insurance_type=Self-Pay: flag rate proportional to no-show rate: flag=19.7%, noshow=15.1%, ratio=1.30 (target: ratio 0.5-2.0)

**22/23 governance checks passed.**
