# Appendix: Simulation Validation

**30/32 checks passed.** Each check verifies a configuration parameter against the actual simulation output.

*Experiment: 20260328_224503 | Seed: 42 | Patients: 2,000 | Days: 60*

| # | Check | Expected | Actual | Result |
|---|-------|----------|--------|--------|
| 1 | Patient panel size | 2000 | 2000 | PASS |
| 2 | Simulation duration | 60 | 60 | PASS |
| 3 | Daily capacity | 72 | 72.0 (mean) | PASS |
| 4 | Population no-show rate | 13.0% | 14.5% | PASS |
| 5 | Utilization accounting | ~100% | 103.0% | **FAIL** |
| 6 | No-show distribution shape | median < mean | median=0.0415, mean=0.1300 | PASS |
| 7 | No-show probability bounds | [0.01, 0.80] | [0.0101, 0.8000] | PASS |
| 8 | No-show probability mean | 0.1300 | 0.1300 | PASS |
| 9 | Visit type: chronic | 20% | 19% (372/2000) | PASS |
| 10 | Visit type: routine | 50% | 52% (1034/2000) | PASS |
| 11 | Visit type: infrequent | 30% | 30% (594/2000) | PASS |
| 12 | Race: White | 58% | 58% | PASS |
| 13 | Race: Black | 13% | 13% | PASS |
| 14 | Race: Hispanic | 18% | 18% | PASS |
| 15 | Race: Asian | 6% | 6% | PASS |
| 16 | Race: Other | 5% | 6% | PASS |
| 17 | Insurance: Commercial | 45% | 47% | PASS |
| 18 | Insurance: Medicare | 25% | 25% | PASS |
| 19 | Insurance: Medicaid | 20% | 20% | PASS |
| 20 | Insurance: Self-Pay | 10% | 9% | PASS |
| 21 | Age band: 18-29 | 15% | 15% | PASS |
| 22 | Age band: 30-44 | 22% | 22% | PASS |
| 23 | Age band: 45-64 | 35% | 35% | PASS |
| 24 | Age band: 65+ | 28% | 28% | PASS |
| 25 | ML model AUC | 0.83 | 0.845 | PASS |
| 26 | Individual overbooking cap | <= 10 | max observed: 0 | PASS |
| 27 | Baseline threshold applied | >= 50% hist rate | verified by overbooking pattern | PASS |
| 28 | Waitlist accumulation (control) | ~300 | 0 | **FAIL** |
| 29 | AR(1) modifier mean | ~1.0 | 0.9968 | PASS |
| 30 | AR(1) modifier spread | ~0.128 | 0.1308 | PASS |
| 31 | AR(1) modifier bounds | [0.5, 2.0] | [0.500, 1.445] | PASS |
| 32 | Total slots resolved | ~4248 | 4320 | PASS |

## Failed Checks

**Utilization accounting:** Utilization + no-show rate should ≈ 100%
- Expected: ~100%
- Actual: 103.0%
- Detail: Util=88.5% + NoShow=14.5% = 103.0%

**Waitlist accumulation (control):** Config: 5/day x 60 days = ~300 max (Poisson arrival)
- Expected: ~300
- Actual: 0
- Detail: Expected ~300, observed 0 (some filtered as recently scheduled)

## Detail

### Population

- [PASS] **Patient panel size**: Config: n_patients=2000
  - Expected: 2000 | Actual: 2000
- [PASS] **Simulation duration**: Config: n_days=60
  - Expected: 60 | Actual: 60
  - Results contain 60 timesteps
### Clinic Capacity

- [PASS] **Daily capacity**: Config: 6 providers x 12 slots = 72
  - Expected: 72 | Actual: 72.0 (mean)
  - Range: [72, 72]
### No-Show Rate

- [PASS] **Population no-show rate**: Config: base_noshow_rate=13%
  - Expected: 13.0% | Actual: 14.5%
  - Difference: 1.5%
- [FAIL] **Utilization accounting**: Utilization + no-show rate should ≈ 100%
  - Expected: ~100% | Actual: 103.0%
  - Util=88.5% + NoShow=14.5% = 103.0%
### Distributions

- [PASS] **No-show distribution shape**: Beta-distributed, right-skewed (median < mean)
  - Expected: median < mean | Actual: median=0.0415, mean=0.1300
- [PASS] **No-show probability bounds**: All probabilities in [0.01, 0.80]
  - Expected: [0.01, 0.80] | Actual: [0.0101, 0.8000]
- [PASS] **No-show probability mean**: Population mean near 13%
  - Expected: 0.1300 | Actual: 0.1300
### Visit Types

- [PASS] **Visit type: chronic**: Expected proportion: 20%
  - Expected: 20% | Actual: 19% (372/2000)
- [PASS] **Visit type: routine**: Expected proportion: 50%
  - Expected: 50% | Actual: 52% (1034/2000)
- [PASS] **Visit type: infrequent**: Expected proportion: 30%
  - Expected: 30% | Actual: 30% (594/2000)
### Demographics

- [PASS] **Race: White**: Expected: 58%
  - Expected: 58% | Actual: 58%
- [PASS] **Race: Black**: Expected: 13%
  - Expected: 13% | Actual: 13%
- [PASS] **Race: Hispanic**: Expected: 18%
  - Expected: 18% | Actual: 18%
- [PASS] **Race: Asian**: Expected: 6%
  - Expected: 6% | Actual: 6%
- [PASS] **Race: Other**: Expected: 5%
  - Expected: 5% | Actual: 6%
- [PASS] **Insurance: Commercial**: Expected: 45%
  - Expected: 45% | Actual: 47%
- [PASS] **Insurance: Medicare**: Expected: 25%
  - Expected: 25% | Actual: 25%
- [PASS] **Insurance: Medicaid**: Expected: 20%
  - Expected: 20% | Actual: 20%
- [PASS] **Insurance: Self-Pay**: Expected: 10%
  - Expected: 10% | Actual: 9%
- [PASS] **Age band: 18-29**: Expected: 15%
  - Expected: 15% | Actual: 15%
- [PASS] **Age band: 30-44**: Expected: 22%
  - Expected: 22% | Actual: 22%
- [PASS] **Age band: 45-64**: Expected: 35%
  - Expected: 35% | Actual: 35%
- [PASS] **Age band: 65+**: Expected: 28%
  - Expected: 28% | Actual: 28%
### Model Performance

- [PASS] **ML model AUC**: Config target: 0.83
  - Expected: 0.83 | Actual: 0.845
  - Measured on predictions vs actual outcomes
### Policy Constraints

- [PASS] **Individual overbooking cap**: Config: max_individual_overbooks=10
  - Expected: <= 10 | Actual: max observed: 0
- [PASS] **Baseline threshold applied**: Config: baseline_threshold=0.5
  - Expected: >= 50% hist rate | Actual: verified by overbooking pattern
  - Baseline overbooked 293 slots
### Waitlist

- [FAIL] **Waitlist accumulation (control)**: Config: 5/day x 60 days = ~300 max (Poisson arrival)
  - Expected: ~300 | Actual: 0
  - Expected ~300, observed 0 (some filtered as recently scheduled)
### AR(1) Behavioral Drift

- [PASS] **AR(1) modifier mean**: Config: rho=0.95, mean-reverts to 1.0
  - Expected: ~1.0 | Actual: 0.9968
- [PASS] **AR(1) modifier spread**: Expected std ≈ 0.128 (sigma/sqrt(1-rho^2))
  - Expected: ~0.128 | Actual: 0.1308
- [PASS] **AR(1) modifier bounds**: All modifiers in [0.5, 2.0]
  - Expected: [0.5, 2.0] | Actual: [0.500, 1.445]
### Accounting

- [PASS] **Total slots resolved**: Expected: ~4248 (72/day x 59 resolved days)
  - Expected: ~4248 | Actual: 4320
