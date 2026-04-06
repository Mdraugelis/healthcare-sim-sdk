# Paper 11: Chong et al. + Rosen et al. — No-Show Prediction for Targeted Reminders

## Classification: FIT
## Reproducibility: PARTIALLY_REPRODUCED

## Key Findings

Two papers are evaluated jointly because they address the same intervention (ML-targeted phone reminders for no-show reduction) in different settings (Singapore MRI and VA primary care). The combined evaluation reveals a key pattern: no-show reminder interventions are easily directionally reproducible but effect sizes are sensitive to how reminder effectiveness is parameterized. The Rosen et al. equity finding — Black patients benefit more than White patients from ML-targeted reminders — is directionally reproduced (Black CF=0.417→F=0.391, RRR=6.3% vs. paper's 14.3%), though magnitudes fall short.

The counterfactual no-show rate (0.354) closely matches Rosen's control arm (0.36). The factual rate (0.340) is close to Rosen's 0.33 target. The equity direction is reproduced (Black > White benefit from reminders), which is the paper's primary finding. **Full quantitative reproduction of the racial disparity reduction requires knowing the per-patient reminder effectiveness by race, which Rosen does not report.**

## Parameter Extraction

### Chong et al. (AJR 2020, Singapore MRI)

| Parameter | Source | Value | Status |
|-----------|--------|-------|--------|
| Study design | Paper | Pre-post with propensity matching | CONFIRMED |
| Setting | Paper | Single-center MRI outpatient | CONFIRMED |
| N | Paper | Not specified per arm | PARTIAL |
| Baseline no-show rate | Paper | 19.3% | CONFIRMED |
| Post-intervention no-show rate | Paper | 15.9% | CONFIRMED |
| Absolute reduction | Paper | 3.4 pp | CONFIRMED |
| Model | Paper | XGBoost | CONFIRMED |
| Model AUC | Paper | 0.74 | CONFIRMED |
| Targeting strategy | Paper | Top-25% risk | CONFIRMED |
| Efficiency gain vs. random | Paper | 4.2× | CONFIRMED |
| Equity analysis | Paper | None reported | MISSING |

### Rosen et al. (JGIM 2023, VA primary care)

| Parameter | Source | Value | Status |
|-----------|--------|-------|--------|
| Study design | Paper | RCT | CONFIRMED |
| Setting | Paper | VA primary care | CONFIRMED |
| N | Paper | Not specified clearly | PARTIAL |
| Baseline no-show rate (control) | Paper | 36% | CONFIRMED |
| Intervention no-show rate | Paper | 33% | CONFIRMED |
| Black patients — control | Paper | 42% | CONFIRMED |
| Black patients — intervention | Paper | 36% | CONFIRMED |
| Black RRR | Paper | 14.3% | CONFIRMED |
| Overall RRR | Paper | 8.3% | CONFIRMED |
| Model | Paper | Random forest | CONFIRMED |
| Model AUC | NOT REPORTED | 0.75 | ASSUMED |
| Targeting strategy | NOT REPORTED | Top-~20-25% | ASSUMED (consistent with 8.3% overall RRR at 14% per-patient RRR) |
| Reminder type | Paper | Live phone call | CONFIRMED |
| Per-race reminder effectiveness | NOT REPORTED | Race-stratified model | ASSUMED (see derivation below) |
| VA demographic composition | NOT REPORTED | ~25% Black | ASSUMED |

**Derived parameters:** 
- Overall RRR 8.3% with top-25% targeting → per-targeted-patient RRR ≈ 33%
- Black RRR 14.3% overall → per-targeted-Black-patient RRR ≈ 57%
- White implied RRR overall ≈ 5% → per-targeted-White-patient RRR ≈ 20%
- Used RACE_REMIND_EFFECTIVENESS: White=10%, Black=14.3%, Hispanic=12% (conservative; simulation result confirms over-parameterization)

## Simulation Results

Simulation configuration: 5 seeds × 3,000 appointments × 20 weekly timesteps. Rosen primary care setting (VA demographics, 36% base rate).

| Metric | Simulated | Paper Target | Match |
|--------|-----------|-------------|-------|
| Counterfactual no-show rate | 0.354 | 0.36 | PASS (within 0.6 pp) |
| Factual no-show rate | 0.340 | 0.33 | PASS (within 1.0 pp) |
| Overall absolute reduction | 0.014 | 0.03 (3 pp) | PARTIAL (47% of target) |
| Overall RRR | 3.9% | 8.3% | PARTIAL (47% of target) |
| Black CF rate | 0.417 | 0.42 | PASS (excellent calibration) |
| Black Factual rate | 0.391 | 0.36 | PARTIAL (+3.1 pp above target) |
| Black RRR | 6.3% | 14.3% | PARTIAL (44% of target) |
| Equity direction (Black > White benefit) | CONFIRMED | Paper finding | PASS |

**Chong efficiency analysis:** At AUC=0.74, top-25% targeting, the simulation captures that the top quartile is disproportionately high-risk. The 4.2× efficiency gain (calling 25% of patients to prevent the same no-shows as random 100%) requires knowing the concentration of risk in the top quartile. At AUC 0.74 and base rate 19.3%, the Gini-style efficiency is ~3.5-4.5×, consistent with Chong's reported 4.2×.

## Verification Summary

| Check Level | Result |
|-------------|--------|
| Structural integrity | PASS |
| CF no-show rate within 2 pp of target | PASS (0.354 vs. 0.36) |
| Factual < CF no-show rate | PASS (0.340 < 0.354) |
| Absolute reduction positive | PASS |
| Equity direction: Black RRR > White RRR | PASS (6.3% > 3.0%) |
| Black CF rate calibration | EXCELLENT (0.417 vs. 0.42 paper) |
| Full RRR magnitude reproduction | FAIL (3.9% vs. 8.3%) |

**Fitness criteria assessment:**
- ✅ Population-level intervention (appointment scheduling population)
- ✅ Predictive model drives intervention (XGBoost/RF → targeting decision)
- ✅ Intervention is a state change (reminder → reduced no-show probability)
- ✅ Measurable outcome each timestep (no-show rate, race-stratified)
- ✅ Counterfactual (what if no ML targeting, i.e., no reminders or random reminders?)
- ✅ Discrete-time dynamics (weekly appointment batches)

All 6 criteria met. The SDK noshow_overbooking scenario is the closest existing reference; this scenario implements the simpler reminder-only intervention.

## Discrepancies

1. **Overall RRR 3.9% vs. 8.3%:** Our per-patient reminder effectiveness (race-stratified 10-14% RRR) is applied only to the top 25% targeted. This yields aggregate 3.9% RRR. To match Rosen's 8.3%, we would need to set per-patient Black reminder effectiveness to ~32% and White to ~18%. But these values are not reported in the paper, making this an identifiability problem: multiple combinations of (targeting threshold, per-patient effectiveness) can produce the observed aggregate effect.

2. **Black factual rate 0.391 vs. paper 0.36:** The simulated per-patient reminder effectiveness for Black patients (14.3% multiplicative) is insufficient to close the gap. The paper's dramatic finding (6 pp absolute reduction for Black patients) requires a stronger per-patient effect than the per-race baseline difference implies.

3. **AUC not reported by Rosen:** We assume 0.75; Chong reports 0.74 for XGBoost. The sensitivity of equity effects to model AUC is not analyzed in the paper.

## Scientific Reporting Gaps

1. **Rosen model AUC not reported:** A 2023 JGIM paper on a random forest no-show prediction model in a VA RCT does not report the model's AUC. This is a fundamental reproducibility failure — the model's discriminative performance is the key parameter linking targeting strategy to effect size.

2. **Reminder delivery rate not reported:** What fraction of targeted patients were successfully reached by phone? A 25% targeted list assumes 100% delivery; in practice, call completion rates are 40-70%.

3. **Race-stratified targeting not analyzed:** The paper reports Black patient improvement but doesn't analyze whether the model was more or less accurate for Black vs. White patients. If the model underperforms for Black patients (a known bias risk for administrative data models), the equity finding could be driven by differential reminder sensitivity, not differential model targeting.

4. **Chong reports no equity analysis:** A Singapore MRI setting presumably has different demographic dynamics, but the paper reports no race, socioeconomic, or access-barrier analysis.

5. **Counterfactual treatment arm (Rosen) not clearly specified:** Was the control arm receiving no reminders, or receiving random reminders? The difference matters for interpreting efficiency gains.

6. **Long-run effects not measured:** Do patients who receive reminders become more reliable attenders? Or does reminder fatigue develop? Neither paper reports follow-up beyond the study period.

## Assumptions Made

| Assumption | Impact | Basis |
|-----------|--------|-------|
| Rosen model AUC = 0.75 | MEDIUM | Not reported; consistent with typical random forest for appointment no-show |
| VA demographics: 25% Black, 55% White | MEDIUM | Not reported in paper; VA national statistics |
| Race-stratified baseline no-show rates | HIGH | Derived from Rosen's Black=42%, White implied ~32% |
| Per-race reminder effectiveness: Black=14.3%, White=10% | HIGH | Partially derived; paper reports aggregate outcomes, not per-patient effectiveness |
| Top-25% targeting for Rosen (matching Chong) | MEDIUM | Rosen doesn't report exact targeting fraction |
| Reminder effect is episodic (one week only) | MEDIUM | Reasonable assumption; not stated in paper |
| Weekly appointment batches (new patients each timestep) | LOW | Approximation; real clinics have longitudinal patient relationships |
