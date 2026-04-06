# Paper 27: Rajkomar et al. — Scalable and Accurate Deep Learning with Electronic Health Records (2018)

## Classification: PARTIAL_FIT
## Reproducibility: UNDERDETERMINED

## Key Findings

Rajkomar et al. demonstrate that a deep learning architecture applied to raw EHR data (Fast Healthcare Interoperability Resources — FHIR format) achieves strong retrospective discrimination across multiple clinical prediction tasks at UCSF and UChicago. Key results:

- **In-hospital mortality AUC: 0.93–0.94** (UCSF and UChicago separately)
- **30-day unplanned readmission AUC: 0.75–0.76**
- **Prolonged length of stay (>3 days) AUC: 0.85–0.86**
- **ICD discharge diagnosis prediction AUC: 0.90**

The model uses all structured and unstructured EHR data in FHIR format without domain-specific feature engineering. Validation was internal to each site (held-out test sets), not cross-institutional — UCSF trained ≠ UChicago tested. The dataset covers 216,221 patients.

This is a **retrospective validation study**, not a clinical deployment study. No patient outcomes were affected; the model's predictive performance was evaluated against future-known events in historical records. There is no intervention, no clinician in the loop, no outcome change attributable to the model.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Patients | 216,221 total | Paper | Extracted |
| Institutions | UCSF + UChicago | Paper | Extracted |
| Study design | Retrospective validation | Paper | Extracted |
| In-hospital mortality AUC | 0.93–0.94 | Paper | Extracted |
| 30-day readmission AUC | 0.75–0.76 | Paper | Extracted |
| LOS >3 days AUC | 0.85–0.86 | Paper | Extracted |
| ICD diagnosis AUC | 0.90 | Paper | Extracted |
| In-hospital mortality base rate | ~2.3% (UCSF), ~1.9% (UChicago) | Paper | Partial (inferred from cohort description) |
| 30-day readmission base rate | ~14–16% | Paper | Partial (estimated from similar inpatient cohorts) |
| External validation | No — each institution uses its own held-out set | Paper | Extracted |
| Clinical threshold | Not specified — no deployment threshold defined | N/A | Missing |
| Intervention | None | N/A | Missing |
| Effect size | None (retrospective validation only) | N/A | Missing |
| Clinician response rate | Not applicable | N/A | Missing |
| Training demographics | Not reported in detail | Paper | Missing |

**Missing parameters for SDK:** Intervention, threshold, clinician response, effect size, training demographics. These are not missing due to reporting gaps — they don't exist because no clinical deployment was described.

## Simulation Results

No simulation executed. Fitness assessment below:

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention — retrospective validation only |
| Predictive model drives intervention | ~ PARTIAL | Model exists and performs well; could drive an intervention in a deployment scenario |
| Intervention is a state change | ✗ FAIL | No state change — no patients were treated differently |
| Measurable outcome at each timestep | ~ PARTIAL | Mortality, readmission, LOS are measurable outcomes — but not at timesteps of an intervention |
| Counterfactual causal question | ✗ FAIL | No counterfactual — retrospective discrimination, not causal |
| Discrete-time dynamics | ~ PARTIAL | Inpatient trajectory is naturally discrete-time, but the paper doesn't model it dynamically |

**3 of 6 criteria partially met; 0 fully met → PARTIAL_FIT** (but requires a hypothetical deployment design, not paper-derived parameters)

**Note on PARTIAL_FIT:** This paper FITs the SDK's model specification layer — the AUC values are usable as `ControlledMLModel` inputs. But the paper does not describe any intervention to simulate. To use this paper in the SDK, a user would need to design a hypothetical deployment (e.g., "what if UCSF deployed this model to trigger early discharge planning?") and the simulation would test that hypothetical, not reproduce the paper's claims. This is a legitimate use but must be labeled as prospective scenario design, not reproduction.

## Verification Summary

Not applicable. No simulation was designed or executed.

**Hypothetical calibration design (for future scenario development):**

If a Geisinger team wanted to simulate deployment of a mortality prediction model with AUC 0.93–0.94:
1. `create_population()`: Inpatient cohort, ~2% in-hospital mortality base rate
2. `predict()`: `ControlledMLModel(mode="discrimination", target_auc=0.93)`
3. Threshold selection: Must be specified by the user — the paper provides no threshold guidance
4. `intervene()`: Clinical intervention (palliative care consult, ICU escalation, rapid response) must be specified by the user — the paper provides no intervention
5. `measure()`: In-hospital mortality, LOS
6. Intervention effectiveness: Must be derived from other literature (e.g., Paper #2 TREWS data)

This would require ≥4 assumed major parameters → UNDERDETERMINED under Phase 3 rules. Proceed as hypothetical scenario design, not paper reproduction.

## Discrepancies

Not applicable — no simulation was run.

**Anticipated discrepancy for any hypothetical deployment:** External validation AUC will likely be lower than 0.93–0.94 (Paper #24, Park & Han: internal vs. external validation performance gap). For Geisinger simulation planning, use 0.80–0.85 as a conservative external AUC estimate based on observed drops in comparable EHR models.

## Scientific Reporting Gaps

1. **No training demographics:** Race, sex, payer mix, and socioeconomic indicators of the 216,221-patient training set are not reported. This is a HIGH-impact gap for equity analysis.
2. **No subgroup performance:** AUC is not broken out by race, sex, age, or insurance status. Given the mortality base rate differences documented in other literature, this is a material omission.
3. **No threshold specification:** The paper reports AUC but does not define an operational threshold. Without a threshold, PPV and sensitivity are unspecified, making clinical deployment evaluation impossible.
4. **No external cross-institutional validation:** UCSF and UChicago train and validate separately; neither tests the other's model. The 0.93–0.94 AUC may not transfer. This is the exact failure mode documented in Paper #1.
5. **No temporal validation:** The held-out test sets are presumably from later time periods, but the paper does not describe temporal splits explicitly. Concept drift is not evaluated.
6. **No calibration metrics:** Only discrimination (AUC) is reported. Calibration (whether predicted probabilities match observed rates) is not assessed (see Paper #18, Van Calster).

## Assumptions Made

No simulation was run, so no parameters were assumed.

**Assumptions required for any future deployment simulation (all flagged HIGH):**
- [HIGH] Operational threshold: Unknown; assumed top-decile for planning purposes
- [HIGH] Intervention type and effectiveness: Absent from paper; must be sourced from deployment literature
- [HIGH] External validation AUC: Conservative estimate 0.80–0.85 based on systematic degradation patterns
- [HIGH] Training demographics: Unknown; Geisinger's own demographics assumed for local calibration

---

## SDK Design Rationale

This paper is best used as a **model performance benchmark** for SDK scenario configuration, not as a direct reproduction target.

**Use in the SDK:** When designing any inpatient mortality prediction scenario, the Rajkomar et al. values (AUC 0.93–0.94 internal) represent the optimistic upper bound on what deep learning on raw EHR can achieve. The SDK should flag when users set target AUC >0.90 without external validation evidence, noting that Rajkomar's internal validation performance is an upper bound, not an operational expectation.

The paper's FHIR-based data representation is also noteworthy for Geisinger's Epic/FHIR infrastructure — the same technical approach is already operationally feasible in Geisinger's environment.
