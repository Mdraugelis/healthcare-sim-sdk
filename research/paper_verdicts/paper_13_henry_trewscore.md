# Paper 13: Henry et al. — TREWScore for Septic Shock (Sci Transl Med 2015)

## Classification: PARTIAL_FIT
## Reproducibility: UNDERDETERMINED

---

## Key Findings

TREWScore (Cox proportional hazards + lasso, 26 predictors) achieved AUC 0.83 for predicting septic shock in ICU patients from MIMIC-II, identifying patients a median 28.2 hours before onset — two-thirds before any organ dysfunction. This is retrospective model development on a research database; the paper reports no deployed intervention, no treatment arm, no effect on outcomes, and no prospective validation. The SDK can partially instantiate the model's performance characteristics for a sepsis prediction scenario, but cannot reproduce any clinical outcome claims because none are made. Reproducibility is UNDERDETERMINED: the paper's claims (AUC 0.83, 28.2h lead time) are model validation claims, not deployment outcome claims — they require a MIMIC-II reconstruction, not an SDK simulation.

**SDK value:** TREWScore's parameters (AUC, base rate, lead time distribution) directly parameterize the `predict()` and `step()` components of a sepsis scenario. The paper is a foundational parameter source for Papers #2 (Adams/TREWS) and the SDK's `sepsis_early_alert` scenario, even though it cannot itself be simulated end-to-end.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Setting | ICU, MIMIC-II database | Inpatient ICU | AVAILABLE |
| Population N | 1,611 ICU admissions, 11,384 unique patients in MIMIC-II | Cohort size | PARTIALLY AVAILABLE |
| Septic shock base rate | ~8.4% (derived: 706 septic shock / ~8,400 eligible) | Base rate for population | INFERRED — HIGH IMPACT |
| Model type | Cox PH + lasso, 26 predictors | Model AUC + calibration | PARTIALLY AVAILABLE |
| Model AUC | 0.83 (AUROC for septic shock prediction) | Target AUC | AVAILABLE |
| Model sensitivity | ~0.85 at optimal threshold (paper reports ROC curve) | Sensitivity at threshold | PARTIALLY AVAILABLE |
| Model specificity | ~0.67 at optimal threshold | Specificity at threshold | PARTIALLY AVAILABLE |
| Lead time | Median 28.2h before septic shock onset | Temporal shift parameter | AVAILABLE |
| Lead time distribution | Not reported (only median) | Full distribution for step() | ABSENT |
| Intervention | None described | State change action | ABSENT |
| Intervention effectiveness | N/A | Multiplicative/additive risk reduction | ABSENT |
| Outcome rate post-intervention | N/A — retrospective development only | Post-intervention outcome rate | ABSENT |
| Calibration | Not reported | Calibration slope/intercept | ABSENT |
| Race/ethnicity breakdown | Not reported | Demographic parameters for equity | ABSENT |
| PPV at deployed threshold | Not reported for clinical threshold | Operational PPV | ABSENT |
| Clinician response model | N/A | Alert compliance curve | ABSENT |
| Feature availability at deployment | Not characterized | Real-time feature feasibility | ABSENT |

**Inferred parameters (HIGH IMPACT):**
- Septic shock base rate: The paper states MIMIC-II contains ~11,000 patients; septic shock patients numbered 706 in the development cohort. Base rate ~8.4% is inferred but uncertain — MIMIC is a critically ill ICU database with higher acuity than general wards.
- Threshold: Paper does not specify the operational threshold; optimal ROC threshold is not the deployed threshold.

**Parameters absent that would be needed for full simulation:**
- Any intervention arm (there is none — this is model development only)
- Clinical workflow integration
- Alert rate at any specific threshold
- Calibration statistics

---

## Simulation Results

**Partial SDK mapping attempted (assessment only — no code run):**

The SDK's `sepsis_early_alert` scenario can incorporate TREWScore parameters as follows:
- `predict()`: `ControlledMLModel(mode="discrimination", target_auc=0.83)`
- `step()`: Temporal dynamics can encode the 28.2h lead time as a temporal offset parameter
- `create_population()`: ICU patient entities with septic shock base rate ~8.4%

**What cannot be simulated from this paper:**
- `intervene()`: No intervention is described. Effectiveness = undefined.
- `measure()`: No outcome comparison exists. The paper measures model discrimination, not clinical outcomes.
- Counterfactual arm: There is no "no-model" comparison in the paper.

**Assessment:** This paper can contribute parameters to another scenario (Paper #2, Adams/TREWS) but cannot itself generate a simulated factual-vs-counterfactual comparison. The SDK would be running a parameterization test, not a reproducibility test.

---

## Verification Summary

No simulation run. Would run if used as parameter source for Adams/TREWS scenario.

SDK fitness criteria assessment:
- [x] Population-level intervention: Potentially (ICU population)
- [ ] Predictive model drives intervention: Model exists but no intervention defined
- [ ] Intervention is a state change: Not described
- [x] Measurable outcome at each timestep: Septic shock onset is measurable
- [ ] Counterfactual causal question: Not posed in this paper (retrospective development)
- [x] Discrete-time dynamics: ICU scoring is discrete-time compatible

**3 of 6 criteria met → NO_FIT for direct simulation; PARTIAL_FIT as parameter source.**

---

## Discrepancies

Not applicable — no simulation run against which to compare. The paper's AUC of 0.83 and 28.2h lead time are accepted as the reported values. No external validation data is provided in this paper; a direct AUC reproduction would require MIMIC-II access, which is outside the SDK's scope.

**Notable gap:** AUC 0.83 in MIMIC-II (retrospective, single-source, enriched ICU cohort) cannot be assumed transferable to other populations. The gap between this 0.83 and Epic's external AUC of 0.63 (Paper #1) illustrates exactly this risk. TREWScore's external performance was later demonstrated in the Adams 2022 prospective study (Paper #2, AUC not directly reported but sensitivity/specificity at clinical threshold documented).

---

## Scientific Reporting Gaps

1. **No calibration statistics reported.** AUC alone tells us nothing about whether predicted probabilities are reliable. A model with AUC 0.83 but E:O ratio of 2.0 would generate systematic overtreatment at any probability-based threshold. This is the exact deficiency Van Calster et al. (Paper #18) later systematized.

2. **No threshold specification.** The paper shows ROC curves but does not recommend an operational threshold. Any deployment of TREWScore would require threshold selection, which requires calibration data and decision analysis — neither of which this paper provides.

3. **No lead time distribution.** "Median 28.2 hours" is not actionable for SDK parameterization of step() dynamics. The shape of the lead time distribution (log-normal? skewed?) determines whether early warning is robust or driven by a few extreme outliers.

4. **MIMIC-II representativeness.** MIMIC-II is a single-center, high-acuity ICU database from Beth Israel Deaconess. Septic shock base rate, feature distributions, and clinical trajectories may not generalize. No external validation is presented in this paper.

5. **Zero equity analysis.** No demographic breakdown. MIMIC-II includes race/ethnicity data; it was not analyzed. The SDK would mandate this at design time.

6. **26 predictors listed but feature availability not characterized.** In a real deployment, some of these 26 predictors may be missing, delayed, or unavailable in electronic form. This gap between retrospective feature availability and prospective feature availability is a known source of performance degradation not addressed in this paper.

---

## Assumptions Made

| Assumption | Impact | Basis |
|------------|--------|-------|
| Septic shock base rate ~8.4% | HIGH | Derived from cohort description; MIMIC-II is enriched ICU population |
| AUC 0.83 represents discrimination at optimal threshold | MEDIUM | Paper presents AUROC; operational threshold not specified |
| Lead time distribution is approximately log-normal | MEDIUM | Median 28.2h reported only; distribution shape assumed from clinical literature |
| 26 predictors are available in real time | HIGH | Paper uses retrospective features; prospective availability not validated |
| Demographic distribution follows MIMIC-II composition | MEDIUM | Not stated in paper; MIMIC-II is ~60% white, ~15% Black, ~7% Hispanic |

---

## SDK Design Contribution

TREWScore provides the performance benchmark for sepsis prediction models in the SDK's model library. The AUC 0.83 and 28.2h lead time should be documented as the **development-set upper bound** for sepsis prediction, distinct from the deployed performance of TREWS (Paper #2). This contrast — development AUC vs. deployed clinical utility — is a core SDK narrative.

The paper's absence of calibration, threshold specification, and equity analysis provides three concrete checklist items that the SDK's pre-simulation gate should require before any model is deployed in a scenario.

**Priority for SDK parameter library:** HIGH — directly parameterizes the `sepsis_early_alert` scenario's `predict()` component.
