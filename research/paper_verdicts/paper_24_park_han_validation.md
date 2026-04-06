# Paper 24: Park & Han — Methodologic Guide for Evaluating Clinical Performance and Effect of AI Technology for Medical Diagnosis and Prediction (2018)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Park and Han reviewed 516 published AI studies in radiology and medical imaging, finding that the field's clinical validation standards are alarmingly weak: only **6% had external validation**, and **0% met all three design quality features simultaneously** — cohort (not case-control) study design, multi-institutional data, and prospective data collection. The paper provides a methodological framework distinguishing diagnostic performance evaluation (technical: does the model discriminate?) from clinical performance evaluation (operational: does it change decisions and outcomes?).

The critical distinction is between **case-control studies** (artificially enriched diseased/non-diseased populations, inflating apparent AUC) and **cohort studies** (representative clinical populations with realistic base rates, measuring true operational performance). A model trained and validated on enriched case-control data will almost always perform worse when deployed into a clinical cohort — which is precisely the failure mode documented in Paper #1 (Wong et al., Epic Sepsis Model).

The paper also distinguishes clinical impact studies (does the AI change physician behavior and patient outcomes?) from the vastly more common technical validation studies (can the model classify correctly?). This distinction directly maps to the SDK's fitness criteria.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Papers reviewed | 516 AI studies in medical imaging | Paper | Extracted |
| External validation rate | 6% (of 516 studies) | Paper | Extracted |
| Studies meeting all 3 design criteria | 0% | Paper | Extracted |
| Design criterion 1 | Cohort (not case-control) study design | Paper | Extracted |
| Design criterion 2 | Multi-institutional data | Paper | Extracted |
| Design criterion 3 | Prospective data collection | Paper | Extracted |
| AUC inflation (case-control vs cohort) | Not quantified; direction documented | Paper | Partial |
| Clinical performance studies | Minority (<6% with external val; clinical impact far rarer) | Paper | Inferred |

**Missing parameters for SDK:** No deployment, no intervention, no measurable patient outcome — this is a methodological review.

## Simulation Results

No simulation conducted. Paper is a methodological framework and survey.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention — this is a survey of study designs |
| Predictive model drives intervention | ✗ FAIL | Models are evaluated for technical performance, not interventions |
| Intervention is a state change | ✗ FAIL | No state change defined |
| Measurable outcome at each timestep | ✗ FAIL | No patient-level temporal outcome |
| Counterfactual causal question | ✗ FAIL | No counterfactual; paper reviews study methodology |
| Discrete-time dynamics | ✗ FAIL | No temporal dynamics |

**0 of 6 criteria met → NO_FIT**

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

The paper itself identifies gaps in the literature it reviews:
1. **0% of studies meet all three design criteria** — the paper correctly flags this but does not estimate the magnitude of AUC inflation from case-control designs, which would be a valuable quantitative companion finding.
2. **No breakdown by disease area or model type:** Are certain specialties (radiology vs. pathology vs. cardiology) more or less likely to achieve cohort design?
3. **No temporal trend analysis:** Is external validation becoming more common over time, or is the 6% rate stable?
4. **No analysis of publication bias:** Studies with weak external validation that also show poor generalization may be less likely to be published, meaning 6% is an optimistic estimate.

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper provides the **empirical foundation for why pre-deployment simulation matters** and directly motivates the SDK's entire counterfactual framework.

**Concrete SDK requirements derived from this paper:**

1. **Case-control vs. cohort AUC adjustment:** When a paper reports AUC from a case-control design (artificially balanced 50/50 or 1:1 enriched), the SDK should flag this and apply a downward AUC adjustment before simulation. The adjustment formula (converting from enriched prevalence AUC to natural prevalence AUC) should be documented in the SDK's `ControlledMLModel` configuration. Park & Han establish the *need* for this; Fawcett (2006) and others provide the mechanics.

2. **External validation gate in Phase 2 (fitness):** The SDK's fitness assessment should include as a metadata field: "Has this paper been externally validated?" If NO (94% of papers per Park & Han), the SDK should display a warning: "Model performance derived from internal validation only; simulation results represent best-case scenario for this model class. External validation performance is expected to be lower (see Paper #24)."

3. **Three-criterion quality score:** Every paper processed through the SDK pipeline should be scored on Park & Han's three design criteria: (a) cohort design (yes/no), (b) multi-institutional (yes/no), (c) prospective (yes/no). This becomes a standard metadata field in the Phase 1 parameter extraction template.

4. **Clinical vs. diagnostic performance distinction in fitness criteria:** The SDK's Criterion 1 (population-level intervention) and Criterion 5 (counterfactual causal question) together operationalize Park & Han's distinction between technical validation (not SDK-appropriate) and clinical impact studies (SDK-appropriate). This paper provides the external justification for why those two criteria exist.

**Connection to other papers:**
- Paper #1 (Wong et al.) is the canonical empirical instantiation of Park & Han's warnings — AUC 0.76–0.83 internal → 0.63 external.
- Paper #23 (Ghassemi) explains why XAI doesn't compensate for the external validation gap.
- Paper #18 (Van Calster calibration) adds the calibration dimension to Park & Han's three-criteria framework.
- The 0% achievement rate for all three criteria is the most important number in this batch for Geisinger procurement: **zero published AI studies in this field met all three design standards as of 2018.**
