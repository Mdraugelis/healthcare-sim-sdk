# Paper 15: Obermeyer et al. — Racial bias in commercial health algorithm (Science 2019)

## Classification: FIT
## Reproducibility: UNDERDETERMINED

---

## Key Findings

An Optum/Change Healthcare algorithm used healthcare cost as a proxy for health need, systematically assigning Black patients lower risk scores than equally sick White patients — because Black patients receive less care for the same level of illness, their costs are lower, not their needs. Correcting the proxy (predicting health need directly rather than cost) would increase the fraction of Black patients identified for high-need care management programs from 17.7% to 46.5%, representing an 84% reduction in racial bias. With approximately 200 million patients/year affected by similar algorithms, this is among the largest systematic biases in U.S. healthcare delivery. The SDK can model this scenario: the key question is simulatable (what happens to demographic parity when we change the proxy variable?), and all 6 fitness criteria are met. Reproducibility is UNDERDETERMINED because the commercial algorithm's training data, feature set, and exact threshold are proprietary — we would be simulating a structural analog, not the exact algorithm.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Setting | Large academic medical center, commercial insurer data | Outpatient / care management | AVAILABLE |
| Population N | ~48,784 patients analyzed | Cohort size | AVAILABLE |
| Patient entity type | Insured patients, continuous enrollment | Enrollment entities | AVAILABLE |
| Risk score input | Algorithm outputs a percentile risk score | Prediction score ∈ [0,1] | AVAILABLE (structural) |
| Black patient fraction | ~46.5% of patients with identical health need | Demographic proportion | AVAILABLE |
| Baseline (biased) Black enrollment rate | 17.7% of top-risk tier | Base rate for equity audit | AVAILABLE |
| Debiased Black enrollment rate | 46.5% of top-risk tier | Post-debiasing target | AVAILABLE |
| Bias reduction magnitude | 84% bias reduction with target reformulation | Effect size | AVAILABLE |
| Algorithm accuracy (predicting cost) | High — cost prediction was accurate | Model AUC for cost proxy | PARTIALLY AVAILABLE |
| Algorithm accuracy (predicting need) | Lower than cost — but paper demonstrates reformulation reduces bias | AUC after reformulation | PARTIALLY AVAILABLE |
| Proxy mechanism | Cost used instead of health need; Black patients receive ~$1,800 less care/year for same health status | Structural bias parameter | AVAILABLE |
| Health need operationalization | Number of active chronic conditions (paper's "true need" measure) | Ground truth label | AVAILABLE |
| Cost gap | Black patients receive $1,800 less/year at equivalent health burden | Differential resource utilization | AVAILABLE |
| Population impact | ~200M patients/year in similar algorithms | Scale parameter | AVAILABLE |
| Intervention threshold | Top risk tier (threshold not precisely stated; ~top 5–10%) | Operational threshold | PARTIALLY AVAILABLE |
| Calibration | Not reported | Calibration slope/intercept | ABSENT |
| Study design | Retrospective observational, single large health system | Study type | AVAILABLE |
| Race categorization method | Not reported (likely EHR-derived) | Demographic labeling method | ABSENT |

**Inferred parameters (HIGH IMPACT):**
- Risk threshold: "High-risk tier" implies top ~5–10% of risk distribution; paper discusses top-N but exact percentile is not specified.
- Algorithm AUC for health need: Not reported. The original algorithm achieves high AUC for cost prediction; health need AUC is lower but unquantified.
- Effect of reformulation on AUC: Paper reports 84% bias reduction but doesn't separately report AUC of the debiased model.

---

## Simulation Results

**Assessment of simulatability — no code run due to missing proprietary parameters.**

**SDK scenario design is feasible:**

The scenario structure maps directly to the SDK:

```
create_population():
    N = 48,784 patients
    race: Black proportion ~17.7% initially selected for care management (biased), 
          true proportion in equivalent-need population = 46.5%
    health_need: continuous variable (chronic condition count)
    cost: derived from health_need * (1 - access_gap)
    access_gap: 0.20 for Black patients (receives ~20% less care for same need)

predict():
    biased_model: predicts cost (high AUC for cost, biased against Black patients)
    debiased_model: predicts health_need directly (lower cost-prediction AUC, equitable)

intervene():
    top-tier selection: patients above threshold enter care management
    state change: care_management_enrolled = True

measure():
    primary: fraction of enrolled patients who are Black (vs. equal-need baseline)
    secondary: health outcomes under care management (not reported in paper)
    equity: demographic parity, equalized odds, equal opportunity
```

**The core finding is directly simulatable:** Switching proxy from cost to health_need shifts Black enrollment from 17.7% to 46.5%. The SDK can reproduce this directional finding with synthetic data matching the paper's structural parameters.

**What cannot be reproduced exactly:** The commercial algorithm's exact AUC, feature set, training data, and threshold are proprietary. The simulation would validate the *mechanism* (proxy variable bias) not the *specific algorithm*.

---

## Verification Summary

No simulation run. SDK fitness criteria:
- [x] Population-level intervention: Yes (care management enrollment)
- [x] Predictive model drives intervention: Yes (risk score → enrollment threshold)
- [x] Intervention is a state change: Yes (enrolled vs. not-enrolled in care management)
- [x] Measurable outcome at each timestep: Yes (enrollment rate, demographic parity)
- [x] Counterfactual causal question: Yes (what if cost proxy replaced with health need?)
- [x] Discrete-time dynamics: Yes (annual enrollment cycles are discrete-time)

**6 of 6 criteria met → FIT.**

The UNDERDETERMINED reproducibility classification reflects proprietary algorithm details, not SDK fitness. The scenario is fully modelable; the exact calibration to the commercial algorithm is blocked by trade secrecy.

---

## Discrepancies

No simulation run. Anticipated discrepancy:

**Proprietary algorithm wall:** The exact algorithm is the Optum/Change Healthcare product, which is not publicly described. Any SDK simulation would be a structural analog demonstrating that the *mechanism* (cost proxy + access differential = demographic bias) produces the direction of bias reported. The magnitude (17.7%→46.5%) can be calibrated in simulation by setting access_gap to the reported $1,800/year differential, but this calibration is assumption-dependent.

**The bias mechanism is not disputed** — Obermeyer et al.'s finding has been widely replicated across similar cost-proxy algorithms. The SDK's value is not proving the finding but quantifying what it looks like in a specific system (e.g., the health system's population demographics and access patterns).

---

## Scientific Reporting Gaps

1. **Algorithm identity not confirmed.** The paper identifies Optum/Change Healthcare as the developer but doesn't confirm the exact commercial product. This protects the paper from legal risk but makes direct replication impossible.

2. **Threshold not specified precisely.** "High-risk tier" is described qualitatively; the exact percentile cutoff affects the magnitude of demonstrated bias. A 5th-percentile threshold vs. 10th-percentile threshold changes the absolute numbers dramatically.

3. **No calibration reported.** A model with high AUC for cost prediction that is poorly calibrated (over- or under-predicts probabilities) compounds the equity problem by introducing an additional source of threshold-sensitivity.

4. **Health outcomes not reported.** The paper demonstrates selection bias (who gets enrolled) but not outcome bias (do enrolled patients benefit equally?). Even a perfectly equitable enrollment algorithm could produce differential outcomes if care management is less effective for Black patients due to structural barriers.

5. **Temporal analysis absent.** The study is cross-sectional. Whether bias has increased or decreased over time (as cost-need correlations change with policy environment) is unknown.

6. **Single-system generalizability.** The paper uses one large academic medical center + insurer data. Whether the $1,800/year access differential holds in integrated payer-provider systems (which control both sides of the cost-need relationship) is not addressed.

---

## Assumptions Made

| Assumption | Impact | Basis |
|------------|--------|-------|
| Top-risk tier = approximately top 5–10% of risk distribution | HIGH | Inferred from "high-need care management" context; not specified |
| Access gap is structurally stable (not confounded by other factors) | HIGH | Paper's causal argument; alternative explanations partially addressed |
| Black patients' health need is accurately measured by active conditions count | MEDIUM | Paper's operational definition; alternative need measures exist |
| Mechanism generalizes to the target health system's population | MEDIUM | the target health system is an integrated payer-provider; access gap dynamics may differ |
| AUC of debiased model is ≥0.70 (functional) | MEDIUM | Not reported; paper claims bias is reduced without performance loss |

---

## SDK Design Contribution

**This is the canonical equity audit scenario.** Obermeyer et al. should be the first scenario in the SDK's equity testing library, implemented as:

1. **`scenarios/proxy_bias_audit/`**: A scenario where `predict()` can be configured to use either:
   - `proxy_target = "cost"` (replicating the biased algorithm structure)
   - `proxy_target = "health_need"` (replicating the debiased alternative)
   
2. The SDK's equity audit module should implement **proxy variable detection** as a flagging mechanism: when a model is trained on a variable that is differentially accessible across demographic groups, the SDK should warn that cost/utilization proxies carry structural bias risk.

3. **The three fairness metrics** from Rajkomar et al. (Paper #21) should be computed against both algorithm configurations:
   - Demographic parity (equal enrollment rates regardless of need)
   - Equalized odds (equal TPR/FPR across groups)
   - Equal opportunity (equal TPR for Black vs. White patients with equivalent need)

4. **Health-system-specific calibration opportunity:** With access to the target health system's integrated claims + clinical data, the actual cost-need gap for Black patients in the target population (vs. Paper's ~$1,800/year) could be directly estimated, allowing more accurate simulation of what this algorithm would do in the target health system's population.

**Priority for SDK equity module:** CRITICAL — this paper motivated the equity audit requirement and provides the clearest numerical targets (17.7%→46.5%, 84% bias reduction) for SDK validation.
