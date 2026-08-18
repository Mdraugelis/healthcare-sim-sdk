# Paper 22: Chen, Pierson, Rose et al. — Ethical Machine Learning in Healthcare (2021)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Chen et al. map the ethical failure modes of clinical ML across five pipeline stages — problem formulation, data collection, model development, deployment, and monitoring — and argue that most bias enters at the problem formulation stage, not the modeling stage. The paper quantifies the 10/90 gap: 10% of diseases account for 90% of NIH research funding. The sickle cell vs. cystic fibrosis funding disparity is particularly stark: $2,094 vs. $6,972 NIH funding per affected individual (3.4×), despite sickle cell affecting ~100,000 Americans and cystic fibrosis ~30,000. These structural funding inequities upstream of any model mean that training data for underserved populations is systematically sparse before any algorithmic choices are made.

The paper's central contribution is a structured taxonomy of ethical risks with stage-specific mitigations. Like Paper 21, this is a normative and analytical framework, not an empirical deployment study.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Pipeline stages | 5: problem formulation, data, modeling, deployment, monitoring | Paper | Extracted |
| 10/90 funding gap | 10% of diseases, 90% of NIH funding | Paper | Extracted |
| Sickle cell NIH funding/patient | ~$2,094 per affected individual | Paper | Extracted |
| Cystic fibrosis NIH funding/patient | ~$6,972 per affected individual | Paper | Extracted |
| Funding disparity ratio | 3.4× (CF vs SCD) | Paper | Extracted |
| SCD US prevalence | ~100,000 | Paper | Extracted |
| CF US prevalence | ~30,000 | Paper | Extracted |
| Model N | Not applicable — framework paper | N/A | — |
| AUC, effect size | Not applicable | N/A | — |

**Missing parameters for SDK:** All operational deployment parameters absent.

## Simulation Results

No simulation conducted. Paper is a multi-stage ethical analysis framework.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention defined |
| Predictive model drives intervention | ✗ FAIL | No specific model |
| Intervention is a state change | ✗ FAIL | No state change |
| Measurable outcome at each timestep | ✗ FAIL | No temporal outcome |
| Counterfactual causal question | ✗ FAIL | No counterfactual structure |
| Discrete-time dynamics | ✗ FAIL | No temporal dynamics |

**0 of 6 criteria met → NO_FIT**

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

As a framework paper, the relevant gaps are analytical omissions:
1. **No quantitative guidance on stage-specific bias magnitude:** The paper identifies where bias enters but does not quantify how much bias per stage across its reviewed studies.
2. **No cross-condition training data availability estimates:** The funding disparity argument would be stronger with empirical estimates of training dataset sizes by disease area.
3. **No prioritization scheme:** When multiple ethical concerns conflict (e.g., data minimization vs. subgroup fairness requiring more data), the paper offers no tiebreaker framework.
4. **Monitoring stage underspecified:** The fifth pipeline stage (monitoring) receives the least treatment despite being where most deployed-model failures are detected.

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper provides the **conceptual scaffolding for SDK equity audit design** at the scenario design phase (Phase 3) and is particularly relevant to the `create_population()` method.

**Concrete SDK requirements derived from this paper:**

1. **Stage 1 — Problem formulation audit (Phase 1 addition):** Every paper extraction should now include: "What disease/condition is being modeled? What is the known research funding density for this condition? Is this condition concentrated in underserved populations?" This prevents the SDK from laundering upstream funding inequities through technically neutral simulation.

2. **Training demographics as a required field:** The 10/90 funding gap means that EHR-trained models for low-prevalence-in-research conditions will be trained on data that systematically underrepresents affected populations. The SDK's `create_population()` documentation should flag this when a user doesn't provide training demographics (linking to Sendak et al., Paper #26, which requires training demographics on the Model Facts label).

3. **Monitoring stage → continuous equity monitoring:** The SDK's Phase 6 (reproduce) should include a longitudinal equity drift check: does the simulated equity gap widen over time as population distributions shift? The five-stage pipeline's monitoring gap motivates this.

4. **10/90 gap as a base-rate uncertainty multiplier:** When a scenario models a condition with sparse training data (low NIH funding density, underrepresented population), the SDK should apply wider confidence intervals on model performance estimates for subgroups.

**Connection to other papers:** 
- Directly extends Paper #21 (Rajkomar fairness) by locating bias before the model exists.
- Directly motivates Paper #15 (Obermeyer) finding — Medicaid-enrolled Black patients are systematically less sick in the training data because they receive less care, not because they are healthier.
- The monitoring gap connects to Paper #25 (Coiera last-mile) and Paper #29 (FDA GMLP Principle 9: post-deployment monitoring).
