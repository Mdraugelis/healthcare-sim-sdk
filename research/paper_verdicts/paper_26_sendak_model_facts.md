# Paper 26: Sendak et al. — "The Real World Is Not a Test Set": A Framework for the Evaluation and Implementation of Clinical Decision Support Systems (2020)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Sendak et al. propose the **Model Facts label** — a standardized one-page disclosure document analogous to a Drug Facts label — for clinical AI/ML models. The paper documents the deployment of Duke's Sepsis Watch system (a deep learning model for sepsis prediction, AUC 0.87 on held-out data) and the lessons learned in making that deployment transparent to clinicians, administrators, and patients.

The Model Facts label includes: intended use, target population, training population demographics, model performance (AUC, PPV, sensitivity, specificity), performance by subgroup, known limitations, and update schedule. The paper argues that without standardized disclosure, clinicians cannot make informed decisions about when to trust or override AI recommendations, and institutions cannot systematically compare models.

The Sepsis Watch system itself (AUC 0.87, deployed at Duke) is a concrete deployment, but the paper's primary contribution is the disclosure framework. The empirical content is the construction and operationalization of the label, not a clinical outcomes study.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Model system | Duke Sepsis Watch (deep learning) | Paper | Extracted |
| Sepsis Watch AUC | 0.87 (held-out validation) | Paper | Extracted |
| Label components | Intended use, population, training demographics, AUC, PPV, sensitivity, specificity, subgroup performance, limitations, update schedule | Paper | Extracted |
| Label analog | FDA Drug Facts label | Paper | Extracted |
| Effect size (outcomes) | Not reported — no clinical outcomes study | N/A | Missing |
| Intervention effectiveness | Not reported | N/A | Missing |
| Control arm | Not applicable — no comparison group for label evaluation | N/A | — |
| Prospective vs. retrospective | Retrospective validation; implementation description | Paper | Extracted |

**Missing parameters for SDK:** No clinical outcomes reported, no intervention effectiveness measured, no control arm. The Sepsis Watch AUC of 0.87 is extractable but cannot be connected to mortality reduction without a clinical impact study.

## Simulation Results

No simulation conducted. Paper is a framework/disclosure proposal with a single-site case study.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ~ PARTIAL | Sepsis Watch is a population-level alert system, but the paper does not evaluate it as an intervention |
| Predictive model drives intervention | ~ PARTIAL | Sepsis Watch uses a predictive model, but intervention effectiveness is not measured |
| Intervention is a state change | ✗ FAIL | No state change (outcomes) measured in this paper |
| Measurable outcome at each timestep | ✗ FAIL | No clinical outcomes reported |
| Counterfactual causal question | ✗ FAIL | No pre/post or control comparison for patient outcomes |
| Discrete-time dynamics | ✗ FAIL | No temporal outcome structure |

**2 of 6 criteria partially met; 0 fully met → NO_FIT**

Note: Sepsis Watch as a system might FIT the SDK if a companion clinical impact paper were available. This paper alone cannot support a simulation because it reports only technical validation, not clinical outcomes.

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

1. **No clinical outcomes data:** The Model Facts label mandates performance disclosure but the paper itself doesn't demonstrate what happened to patients after Sepsis Watch was deployed. This is the central gap.
2. **No label compliance data:** Were labels actually read by clinicians? Did they change behavior or trust calibration?
3. **No subgroup performance in the case study:** The label template requires subgroup performance, but the paper doesn't report whether Duke's Sepsis Watch met its own subgroup disclosure requirements.
4. **No guidance on update triggers:** When should a Model Facts label be updated? The paper proposes an "update schedule" field but gives no criteria for what triggers a label revision.
5. **Single-site case study:** Duke Sepsis Watch is one deployment. The paper does not demonstrate whether the label format generalizes to different model types (radiology vs. NLP vs. tabular EHR).

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper provides the **specification for required metadata in every SDK scenario** and is the most directly actionable design input of the batch.

**Concrete SDK requirements derived from this paper:**

1. **Model Facts as mandatory scenario metadata:** Every SDK scenario must include a `model_facts.yaml` alongside the Hydra config, containing all fields from the Sendak label:
   - `intended_use`: Clinical context, target population
   - `training_population`: N, demographics, institution(s), time period
   - `performance`: AUC, sensitivity, specificity, PPV at deployed threshold
   - `subgroup_performance`: Performance broken out by race, sex, age band at minimum
   - `known_limitations`: What the model doesn't do, populations excluded from training
   - `update_schedule`: When performance was last checked, when it will be rechecked

2. **SDK output includes auto-generated Model Facts:** After a simulation run, the SDK should output a `model_facts_simulated.md` that fills in the Model Facts template from the simulation parameters, making it easy for health system teams to draft a label for a real deployment by running the SDK first.

3. **Missing training demographics → required warning:** If a paper does not report training population demographics (a required Model Facts field), Phase 1 parameter extraction must flag this as a HIGH-impact gap. This connects to Paper #22 (Chen) on upstream data equity and Paper #21 (Rajkomar) on fairness definitions.

4. **AUC alone is insufficient disclosure:** The Model Facts label requires PPV at the deployed threshold — not just AUC. This operationalizes Paper #25 (Coiera's AUC as necessary-not-sufficient). SDK scenario configs must require both `auc` and `ppv_at_threshold` fields; if PPV is missing, it must be computed from the simulation and flagged as a simulation-derived estimate.

**Connection to other papers:**
- Paper #1 (Wong/Epic): The ESM never had a Model Facts label; clinicians didn't know AUC was 0.76–0.83 internally or that external validation had never been published.
- Paper #21 (Rajkomar fairness): Model Facts subgroup performance field operationalizes the fairness reporting requirements.
- Paper #29 (FDA GMLP): Principle 5 (make users aware of capabilities and limitations) is operationalized by the Model Facts label.
- Paper #23 (Ghassemi XAI): Model Facts replaces post-hoc explanation with transparent performance disclosure — a more defensible trust mechanism.
