# Paper 25: Coiera — The Last Mile: Where Artificial Intelligence Meets Reality (2019)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Coiera introduces a three-stage framework for AI system failure: the **first mile** (data quality and model development), the **middle mile** (clinical workflow integration and alert design), and the **last mile** (the point where AI output meets clinician decision-making and patient behavior). He argues that most AI investment targets the first mile while most deployment failures occur in the last mile.

The paper's most counterintuitive claim — which Coiera calls a "paradox of success" — is that **AI-driven performance improvement can cause performance degradation**: when an AI system correctly reduces a problem's incidence (e.g., fewer sepsis cases because early detection improved), the remaining cases become harder (the model was good at catching easy ones) and clinician skills atrophy from reduced exposure to the full disease spectrum. This is an adversarial feedback loop that static validation cannot detect.

Coiera also formalizes AUC as a **necessary but not sufficient** condition for clinical impact. High AUC is required but cannot predict whether the model will change behavior, whether behavior change will improve outcomes, or whether the workflow can absorb the alert volume.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Framework stages | 3: first mile, middle mile, last mile | Paper | Extracted |
| "Paradox of success" mechanism | Performance improvement → harder cases remain → skill atrophy | Paper | Extracted |
| AUC claim | Necessary but not sufficient for clinical impact | Paper | Extracted |
| Alert fatigue context | Referenced as key last-mile barrier | Paper | Extracted |
| Empirical study N | Not applicable — framework paper | N/A | — |
| Model AUC, effect size | Not applicable | N/A | — |
| Counterfactual comparison | Not applicable | N/A | — |

**Missing parameters for SDK:** No deployment, no intervention, no measured patient outcomes.

## Simulation Results

No simulation conducted. Paper is a theoretical/analytical framework.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No specific intervention defined |
| Predictive model drives intervention | ✗ FAIL | Framework describes AI systems generically |
| Intervention is a state change | ✗ FAIL | No state change defined |
| Measurable outcome at each timestep | ✗ FAIL | No quantitative outcome defined |
| Counterfactual causal question | ✗ FAIL | Conceptual, not counterfactual |
| Discrete-time dynamics | ~ PARTIAL | The paradox-of-success dynamic is inherently temporal but not formalized |

**0–1 of 6 criteria met → NO_FIT**

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

1. **Paradox of success is not empirically demonstrated:** This is a compelling theoretical argument but Coiera does not cite a study where this feedback loop was measured in practice. The SDK could test this computationally.
2. **No quantification of last-mile vs. first-mile failure rate:** The claim that most failures are last-mile is asserted without a systematic review.
3. **No operationalized last-mile metrics:** "Clinician decision-making" is not defined precisely enough to be measured in an implementation study.
4. **Skill atrophy mechanism unformalized:** The skill atrophy component of the paradox of success requires a model of clinician learning and forgetting — Coiera identifies the phenomenon but provides no parameterization.

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper provides the **theoretical motivation for the SDK's most distinctive feature: branched counterfactual simulation.** It also suggests a novel simulation experiment (paradox of success) that no existing SDK scenario implements.

**Concrete SDK requirements derived from this paper:**

1. **Three-mile failure taxonomy in scenario design (Phase 3):** Every scenario design document should classify which mile(s) the scenario is primarily testing:
   - First mile: Model AUC sensitivity analysis
   - Middle mile: Alert delivery latency, threshold optimization, alert volume
   - Last mile: Clinician response rate, alert fatigue (linked to Paper #30), compliance degradation

2. **Paradox-of-success simulation experiment (new scenario type):** Coiera's adversarial feedback loop should be implemented as a multi-period SDK scenario where: (a) the model correctly reduces disease incidence in period 1; (b) remaining cases in period 2 are drawn from a harder distribution (higher base acuity, lower model sensitivity); (c) clinician skill degrades as a function of reduced disease exposure (skill atrophy parameter). This is novel SDK science — no existing paper implements it. It would be a publishable SDK validation experiment.

3. **AUC is necessary-not-sufficient → SDK gate:** The SDK should display a hard warning when a user attempts to evaluate a model with AUC < 0.65: "Model discrimination is below minimum threshold for clinical impact (Coiera 2019). Simulation may produce technically valid results that are clinically irrelevant." This operationalizes Coiera's claim without being prescriptive.

4. **Last-mile clinician response model:** The SDK's `intervene()` method already supports compliance rates. This should be documented explicitly as a "last-mile" parameter: `clinician_response_rate` (0–1), which degrades as a function of false-positive volume (linked to Paper #30 alert fatigue data).

**Connection to other papers:**
- Paper #1 (Wong et al.): AUC 0.63 external = first-mile failure confirmed; 18% alert rate = middle-mile failure; 33% sensitivity = last-mile failure (clinicians can't act on noise).
- Paper #2 (Adams/TREWS): The alert confirmation step (clinician verifies alert within 3h) is a last-mile design solution.
- Paper #3 (Kaiser/AAM): Virtual Quality Nurse layer is an explicit middle-to-last-mile bridge.
- Paper #30 (Hussain alert fatigue): Quantifies the last-mile degradation parameters Coiera describes qualitatively.
- The paradox-of-success scenario would be a first-of-its-kind SDK contribution.
