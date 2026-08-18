# Paper 23: Ghassemi, Oakden-Rayner, Beam — The False Hope of Current Approaches to Explainable AI in Healthcare (2021)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Ghassemi, Oakden-Rayner, and Beam systematically critique post-hoc explainability methods (SHAP, LIME, saliency maps, attention weights) as unreliable tools for validating clinical AI, arguing that: (1) these methods often disagree with each other on the same model; (2) they do not expose clinically meaningful reasoning — they expose mathematical properties of the model; (3) XAI explanations can *increase* clinician overconfidence by providing a false sense of model transparency; and (4) rigorous external validation is both more informative and more protective than any post-hoc explanation. 

The paper's empirical core includes demonstration that SHAP/LIME attributions are unstable across perturbations and that XAI-exposed reasoning does not match clinical expert reasoning in chest X-ray classification. The policy implication is clear: explainability requirements in clinical AI governance should not be met by post-hoc explanation alone.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| XAI methods evaluated | SHAP, LIME, saliency maps, attention | Paper | Extracted |
| Agreement rate across XAI methods | Unreliable / inconsistent (no single number given) | Paper | Partial |
| Overconfidence effect | XAI increases overconfidence (no % given) | Paper | Partial |
| Study designs reviewed | Multiple (literature review + empirical demonstration) | Paper | Extracted |
| Clinical domain | General, with radiology examples | Paper | Extracted |
| Model N | Not applicable — framework/critique paper | N/A | — |
| AUC, effect size | Not applicable | N/A | — |

**Missing parameters for SDK:** No deployment, no intervention, no outcome, no population.

## Simulation Results

No simulation conducted. Paper is a methodological critique.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention; this is a methods critique |
| Predictive model drives intervention | ✗ FAIL | Models are evaluated for explanations, not interventions |
| Intervention is a state change | ✗ FAIL | No state change |
| Measurable outcome at each timestep | ✗ FAIL | No patient outcome measured |
| Counterfactual causal question | ✗ FAIL | No counterfactual deployment comparison |
| Discrete-time dynamics | ✗ FAIL | No temporal structure |

**0 of 6 criteria met → NO_FIT**

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

1. **No quantification of overconfidence effect size:** The claim that XAI increases overconfidence is compelling but would be stronger with a controlled experiment measuring clinician decision accuracy with vs. without XAI explanations.
2. **Agreement metrics between XAI methods not systematically quantified:** The instability claim is illustrated but not measured with a reproducible benchmark.
3. **No distinction between explanation fidelity and explanation utility:** A SHAP attribution can be technically faithful to the model while being clinically useless — the paper conflates these, which weakens the policy argument.
4. **No constructive path forward:** The paper argues against XAI but doesn't specify what rigorous validation *does* look like (left to the reader to infer: external validation, prospective cohort design, DCA).

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper is a **direct warning against building XAI into the SDK as a trust mechanism** and has concrete implications for how simulation outputs are presented.

**Concrete SDK requirements derived from this paper:**

1. **No post-hoc explanation modules:** The SDK should not include SHAP/LIME attribution as a feature for validating simulated model behavior. If the SDK's `ControlledMLModel` needs to be validated, the validation is: does it achieve the specified AUC, sensitivity, and PPV across subgroups? Not: does it explain its predictions correctly.

2. **Explicit anti-XAI documentation in SDK:** The SDK's design philosophy document should include a note: "We do not implement post-hoc explainability for the ML model simulator. Ghassemi et al. (2021) demonstrate that such methods do not reliably validate model reasoning and may increase user overconfidence. Validation is via statistical performance metrics, not attribution."

3. **Overconfidence mitigation in output presentation:** SDK simulation outputs should display uncertainty ranges prominently, not just point estimates, to counteract the overconfidence dynamic Ghassemi documents. When a simulation produces a mortality reduction estimate, it must be displayed with confidence intervals and sensitivity ranges, not as a single number.

4. **External validation priority in fitness assessment:** When classifying papers (Phase 2), the SDK documentation should cite Ghassemi when noting that a paper's internal validation with XAI is weaker evidence than external validation with held-out cohorts. This strengthens the rationale for Paper #24's (Park & Han) finding that only 6% of AI studies have external validation.

**Connection to other papers:**
- Directly pairs with Paper #24 (Park & Han) — low external validation rates mean that XAI is filling a gap it cannot fill.
- Connects to Paper #26 (Sendak Model Facts) — the Model Facts label addresses transparency through standardized performance metrics, not explanations.
- Connects to Paper #29 (FDA GMLP Principle 10) — good ML practice requires testing performance, not just documenting explanations.
