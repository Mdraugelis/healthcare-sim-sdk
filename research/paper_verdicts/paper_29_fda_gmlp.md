# Paper 29: FDA/Health Canada/MHRA — Good Machine Learning Practice for Medical Device Development: Guiding Principles (2021)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

The FDA (in collaboration with Health Canada and the UK MHRA) published ten guiding principles for Good Machine Learning Practice (GMLP) in medical device development, released October 2021. This is regulatory guidance — not a research paper — and represents the joint position of three major regulatory bodies on what constitutes responsible AI/ML development in healthcare.

The ten principles are:
1. Multi-disciplinary expertise throughout product lifecycle
2. Good software engineering and security practices
3. Clinical study participants and data reflect intended use population
4. Training data independence from test data
5. Model selection reflects intended use and clinical environment
6. Focus on task-appropriate performance
7. **Focus on human-AI team performance** (not model-alone performance)
8. **Test under clinically relevant conditions** (not just benchmark datasets)
9. Users receive clear, essential information about the AI/ML device
10. Deployed models are monitored with managed updates

Principles 7 and 8 are particularly relevant to the SDK: they mandate evaluating the combined performance of the human-AI team in realistic conditions, which is precisely what pre-deployment simulation provides. As of December 2024, FDA has approved **1,016 AI/ML medical devices**.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Regulatory bodies | FDA (US), Health Canada, MHRA (UK) | Document | Extracted |
| Publication date | October 2021 | Document | Extracted |
| Number of principles | 10 | Document | Extracted |
| FDA-approved AI/ML devices (Dec 2024) | 1,016 | USER.md | Extracted |
| Human-AI team performance | Principle 7 — evaluate team, not model alone | Document | Extracted |
| Clinically relevant testing | Principle 8 — test under operational conditions | Document | Extracted |
| Post-deployment monitoring | Principle 10 — managed updates, performance tracking | Document | Extracted |
| Empirical study N | Not applicable — regulatory guidance | N/A | — |
| AUC, effect size, outcome | Not applicable | N/A | — |

**Missing parameters for SDK:** All operational parameters absent — regulatory guidance document.

## Simulation Results

No simulation conducted. Regulatory guidance documents do not describe deployable interventions with measurable outcomes.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention — guidance document |
| Predictive model drives intervention | ✗ FAIL | No specific model |
| Intervention is a state change | ✗ FAIL | No state change |
| Measurable outcome at each timestep | ✗ FAIL | No outcome |
| Counterfactual causal question | ✗ FAIL | No counterfactual |
| Discrete-time dynamics | ✗ FAIL | No temporal structure |

**0 of 6 criteria met → NO_FIT**

**Classification note:** Regulatory guidance documents are structurally incompatible with the SDK's simulation framework. This is not a deficiency of the guidance — it is the correct tool for its purpose. The SDK is the computational implementation of what these principles require in practice.

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

These are not gaps in a conventional scientific sense — this is regulatory guidance, and incompleteness is expected (principles require interpretation). Noteworthy underspecifications:

1. **Principle 7 (human-AI team performance) is undefined operationally:** "Human-AI team performance" is mandated but not defined. What metric? Measured how? In which study design? The guidance opens the door for simulation-based pre-deployment evaluation but does not specify it.
2. **No minimum AUC or performance standards:** The guidance is principles-based, not prescriptive. A model with AUC 0.55 technically complies if it's transparent about its limitations.
3. **Principle 10 (post-deployment monitoring) lacks triggers:** No guidance on what performance degradation threshold requires a label update or model withdrawal.
4. **No equity/fairness principle:** Among the 10 principles, there is no explicit requirement for subgroup performance evaluation or demographic fairness reporting. This is a significant omission given Rajkomar (Paper #21) and Chen (Paper #22).

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This regulatory document provides the **compliance framework** that the SDK should be designed against. Every SDK scenario, when used by a Geisinger team evaluating a vendor AI product, should generate output that directly addresses each of the 10 GMLP principles.

**Concrete SDK requirements derived from each relevant principle:**

**Principle 3 (intended use population):** Phase 1 extraction must capture training population demographics and compare to the target deployment population (Geisinger's ~85% White, ~5% Black, ~6% Hispanic demographics in Central PA). The SDK should flag demographic mismatches as a compliance risk.

**Principle 7 (human-AI team performance):** This is the most direct regulatory mandate for simulation. The SDK's `intervene()` method, `clinician_response_rate` parameter, and branched counterfactual framework collectively operationalize Principle 7. SDK outputs should be labeled: "This simulation evaluates human-AI team performance under [specified conditions] per FDA GMLP Principle 7."

**Principle 8 (clinically relevant conditions):** The SDK's `create_population()` method, when calibrated to Geisinger's actual patient demographics and base rates, produces the "clinically relevant conditions" Principle 8 requires. This is the core justification for local SDK calibration: a nationally validated model must be re-evaluated against local conditions.

**Principle 9 (clear information to users):** Operationalized by the Sendak Model Facts label (Paper #26). The SDK's auto-generated `model_facts_simulated.md` output directly addresses Principle 9.

**Principle 10 (post-deployment monitoring):** The SDK's longitudinal equity drift analysis (motivated by Chen, Paper #22) addresses monitoring for performance degradation. The SDK should include a "monitoring plan" output that specifies: what metrics to track, at what frequency, with what alert thresholds.

**GMLP compliance scorecard output:** After each simulation run, the SDK should generate a `gmlp_compliance.md` that maps simulation outputs to each of the 10 principles with a status (addressed / not addressed / partially addressed). This transforms the SDK from a research tool into a regulatory-readiness assessment tool — high value for Geisinger procurement.

**Connection to other papers:**
- The 1,016 FDA-approved AI/ML devices (Dec 2024) is the market context: Principle 7 applies to all of them, but only a fraction will have been evaluated under clinically relevant conditions (Paper #24 shows 0% meet all three design criteria).
- Principle 7 (human-AI team) = Paper #25 (Coiera last-mile) in regulatory language.
- Principle 9 (clear information) = Paper #26 (Sendak Model Facts) in implementation language.
- The missing equity principle in GMLP is addressed by Papers #21 and #22 and should be treated as a Geisinger-specific additional requirement above the FDA baseline.
