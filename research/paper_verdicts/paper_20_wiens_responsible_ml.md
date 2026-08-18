# Paper 20: Wiens, Saria, Sendak et al. — Responsible ML roadmap (Nature Medicine 2019)

## Classification: NO_FIT
## Reproducibility: N/A

---

## Key Findings

This multi-institutional consensus statement (Michigan, Johns Hopkins, Duke, MIT, Kaiser, Harvard, Stanford, Toronto) defines the responsible ML development lifecycle for healthcare: problem formulation → data collection → model development → validation → deployment → monitoring. It emphasizes multidisciplinary stakeholders, transparency, bias/fairness attention, prospective validation of clinical impact, and post-deployment monitoring. The paper reports no original clinical data, no model, no deployed intervention, and no patient outcomes. It is a normative framework paper. The SDK cannot simulate a framework. However, the Wiens et al. roadmap is the closest published analog to the SDK's own design philosophy — and every phase of their roadmap maps to a phase in the SDK's 8-phase processing workflow.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Study population | None (framework paper) | Patient entities | ABSENT |
| Base outcome rates | None | Population base rate | ABSENT |
| Model performance | None | Target AUC | ABSENT |
| Intervention | None | State-change action | ABSENT |
| Effect size | None | Absolute risk reduction | ABSENT |
| Equity analysis | Conceptual framework only | Demographic breakdown | ABSENT |
| Study design | N/A | RCT or quasi-experimental structure | ABSENT |

No simulatable parameters. Every element of this paper is normative guidance, not empirical data.

---

## Simulation Results

Not attempted. Classification: NO_FIT.

Wiens et al. describe how healthcare ML should be done. The SDK is an implementation of that description. Simulating the framework paper would be circular — the SDK is already the simulation of their recommendations.

---

## Verification Summary

No simulation run. SDK fitness criteria:
- [ ] Population-level intervention: Not applicable (framework paper)
- [ ] Predictive model drives intervention: Not applicable
- [ ] Intervention is a state change: Not applicable
- [ ] Measurable outcome at each timestep: Not applicable
- [ ] Counterfactual causal question: Not applicable
- [ ] Discrete-time dynamics: Not applicable

**0 of 6 criteria met → NO_FIT.**

---

## Discrepancies

None applicable.

**One structural observation:** The Wiens et al. roadmap identifies monitoring as a mandatory post-deployment phase, but in practice monitoring is rarely implemented rigorously (see Paper #1: Epic's ESM was deployed broadly without systematic outcome monitoring at external sites). The SDK's counterfactual simulation framework could serve as a prospective monitoring tool — running the simulation against incoming real-world data to detect when deployed model performance diverges from simulated expectations. This is a use case beyond the paper's scope but directly implied by its monitoring requirement.

---

## Scientific Reporting Gaps

This paper is a consensus framework, not an empirical study. "Gaps" mean underspecified framework elements:

1. **Monitoring criteria are not operationalized.** Wiens et al. recommend monitoring but do not specify: monitoring frequency, which metrics to track, statistical thresholds for retraining triggers, or minimum sample sizes for meaningful drift detection. The SDK's `step()` function models temporal dynamics that could provide these operational specifics.

2. **Equity evaluation timing is underspecified.** The roadmap addresses bias/fairness but doesn't specify whether equity evaluation should occur at development, validation, or deployment — or all three. The Obermeyer finding (Paper #15) demonstrates that a model can appear fair in development but produce inequitable outcomes in deployment. The SDK's equity audit should run at each phase.

3. **Problem formulation feedback is one-directional.** The roadmap flows linearly from formulation to deployment. In practice, deployment reveals problems that require reformulating the problem (as Obermeyer's target reformulation demonstrates). The SDK's iterative simulation capability supports this feedback loop but the paper doesn't explicitly model it.

4. **Stakeholder engagement is described but not operationalized.** "Multidisciplinary teams" is recommended but the paper doesn't specify who must be present or what decisions require clinical vs. technical authority. The FDA GMLP principles (Paper #29) later operationalized some of this through the concept of human-AI team performance evaluation.

5. **No failure mode taxonomy.** The roadmap identifies phases where things can go wrong but doesn't catalog specific failure modes at each phase. Papers in this pipeline provide that taxonomy empirically (AUC degradation at external sites, calibration failures, proxy bias, alert fatigue). The SDK's verdict library is building exactly this taxonomy.

---

## Assumptions Made

None — no simulation attempted.

---

## SDK Design Contribution

**The Wiens et al. roadmap is the closest published external specification for the SDK's 8-phase workflow.** The alignment is direct:

| Wiens et al. Phase | SDK Phase | Notes |
|-------------------|-----------|-------|
| Problem formulation | Phase 1 (Extract) + Phase 2 (Fitness) | SDK adds formal fitness criteria |
| Data collection | Phase 1 (Parameter Extraction) | SDK works from published parameters, not raw data |
| Model development | Phase 3 (Design) → `predict()` | SDK uses `ControlledMLModel`, not raw training |
| Validation | Phase 4 (Calibrate) + Phase 5 (Verify) | SDK adds 4-level verification protocol |
| Deployment | Phase 6 (Reproduce) | SDK simulates deployment counterfactually |
| Monitoring | Phase 8 (Synthesize) + ongoing scenarios | SDK could support ongoing monitoring |
| Equity/bias | Every phase → equity audit | SDK makes equity non-optional |

**The key SDK contribution beyond Wiens et al.:**
Wiens et al. recommend prospective clinical validation as the gold standard. This is correct but expensive and slow. The SDK provides a computational intermediate step: pre-deployment simulation that can reject obviously harmful configurations, quantify uncertainty, and inform prospective trial design — before any patient is exposed. This is the gap between the roadmap and its implementation.

### Specific SDK Implementation Items Derived from This Paper

1. **Pre-deployment checklist gate (Phase 2):** Before accepting a paper as FIT, the SDK should verify that the source study completed at minimum:
   - Problem formulation with measurable outcome
   - External validation (not same-dataset evaluation)
   - Bias/fairness assessment
   - Clinical stakeholder involvement
   
2. **Post-deployment monitoring template:** The SDK should output a monitoring plan alongside each simulation:
   - Which metrics to track in production (AUC, calibration, alert rate, demographic parity)
   - Statistical thresholds triggering review
   - Recommended monitoring frequency

3. **Deployment readiness score:** A composite score (0–10) based on:
   - Study design quality (Nagendran criteria, Paper #19)
   - Reporting completeness (CONSORT-AI, TRIPOD)
   - Calibration reporting (Van Calster, Paper #18)
   - Equity analysis (Obermeyer, Paper #15)
   - External validation presence (Park, Paper #24 in pipeline)
   - Effect size estimate from simulation

   This score could be a standardized SDK output, supporting health system procurement decisions.

**Cross-paper synthesis:** Wiens et al. (Paper #20) + Plana et al. (Paper #12) + Nagendran et al. (Paper #19) together form the evidence base for why pre-deployment simulation is not just useful but necessary:
- Plana: 0% of RCTs met all reporting standards (supply problem)
- Nagendran: 93% of studies are retrospective/non-deployed (evidence quality problem)  
- Wiens: The full roadmap is specified but rarely followed (implementation problem)

The SDK addresses all three gaps simultaneously: it enforces reporting standards (Plana), requires prospective-equivalent evaluation through simulation (Nagendran), and implements the full roadmap computationally (Wiens).

**Priority for SDK documentation:** HIGH — Wiens et al. should be cited in the SDK's design rationale document as the external specification that the SDK operationalizes.
