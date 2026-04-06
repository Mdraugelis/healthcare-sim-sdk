# Paper 28: Li, Akel, Shah — The Future of AI in Medicine: A Perspective from a Global Pediatric Academic Medical Center (2020) / AI Delivery Science

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Li, Akel, and Shah (Stanford, npj Digital Medicine 2020) argue that clinical AI implementation requires a new discipline — "AI delivery science" — that integrates design thinking, process improvement (Lean/Six Sigma), and implementation science. The core argument is that AI-enabled clinical tools should be understood as **new care delivery systems**, not as point tools dropped into existing workflows. This reframes the question from "does this model work?" to "does this system of care work, of which the AI is one component?"

The three-framework synthesis positions:
- **Design thinking:** Understanding user needs, iterative prototyping, workflow mapping before deployment
- **Process improvement:** Reducing waste, measuring system throughput, identifying bottlenecks (Coiera's middle and last mile in operational terms)
- **Implementation science:** Evidence-based frameworks (e.g., RE-AIM, CFIR) for adoption, fidelity, and sustainability

The paper is a perspective/opinion piece from Stanford Children's Health context, with no empirical study, no measured outcomes, and no specific AI system evaluated.

## Parameter Extraction

| Parameter | Value | Source | Status |
|---|---|---|---|
| Frameworks integrated | Design thinking, process improvement, implementation science | Paper | Extracted |
| AI framing | AI-enabled systems as new care delivery systems | Paper | Extracted |
| Implementation frameworks cited | RE-AIM, CFIR | Paper | Extracted |
| Setting | Pediatric academic medical center (Stanford Children's) | Paper | Extracted |
| Empirical study N | Not applicable — perspective piece | N/A | — |
| AUC, effect size, outcomes | Not applicable | N/A | — |
| Counterfactual | Not applicable | N/A | — |

**Missing parameters for SDK:** All operational parameters absent — perspective/opinion piece.

## Simulation Results

No simulation conducted. Paper is a perspective piece proposing a new discipline.

**SDK Fitness Criteria Assessment:**

| Criterion | Met? | Notes |
|---|---|---|
| Population-level intervention | ✗ FAIL | No intervention — framework paper |
| Predictive model drives intervention | ✗ FAIL | No specific model described |
| Intervention is a state change | ✗ FAIL | No state change defined |
| Measurable outcome at each timestep | ✗ FAIL | No outcome defined |
| Counterfactual causal question | ✗ FAIL | No counterfactual structure |
| Discrete-time dynamics | ✗ FAIL | No temporal dynamics |

**0 of 6 criteria met → NO_FIT**

## Verification Summary

Not applicable. No simulation was designed or executed.

## Discrepancies

Not applicable.

## Scientific Reporting Gaps

As a perspective piece, the relevant gaps are analytical:
1. **No implementation outcomes data:** The paper advocates for implementation science without reporting any implementation outcomes from Stanford Children's AI deployments.
2. **No adoption rates or fidelity metrics:** RE-AIM and CFIR are well-validated frameworks, but the paper doesn't apply them to any concrete example to demonstrate what good AI delivery science looks like in practice.
3. **No operational workflow maps:** Design thinking requires workflow mapping, but the paper doesn't include a workflow diagram for any AI system.
4. **No conflict between frameworks documented:** Design thinking (embrace ambiguity, prototype fast) and process improvement (reduce variance, standardize) have real tensions that the paper glosses over.
5. **Pediatric context not leveraged:** The pediatric setting has specific AI challenges (small populations, weight-based dosing, developmental variation) that the paper doesn't address, making the framing unnecessarily generic for a journal claiming to represent a global pediatric center.

## Assumptions Made

None — no simulation was run.

---

## SDK Design Rationale (Primary Contribution of This Paper)

This paper provides the **organizational and methodological framing for how SDK outputs should be used in Geisinger's deployment process**, rather than contributing directly to simulation design.

**Concrete SDK requirements derived from this paper:**

1. **"AI-enabled care delivery system" framing in SDK documentation:** The SDK README and Phase 3 scenario design template should adopt Li et al.'s framing: we are not simulating a model, we are simulating a *system of care*. The model is one component. The other components (clinician workflow, alert routing, patient population, time constraints) are equally important. This motivates why the SDK requires `intervene()`, `measure()`, and `step()` — not just `predict()`.

2. **Design thinking phase → Phase 0 (pre-extraction):** The SDK's workflow (AGENTS.md) could benefit from a Phase 0 before paper extraction: "What clinical problem is this solving? Who are the users? What is the workflow?" This ensures teams aren't simulating the right model for the wrong problem.

3. **Implementation science metrics as Phase 8 additions:** The synthesis report (Phase 8) should include RE-AIM dimensions for each paper: Reach (who gets the intervention?), Effectiveness (what is the effect?), Adoption (what fraction of eligible providers adopted it?), Implementation (fidelity to protocol?), Maintenance (sustained effects?). Most papers only report Effectiveness; the SDK synthesis should flag the other dimensions as standard gaps.

4. **Process improvement connection to threshold optimization:** Lean's "reduce waste" principle maps directly to threshold optimization — a threshold that generates 18% alert rate (Paper #1) is clinical waste. The SDK's threshold sweep analysis can be framed as a process improvement exercise: find the threshold that minimizes waste (false positives) while maintaining acceptable yield (sensitivity).

**Connection to other papers:**
- Paper #25 (Coiera): "Last mile" = implementation science's adoption and fidelity dimensions.
- Paper #3 (Kaiser/AAM): The Virtual Quality Nurse layer is a process improvement solution to the middle-mile problem.
- Paper #26 (Sendak Model Facts): The Model Facts label is a design thinking deliverable — it documents the system's intended use before deployment.
- Paper #29 (FDA GMLP): Principle 7 (human-AI team performance) operationalizes the "system of care" framing: it's not the model's AUC that matters, it's the team's performance.
