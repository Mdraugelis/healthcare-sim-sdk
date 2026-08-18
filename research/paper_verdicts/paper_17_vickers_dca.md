# Paper 17: Vickers & Elkin — Decision Curve Analysis (Med Decision Making 2006)

## Classification: NO_FIT
## Reproducibility: N/A

---

## Key Findings

Decision curve analysis (DCA) introduces net benefit — Net Benefit = (TP/N) − (FP/N) × (p_t / (1−p_t)) — as the metric that bridges prediction model performance and clinical utility, by incorporating clinician threshold preferences (p_t) directly into the evaluation. Plotting net benefit against threshold probability reveals the range of preferences at which a given model outperforms "treat all" or "treat none" baselines. This is a methods paper: it introduces and validates a statistical framework. It reports no clinical population, no ML model deployment, and no patient outcomes — only a demonstration of the DCA methodology applied to a prostate cancer decision example. The SDK cannot simulate a methods paper's illustrative example as a healthcare intervention. However, the DCA framework is directly implemented as the SDK's net benefit calculation and threshold sweep module.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Study population | Illustrative (prostate biopsy decisions) | Discrete patient entities with outcomes | ILLUSTRATIVE ONLY |
| Base outcome rate | ~25% (prostate cancer in biopsy population, illustrative) | Population base rate | ILLUSTRATIVE ONLY |
| Model AUC | Not specified for the illustrative model | Target AUC | ABSENT |
| Net benefit formula | NB = TP/N − FP/N × (p_t / 1−p_t) | SDK formula specification | FULLY AVAILABLE |
| Treat-all baseline | NB_all = prevalence − (1−prevalence) × (p_t / 1−p_t) | Comparator curve | FULLY AVAILABLE |
| Treat-none baseline | NB_none = 0 for all p_t | Comparator curve | FULLY AVAILABLE |
| Threshold probability range | p_t ∈ (0, 1), plotted at clinical decision points | Sweep range | FULLY AVAILABLE |
| Weighted harms ratio | Exchange rate: w = p_t / (1−p_t) | FP penalty weight | FULLY AVAILABLE |
| Intervention | Prostate biopsy (illustrative) | Not needed for methods paper | N/A |
| Counterfactual | Treat-all / treat-none comparators | SDK comparator arms | DERIVED |

**The paper's contribution is the formula and framework, not empirical parameters.** No simulation parameters are missing — they are simply not applicable to a methods paper.

---

## Simulation Results

Not attempted. Classification: NO_FIT.

The DCA framework is a **calculation method**, not a deployment scenario. The SDK does not simulate papers that describe analytical tools — it uses those tools. Implementing DCA in the SDK's output module is the correct response to this paper, not running a simulation of the prostate biopsy illustration.

**The SDK already implements this:** Per TOOLS.md, the SDK's threshold optimization module incorporates decision curve analysis. This paper is the specification for that module.

---

## Verification Summary

No simulation run. SDK fitness criteria:
- [ ] Population-level intervention: Not applicable (methods paper)
- [ ] Predictive model drives intervention: Not applicable
- [ ] Intervention is a state change: Not applicable
- [ ] Measurable outcome at each timestep: Not applicable
- [ ] Counterfactual causal question: Not applicable (DCA defines the counterfactual comparators)
- [ ] Discrete-time dynamics: Not applicable

**0 of 6 criteria met → NO_FIT for simulation; fully applicable as SDK specification.**

---

## Discrepancies

None applicable.

**One technical note worth preserving:** The paper demonstrates that a model can achieve positive net benefit in a range of threshold probabilities where treat-all is better than treat-none, but that within that range, using a model (even an imperfect one) improves on treat-all. This is the mathematical basis for why a model with AUC 0.63 (Epic ESM, Paper #1) can still theoretically provide clinical value at specific threshold probabilities — just not at the deployed threshold. The SDK's DCA output for Paper #1's scenario should make this explicit.

---

## Scientific Reporting Gaps

This paper is foundational and methodologically sound. "Gaps" are evolutionary, not deficiencies:

1. **Temporal extension not addressed.** Standard DCA assumes a single decision point. The SDK extends to multiple timesteps (discrete-time simulation), where net benefit accumulates across time. The correct extension is to compute net benefit per-timestep and aggregate — a methodology not defined in Vickers 2006 but required for any longitudinal simulation.

2. **Multi-intervention DCA not addressed.** The original paper handles one intervention (treat vs. not treat). The SDK models complex interventions with intermediate steps (alert → clinician decision → treatment compliance → outcome). Net benefit in this setting should be computed at the population level, accounting for alert fatigue attenuation — an extension Vickers and colleagues addressed in later work (Vickers et al., 2019) but not here.

3. **Equity-weighted DCA not addressed.** Standard DCA treats all patients as equivalently weighted. A disparity-aware extension would weight net benefit by demographic parity: a model that benefits high-risk White patients but misses equivalent Black patients has lower "equity-adjusted net benefit" than one that performs uniformly. This extension is not in the literature and would be an original SDK contribution.

---

## Assumptions Made

None — no simulation attempted.

---

## SDK Design Contribution

**This paper is the formal specification for the SDK's threshold optimization output.** Every scenario should produce:

1. **Decision curve plot**: Net benefit vs. threshold probability (p_t from 0.01 to 0.50) for:
   - The simulated intervention model
   - Treat-all baseline
   - Treat-none baseline (NB = 0)

2. **Net benefit formula implementation:**
   ```python
   def net_benefit(tp, fp, n, p_t):
       return (tp / n) - (fp / n) * (p_t / (1 - p_t))
   
   def treat_all_net_benefit(outcome_rate, p_t):
       return outcome_rate - (1 - outcome_rate) * (p_t / (1 - p_t))
   ```

3. **Clinical utility range:** Report the range of p_t values where the model outperforms both treat-all and treat-none.

4. **Threshold recommendation:** Report the p_t that maximizes net benefit, with uncertainty bounds from the simulation's stochastic replicates.

**Implementation note:** The SDK's threshold sweep should sweep both the model score threshold AND p_t independently — these are distinct parameters. The score threshold determines who receives the intervention (operational); p_t determines the harm-benefit weighting (analytical). Conflating them is a common error.

**Cross-paper relevance:** 
- Paper #1 (Wong/Epic ESM): DCA would show that at threshold p_t where most clinicians operate, AUC 0.63 yields net benefit near zero or negative.
- Paper #2 (TREWS): DCA would show positive net benefit at the published threshold (sensitivity 0.8, PPV 0.27).
- Paper #15 (Obermeyer): Equity-adjusted DCA would show that cost-proxy selection has negative equity-adjusted net benefit even if aggregate net benefit is positive.

**Priority for SDK implementation:** CRITICAL — DCA should be a mandatory output of every simulation run, as required by JAMA, BMJ, and Annals of Internal Medicine reporting standards.
