# Paper 10: Beede et al. — Google DR/ARDA (CHI 2020)

## Classification: NO_FIT
## Reproducibility: N/A

## Key Findings

Beede et al. is the most important paper in our batch for what it demonstrates about the SDK's scope boundaries. This is not a trial — it is a human-centered evaluation of a deployed AI system using mixed methods (semi-structured interviews, observation, system logs). The findings are qualitative or operational: nurses frustrated by workflow disruptions, 21% image rejection rate, connectivity failures causing multi-hour delays, and referral burden leading staff to actively discourage patient enrollment. These findings are real and clinically important, but they are not expressible in the SDK's 5-method contract.

The paper's core contribution is precisely what the SDK *cannot* model without substantial extension: the social, organizational, and infrastructural determinants of AI deployment failure. You cannot write a `step()` function for "trust erodes as nurses experience repeated connectivity failures."

**No simulation run.** This paper is a NO_FIT finding about the SDK's scope. The SDK models quantitative intervention effects; Beede et al. documents why quantitative effects never materialized.

## Parameter Extraction

| Parameter | Source | Value | Status |
|-----------|--------|-------|--------|
| Study design | Paper | Mixed-methods deployment evaluation | CONFIRMED (not an RCT) |
| Setting | Paper | 11 Thai primary care clinics | CONFIRMED |
| AI system | Paper | Google ARDA (diabetic retinopathy detection) | CONFIRMED |
| Lab-setting accuracy | Paper | 94.7% | CONFIRMED |
| Field image rejection rate | Paper | ~21% | CONFIRMED (ungradeable images rejected) |
| Primary outcome | Paper | None specified — qualitative | NO QUANTITATIVE PRIMARY ENDPOINT |
| Patient throughput | NOT REPORTED | Unclear | MISSING |
| Referral rate pre/post | NOT REPORTED | Unclear | MISSING |
| DR detection rate in field | NOT REPORTED | Unclear | MISSING |
| Connectivity failure rate | Paper (partial) | Frequent | QUALITATIVE ONLY |
| Nurse satisfaction | Paper | Low | QUALITATIVE ONLY |
| Model AUC | NOT REPORTED (in this paper) | 94.7% accuracy (not AUC) | MISSING AUC |
| Counterfactual (pre-AI) DR rate | NOT REPORTED | Unknown | MISSING |

## Simulation Results

**No simulation executed.** NO_FIT classification.

**Why NO_FIT — fitness criteria assessment:**

- ❌ **Population-level intervention:** Technically yes (11 clinics, population-level deployment) — PASS
- ⚠️ **Predictive model drives intervention:** The model exists, but the paper's findings are about cases where it *didn't* drive intervention due to operational failures — PARTIAL
- ❌ **Intervention is a state change:** The "intervention" here is deployment of an AI system into a workflow. The outcome studied is whether and how staff use the system. This is not a binary state change on individual patients. — FAIL
- ❌ **Measurable outcome each timestep:** There is no quantitative primary outcome. The paper reports qualitative findings, image rejection rates, and observational workflow disruptions. — FAIL
- ❌ **Counterfactual causal question:** No counterfactual is constructed. There is no "without AI" comparison for patient outcomes. — FAIL
- ✅ **Discrete-time dynamics:** Could be discretized if other criteria were met.

**3 of 6 criteria fail.** Per AGENTS.md: NO_FIT.

## Why This Paper Is a Finding About SDK Scope

The paper's value to this project is not in reproducing a quantitative effect — it's in documenting the **pre-conditions that must hold** for any quantitative simulation to be valid:

1. **Image quality distribution matters:** In lab conditions, 0% rejection; in field, 21% rejection. Any simulation of AI-assisted retinopathy screening that doesn't model image quality distribution will systematically overestimate effectiveness. The SDK's `predict()` method assumes the model operates on all entities; a realistic extension would sample from an ungradeable-image distribution before prediction occurs.

2. **Connectivity is not binary:** The paper reveals that connectivity failures create multi-hour queues, changing the entire workflow. The SDK has no concept of system latency or queue dynamics.

3. **Nurse compliance is not constant:** Staff actively discouraged enrollment as frustration mounted. The SDK models a fixed intervention delivery rate; real-world compliance is a time-varying function of frustration, workload, and trust.

4. **The intervention-reversal problem:** When staff discourage patients from using the AI, the "intervention" becomes a *harm* — patients who would have been screened without the AI aren't screened because of it. The SDK can model treatment-arm outcomes but not this kind of system-induced harm via workflow displacement.

**Bottom line for the target health system:** Before deploying any imaging AI, simulate the operational failure modes that Beede documents. The 94.7% lab accuracy means nothing if 21% of images are rejected, nurses are discouraged, and the referral pathway doesn't scale. This paper should be mandatory reading for any AI procurement process involving imaging at the target health system.

## Verification Summary

Not applicable — no simulation run.

## Discrepancies

Not applicable — no quantitative findings to reproduce.

## Scientific Reporting Gaps

1. **No quantitative primary outcome:** CHI 2020 papers often use mixed methods, but without a pre-specified primary endpoint and quantitative effect size, this paper cannot be used to validate AI-assisted DR screening effectiveness.

2. **No pre/post or control comparison:** Without a comparison condition, we cannot know whether DR detection rates changed, whether patient outcomes improved, or whether the system helped any patient at all.

3. **94.7% accuracy is the wrong metric:** Accuracy conflates sensitivity and specificity in a disease with ~5-15% prevalence. At 5% prevalence, a model that always says "no DR" achieves 95% accuracy. The paper should report sensitivity, specificity, and PPV.

4. **Sample sizes for qualitative findings not reported:** "Nurses frustrated" — how many nurses? How many observations? This limits generalizability.

5. **No model AUC:** Despite being a Google deep learning paper, no AUROC is reported for the Thai deployment. Lab accuracy (94.7%) is reported but this is not a discriminative performance metric.

6. **Image rejection mechanism not specified:** What makes an image "ungradeable"? The paper doesn't provide an image quality model, making the 21% rejection rate uninterpretable for planning purposes.

## Assumptions Made

No simulation-level assumptions. 

**Analytical note:** If forced to build a partial simulation of the operational aspects the SDK *does* support (e.g., what would happen if we modeled the ~79% of gradeable images and ignored the rest?), we could estimate the effective reach and impact. But this would be simulating a best-case scenario that directly contradicts the paper's findings — which is exactly the kind of misleading analysis this project exists to prevent.
