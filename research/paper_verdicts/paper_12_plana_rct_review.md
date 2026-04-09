# Paper 12: Plana et al. — Systematic review of ML RCTs (JAMA Network Open 2022)

## Classification: NO_FIT
## Reproducibility: N/A

---

## Key Findings

This systematic review cataloged the near-total absence of rigorous RCT evidence for clinical ML interventions as of October 2021: only 41 RCTs existed globally, 85% had n<1,000, 56% were single-center, 0% met all CONSORT-AI criteria, and only 27% reported race/ethnicity data. The paper does not describe a specific deployed intervention — it characterizes the *landscape* of evidence. There is no population, no model, no intervention arm, and no effect size to parameterize. The SDK cannot simulate a meta-review. However, this paper's findings are among the strongest motivators for the SDK's existence: the scarcity of rigorous trial evidence makes pre-deployment computational evaluation not merely useful but structurally necessary.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Study population | 41 RCTs (aggregate meta-data) | Discrete patient entities with outcomes | ABSENT |
| Base outcome rate | Not applicable (meta-level) | Population-level base rate | ABSENT |
| Model AUC | Varies across reviewed trials | Single calibrated AUC | ABSENT |
| Intervention type | Varies (no single intervention) | Specific state-change action | ABSENT |
| Effect size | 70–81% trials reported positive endpoints (qualitative) | Absolute risk reduction | ABSENT |
| Targeting rule | Varies | Threshold or top-K rule | ABSENT |
| Demographics | 27% of reviewed trials reported race/ethnicity | Demographic breakdown for equity audit | ABSENT |
| Sample sizes | 85% n<1,000 | N≥1,000 for stable rates | ABSENT |
| Follow-up period | Varies | Defined timestep window | ABSENT |
| Counterfactual structure | Varies (RCT, but many unblinded) | Branched factual/counterfactual | ABSENT |

**Nothing is simulatable.** Every parameter that would allow SDK instantiation is either absent (by design — this is a review) or heterogeneous across 41 source papers.

---

## Simulation Results

Not attempted. Classification: NO_FIT.

The SDK requires a single, coherent scenario: one population, one model, one intervention, one outcome. A meta-analysis of 41 heterogeneous trials does not constitute a scenario — it constitutes evidence *about* scenarios. Attempting to simulate "the average RCT" would be epistemically meaningless.

---

## Verification Summary

No simulation run. No verification checks performed.

---

## Discrepancies

None applicable. The paper's findings are internally consistent and well-documented. The gap is structural: meta-reviews describe the space of experiments, not a single experiment.

---

## Scientific Reporting Gaps

The paper itself documents the field's reporting gaps — which is the point. Specific gaps relevant to SDK users:

1. **Zero percent CONSORT-AI compliance**: No reviewed trial reported all required elements. The SDK's verification protocol addresses multiple CONSORT-AI criteria computationally (threshold reporting, subgroup analysis, model performance in deployment context).

2. **73% missing race/ethnicity data**: The SDK's equity audit module would catch this at design time, forcing parameterization of demographic distributions before any simulation runs.

3. **85% underpowered (n<1,000)**: The SDK can compute minimum detectable effect sizes for any parameterized scenario, flagging whether a proposed trial would be adequately powered before it is run.

4. **56% single-center**: The SDK's multi-site simulation capability (see Kaiser Paper #3) models the variance introduced by site-level heterogeneity — a risk that single-center RCTs cannot detect.

5. **Publication bias in positive endpoints (70–81% positive)**: The SDK runs counterfactuals by construction, making it impossible to report only the favorable arm.

---

## Assumptions Made

None — no simulation was attempted. All statements above are assessments of paper structure, not modeled outputs.

---

## SDK Design Contribution

**This paper is a design mandate, not a simulation target.**

Plana et al. establish that the clinical ML field operates with:
- Thin evidence base (41 trials globally after years of deployment)
- Systematic under-reporting of equity data
- Near-zero adherence to reporting standards
- Insufficient statistical power to detect modest effects
- Possible publication bias toward positive results

Every one of these failures is partially addressable by a pre-deployment simulation SDK. Specifically:

1. The SDK should implement CONSORT-AI checklist verification as a pre-simulation gate, refusing to proceed until required parameters are specified.
2. The SDK's equity audit should be non-optional — demographic distributions must be parameterized or the simulation is flagged as incomplete.
3. The SDK should output statistical power calculations alongside effect size estimates, making sample size inadequacy explicit.
4. Multi-site variance modeling should be a default feature, not an optional extension.

**Priority for SDK roadmap:** HIGH. This paper motivates mandatory equity parameterization and CONSORT-AI compliance checking as pre-simulation gates.
