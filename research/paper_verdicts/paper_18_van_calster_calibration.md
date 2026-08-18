# Paper 18: Van Calster et al. — Calibration as Achilles heel (BMC Medicine 2019)

## Classification: NO_FIT
## Reproducibility: N/A

---

## Key Findings

Only 36% of published prediction models report calibration statistics, despite calibration being clinically essential: a high-AUC model with poor calibration can generate lower net benefit than "treat none." Van Calster et al. demonstrate through simulation and analytical argument that calibration failures (E:O ≠ 1.0, calibration slope ≠ 1.0) systematically reduce net benefit, and that the magnitude of damage depends on the threshold probability at which the model is applied. This is a methods/review paper. It contains no new clinical deployment data. The SDK cannot simulate this paper as a healthcare scenario, but the paper directly specifies how the SDK must implement calibration-aware model simulation and flag calibration failures as high-priority safety risks.

---

## Parameter Extraction

| Parameter | Paper Reports | SDK Needs | Status |
|-----------|--------------|-----------|--------|
| Study population | Systematic review of prediction models + simulation demonstrations | Patient entities | NOT APPLICABLE |
| Calibration reporting rate | 36% of published models report calibration | Baseline calibration compliance | AVAILABLE (meta-statistic) |
| E:O ratio acceptable range | 0.8–1.2 (implied from simulations) | E:O threshold for warning | DERIVABLE |
| Calibration slope acceptable range | 0.8–1.2 (good calibration) | Slope threshold for warning | DERIVABLE |
| Calibration-in-the-large | Defined: mean predicted probability vs. observed prevalence | SDK calibration check | FULLY AVAILABLE |
| Weak vs. strong calibration | Hierarchy defined: calibration-in-the-large < weak < moderate < strong | SDK calibration taxonomy | FULLY AVAILABLE |
| Hosmer-Lemeshow test | Discussed; authors argue for calibration plots over omnibus tests | SDK test selection | AVAILABLE (recommendation) |
| Net benefit under miscalibration | Demonstrated: miscalibration reduces NB; can make NB < 0 | SDK warning threshold | DERIVABLE |
| Van Calster & Vickers 2015 result | Miscalibration always reduces net benefit (companion paper) | SDK validation | AVAILABLE |
| Intervention | None (methods/review paper) | State-change action | ABSENT |
| Patient outcomes | None (methods paper) | Outcome rates | ABSENT |

---

## Simulation Results

Not attempted. Classification: NO_FIT.

Van Calster et al. themselves run simulations in this paper to demonstrate that calibration failures degrade net benefit. The SDK should not re-simulate their demonstration — it should implement their findings as specifications. Specifically:

**What the SDK must implement (derived from this paper):**
1. `ControlledMLModel(mode="calibration")` already exists in the SDK. The calibration check in Phase 4 (Calibrate) must include:
   - E:O ratio (expected/observed events at deployed threshold)
   - Calibration slope (logistic regression of outcome on predicted probability)
   - Calibration plot (decile-level observed vs. predicted)
2. A model failing calibration checks (E:O outside [0.8, 1.2]) should be flagged as HIGH RISK before Phase 5 proceeds.
3. The SDK should compute net benefit under the observed calibration error and report how much net benefit is lost relative to a perfectly calibrated model at the same AUC.

---

## Verification Summary

No simulation run. SDK fitness criteria:
- [ ] Population-level intervention: Not applicable (methods/review paper)
- [ ] Predictive model drives intervention: Not applicable
- [ ] Intervention is a state change: Not applicable
- [ ] Measurable outcome at each timestep: Not applicable
- [ ] Counterfactual causal question: Not applicable
- [ ] Discrete-time dynamics: Not applicable

**0 of 6 criteria met → NO_FIT for simulation; critical specification source for SDK calibration module.**

---

## Discrepancies

None applicable. The paper's core findings — 36% reporting rate, miscalibration degrades net benefit — are internally consistent and widely replicated in subsequent literature.

**Extension worth noting:** The companion paper (Van Calster & Vickers, Medical Decision Making 2015) proved analytically that for any decision threshold, a miscalibrated model with positive net benefit will have *lower* net benefit than a perfectly calibrated version of the same model. This is not a simulation finding — it's a mathematical proof. The SDK should treat this as a hard constraint, not a soft recommendation.

---

## Scientific Reporting Gaps

The paper is a systematic critique of the field's reporting standards. Its findings *are* the reporting gaps:

1. **64% of models report no calibration at all.** This is not a gap in Van Calster et al. — it's a gap in the papers they reviewed. The SDK's pre-simulation gate should refuse to proceed to Phase 4 (Calibrate) without at least an E:O ratio from the source paper, or an explicit flagging of this absence.

2. **The calibration hierarchy is underused.** Van Calster et al. define a five-level calibration hierarchy (mean, weak, moderate, moderate overall, strong). In practice, most papers that do report calibration use only Hosmer-Lemeshow p-values, which the authors argue are inadequate. The SDK should implement at minimum: calibration-in-the-large + calibration slope.

3. **Temporal calibration degradation is not addressed.** This paper focuses on calibration at a single time point. In deployment, calibration drifts — case mix changes, treatment patterns change, and the model's predicted probabilities become progressively miscalibrated relative to current reality. The SDK's `step()` function should optionally model calibration drift over time, informed by the observation that clinical AI models degrade post-deployment (Papers #1, #25).

4. **Threshold-specific calibration is not standard.** A model may be well-calibrated overall but poorly calibrated in the high-risk tail — exactly where threshold-based interventions operate. Standard calibration checks may miss this. The SDK should compute calibration separately in the high-risk stratum (top 10% by predicted probability) as a mandatory check.

---

## Assumptions Made

None — no simulation attempted.

---

## SDK Design Contribution

**This paper is the specification for the SDK's calibration verification layer.** Every scenario simulation must include:

### Mandatory Calibration Checks (Phase 4)

```python
def calibration_checks(predicted_probs, observed_outcomes, threshold):
    """
    Implements Van Calster et al. calibration hierarchy.
    """
    # Level 1: Calibration-in-the-large (mean calibration)
    E_O_ratio = np.mean(predicted_probs) / np.mean(observed_outcomes)
    
    # Level 2: Calibration slope
    # logit(p) ~ alpha + beta * logit(p_hat)
    # Perfect calibration: alpha=0, beta=1
    
    # Level 3: Calibration plot (decile-level)
    # Plot observed vs. predicted probability by risk decile
    
    # Level 4: High-risk stratum calibration
    # Subset to top 10% by predicted probability
    # Compute E:O in this stratum specifically
    
    # Warnings:
    if E_O_ratio < 0.8 or E_O_ratio > 1.2:
        raise CalibrationWarning(f"E:O ratio {E_O_ratio:.2f} outside acceptable range [0.8, 1.2]")
    
    return calibration_metrics
```

### Net Benefit Under Miscalibration (Phase 5 extension)

```python
def net_benefit_under_miscalibration(auc, E_O_ratio, outcome_rate, p_t_range):
    """
    Computes net benefit degradation as a function of miscalibration.
    Implements Van Calster & Vickers 2015 result.
    """
    # Perfect calibration NB baseline
    # Miscalibrated NB = f(E:O, threshold probability)
    # Returns: NB_degradation = NB_perfect - NB_miscalibrated
```

### Integration with Paper #17 (Vickers DCA)

Van Calster and Vickers are complementary: DCA (Paper #17) gives the framework; calibration (this paper) gives the condition under which that framework produces reliable results. Together they form the SDK's complete model evaluation specification:
- High AUC + poor calibration → unreliable net benefit → poor clinical decisions
- High AUC + good calibration → reliable net benefit → valid threshold selection

**Cross-paper calibration findings from this pipeline:**
- Paper #1 (Epic ESM): AUC 0.63 at external site — likely accompanied by calibration failure (not reported)
- Paper #13 (TREWScore): No calibration reported despite AUC 0.83 on MIMIC-II
- Paper #14 (Edelson EWS): No calibration reported for any of six EWS systems
- Paper #15 (Obermeyer): Calibration irrelevant for cost-proxy (not predicting probabilities)
- The across-batch pattern: calibration is consistently absent from validation papers, confirming Van Calster's 36% reporting rate finding

**Priority for SDK implementation:** CRITICAL — calibration checking is a safety-critical function. A model deployed without calibration verification may generate net harm while appearing to have positive AUC.
