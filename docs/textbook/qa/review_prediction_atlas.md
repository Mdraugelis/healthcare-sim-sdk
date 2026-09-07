# Independent review: prediction, policies, and scenario atlas

Reviewed chapters 06, 07, 08, and 13 against the current source on 2026-09-05. The chapters are suitable for the textbook after the corrections below. This is a documentation review, not clinical validation or statistical certification of the scenarios.

## Corrections applied

- **07, controlled models:** replaced `\frac12` with mathtext-compatible `\frac{1}{2}`. The original expression failed the actual MathTextParser.
- **07, subgroup scoring:** clarified that the severity adjustment has zero mean on the raw latent-score scale at the configured positive severity mix. A subsequent sigmoid does not generally preserve the mean reported probability; a changing severity mix can also change the raw-score mean.
- **08, capacity:** clarified that consults and screening share assessment capacity within each simulated world. Factual and counterfactual worlds use the same configured capacity; they do not compete for a combined cross-world resource pool.
- **13, MASAI:** separated detections per screened person, reads per screen, and relative reading workload. The original wording unnecessarily implied that detection and workload must use different denominators.

## Evidence checked

- All four JSON files parse; every fenced equation renders with Matplotlib mathtext; every cited local file and exact function/class symbol resolves by AST inspection.
- Recomputed the PPV example (16/114), four-score AP (5/6), equal-variance binormal operating-point AUC (approximately 0.916), expected reminder benefits (0.1525 versus 0.2325), and annual-to-weekly probability conversion (0.004766706). The exact pair-count AUC example is 3/4.
- Read the metric implementations: inclusive threshold ties; finite-grid legacy AUC; average tied ranks in fidelity AUC; stable but ungrouped AP ties; fractional-budget rounding; binned linear calibration versus logistic recalibration. The chapters correctly distinguish these behaviors and their limits.
- Matched the main controlled-model formula to `_generate_scores`, including Gaussian scales and label-conditional means. Checked its four modes, 15×15×6 default grid, weighted objectives, calibration on reused fitting data, unconstrained fitted Platt slope, and empirical latent-risk ceiling. The chapter appropriately qualifies both ranking preservation and the ceiling; neither is treated as an unconditional theorem of this implementation.
- Read both alternative controlled classes and the frozen subgroup scorer. Their APIs and generators differ. The subgroup path uses persistent per-entity noise, current simulated truth, subgroup dispersion/scrambling, and the current severity adjustment; it is not a deployable clinical feature model.
- Checked operational semantics against reminder, malnutrition, retention, and sepsis methods: attempted calls versus contact; worklist proposal versus assessment; separate SOC/AI budgets with maximum protection; responders versus newly treated patients and response capacity.
- Inspected scenario clocks, defaults, absorbing/repeated outcomes, and simplified paper-family mechanisms. AST enumeration confirms 14 domain classes plus the teaching template. The atlas correctly warns about septic-state occupancy, repeated stroke events, the closed retention cohort, SHIELD-RT repeated draws, absent Kaiser hospital rollout gating, InSight arm masking, and the deterministic Manz rollout approximation.
- Retrieved primary-paper records for the ESM external-validation study, Kaiser deterioration program, InSight randomized study, and SHIELD-RT trial. These support the atlas's study-context descriptions. Direct PubMed opens sometimes returned empty content; targeted search recovered the cited records. This review did not independently reproduce every published study or refresh all of the author-verified bibliography.

## Limits and remaining review

No production code was changed and no full simulation suite was run. Small arithmetic checks do not establish that a stochastic generator attains every requested target. Paper-family naming does not certify reproduction of original populations, assignments, models, or clinical estimators. The offline HTML build and browser review must still check page layout, interactions, navigation, and links. Any claim of prospective clinical usefulness requires evidence beyond these synthetic mechanisms and documentation checks.
