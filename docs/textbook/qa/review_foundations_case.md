# Internal review: foundations, malnutrition and peer-review workflow

Date: 2026-09-05. Scope: chapters 01, 02, 03, 12 and 16. This was an internal agent review conducted separately from authorship of these five chapters. It is not external scientific peer review or clinical validation.

## Checks completed

- Read all five chapter bodies, review questions, exercises and cited code claims. All five JSON files parse; 15 named Python source references resolve through the AST. Referenced non-Python files exist.
- Parsed all 12 mathematics blocks with Matplotlib mathtext. Independently recalculated first-pass benefit, false-positive workload, Bayes PPV, beta moments, total-variance examples, annual-to-shorter-window risk conversion, AR(1) standard deviations, and malnutrition readmission/LOS examples without calling production probability helpers.
- Compared distribution transformation and random-stream descriptions with `risk_distributions.py`, `temporal_dynamics.py` and `core/rng.py`; compared recurrence, clipping, state storage and occupancy claims with stroke and sepsis source.
- Inspected malnutrition's 37 state rows, worklist ordering, diagnosis, coding, LOS, readmission, primary outcomes, reduction denominators, scorer configuration and retained public demonstration artifacts. No production experiment or full test suite was rerun.
- Reopened the official NIST exponential and beta references and Berkeley author/course AR-model notes. Their displayed formulas support the associated mathematical claims.

## Worked quantities checked

| Quantity | Independent result | Manuscript result |
| --- | --- | --- |
| First-pass benefit | 10,000 × .05 × .80 × .50 × .25 = 50 | Correct |
| Positive predictive value | 400 / 1,350 = 29.6296% | Correct |
| Annual 5% risk to hazard | −log(.95) = .0512932944 per year | Correct |
| Weekly / four-week probability | .0985923166% / .3937864228%, using 1/52 and 4/52 years | Correct |
| Beta mean .05, alpha .5 | beta 9.5; variance .0043181818; SD .0657128741 | Correct |
| AR(1), rho .95, innovation SD .05 | stationary SD .1601281538 under stated unclipped assumptions | Correct |
| Readmission example | .20 × .90 × .94 = .1692; 3.08 percentage-point absolute reduction | Correct |
| LOS example | min(.5 + .2 × 3, .8) = .8 days | Correct |

The saved ranked-worklist analysis has mean 10.6998421 percentage points, sample SD 4.8349638, and three seeds. Recomputing SD/√3 gives MCSE 2.7914677. Applying the two-sided t critical value with two degrees of freedom reconstructs the saved interval [−1.3108739, 22.7105581]. The reported empirical seed percentiles match the retained analysis record; they were not mislabeled as the mean interval.

The seed-11 trace independently confirms patient 26's factual assessment at timestep 5 with no comparator assessment recorded through timestep 34, and patient 81's comparator assessment at timestep 7 with no factual assessment recorded through timestep 34. These observations support selected mechanisms, not their population frequency.

## Corrections applied

Only chapter 12 required substantive edits:

1. **Primary endpoint meaning:** added that a primary miss is a true case discharged without any assessment. An assessed patient without a confirmed diagnosis is not a primary miss. This distinction follows `MalnutritionPrioritizationScenario.measure` and avoids overstating diagnosis capture.
2. **Denominator dependence:** added that `reduce_results` uses each branch's own discharged true-case denominator. An intervention that changes length of stay can change denominator membership; the recorded contrast is not automatically a fixed-cohort causal contrast.
3. **Additional policy channels:** qualified the opening allocation description to acknowledge configurable flag-driven coding and screening-floor erosion, rather than attributing every possible recorded change solely to ranking.
4. **Input and timing contracts:** specified nonnegative reference timing, minimum-stay-adjusted runway, and application of LOS credit only to new diagnoses.
5. **Evidence scope:** identified the example case seed and bounded absent assessments to the observation horizon. Clarified that the roughly 14–15% incomplete-follow-up figure is the mean factual fraction across demonstration cells, rather than every seed or both branches.

Added exact source references for primary measurement and result reduction. No arithmetic corrections were needed in the reviewed chapters.

## Remaining limits

- Chapter 16's browser-local note storage, manuscript identity comparison and export behavior require the root author's UI implementation and browser QA. This review did not execute or certify those interactions. The final documentation quality record must report their actual completed checks.
- Retained demonstration summaries and selected traces were inspected; this was not a fresh reconstruction of all original malnutrition efficacy, financial, LOS/readmission or equity sweeps.
- The manuscript correctly preserves incomplete model/scenario applicability and real-world study-design evidence, selected-case limitations, and pending human review. Passing computational checks do not establish clinical efficacy or external validity.
- Later edits to these chapters or the referenced code may require renewed applicability review.
