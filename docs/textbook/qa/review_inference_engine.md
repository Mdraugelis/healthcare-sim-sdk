# Independent review: inference, monitoring, and execution chapters

Reviewed 2026-09-05: chapters 04, 05, 09, 10, 11, 14, and 15. This is an editorial and source-consistency review, not clinical approval or certification of every simulator configuration.

The chapters correctly distinguish policy estimands from individual treatment comparisons, Monte Carlo error from seed variability and clinical uncertainty, observation time from prediction time, and machine evidence from human approval. No unresolved blocking mathematical error was identified in the reviewed text. Four chapter files received bounded corrections; production code was unchanged.

## Corrections made

| Chapter | Finding | Correction and source |
|---|---|---|
| 04 | Matching RNG streams require a fresh scenario and correct stream use. An already advanced scenario does not restart merely because the counterfactual partitioner forks. | Made this precondition explicit. Checked `core/engine.py::_run_branched` and `core/rng.py::{create_streams,fork}`. |
| 10 | Certification gates include equality at the configured boundary; “below/above” suggested strict comparisons. Pointwise MC bounds could also be mistaken for joint certification. | Changed the gate wording to “at or below/at or above” and explicitly denied a simultaneous 95% guarantee across metrics/design cells. Checked `experiments/analysis/certification.py::{_upper_gate,_lower_gate,limitations}`. |
| 11 | “Joins future departures to scored identities” implied a generic ID-based join. The retention harness actually uses fixed full-cohort array positions. The text did not clearly distinguish replay from live monitoring. | Documented positional alignment, the active-at-prediction mask, review-packet ID/shape checks, offline replay, and stamping statistics at outcome-maturity week. Checked `monitoring/harness.py::run_from_results` and `monitoring/review.py::write_monitoring_review_packet`. |
| 11 | Evaluation completeness might be read as proof that all monitor inputs were available. | Explained that `evaluate_claim` checks weekly records; relevant metric and mature-label availability need separate review. |
| 15 | Zero capacity could be interpreted as disabling all ordinary care. | Restricted the invariant to the service governed by that resource constraint; separately resourced care can continue. |

## Checks completed

- Read all seven chapter bodies, worked examples, exercises, and code references against the current checkout. Confirmed branched versus snapshot call order, shared scenario ownership, RNG spawning, lagged appointment resolution, export identity mapping, exposure semantics, missing measurements, and the frozen outcome overlay’s limitations.
- Checked segmented ITS design columns and effect interpretation, OLS versus HAC inference, null/effect generation, selective observation, failure retention, bias, coverage, directional power, and exact Monte Carlo gate behavior. RMSE is correctly described as derivable from saved records rather than already being a named summary.
- Checked Shewhart and CUSUM rules, logistic recalibration, prediction-horizon maturity, null false alarms in either direction, censoring, conditional detection delay, duplicate-seed rejection, and incomplete-run exclusions.
- Confirmed the fault-injection and hand-worked equity examples exist in the cited tests. Confirmed trace projection uses defensive copies and immutable serialized facts, and provenance describes unsigned integrity rather than a digital signature or clinical endorsement.
- Recomputed the numerical examples. The four-run mean interval is `[7.4015, 12.5985]`; zero alarms in 30 independent runs give one-sided/two-sided upper bounds `0.09503/0.11570`; 52/208 independent 1% checks give any-alarm probabilities `0.40703/0.87637`.
- Executed chapter 05’s uncertainty example and chapter 10’s exact configuration. The latter completed all 200 null/effect fits and returned `INCONCLUSIVE`, consistent with the chapter’s explicit refusal to promise a passing certificate.
- Spot-checked primary/official references for [mean intervals](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm), [CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc313.htm), [calibration hierarchy](https://pubmed.ncbi.nlm.nih.gov/26772608/), [HAC inference](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html), common-random-number comparisons, and Sargent’s verification/validation paper. One PMC target-trial page returned a browser challenge during this spot-check; its citation was retained and is not represented as independently reread in full here.

## Remaining limits

HTML rendering and interactive-lab behavior belong to the separate build review. No production test suite was rerun for these documentation edits. This review checks the described mechanisms and bounded calculations; it does not exhaustively validate all scenario parameter combinations, clinical inputs, identification assumptions, or external-study reproductions. The positional identity requirement of retention monitoring is now disclosed, not repaired in production by this task.
