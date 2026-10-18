# Methodological foundations and validation alignment

Reviewed 17 October 2026 against revision `7bcdc3c` and the existing rerun reports.
Companion to [Paper-based validation](paper_based_validation.md).

**Assessment:** the SDK has substantial executable checks of its internal
behavior. The proposed paper benchmark is consistent with established validation
principles, but its evidence ledgers, masking controls, uncertainty propagation,
and study-specific analyses remain to be implemented. The existing calibrated
paper exercises do not establish independent external validation.

This is a focused methodological review, not a systematic literature review,
certification, or completed reassessment of all 30 entries. Below, **implemented**
means inspected code or recorded execution supports the specific statement;
**partial** means important conditions remain; **proposed** means documented design
without a completed benchmark. These are our judgments, not the cited authors'
endorsement of the SDK.

## 1. Foundations to cite in the benchmark protocol

| Foundation | What it contributes | Current alignment and required action |
|---|---|---|
| **Morris, White & Crowther (2019)**, [Using simulation studies to evaluate statistical methods](https://doi.org/10.1002/sim.8086), *Statistics in Medicine* 38:2074–2102 | ADEMP: aims, data-generating mechanisms, estimands, methods, performance measures; Monte Carlo uncertainty. | **Partial.** Configured sweeps, seeds and reports exist. Require one explicit ADEMP specification per benchmark, including failure handling, Monte Carlo standard errors, and precision-based repetition counts. This organizes method evaluation; it does not by itself validate a clinical mechanism. |
| **Sargent (2013)**, [Verification and validation of simulation models](https://doi.org/10.1057/jos.2012.20), *Journal of Simulation* 7:12–24; [open author conference treatment](https://informs-sim.org/wsc13papers/includes/files/027.pdf) | Distinguishes conceptual-model validity, implementation verification, data validity and operational validity. | **Partial.** Branch, RNG, accounting and boundary checks support verification. Add a clinical/workflow review of each conceptual model and empirical checks tied to its intended application. Passing software checks cannot establish all of these forms of validity. |
| **Eddy et al. (2012)**, [Model transparency and validation, ISPOR–SMDM Task Force–7](https://doi.org/10.1016/j.jval.2012.04.012), *Value in Health* 15:843–850 | Distinguishes dependent, partially dependent, independent and blinded validation; validation is application-specific. | **Partial; central gap.** Documentation and verification exist. Record how each validation source influenced model development or calibration. Outcome masking is proposed; independent blinded validation has not been demonstrated. |
| **Briggs et al. (2012)**, [Model parameter estimation and uncertainty, ISPOR–SMDM Task Force–6](https://doi.org/10.1016/j.jval.2012.04.014), *Value in Health* 15:835–842 | Separates stochastic variation, parameter uncertainty, heterogeneity and structural uncertainty. | **Partial.** Seed and selected parameter sweeps exist. Add justified joint input distributions and separate alternative mechanisms. Preserve correlations and distinguish estimated distributions from scenario ranges lacking probability weights. |
| **Kennedy & O'Hagan (2001)**, [Bayesian calibration of computer models](https://doi.org/10.1111/1467-9868.00294), *JRSS B* 63:425–464 | Separates fitting unknown parameters from remaining uncertainty and model discrepancy. | **Partial.** Existing exercises fit efficacy or capacity to clinical outcomes. Label those fits as calibration, preserve their targets and search procedures, and test elsewhere. We need not implement their full Bayesian method to respect this distinction. |
| **Gneiting & Raftery (2007)**, [Strictly proper scoring rules, prediction, and estimation](https://doi.org/10.1198/016214506000001437), *JASA* 102:359–378; [author text](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | Evaluates probabilistic predictions with scores that reward honest uncertainty; interval scores assess width and misses together. | **Proposed at the paper level.** Replace “result falls in interval = reproduced” with prespecified predictive scores, interval width, and comparison against a simple forecast. Do not confuse individual-patient Brier scores with scoring a paper's estimated effect. |
| **Holland (1986)**, [Statistics and causal inference](https://doi.org/10.1080/01621459.1986.10478354), *JASA* 81:945–960 | Potential outcomes clarify the causal contrast and the unobserved counterfactual. | **Partial.** Paired branches provide simulator truth conditional on its mechanisms. Declare policy, population, horizon and aggregation for that truth. Matching trial arm averages does not empirically validate each individual's simulated counterfactual. |
| **Hernán & Robins (2016)**, [Using big data to emulate a target trial](https://doi.org/10.1093/aje/kwv254), *AJE* 183:758–764 | Makes eligibility, strategies, assignment, follow-up, outcomes, causal contrasts and analysis explicit. | **Proposed per paper.** Entity-level exports include treatment timing, but those exports alone do not supply identification. Build and check a study protocol before choosing an estimator. Preserve randomized designs when reproducing randomized papers. |
| **Hudgens & Halloran (2008)**, [Toward causal inference with interference](https://doi.org/10.1198/016214508000000292), *JASA* 103:832–842 | Distinguishes direct and spillover effects when one person's treatment affects others. | **Partial.** Contract tests already restrict untreated-identity assertions where capacity/SOC spillovers can occur. Explicitly define a team/hospital policy effect and the interference boundary. Shared staffing means an untreated patient can still be affected by deployment. |

## 2. Design-specific references and decisions

These references apply when the corresponding design or claim is used. A single
estimator should not be imposed on the entire paper collection.

| Design or question | Reference | SDK assessment and decision |
|---|---|---|
| Interrupted time series | **Lopez Bernal, Cummins & Gasparrini (2017)**, [ITS regression tutorial](https://doi.org/10.1093/ije/dyw098), *IJE* 46:348–355 | **Partial.** Segmented regression and optional HAC standard errors exist. Prespecify impact shape, lag and duration; assess seasonality, overdispersion, serial dependence and concurrent changes. Validate coverage in the proposed setting. A significant baseline slope is not itself a validity failure: ITS explicitly models trends. |
| Observational staggered rollout, such as AAM | **Callaway & Sant'Anna (2021)**, [Difference-in-differences with multiple time periods](https://doi.org/10.1016/j.jeconom.2020.12.001), *Journal of Econometrics* 225:200–230; [author preprint](https://arxiv.org/abs/1803.09015) | **Proposed.** Panel exports exist; the inspected analysis package does not implement this estimator. Consider group/time effects using suitable not-yet-treated or never-treated controls, with explicit parallel-trends, overlap and anticipation assumptions. Test violations and cluster-level inference; late periods may lack eligible controls. |
| Randomized stepped-wedge studies, such as Manz | **Hemming et al. (2015)**, [The stepped wedge cluster randomised trial](https://doi.org/10.1136/bmj.h391), *BMJ* 350:h391 | **Not yet a faithful study reconstruction.** Reproduce randomized cluster crossover, calendar effects and within-cluster dependence. A common intervention-start week does not reproduce a stepped-wedge trial. Staggered implementation alone does not make an observational rollout randomized. |
| Exposure defined after baseline, such as confirmation within three hours | **Hernán et al. (2016)**, [Specifying a target trial prevents immortal time bias](https://doi.org/10.1016/j.jclinepi.2016.04.014), *JCE* 79:70–75 | **Timing audit required.** Align eligibility, strategy assignment and follow-up; specify handling of grace periods and early events. The TREWS confirmation comparison raises this audit question. This review does not establish that the original paper's analysis is biased. |
| Probability calibration | **Van Calster et al. (2019)**, [Calibration: the Achilles heel of predictive analytics](https://doi.org/10.1186/s12916-019-1466-7), *BMC Medicine* 17:230 | **Partial.** Feasibility/reference checks and mature-label calibration monitoring exist. Require calibration curves, intercept, slope and operating points in the correct cohort and outcome horizon. Achieving an AUC does not establish calibrated probabilities or correct subgroup performance. |
| Prediction usefulness | **Vickers & Elkin (2006)**, [Decision curve analysis](https://doi.org/10.1177/0272989X06295361), *Medical Decision Making* 26:565–574 | **Proposed where appropriate.** Assess decision consequences, not just AUC. A threshold's harm/benefit interpretation must be defensible. Fixed-capacity ranking and staffing displacement also need an explicit operational utility model; decision-curve net benefit alone is not a simulated causal reduction in mortality. |
| Predictions alter their own outcome labels | **Perdomo et al. (2020)**, [Performative prediction](https://proceedings.mlr.press/v119/perdomo20a.html), *ICML/PMLR* 119:7599–7609 | **Partial.** Tier 4 uses mature factual labels and has an oracle comparison for diagnosis. Successful prevention, selective survival and case-mix change can alter measured performance. Do not automatically interpret a factual AUC/calibration alarm as deterioration of the underlying model. |

## 3. Questions the literature resolves for our approach

### Can the simulator miss the published outcome and still be useful?

Yes, if its implementation is correct and its intended use supports the degree
of uncertainty disclosed. Preserve the original frozen prediction, the observed
discrepancy, and the assumptions that explain sensitivity. A plausible explanation
can motivate revision; it cannot retroactively turn an unsuccessful prediction
into a successful one. This is our benchmark decision rule, consistent with the
application-specific validation distinction above.

Grade external prediction and internal verification separately. Also grade
usefulness: a very wide prediction can be compatible with a result while offering
little evidence for a decision. Repeated misses can reveal a systematically
inadequate mechanism even when each individual miss has a plausible story.

### Is concealing the published effect sufficient for blinded validation?

No. Eddy et al. explicitly discuss blinded validation, including predicting trials
from their design and initial conditions. Record source dependence separately
from outcome access. Using a source to develop or fit mechanisms can leave a
comparison dependent even after its outcome is hidden. Trial-design inputs need
not themselves defeat masking; disclose their role and what was frozen first.

For this collection, previously inspected and tuned studies are **retrospectively
masked benchmarks**. Use an untouched study, site, cohort or future period for a
new outcome-blinded test. Log who could access targets, including prior code,
comments and reports, and freeze the permitted packet before prediction.

### What if we do not know intervention efficacy?

Do not manufacture it from predictive performance. Independently source it,
report conditional predictions across justified possibilities, or declare the
clinical prediction underdetermined. In a deliberately simplified homogeneous
example, a 50% contact probability with 20% efficacy and a 25% contact probability
with 40% efficacy have the same product. One aggregate improvement therefore
does not uniquely determine either mechanism. Real targeting, effect modification,
timing and capacity require additional structure beyond this toy product.

Calibration to a clinical result is legitimate, but that result is then a fitting
target. The Chong configuration explicitly calls reminder effectiveness 0.4437
calibrated; SHIELD-RT's scenario describes efficacy fitted from arm event rates.
Those cases must not be counted as independent predictions of those rates.

### How should uncertainty be generated and scored?

Separate four operations:

1. Sample justified uncertain inputs jointly, preserving dependencies. A sourced
   estimate still has uncertainty and may require a transport assumption.
2. Simulate the study's actual sampling, assignment and observation processes.
   Repeat stochastic realizations within input settings as needed for precision.
3. Evaluate plausible alternative mechanisms separately. Without defensible
   probabilities for those alternatives, report scenario-specific predictions
   and an envelope rather than calling the envelope a 95% predictive interval.
4. Score the frozen prediction of the same reported statistic after unblinding.
   Report Monte Carlo error separately from the study's sampling uncertainty.

For a central interval [l, u] with nominal coverage 1−alpha and observed statistic
y, one option from Gneiting & Raftery is the interval score (lower is better):

```text
IS = (u - l)
     + (2 / alpha) * (l - y) * I(y < l)
     + (2 / alpha) * (y - u) * I(y > u)
```

Choose coverage, effect scale, a practical width criterion and a simple comparison
forecast before seeing the outcome. For full predictive samples, CRPS is another
option. Estimate coverage across independent validation cases; one paper cannot
establish 95% calibration. Do not average raw scores on incompatible scales.

If the simulated studies already represent sampling variation in the published
estimator, adding the paper's reported sampling error again double-counts it.
Conversely, a distribution of population effects alone omits study sampling
variation and cannot be compared as though it forecasts the observed estimate.

### Does a paired oracle prove the causal analysis works?

Only for the specified simulated world and causal contrast. Keep oracle risks,
counterfactual branches and target effects outside the evaluator's observed data.
Use observed simulated controls for the evaluation, and oracle effects for grading.
If the evaluator knows the generating model, test structural alternatives too.

Our additional implication from the potential-outcome framework is that matching
arm-level event rates cannot identify the joint pair of individual potential
outcomes. Shared-random-number coupling is an explicit simulation assumption and
a useful variance-reduction device. Claims about *which individuals* were saved
need stronger justification than agreement on an average treatment effect.

With shared capacity, define the contrast between whole deployment policies,
including usual care, funding and spillovers. “No direct AI action” is not always
equivalent to “unaffected by AI.”

### Must quasi-experiments reproduce the original paper's estimator?

For a paper-reproduction claim, first reproduce its cohort, comparison and
reported estimator as closely as the available information permits. Label any
remaining approximation. An alternative estimator answers a separate question
and should be reported alongside the reconstruction.

For method evaluation, choose the estimator for the simulated identification
problem. Include null effects with secular trends, unmeasured confounding,
heterogeneous effects, anticipation, concurrent interventions and realistic
serial/cluster dependence. A more sophisticated standard-error calculation does
not remove confounding. Neither a nonsignificant pretrend test nor successful
recovery in a favorable simulation establishes identification in actual hospitals.

## 4. Evidence behind the alignment judgments

| Inspected artifact | What it supports | What it does not establish |
|---|---|---|
| [Issue-register verification and reruns](issue_register_validation.md) and [provenance](issue_register_rerun_provenance.json) | 818 completed parameter/seed cells; recorded checks, limitations and source revision | Independent validation of 30 publications |
| [Scenario contract matrix](../tests/integration/test_all_scenario_contracts.py) | RNG coupling and factual-trajectory noninterference across 13 domain scenarios; restricted untreated-identity checks | Nominal paper fidelity: several tests deliberately use lower AUCs or an explicit exploratory override |
| [Controlled model](../healthcare_sim_sdk/ml/model.py) | Independent-reference simulation, feasibility checks and reported performance tolerances | Independent clinical validation; bootstrapping an assumed risk distribution cannot establish that distribution's realism |
| [Analysis exports](../healthcare_sim_sdk/core/results.py) | Entity panels and first treatment times | Automatically valid causal contrasts, complete confounders or a staggered DiD estimator |
| [ITS implementation](../healthcare_sim_sdk/experiments/analysis/its.py) and [tests](../tests/unit/test_its_estimator.py) | Segmented regression, optional HAC, paired oracle difference analysis and selected statistical checks | General coverage under every serial or hospital-level dependence structure; independent-noise controls alone do not establish exchangeability |
| [Chong configuration](../healthcare_sim_sdk/scenarios/noshow_targeted_reminders/configs/chong_replication.yaml) and [SHIELD-RT scenario](../healthcare_sim_sdk/scenarios/paper06_shield_rt/scenario.py) | Direct evidence that clinical outcome information was used to select efficacy | A blinded prediction of those same clinical outcomes |
| [Experiment lifecycle](../healthcare_sim_sdk/experiments/lifecycle.py) | Saved configurations, metrics and catalog registration | A sealed target packet, access control or enforced pre-unblinding freeze |
| [Retention monitoring report](../reports/monitoring_validation/monitoring_validation.md) | See report for currently observed monitoring behavior and limitations | A universal guarantee that a monitoring alarm identifies model deterioration |

The rerun report explicitly flags 25 of 90 distinct retention reference fits
outside the 0.02 AUC tolerance. Such runs can support exploration with disclosure,
but cannot support a claim that all requested paper conditions were achieved.

## 5. Reporting references to use selectively

These help make an experiment inspectable. Completed checklists are not proof
that a mechanism is correct or a causal effect is identified.

| Reference | How we should use it |
|---|---|
| **Monks et al. (2019)**, [STRESS guidelines](https://doi.org/10.1080/17477778.2018.1442155), *Journal of Simulation* 13:55–67 (online 2018) | Map the applicable simulation-reporting items to model logic, data, initialization, experiment design, implementation and reproducibility. Choose and explain the relevant DES/ABS/SD checklist adaptation rather than claiming automatic compliance. |
| **Collins et al. (2024)**, [TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378), *BMJ* 385:e078378 | Extract and report prediction population, intended use, predictors, outcomes, evaluation data and performance. It supersedes TRIPOD 2015; it is a reporting guideline, not a clinical-effect validation method. |
| **Vasey et al. (2022)**, [DECIDE-AI](https://doi.org/10.1038/s41591-022-01772-9), *Nature Medicine* 28:924–933 | Check early clinical workflow, user interaction, delivery and error descriptions. The simulated intervention must represent the care process, not just a score. |
| **Cruz Rivera et al. (2020)**, [SPIRIT-AI](https://doi.org/10.1038/s41591-020-1037-7), and **Liu et al. (2020)**, [CONSORT-AI](https://doi.org/10.1038/s41591-020-1034-x) | Use the AI-specific protocol/report items for algorithm version, inputs, outputs, human decisions and error cases in randomized intervention benchmarks. Use alongside the updated core [SPIRIT 2025](https://doi.org/10.1038/s41591-025-03668-w) and [CONSORT 2025](https://doi.org/10.1136/bmj-2024-081123) statements, mapping older extension item numbers explicitly. |

## 6. Work required before a new validation claim

| Priority | Deliverable | Acceptance condition | Status at review |
|---|---|---|---|
| 1 | Source and assumption ledgers for Tarabichi and AAM | Every material input has population, horizon, units, source location, uncertainty, transformation and role; outcome-derived inputs are identified | Proposed; initial source corrections documented in the companion approach |
| 1 | Study/estimand specifications | Eligibility, time zero, usual care, assignment/rollout, interference boundary, follow-up, outcome and reported contrast agree with the selected study | Proposed |
| 1 | Masking and freeze record | Separate input, observed-data and grading packets; target-access history; immutable pre-unblinding configuration and code record | Proposed; current studies have prior outcome exposure |
| 1 | Application verification report | Accounting, timing, declared budgets, numerical checks, model performance, analytic cases and assumption-valid invariants pass | Existing checks provide a base; study-specific gates remain |
| 2 | Uncertainty and scoring specification | Joint parameter assumptions, structural alternatives, study sampling, baseline forecast, predictive score, useful width and MCSE rule fixed in advance | Proposed |
| 2 | Causal-method benchmark | Observed-data estimator tested against hidden truth for null/positive effects and explicit identification violations; bias, coverage, power and failures reported | Components exist; paper-matched benchmark proposed |
| 2 | External comparison report | Frozen result preserved; source dependence, discrepancy, informativeness and sensitivity reported without post hoc success relabeling | Proposed |

The current documentation change supplies the reference framework and decisions.
The pending benchmarks have not run. This review also corrects the ITS baseline
trend explanation and reclassifies the README's historical calibration examples;
executable simulation behavior is unchanged. Review checks: 626 tests passed,
11 private-artifact tests skipped, lint passed, and documentation links passed.
