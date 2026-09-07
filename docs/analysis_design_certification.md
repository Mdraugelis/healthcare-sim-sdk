# Synthetic analysis-design certification

The certification harness evaluates the existing `segmented_regression` estimator against explicit synthetic ground truth. It is evidence about configured statistical behavior, not certification of a clinical design or its assumptions.

## Supported scope

Each design supplies the preperiod and follow-up lengths, OLS versus HAC covariance settings, stationary AR(1) period shocks, continuous individual-outcome variation, period denominators, and optional seasonality/secular slope. Denominators can vary sinusoidally. An adverse observation mechanism removes a configured fraction of the highest outcomes after rollout; the target remains the full-population additive level change. This deliberately tests whether selective observation creates bias despite apparently strong power.

The generator creates independent individuals within each period. It does not model binary event rates, clustered sites, delayed labels, missing timepoints, staggered rollout, or a specific clinical workflow. Those areas remain explicitly `NOT_RUN`. DiD is `NOT_RUN` because the SDK currently prepares DiD panels but provides no DiD estimator; the CITS group-trend estimator is not mislabeled as DiD.

Only the listed parameter combinations are evaluated. A certificate does not establish performance between those combinations or outside them. A statistical reviewer should specify a design grid and requirements before evaluating a proposed analysis, and keep development/tuning seeds separate from final validation seeds.

## Evidence and decisions

Each cell has independent null and planted-effect repetitions with saved seeds, true effect, estimate, standard error, interval, p-value, and denominator ranges. The null metric is any two-sided rejection at configured alpha. Power requires significance and the correct planted-effect direction. Coverage uses the estimator's Student-t interval for OLS or asymptotic-normal interval for HAC.

`binomial_performance()` supplies exact Clopper-Pearson 95% Monte Carlo intervals for Type I error, power, and coverage. Mean bias includes its Monte Carlo standard error and Student-t interval. These intervals describe simulation precision conditional on configured assumptions; they are distinct from each replicate's estimator interval.

Requirements are configurable: maximum Type I error, minimum power and coverage, and maximum absolute bias. Defaults are alpha=.05, maximum Type I=.10, minimum coverage=.90, minimum power=.80, and maximum absolute bias=.10 outcome units. These are demo requirements, not universal statistical standards or a claim of exact nominal 5% calibration.

A criterion passes only when its entire Monte Carlo interval satisfies the requirement. A confidently missed requirement fails. An overlapping interval is `INCONCLUSIVE`, including when too few repetitions prevent a useful conclusion. A failed fit remains recorded and fails its arm; metrics are not summarized over only successful repetitions. Bounds are pointwise and do not imply simultaneous family-wise certification across a large grid.

## Run a reproducible demonstration

```sh
.venv/bin/python scripts/certify_analysis_design.py --repetitions 100
```

The default six-design demonstration covers long and short observation windows, autocorrelation with OLS/HAC, variable denominators, and informative censoring. It intentionally includes adverse designs and is expected to return `FAIL`, with exit code 1. Exit codes are 0 for PASS, 1 for FAIL, and 2 for INCONCLUSIVE.

Results are written to `reports/analysis_design_demo/design_evidence.json` and `.md`. JSON contains the full configuration, estimator/generator/config SHA-256 digests, dependency versions, all repetition evidence, status criteria, and unsupported scope. Markdown gives statistical reviewers a concise linked readout. These local generated files can be linked in an experiment review packet as synthetic design evidence; their scope must remain visible.

To supply a prespecified configuration:

```json
{
  "designs": [
    {
      "name": "planned_window_ar1_stress",
      "n_pre": 48,
      "n_post": 24,
      "autocorrelation": 0.6,
      "hac_maxlags": 4,
      "denominator": 100,
      "denominator_variation": 0.5
    }
  ],
  "repetitions": 1000,
  "seed": 20260907,
  "planted_effect": -0.5,
  "alpha": 0.05,
  "max_type_i": 0.075,
  "min_coverage": 0.925,
  "min_power": 0.8,
  "max_abs_bias": 0.1
}
```

```sh
.venv/bin/python scripts/certify_analysis_design.py \
  --config design.json --output reports/planned_design_evidence
```

The library API is `run_design_certification(DesignCertificationConfig(...))`, followed by `write_design_evidence(result, output_dir)`. The writer returns absolute `json` and `markdown` paths for attachment to review packets. The configuration is a tuple of `ITSDesign` objects; all unspecified fields use documented dataclass defaults.
