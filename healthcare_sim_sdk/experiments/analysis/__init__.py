"""Analysis utilities for post-hoc estimation on simulation results.

Currently provides:

- ``its``: Interrupted Time Series segmented regression and related
  estimators (Wagner 2002; Bernal, Cummins, Gasparrini 2017). Supports
  classic pre/post segmented regression with HAC standard errors
  for autocorrelation-penalised inference, a CITS-with-control-series
  mode lifted from ``nurse_retention/monitoring/tier3_cits.py``, and
  a slope-only floor mode.
"""

from healthcare_sim_sdk.experiments.analysis import its

__all__ = ["its"]
