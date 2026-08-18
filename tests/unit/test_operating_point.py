"""Unit tests for well-posed operating-point targeting.

The legacy `classification` mode let callers pin BOTH sensitivity and PPV.
At a known prevalence those two pin specificity through Bayes' theorem, so
the pair names a single POINT in ROC space rather than a target region --
reachable only if some attainable ROC curve passes through it. When it was
not reachable the fit failed silently, producing a model 16x less sensitive
than configured while reporting success.

`operating_point` mode is well posed by construction:
  * exactly ONE binding constraint (ppv | sensitivity) fixes the threshold,
  * exactly ONE quality spec (target_auc, or a goal for the other metric,
    which implies an AUC) fixes the curve.

Fits use tiny search grids: these tests exercise contract and control flow,
not fit quality, and the default 15x15x6 grid is far too slow for a unit
test.
"""

import warnings

import numpy as np
import pytest

from healthcare_sim_sdk.ml.model import (
    ControlledMLModel,
    OperatingPointError,
    gate_tolerance,
)
from healthcare_sim_sdk.ml.performance import (
    iso_ppv_slope,
    ppv_at_sensitivity,
    required_auc_for_operating_point,
    sensitivity_at_ppv,
    validate_operating_point,
)
from healthcare_sim_sdk.population.risk_distributions import (
    beta_distributed_risks,
)

FAST_GRID = dict(
    correlation_grid=np.linspace(0.5, 0.99, 3),
    scale_grid=np.linspace(0.05, 0.5, 3),
)


def make_population(annual_rate, n=4000, seed=17):
    rng = np.random.default_rng(seed)
    risks = beta_distributed_risks(
        n_patients=n, annual_incident_rate=annual_rate,
        concentration=0.5, rng=rng,
    )
    labels = (rng.random(n) < risks).astype(int)
    return labels, risks, rng


# -- Degrees of freedom ---------------------------------------------------

class TestDegreesOfFreedom:
    """The configuration must leave exactly one free parameter."""

    def test_both_auc_and_goal_rejected(self):
        with pytest.raises(ValueError, match="over-determined"):
            ControlledMLModel(
                mode="operating_point", binding="ppv", binding_value=0.15,
                goal_value=0.80, target_auc=0.90,
            )

    def test_neither_auc_nor_goal_rejected(self):
        with pytest.raises(ValueError, match="under-determined"):
            ControlledMLModel(
                mode="operating_point", binding="ppv", binding_value=0.15,
            )

    def test_missing_binding_rejected(self):
        with pytest.raises(ValueError, match="binding="):
            ControlledMLModel(
                mode="operating_point", goal_value=0.80, target_auc=0.90,
            )

    def test_binding_must_be_a_known_metric(self):
        with pytest.raises(ValueError, match="binding="):
            ControlledMLModel(
                mode="operating_point", binding="auc", binding_value=0.9,
                goal_value=0.8,
            )

    def test_missing_binding_value_rejected(self):
        with pytest.raises(ValueError, match="binding_value"):
            ControlledMLModel(
                mode="operating_point", binding="ppv", goal_value=0.80,
            )

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
    def test_out_of_range_values_rejected(self, bad):
        with pytest.raises(ValueError):
            ControlledMLModel(
                mode="operating_point", binding="ppv", binding_value=bad,
                goal_value=0.80,
            )

    @pytest.mark.parametrize("kwargs", [
        dict(binding="ppv", binding_value=0.15, goal_value=0.80),
        dict(binding="ppv", binding_value=0.15, target_auc=0.90),
        dict(binding="sensitivity", binding_value=0.80, goal_value=0.15),
        dict(binding="sensitivity", binding_value=0.80, target_auc=0.90),
    ])
    def test_well_posed_configs_accepted(self, kwargs):
        ControlledMLModel(mode="operating_point", **kwargs)


# -- Operating-point geometry --------------------------------------------

class TestGeometry:
    """Fixing PPV pins a ray through the origin in ROC space."""

    def test_ppv_at_or_below_prevalence_is_degenerate(self):
        """Such a flagged set is no more enriched than the base rate."""
        result = validate_operating_point(prevalence=0.25, ppv=0.15)
        assert result["valid"] is False
        assert "worse-than-random" in result["reason"]
        assert result["slope"] < 1.0     # ray below the chance diagonal

    def test_ppv_above_prevalence_is_valid(self):
        result = validate_operating_point(prevalence=0.05, ppv=0.15)
        assert result["valid"] is True
        assert result["slope"] > 1.0

    def test_iso_ppv_slope_crosses_one_at_prevalence(self):
        prev = 0.10
        assert iso_ppv_slope(prev, prev + 1e-6) > 1.0
        assert iso_ppv_slope(prev, prev - 1e-6) < 1.0

    def test_sensitivity_increases_with_auc_at_fixed_ppv(self):
        """The whole design rests on this monotonicity."""
        vals = [sensitivity_at_ppv(a, 0.05, 0.15)
                for a in (0.65, 0.75, 0.85, 0.95)]
        assert vals == sorted(vals)
        assert vals[-1] > vals[0]

    def test_ppv_increases_with_auc_at_fixed_sensitivity(self):
        vals = [ppv_at_sensitivity(a, 0.05, 0.80)
                for a in (0.65, 0.75, 0.85, 0.95)]
        assert vals == sorted(vals)

    def test_required_auc_round_trips(self):
        solve = required_auc_for_operating_point(
            prevalence=0.05, binding="ppv",
            binding_value=0.15, goal_value=0.70,
        )
        back = sensitivity_at_ppv(solve["required_auc"], 0.05, 0.15)
        assert back == pytest.approx(0.70, abs=1e-3)

    def test_unreachable_goal_reports_the_ceiling(self):
        solve = required_auc_for_operating_point(
            prevalence=0.0005, binding="ppv",
            binding_value=0.95, goal_value=0.99,
        )
        assert solve["required_auc"] is None
        assert solve["attainable_at_auc_hi"] < 0.99


# -- Gate tolerance -------------------------------------------------------

class TestGateTolerance:

    def test_relative_for_large_targets(self):
        assert gate_tolerance(0.80) == pytest.approx(0.16)

    def test_absolute_floor_for_small_targets(self):
        """20% of a 0.02 target is 0.004 -- inside sampling noise."""
        assert gate_tolerance(0.02) == pytest.approx(0.02)

    def test_floor_and_relative_cross_over(self):
        assert gate_tolerance(0.10) == pytest.approx(0.02)
        assert gate_tolerance(0.50) == pytest.approx(0.10)


# -- The gate itself ------------------------------------------------------

class TestGate:
    """Enforcement is driven from the report, so it can be unit-tested
    without paying for a fit."""

    def _model(self, **kw):
        base = dict(
            mode="operating_point", binding="ppv", binding_value=0.15,
            goal_value=0.80,
        )
        base.update(kw)
        return ControlledMLModel(**base)

    def _report(self, ppv, sens, **extra):
        report = {
            "achieved_ppv": ppv, "achieved_sensitivity": sens,
            "achieved_auc": 0.85, "required_auc": 0.95,
        }
        report.update(extra)
        return report

    def test_on_target_passes(self):
        self._model()._enforce_operating_point(
            self._report(ppv=0.15, sens=0.80)
        )

    def test_undershoot_raises(self):
        with pytest.raises(OperatingPointError, match="under by"):
            self._model()._enforce_operating_point(
                self._report(ppv=0.15, sens=0.30)
            )

    def test_overshoot_also_raises(self):
        """Landing ABOVE target still means simulating the wrong model.

        ControlledMLModel exists to reproduce SPECIFIED characteristics,
        so 'better than asked' is still not what was asked.
        """
        with pytest.raises(OperatingPointError, match="over by"):
            self._model()._enforce_operating_point(
                self._report(ppv=0.15, sens=0.999)
            )

    def test_binding_miss_raises(self):
        with pytest.raises(OperatingPointError, match="binding ppv"):
            self._model()._enforce_operating_point(
                self._report(ppv=0.40, sens=0.80)
            )

    def test_error_names_the_auc_shortfall(self):
        with pytest.raises(OperatingPointError, match="required 0.9500"):
            self._model()._enforce_operating_point(
                self._report(ppv=0.15, sens=0.30)
            )

    def test_converged_search_blames_the_distribution(self):
        with pytest.raises(OperatingPointError, match="lower the goal"):
            self._model()._enforce_operating_point(
                self._report(ppv=0.15, sens=0.30,
                             auc_search_boundary_pinned=False)
            )

    def test_pinned_search_blames_the_grid(self):
        with pytest.raises(OperatingPointError, match="grid boundary"):
            self._model()._enforce_operating_point(
                self._report(
                    ppv=0.15, sens=0.30,
                    auc_search_boundary_pinned=True,
                    auc_search_pinned_params=["noise_scale=0.7000"],
                )
            )

    def test_override_warns_and_annotates(self):
        """The override must record what it accepted, not silence it."""
        model = self._model(allow_infeasible=True)
        report = self._report(ppv=0.15, sens=0.30)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._enforce_operating_point(report)
        assert len(caught) == 1
        assert report["accepted_infeasible"] is True
        assert any("sensitivity" in m for m in report["gate_misses"])


# -- End to end -----------------------------------------------------------

class TestEndToEnd:

    def test_binding_ppv_is_held_and_goal_reported(self):
        labels, risks, rng = make_population(0.10)
        model = ControlledMLModel(
            mode="operating_point", binding="ppv", binding_value=0.15,
            goal_value=0.60, allow_infeasible=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = model.fit(labels, risks, rng, n_iterations=1,
                               **FAST_GRID)
        # The binding constraint is placed exactly, by sweeping every
        # achievable cut rather than a fixed threshold grid.
        assert report["achieved_ppv"] >= 0.15 - 1e-9
        assert report["achieved_ppv"] == pytest.approx(0.15, abs=0.02)
        assert report["binding"] == "ppv"
        assert report["goal_metric"] == "sensitivity"
        assert report["required_auc_source"] == "derived_from_goal"

    def test_binding_sensitivity_is_symmetric(self):
        labels, risks, rng = make_population(0.10)
        model = ControlledMLModel(
            mode="operating_point", binding="sensitivity",
            binding_value=0.60, goal_value=0.15, allow_infeasible=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = model.fit(labels, risks, rng, n_iterations=1,
                               **FAST_GRID)
        assert report["achieved_sensitivity"] >= 0.60 - 1e-9
        assert report["binding"] == "sensitivity"
        assert report["goal_metric"] == "ppv"

    def test_target_auc_path_skips_the_solve(self):
        labels, risks, rng = make_population(0.10)
        model = ControlledMLModel(
            mode="operating_point", binding="ppv", binding_value=0.15,
            target_auc=0.80, allow_infeasible=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = model.fit(labels, risks, rng, n_iterations=1,
                               **FAST_GRID)
        assert report["required_auc_source"] == "target_auc"
        assert report["required_auc"] == pytest.approx(0.80)
        assert report.get("goal_value") is None

    def test_degenerate_ppv_rejected_at_fit_time(self):
        """Prevalence is only known once labels arrive."""
        labels, risks, rng = make_population(0.50)
        model = ControlledMLModel(
            mode="operating_point", binding="ppv", binding_value=0.05,
            goal_value=0.80,
        )
        with pytest.raises(OperatingPointError, match="Degenerate"):
            model.fit(labels, risks, rng, n_iterations=1, **FAST_GRID)

    def test_unreachable_goal_fails_before_searching(self):
        labels, risks, rng = make_population(0.01)
        model = ControlledMLModel(
            mode="operating_point", binding="ppv", binding_value=0.95,
            goal_value=0.99,
        )
        with pytest.raises(OperatingPointError, match="unreachable"):
            model.fit(labels, risks, rng, n_iterations=1, **FAST_GRID)


# -- Backward compatibility ----------------------------------------------

class TestBackwardCompatibility:

    @pytest.mark.parametrize(
        "mode", ["discrimination", "classification", "threshold_ppv"]
    )
    def test_legacy_modes_keep_their_auc_default(self, mode):
        """target_auc became Optional so operating_point can tell 'set'
        from 'defaulted'; legacy modes must not notice."""
        assert ControlledMLModel(mode=mode).target_auc == 0.83

    def test_legacy_modes_need_no_operating_point_args(self):
        model = ControlledMLModel(mode="discrimination")
        assert model.binding is None
        assert model.binding_value is None

    def test_unknown_mode_still_rejected(self):
        with pytest.raises(ValueError, match="mode must be one of"):
            ControlledMLModel(mode="nonsense")
