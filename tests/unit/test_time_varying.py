"""Unit tests for TimeVaryingParameter and resolve() helper."""

import pytest

from healthcare_sim_sdk.scenarios.nurse_retention.time_varying import (
    TimeVaryingParameter,
    resolve,
)


class TestConstantValue:
    """A TimeVaryingParameter with no change points is constant."""

    def test_no_change_points_returns_base(self):
        p = TimeVaryingParameter(base=0.5)
        for t in [0, 1, 10, 52, 100, 1000]:
            assert p.value_at(t) == 0.5

    def test_base_zero(self):
        p = TimeVaryingParameter(base=0.0)
        assert p.value_at(0) == 0.0
        assert p.value_at(100) == 0.0

    def test_base_negative(self):
        # Negative base is fine for step/linear (not exponential)
        p = TimeVaryingParameter(base=-2.5)
        assert p.value_at(0) == -2.5


class TestStepInterpolation:
    """Step mode: value jumps at change points, holds between."""

    def test_single_step_change(self):
        p = TimeVaryingParameter(
            base=6.0,
            change_points=((30, 2.0),),
            interpolation="step",
        )
        assert p.value_at(0) == 6.0
        assert p.value_at(29) == 6.0
        assert p.value_at(30) == 2.0
        assert p.value_at(31) == 2.0
        assert p.value_at(100) == 2.0

    def test_multiple_step_changes(self):
        p = TimeVaryingParameter(
            base=1.0,
            change_points=((10, 2.0), (20, 3.0), (30, 1.5)),
            interpolation="step",
        )
        assert p.value_at(0) == 1.0
        assert p.value_at(9) == 1.0
        assert p.value_at(10) == 2.0
        assert p.value_at(19) == 2.0
        assert p.value_at(20) == 3.0
        assert p.value_at(29) == 3.0
        assert p.value_at(30) == 1.5
        assert p.value_at(100) == 1.5


class TestLinearInterpolation:
    """Linear mode: interpolate linearly between change points."""

    def test_linear_ramp(self):
        # 0.5 to 0.0 between weeks 26 and 52
        p = TimeVaryingParameter(
            base=0.5,
            change_points=((26, 0.5), (52, 0.0)),
            interpolation="linear",
        )
        assert p.value_at(0) == 0.5
        assert p.value_at(26) == 0.5
        assert p.value_at(52) == 0.0
        assert p.value_at(100) == 0.0
        # Midpoint check: week 39 is halfway through ramp
        assert abs(p.value_at(39) - 0.25) < 1e-9

    def test_linear_ramp_quarter_points(self):
        p = TimeVaryingParameter(
            base=1.0,
            change_points=((0, 1.0), (40, 0.0)),
            interpolation="linear",
        )
        # At quarters: 1.0, 0.75, 0.5, 0.25, 0.0
        assert abs(p.value_at(0) - 1.0) < 1e-9
        assert abs(p.value_at(10) - 0.75) < 1e-9
        assert abs(p.value_at(20) - 0.5) < 1e-9
        assert abs(p.value_at(30) - 0.25) < 1e-9
        assert abs(p.value_at(40) - 0.0) < 1e-9

    def test_linear_multi_segment(self):
        # Rise then fall: 1.0 → 3.0 → 2.0
        p = TimeVaryingParameter(
            base=1.0,
            change_points=((0, 1.0), (10, 3.0), (20, 2.0)),
            interpolation="linear",
        )
        assert abs(p.value_at(5) - 2.0) < 1e-9   # midpoint of 1→3
        assert abs(p.value_at(10) - 3.0) < 1e-9
        assert abs(p.value_at(15) - 2.5) < 1e-9  # midpoint of 3→2
        assert abs(p.value_at(20) - 2.0) < 1e-9


class TestExponentialInterpolation:
    """Exponential mode: geometric interpolation (half-life decay etc)."""

    def test_exponential_halving(self):
        # 1.0 at t=0, 0.5 at t=10, 0.25 at t=20
        p = TimeVaryingParameter(
            base=1.0,
            change_points=((0, 1.0), (10, 0.5), (20, 0.25)),
            interpolation="exponential",
        )
        assert abs(p.value_at(0) - 1.0) < 1e-9
        assert abs(p.value_at(10) - 0.5) < 1e-9
        assert abs(p.value_at(20) - 0.25) < 1e-9
        # Exponential midpoint of [1.0, 0.5] at t=5 should be sqrt(0.5)
        expected = 0.5 ** 0.5
        assert abs(p.value_at(5) - expected) < 1e-9

    def test_exponential_rejects_sign_change(self):
        with pytest.raises(ValueError, match="same.*non-zero sign"):
            TimeVaryingParameter(
                base=1.0,
                change_points=((10, -1.0),),
                interpolation="exponential",
            )

    def test_exponential_rejects_zero(self):
        with pytest.raises(ValueError, match="same.*non-zero sign"):
            TimeVaryingParameter(
                base=1.0,
                change_points=((10, 0.0),),
                interpolation="exponential",
            )


class TestValidation:
    """Constructor validates change_points ordering and interpolation."""

    def test_invalid_interpolation_mode(self):
        with pytest.raises(ValueError, match="interpolation must be"):
            TimeVaryingParameter(
                base=1.0,
                interpolation="cubic",
            )

    def test_unsorted_change_points_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeVaryingParameter(
                base=1.0,
                change_points=((10, 2.0), (5, 3.0)),
                interpolation="step",
            )

    def test_duplicate_weeks_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            TimeVaryingParameter(
                base=1.0,
                change_points=((10, 2.0), (10, 3.0)),
                interpolation="step",
            )


class TestPurityAndDeterminism:
    """value_at is a pure function of (self, t)."""

    def test_deterministic_across_calls(self):
        p = TimeVaryingParameter(
            base=0.5,
            change_points=((10, 0.3), (20, 0.1)),
            interpolation="linear",
        )
        # Same input, same output across many calls
        for _ in range(100):
            assert p.value_at(15) == p.value_at(15)

    def test_frozen_dataclass(self):
        p = TimeVaryingParameter(base=0.5)
        with pytest.raises((AttributeError, Exception)):
            p.base = 0.7  # type: ignore

    def test_hashable(self):
        # Frozen dataclasses with hashable fields are hashable
        p1 = TimeVaryingParameter(base=0.5, change_points=((10, 0.3),))
        p2 = TimeVaryingParameter(base=0.5, change_points=((10, 0.3),))
        assert hash(p1) == hash(p2)
        assert p1 == p2


class TestResolveHelper:
    """resolve() handles both plain floats and TimeVaryingParameters."""

    def test_resolve_float(self):
        assert resolve(0.5, t=0) == 0.5
        assert resolve(0.5, t=100) == 0.5

    def test_resolve_int_becomes_float(self):
        assert resolve(6, t=0) == 6.0
        assert isinstance(resolve(6, t=0), float)

    def test_resolve_tvp(self):
        p = TimeVaryingParameter(
            base=6.0,
            change_points=((30, 2.0),),
            interpolation="step",
        )
        assert resolve(p, t=0) == 6.0
        assert resolve(p, t=30) == 2.0
        assert resolve(p, t=100) == 2.0

    def test_resolve_is_pure(self):
        p = TimeVaryingParameter(
            base=0.5,
            change_points=((10, 0.1),),
            interpolation="step",
        )
        # Same call should give same result
        assert resolve(p, 15) == resolve(p, 15)
        # Calling with the same t shouldn't mutate p
        _ = resolve(p, 5)
        _ = resolve(p, 50)
        assert resolve(p, 15) == 0.1


class TestMonitoringRegimeSchedules:
    """Schedules used by the actual monitoring validation regimes."""

    def test_gradual_decay_regime(self):
        # Regime C: effectiveness 0.5 from 0-26, linear ramp to 0 by 52
        p = TimeVaryingParameter(
            base=0.5,
            change_points=((26, 0.5), (52, 0.0)),
            interpolation="linear",
        )
        assert p.value_at(0) == 0.5
        assert p.value_at(26) == 0.5
        assert abs(p.value_at(39) - 0.25) < 1e-9  # ramp midpoint
        assert p.value_at(52) == 0.0
        assert p.value_at(104) == 0.0

    def test_capacity_collapse_regime(self):
        # Regime D: capacity 6 from 0-29, then 2 from week 30 onward
        p = TimeVaryingParameter(
            base=6.0,
            change_points=((30, 2.0),),
            interpolation="step",
        )
        assert p.value_at(29) == 6.0
        assert p.value_at(30) == 2.0
        assert p.value_at(104) == 2.0

    def test_model_drift_regime(self):
        # Regime E: AUC 0.80 from 0-19, linear to 0.60 by week 40
        p = TimeVaryingParameter(
            base=0.80,
            change_points=((20, 0.80), (40, 0.60)),
            interpolation="linear",
        )
        assert p.value_at(0) == 0.80
        assert p.value_at(20) == 0.80
        assert abs(p.value_at(30) - 0.70) < 1e-9  # ramp midpoint
        assert p.value_at(40) == 0.60
        assert p.value_at(104) == 0.60
