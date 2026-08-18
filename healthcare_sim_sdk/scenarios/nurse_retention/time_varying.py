"""Time-varying parameter support for nurse retention scenarios.

A ``TimeVaryingParameter`` wraps a base value and a schedule of change
points, allowing simulation parameters to evolve over weeks without
breaking step() purity. The ``value_at(t)`` method is a pure function
of the schedule and ``t`` — no RNG, no hidden state.

Used by the monitoring validation task to model:
- Gradual decay of intervention effectiveness over time
- Abrupt capacity changes (staffing crises)
- Linear model AUC drift from population shift
- Any other parameter that should not be constant during a run

For scenarios that do not need time variation, plain floats work as
before; ``RetentionConfig`` fields accept ``float | TimeVaryingParameter``.

Design notes:

- The dataclass is frozen so it can be treated as immutable configuration
  data. No mutation after construction.
- ``change_points`` is a tuple of ``(week, value)`` pairs sorted by week.
- Three interpolation modes are supported:
  * ``"step"`` — value changes abruptly at each change point
  * ``"linear"`` — values interpolate linearly between consecutive points
  * ``"exponential"`` — values interpolate exponentially (useful for decay)
- Before the first change point, ``base`` is returned.
- At or after the last change point, the last value is returned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class TimeVaryingParameter:
    """A parameter whose value changes over simulation weeks.

    Supports step, linear, and exponential interpolation between change
    points. Immutable and deterministic: ``value_at(t)`` is a pure
    function of ``(t, base, change_points, interpolation)``.

    Attributes
    ----------
    base : float
        The value returned for ``t`` before the first change point.
    change_points : tuple of (int, float)
        Sorted sequence of ``(week, value)`` pairs. Weeks must be
        strictly increasing. Empty tuple means "constant at ``base``".
    interpolation : str
        One of ``"step"``, ``"linear"``, or ``"exponential"``.
        - ``"step"``: value jumps to the new value at each change point
          and stays there until the next change point
        - ``"linear"``: value interpolates linearly between consecutive
          change points; before the first, returns ``base``; after the
          last, returns the last value
        - ``"exponential"``: value interpolates exponentially
          (``value_a * (value_b/value_a)^fraction``); useful for
          half-life style decay schedules

    Examples
    --------
    Constant value (degenerate case, equivalent to plain float):

    >>> p = TimeVaryingParameter(base=0.5)
    >>> p.value_at(0)
    0.5
    >>> p.value_at(100)
    0.5

    Step change at week 30 from 6 to 2 (capacity collapse):

    >>> p = TimeVaryingParameter(
    ...     base=6.0,
    ...     change_points=((30, 2.0),),
    ...     interpolation="step",
    ... )
    >>> p.value_at(0)
    6.0
    >>> p.value_at(29)
    6.0
    >>> p.value_at(30)
    2.0
    >>> p.value_at(100)
    2.0

    Linear ramp from 0.5 to 0.0 between weeks 26 and 52 (gradual decay):

    >>> p = TimeVaryingParameter(
    ...     base=0.5,
    ...     change_points=((26, 0.5), (52, 0.0)),
    ...     interpolation="linear",
    ... )
    >>> p.value_at(0)
    0.5
    >>> p.value_at(26)
    0.5
    >>> round(p.value_at(39), 4)  # midpoint
    0.25
    >>> p.value_at(52)
    0.0
    >>> p.value_at(100)
    0.0
    """

    base: float
    change_points: Tuple[Tuple[int, float], ...] = field(default_factory=tuple)
    interpolation: str = "step"

    def __post_init__(self) -> None:
        # Validate interpolation mode
        if self.interpolation not in ("step", "linear", "exponential"):
            raise ValueError(
                f"interpolation must be 'step', 'linear', or "
                f"'exponential', got {self.interpolation!r}"
            )

        # Validate change_points are sorted and strictly increasing in t
        if self.change_points:
            weeks = [cp[0] for cp in self.change_points]
            for i in range(1, len(weeks)):
                if weeks[i] <= weeks[i - 1]:
                    raise ValueError(
                        "change_points must have strictly increasing "
                        f"weeks, got {weeks}"
                    )
            # For exponential interpolation, values cannot cross zero
            if self.interpolation == "exponential":
                all_values = [self.base] + [cp[1] for cp in self.change_points]
                signs = {
                    1 if v > 0 else (-1 if v < 0 else 0)
                    for v in all_values
                }
                if 0 in signs or len(signs) > 1:
                    raise ValueError(
                        "exponential interpolation requires all values "
                        "(base and change_points) to have the same "
                        "non-zero sign"
                    )

    def value_at(self, t: int) -> float:
        """Return the parameter value at simulation week ``t``.

        This is a pure function: no side effects, no RNG, no hidden
        state. Deterministic for a given ``(t, self)`` pair.
        """
        if not self.change_points:
            return self.base

        first_week = self.change_points[0][0]

        # Before the first change point: return base
        if t < first_week:
            return self.base

        # At or after the last change point: return last value
        last_week, last_value = self.change_points[-1]
        if t >= last_week:
            return last_value

        # Find the two change points bracketing t
        # (we know first_week <= t < last_week here)
        prev_week = first_week
        prev_value = self.change_points[0][1]
        for week, value in self.change_points[1:]:
            if t < week:
                break
            prev_week, prev_value = week, value
        else:
            # Should never reach here given the guard above
            return last_value

        next_week, next_value = week, value

        if self.interpolation == "step":
            # Hold previous value until we cross next_week
            return prev_value

        # Linear or exponential: interpolate
        span = next_week - prev_week
        if span == 0:
            return prev_value
        fraction = (t - prev_week) / span

        if self.interpolation == "linear":
            return prev_value + fraction * (next_value - prev_value)

        # Exponential: prev_value * (next_value/prev_value) ** fraction
        # __post_init__ guarantees same sign and non-zero
        ratio = next_value / prev_value
        return prev_value * math.pow(ratio, fraction)


def resolve(field_value, t: int) -> float:
    """Return a numeric value from a float or a TimeVaryingParameter.

    Convenience helper for scenario code that holds either type in a
    config field. If ``field_value`` is already a number, it is returned
    as-is. If it is a ``TimeVaryingParameter``, its ``value_at(t)`` is
    returned.

    Pure function: does not mutate ``field_value`` or depend on global
    state.
    """
    if isinstance(field_value, TimeVaryingParameter):
        return field_value.value_at(t)
    return float(field_value)
