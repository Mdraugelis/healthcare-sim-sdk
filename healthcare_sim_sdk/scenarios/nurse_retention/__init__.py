"""Nurse Retention Scenario — Laudio-style predict + intervene.

Simulates nurse turnover in a health system where an ML model
identifies at-risk nurses and nurse managers conduct targeted
check-ins to improve retention. Manager capacity is the binding
constraint: each manager can only reach a limited number of
nurses per week, so threshold selection and model quality
jointly determine how many preventable departures are caught.

Inspired by Laudio/AONL research (April 2025) showing targeted
manager engagement at key milestones improves retention by
+6-13 percentage points in the first year.
"""

from .scenario import NurseRetentionScenario, RetentionConfig

__all__ = ["NurseRetentionScenario", "RetentionConfig"]
