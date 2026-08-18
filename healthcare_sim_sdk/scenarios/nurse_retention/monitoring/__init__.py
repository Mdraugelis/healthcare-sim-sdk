"""Tiered monitoring dashboard validation for nurse retention.

See ``tasks/tiered_monitoring_validation.md`` for the full task
specification. This module implements the four monitoring tiers
as stateful consumers of the nurse retention scenario's weekly
outputs, plus an offline Tier 4 model drift analyzer.
"""

from .aggregator import ManagerWeeklyRow, WeeklyDashboardRow, aggregate
from .events import DetectionEvent, MonitoringRun, Tier3Estimate
from .harness import MonitoringHarness
from .tier1_shewhart import ShewhartChart, Tier1Shewhart
from .tier2_cusum import Tier2CUSUM
from .tier3_cits import Tier3CITS
from .tier4_model_drift import (
    Tier4Result,
    Tier4RollingStats,
    analyze_tier4,
)

__all__ = [
    "DetectionEvent",
    "MonitoringHarness",
    "MonitoringRun",
    "ManagerWeeklyRow",
    "ShewhartChart",
    "Tier1Shewhart",
    "Tier2CUSUM",
    "Tier3CITS",
    "Tier3Estimate",
    "Tier4Result",
    "Tier4RollingStats",
    "WeeklyDashboardRow",
    "aggregate",
    "analyze_tier4",
]
