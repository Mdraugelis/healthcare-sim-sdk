"""No-Show Overbooking Evaluation Harness.

Runs the no-show scenario across multiple clinic configurations,
model types, and overbooking thresholds. Produces metric tables
for both baseline (patient historical rate) and ML predictor
evaluation.

Usage:
    from scenarios.noshow_overbooking.evaluation_harness import (
        run_evaluation_sweep, CLINIC_PROFILES, summarize_results
    )
    results = run_evaluation_sweep(
        clinics=CLINIC_PROFILES,
        model_configs=[
            {"model_type": "baseline"},
            {"model_type": "predictor", "model_auc": 0.83},
        ],
        thresholds=[0.15, 0.20, 0.25, 0.30, 0.40],
        n_days=90,
        seed=42,
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from scenarios.noshow_overbooking.scenario import (
    ClinicConfig,
    NoShowOverbookingScenario,
)
from sdk.core.engine import BranchedSimulationEngine, CounterfactualMode
from sdk.core.scenario import TimeConfig
from sdk.ml.performance import auc_score


@dataclass
class ClinicProfile:
    """A clinic archetype for evaluation."""
    name: str
    campus: str
    n_providers: int
    slots_per_provider: int
    target_utilization: float  # e.g. 0.80 or 1.15
    base_noshow_rate: float = 0.13
    n_patients: int = 2000

    @property
    def daily_capacity(self) -> int:
        return self.n_providers * self.slots_per_provider

    @property
    def max_overbook_per_provider(self) -> int:
        """Scale overbooking capacity by target utilization."""
        if self.target_utilization <= 1.0:
            return 2
        return max(1, int((self.target_utilization - 1.0) * 10) + 2)


# Default clinic profiles spanning the utilization spectrum
CLINIC_PROFILES = [
    ClinicProfile(
        "Under-booked Rural", "Rural", 4, 10,
        target_utilization=0.80, n_patients=800,
    ),
    ClinicProfile(
        "Low-volume Primary", "Suburban", 6, 10,
        target_utilization=0.85, n_patients=1200,
    ),
    ClinicProfile(
        "Standard Primary Care", "Main", 8, 12,
        target_utilization=0.92, n_patients=2000,
    ),
    ClinicProfile(
        "High-demand Specialty", "Main", 6, 14,
        target_utilization=1.05, n_patients=2000,
    ),
    ClinicProfile(
        "Over-booked Urban", "Urban", 10, 12,
        target_utilization=1.15, n_patients=3000,
    ),
]


@dataclass
class EvalResult:
    """Results from a single evaluation run."""
    clinic_name: str
    campus: str
    model_type: str
    model_auc: Optional[float]
    threshold: float
    n_days: int
    daily_capacity: int
    target_utilization: float

    # Core metrics
    observed_noshow_rate: float = 0.0
    baseline_predictor_auc: float = 0.0
    utilization_rate: float = 0.0
    collision_rate: float = 0.0
    overbooked_show_rate: float = 0.0
    overbookings_per_week: float = 0.0
    mean_wait_time_minutes: float = 0.0
    mean_overbooking_burden: float = 0.0
    max_overbooking_burden: int = 0

    # Subgroup breakdowns (dicts keyed by subgroup label)
    noshow_by_race: Dict[str, float] = field(default_factory=dict)
    noshow_by_insurance: Dict[str, float] = field(default_factory=dict)
    noshow_by_age: Dict[str, float] = field(default_factory=dict)
    collision_by_race: Dict[str, float] = field(default_factory=dict)
    overbooking_burden_by_race: Dict[str, float] = field(
        default_factory=dict
    )
    overbooking_burden_by_insurance: Dict[str, float] = field(
        default_factory=dict
    )


def run_single_evaluation(
    clinic: ClinicProfile,
    model_type: str = "baseline",
    model_auc: float = 0.83,
    threshold: float = 0.30,
    n_days: int = 90,
    seed: int = 42,
    max_individual_overbooks: int = 5,
) -> EvalResult:
    """Run a single clinic evaluation and compute all metrics."""
    cc = ClinicConfig(
        name=clinic.name,
        n_providers=clinic.n_providers,
        slots_per_provider_per_day=clinic.slots_per_provider,
        max_overbook_per_provider=clinic.max_overbook_per_provider,
    )

    tc = TimeConfig(
        n_timesteps=n_days,
        timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(n_days)),
    )

    scenario = NoShowOverbookingScenario(
        time_config=tc,
        seed=seed,
        n_patients=clinic.n_patients,
        base_noshow_rate=clinic.base_noshow_rate,
        model_type=model_type,
        model_auc=model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=max_individual_overbooks,
        clinic_config=cc,
        campus=clinic.campus,
    )

    engine = BranchedSimulationEngine(
        scenario, CounterfactualMode.BRANCHED,
    )
    results = engine.run(clinic.n_patients)

    # Aggregate metrics across all days
    return _compute_metrics(
        results, clinic, model_type, model_auc, threshold, n_days,
    )


def _compute_metrics(
    results, clinic, model_type, model_auc, threshold, n_days,
) -> EvalResult:
    """Compute all evaluation metrics from simulation results."""
    n_t = n_days

    # Collect arrays across timesteps
    all_noshows = []
    all_utilized = []
    all_wait = []
    all_race = []
    all_insurance = []
    all_age = []
    all_predicted = []
    all_true_probs = []

    for t in range(n_t):
        out = results.outcomes[t]
        all_noshows.append(out.events)
        all_utilized.append(out.secondary["utilization"])
        all_wait.append(out.secondary["wait_times"])
        all_race.append(out.secondary["race_ethnicity"])
        all_insurance.append(out.secondary["insurance_type"])
        all_age.append(out.secondary["age_band"])

        if t in results.predictions:
            pred = results.predictions[t]
            all_predicted.append(pred.scores)
            all_true_probs.append(pred.metadata["true_probs"])

    noshows = np.concatenate(all_noshows)
    utilized = np.concatenate(all_utilized)
    wait = np.concatenate(all_wait)
    race = np.concatenate(all_race)
    insurance = np.concatenate(all_insurance)
    age = np.concatenate(all_age)

    # Final-day state metadata for cumulative counters
    final_meta = results.outcomes[n_t - 1].metadata
    total_overbooked = final_meta["total_overbooked"]
    total_ob_showed = final_meta["total_overbooked_showed"]
    total_collisions = final_meta["total_collisions"]

    # Predictor AUC: compare predicted no-show prob vs actual outcome.
    # Predictions at time t are for the schedule that gets resolved at
    # time t+1, with results reported in outcomes[t+1].
    predictor_auc = 0.0
    if all_predicted:
        pred_all = np.concatenate(all_predicted)
        # Outcomes at t+1 correspond to predictions at t
        actual_noshows_for_auc = []
        for t in range(n_t):
            if t in results.predictions and (t + 1) < n_t:
                actual_noshows_for_auc.append(
                    results.outcomes[t + 1].events
                )
        if actual_noshows_for_auc:
            actual_flat = np.concatenate(actual_noshows_for_auc)
            # Trim predictions to match (last prediction has no outcome)
            n_actual = len(actual_flat)
            if n_actual > 0 and n_actual <= len(pred_all):
                predictor_auc = auc_score(
                    actual_flat, pred_all[:n_actual]
                )

    # Subgroup breakdowns
    noshow_by_race = _subgroup_rate(noshows, race)
    noshow_by_insurance = _subgroup_rate(noshows, insurance)
    noshow_by_age = _subgroup_rate(noshows, age)

    er = EvalResult(
        clinic_name=clinic.name,
        campus=clinic.campus,
        model_type=model_type,
        model_auc=model_auc if model_type == "predictor" else None,
        threshold=threshold,
        n_days=n_days,
        daily_capacity=clinic.daily_capacity,
        target_utilization=clinic.target_utilization,
        observed_noshow_rate=float(noshows.mean()) if len(noshows) else 0,
        baseline_predictor_auc=predictor_auc,
        utilization_rate=float(utilized.mean()) if len(utilized) else 0,
        collision_rate=(
            total_collisions / max(total_overbooked, 1)
            if total_overbooked > 0 else 0.0
        ),
        overbooked_show_rate=(
            total_ob_showed / max(total_overbooked, 1)
            if total_overbooked > 0 else 0.0
        ),
        overbookings_per_week=(
            total_overbooked / max(n_days / 7, 1)
        ),
        mean_wait_time_minutes=float(
            wait[wait > 0].mean() if np.any(wait > 0) else 0
        ),
        mean_overbooking_burden=final_meta["mean_overbooking_burden"],
        max_overbooking_burden=0,  # computed below
        noshow_by_race=noshow_by_race,
        noshow_by_insurance=noshow_by_insurance,
        noshow_by_age=noshow_by_age,
    )

    return er


def _subgroup_rate(
    values: np.ndarray, groups: np.ndarray,
) -> Dict[str, float]:
    """Compute mean of values within each group."""
    result = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() > 0:
            result[str(g)] = float(values[mask].mean())
    return result


def run_evaluation_sweep(
    clinics: Optional[List[ClinicProfile]] = None,
    model_configs: Optional[List[Dict[str, Any]]] = None,
    thresholds: Optional[List[float]] = None,
    n_days: int = 90,
    seed: int = 42,
    max_individual_overbooks: int = 5,
) -> List[EvalResult]:
    """Run full evaluation sweep across clinics, models, and thresholds.

    Returns a list of EvalResult objects, one per combination.
    """
    if clinics is None:
        clinics = CLINIC_PROFILES
    if model_configs is None:
        model_configs = [
            {"model_type": "baseline"},
            {"model_type": "predictor", "model_auc": 0.83},
        ]
    if thresholds is None:
        thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    all_results: List[EvalResult] = []

    for clinic in clinics:
        for mc in model_configs:
            for thresh in thresholds:
                mt = mc["model_type"]
                auc = mc.get("model_auc", 0.83)
                result = run_single_evaluation(
                    clinic=clinic,
                    model_type=mt,
                    model_auc=auc,
                    threshold=thresh,
                    n_days=n_days,
                    seed=seed,
                    max_individual_overbooks=max_individual_overbooks,
                )
                all_results.append(result)
                print(
                    f"  {clinic.name:25s} | {mt:10s} | "
                    f"thresh={thresh:.2f} | "
                    f"util={result.utilization_rate:.3f} | "
                    f"collision={result.collision_rate:.3f} | "
                    f"noshow={result.observed_noshow_rate:.3f}"
                )

    return all_results


def summarize_results(
    results: List[EvalResult],
) -> Dict[str, Any]:
    """Create summary tables from evaluation results.

    Returns dict with:
    - 'baseline_table': Phase 1 metrics by clinic
    - 'predictor_table': Phase 2 metrics by clinic
    - 'threshold_comparison': threshold sweep results
    """
    baseline = [r for r in results if r.model_type == "baseline"]
    predictor = [r for r in results if r.model_type == "predictor"]

    return {
        "baseline_results": baseline,
        "predictor_results": predictor,
        "all_results": results,
    }


def find_optimal_threshold(
    results: List[EvalResult],
    clinic_name: str,
    model_type: str = "predictor",
    target_utilization: float = 0.95,
    max_collision_rate: float = 0.15,
) -> Optional[EvalResult]:
    """Find threshold closest to target utilization within collision limit.

    Returns the EvalResult with the best threshold for the given clinic.
    """
    clinic_results = [
        r for r in results
        if r.clinic_name == clinic_name and r.model_type == model_type
    ]

    eligible = [
        r for r in clinic_results
        if r.collision_rate <= max_collision_rate
    ]

    if not eligible:
        return min(clinic_results, key=lambda r: r.collision_rate)

    return min(
        eligible,
        key=lambda r: abs(r.utilization_rate - target_utilization),
    )
