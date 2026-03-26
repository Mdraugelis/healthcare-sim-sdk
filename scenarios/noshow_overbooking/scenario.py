"""No-Show Overbooking Scenario — Reference Implementation #2.

Simulates a clinic where a predictive model identifies likely no-shows.
High no-show probability slots are double-booked, improving utilization
but potentially increasing wait times and creating equity concerns
through compounding overbooking burden.

Supports two model types:
- 'baseline': Uses patient's historical no-show rate as the predictor
- 'predictor': Uses a ControlledProbabilityModel (simulates Epic-like ML)

Unit of analysis: appointment
State: NoShowState dataclass (patient dict + schedule list + counters)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from sdk.ml.model import ControlledMLModel


# -- Demographic distributions (realistic clinical proportions) -----------

RACE_ETHNICITY = {
    "White": {"prob": 0.58, "noshow_mult": 0.90},
    "Black": {"prob": 0.13, "noshow_mult": 1.35},
    "Hispanic": {"prob": 0.18, "noshow_mult": 1.25},
    "Asian": {"prob": 0.06, "noshow_mult": 0.80},
    "Other": {"prob": 0.05, "noshow_mult": 1.10},
}

INSURANCE_TYPE = {
    "Commercial": {"prob": 0.45, "noshow_mult": 0.80},
    "Medicare": {"prob": 0.25, "noshow_mult": 0.95},
    "Medicaid": {"prob": 0.20, "noshow_mult": 1.40},
    "Self-Pay": {"prob": 0.10, "noshow_mult": 1.60},
}

AGE_BAND = {
    "18-29": {"prob": 0.15, "noshow_mult": 1.40},
    "30-44": {"prob": 0.22, "noshow_mult": 1.15},
    "45-64": {"prob": 0.35, "noshow_mult": 0.90},
    "65+": {"prob": 0.28, "noshow_mult": 0.80},
}


@dataclass
class ClinicConfig:
    """Configuration for a clinic."""
    name: str = "Primary Care"
    n_providers: int = 8
    slots_per_provider_per_day: int = 12
    max_overbook_per_provider: int = 2
    appointment_duration_minutes: int = 20
    buffer_minutes: int = 10


@dataclass
class Patient:
    """Persistent patient attributes across appointments."""
    patient_id: int
    base_noshow_probability: float
    n_past_appointments: int = 0
    n_past_noshows: int = 0
    n_times_overbooked: int = 0
    race_ethnicity: str = "Unknown"
    insurance_type: str = "Unknown"
    age_band: str = "Unknown"
    campus: str = "Unknown"

    @property
    def historical_noshow_rate(self) -> float:
        """Patient's own historical no-show rate."""
        if self.n_past_appointments == 0:
            return 0.13  # population default
        return self.n_past_noshows / self.n_past_appointments


@dataclass
class AppointmentSlot:
    """A single appointment slot in the daily schedule."""
    slot_id: int
    day: int
    provider_id: int
    patient_id: int
    true_noshow_prob: float
    predicted_noshow_prob: float = 0.0
    is_overbooked: bool = False
    overbooked_patient_id: Optional[int] = None
    original_showed: bool = False
    overbooked_showed: bool = False
    wait_time_minutes: float = 0.0
    resolved: bool = False


@dataclass
class NoShowState:
    """Complete state for the no-show overbooking scenario.

    All mutable data lives here to satisfy the step purity contract.

    `resolved_slots` holds the previous day's slots AFTER resolution
    (show/no-show determined). `schedule` holds the current day's
    slots for predict/intervene. measure() reports from resolved_slots.
    """
    day: int
    patients: Dict[int, Patient]
    schedule: List[AppointmentSlot]  # current day (for predict/intervene)
    resolved_slots: List[AppointmentSlot]  # previous day (resolved)
    clinic_config: ClinicConfig
    overbook_budget: Dict[int, int]
    total_slots_resolved: int = 0
    total_noshows: int = 0
    total_collisions: int = 0
    total_overbooked_slots: int = 0
    total_overbooked_showed: int = 0


class NoShowOverbookingScenario(BaseScenario["NoShowState"]):
    """No-show prediction with overbooking policy simulation.

    Args:
        model_type: 'baseline' (patient historical rate) or
            'predictor' (ML model with target AUC).
        model_auc: Target AUC for 'predictor' mode.
        base_noshow_rate: Population-level no-show rate (default 0.13).
        overbooking_threshold: Predicted prob above which to overbook.
        campus: Clinic campus label for subgroup analysis.
    """

    unit_of_analysis = "appointment"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        n_patients: int = 2000,
        base_noshow_rate: float = 0.13,
        noshow_concentration: float = 0.3,
        noshow_variability: float = 0.03,
        model_type: str = "predictor",
        model_auc: float = 0.83,
        overbooking_threshold: float = 0.30,
        max_individual_overbooks: int = 5,
        clinic_config: Optional[ClinicConfig] = None,
        campus: str = "Main",
    ):
        super().__init__(time_config=time_config, seed=seed)

        self.n_patients = n_patients
        self.base_noshow_rate = base_noshow_rate
        self.noshow_concentration = noshow_concentration
        self.noshow_variability = noshow_variability
        self.model_type = model_type
        self.overbooking_threshold = overbooking_threshold
        self.max_individual_overbooks = max_individual_overbooks
        self.clinic_config = clinic_config or ClinicConfig()
        self.campus = campus

        self.model_auc = model_auc
        if model_type == "predictor":
            self._model = ControlledMLModel(
                mode="discrimination",
                target_auc=model_auc,
            )
        else:
            self._model = None
        self._model_fitted = False

    def create_population(self, n_entities: int) -> NoShowState:
        """Create patient panel with realistic demographics."""
        rng = self.rng.population
        cc = self.clinic_config

        # Beta-distributed base no-show probabilities
        alpha = self.noshow_concentration
        beta_p = alpha * (1 / self.base_noshow_rate - 1)
        raw_probs = rng.beta(alpha, beta_p, self.n_patients)
        scaling = self.base_noshow_rate / np.mean(raw_probs)
        base_probs = np.clip(raw_probs * scaling, 0.01, 0.80)

        # Assign demographics
        race_names = list(RACE_ETHNICITY.keys())
        race_probs = [RACE_ETHNICITY[r]["prob"] for r in race_names]
        races = rng.choice(race_names, self.n_patients, p=race_probs)

        ins_names = list(INSURANCE_TYPE.keys())
        ins_probs = [INSURANCE_TYPE[i]["prob"] for i in ins_names]
        insurances = rng.choice(ins_names, self.n_patients, p=ins_probs)

        age_names = list(AGE_BAND.keys())
        age_probs = [AGE_BAND[a]["prob"] for a in age_names]
        ages = rng.choice(age_names, self.n_patients, p=age_probs)

        # Apply demographic disparity multipliers
        for i in range(self.n_patients):
            mult = (
                RACE_ETHNICITY[races[i]]["noshow_mult"]
                * INSURANCE_TYPE[insurances[i]]["noshow_mult"]
                * AGE_BAND[ages[i]]["noshow_mult"]
            ) ** (1 / 3)  # geometric mean to avoid extreme compounding
            base_probs[i] *= mult
        base_probs = np.clip(base_probs, 0.01, 0.80)

        # Re-scale to hit target population rate
        scaling = self.base_noshow_rate / np.mean(base_probs)
        base_probs = np.clip(base_probs * scaling, 0.01, 0.80)

        # Create patients with history
        patients: Dict[int, Patient] = {}
        for i in range(self.n_patients):
            past_appts = int(rng.poisson(8))
            past_noshows = int(rng.binomial(
                max(past_appts, 0), base_probs[i]
            ))
            patients[i] = Patient(
                patient_id=i,
                base_noshow_probability=float(base_probs[i]),
                n_past_appointments=past_appts,
                n_past_noshows=past_noshows,
                race_ethnicity=str(races[i]),
                insurance_type=str(insurances[i]),
                age_band=str(ages[i]),
                campus=self.campus,
            )

        schedule = _generate_schedule(
            patients, day=0, clinic_config=cc,
            noshow_variability=self.noshow_variability,
            rng=self.rng.temporal,
        )

        overbook_budget = {
            pid: cc.max_overbook_per_provider
            for pid in range(cc.n_providers)
        }

        return NoShowState(
            day=0,
            patients=patients,
            schedule=schedule,
            resolved_slots=[],
            clinic_config=cc,
            overbook_budget=overbook_budget,
        )

    def step(self, state: NoShowState, t: int) -> NoShowState:
        """Resolve appointments, update histories, generate new schedule.

        PURITY: uses only self.rng.temporal and state.
        """
        rng = self.rng.temporal

        for slot in state.schedule:
            if slot.resolved:
                continue

            slot.original_showed = (
                rng.random() >= slot.true_noshow_prob
            )

            if (slot.is_overbooked
                    and slot.overbooked_patient_id is not None):
                ob_patient = state.patients[slot.overbooked_patient_id]
                ob_prob = ob_patient.base_noshow_probability
                ob_prob += rng.normal(0, self.noshow_variability)
                ob_prob = float(np.clip(ob_prob, 0.01, 0.80))
                slot.overbooked_showed = (rng.random() >= ob_prob)

                if slot.original_showed and slot.overbooked_showed:
                    slot.wait_time_minutes = float(
                        state.clinic_config.appointment_duration_minutes
                        + state.clinic_config.buffer_minutes
                    )
                    state.total_collisions += 1

                ob_patient.n_times_overbooked += 1
                state.total_overbooked_slots += 1
                if slot.overbooked_showed:
                    state.total_overbooked_showed += 1

            patient = state.patients[slot.patient_id]
            patient.n_past_appointments += 1
            if not slot.original_showed:
                patient.n_past_noshows += 1
                state.total_noshows += 1

            state.total_slots_resolved += 1
            slot.resolved = True

        # Save resolved slots for measure() to report on
        state.resolved_slots = list(state.schedule)

        state.day = t
        state.schedule = _generate_schedule(
            state.patients, day=t,
            clinic_config=state.clinic_config,
            noshow_variability=self.noshow_variability,
            rng=rng,
        )

        state.overbook_budget = {
            pid: state.clinic_config.max_overbook_per_provider
            for pid in range(state.clinic_config.n_providers)
        }

        return state

    def predict(self, state: NoShowState, t: int) -> Predictions:
        """Generate predictions using configured model type."""
        true_probs = np.array([
            s.true_noshow_prob for s in state.schedule
        ])

        if self.model_type == "baseline":
            # Use each patient's historical no-show rate
            predicted = np.array([
                state.patients[s.patient_id].historical_noshow_rate
                for s in state.schedule
            ])
            # Add small noise for variability
            noise = self.rng.prediction.normal(
                0, 0.02, len(predicted)
            )
            predicted = np.clip(predicted + noise, 0.01, 0.99)
        else:
            # Fit model on first prediction call
            if not self._model_fitted:
                true_labels = (
                    self.rng.prediction.random(len(true_probs))
                    < true_probs
                ).astype(int)
                self._model.fit(
                    true_labels, true_probs,
                    self.rng.prediction, n_iterations=3,
                )
                self._model_fitted = True

            predicted = self._model.predict(
                true_probs, self.rng.prediction,
            )

        for i, slot in enumerate(state.schedule):
            slot.predicted_noshow_prob = float(predicted[i])

        return Predictions(
            scores=predicted,
            metadata={
                "slot_ids": [s.slot_id for s in state.schedule],
                "true_probs": true_probs,
                "model_type": self.model_type,
            },
        )

    def intervene(
        self, state: NoShowState, predictions: Predictions, t: int,
    ) -> tuple[NoShowState, Interventions]:
        """Overbook high no-show probability slots with guardrails."""
        rng = self.rng.intervention
        overbooked_indices: List[int] = []

        scored = sorted(
            enumerate(state.schedule),
            key=lambda x: x[1].predicted_noshow_prob,
            reverse=True,
        )

        scheduled_ids = {s.patient_id for s in state.schedule}

        for idx, slot in scored:
            if slot.predicted_noshow_prob < self.overbooking_threshold:
                break

            if state.overbook_budget.get(slot.provider_id, 0) <= 0:
                continue

            candidates = [
                p for pid, p in state.patients.items()
                if pid not in scheduled_ids
                and p.n_times_overbooked < self.max_individual_overbooks
            ]
            if not candidates:
                continue

            candidate = candidates[int(rng.integers(len(candidates)))]

            slot.is_overbooked = True
            slot.overbooked_patient_id = candidate.patient_id
            state.overbook_budget[slot.provider_id] -= 1
            scheduled_ids.add(candidate.patient_id)
            overbooked_indices.append(idx)

        return state, Interventions(
            treated_indices=np.array(overbooked_indices, dtype=int),
            metadata={
                "n_overbooked": len(overbooked_indices),
                "overbooking_rate": (
                    len(overbooked_indices) / max(len(state.schedule), 1)
                ),
            },
        )

    def measure(self, state: NoShowState, t: int) -> Outcomes:
        """Record appointment-level outcomes with demographics.

        Reports from resolved_slots (previous day's resolved
        appointments). On day 0 before any resolution, reports
        from the current schedule (unresolved).
        """
        slots = (
            state.resolved_slots if state.resolved_slots
            else state.schedule
        )

        slot_ids = np.array([s.slot_id for s in slots])
        noshows = np.array([
            0.0 if s.original_showed else 1.0
            for s in slots
        ])

        utilized = np.array([
            1.0 if s.original_showed
            else (
                1.0 if s.is_overbooked and s.overbooked_showed
                else 0.0
            )
            for s in slots
        ])

        wait_times = np.array([
            s.wait_time_minutes for s in slots
        ])

        patient_ids = np.array([
            s.patient_id for s in slots
        ])

        race = np.array([
            state.patients[s.patient_id].race_ethnicity
            for s in slots
        ])
        insurance = np.array([
            state.patients[s.patient_id].insurance_type
            for s in slots
        ])
        age_band = np.array([
            state.patients[s.patient_id].age_band
            for s in slots
        ])

        return Outcomes(
            events=noshows,
            entity_ids=slot_ids,
            secondary={
                "utilization": utilized,
                "wait_times": wait_times,
                "race_ethnicity": race,
                "insurance_type": insurance,
                "age_band": age_band,
            },
            metadata={
                "patient_ids": patient_ids,
                "campus": self.campus,
                "total_collisions": state.total_collisions,
                "total_overbooked": state.total_overbooked_slots,
                "total_overbooked_showed": state.total_overbooked_showed,
                "total_noshows": state.total_noshows,
                "total_resolved": state.total_slots_resolved,
                "mean_overbooking_burden": float(np.mean([
                    p.n_times_overbooked
                    for p in state.patients.values()
                ])),
            },
        )


def _generate_schedule(
    patients: Dict[int, Patient],
    day: int,
    clinic_config: ClinicConfig,
    noshow_variability: float,
    rng: np.random.Generator,
) -> List[AppointmentSlot]:
    """Generate a day's appointment slots. Pure function."""
    cc = clinic_config
    total_slots = cc.n_providers * cc.slots_per_provider_per_day
    n_to_schedule = min(total_slots, len(patients))

    patient_ids = rng.choice(
        list(patients.keys()),
        size=n_to_schedule,
        replace=False,
    )

    schedule: List[AppointmentSlot] = []
    for i, pid in enumerate(patient_ids):
        patient = patients[pid]
        appt_prob = patient.base_noshow_probability + rng.normal(
            0, noshow_variability
        )
        appt_prob = float(np.clip(appt_prob, 0.01, 0.80))

        schedule.append(AppointmentSlot(
            slot_id=day * 10000 + i,
            day=day,
            provider_id=i % cc.n_providers,
            patient_id=pid,
            true_noshow_prob=appt_prob,
        ))
    return schedule
