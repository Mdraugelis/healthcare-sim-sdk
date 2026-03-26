"""No-Show Overbooking Scenario — Reference Implementation #2.

Simulates a clinic where an ML model predicts patient no-shows.
High no-show probability slots are double-booked, improving
utilization but potentially increasing wait times and creating
equity concerns through compounding overbooking burden.

Unit of analysis: appointment
State: NoShowState dataclass (patient dict + schedule list + counters)

This scenario validates SDK generality: fundamentally different state
representation (dataclass graph vs numpy arrays), different unit of
analysis, multiple outcome dimensions, and compounding effects.
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
from sdk.ml.probability_model import ControlledProbabilityModel


@dataclass
class ClinicConfig:
    """Configuration for a clinic."""
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
    subgroup: str = "unknown"


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

    All mutable data lives here — not on the scenario object —
    to satisfy the step purity contract.
    """
    day: int
    patients: Dict[int, Patient]
    schedule: List[AppointmentSlot]
    clinic_config: ClinicConfig
    overbook_budget: Dict[int, int]
    total_slots_resolved: int = 0
    total_noshows: int = 0
    total_collisions: int = 0
    total_overbooked_slots: int = 0


class NoShowOverbookingScenario(BaseScenario["NoShowState"]):
    """No-show prediction with overbooking policy simulation.

    State is a NoShowState dataclass containing patients, schedule,
    and counters. step() uses only rng.temporal and the passed-in
    state for purity.
    """

    unit_of_analysis = "appointment"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        n_patients: int = 1000,
        base_noshow_rate: float = 0.12,
        noshow_concentration: float = 0.3,
        noshow_variability: float = 0.03,
        model_auc: float = 0.75,
        overbooking_threshold: float = 0.30,
        max_individual_overbooks: int = 3,
        intervention_effectiveness: float = 1.0,
        clinic_config: Optional[ClinicConfig] = None,
    ):
        super().__init__(time_config=time_config, seed=seed)

        self.n_patients = n_patients
        self.base_noshow_rate = base_noshow_rate
        self.noshow_concentration = noshow_concentration
        self.noshow_variability = noshow_variability
        self.overbooking_threshold = overbooking_threshold
        self.max_individual_overbooks = max_individual_overbooks
        self.intervention_effectiveness = intervention_effectiveness
        self.clinic_config = clinic_config or ClinicConfig()

        self._model = ControlledProbabilityModel(target_auc=model_auc)

    def create_population(self, n_entities: int) -> NoShowState:
        """Create patient panel and initial schedule."""
        rng = self.rng.population
        cc = self.clinic_config

        # Beta-distributed no-show probabilities
        alpha = self.noshow_concentration
        beta_p = alpha * (1 / self.base_noshow_rate - 1)
        raw_probs = rng.beta(alpha, beta_p, self.n_patients)
        scaling = self.base_noshow_rate / np.mean(raw_probs)
        base_probs = np.clip(raw_probs * scaling, 0.01, 0.80)

        # Subgroup assignments with disparity multipliers
        subgroups = rng.choice(
            ["group_A", "group_B", "group_C", "group_D"],
            size=self.n_patients,
            p=[0.55, 0.20, 0.15, 0.10],
        )
        multipliers = {
            "group_A": 0.85, "group_B": 1.10,
            "group_C": 1.30, "group_D": 1.50,
        }
        for i, sg in enumerate(subgroups):
            base_probs[i] *= multipliers[sg]
        base_probs = np.clip(base_probs, 0.01, 0.80)

        # Create patients
        patients: Dict[int, Patient] = {}
        for i in range(self.n_patients):
            past_appts = int(rng.poisson(6))
            past_noshows = int(rng.binomial(
                max(past_appts, 0), base_probs[i]
            ))
            patients[i] = Patient(
                patient_id=i,
                base_noshow_probability=float(base_probs[i]),
                n_past_appointments=past_appts,
                n_past_noshows=past_noshows,
                subgroup=str(subgroups[i]),
            )

        # Initial schedule (uses temporal RNG via helper)
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
            clinic_config=cc,
            overbook_budget=overbook_budget,
        )

    def step(self, state: NoShowState, t: int) -> NoShowState:
        """Resolve appointments, update histories, generate new schedule.

        PURITY: uses only self.rng.temporal and state. All mutable
        counters and patient histories live in the state object.
        """
        rng = self.rng.temporal

        # Resolve current schedule
        for slot in state.schedule:
            if slot.resolved:
                continue

            # Original patient show/no-show
            slot.original_showed = (
                rng.random() >= slot.true_noshow_prob
            )

            # Overbooked patient show/no-show
            if (slot.is_overbooked
                    and slot.overbooked_patient_id is not None):
                ob_patient = state.patients[slot.overbooked_patient_id]
                ob_prob = ob_patient.base_noshow_probability
                ob_prob += rng.normal(0, self.noshow_variability)
                ob_prob = np.clip(ob_prob, 0.01, 0.80)
                slot.overbooked_showed = (rng.random() >= ob_prob)

                # Collision: both showed
                if slot.original_showed and slot.overbooked_showed:
                    slot.wait_time_minutes = float(
                        state.clinic_config.appointment_duration_minutes
                        + state.clinic_config.buffer_minutes
                    )
                    state.total_collisions += 1

                # Track overbooking burden
                ob_patient.n_times_overbooked += 1
                state.total_overbooked_slots += 1

            # Update patient history
            patient = state.patients[slot.patient_id]
            patient.n_past_appointments += 1
            if not slot.original_showed:
                patient.n_past_noshows += 1
                state.total_noshows += 1

            state.total_slots_resolved += 1
            slot.resolved = True

        # Generate new schedule for day t
        state.day = t
        state.schedule = _generate_schedule(
            state.patients, day=t,
            clinic_config=state.clinic_config,
            noshow_variability=self.noshow_variability,
            rng=rng,
        )

        # Reset daily overbook budget
        state.overbook_budget = {
            pid: state.clinic_config.max_overbook_per_provider
            for pid in range(state.clinic_config.n_providers)
        }

        return state

    def predict(self, state: NoShowState, t: int) -> Predictions:
        """Run probability model on current schedule."""
        true_probs = np.array([
            s.true_noshow_prob for s in state.schedule
        ])

        predicted = self._model.predict(true_probs, self.rng.prediction)

        for i, slot in enumerate(state.schedule):
            slot.predicted_noshow_prob = float(predicted[i])

        return Predictions(
            scores=predicted,
            metadata={
                "slot_ids": [s.slot_id for s in state.schedule],
                "true_probs": true_probs,
            },
        )

    def intervene(
        self, state: NoShowState, predictions: Predictions, t: int,
    ) -> tuple[NoShowState, Interventions]:
        """Overbook high no-show probability slots."""
        rng = self.rng.intervention
        overbooked_indices: List[int] = []

        # Sort by predicted no-show prob (highest first)
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

            # Find candidate not already scheduled today
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
        """Record appointment-level outcomes."""
        slot_ids = np.array([s.slot_id for s in state.schedule])
        noshows = np.array([
            0.0 if s.original_showed else 1.0
            for s in state.schedule
        ])

        utilized = np.array([
            1.0 if s.original_showed
            else (1.0 if s.is_overbooked and s.overbooked_showed else 0.0)
            for s in state.schedule
        ])

        wait_times = np.array([
            s.wait_time_minutes for s in state.schedule
        ])

        # Per-patient overbooking burden (for equity analysis)
        patient_ids = np.array([
            s.patient_id for s in state.schedule
        ])
        patient_subgroups = np.array([
            state.patients[s.patient_id].subgroup
            for s in state.schedule
        ])

        return Outcomes(
            events=noshows,
            entity_ids=slot_ids,
            secondary={
                "utilization": utilized,
                "wait_times": wait_times,
                "subgroup": patient_subgroups,
            },
            metadata={
                "patient_ids": patient_ids,
                "total_collisions": state.total_collisions,
                "total_overbooked": state.total_overbooked_slots,
                "mean_overbooking_burden": np.mean([
                    p.n_times_overbooked
                    for p in state.patients.values()
                ]),
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
