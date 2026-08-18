"""No-Show Targeted Reminder Scenario.

Simulates ML-targeted phone reminders to reduce outpatient appointment
no-shows. High-risk patients identified by a predictive model receive
phone call reminders, which reduce their no-show probability.

Replicates findings from:
- Chong et al. (AJR, 2020): XGBoost AUC 0.74, top-25% targeting, 4.2x
  efficiency gain over random calling.
- Rosen et al. (JGIM, 2023): RCT showing ML-targeted reminders reduce
  racial disparities in no-show rates (Black patients benefit more).

Unit of analysis: appointment
State: ReminderState dataclass (patient dict + schedule list + counters)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from healthcare_sim_sdk.core.scenario import (
    BaseScenario,
    Interventions,
    Outcomes,
    Predictions,
    TimeConfig,
)
from healthcare_sim_sdk.ml.model import ControlledMLModel


# -- Default demographic distributions (Rosen VA population) ---------------

DEFAULT_RACE_ETHNICITY: Dict[str, Dict[str, float]] = {
    "White": {"prob": 0.55, "noshow_mult": 0.86},
    "Black": {"prob": 0.30, "noshow_mult": 1.17},
    "Hispanic": {"prob": 0.10, "noshow_mult": 1.05},
    "Asian": {"prob": 0.02, "noshow_mult": 0.80},
    "Other": {"prob": 0.03, "noshow_mult": 1.00},
}

DEFAULT_INSURANCE_TYPE: Dict[str, Dict[str, float]] = {
    "Commercial": {"prob": 0.45, "noshow_mult": 0.80},
    "Medicare": {"prob": 0.25, "noshow_mult": 0.95},
    "Medicaid": {"prob": 0.20, "noshow_mult": 1.40},
    "Self-Pay": {"prob": 0.10, "noshow_mult": 1.60},
}

DEFAULT_AGE_BAND: Dict[str, Dict[str, float]] = {
    "18-29": {"prob": 0.15, "noshow_mult": 1.40},
    "30-44": {"prob": 0.22, "noshow_mult": 1.15},
    "45-64": {"prob": 0.35, "noshow_mult": 0.90},
    "65+": {"prob": 0.28, "noshow_mult": 0.80},
}


@dataclass
class CallerConfig:
    """Configuration for the reminder calling program."""
    name: str = "Outpatient Reminder Program"
    n_providers: int = 8
    slots_per_provider_per_day: int = 12
    call_capacity_per_day: int = 24
    call_success_rate: float = 0.65
    reminder_effectiveness: float = 0.35
    max_calls_per_patient_per_period: int = 1


@dataclass
class Patient:
    """Persistent patient state across appointments."""
    patient_id: int
    base_noshow_probability: float
    n_past_appointments: int = 0
    n_past_noshows: int = 0
    n_calls_received: int = 0
    n_calls_reached: int = 0
    was_called_today: bool = False
    was_reached_today: bool = False
    reminder_effect_active: bool = False
    race_ethnicity: str = "Unknown"
    insurance_type: str = "Unknown"
    age_band: str = "Unknown"

    @property
    def historical_noshow_rate(self) -> float:
        """Patient's own historical no-show rate."""
        if self.n_past_appointments == 0:
            return 0.20  # population default
        return self.n_past_noshows / self.n_past_appointments


@dataclass
class AppointmentSlot:
    """Single appointment slot."""
    slot_id: int
    day: int
    provider_id: int
    patient_id: int
    true_noshow_prob: float
    effective_noshow_prob: float
    predicted_noshow_prob: float = 0.0
    was_called: bool = False
    was_reached: bool = False
    showed: bool = False
    resolved: bool = False


@dataclass
class ReminderState:
    """Complete scenario state."""
    day: int
    patients: Dict[int, Patient]
    schedule: List[AppointmentSlot]
    resolved_slots: List[AppointmentSlot]
    caller_config: CallerConfig
    total_slots_resolved: int = 0
    total_noshows: int = 0
    total_calls_made: int = 0
    total_calls_reached: int = 0
    total_reminded_showed: int = 0
    total_reminded_noshowed: int = 0


class NoShowTargetedReminderScenario(BaseScenario["ReminderState"]):
    """No-show prediction with targeted phone reminder intervention.

    Args:
        model_type: 'baseline' (patient historical rate) or
            'predictor' (ML model with target AUC).
        model_auc: Target AUC for 'predictor' mode.
        base_noshow_rate: Population-level no-show rate.
        targeting_mode: 'top_k', 'top_fraction', or 'threshold'.
        targeting_fraction: Fraction of schedule to target (top_fraction
            mode only).
        targeting_threshold: Probability threshold for calling (threshold
            mode only).
    """

    unit_of_analysis = "appointment"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        n_patients: int = 5000,
        base_noshow_rate: float = 0.36,
        noshow_concentration: float = 0.3,
        noshow_variability: float = 0.03,
        model_type: str = "predictor",
        model_auc: float = 0.72,
        targeting_mode: str = "top_k",
        targeting_fraction: float = 0.25,
        targeting_threshold: float = 0.30,
        caller_config: Optional[CallerConfig] = None,
        race_ethnicity: Optional[Dict[str, Dict[str, float]]] = None,
        insurance_type: Optional[Dict[str, Dict[str, float]]] = None,
        age_band: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        super().__init__(time_config=time_config, seed=seed)

        self.n_patients = n_patients
        self.base_noshow_rate = base_noshow_rate
        self.noshow_concentration = noshow_concentration
        self.noshow_variability = noshow_variability
        self.model_type = model_type
        self.model_auc = model_auc
        self.targeting_mode = targeting_mode
        self.targeting_fraction = targeting_fraction
        self.targeting_threshold = targeting_threshold
        self.caller_config = caller_config or CallerConfig()

        self._race_ethnicity = race_ethnicity or DEFAULT_RACE_ETHNICITY
        self._insurance_type = insurance_type or DEFAULT_INSURANCE_TYPE
        self._age_band = age_band or DEFAULT_AGE_BAND

        if model_type == "predictor":
            self._model: Optional[ControlledMLModel] = ControlledMLModel(
                mode="discrimination",
                target_auc=model_auc,
            )
        else:
            self._model = None
        self._model_fitted = False

    def create_population(self, n_entities: int) -> ReminderState:
        """Create patient panel with realistic demographics."""
        rng = self.rng.population
        cc = self.caller_config

        # Beta-distributed base no-show probabilities
        alpha = self.noshow_concentration
        beta_p = alpha * (1 / self.base_noshow_rate - 1)
        raw_probs = rng.beta(alpha, beta_p, self.n_patients)
        scaling = self.base_noshow_rate / np.mean(raw_probs)
        base_probs = np.clip(raw_probs * scaling, 0.01, 0.80)

        # Assign demographics
        race_names = list(self._race_ethnicity.keys())
        race_probs = [self._race_ethnicity[r]["prob"] for r in race_names]
        races = rng.choice(race_names, self.n_patients, p=race_probs)

        ins_names = list(self._insurance_type.keys())
        ins_probs = [self._insurance_type[i]["prob"] for i in ins_names]
        insurances = rng.choice(ins_names, self.n_patients, p=ins_probs)

        age_names = list(self._age_band.keys())
        age_probs = [self._age_band[a]["prob"] for a in age_names]
        ages = rng.choice(age_names, self.n_patients, p=age_probs)

        # Apply demographic disparity multipliers (geometric mean)
        for i in range(self.n_patients):
            mult = (
                self._race_ethnicity[races[i]]["noshow_mult"]
                * self._insurance_type[insurances[i]]["noshow_mult"]
                * self._age_band[ages[i]]["noshow_mult"]
            ) ** (1 / 3)
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
            )

        # Initial schedule uses population RNG (not temporal)
        schedule = _generate_schedule(
            patients, day=0, caller_config=cc,
            noshow_variability=self.noshow_variability,
            rng=rng,
        )

        return ReminderState(
            day=0,
            patients=patients,
            schedule=schedule,
            resolved_slots=[],
            caller_config=cc,
        )

    def step(self, state: ReminderState, t: int) -> ReminderState:
        """Resolve appointments, update histories, generate new schedule.

        PURITY: uses only self.rng.temporal and state.
        KEY DIFFERENCE from overbooking: resolves using effective_noshow_prob
        (which incorporates reminder effect from intervene()).
        """
        rng = self.rng.temporal

        for slot in state.schedule:
            if slot.resolved:
                continue

            # Use effective_noshow_prob — this is the intervention mechanism
            slot.showed = (rng.random() >= slot.effective_noshow_prob)

            patient = state.patients[slot.patient_id]
            patient.n_past_appointments += 1
            if not slot.showed:
                patient.n_past_noshows += 1
                state.total_noshows += 1
                if slot.was_called and slot.was_reached:
                    state.total_reminded_noshowed += 1
            else:
                if slot.was_called and slot.was_reached:
                    state.total_reminded_showed += 1

            state.total_slots_resolved += 1
            slot.resolved = True

        # Reset reminder flags on all patients
        for patient in state.patients.values():
            patient.was_called_today = False
            patient.was_reached_today = False
            patient.reminder_effect_active = False

        # Save resolved slots for measure() to report on
        state.resolved_slots = list(state.schedule)

        state.day = t
        state.schedule = _generate_schedule(
            state.patients, day=t,
            caller_config=state.caller_config,
            noshow_variability=self.noshow_variability,
            rng=rng,
        )

        return state

    def predict(self, state: ReminderState, t: int) -> Predictions:
        """Generate predictions using configured model type."""
        true_probs = np.array([
            s.true_noshow_prob for s in state.schedule
        ])

        if self.model_type == "baseline":
            predicted = np.array([
                state.patients[s.patient_id].historical_noshow_rate
                for s in state.schedule
            ])
            noise = self.rng.prediction.normal(
                0, 0.02, len(predicted)
            )
            predicted = np.clip(predicted + noise, 0.01, 0.99)
        else:
            if not self._model_fitted:
                true_labels = (
                    self.rng.prediction.random(len(true_probs))
                    < true_probs
                ).astype(int)
                assert self._model is not None
                self._model.fit(
                    true_labels, true_probs,
                    self.rng.prediction, n_iterations=3,
                )
                self._model_fitted = True

            assert self._model is not None
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
        self, state: ReminderState, predictions: Predictions, t: int,
    ) -> tuple[ReminderState, Interventions]:
        """Call high-risk patients to reduce their no-show probability.

        Targeting modes:
        - top_k: Call top call_capacity_per_day highest-risk patients.
        - top_fraction: Call top X% of schedule, capped by capacity.
        - threshold: Call everyone above a probability threshold, capped.
        """
        rng = self.rng.intervention
        cc = state.caller_config
        called_indices: List[int] = []
        n_reached = 0

        # Sort schedule indices by predicted risk (descending)
        scored = sorted(
            enumerate(state.schedule),
            key=lambda x: x[1].predicted_noshow_prob,
            reverse=True,
        )

        # Determine how many to call based on targeting mode
        n_scheduled = len(state.schedule)
        if self.targeting_mode == "top_k":
            n_to_call = min(cc.call_capacity_per_day, n_scheduled)
        elif self.targeting_mode == "top_fraction":
            n_to_call = min(
                int(self.targeting_fraction * n_scheduled),
                cc.call_capacity_per_day,
            )
        elif self.targeting_mode == "threshold":
            n_above = sum(
                1 for _, s in scored
                if s.predicted_noshow_prob >= self.targeting_threshold
            )
            n_to_call = min(n_above, cc.call_capacity_per_day)
        else:
            n_to_call = 0

        calls_made = 0
        for idx, slot in scored:
            if calls_made >= n_to_call:
                break

            # Per-patient constraint
            patient = state.patients[slot.patient_id]
            if patient.was_called_today:
                continue

            # Mark as called
            slot.was_called = True
            patient.was_called_today = True
            patient.n_calls_received += 1
            state.total_calls_made += 1
            calls_made += 1
            called_indices.append(idx)

            # Determine if call reaches patient
            reached = rng.random() < cc.call_success_rate
            if reached:
                slot.was_reached = True
                patient.was_reached_today = True
                patient.n_calls_reached += 1
                patient.reminder_effect_active = True
                state.total_calls_reached += 1
                n_reached += 1

                # Apply reminder effect: multiplicative reduction
                slot.effective_noshow_prob = float(np.clip(
                    slot.true_noshow_prob * (1 - cc.reminder_effectiveness),
                    0.01, 0.99,
                ))

        return state, Interventions(
            treated_indices=np.array(called_indices, dtype=int),
            metadata={
                "n_called": calls_made,
                "n_reached": n_reached,
                "targeting_mode": self.targeting_mode,
                "call_capacity": cc.call_capacity_per_day,
            },
        )

    def measure(self, state: ReminderState, t: int) -> Outcomes:
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
            0.0 if s.showed else 1.0
            for s in slots
        ])

        was_called = np.array([
            1.0 if s.was_called else 0.0
            for s in slots
        ])
        was_reached = np.array([
            1.0 if s.was_reached else 0.0
            for s in slots
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
                "was_called": was_called,
                "was_reached": was_reached,
                "race_ethnicity": race,
                "insurance_type": insurance,
                "age_band": age_band,
            },
            metadata={
                "total_noshows": state.total_noshows,
                "total_resolved": state.total_slots_resolved,
                "total_calls_made": state.total_calls_made,
                "total_calls_reached": state.total_calls_reached,
                "total_reminded_showed": state.total_reminded_showed,
                "total_reminded_noshowed": state.total_reminded_noshowed,
            },
        )


def _generate_schedule(
    patients: Dict[int, Patient],
    day: int,
    caller_config: CallerConfig,
    noshow_variability: float,
    rng: np.random.Generator,
) -> List[AppointmentSlot]:
    """Generate a day's appointment slots. Pure function."""
    cc = caller_config
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
            effective_noshow_prob=appt_prob,  # No intervention yet
        ))
    return schedule
