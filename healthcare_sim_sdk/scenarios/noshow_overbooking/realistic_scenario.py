"""Realistic No-Show Overbooking Scenario.

Models a clinic with:
- Visit-frequency-weighted daily panels (chronic patients appear often)
- Accumulating waitlist (unmet demand carries over day-to-day)
- Overbooking policy that fills from the waitlist
- Two prediction modes: baseline (historical rate >= threshold)
  and ML predictor (configurable AUC)

The core question: at what overbooking threshold do we achieve good
utilization without excessive collisions, and how does the ML predictor
improve upon the staff's current 50% historical rate threshold?

Unit of analysis: appointment
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


# -- Demographic distributions -------------------------------------------

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

# Visit frequency: how often each type is scheduled
VISIT_TYPES = {
    "chronic": {
        "prob": 0.20,
        "mean_interval_days": 21,   # every 2-4 weeks
        "noshow_mult": 1.15,        # slightly higher no-show
    },
    "routine": {
        "prob": 0.50,
        "mean_interval_days": 75,   # every 2-3 months
        "noshow_mult": 1.00,
    },
    "infrequent": {
        "prob": 0.30,
        "mean_interval_days": 240,  # 1-2x per year
        "noshow_mult": 0.85,        # lower no-show (planned visits)
    },
}


@dataclass
class ClinicConfig:
    """Clinic configuration."""
    name: str = "Primary Care"
    n_providers: int = 8
    slots_per_provider_per_day: int = 12
    max_overbook_per_provider: int = 2
    appointment_duration_minutes: int = 20
    buffer_minutes: int = 10
    new_waitlist_requests_per_day: int = 5


@dataclass
class Patient:
    """Patient with visit history and demographics."""
    patient_id: int
    base_noshow_prob: float
    visit_type: str  # chronic, routine, infrequent
    daily_schedule_weight: float  # probability of appearing on any given day
    n_past_appointments: int = 0
    n_past_noshows: int = 0
    n_times_overbooked: int = 0
    last_scheduled_day: int = -999
    race_ethnicity: str = "Unknown"
    insurance_type: str = "Unknown"
    age_band: str = "Unknown"

    @property
    def historical_noshow_rate(self) -> float:
        if self.n_past_appointments < 3:
            return 0.13  # population default when insufficient history
        return self.n_past_noshows / self.n_past_appointments


@dataclass
class Slot:
    """An appointment slot for one day."""
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
class WaitlistEntry:
    """A patient waiting for an appointment."""
    patient_id: int
    request_day: int
    priority: str = "routine"  # urgent, routine


@dataclass
class ClinicState:
    """Complete state. All mutable data lives here (step purity)."""
    day: int
    patients: Dict[int, Patient]
    schedule: List[Slot]
    resolved_slots: List[Slot]
    waitlist: List[WaitlistEntry]
    clinic_config: ClinicConfig
    overbook_budget: Dict[int, int]

    # AR(1) modifiers: per-patient temporal drift of no-show probability.
    # Stored in state (not Patient) for step purity.
    # current_noshow_prob = base_noshow_prob * noshow_modifiers[pid]
    noshow_modifiers: Dict[int, float] = None  # type: ignore

    # Cumulative counters
    total_slots_resolved: int = 0
    total_noshows: int = 0
    total_collisions: int = 0
    total_overbooked: int = 0
    total_overbooked_showed: int = 0
    total_waitlist_served: int = 0
    cumulative_wait_days: int = 0


class RealisticNoShowScenario(BaseScenario["ClinicState"]):
    """No-show overbooking with visit-frequency scheduling and waitlist.

    Two prediction modes:
    - 'baseline': Staff uses patient historical no-show rate >= threshold
    - 'predictor': ML model with configurable AUC

    Args:
        n_patients: Size of the patient panel.
        base_noshow_rate: Population average no-show rate.
        model_type: 'baseline' or 'predictor'.
        model_auc: Target AUC for predictor mode.
        overbooking_threshold: Predicted prob above which to overbook.
        max_individual_overbooks: Per-patient cap over the simulation.
    """

    unit_of_analysis = "appointment"

    def __init__(
        self,
        time_config: TimeConfig,
        seed: Optional[int] = None,
        n_patients: int = 3000,
        base_noshow_rate: float = 0.13,
        noshow_concentration: float = 0.3,
        noshow_variability: float = 0.03,
        model_type: str = "baseline",
        model_auc: float = 0.83,
        overbooking_threshold: float = 0.50,
        max_individual_overbooks: int = 10,
        overbooking_policy: str = "threshold",
        clinic_config: Optional[ClinicConfig] = None,
        ar1_rho: float = 0.95,
        ar1_sigma: float = 0.04,
    ):
        super().__init__(time_config=time_config, seed=seed)
        self.n_patients = n_patients
        self.base_noshow_rate = base_noshow_rate
        self.noshow_concentration = noshow_concentration
        self.noshow_variability = noshow_variability
        self.model_type = model_type
        self.model_auc = model_auc
        self.overbooking_threshold = overbooking_threshold
        self.max_individual_overbooks = max_individual_overbooks
        self.overbooking_policy = overbooking_policy
        self.clinic_config = clinic_config or ClinicConfig()
        self.ar1_rho = ar1_rho
        self.ar1_sigma = ar1_sigma

        if model_type == "predictor":
            self._model = ControlledMLModel(
                mode="discrimination", target_auc=model_auc,
            )
        else:
            self._model = None
        self._model_fitted = False

    def create_population(self, n_entities: int) -> ClinicState:
        rng = self.rng.population
        cc = self.clinic_config

        # Assign visit types
        vt_names = list(VISIT_TYPES.keys())
        vt_probs = [VISIT_TYPES[v]["prob"] for v in vt_names]
        visit_types = rng.choice(vt_names, self.n_patients, p=vt_probs)

        # Base no-show probabilities (beta distributed)
        alpha = self.noshow_concentration
        beta_p = alpha * (1 / self.base_noshow_rate - 1)
        raw_probs = rng.beta(alpha, beta_p, self.n_patients)
        scaling = self.base_noshow_rate / np.mean(raw_probs)
        base_probs = np.clip(raw_probs * scaling, 0.01, 0.80)

        # Demographics
        race_names = list(RACE_ETHNICITY.keys())
        race_probs = [RACE_ETHNICITY[r]["prob"] for r in race_names]
        races = rng.choice(race_names, self.n_patients, p=race_probs)

        ins_names = list(INSURANCE_TYPE.keys())
        ins_probs = [INSURANCE_TYPE[i]["prob"] for i in ins_names]
        insurances = rng.choice(ins_names, self.n_patients, p=ins_probs)

        age_names = list(AGE_BAND.keys())
        age_probs = [AGE_BAND[a]["prob"] for a in age_names]
        ages = rng.choice(age_names, self.n_patients, p=age_probs)

        # Apply demographic + visit type multipliers
        for i in range(self.n_patients):
            demo_mult = (
                RACE_ETHNICITY[races[i]]["noshow_mult"]
                * INSURANCE_TYPE[insurances[i]]["noshow_mult"]
                * AGE_BAND[ages[i]]["noshow_mult"]
            ) ** (1 / 3)
            vt_mult = VISIT_TYPES[visit_types[i]]["noshow_mult"]
            base_probs[i] *= demo_mult * vt_mult
        base_probs = np.clip(base_probs, 0.01, 0.80)

        # Rescale to target rate
        scaling = self.base_noshow_rate / np.mean(base_probs)
        base_probs = np.clip(base_probs * scaling, 0.01, 0.80)

        # Create patients with scheduling weights
        patients: Dict[int, Patient] = {}
        for i in range(self.n_patients):
            vt = str(visit_types[i])
            interval = VISIT_TYPES[vt]["mean_interval_days"]
            # Daily probability of being scheduled ≈ 1/interval
            daily_weight = 1.0 / interval

            past_appts = int(rng.poisson(3))
            past_noshows = int(rng.binomial(
                max(past_appts, 0), base_probs[i]
            ))
            patients[i] = Patient(
                patient_id=i,
                base_noshow_prob=float(base_probs[i]),
                visit_type=vt,
                daily_schedule_weight=daily_weight,
                n_past_appointments=past_appts,
                n_past_noshows=past_noshows,
                race_ethnicity=str(races[i]),
                insurance_type=str(insurances[i]),
                age_band=str(ages[i]),
            )

        # Initialize AR(1) modifiers at 1.0 (no drift yet)
        noshow_modifiers = {i: 1.0 for i in range(self.n_patients)}

        # Generate initial schedule using POPULATION rng
        schedule = _build_daily_schedule(
            patients, day=0, clinic_config=cc,
            noshow_variability=self.noshow_variability,
            noshow_modifiers=noshow_modifiers, rng=rng,
        )

        waitlist: List[WaitlistEntry] = []
        overbook_budget = {
            pid: cc.max_overbook_per_provider
            for pid in range(cc.n_providers)
        }

        return ClinicState(
            day=0, patients=patients, schedule=schedule,
            resolved_slots=[], waitlist=waitlist,
            clinic_config=cc, overbook_budget=overbook_budget,
            noshow_modifiers=noshow_modifiers,
        )

    def step(self, state: ClinicState, t: int) -> ClinicState:
        """Resolve appointments, update histories, add waitlist requests,
        generate new schedule.

        PURITY: uses only self.rng.temporal and state.
        """
        rng = self.rng.temporal
        cc = state.clinic_config

        # 1. Resolve current schedule
        for slot in state.schedule:
            if slot.resolved:
                continue

            slot.original_showed = (
                rng.random() >= slot.true_noshow_prob
            )

            if (slot.is_overbooked
                    and slot.overbooked_patient_id is not None):
                ob_patient = state.patients[slot.overbooked_patient_id]
                ob_mod = state.noshow_modifiers.get(
                    slot.overbooked_patient_id, 1.0
                )
                ob_prob = ob_patient.base_noshow_prob * ob_mod
                ob_prob += rng.normal(0, self.noshow_variability)
                ob_prob = float(np.clip(ob_prob, 0.01, 0.80))
                slot.overbooked_showed = (rng.random() >= ob_prob)

                if slot.original_showed and slot.overbooked_showed:
                    slot.wait_time_minutes = float(
                        cc.appointment_duration_minutes
                        + cc.buffer_minutes
                    )
                    state.total_collisions += 1

                ob_patient.n_times_overbooked += 1
                state.total_overbooked += 1
                if slot.overbooked_showed:
                    state.total_overbooked_showed += 1

            # Update patient history
            patient = state.patients[slot.patient_id]
            patient.n_past_appointments += 1
            patient.last_scheduled_day = state.day
            if not slot.original_showed:
                patient.n_past_noshows += 1
                state.total_noshows += 1
            state.total_slots_resolved += 1
            slot.resolved = True

        state.resolved_slots = list(state.schedule)

        # 2. Evolve AR(1) modifiers for ALL patients
        # Behavior drifts every day, even when not scheduled.
        # modifier_t = rho * modifier_{t-1} + (1-rho) * 1.0 + N(0, sigma)
        for pid in state.noshow_modifiers:
            mod = state.noshow_modifiers[pid]
            noise = rng.normal(0, self.ar1_sigma)
            mod = self.ar1_rho * mod + (1 - self.ar1_rho) * 1.0 + noise
            state.noshow_modifiers[pid] = float(np.clip(mod, 0.5, 2.0))

        # 4. New waitlist requests arrive
        n_new = int(rng.poisson(cc.new_waitlist_requests_per_day))
        # New requests come from patients not recently scheduled
        eligible = [
            pid for pid, p in state.patients.items()
            if (t - p.last_scheduled_day) > 14  # not scheduled recently
        ]
        if eligible and n_new > 0:
            n_new = min(n_new, len(eligible))
            new_ids = rng.choice(eligible, size=n_new, replace=False)
            for pid in new_ids:
                priority = "urgent" if rng.random() < 0.2 else "routine"
                state.waitlist.append(WaitlistEntry(
                    patient_id=int(pid), request_day=t,
                    priority=priority,
                ))

        # 5. Generate new schedule (uses current drifted modifiers)
        state.day = t
        state.schedule = _build_daily_schedule(
            state.patients, day=t, clinic_config=cc,
            noshow_variability=self.noshow_variability,
            noshow_modifiers=state.noshow_modifiers, rng=rng,
        )

        # 6. Reset daily overbook budget
        state.overbook_budget = {
            pid: cc.max_overbook_per_provider
            for pid in range(cc.n_providers)
        }

        return state

    def predict(self, state: ClinicState, t: int) -> Predictions:
        """Predict no-show probability for today's schedule."""
        true_probs = np.array([
            s.true_noshow_prob for s in state.schedule
        ])

        if self.model_type == "baseline":
            # Staff uses patient's historical rate
            predicted = np.array([
                state.patients[s.patient_id].historical_noshow_rate
                for s in state.schedule
            ])
            # Small noise for variability
            noise = self.rng.prediction.normal(
                0, 0.02, len(predicted)
            )
            predicted = np.clip(predicted + noise, 0.01, 0.99)
        else:
            if not self._model_fitted:
                labels = (
                    self.rng.prediction.random(len(true_probs))
                    < true_probs
                ).astype(int)
                self._model.fit(
                    labels, true_probs,
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
                "true_probs": true_probs,
                "model_type": self.model_type,
            },
        )

    def intervene(
        self, state: ClinicState, predictions: Predictions, t: int,
    ) -> tuple[ClinicState, Interventions]:
        """Overbook slots using configured policy.

        Policies:
            'threshold': Scan slots with predicted no-show prob above
                threshold, fill from waitlist (current practice model).
            'urgent_first': Start from the waitlist demand side. Place
                all urgent patients first into the highest no-show
                slots, then routine patients if budget remains.
        """
        self.rng.intervention.random()  # RNG discipline

        if self.overbooking_policy == "urgent_first":
            return self._intervene_urgent_first(state, predictions, t)
        return self._intervene_threshold(state, predictions, t)

    def _intervene_threshold(
        self, state: ClinicState, predictions: Predictions, t: int,
    ) -> tuple[ClinicState, Interventions]:
        """Threshold policy: overbook slots above predicted no-show threshold."""
        overbooked_indices: List[int] = []

        scored = sorted(
            enumerate(state.schedule),
            key=lambda x: x[1].predicted_noshow_prob,
            reverse=True,
        )

        for idx, slot in scored:
            if slot.predicted_noshow_prob < self.overbooking_threshold:
                break
            if state.overbook_budget.get(slot.provider_id, 0) <= 0:
                continue
            if not state.waitlist:
                break

            state.waitlist.sort(key=lambda w: (
                0 if w.priority == "urgent" else 1,
                w.request_day,
            ))
            candidate_entry = None
            for i, entry in enumerate(state.waitlist):
                p = state.patients[entry.patient_id]
                if p.n_times_overbooked < self.max_individual_overbooks:
                    candidate_entry = state.waitlist.pop(i)
                    break
            if candidate_entry is None:
                continue

            slot.is_overbooked = True
            slot.overbooked_patient_id = candidate_entry.patient_id
            state.overbook_budget[slot.provider_id] -= 1
            state.total_waitlist_served += 1
            state.cumulative_wait_days += (t - candidate_entry.request_day)
            overbooked_indices.append(idx)

        return state, self._make_intervention_result(
            overbooked_indices, state, t,
        )

    def _intervene_urgent_first(
        self, state: ClinicState, predictions: Predictions, t: int,
    ) -> tuple[ClinicState, Interventions]:
        """Urgent-first policy: place waitlist patients into best slots.

        1. Sort waitlist: urgent first, then by wait time (longest first)
        2. For each waitlist patient, find the available slot with the
           highest predicted no-show probability (best chance the
           original patient won't show)
        3. Urgent patients always get placed; routine patients only
           if provider budget remains
        """
        overbooked_indices: List[int] = []

        # Sort waitlist: urgent first, then longest-waiting
        state.waitlist.sort(key=lambda w: (
            0 if w.priority == "urgent" else 1,
            w.request_day,
        ))

        # Build available slots ranked by no-show prob (highest first)
        # Track which slots are already taken
        available = sorted(
            enumerate(state.schedule),
            key=lambda x: x[1].predicted_noshow_prob,
            reverse=True,
        )
        used_slots = set()

        patients_to_place = []
        for entry in state.waitlist:
            p = state.patients[entry.patient_id]
            if p.n_times_overbooked >= self.max_individual_overbooks:
                continue
            patients_to_place.append(entry)

        placed_entries = []
        for entry in patients_to_place:
            # Find best available slot
            best_idx = None
            best_slot = None
            for idx, slot in available:
                if idx in used_slots:
                    continue
                if state.overbook_budget.get(slot.provider_id, 0) <= 0:
                    continue
                best_idx = idx
                best_slot = slot
                break  # already sorted, first available is best

            if best_slot is None:
                break  # no slots available

            best_slot.is_overbooked = True
            best_slot.overbooked_patient_id = entry.patient_id
            state.overbook_budget[best_slot.provider_id] -= 1
            state.total_waitlist_served += 1
            state.cumulative_wait_days += (t - entry.request_day)
            overbooked_indices.append(best_idx)
            used_slots.add(best_idx)
            placed_entries.append(entry)

        # Remove placed patients from waitlist
        placed_ids = {e.patient_id for e in placed_entries}
        state.waitlist = [
            w for w in state.waitlist
            if w.patient_id not in placed_ids
        ]

        return state, self._make_intervention_result(
            overbooked_indices, state, t,
        )

    def _make_intervention_result(
        self, overbooked_indices, state, t,
    ) -> Interventions:
        return Interventions(
            treated_indices=np.array(overbooked_indices, dtype=int),
            metadata={
                "n_overbooked": len(overbooked_indices),
                "policy": self.overbooking_policy,
                "waitlist_size": len(state.waitlist),
                "avg_wait_days": (
                    state.cumulative_wait_days
                    / max(state.total_waitlist_served, 1)
                ),
            },
        )

    def measure(self, state: ClinicState, t: int) -> Outcomes:
        """Record outcomes from resolved slots."""
        slots = (
            state.resolved_slots if state.resolved_slots
            else state.schedule
        )

        slot_ids = np.array([s.slot_id for s in slots])
        noshows = np.array([
            0.0 if s.original_showed else 1.0 for s in slots
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
        race = np.array([
            state.patients[s.patient_id].race_ethnicity
            for s in slots
        ])
        insurance = np.array([
            state.patients[s.patient_id].insurance_type
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
            },
            metadata={
                "total_collisions": state.total_collisions,
                "total_overbooked": state.total_overbooked,
                "total_overbooked_showed": state.total_overbooked_showed,
                "total_noshows": state.total_noshows,
                "total_resolved": state.total_slots_resolved,
                "waitlist_size": len(state.waitlist),
                "total_waitlist_served": state.total_waitlist_served,
                "avg_wait_days": (
                    state.cumulative_wait_days
                    / max(state.total_waitlist_served, 1)
                ),
                "mean_overbooking_burden": float(np.mean([
                    p.n_times_overbooked
                    for p in state.patients.values()
                ])),
            },
        )


def _build_daily_schedule(
    patients: Dict[int, Patient],
    day: int,
    clinic_config: ClinicConfig,
    noshow_variability: float,
    noshow_modifiers: Dict[int, float],
    rng: np.random.Generator,
) -> List[Slot]:
    """Build a day's schedule using visit-frequency weighting.

    Patients are selected with probability proportional to their
    daily_schedule_weight. The true no-show probability reflects the
    current AR(1) drift: base_prob * modifier + appointment noise.
    """
    cc = clinic_config
    total_slots = cc.n_providers * cc.slots_per_provider_per_day

    pids = list(patients.keys())
    weights = np.array([
        patients[pid].daily_schedule_weight for pid in pids
    ])

    # Don't schedule someone who was just scheduled
    for i, pid in enumerate(pids):
        if (day - patients[pid].last_scheduled_day) < 3:
            weights[i] = 0.0

    w_sum = weights.sum()
    if w_sum < 1e-10:
        weights = np.ones(len(pids)) / len(pids)
    else:
        weights = weights / w_sum

    n_to_schedule = min(total_slots, len(pids))
    selected = rng.choice(
        pids, size=n_to_schedule, replace=False, p=weights,
    )

    schedule: List[Slot] = []
    for i, pid in enumerate(selected):
        patient = patients[pid]
        modifier = noshow_modifiers.get(pid, 1.0)
        # Current drifted probability + per-appointment noise
        current_prob = patient.base_noshow_prob * modifier
        appt_prob = current_prob + rng.normal(0, noshow_variability)
        appt_prob = float(np.clip(appt_prob, 0.01, 0.80))

        schedule.append(Slot(
            slot_id=day * 10000 + i,
            day=day,
            provider_id=i % cc.n_providers,
            patient_id=pid,
            true_noshow_prob=appt_prob,
        ))
    return schedule
