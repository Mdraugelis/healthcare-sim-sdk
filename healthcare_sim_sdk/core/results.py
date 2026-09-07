"""Simulation records and explicit, alignment-checked analysis exports.

``events`` selects the primary outcome; other keys select per-entity secondary
outcomes. Missing measurements are unknown, never silently zero. Treatment
indices are positions in the prediction population, not entity identifiers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .scenario import TimeConfig, Predictions, Interventions, Outcomes


def _vector(value: Any, label: str, length: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or (length is not None and len(arr) != length):
        raise ValueError(f"{label} must be a one-dimensional array"
                         + (f" of length {length}" if length is not None else ""))
    return arr


def _outcome(out: Outcomes, key: str) -> np.ndarray:
    events = _vector(out.events, "events")
    if key == "events":
        return events
    if key not in out.secondary:
        raise ValueError(f"Outcome {key!r} not found in Outcomes.secondary")
    return _vector(out.secondary[key], f"secondary outcome {key!r}", len(events))


def _ids(out: Outcomes, *, required: bool = True) -> np.ndarray:
    n = len(_vector(out.events, "events"))
    if out.entity_ids is None:
        if required:
            raise ValueError("entity_ids not set on Outcomes. Set entity_ids in "
                             "measure() for panel export.")
        return np.arange(n)
    return _unique_ids(out.entity_ids, n, "entity_ids")


def _unique_ids(value: Any, n: int, label: str) -> np.ndarray:
    ids = _vector(value, label, n)
    if any(e is None or (isinstance(e, (float, np.floating)) and not np.isfinite(e))
           for e in ids):
        raise ValueError(f"{label} must not contain missing or nonfinite IDs")
    if len(set(ids.tolist())) != n:
        raise ValueError(f"{label} must be unique within a timestep")
    return ids


@dataclass
class SimulationResults:
    """Standardized records; the scenario owns entity identity and outcomes."""
    n_entities: int = 0
    time_config: Optional[TimeConfig] = None
    unit_of_analysis: str = "entity"
    counterfactual_mode: str = "branched"
    predictions: Dict[int, Predictions] = field(default_factory=dict)
    interventions: Dict[int, Interventions] = field(default_factory=dict)
    outcomes: Dict[int, Outcomes] = field(default_factory=dict)
    counterfactual_outcomes: Dict[int, Outcomes] = field(default_factory=dict)
    validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_predictions(self, t: int, predictions: Predictions) -> None:
        self.predictions[t] = predictions

    def record_interventions(self, t: int, interventions: Interventions) -> None:
        self.interventions[t] = interventions

    def record_outcomes(self, t: int, outcomes: Outcomes,
                        counterfactual: bool = False) -> None:
        target = self.counterfactual_outcomes if counterfactual else self.outcomes
        target[t] = outcomes

    def record_validation(self, phase: str, results: Dict[str, Any]) -> None:
        self.validations[phase] = results

    def _n_timesteps(self) -> int:
        times = set(self.outcomes) | set(self.counterfactual_outcomes)
        times |= set(self.predictions) | set(self.interventions)
        if any(not isinstance(t, (int, np.integer)) or t < 0 for t in times):
            raise ValueError("Recorded timesteps must be nonnegative integers")
        n = (self.time_config.n_timesteps if self.time_config is not None
             else max(times, default=-1) + 1)
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("time_config.n_timesteps must be a nonnegative integer")
        if any(t >= n for t in times):
            raise ValueError("Recorded timestep outside time_config horizon")
        return n

    def get_outcome_series(self, outcome_key: str = "events") -> np.ndarray:
        """Totals with NaN for unmeasured timesteps (empty measured arrays sum to 0)."""
        return self.to_analysis().to_time_series(outcome_key)["outcomes"]

    def get_counterfactual_outcome_series(
        self, outcome_key: str = "events",
    ) -> np.ndarray:
        return self.to_analysis().to_time_series(
            outcome_key, branch="counterfactual")["outcomes"]

    def get_treatment_indicator(
        self, branch: str = "factual", treatment_semantics: str = "scheduled",
    ) -> np.ndarray:
        """Intervention-call indicator; ``any_treated`` requires a nonempty action.

        This is a timestep-level process indicator, not persistent exposure.
        Counterfactuals have no AI assignments, including snapshot mode (which
        can retain effects of earlier factual treatment in its inherited state).
        """
        self.to_analysis()._get_outcomes_dict(branch)
        if treatment_semantics not in {"scheduled", "any_treated"}:
            raise ValueError("treatment_semantics must be 'scheduled' or 'any_treated'")
        indicator = np.zeros(self._n_timesteps(), dtype=int)
        if branch == "factual":
            for t, intervention in self.interventions.items():
                indicator[t] = int(treatment_semantics == "scheduled"
                                   or len(intervention.treated_indices) > 0)
        return indicator

    def to_analysis(self) -> "AnalysisDataset":
        return AnalysisDataset(results=self)


class AnalysisDataset:
    """Analysis exports with explicit assignment and observation semantics.

    Persistent ``ever_treated`` means assigned at or before this timestep;
    ``current_step`` means assigned at exactly this timestep. Neither infers
    biological effect duration, eligibility, adoption, or baseline clinical care.
    Entity IDs must be stable across time for persistent exposure to be useful.
    """

    def __init__(self, results: SimulationResults):
        self.results = results

    @property
    def unit_of_analysis(self) -> str:
        return self.results.unit_of_analysis

    def _get_outcomes_dict(self, branch: str = "factual") -> Dict[int, Outcomes]:
        if branch not in {"factual", "counterfactual"}:
            raise ValueError("branch must be 'factual' or 'counterfactual'")
        return (self.results.counterfactual_outcomes if branch == "counterfactual"
                else self.results.outcomes)

    def _prediction_ids(self, t: int) -> Optional[np.ndarray]:
        pred = self.results.predictions.get(t)
        if pred is None:
            return None
        n = len(_vector(pred.scores, "prediction scores"))
        # slot_ids is the existing appointment-scenario convention. Slots are
        # globally unique appointments, not patients, and outcomes can lag them.
        for key in ("entity_ids", "slot_ids"):
            if key in pred.metadata:
                return _unique_ids(pred.metadata[key], n, f"prediction {key}")
        out = self.results.outcomes.get(t)
        if out is None:
            raise ValueError(f"Cannot align prediction positions at timestep {t}; "
                             "set Predictions.metadata['entity_ids']")
        return _unique_ids(_ids(out), n, "prediction/outcome entity_ids")

    def _assignments(self) -> Dict[int, set]:
        assignments: Dict[int, set] = {}
        for t, intv in self.results.interventions.items():
            indices = _vector(intv.treated_indices, "treated_indices")
            if len(indices) == 0:
                assignments[t] = set()
                continue
            if indices.dtype.kind not in "iu" or np.any(indices < 0):
                raise ValueError(
                    "treated_indices must contain nonnegative integer positions")
            ids = self._prediction_ids(t)
            if ids is None:
                out = self.results.outcomes.get(t)
                if out is None:
                    raise ValueError(
                        f"Cannot align treatment positions at timestep {t}")
                ids = _ids(out)
            if np.any(indices >= len(ids)):
                raise ValueError(
                    f"treated_indices outside entity population at timestep {t}")
            assignments[t] = set(ids[indices].tolist())
        return assignments

    def to_time_series(
        self, outcome_key: str = "events", branch: str = "factual", *,
        treatment_semantics: str = "scheduled",
    ) -> Dict[str, np.ndarray]:
        """Totals for ITS; ``observed`` distinguishes missing from measured zero.

        treatment_indicator uses intervention calls by default; use
        treatment_semantics='any_treated' for at least one assigned entity.
        A recorded empty outcome normally sums to zero. Set boolean metadata
        ``outcome_observed=False`` for an empty, unobserved measurement (e.g.
        appointment slots that have not yet resolved); its total remains NaN.
        """
        outcomes_dict = self._get_outcomes_dict(branch)
        n = self.results._n_timesteps()
        values = np.full(n, np.nan)
        observed = np.zeros(n, dtype=bool)
        for t, out in outcomes_dict.items():
            outcome_observed = out.metadata.get("outcome_observed", True)
            if not isinstance(outcome_observed, (bool, np.bool_)):
                raise ValueError(
                    "Outcomes.metadata['outcome_observed'] must be boolean")
            if not outcome_observed:
                if len(_outcome(out, outcome_key)):
                    raise ValueError(
                        "An unobserved outcome must have no entity values")
                continue
            try:
                values[t] = np.sum(_outcome(out, outcome_key).astype(float))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cannot aggregate {outcome_key!r} at timestep {t}: {exc}"
                ) from exc
            observed[t] = True
        return {"timesteps": np.arange(n), "outcomes": values, "observed": observed,
                "treatment_indicator": self.results.get_treatment_indicator(
                    branch, treatment_semantics)}

    def to_panel(
        self, outcome_key: str = "events", branch: str = "factual", *,
        treatment_semantics: str = "ever_treated",
    ) -> Dict[str, Any]:
        """Entity panel; treatment starts at each entity's own assignment time.

        The counterfactual treated column always records zero AI assignments.
        Missing periods contribute no rows; all measured entity IDs and selected
        secondary arrays are checked for consistent length and uniqueness.
        """
        outcomes_dict = self._get_outcomes_dict(branch)
        self.results._n_timesteps()
        if treatment_semantics not in {"ever_treated", "current_step"}:
            raise ValueError(
                "treatment_semantics must be 'ever_treated' or 'current_step'")
        assignments = self._assignments() if branch == "factual" else {}
        first: Dict[Any, int] = {}
        for t, entities in assignments.items():
            for entity in entities:
                first[entity] = min(t, first.get(entity, t))
        parts: Dict[str, list[np.ndarray]] = {
            key: [] for key in ("entity_ids", "timesteps", "outcomes", "treated")}
        for t, out in sorted(outcomes_dict.items()):
            ids = _ids(out)
            values = _outcome(out, outcome_key)
            if treatment_semantics == "ever_treated":
                treated = np.array([first.get(e, np.inf) <= t for e in ids], dtype=int)
            else:
                treated = np.array(
                    [e in assignments.get(t, set()) for e in ids], dtype=int)
            columns = (ids, np.full(len(ids), t, dtype=int), values, treated)
            for key, arr in zip(parts, columns):
                parts[key].append(arr)
        panel: Dict[str, Any] = {
            key: (np.concatenate(arrays) if arrays else
                  np.array([], dtype=float if key == "outcomes" else int))
            for key, arrays in parts.items()}
        panel["unit_of_analysis"] = self.unit_of_analysis
        panel["treatment_semantics"] = treatment_semantics
        return panel

    def to_entity_snapshots(
        self, t: int, outcome_key: str = "events", branch: str = "factual",
    ) -> Dict[str, Any]:
        """Cross-section with scores matched by ID from this timestep only.

        Unmatched entities have NaN scores, e.g. yesterday's resolved slots
        measured alongside today's predictions. Counterfactual scores are None,
        because the engine never calls predict on that branch.
        """
        outcomes_dict = self._get_outcomes_dict(branch)
        self.results._n_timesteps()
        if t not in outcomes_dict:
            raise ValueError(f"No outcomes at timestep {t}")
        out = outcomes_dict[t]
        ids = _ids(out, required=False)
        scores = None
        if branch == "factual" and t in self.results.predictions:
            pred = self.results.predictions[t]
            pred_ids = self._prediction_ids(t)
            assert pred_ids is not None  # Prediction is present at this timestep.
            score_by_id = dict(zip(
                pred_ids.tolist(), _vector(pred.scores, "scores")))
            scores = np.array([score_by_id.get(e, np.nan) for e in ids])
        return {"entity_ids": ids, "outcomes": _outcome(out, outcome_key),
                "scores": scores}

    def to_subgroup_panel(
        self, outcome_key: str = "events", subgroup_key: str = "subgroup",
        branch: str = "factual", *, treatment_semantics: str = "ever_treated",
    ) -> Dict[str, Any]:
        """Panel plus per-entity groups from secondary outcomes or metadata.

        A scalar metadata group applies to every entity. Missing groups retain
        the explicit 'unknown' label; malformed arrays fail instead of shifting
        group membership between entities.
        """
        panel = self.to_panel(
            outcome_key, branch, treatment_semantics=treatment_semantics)
        groups = []
        for t, out in sorted(self._get_outcomes_dict(branch).items()):
            n = len(out.events)
            if subgroup_key in out.secondary:
                arr = _vector(
                    out.secondary[subgroup_key], f"subgroup {subgroup_key!r}", n)
            else:
                value = out.metadata.get(subgroup_key, "unknown")
                arr = (np.full(n, value) if np.asarray(value).ndim == 0 else
                       _vector(value, f"subgroup {subgroup_key!r}", n))
            groups.append(arr)
        panel["subgroup"] = (
            np.concatenate(groups) if groups else np.array([], dtype=str))
        return panel
