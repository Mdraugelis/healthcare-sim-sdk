"""Build role-specific human review packets from an actual monitoring run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from healthcare_sim_sdk.core.rng import RNGPartitioner
from healthcare_sim_sdk.experiments.review import (
    ReviewPacket, aggregate_status, evidence,
)
from healthcare_sim_sdk.scenarios.nurse_retention.time_varying import resolve


def write_monitoring_review_packet(
    results, config, run, resolved_config, metrics, output_dir,
):
    """Write current-run evidence; never certify monitor power from one seed."""
    output_dir = Path(output_dir)
    registry = (Path(__file__).parents[1] / "configs" / "monitoring"
                / "review_claims.yaml")
    packet = ReviewPacket(
        output_dir, registry, resolved_config, [(run.regime, run.seed)],
    )
    checks = []
    expected = set(range(config.n_weeks))
    checks.append(evidence("timesteps_complete", "PASS" if (
        set(results.outcomes) == expected
        and set(results.counterfactual_outcomes) == expected
    ) else "FAIL", detail="Both branches must contain every configured week."))
    accounting_errors, capacity_errors, bounds_errors = [], [], []
    for branch_name, branch in (
        ("factual", results.outcomes),
        ("counterfactual", results.counterfactual_outcomes),
    ):
        previous_total = 0
        for t, outcome in sorted(branch.items()):
            meta = outcome.metadata
            events = np.asarray(outcome.events)
            total = meta["total_departures"]
            if (meta["n_active"] + total != config.n_nurses
                    or events.shape != (config.n_nurses,)
                    or not np.array_equal(outcome.entity_ids,
                                          np.arange(config.n_nurses))
                    or events.sum() != total - previous_total):
                accounting_errors.append([branch_name, t])
            if not np.isfinite(events).all() or not np.isin(events, [0, 1]).all():
                bounds_errors.append([branch_name, t])
            k = int(resolve(config.max_interventions_per_manager_per_week, t))
            counts = np.asarray(meta.get("soc_per_manager_this_week", []))
            if (len(counts) != meta["n_managers"] or (counts > k).any()
                    or (counts < 0).any()):
                capacity_errors.append([branch_name, t, "soc"])
            previous_total = total
    for t, actions in results.interventions.items():
        n_managers = results.outcomes[t].metadata["n_managers"]
        # Round-robin manager assignment is fixed at population creation.
        nurse_ids = np.asarray(actions.treated_indices, dtype=int)
        counts = np.bincount(nurse_ids % n_managers, minlength=n_managers)
        k = int(resolve(config.max_interventions_per_manager_per_week, t))
        if (counts > k).any() or len(np.unique(nurse_ids)) != len(nurse_ids):
            capacity_errors.append(["factual", t, "ai"])
    for name, errors in (("population_event_accounting", accounting_errors),
                         ("separate_budget_capacity", capacity_errors),
                         ("finite_binary_outcomes", bounds_errors)):
        checks.append(evidence(name, "FAIL" if errors else "PASS", actual=errors,
                               detail="Every recorded week and branch checked."))
    prediction_bad = [t for t, pred in results.predictions.items() if (
        np.asarray(pred.scores).shape != (config.n_nurses,)
        or not np.isfinite(pred.scores).all()
        or ((pred.scores < 0) | (pred.scores > 1)).any()
    )]
    missing_predictions = (set(results.time_config.prediction_schedule)
                           - set(results.predictions))
    unexpected_predictions = (set(results.predictions)
                              - set(results.time_config.prediction_schedule))
    prediction_bad += sorted(missing_predictions | unexpected_predictions)
    checks.append(evidence("prediction_bounds", "FAIL" if prediction_bad else "PASS",
                           actual=prediction_bad))
    checks.append(evidence("factual_adoption_metadata", "INCONCLUSIVE" if (
        run.metadata["missing_adoption_weeks"]
    ) else "PASS", actual=run.metadata["missing_adoption_weeks"],
        detail="Missing adoption is never filled as full adoption."))
    model = run.metadata["independent_frozen_scoring_validation"]
    checks.append(evidence("frozen_synthetic_score_fidelity", model["status"],
                           detail=model.get("reason", ""), actual=model,
                           category="model"))
    future = [record.get("actual_future_outcomes")
              for record in run.metadata["prediction_performance"]]
    statuses = [evidence(str(i), record["status"] if record else "NOT_RUN")
                for i, record in enumerate(future)]
    checks.append(evidence(
        "realized_future_score_fidelity", aggregate_status(statuses), actual=future,
        detail="Actual post-intervention factual departure windows; terminal horizons "
               "remain NOT_RUN. Each window must have enough positives and negatives.",
        category="model",
    ))
    evaluations = run.metadata["monitoring_claim_evaluations"]
    statuses = [evidence(row["claim_id"], row["evaluation_status"])
                for row in evaluations]
    checks.append(evidence(
        "monitoring_event_evaluation", aggregate_status(statuses), actual=evaluations,
        detail="Status describes completeness of event evaluation. A nondetection is "
               "an observed finding, not a software correctness failure.",
        category="analysis",
    ))
    checks.append(evidence(
        "monitoring_mode_disclosed", "PASS", actual={
            key: run.metadata[key] for key in (
                "monitoring_mode", "uses_counterfactual_for_estimation",
                "tier3_estimand",
            )
        }, detail="Oracle comparison and factual-only trend have distinct estimands.",
        category="analysis",
    ))
    cases, traces = _observed_cases(results, config, run.seed)
    (output_dir / "case_traces.json").write_text(json.dumps(
        traces, indent=2, allow_nan=False,
    ))
    (output_dir / "monitoring_evidence.json").write_text(json.dumps(
        run.to_dict(), indent=2, allow_nan=False,
    ))
    checks.append(evidence(
        "observed_case_traces", "PASS", actual={
            "n_categories": len(cases), "n_trace_rows": len(traces),
        }, detail="Cases use actual event/score/action records from both branches. "
                  "Pre-simulation state and decision ranks are not reconstructed.",
        artifacts=["case_traces.json"],
    ))
    scalar_metrics = {key: value for key, value in metrics.items()
                      if isinstance(value, (int, float)) and np.isfinite(value)}
    packet.add_run(
        run.regime, run.seed, metrics=scalar_metrics, checks=checks, cases=cases,
        artifacts=["case_traces.json", "monitoring_evidence.json"],
    )
    packet.add_evidence(evidence(
        "monitoring_operating_characteristics", "NOT_RUN", category="analysis",
        detail="This packet contains one seed. Repeated independently held-out seeds "
               "are needed for false-alert probability, detection probability, "
               "conditional delay, censoring and Monte Carlo uncertainty.",
    ))
    packet.add_evidence(evidence(
        "clinical_workflow_review", "NOT_RUN", category="human_review",
        detail="Review separate SOC/AI resource budgets, intervention assumptions, "
               "prediction label generation and local workflow plausibility.",
    ))
    return packet.write()


def _observed_cases(results, config, seed):
    """One seeded random case plus explicit favorable/adverse/neutral cases."""
    last = max(results.outcomes)
    f_active = np.asarray(results.outcomes[last].secondary["active_mask"], dtype=bool)
    c_active = np.asarray(
        results.counterfactual_outcomes[last].secondary["active_mask"], dtype=bool,
    )
    random = RNGPartitioner(seed + 20_000_000).create_streams().population
    groups = [
        ("random sample", np.arange(config.n_nurses), True),
        ("retained only on factual branch",
         np.flatnonzero(f_active & ~c_active), False),
        ("departed only on factual branch",
         np.flatnonzero(~f_active & c_active), False),
        ("same final retention status", np.flatnonzero(f_active == c_active), False),
    ]
    cases, traces = [], []
    for category, candidates, random_sample in groups:
        if not len(candidates):
            cases.append({"category": category, "status": "NOT_APPLICABLE",
                          "candidate_count": 0,
                          "selection": "declared outcome category",
                          "narrative": "No observed nurse met this category."})
            continue
        nurse = int(random.choice(candidates) if random_sample else candidates[0])
        first_row = len(traces)
        for t in sorted(results.outcomes):
            pred = results.predictions.get(t)
            action = results.interventions.get(t)
            factual = results.outcomes[t]
            control = results.counterfactual_outcomes[t]
            traces.append({
                "row": len(traces), "nurse_id": nurse, "week": t,
                "score": float(pred.scores[nurse]) if pred is not None else None,
                "ai_check_in": bool(
                    action is not None and nurse in action.treated_indices
                ),
                "factual_departure": int(factual.events[nurse]),
                "counterfactual_departure": int(control.events[nurse]),
                "factual_active": bool(factual.secondary["active_mask"][nurse]),
                "counterfactual_active": bool(control.secondary["active_mask"][nurse]),
                "source": f"results.outcomes/predictions/interventions at week {t}",
            })
        factual_status = "retained" if f_active[nurse] else "departed"
        control_status = "retained" if c_active[nurse] else "departed"
        cases.append({
            "category": category, "status": "PASS", "nurse_id": nurse,
            "candidate_count": len(candidates),
            "selection": ("one random nurse from the full initial cohort; independent "
                          "partitioned seed + 20,000,000" if random_sample
                          else "lowest nurse ID in declared final-outcome category"),
            "narrative": f"Nurse {nurse} is {factual_status} "
                         f"on the factual branch and "
                         f"{control_status} on the "
                         f"counterfactual branch at week {last}. This is a selected "
                         "paired simulation path, not an estimate of average benefit.",
            "artifact": "case_traces.json", "trace_row_start": first_row,
            "trace_row_end": len(traces) - 1,
        })
    return cases, traces
