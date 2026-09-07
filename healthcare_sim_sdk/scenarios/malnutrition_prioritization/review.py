"""Pilot review packet: checks and case traces from each actual simulation."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ...core.rng import RNGPartitioner
from ...experiments.review import (
    ReviewPacket, canonical_json, evidence, file_digest, run_identity,
)
from .runner_common import reduce_results, run_once, verify_results
from .scenario import SOURCE_ML, SOURCE_CONSULT


REGISTRY = Path(__file__).parent / "configs" / "review_claims.yaml"
CERTIFICATES = Path("outputs/malnutrition_verification")


def model_path(cfg):
    return Path(
        cfg.model_params_path or Path(__file__).parent / "configs/model_frozen.yaml"
    )


def certificate_identity(stage, cfg, targets_path=None):
    from ...experiments.review import file_digest
    if stage == "A":
        # The certification cohort uses default demographic proportions.
        config = {"stage": "A", "severity_score_gradient": cfg.severity_score_gradient,
                  "severity_moderate_fraction": cfg.severity_moderate_fraction,
                  "sex_proportions": cfg.sex_proportions,
                  "race_proportions": cfg.race_proportions,
                  "ethnicity_proportions": cfg.ethnicity_proportions,
                  "hl_prevalence_multiplier": cfg.hl_prevalence_multiplier,
                  "degenerate_auroc05": cfg.degenerate_auroc05,
                  "threshold": cfg.threshold,
                  "targets_sha256": file_digest(targets_path) if targets_path else None}
    else:
        config = {"stage": "B", "scenario": asdict(cfg), "horizon": 84,
                  "seed_set": [1, 2, 3, 4, 5, 42]}
        config["scenario"].pop("model_params_path", None)
    return run_identity(config, model_path(cfg) if cfg.ml_enabled else None)


def create_packet(output_dir, config, cells, seeds, registry=REGISTRY):
    from .runner_common import config_from_dict
    resolved_cells = []
    for i, cell in enumerate(cells):
        cfg = config_from_dict(cell["config"])
        path = model_path(cfg) if cfg.ml_enabled else None
        resolved_cells.append({
            "cell": f"cell_{i:02d}", "label": cell.get("label", f"cell_{i:02d}"),
            "scenario": asdict(cfg),
            "model_sha256": file_digest(path) if path and path.exists() else None,
            "model_status": "PASS" if path and path.exists() else
                            "NOT_RUN" if path else "NOT_APPLICABLE",
        })
    packet = ReviewPacket(
        output_dir, registry, {"runner": config, "resolved_cells": resolved_cells},
        [(f"cell_{i:02d}", seed) for i in range(len(cells)) for seed in seeds],
    )
    # Exact applicability is deliberate: a changed cell is not silently
    # certified by a default-case boundary run.
    from .subgroup_performance import TARGETS_PATH
    for stage in ("A", "B"):
        name = f"stage_{stage.lower()}"
        if not any(c["config"].get("ml_enabled", True) for c in cells):
            packet.add_evidence(evidence(
                name, "NOT_APPLICABLE",
                rationale=(
                    "ML disabled in every cell; no ML deployment benefit claimed."
                ),
            ))
            continue
        identities = []
        for cell in cells:
            cfg = config_from_dict(cell["config"])
            try:
                identities.append(certificate_identity(
                    stage, cfg, TARGETS_PATH if TARGETS_PATH.exists() else None,
                ))
            except FileNotFoundError:
                identities.append(None)
        if not identities[0] or any(x != identities[0] for x in identities):
            packet.add_evidence(evidence(
                name, "INCONCLUSIVE",
                detail=("Cells have differing certification scopes; "
                        "certify them individually."),
            ))
        else:
            packet.add_certificate(
                name, CERTIFICATES / f"stage_{stage.lower()}_certificate.json",
                identities[0],
            )
    packet.add_evidence(evidence(
        "study_design_validation", "NOT_RUN", category="analysis",
        detail="No real-world study design is specified by this simulation runner.",
    ))
    return packet


def case_traces(res, seed):
    """Select random and mechanism cases, then trace recorded outputs only.

    Selection randomness uses a separate partitioner and cannot alter the run.
    Rank is among scoreable entities at that scoring call, not asserted to be
    the final consult/MST-adjusted assessment queue rank.
    """
    last = max(res.outcomes)
    f = res.outcomes[last].secondary
    c = res.counterfactual_outcomes[last].secondary
    n = len(f["status"])
    reviewer_seed = seed + 20_000_000
    sample = RNGPartitioner(reviewer_seed).create_streams().population.choice(
        n, min(3, n), replace=False,
    )
    groups = [
        (f"random sample {j+1}", np.array([i]),
         f"uniform random sample; independent reviewer seed {reviewer_seed}")
        for j, i in enumerate(sample)
    ]
    groups += [
        ("ML catch / baseline miss", np.flatnonzero(
            (f["assess_source"] == SOURCE_ML) & (c["missed"] == 1)
        ), "first eligible entity by ID"),
        ("reordering harm", np.flatnonzero(
            (f["missed"] == 1) & (c["assessed"] == 1)
        ), "first eligible entity by ID"),
        ("false-positive workup", np.flatnonzero(
            (f["assess_source"] == SOURCE_ML) & (f["mal_at_discharge"] == 0)
            & (f["status"] == 2)
        ), "first eligible entity by ID"),
        ("retained screening catch", np.flatnonzero(
            (f["assessed"] == 1) & (f["assess_source"] > 0)
            & ~np.isin(f["assess_source"], [SOURCE_ML, SOURCE_CONSULT])
        ), "first eligible entity by ID"),
        ("equity example", np.flatnonzero(
            (f["ethnicity"] == "HispanicLatino") & (f["true_malnourished"] == 1)
        ), "first eligible entity by ID; illustrative subgroup case"),
    ]
    cases, rows = [], []
    selected = set()
    for category, candidates, rule in groups:
        if not len(candidates):
            cases.append({"category": category, "status": "NOT_APPLICABLE",
                          "candidate_count": 0, "selection": rule,
                          "narrative": (
                              "No matching case occurred in this run; this is not "
                              "evidence that the mechanism is impossible."
                          )})
            continue
        i = int(candidates[0])
        selected.add(i)
        cases.append({"category": category, "status": "PASS", "entity_id": i,
                      "candidate_count": (
                          int(len(candidates)) if "random" not in category else n
                      ),
                      "selection": rule,
                      "narrative": (
                          f"Entity {int(res.outcomes[last].entity_ids[i])}: "
                          f"factual assessment day {int(f['assess_day'][i])}, "
                          f"counterfactual {int(c['assess_day'][i])}; "
                          f"missed {int(f['missed'][i])} vs {int(c['missed'][i])}; "
                          f"harm {f['harm'][i]:.2f} vs {c['harm'][i]:.2f}. "
                          "Day -1 denotes no assessment."
                      )})
    for t in sorted(res.outcomes):
        pred = res.predictions.get(t)
        ranks = {}
        if pred is not None:
            scoreable = np.isfinite(pred.scores) & pred.metadata.get(
                "scoreable", np.ones(n, bool),
            )
            order = np.flatnonzero(scoreable)[
                np.argsort(-pred.scores[scoreable], kind="stable")
            ]
            ranks = {int(i): j + 1 for j, i in enumerate(order)}
        for branch, outputs in (
            ("factual", res.outcomes), ("counterfactual", res.counterfactual_outcomes),
        ):
            out = outputs[t]
            s = out.secondary
            for i in sorted(selected):
                score = (float(pred.scores[i])
                         if branch == "factual" and pred is not None else None)
                rows.append({"trace_id": f"{branch}-{t}-{i}", "entity_id": i,
                             "timestep": t, "branch": branch,
                             "decision_detail_scope": (
                                 "End-step outcomes; exact pre-assessment eligibility "
                                 "and queue displacement reason are not recorded."
                             ),
                             "observation_phase": "end of actual engine timestep",
                             "status": int(s["status"][i]),
                             "severity": int(s["severity"][i]),
                             "admit_day": float(s["admit_day"][i]),
                             "score": (score if score is not None
                                       and np.isfinite(score) else None),
                             "rank_among_scoreable": (
                                 ranks.get(i) if branch == "factual" else None
                             ),
                             "assessment_source": int(s["assess_source"][i]),
                             "assessed_today": bool(s["assess_day"][i] == t),
                             "queue_depth": out.metadata["queue_depth"],
                             "total_assessments_today": out.metadata["assessed_today"],
                             "missed_event": float(out.events[i]),
                             "harm": float(s["harm"][i]), "coded": int(s["coded"][i])})
    return cases, rows


def _actual_run_identity(cfg, horizon, seed):
    return {
        "scenario": asdict(cfg), "horizon_days": int(horizon), "seed": int(seed),
        "model_sha256": file_digest(model_path(cfg)) if cfg.ml_enabled else None,
    }


def _check_run_identity(packet, cell_id, actual, seeds):
    declared = next((cell for cell in packet.config["resolved_cells"]
                     if cell["cell"] == cell_id), None)
    if declared is None:
        raise ValueError(f"Unplanned cell: {cell_id}")
    if canonical_json(actual["scenario"]) != canonical_json(declared["scenario"]):
        raise ValueError("Actual scenario configuration differs from reviewed plan")
    if actual["model_sha256"] != declared["model_sha256"]:
        raise ValueError("Frozen model bytes changed since the packet was created")
    planned_seeds = [seed for cell, seed in packet.expected if cell == cell_id]
    if sorted(seeds) != sorted(planned_seeds):
        raise ValueError("Actual seed schedule differs from reviewed plan")
    runner = packet.config.get("runner", {})
    # Hydra runner configs nest horizon under runner; the public demo is flat.
    planned_horizon = runner.get("runner", runner).get("horizon_days")
    if planned_horizon is not None and actual["horizon_days"] != planned_horizon:
        raise ValueError("Actual horizon differs from reviewed plan")


def run_reviewed_cell(packet, cell_id, cfg, horizon, seeds, result_writer=None):
    rows = []
    seeds = list(seeds)
    if not seeds:
        packet.add_evidence(evidence(
            f"execution_plan:{cell_id}", "FAIL", category="reproducibility",
            detail="Actual seed schedule is empty; reviewed runs were not executed.",
        ))
        packet.write()
        raise ValueError("Actual seed schedule differs from reviewed plan")
    for seed in seeds:
        prefix = f"{cell_id}_seed_{seed}"
        manifest_name = prefix + "_run_manifest.json"
        manifest_path = packet.output_dir / manifest_name
        try:
            before = _actual_run_identity(cfg, horizon, seed)
            run_manifest = {"cell": cell_id, "before_run": before}
            manifest_path.write_text(canonical_json(run_manifest))
            _check_run_identity(packet, cell_id, before, seeds)
            _, res = run_once(cfg, horizon, seed)
            after = _actual_run_identity(cfg, horizon, seed)
            run_manifest["after_run"] = after
            manifest_path.write_text(canonical_json(run_manifest))
            _check_run_identity(packet, cell_id, after, seeds)
            verification = verify_results(cfg, res)
            metrics = reduce_results(res)
            metrics["seed"] = seed
            cases, traces = case_traces(res, seed)
            name = prefix + "_traces.json"
            (packet.output_dir / name).write_text(json.dumps(traces, allow_nan=False))
            for case in cases:
                if case["status"] == "PASS":
                    case["artifact"] = name
            checks = verification["checks"] + [evidence(
                "case_traces", "PASS",
                detail=("Actual engine output histories; "
                        "missing categories explicitly identified."),
                artifacts=[name],
            ), evidence(
                "run_provenance", "PASS", category="reproducibility",
                detail=("Scenario, seed schedule, declared horizon and model bytes "
                        "match the reviewed plan before and after execution. "
                        "The manifest records actual horizon when none was planned."),
                artifacts=[manifest_name],
            )]
            artifacts = [name, manifest_name]
            if result_writer is not None:
                artifacts.extend(result_writer(seed, res))
            packet.add_run(
                cell_id, seed, metrics=metrics, checks=checks,
                artifacts=artifacts, cases=cases,
            )
            rows.append(metrics)
            if not verification["all_pass"]:
                packet.write()
                raise ValueError(
                    f"{prefix}: structural verification failed; evidence saved"
                )
        except Exception as error:
            key = (str(cell_id), int(seed))
            if key not in packet.runs and key in packet.expected:
                packet.add_run(
                    cell_id, seed, error=str(error),
                    artifacts=[manifest_name] if manifest_path.exists() else [],
                )
            elif key not in packet.expected:
                packet.add_evidence(evidence(
                    f"execution_plan:{cell_id}:{seed}", "FAIL", detail=str(error),
                    category="reproducibility",
                ))
            packet.write()
            raise
    return rows
