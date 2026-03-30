"""Threshold optimizer for Access Operations.

Sweeps 9 clinic archetypes x (1 baseline + 7 thresholds x 3 AUCs)
= 198 runs. Produces per-archetype threshold recommendations.

Usage:
    python -m healthcare_sim_sdk.scenarios.noshow_overbooking.\
run_threshold_optimizer --n-days 90
"""

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from healthcare_sim_sdk.core.engine import (
    BranchedSimulationEngine, CounterfactualMode,
)
from healthcare_sim_sdk.core.scenario import TimeConfig
from healthcare_sim_sdk.ml.performance import (
    auc_score, confusion_matrix_metrics,
)
from healthcare_sim_sdk.scenarios.noshow_overbooking.realistic_scenario import (
    ClinicConfig, RealisticNoShowScenario,
)

logger = logging.getLogger(__name__)


@dataclass
class ClinicArchetype:
    name: str
    noshow_rate: float
    utilization_label: str
    n_providers: int
    slots_per_provider: int
    max_overbook_per_provider: int
    new_waitlist_requests_per_day: int
    n_patients: int = 2000

    def to_config(self) -> ClinicConfig:
        return ClinicConfig(
            n_providers=self.n_providers,
            slots_per_provider_per_day=self.slots_per_provider,
            max_overbook_per_provider=(
                self.max_overbook_per_provider
            ),
            new_waitlist_requests_per_day=(
                self.new_waitlist_requests_per_day
            ),
        )


def build_archetypes() -> List[ClinicArchetype]:
    util_params = {
        "80%": {"max_ob": 3, "wl": 2},
        "90%": {"max_ob": 2, "wl": 5},
        "110%": {"max_ob": 1, "wl": 10},
    }
    out = []
    for ns in [0.07, 0.13, 0.20]:
        for ul, p in util_params.items():
            out.append(ClinicArchetype(
                name=f"NS{ns:.0%}_Util{ul}",
                noshow_rate=ns,
                utilization_label=ul,
                n_providers=6, slots_per_provider=12,
                max_overbook_per_provider=p["max_ob"],
                new_waitlist_requests_per_day=p["wl"],
            ))
    return out


@dataclass
class Config:
    experiment_name: str = "threshold_optimizer"
    timestamp: str = ""
    seed: int = 42
    n_days: int = 90
    baseline_threshold: float = 0.50
    model_aucs: List[float] = field(
        default_factory=lambda: [0.65, 0.75, 0.83],
    )
    thresholds: List[float] = field(
        default_factory=lambda: [
            0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
        ],
    )
    max_individual_overbooks: int = 10
    ar1_rho: float = 0.95
    ar1_sigma: float = 0.04

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )


def run_one(
    arch: ClinicArchetype, cfg: Config,
    model_type: str, model_auc: float, threshold: float,
) -> Dict[str, Any]:
    cc = arch.to_config()
    tc = TimeConfig(
        n_timesteps=cfg.n_days, timestep_duration=1 / 365,
        timestep_unit="day",
        prediction_schedule=list(range(cfg.n_days)),
    )
    sc = RealisticNoShowScenario(
        time_config=tc, seed=cfg.seed,
        n_patients=arch.n_patients,
        base_noshow_rate=arch.noshow_rate,
        model_type=model_type, model_auc=model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=cfg.max_individual_overbooks,
        overbooking_policy="threshold", clinic_config=cc,
        ar1_rho=cfg.ar1_rho, ar1_sigma=cfg.ar1_sigma,
    )
    results = BranchedSimulationEngine(
        sc, CounterfactualMode.NONE,
    ).run(arch.n_patients)

    meta = results.outcomes[cfg.n_days - 1].metadata
    n_ob = meta["total_overbooked"]
    n_res = meta["total_resolved"]

    # Utilization
    uv = []
    for t in range(1, cfg.n_days):
        u = results.outcomes[t].secondary.get("utilization")
        if u is not None and len(u) > 0:
            uv.append(float(u.mean()))

    # AUC + classification
    ap, aa, ar = [], [], []
    for t in range(cfg.n_days):
        if t in results.predictions and (t + 1) < cfg.n_days:
            p = results.predictions[t].scores
            a = results.outcomes[t + 1].events
            if len(p) == len(a):
                ap.append(p)
                aa.append(a)
                ar.append(
                    results.outcomes[t + 1].secondary.get(
                        "race_ethnicity", np.array([])
                    )
                )
    pa = np.concatenate(ap) if ap else np.array([])
    ya = np.concatenate(aa) if aa else np.array([])
    ra = np.concatenate(ar) if ar else np.array([])

    auc = float(auc_score(ya, pa)) if len(pa) > 100 else 0
    cm = (
        confusion_matrix_metrics(ya, pa, threshold)
        if len(pa) > 100 else {}
    )

    # Subgroup parity
    sg_ratio = 0.0
    ons = float(ya.mean()) if len(ya) > 0 else 0
    if len(ra) == len(ya) and len(ya) > 0:
        for g in np.unique(ra):
            m = ra == g
            if m.sum() < 30:
                continue
            gr = float(ya[m].mean())
            if ons > 0:
                sg_ratio = max(sg_ratio, gr / ons)

    # Re-run for burden
    sc2 = RealisticNoShowScenario(
        time_config=tc, seed=cfg.seed,
        n_patients=arch.n_patients,
        base_noshow_rate=arch.noshow_rate,
        model_type=model_type, model_auc=model_auc,
        overbooking_threshold=threshold,
        max_individual_overbooks=cfg.max_individual_overbooks,
        overbooking_policy="threshold", clinic_config=cc,
        ar1_rho=cfg.ar1_rho, ar1_sigma=cfg.ar1_sigma,
    )
    st = sc2.create_population(arch.n_patients)
    for t in range(cfg.n_days):
        st = sc2.step(st, t)
        pr = sc2.predict(st, t)
        st, _ = sc2.intervene(st, pr, t)
    max_b = max(p.n_times_overbooked for p in st.patients.values())

    return {
        "archetype": arch.name,
        "noshow_config": arch.noshow_rate,
        "util_level": arch.utilization_label,
        "ob_cap": arch.max_overbook_per_provider,
        "wl_pressure": arch.new_waitlist_requests_per_day,
        "model_type": model_type,
        "auc_target": model_auc,
        "threshold": threshold,
        "label": "",
        "auc_achieved": auc,
        "ppv": cm.get("ppv", 0),
        "sensitivity": cm.get("sensitivity", 0),
        "specificity": cm.get("specificity", 0),
        "flag_rate": cm.get("flag_rate", 0),
        "utilization": float(np.mean(uv)) if uv else 0,
        "collision_rate": (
            meta["total_collisions"] / max(n_ob, 1)
            if n_ob > 0 else 0
        ),
        "collisions": meta["total_collisions"],
        "overbooked": n_ob,
        "ob_per_week": n_ob / max(cfg.n_days / 7, 1),
        "waitlist": meta["waitlist_size"],
        "wl_served": meta["total_waitlist_served"],
        "avg_wait_days": meta["avg_wait_days"],
        "max_burden": max_b,
        "mean_burden": meta["mean_overbooking_burden"],
        "sg_ratio": sg_ratio,
        "noshow_observed": (
            meta["total_noshows"] / max(n_res, 1)
        ),
    }


def run_sweep(cfg: Config) -> List[Dict]:
    archs = build_archetypes()
    out = []
    total = len(archs) * (
        1 + len(cfg.thresholds) * len(cfg.model_aucs)
    )
    i = 0
    t0 = time.time()

    for a in archs:
        i += 1
        logger.info("[%d/%d] %s baseline", i, total, a.name)
        r = run_one(a, cfg, "baseline", 0, cfg.baseline_threshold)
        r["label"] = "baseline"
        out.append(r)

        for auc in cfg.model_aucs:
            for th in cfg.thresholds:
                i += 1
                logger.info(
                    "[%d/%d] %s auc=%.2f t=%.2f",
                    i, total, a.name, auc, th,
                )
                r = run_one(a, cfg, "predictor", auc, th)
                r["label"] = f"auc{auc:.2f}_t{th:.2f}"
                out.append(r)

    elapsed = time.time() - t0
    logger.info(
        "Done: %d runs in %.0fs (%.1fs/run)",
        len(out), elapsed, elapsed / max(len(out), 1),
    )
    return out


def generate_report(cfg: Config, results: List[Dict]) -> str:
    L = []  # noqa: E741
    L.append("# Threshold Optimizer: Operations Report")
    L.append("")
    L.append(
        f"*{cfg.timestamp} | {cfg.n_days} days | "
        f"{len(results)} configs*"
    )
    L.append("")

    L.append("## Clinic Archetypes")
    L.append("")
    L.append("| Name | NoShow | Util | OB Cap | WL/Day |")
    L.append("|------|--------|------|--------|--------|")
    for a in build_archetypes():
        L.append(
            f"| {a.name} | {a.noshow_rate:.0%} "
            f"| {a.utilization_label} "
            f"| {a.max_overbook_per_provider} "
            f"| {a.new_waitlist_requests_per_day} |"
        )
    L.append("")

    for arch in build_archetypes():
        L.append(f"## {arch.name}")
        L.append("")

        bl = next(
            (r for r in results
             if r["archetype"] == arch.name
             and r["model_type"] == "baseline"),
            None,
        )
        if bl:
            L.append(
                f"**Baseline (hist>=50%):** "
                f"util={bl['utilization']:.1%}, "
                f"coll={bl['collision_rate']:.1%}, "
                f"WL={bl['waitlist']}, "
                f"burden={bl['max_burden']}"
            )
            L.append("")

        L.append(
            "| AUC | Thresh | Util | Coll% "
            "| PPV | Sens | WL | Wait | Burden |"
        )
        L.append(
            "|-----|--------|------|-------"
            "|-----|------|----|------|--------|"
        )
        preds = sorted(
            [r for r in results
             if r["archetype"] == arch.name
             and r["model_type"] == "predictor"],
            key=lambda r: (r["auc_target"], r["threshold"]),
        )
        for r in preds:
            L.append(
                f"| {r['auc_target']:.2f} "
                f"| {r['threshold']:.2f} "
                f"| {r['utilization']:.1%} "
                f"| {r['collision_rate']:.1%} "
                f"| {r['ppv']:.1%} "
                f"| {r['sensitivity']:.1%} "
                f"| {r['waitlist']} "
                f"| {r['avg_wait_days']:.1f} "
                f"| {r['max_burden']} |"
            )
        L.append("")

    # Prevalence-PPV
    L.append("## Prevalence-PPV Relationship")
    L.append("")
    L.append(
        "| NoShow | AUC 0.65 @0.30 "
        "| AUC 0.75 @0.30 | AUC 0.83 @0.30 |"
    )
    L.append(
        "|--------|----------------"
        "|----------------|----------------|"
    )
    for ns in [0.07, 0.13, 0.20]:
        ppvs = []
        for at in [0.65, 0.75, 0.83]:
            m = [
                r for r in results
                if abs(r["noshow_config"] - ns) < 0.01
                and r["model_type"] == "predictor"
                and abs(r["auc_target"] - at) < 0.01
                and abs(r["threshold"] - 0.30) < 0.01
            ]
            ppvs.append(
                f"{m[0]['ppv']:.1%}" if m else "N/A"
            )
        L.append(
            f"| {ns:.0%} | {ppvs[0]} "
            f"| {ppvs[1]} | {ppvs[2]} |"
        )
    L.append("")
    L.append(
        "**The optimal threshold varies by clinic.** "
        "Use the tables above."
    )
    L.append("")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser(
        description="Threshold optimizer",
    )
    parser.add_argument("--n-days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = Config(seed=args.seed, n_days=args.n_days)
    od = (
        Path(args.output_dir)
        / f"{cfg.experiment_name}_{cfg.timestamp}"
    )
    od.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(od / "experiment.log")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logging.getLogger().addHandler(fh)
    logger.info("Config: %s", json.dumps(asdict(cfg), indent=2))

    results = run_sweep(cfg)

    with open(od / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(od / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    if results:
        flds = list(results[0].keys())
        with open(od / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=flds)
            w.writeheader()
            w.writerows(results)

    report = generate_report(cfg, results)
    with open(od / "ops_report.md", "w") as f:
        f.write(report)

    try:
        from healthcare_sim_sdk.experiments.catalog import (
            ExperimentCatalog,
        )
        ExperimentCatalog().register(
            od, asdict(cfg),
            {"type": "threshold_optimizer",
             "runs": len(results)},
            notes=f"{len(results)} runs, 9 archetypes",
        )
    except Exception as e:
        logger.warning("Catalog: %s", e)

    print(f"\n{'=' * 60}")
    print("THRESHOLD OPTIMIZER COMPLETE")
    print(f"{'=' * 60}")
    print(f"Runs:   {len(results)}")
    print(f"Output: {od}")
    print(f"Report: {od / 'ops_report.md'}")
    print(f"CSV:    {od / 'results.csv'}")


if __name__ == "__main__":
    main()
