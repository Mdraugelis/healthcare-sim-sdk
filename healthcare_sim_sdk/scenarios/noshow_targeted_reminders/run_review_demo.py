"""Write a public review packet for targeted no-show reminders.

Usage (from the repository root):
    python -m healthcare_sim_sdk.scenarios.noshow_targeted_reminders.run_review_demo

Output: outputs/review_demo/noshow_reminders/review_packet.md, five role
views, evidence.json, manifest.json and per-run trace files. The cells and
seeds come from configs/review_demo.yaml. The packet is a worked example of
the review workflow on synthetic data; it certifies nothing.
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from ...experiments.review import ReviewPacket, evidence, load_registry
from ...experiments.uncertainty import summarize_metrics
from .review import REGISTRY, run_reviewed

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "review_demo.yaml"


def run_demo(output_dir, config_path=DEFAULT_CONFIG) -> Path:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = {f"cell_{i:02d}": cell for i, cell in enumerate(config["cells"])}
    config["resolved_cells"] = [
        {"cell": cell_id, "label": cell["label"]}
        for cell_id, cell in cells.items()]
    expected = [(cell_id, seed) for cell_id in cells
                for seed in config["seeds"]]
    packet = ReviewPacket(output_dir, load_registry(REGISTRY), config,
                          expected)

    analysis = {}
    for cell_id, cell in cells.items():
        rows = []
        for seed in config["seeds"]:
            try:
                metrics, checks, cases, artifacts = run_reviewed(
                    cell_id, cell, config, seed, output_dir)
            except Exception as exc:  # recorded, never hidden
                packet.add_run(cell_id, seed, error=f"{type(exc).__name__}: "
                               f"{exc}")
                continue
            packet.add_run(cell_id, seed, metrics=metrics, checks=checks,
                           cases=cases, artifacts=artifacts)
            rows.append({"seed": seed, **metrics})
        if len(rows) == len(config["seeds"]):
            analysis[cell_id] = {
                "label": cell["label"],
                "summary": summarize_metrics(
                    rows, ["reduction_pp", "noshows_prevented"],
                    expected_seeds=config["seeds"]),
            }
    (output_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False))
    packet.add_artifact("analysis.json")

    # Experiment-level claims this demo does not attempt; left visible.
    packet.add_evidence(evidence(
        "model_heldout_auc", "NOT_RUN",
        detail="No held-out check of the simulated model's AUC was run for "
               "this packet.", category="model"))
    packet.add_evidence(evidence(
        "study_design_validation", "NOT_RUN",
        detail="No real-world study design has been specified or tested.",
        category="analysis"))
    return packet.write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",
                        default="outputs/review_demo/noshow_reminders")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(run_demo(args.output_dir, args.config))


if __name__ == "__main__":
    main()
