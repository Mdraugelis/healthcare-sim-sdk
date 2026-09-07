"""Reproduce a public, explicitly uncertified, role-specific review packet."""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from ...experiments.review import load_registry
from .review import REGISTRY, create_packet, run_reviewed_cell
from .run_experiment import _equity_table
from .runner_common import config_from_dict, summarize


DEFAULT_CONFIG = Path(__file__).parent / "configs/review_demo.yaml"


def run_demo(output_dir, config_path=DEFAULT_CONFIG):
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = output_dir / "synthetic_model.yaml"
    OmegaConf.save(OmegaConf.create({"params": config["synthetic_model"]}), model)
    cells = [{"label": c["label"], "overrides": c["overrides"],
              "config": {**config["scenario"], **c["overrides"],
                         "model_params_path": str(model.resolve())}}
             for c in config["cells"]]
    registry = load_registry(REGISTRY)
    registry.setdefault("assumptions", []).append({
        "id": "public_synthetic_demonstration",
        "description": (
            "Public synthetic demonstration: model parameters are illustrative, "
            "not calibrated to an institution or certified clinical model."
        ),
        "reviewers": ["clinical", "operations", "statistics", "governance",
                      "engineering"],
        "provenance": "hypothetical",
        "source": "Review demo YAML synthetic_model parameters",
        "range": "Illustrative demonstration only; no certified parameter range",
        "owner": "Clinical and statistical reviewers (unassigned)",
    })
    packet = create_packet(output_dir, config, cells, config["seeds"], registry)
    packet.add_artifact("synthetic_model.yaml")
    analyses = {}
    for i, cell in enumerate(cells):
        cell_id = f"cell_{i:02d}"
        rows = run_reviewed_cell(packet, cell_id,
                                 config_from_dict(cell["config"]),
                                 config["horizon_days"], config["seeds"])
        analyses[cell_id] = {
            "label": cell["label"],
            "metrics": summarize(rows, ["delta_missed_pp", "delta_harm",
                                        "ml_sourced_assessments"]),
            "subgroups": _equity_table(rows),
        }
    (output_dir / "analysis.json").write_text(json.dumps(
        analyses, indent=2, allow_nan=False))
    packet.add_artifact("analysis.json")
    packet.write()
    return output_dir / "review_packet.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/review_demo/malnutrition")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(run_demo(args.output_dir, args.config))


if __name__ == "__main__":
    main()
