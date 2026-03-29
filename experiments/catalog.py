"""Experiment catalog: registry, retrieval, and comparison.

Maintains a catalog.json index of all experiment runs. Each run is
registered after completion with key metadata for quick lookup.
Supports loading, comparing, and querying past experiments.

Usage:
    from experiments.catalog import ExperimentCatalog

    catalog = ExperimentCatalog()
    catalog.list_experiments()
    exp = catalog.load("20260328_220210")
    catalog.compare(["20260328_220210", "20260328_223015"])
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


CATALOG_PATH = Path(__file__).parent / "catalog.json"
OUTPUTS_DIR = Path(__file__).parent / "outputs"


class ExperimentCatalog:
    """Registry of all experiment runs."""

    def __init__(self, catalog_path: Path = CATALOG_PATH):
        self.catalog_path = catalog_path
        self._entries: List[Dict[str, Any]] = []
        if self.catalog_path.exists():
            with open(self.catalog_path) as f:
                self._entries = json.load(f)

    def register(
        self,
        output_dir: Path,
        config: Dict,
        summary: Dict,
        notes: str = "",
    ) -> None:
        """Register a completed experiment run."""
        entry = {
            "timestamp": config.get("timestamp", ""),
            "experiment_name": config.get("experiment_name", ""),
            "output_dir": str(output_dir),
            "n_patients": config.get("n_patients"),
            "n_days": config.get("n_days"),
            "seed": config.get("seed"),
            "model_auc": config.get("model_auc"),
            "ar1_rho": config.get("ar1_rho"),
            "ar1_sigma": config.get("ar1_sigma"),
            "baseline_threshold": config.get("baseline_threshold"),
            "baseline_auc": summary.get("baseline_auc"),
            "best_predictor_auc": summary.get("best_predictor_auc"),
            "best_predictor_threshold": summary.get(
                "best_predictor_threshold"
            ),
            "collision_rate_reduction": summary.get(
                "collision_rate_reduction"
            ),
            "control_utilization": summary.get("control_utilization"),
            "notes": notes,
            "registered_at": datetime.now().isoformat(),
        }

        # Replace existing entry with same timestamp, or append
        self._entries = [
            e for e in self._entries
            if e["timestamp"] != entry["timestamp"]
        ]
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e["timestamp"])
        self._save()

    def list_experiments(self) -> List[Dict]:
        """List all registered experiments."""
        return list(self._entries)

    def find(self, timestamp: str) -> Optional[Dict]:
        """Find an experiment entry by timestamp (partial match)."""
        for e in self._entries:
            if timestamp in e["timestamp"]:
                return e
        return None

    def load(self, timestamp: str) -> Optional[Dict]:
        """Load full results for an experiment."""
        entry = self.find(timestamp)
        if not entry:
            return None
        output_dir = Path(entry["output_dir"])
        results_path = output_dir / "results.json"
        if not results_path.exists():
            return None
        with open(results_path) as f:
            results = json.load(f)
        config_path = output_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        return {
            "entry": entry,
            "config": config,
            "results": results,
        }

    def compare(
        self, timestamps: List[str],
    ) -> List[Dict]:
        """Load summary data for multiple experiments."""
        entries = []
        for ts in timestamps:
            entry = self.find(ts)
            if entry:
                entries.append(entry)
        return entries

    def _save(self):
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(self._entries, f, indent=2)


def register_from_output_dir(output_dir: Path, notes: str = ""):
    """Register an experiment from its output directory."""
    catalog = ExperimentCatalog()
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    config = {}
    summary = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    catalog.register(output_dir, config, summary, notes)
    return catalog
