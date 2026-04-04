"""Experiment catalog: registry, retrieval, and comparison.

Generic persistence layer — stores experiment metadata and metrics
for any scenario. The catalog does not import scenario-specific code.

Usage:
    from healthcare_sim_sdk.experiments.catalog import ExperimentCatalog

    catalog = ExperimentCatalog()
    catalog.register(output_dir, config, metrics)
    catalog.list_experiments()
    catalog.list_by_scenario("noshow_overbooking")
    exp = catalog.load("20260328_220210")
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# Default to CWD so users' catalogs stay in their project
CATALOG_PATH = Path("catalog.json")


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
        metrics: Dict,
        notes: str = "",
    ) -> None:
        """Register a completed experiment.

        Stores a standard set of fields plus all metrics as a
        nested dict. Any scenario can store domain-specific
        metrics without changing the catalog schema.
        """
        entry = {
            "timestamp": config.get("timestamp", ""),
            "experiment_name": config.get(
                "experiment_name", ""
            ),
            "scenario": config.get("scenario", "unknown"),
            "output_dir": str(output_dir),
            "seed": config.get("seed"),
            "n_entities": config.get(
                "n_patients", config.get("n_entities")
            ),
            "n_timesteps": config.get(
                "n_days", config.get("n_timesteps")
            ),
            "metrics": metrics,
            "notes": notes,
            "registered_at": datetime.now().isoformat(),
        }

        # Idempotent: replace existing entry with same timestamp
        self._entries = [
            e for e in self._entries
            if e.get("timestamp") != entry["timestamp"]
        ]
        self._entries.append(entry)
        self._entries.sort(key=lambda e: e.get("timestamp", ""))
        self._save()

    def list_experiments(self) -> List[Dict]:
        """List all registered experiments."""
        return list(self._entries)

    def list_by_scenario(self, scenario: str) -> List[Dict]:
        """List experiments for a specific scenario."""
        return [
            e for e in self._entries
            if e.get("scenario") == scenario
        ]

    def find(self, timestamp: str) -> Optional[Dict]:
        """Find an experiment entry by timestamp (partial match)."""
        for e in self._entries:
            if timestamp in e.get("timestamp", ""):
                return e
        return None

    def load(self, timestamp: str) -> Optional[Dict]:
        """Load full results for an experiment."""
        entry = self.find(timestamp)
        if not entry:
            return None
        output_dir = Path(entry["output_dir"])

        config = _load_json(output_dir / "config.json")
        metrics = _load_json(output_dir / "metrics.json")
        results = _load_json(output_dir / "results.json")

        return {
            "entry": entry,
            "config": config,
            "metrics": metrics,
            "results": results,
        }

    def compare(
        self, timestamps: List[str],
    ) -> List[Dict]:
        """Load summary data for multiple experiments."""
        return [
            e for ts in timestamps
            for e in [self.find(ts)] if e
        ]

    def _save(self):
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(self._entries, f, indent=2)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load JSON, return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
