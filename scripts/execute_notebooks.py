"""Execute every example in a fresh kernel; fail CI on any cell error.

Executed output is written under gitignored outputs/, never into sources.
"""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "outputs" / "notebook_validation"
    destination.mkdir(parents=True, exist_ok=True)
    failed = []
    for path in sorted((root / "examples").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        try:
            NotebookClient(
                notebook,
                timeout=600,
                kernel_name="python3",
                resources={"metadata": {"path": str(root / "examples")}},
            ).execute()
            nbformat.write(notebook, destination / path.name)
            print(f"PASS {path.name}", flush=True)
        except Exception as exc:
            failed.append(path.name)
            print(f"FAIL {path.name}: {exc}", flush=True)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
