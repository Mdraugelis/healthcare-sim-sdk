#!/usr/bin/env python3
"""Execute notebook Python cells in order in a fresh process without sockets.

Usage: .venv/bin/python scripts/execute_notebook_headless.py

Uses matplotlib's Agg backend and embeds output figures/text in an executed
notebook under reports/. This validates ordinary Python cells, not Jupyter
magics, widget behavior, or front-end rendering. Use nbconvert for those checks.
"""
import argparse
import base64
import contextlib
import io
import os
from pathlib import Path
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default="examples/sanity_checks.ipynb")
    parser.add_argument("--output", default="reports/sanity_checks_executed.ipynb")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.notebook).resolve()
    output = (root / args.output).resolve()
    os.chdir(root)
    with tempfile.TemporaryDirectory(prefix="sim-notebook-") as cache:
        os.environ["MPLCONFIGDIR"] = str(Path(cache) / "matplotlib")
        os.environ["XDG_CACHE_HOME"] = str(Path(cache) / "cache")
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib.pyplot as plt
        import nbformat

        notebook = nbformat.read(source, as_version=4)
        # Remove saved evidence before execution, including cells not reached.
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        namespace = {"__name__": "__main__"}
        count = 0
        failed = False
        for i, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            outputs = []

            def show():
                for num in plt.get_fignums():
                    fig = plt.figure(num)
                    image = io.BytesIO()
                    fig.savefig(image, format="png", bbox_inches="tight")
                    outputs.append(nbformat.v4.new_output("display_data", data={
                        "image/png": base64.b64encode(image.getvalue()).decode(),
                        "text/plain": f"Figure from cell {i}",
                    }))
                plt.close("all")

            plt.show = show
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with (contextlib.redirect_stdout(stdout),
                      contextlib.redirect_stderr(stderr)):
                    code = compile(cell.source, f"{source.name} cell {i}", "exec")
                    exec(code, namespace)
            except Exception as exc:
                failed = True
                outputs.append(nbformat.v4.new_output(
                    "error", ename=type(exc).__name__, evalue=str(exc),
                    traceback=traceback.format_exc().splitlines()))
                traceback.print_exc()
            if stdout.getvalue():
                outputs.append(nbformat.v4.new_output(
                    "stream", name="stdout", text=stdout.getvalue()))
            if stderr.getvalue():
                outputs.append(nbformat.v4.new_output(
                    "stream", name="stderr", text=stderr.getvalue()))
            cell.outputs = outputs
            cell.execution_count = count
            print(f"Cell {i}: {'FAILED' if failed else 'passed'}", flush=True)
            if failed:
                break
        notebook.metadata["execution_method"] = (
            "fresh Python process; sequential execution of Python cells; "
            "matplotlib Agg; no Jupyter kernel or sockets")
        notebook.metadata["execution_passed"] = not failed
        output.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, output)
        print(f"{'Failed' if failed else 'Passed'} {count} code cells: {output}")
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
