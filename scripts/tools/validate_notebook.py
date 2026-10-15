"""Execute the standard-Python notebook cells and save outputs; no Jupyter required."""
import contextlib
import io
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
NOTEBOOK = PROJECT / "notebooks" / "data_preparation.ipynb"


def main():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if notebook["nbformat"] != 4:
        raise ValueError("Unsupported notebook format")
    identifiers = [cell["id"] for cell in notebook["cells"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate cell IDs")
    namespace = {"__name__": "__main__"}
    count = 0
    for index, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        count += 1
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
        cell["execution_count"] = count
        output = captured.getvalue()
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}]
                           if output else [])
        print(f"Cell {index} passed")
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"All {count} code cells passed; outputs saved.")


if __name__ == "__main__":
    main()
