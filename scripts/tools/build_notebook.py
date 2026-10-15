"""Build a self-contained data notebook from the maintained standard-library scripts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
            "source": source.strip().splitlines(keepends=True)}
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    if kind == "code":
        cell.update({"execution_count": None, "outputs": []})
    cells.append(cell)


add("markdown", """
# Data preparation

Download, inspect, clean, and align Dolly-15k and FreshQA. The datasets remain separate.
""")
add("markdown", """
## Setup

Select the project `.venv` kernel. See the README for installation commands.
""")
add("code", r"""
from pathlib import Path
import sys
import json
import csv
import hashlib
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / "README.md").is_file() and (p / "pyproject.toml").is_file()), None)
if PROJECT_ROOT is None:
    raise RuntimeError("Open the notebook inside a copy of this project.")
if Path(sys.prefix).resolve() != (PROJECT_ROOT / ".venv").resolve():
    raise RuntimeError("Select this project's .venv kernel before running data operations.")
RUN_DATE = datetime.now(timezone(timedelta(hours=5.5))).date().isoformat()
print("Project venv verified:", PROJECT_ROOT.name)
print("Python version:", sys.version.split()[0])
print("Audit date (IST):", RUN_DATE)
""")
add("markdown", """
## Download

Fetch the original datasets; keep existing downloads.
""")
download_source = (ROOT / "scripts" / "phase_1" / "download_data.py").read_text(encoding="utf-8")
download_source = download_source.replace("PROJECT = Path(__file__).resolve().parents[2]", "PROJECT = PROJECT_ROOT")
download_source = download_source.split('\nif __name__ == "__main__":', 1)[0]
add("code", download_source + "\n\ndownload()\nRAW_DIR = PROJECT / 'data' / 'raw'\n")
add("markdown", """
## Inspect

Check record counts, columns, and source categories.
""")
add("code", r"""
from collections import Counter

dolly_raw = [json.loads(line) for line in (RAW_DIR / "databricks-dolly-15k.jsonl")
             .read_text(encoding="utf-8-sig").splitlines() if line.strip()]
with (RAW_DIR / "freshqa_2026-04-21.csv").open(encoding="utf-8-sig", newline="") as handle:
    fresh_raw_rows = list(csv.reader(handle))
fresh_header_index = next(i for i, row in enumerate(fresh_raw_rows) if "question" in row and "id" in row)
fresh_header = fresh_raw_rows[fresh_header_index]
fresh_raw = [dict(zip(fresh_header, row)) for row in fresh_raw_rows[fresh_header_index + 1:]
             if any(value.strip() for value in row)]
print("Dolly rows:", len(dolly_raw))
print("Dolly columns:", list(dolly_raw[0]))
print("Dolly categories:", dict(Counter(row["category"] for row in dolly_raw)))
print("FreshQA rows:", len(fresh_raw))
print("FreshQA header starts at CSV record:", fresh_header_index + 1)
print("FreshQA columns:", fresh_header)
print("FreshQA fact types:", dict(Counter(row["fact_type"] for row in fresh_raw)))
""")
add("markdown", """
## Clean

Normalize Unicode and line endings, remove fully identical records, and flag review cases.
""")
clean_source = (ROOT / "scripts" / "phase_1" / "clean_datasets.py").read_text(encoding="utf-8")
clean_source = clean_source.replace("PROJECT = Path(__file__).resolve().parents[2]", "PROJECT = PROJECT_ROOT")
prefix, remaining = clean_source.split("def clean(", 1)
clean_function, remaining = ("def clean(" + remaining).split("def csv_text(", 1)
helpers, run_source = ("def csv_text(" + remaining).split("def run():", 1)
run_source = ("def run():" + run_source).split('\nif __name__ == "__main__":', 1)[0]
run_start = run_source.index('    raw = PROJECT / "data" / "raw"')
run_source = "def clean_sources(record_date=None):\n    args = argparse.Namespace(record_date=record_date)\n" + run_source[run_start:]
add("code", prefix)
add("code", clean_function)
add("code", helpers + "\n\n" + run_source)
add("code", r"""
# Hand-checkable checks for the important cleaning decisions.
assert normalize("line one\r\n    code\rline three") == "line one\n    code\nline three"
assert normalize("e\u0301") == "\u00e9"
assert normalize(" \t\n") == ""
sample = {"instruction": "Explain this example", "context": "", "response": "First answer", "category": "general_qa"}
alternative = {**sample, "response": "Another answer"}
empty = {**sample, "instruction": "  "}
kept, _, _, check_audit = clean([(1, sample, None), (2, dict(sample), None),
                                (3, alternative, None), (4, empty, None)],
                               DOLLY_FIELDS, "instruction", ["instruction", "context"], ["category"], "dolly")
assert len(kept) == 2 and kept[1]["response"] == "Another answer"
assert check_audit["identical_records_removed"] == 1
assert check_audit["empty_requests_excluded"] == 1
assert kept[0]["context"] == ""
quoted = [{"a": 'a comma, then a quote "', "b": "line one\nline two"}]
assert list(csv.DictReader(io.StringIO(csv_text(quoted, ["a", "b"]), newline=""))) == quoted
print("Cleaning checks passed.")
""")
add("code", r"""
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"
expected_cleaned = ["databricks-dolly-15k.cleaned.jsonl", "freshqa_2026-04-21.cleaned.csv",
                    "dolly.provenance.csv", "dolly.review.csv", "freshqa.provenance.csv",
                    "freshqa.review.csv", "cleaning_audit.json"]
present = [(CLEANED_DIR / filename).exists() for filename in expected_cleaned]
if any(present) and not all(present):
    raise RuntimeError("Partial cleaned output exists. Inspect it before continuing; nothing was overwritten.")
if all(present):
    print("Kept existing cleaned datasets and audit.")
else:
    clean_sources(record_date=RUN_DATE)
cleaning_audit = json.loads((CLEANED_DIR / "cleaning_audit.json").read_text(encoding="utf-8"))
for name, info in cleaning_audit["datasets"].items():
    assert hashlib.sha256((PROJECT_ROOT / info["raw_file"]).read_bytes()).hexdigest() == info["raw_sha256"]
    assert hashlib.sha256((PROJECT_ROOT / info["cleaned_file"]).read_bytes()).hexdigest() == info["cleaned_sha256"]
    print(name, ":", info["input_records"], "->", info["output_records"], "records")
""")
add("markdown", """
## Align columns

Both tables use `id`, `prompt`, `user_context`, `route`, and `label_reason`.
Answers and source metadata stay in companion files. Annotation fields start blank.
""")
prep_source = (ROOT / "scripts" / "phase_1" / "prepare_columns.py").read_text(encoding="utf-8")
prep_source = prep_source.replace("PROJECT = Path(__file__).resolve().parents[2]", "PROJECT = PROJECT_ROOT")
prep_prefix, prep_run = prep_source.split("def main():", 1)
prep_run = ("def main():" + prep_run).split('\nif __name__ == "__main__":', 1)[0]
prep_start = prep_run.index('    cleaned = PROJECT / "data" / "cleaned"')
prep_run = "def prepare_separate_tables(record_date=None):\n    args = argparse.Namespace(record_date=record_date)\n" + prep_run[prep_start:]
add("code", prep_prefix)
add("code", prep_run)
add("code", r"""
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
expected_prepared = ["dolly.annotation.csv", "dolly.metadata.jsonl", "freshqa.annotation.csv",
                     "freshqa.metadata.jsonl", "column_preparation_audit.json"]
present = [(PREPARED_DIR / filename).exists() for filename in expected_prepared]
if any(present) and not all(present):
    raise RuntimeError("Partial prepared output exists. Inspect it first; nothing was overwritten.")
if all(present):
    print("Kept existing annotation files: possible human labels will not be overwritten.")
else:
    prepare_separate_tables(record_date=RUN_DATE)
""")
add("markdown", """
## Validate

Check row counts, text preservation, IDs, metadata joins, and allowed route values.
""")
add("code", r"""
with (CLEANED_DIR / "freshqa_2026-04-21.cleaned.csv").open(encoding="utf-8", newline="") as handle:
    cleaned_fresh = list(csv.DictReader(handle))
cleaned_sources = {
    "dolly": [json.loads(line) for line in (CLEANED_DIR / "databricks-dolly-15k.cleaned.jsonl")
              .read_text(encoding="utf-8").splitlines()],
    "freshqa": cleaned_fresh,
}
all_prepared_ids = set()
for name, original in cleaned_sources.items():
    with (PREPARED_DIR / f"{name}.annotation.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == FIELDS
        prepared = list(reader)
    metadata = [json.loads(line) for line in (PREPARED_DIR / f"{name}.metadata.jsonl")
                .read_text(encoding="utf-8").splitlines()]
    assert len(prepared) == len(original) == len(metadata)
    identifiers = [row["id"] for row in prepared]
    assert len(set(identifiers)) == len(identifiers)
    assert not all_prepared_ids.intersection(identifiers)
    all_prepared_ids.update(identifiers)
    assert identifiers == [row["id"] for row in metadata]
    for old, new in zip(original, prepared, strict=True):
        assert new["prompt"] == old["instruction" if name == "dolly" else "question"]
        assert new["user_context"] == (old["context"] if name == "dolly" else "")
        assert new["route"] in {"", "rag", "internet", "direct"}
    print(name, ":", len(prepared), "rows;", sum(bool(row["route"]) for row in prepared), "route labels assigned")
    for row in prepared[:2]:
        print({"id": row["id"], "prompt_preview": row["prompt"][:100], "route": row["route"]})
print("Validation passed. No datasets were merged and no route labels were generated.")
""")
add("markdown", """
## Output

Separate annotation tables are in `data/prepared/`. No routes have been assigned.
""")

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python (select project .venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python", "pygments_lexer": "ipython3"}},
    "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks" / "data_preparation.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Created {destination.name}: {len(cells)} cells")
