# Query Route Mining

A study of frequent itemset mining, association rules, and classification for
routing assistant requests to document retrieval, internet search, or a direct answer.

**Status:** dataset cleaning and column preparation are implemented. Route labelling
and the three-route classifier are not complete. No accuracy improvement is claimed.

## Setup

Python 3.12+ is required for the listed dependencies; development has used Python 3.13.
From a new checkout:

```powershell
git clone https://github.com/farazn7/query-route-mining.git
cd query-route-mining
python -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

`requirements.txt` lists the notebook tools. The data scripts use the standard library.

## Prepare the data

Run from the repository root:

```powershell
& '.\.venv\Scripts\python.exe' -B scripts/phase_1/download_data.py
& '.\.venv\Scripts\python.exe' -B scripts/phase_1/clean_datasets.py
& '.\.venv\Scripts\python.exe' -B scripts/phase_1/prepare_columns.py
```

These three scripts use only Python's standard library, so installing requirements
is optional if you only need data preparation. Downloads remain in `data/raw/`,
cleaned copies in `data/cleaned/`, and separate annotation tables in `data/prepared/`.
These directories are excluded from Git. Existing annotation files are not overwritten.

The prepared tables share `id`, `prompt`, `user_context`, `route`, and `label_reason`.
Route labels initially remain blank. Answers and source annotations are stored in
companion metadata files, not classifier inputs.

## Notebook

```powershell
& '.\.venv\Scripts\python.exe' -m jupyterlab
```

Open [data_preparation.ipynb](notebooks/data_preparation.ipynb), select the
project's `.venv` kernel, and run its cells in order. It covers the same preparation
workflow. The notebook-processing cells have been checked with a Python runner;
a fresh Jupyter installation has not been tested in this project.

## Layout

- `scripts/phase_1/`: download, clean, and prepare the datasets.
- `scripts/tools/`: notebook generation and validation helpers.
- `notebooks/data_preparation.ipynb`: one executable walkthrough.
- `data/`: local datasets and generated outputs, excluded from Git.

Sources: [Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
and [FreshQA](https://github.com/freshllms/freshqa). The FreshQA download uses the
21 April 2026 sheet linked by its maintainers; its stored answers are historical.
