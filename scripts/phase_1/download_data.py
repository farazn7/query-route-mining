"""Download official raw datasets without replacing existing local files."""
import csv
import io
import json
from pathlib import Path
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
SOURCES = {
    "databricks-dolly-15k.jsonl": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl?download=true",
    "freshqa_2026-04-21.csv": "https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv",
}


def download():
    raw = PROJECT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = dict(SOURCES)
    for filename, url in sources.items():
        target = raw / filename
        if target.exists():
            print("Kept existing:", filename, target.stat().st_size, "bytes")
            continue
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        # Validate before saving, so an HTML error/login page cannot masquerade as data.
        if filename.endswith(".jsonl"):
            records = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
            if not records or not all(isinstance(row, dict) for row in records):
                raise ValueError("Invalid Dolly response")
        elif filename.endswith(".csv"):
            records = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"), newline="")))
            if not any("question" in row and "id" in row for row in records):
                raise ValueError("FreshQA response has no expected header")
        with target.open("xb") as handle:
            handle.write(payload)
        print("Downloaded:", filename, len(payload), "bytes")


if __name__ == "__main__":
    download()
