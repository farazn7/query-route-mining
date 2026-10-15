"""Align and prune columns in separate datasets; do not merge or assign routes."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
FIELDS = ["id", "prompt", "user_context", "route", "label_reason"]


def csv_content(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def prepare(dataset, rows, provenance, review):
    if len(rows) != len(provenance):
        raise ValueError("Cleaned records and provenance have different lengths")
    flags = defaultdict(list)
    for issue in review:
        if issue["cleaned_row"] and issue["issue"] != "identical_record_removed":
            flags[int(issue["cleaned_row"])].append(issue["issue"])
    primary, metadata = [], []
    for index, (row, origin) in enumerate(zip(rows, provenance, strict=True), 1):
        if int(origin["cleaned_row"]) != index:
            raise ValueError("Provenance row order does not match cleaned data")
        source_id = origin["source_row"] if dataset == "dolly" else row["id"]
        identifier = f"{dataset}-{source_id}"
        prompt = row["instruction"] if dataset == "dolly" else row["question"]
        context = row["context"] if dataset == "dolly" else ""
        if not prompt.strip():
            raise ValueError("A source request is empty")
        primary.append({"id": identifier, "prompt": prompt, "user_context": context,
                        "route": "", "label_reason": ""})
        record = {"id": identifier, "source_dataset": dataset, "original_id": row.get("id"),
                  "source_row": int(origin["source_row"]),
                  "source_input_group_id": origin["input_group_id"],
                  "cleaning_flags": sorted(set(flags[index]))}
        if dataset == "dolly":
            record.update({"reference_answers": [row["response"]],
                           "source_metadata": {"category": row["category"]}})
        else:
            record.update({"reference_answers": [row[f"answer_{n}"] for n in range(10)
                                                 if row[f"answer_{n}"].strip()],
                           "source_metadata": {key: row[key] for key in
                                               ("split", "effective_year", "next_review", "false_premise",
                                                "num_hops", "fact_type", "source")}})
            if row["note"].strip():
                # Current snapshot has no notes; preserve unexpected future notes.
                record["source_metadata"]["note"] = row["note"]
        metadata.append(record)
    ids = [row["id"] for row in primary]
    if len(set(ids)) != len(ids):
        raise ValueError("Prepared IDs are not unique")
    return primary, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-date", help="Recorded audit date, YYYY-MM-DD")
    args = parser.parse_args()
    cleaned = PROJECT / "data" / "cleaned"
    output = PROJECT / "data" / "prepared"
    if not output.resolve().is_relative_to(PROJECT / "data"):
        raise ValueError("Output is outside the project data directory")
    filenames = [f"{name}.{suffix}" for name in ("dolly", "freshqa")
                 for suffix in ("annotation.csv", "metadata.jsonl")]
    filenames.append("column_preparation_audit.json")
    if any((output / name).exists() for name in filenames):
        raise FileExistsError("Prepared files already exist. Refusing to overwrite possible annotations.")
    source_paths = {
        "dolly": cleaned / "databricks-dolly-15k.cleaned.jsonl",
        "freshqa": cleaned / "freshqa_2026-04-21.cleaned.csv",
    }
    tracked = [*source_paths.values(), *[cleaned / f"{name}.{suffix}.csv"
               for name in ("dolly", "freshqa") for suffix in ("provenance", "review")]]
    before = {path: path.read_bytes() for path in tracked}
    with source_paths["freshqa"].open(encoding="utf-8", newline="") as handle:
        fresh = list(csv.DictReader(handle))
    sources = {"dolly": [json.loads(line) for line in before[source_paths["dolly"]]
                         .decode("utf-8").splitlines() if line.strip()], "freshqa": fresh}
    date = args.record_date or datetime.now(timezone(timedelta(hours=5.5))).date().isoformat()
    audit = {"record_date": date, "scope": "column alignment/pruning only; no merge, row selection or route labels",
             "shared_columns": FIELDS, "prediction_inputs": ["prompt", "user_context"], "datasets": {}}
    pending = {}
    all_ids = set()
    for name, rows in sources.items():
        with (cleaned / f"{name}.provenance.csv").open(encoding="utf-8", newline="") as handle:
            provenance = list(csv.DictReader(handle))
        with (cleaned / f"{name}.review.csv").open(encoding="utf-8", newline="") as handle:
            review = list(csv.DictReader(handle))
        primary, metadata = prepare(name, rows, provenance, review)
        ids = {row["id"] for row in primary}
        assert not all_ids & ids, "IDs overlap across datasets"
        all_ids.update(ids)
        content = csv_content(primary)
        assert list(csv.DictReader(io.StringIO(content, newline=""))) == primary
        assert len(primary) == len(rows) == len(metadata)
        assert all(not row["route"] and not row["label_reason"] for row in primary)
        assert all(list(row) == FIELDS for row in primary)
        assert [row["id"] for row in primary] == [row["id"] for row in metadata]
        # Assert the prediction text was not rewritten or confused with an answer.
        for source, prepared in zip(rows, primary, strict=True):
            assert prepared["prompt"] == source["instruction" if name == "dolly" else "question"]
            assert prepared["user_context"] == (source["context"] if name == "dolly" else "")
        pending[f"{name}.annotation.csv"] = content
        pending[f"{name}.metadata.jsonl"] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metadata)
        audit["datasets"][name] = {"source_columns": list(rows[0]), "source_rows": len(rows),
                                   "prepared_rows": len(primary), "blank_route_labels": len(primary),
                                   "metadata_records": len(metadata), "id_uniqueness_checked": True,
                                   "prompt_and_context_preserved": True}
    for path, content in before.items():
        assert path.read_bytes() == content, "An input file changed"
    audit["inputs_unchanged"] = True
    audit["input_sha256"] = {path.relative_to(PROJECT).as_posix(): hashlib.sha256(content).hexdigest()
                              for path, content in before.items()}
    audit["output_sha256"] = {name: hashlib.sha256(content.encode("utf-8")).hexdigest()
                               for name, content in pending.items()}
    pending["column_preparation_audit.json"] = json.dumps(audit, indent=2) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in pending.items():
        with (output / filename).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    for name in sources:
        with (output / f"{name}.annotation.csv").open(encoding="utf-8", newline="") as handle:
            assert len(list(csv.DictReader(handle))) == len(sources[name])
        actual_metadata = [json.loads(line) for line in (output / f"{name}.metadata.jsonl")
                           .read_text(encoding="utf-8").splitlines()]
        assert len(actual_metadata) == len(sources[name])
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
