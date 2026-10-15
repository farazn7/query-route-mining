"""Combine prepared tables and metadata without selecting rows or assigning routes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata

PROJECT = Path(__file__).resolve().parents[2]
FIELDS = ["id", "prompt", "user_context", "route", "label_reason"]
REVIEW_FIELDS = ["input_group_id", "issue", "ids", "source_datasets", "route_labels"]
ROUTES = {"", "rag", "internet", "direct"}


def read_source(folder, name):
    with (folder / f"{name}.annotation.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"{name}: unexpected annotation columns")
        rows = list(reader)
    metadata = [json.loads(line) for line in (folder / f"{name}.metadata.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()]
    return {"name": name, "rows": rows, "metadata": metadata}


def group_key(row):
    values = [" ".join(unicodedata.normalize("NFC", row[key]).casefold().split())
              for key in ("prompt", "user_context")]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()


def combine(sources):
    rows, metadata, groups = [], [], defaultdict(list)
    seen_ids = set()
    for source in sources:
        original_metadata = source["metadata"]
        lookup = {record["id"]: record for record in original_metadata}
        source_ids = [row["id"] for row in source["rows"]]
        if len(lookup) != len(original_metadata) or len(set(source_ids)) != len(source_ids):
            raise ValueError(f"{source['name']}: duplicate IDs")
        if set(source_ids) != set(lookup):
            raise ValueError(f"{source['name']}: annotation/metadata IDs do not match")
        for row in source["rows"]:
            if set(row) != set(FIELDS) or any(not isinstance(row[key], str) for key in FIELDS):
                raise ValueError("Invalid annotation record")
            if not row["id"] or not row["prompt"].strip() or row["route"] not in ROUTES:
                raise ValueError("Invalid ID, prompt, or route label")
            if row["id"] in seen_ids:
                raise ValueError("ID collision across sources")
            seen_ids.add(row["id"])
            original = lookup[row["id"]]
            if original.get("source_dataset") != source["name"]:
                raise ValueError("Metadata source does not match its table")
            key = group_key(row)
            rows.append(dict(row))
            metadata.append({**original, "input_group_id": key})
            groups[key].append({"id": row["id"], "source": source["name"], "route": row["route"]})
    review = []
    cross_source = conflicts = 0
    for key, members in groups.items():
        if len(members) < 2:
            continue
        names = sorted({member["source"] for member in members})
        labels = sorted({member["route"] for member in members if member["route"]})
        conflict = len(labels) > 1
        cross_source += len(names) > 1
        conflicts += conflict
        review.append({"input_group_id": key,
                       "issue": "repeated_input;route_conflict" if conflict else "repeated_input",
                       "ids": ";".join(member["id"] for member in members),
                       "source_datasets": ";".join(names), "route_labels": ";".join(labels)})
    stats = {"records": len(rows), "metadata_records": len(metadata),
             "sources": {source["name"]: len(source["rows"]) for source in sources},
             "route_counts": dict(Counter(row["route"] or "unlabelled" for row in rows)),
             "repeated_input_groups": len(review), "cross_source_repeated_input_groups": cross_source,
             "conflicting_route_groups": conflicts, "records_removed": 0}
    return rows, metadata, review, stats


def as_csv(rows, fields):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def integrate(record_date=None):
    prepared = PROJECT / "data" / "prepared"
    output = PROJECT / "data" / "integrated"
    filenames = ["queries.csv", "metadata.jsonl", "review.csv", "integration_audit.json"]
    if any((output / name).exists() for name in filenames):
        raise FileExistsError("Integrated files already exist; preserving possible annotations.")
    paths = [prepared / f"{name}.{suffix}" for name in ("dolly", "freshqa")
             for suffix in ("annotation.csv", "metadata.jsonl")]
    before = {path: path.read_bytes() for path in paths}
    sources = [read_source(prepared, name) for name in ("dolly", "freshqa")]
    rows, metadata, review, stats = combine(sources)
    assert rows == [row for source in sources for row in source["rows"]]
    assert [row["id"] for row in rows] == [record["id"] for record in metadata]
    assert len(rows) == sum(len(source["rows"]) for source in sources)
    query_csv = as_csv(rows, FIELDS)
    assert list(csv.DictReader(io.StringIO(query_csv, newline=""))) == rows
    payloads = {"queries.csv": query_csv,
                "metadata.jsonl": "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in metadata),
                "review.csv": as_csv(review, REVIEW_FIELDS)}
    audit = {"record_date": record_date or datetime.now(timezone(timedelta(hours=5.5))).date().isoformat(),
             "scope": "integration only; no row removal or route assignment", "columns": FIELDS, **stats,
             "input_sha256": {path.relative_to(PROJECT).as_posix(): hashlib.sha256(data).hexdigest()
                              for path, data in before.items()},
             "output_sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest()
                               for name, text in payloads.items()}}
    assert all(path.read_bytes() == data for path, data in before.items())
    audit["inputs_unchanged"] = True
    payloads["integration_audit.json"] = json.dumps(audit, indent=2) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    for filename, content in payloads.items():
        with (output / filename).open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    with (output / "queries.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == rows
    assert [json.loads(line) for line in (output / "metadata.jsonl")
            .read_text(encoding="utf-8").splitlines()] == metadata
    print(json.dumps(stats, indent=2))
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-date", help="Recorded audit date, YYYY-MM-DD")
    integrate(record_date=parser.parse_args().record_date)
