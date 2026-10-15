"""Conservative, reproducible dataset cleaning using only the standard library."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import unicodedata

PROJECT = Path(__file__).resolve().parents[2]
DOLLY_FIELDS = ["instruction", "context", "response", "category"]
FRESH_FIELDS = ["id", "split", "question", "effective_year", "next_review",
                "false_premise", "num_hops", "fact_type", "source",
                *[f"answer_{i}" for i in range(10)], "note"]
CATEGORIES = {"closed_qa", "classification", "open_qa", "information_extraction",
              "brainstorming", "general_qa", "summarization", "creative_writing"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def normalize(value):
    """Preserve meaningful spacing/indentation and paragraphs; no token filtering."""
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return "" if not value.strip() else value


def fingerprint(row, fields):
    # Review grouping only, never a reason to delete differently answered prompts.
    values = [" ".join(row[field].casefold().split()) for field in fields]
    return digest(json.dumps(values, ensure_ascii=False).encode("utf-8"))


def clean(records, fields, request_field, context_fields, metadata_fields, dataset):
    kept, provenance, review = [], [], []
    seen, groups = {}, defaultdict(list)
    counts = Counter()
    empty_before = Counter()
    for source_row, row, parse_error in records:
        counts["input_records"] += 1
        if parse_error or not isinstance(row, dict) or any(
            field not in row or not isinstance(row[field], str) for field in fields
        ):
            counts["invalid_records_excluded"] += 1
            review.append({"source_row": source_row, "cleaned_row": "",
                           "issue": "invalid_record_excluded", "detail": parse_error or "invalid field schema"})
            continue
        if set(row) != set(fields):
            raise ValueError(f"Unexpected {dataset} columns; review the schema before cleaning.")
        empty_before.update(field for field in fields if not row[field].strip())
        cleaned = {field: normalize(row[field]) for field in fields}
        for field in metadata_fields:
            cleaned[field] = cleaned[field].strip()
        changed = [field for field in fields if cleaned[field] != row[field]]
        counts["normalized_fields"] += len(changed)
        counts["normalized_records"] += bool(changed)
        if not cleaned[request_field].strip():
            counts["empty_requests_excluded"] += 1
            review.append({"source_row": source_row, "cleaned_row": "",
                           "issue": "empty_request_excluded", "detail": "request is whitespace-only"})
            continue
        key = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
        if key in seen:
            counts["identical_records_removed"] += 1
            review.append({"source_row": source_row, "cleaned_row": seen[key],
                           "issue": "identical_record_removed", "detail": "all source fields match retained record"})
            continue
        cleaned_row = len(kept) + 1
        seen[key] = cleaned_row
        kept.append(cleaned)
        group_id = fingerprint(cleaned, context_fields)
        groups[group_id].append(cleaned_row)
        provenance.append({"cleaned_row": cleaned_row, "source_row": source_row,
                           "original_id": cleaned.get("id", ""), "input_group_id": group_id,
                           "normalized_fields": ",".join(changed)})
        issues = []
        if len(cleaned[request_field].split()) < 2:
            issues.append(("short_request_review", "fewer than two whitespace-separated words; retained"))
        if any("\ufffd" in value for value in cleaned.values()):
            issues.append(("replacement_character_review", "Unicode replacement character present; retained"))
        if any(unicodedata.category(char) == "Cc" and char not in "\n\t"
               for value in cleaned.values() for char in value):
            issues.append(("control_character_review", "unusual control character present; retained"))
        if dataset == "dolly" and cleaned["category"] not in CATEGORIES:
            issues.append(("unexpected_category_review", cleaned["category"]))
        if dataset == "freshqa":
            expected = {"split": {"DEV", "TEST"}, "false_premise": {"TRUE", "FALSE"},
                        "num_hops": {"one-hop", "multi-hop"},
                        "fact_type": {"never-changing", "slow-changing", "fast-changing"}}
            for field, values in expected.items():
                if cleaned[field] not in values:
                    issues.append(("unexpected_metadata_review", field + ": " + cleaned[field]))
            if not cleaned["answer_0"].strip():
                issues.append(("empty_reference_answer_review", "answer_0 is empty; retained"))
        for issue, detail in issues:
            review.append({"source_row": source_row, "cleaned_row": cleaned_row,
                           "issue": issue, "detail": detail})
    repeated = {key: values for key, values in groups.items() if len(values) > 1}
    for group_id, members in repeated.items():
        for member in members:
            review.append({"source_row": provenance[member - 1]["source_row"],
                           "cleaned_row": member, "issue": "repeated_input_review",
                           "detail": group_id + "; cleaned rows " + ",".join(map(str, members))})
    counts["output_records"] = len(kept)
    counts["repeated_input_groups_retained"] = len(repeated)
    counts["records_in_repeated_input_groups"] = sum(map(len, repeated.values()))
    counts["review_entries"] = len(review)
    for field in ("invalid_records_excluded", "empty_requests_excluded", "identical_records_removed"):
        counts.setdefault(field, 0)
    assert counts["input_records"] == (len(kept) + counts["invalid_records_excluded"]
                                        + counts["empty_requests_excluded"]
                                        + counts["identical_records_removed"])
    audit = {**dict(counts), "empty_fields_before": dict(empty_before),
             "empty_fields_after": {field: sum(not row[field].strip() for row in kept) for field in fields},
             "review_issue_counts": dict(Counter(row["issue"] for row in review))}
    return kept, provenance, review, audit


def csv_text(rows, fields):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def write_text(path, content):
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-date", help="Date for the audit record, YYYY-MM-DD")
    args = parser.parse_args()
    raw = PROJECT / "data" / "raw"
    output = PROJECT / "data" / "cleaned"
    if not output.resolve().is_relative_to(PROJECT / "data") or output.resolve() == raw.resolve():
        raise ValueError("Output must be a separate project data directory.")
    inputs = {"dolly": raw / "databricks-dolly-15k.jsonl",
              "freshqa": raw / "freshqa_2026-04-21.csv"}
    originals = {name: path.read_bytes() for name, path in inputs.items()}
    dolly_records = []
    for source_row, line in enumerate(originals["dolly"].decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            dolly_records.append((source_row, json.loads(line), None))
        except json.JSONDecodeError as exc:
            dolly_records.append((source_row, None, str(exc)))
    csv_rows = list(csv.reader(io.StringIO(originals["freshqa"].decode("utf-8-sig"), newline="")))
    header_index = next(i for i, row in enumerate(csv_rows) if row == FRESH_FIELDS)
    fresh_records = []
    blank_rows = 0
    for source_row, row in enumerate(csv_rows[header_index + 1:], header_index + 2):
        if not any(value.strip() for value in row):
            blank_rows += 1
            continue
        fresh_records.append((source_row, dict(zip(FRESH_FIELDS, row)),
                              None if len(row) == len(FRESH_FIELDS) else "incorrect CSV field count"))
    jobs = {
        "dolly": (dolly_records, DOLLY_FIELDS, "instruction", ["instruction", "context"], ["category"]),
        "freshqa": (fresh_records, FRESH_FIELDS, "question", ["question"],
                    ["id", "split", "effective_year", "next_review", "false_premise", "num_hops", "fact_type"]),
    }
    output.mkdir(parents=True, exist_ok=True)
    audit = {"record_date": args.record_date or datetime.now(timezone(timedelta(hours=5.5))).date().isoformat(),
             "scope": "cleaning only; no integration, selection, labelling, splitting or training",
             "datasets": {}}
    for name, parameters in jobs.items():
        kept, provenance, review, summary = clean(*parameters, dataset=name)
        filename = "databricks-dolly-15k.cleaned.jsonl" if name == "dolly" else "freshqa_2026-04-21.cleaned.csv"
        destination = output / filename
        if name == "dolly":
            content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept)
            assert [json.loads(line) for line in content.splitlines()] == kept
        else:
            content = csv_text(kept, FRESH_FIELDS)
            assert list(csv.DictReader(io.StringIO(content, newline=""))) == kept
            summary["introductory_rows_removed"] = header_index
            summary["blank_data_rows_removed"] = blank_rows
            summary["id_collision_groups"] = sum(count > 1 for count in Counter(row["id"] for row in kept).values())
            summary["metadata_distributions"] = {field: dict(Counter(row[field] for row in kept))
                                                  for field in ("split", "fact_type", "false_premise", "num_hops")}
        write_text(destination, content)
        write_text(output / f"{name}.provenance.csv", csv_text(provenance,
                   ["cleaned_row", "source_row", "original_id", "input_group_id", "normalized_fields"]))
        write_text(output / f"{name}.review.csv", csv_text(review,
                   ["source_row", "cleaned_row", "issue", "detail"]))
        if name == "dolly":
            assert [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()] == kept
            summary["metadata_distributions"] = {"category": dict(Counter(row["category"] for row in kept))}
        else:
            with destination.open(encoding="utf-8", newline="") as handle:
                assert list(csv.DictReader(handle)) == kept
        summary.update({"raw_file": inputs[name].relative_to(PROJECT).as_posix(),
                        "cleaned_file": destination.relative_to(PROJECT).as_posix(),
                        "raw_sha256": digest(originals[name]),
                        "cleaned_sha256": digest(destination.read_bytes())})
        audit["datasets"][name] = summary
    for name, path in inputs.items():
        assert path.read_bytes() == originals[name], "Raw source changed during cleaning"
    audit["raw_files_unchanged"] = True
    write_text(output / "cleaning_audit.json", json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    run()
