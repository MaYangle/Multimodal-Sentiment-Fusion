"""Freeze native train/validation scores and label-free final test IDs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def prepare(source, dataset, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids, records, test = set(), [], []
    counts = Counter()
    for row in rows:
        sid = row["video_id"] + ("/" if dataset == "simsv2" else "_") + row["clip_id"]
        if sid in ids or row["mode"] not in ("train", "valid", "test"):
            raise ValueError("Duplicate ID or unknown split")
        ids.add(sid); counts[row["mode"]] += 1
        if row["mode"] == "test":
            test.append(sid)
            continue
        score = float(row["label"])
        if not - (1 if dataset == "simsv2" else 3) <= score <= (1 if dataset == "simsv2" else 3):
            raise ValueError("Invalid native label")
        rec = {"id": sid, "video_id": row["video_id"], "clip_id": row["clip_id"], "split": row["mode"],
               "dataset": dataset, "score": score, "transcript": row["text"], "source_csv_sha256": source_hash}
        for m in ("T", "A", "V"):
            value = row.get("label_" + m)
            if value not in (None, ""):
                rec["score_" + m] = float(value)
        records.append(rec)
    expected = {"simsv2": {"train": 2722, "valid": 647, "test": 1034}, "mosei": {"train": 16326, "valid": 1871, "test": 4659}}
    if dict(counts) != expected[dataset]:
        raise ValueError("Unexpected author cohort size; review version before accepting")
    manifest = output / "train_valid.jsonl"
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    test_file = output / "test_ids.txt"
    test_file.write_text("".join(sid + "\n" for sid in sorted(test)), encoding="utf-8")
    report = {"dataset": dataset, "native_range": [-1, 1] if dataset == "simsv2" else [-3, 3],
              "source_csv_sha256": source_hash, "split_counts": dict(counts), "mapped_labels": False,
              "train_valid_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "test_ids_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(), "test_labels_exported": False}
    (output / "manifest_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--dataset", choices=("simsv2", "mosei"), required=True)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.dataset, a.output_dir), indent=2))
