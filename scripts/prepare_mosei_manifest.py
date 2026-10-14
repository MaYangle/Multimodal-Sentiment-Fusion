"""Audit CMU-MOSEI competition split labels against Kaggle's file inventory.

This standard-library script performs metadata and file-inventory auditing
only. It does not train, score, download, or inspect media contents. The
competition describes the MOSEI score boundaries as one possible mapping; this
project applies the shared ``jevaffect.protocol`` mapping.

``FileName`` is parsed at its final underscore: the suffix must be a numeric
segment ID, while the complete prefix is retained as the source-video key.
Eleven-character YouTube IDs may themselves contain underscores. Numeric
prefixes found in the supplied competition CSV are retained as opaque numeric
source aliases and are not called YouTube IDs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.protocol import ProtocolError, map_mosei_sentiment


SPLITS = ("train", "valid", "test")
REQUIRED_COLUMNS = ("FileName", "sentiment_score")
YOUTUBE_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}\Z")
SEGMENT_RE = re.compile(r"\d+\Z")
SOURCE_PREFIX_RE = re.compile(r"[A-Za-z0-9_-]+\Z")


class ManifestError(ValueError):
    """Raised when split metadata, IDs, labels, or source grouping is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_filename(file_name: str) -> tuple[str, str, str]:
    """Return (source-video key, segment suffix, source-key type).

    Split at the final underscore because valid YouTube IDs can contain
    underscores. Accepted source keys are 11-character YouTube IDs or numeric
    aliases present in the supplied competition data.
    """
    if not isinstance(file_name, str) or not file_name or file_name != file_name.strip():
        raise ManifestError("FileName must be a non-empty, unpadded string")
    source_video, separator, segment_id = file_name.rpartition("_")
    if not separator or not segment_id or not SEGMENT_RE.fullmatch(segment_id):
        raise ManifestError("FileName must end with an underscore and numeric segment ID")
    if YOUTUBE_ID_RE.fullmatch(source_video):
        source_type = "youtube_id_11"
    elif source_video.isdigit():
        source_type = "numeric_alias"
    elif SOURCE_PREFIX_RE.fullmatch(source_video):
        raise ManifestError("source-video prefix is neither an 11-character YouTube ID nor numeric alias")
    else:
        raise ManifestError("source-video prefix contains unsupported characters")
    return source_video, segment_id, source_type


def _expected_media_paths(split: str, file_name: str) -> dict[str, str]:
    root = "CMU-MOSEI/cmumosei/{}/".format(split)
    return {
        "video": root + "video/{}.mp4".format(file_name),
        "audio": root + "audio/{}.wav".format(file_name),
        "text": root + "text/{}.txt".format(file_name),
    }


def _read_inventory(inventory_json: Path) -> tuple[set[str], dict[str, Any]]:
    if not inventory_json.is_file():
        raise ManifestError("files inventory does not exist: {}".format(inventory_json))
    inventory_hash = sha256_file(inventory_json)
    try:
        with inventory_json.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ManifestError("files inventory is not valid UTF-8 JSON") from exc

    entries = data.get("files") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ManifestError("files inventory must contain a 'files' list")
    names: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError("inventory entry {} is not an object".format(index))
        name = entry.get("name") or entry.get("ref")
        if not isinstance(name, str) or not name:
            raise ManifestError("inventory entry {} has no file name".format(index))
        names.append(name)
    unique_names = set(names)
    duplicate_paths = len(names) - len(unique_names)
    if duplicate_paths:
        raise ManifestError("files inventory contains duplicate paths: {}".format(duplicate_paths))

    return unique_names, {
        "inventory_sha256": inventory_hash,
        "inventory_entries": len(entries),
        "inventory_unique_paths": len(unique_names),
        "inventory_duplicate_paths": duplicate_paths,
        "inventory_complete": data.get("complete") is True,
    }


def _read_split_csv(path: Path, split: str) -> tuple[list[dict[str, Any]], str]:
    if not path.is_file():
        raise ManifestError("{} split CSV does not exist: {}".format(split, path))
    source_hash = sha256_file(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError("{} split CSV is empty".format(split))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ManifestError("{} split CSV contains duplicate headers".format(split))
        missing = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing:
            raise ManifestError("{} split CSV is missing columns: {}".format(split, ", ".join(missing)))
        for row_number, row in enumerate(reader, start=2):
            if row is None or all(value is None for value in row.values()):
                continue
            if None in row:
                raise ManifestError("{} row {} has more fields than the header".format(split, row_number))
            rows.append(row)
    return rows, source_hash


def _build_row(
    row: dict[str, Any],
    split: str,
    source_csv_sha256: str,
    inventory_paths: set[str],
    inventory_sha256: str,
) -> dict[str, Any]:
    file_name = row.get("FileName")
    source_video, segment_id, source_type = parse_filename(file_name)
    raw_label = row.get("sentiment_score")
    if raw_label is None or not raw_label.strip():
        raise ManifestError("{} row {} has an empty sentiment_score".format(split, file_name))
    try:
        score = float(raw_label)
    except ValueError as exc:
        raise ManifestError("{} row {} has a non-numeric sentiment_score".format(split, file_name)) from exc
    if not math.isfinite(score):
        raise ManifestError("{} row {} has a non-finite sentiment_score".format(split, file_name))
    try:
        sentiment_class = map_mosei_sentiment(score)
    except ProtocolError as exc:
        raise ManifestError("{} row {} has an unmappable sentiment_score".format(split, file_name)) from exc

    media_paths = _expected_media_paths(split, file_name)
    return {
        "ID": file_name,
        "raw_label": raw_label,
        "class": sentiment_class,
        "language": "en",
        "dataset": "CMU-MOSEI",
        "split": split,
        "source_video": source_video,
        "source_video_id_type": source_type,
        "segment_id": segment_id,
        "video_path": media_paths["video"],
        "audio_path": media_paths["audio"],
        "text_path": media_paths["text"],
        "video_present_in_inventory": media_paths["video"] in inventory_paths,
        "audio_present_in_inventory": media_paths["audio"] in inventory_paths,
        "text_present_in_inventory": media_paths["text"] in inventory_paths,
        "source_csv_sha256": source_csv_sha256,
        "inventory_sha256": inventory_sha256,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def prepare_mosei_manifest(
    label_csv_dir: Path,
    inventory_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Audit split CSVs and inventory, then emit per-split manifests and frozen test IDs."""
    inventory_paths, inventory_info = _read_inventory(inventory_json)
    all_records: list[dict[str, Any]] = []
    split_records: dict[str, list[dict[str, Any]]] = {}
    source_csv_hashes: dict[str, str] = {}
    seen_ids: set[str] = set()
    source_video_splits: dict[str, set[str]] = defaultdict(set)
    source_type_counts: Counter[str] = Counter()
    media_presence: dict[str, dict[str, int]] = {}
    split_row_counts: dict[str, int] = {}
    duplicate_ids_within_split = 0
    duplicate_ids_across_splits = 0

    for split in SPLITS:
        csv_path = label_csv_dir / "label_file_{}.csv".format(split)
        input_rows, source_hash = _read_split_csv(csv_path, split)
        source_csv_hashes[split] = source_hash
        records: list[dict[str, Any]] = []
        seen_in_split: set[str] = set()
        for row in input_rows:
            record = _build_row(row, split, source_hash, inventory_paths, inventory_info["inventory_sha256"])
            identifier = record["ID"]
            if identifier in seen_in_split:
                duplicate_ids_within_split += 1
                raise ManifestError("duplicate ID within {} split: {}".format(split, identifier))
            if identifier in seen_ids:
                duplicate_ids_across_splits += 1
                raise ManifestError("ID overlaps splits: {}".format(identifier))
            seen_in_split.add(identifier)
            seen_ids.add(identifier)
            source_video_splits[record["source_video"]].add(split)
            source_type_counts[record["source_video_id_type"]] += 1
            records.append(record)

        split_records[split] = records
        all_records.extend(records)
        split_row_counts[split] = len(records)
        media_presence[split] = {
            modality: sum(bool(record["{}_present_in_inventory".format(modality)]) for record in records)
            for modality in ("video", "audio", "text")
        }

    cross_split_sources = sum(len(splits) > 1 for splits in source_video_splits.values())
    if cross_split_sources:
        raise ManifestError(
            "source-video groups overlap splits: {} groups".format(cross_split_sources)
        )

    test_ids = sorted(record["ID"] for record in split_records["test"])
    test_id_bytes = ("\n".join(test_ids) + "\n").encode("utf-8")
    test_ids_sha256 = hashlib.sha256(test_id_bytes).hexdigest()

    summary = {
        "dataset": "CMU-MOSEI",
        "split_row_counts": split_row_counts,
        "split_csv_sha256": source_csv_hashes,
        **inventory_info,
        "duplicate_ids_within_split": duplicate_ids_within_split,
        "duplicate_ids_across_splits": duplicate_ids_across_splits,
        "source_video_id_type_row_counts": dict(sorted(source_type_counts.items())),
        "source_video_groups_cross_split": cross_split_sources,
        "media_inventory_present_by_split": media_presence,
        "media_inventory_missing_by_split": {
            split: {
                modality: split_row_counts[split] - media_presence[split][modality]
                for modality in ("video", "audio", "text")
            }
            for split in SPLITS
        },
        "test_ids_count": len(test_ids),
        "test_ids_sha256": test_ids_sha256,
        "mapping_note": "project-adopted application of the competition's one possible MOSEI mapping",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        _write_jsonl(output_dir / "mosei_{}_manifest.jsonl".format(split), records)
    test_ids_path = output_dir / "mosei_test_ids.txt"
    test_ids_path.write_bytes(test_id_bytes)
    (output_dir / "mosei_test_ids.sha256").write_text(
        "{}  mosei_test_ids.txt\n".format(test_ids_sha256), encoding="ascii", newline="\n"
    )
    (output_dir / "mosei_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-csv-dir", required=True, type=Path, help="Directory containing label_file_{train,valid,test}.csv")
    parser.add_argument("--files-json", required=True, type=Path, help="Kaggle files.json inventory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated manifests and audit summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = prepare_mosei_manifest(args.label_csv_dir, args.files_json, args.output_dir)
    except (ManifestError, OSError, UnicodeError) as exc:
        print("MOSEI manifest preparation failed: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
