"""Build an auditable CH-SIMS v2 manifest without training or scoring.

The manifests preserve ``meta.csv``'s official ``mode`` values. By default,
unmappable test labels block the all-split manifest while still allowing a
train/valid-only manifest and a frozen test-ID list. The explicit
``--allow-unresolved-test-labels`` option writes an all-split JSONL with test
``Label: null`` and ``label_status: unresolved``. Training code must isolate
``train`` and ``valid`` from ``test``; unresolved test records are not scored.
Speaker IDs are absent from the expected metadata, so this script records
``speaker_known=false`` and makes no speaker-leakage claim.

Normalized transcript hashes use Unicode NFKC, case-folding, and collapsed
whitespace. Cross-split repetitions and differing mapped labels are reported,
never removed automatically. The summary intentionally omits label counts by
split, including the test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.protocol import ProtocolError, map_simsv2_label


REQUIRED_COLUMNS = ("video_id", "clip_id", "text", "label", "mode")
SPLIT_ORDER = ("train", "valid", "test")
MANIFEST_COLUMNS = (
    "ID",
    "raw_relative_path",
    "language",
    "dataset",
    "source_video",
    "speaker_known",
    "mode",
    "Label",
    "label_status",
    "label_reason",
    "raw_label",
    "source_annotation",
    "transcript_sha256",
    "source_csv_sha256",
)


class ManifestError(ValueError):
    """Raised when the metadata cannot be represented safely by this manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_transcript(text: str) -> str:
    """Return the normalization used for duplicate checks and transcript hashes."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def transcript_sha256(text: str) -> str:
    normalized = normalize_transcript(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_component(value: Any, field_name: str, row_number: int) -> str:
    if value is None:
        raise ManifestError("row {} has an empty {}".format(row_number, field_name))
    component = value.strip()
    if not component or component in (".", "..") or "/" in component or "\\" in component:
        raise ManifestError("row {} has an unsafe {}".format(row_number, field_name))
    return component


def _safe_raw_prefix(raw_dir_name: str) -> PurePosixPath:
    prefix = PurePosixPath(raw_dir_name.replace("\\", "/"))
    if prefix.is_absolute() or not prefix.parts or any(part in ("", ".", "..") for part in prefix.parts):
        raise ManifestError("raw directory must be a non-empty relative path without '..'")
    return prefix


def _row_to_record(
    row: dict[str, Any],
    row_number: int,
    raw_prefix: PurePosixPath,
    source_csv_sha256: str,
) -> tuple[dict[str, Any], str, Any, bool]:
    video_id = _safe_component(row.get("video_id"), "video_id", row_number)
    clip_id = _safe_component(row.get("clip_id"), "clip_id", row_number)
    mode = (row.get("mode") or "").strip()
    if mode not in SPLIT_ORDER:
        raise ManifestError(
            "row {} has unsupported mode {!r}; expected train, valid, or test".format(
                row_number, mode
            )
        )

    raw_label = (row.get("label") or "").strip()
    numeric_label = None
    label_is_valid = False
    mapped_label = ""
    label_reason = ""
    try:
        numeric_label = float(raw_label)
        if math.isfinite(numeric_label):
            mapped_label = map_simsv2_label(numeric_label)
            label_is_valid = True
        else:
            label_reason = "raw label is non-finite"
    except (ValueError, ProtocolError):
        label_reason = "raw label is not in the competition's discrete CH-SIMS v2 mapping"
    if not raw_label:
        label_reason = "raw label is empty"

    text = row.get("text") or ""
    if not isinstance(text, str):
        raise ManifestError("row {} has a non-text transcript".format(row_number))

    sample_id = "{}/{}".format(video_id, clip_id)
    raw_path = raw_prefix / video_id / (clip_id + ".mp4")
    normalized = normalize_transcript(text)
    record = {
        "ID": sample_id,
        "raw_relative_path": raw_path.as_posix(),
        "language": "zh",
        "dataset": "CH-SIMS-v2.0",
        "source_video": video_id,
        "speaker_known": "false",
        "mode": mode,
        "Label": mapped_label if label_is_valid else None,
        "label_status": "mapped" if label_is_valid else "unresolved",
        "label_reason": "" if label_is_valid else label_reason,
        "raw_label": raw_label,
        "source_annotation": (row.get("annotation") or "").strip(),
        "transcript_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "source_csv_sha256": source_csv_sha256,
    }
    return record, normalized, numeric_label, label_is_valid


def _read_records(
    meta_csv: Path,
    raw_dir_name: str,
) -> tuple[list[dict[str, Any]], list[str], list[Any], int, dict[str, Any]]:
    if not meta_csv.is_file():
        raise ManifestError("metadata CSV does not exist: {}".format(meta_csv))
    source_digest = sha256_file(meta_csv)
    raw_prefix = _safe_raw_prefix(raw_dir_name)
    records: list[dict[str, Any]] = []
    normalized_texts: list[str] = []
    raw_label_values: list[Any] = []
    invalid_label_rows = 0
    seen_ids: set[str] = set()

    with meta_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestError("metadata CSV is empty or has no header")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ManifestError("metadata CSV contains duplicate column names")
        missing_columns = sorted(set(REQUIRED_COLUMNS) - set(reader.fieldnames))
        if missing_columns:
            raise ManifestError("metadata CSV is missing columns: {}".format(", ".join(missing_columns)))

        for row_number, row in enumerate(reader, start=2):
            if row is None or all(value is None for value in row.values()):
                continue
            if None in row:
                raise ManifestError("row {} contains more fields than the header".format(row_number))
            record, normalized, raw_label_value, label_is_valid = _row_to_record(
                row, row_number, raw_prefix, source_digest
            )
            if not label_is_valid and record["mode"] != "test":
                raise ManifestError(
                    "row {} has an unmappable label in {}; train and valid labels must map".format(
                        row_number, record["mode"]
                    )
                )
            if record["ID"] in seen_ids:
                raise ManifestError("duplicate ID at row {}: {}".format(row_number, record["ID"]))
            seen_ids.add(record["ID"])
            records.append(record)
            normalized_texts.append(normalized)
            raw_label_values.append(raw_label_value)
            if not label_is_valid:
                invalid_label_rows += 1

    if not records:
        raise ManifestError("metadata CSV contains no data rows")
    return (
        records,
        normalized_texts,
        raw_label_values,
        invalid_label_rows,
        {"source_csv_sha256": source_digest},
    )


def _audit(
    records: list[dict[str, Any]],
    normalized_texts: list[str],
    raw_label_values: list[Any],
    invalid_label_rows: int,
    source_info: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    split_counts = {split: 0 for split in SPLIT_ORDER}
    source_splits: dict[str, set[str]] = defaultdict(set)
    text_groups: dict[str, list[tuple[str, str, Any]]] = defaultdict(list)
    test_ids: list[str] = []

    for record, normalized, raw_label_value in zip(records, normalized_texts, raw_label_values):
        split = record["mode"]
        split_counts[split] += 1
        source_splits[record["source_video"]].add(split)
        if normalized:
            text_groups[normalized].append((split, record["Label"], raw_label_value))
        if split == "test":
            test_ids.append(record["ID"])

    duplicate_text_groups = 0
    mapped_conflicting_text_groups = 0
    raw_score_disagreement_groups = 0
    for occurrences in text_groups.values():
        splits = {split for split, _label, _raw_value in occurrences}
        if len(splits) > 1:
            duplicate_text_groups += 1
            mapped_labels = {label for _split, label, _raw_value in occurrences if label}
            raw_values = {raw_value for _split, _label, raw_value in occurrences if raw_value is not None}
            if len(mapped_labels) > 1:
                mapped_conflicting_text_groups += 1
            if len(raw_values) > 1:
                raw_score_disagreement_groups += 1

    overlapping_source_videos = sum(len(splits) > 1 for splits in source_splits.values())
    test_ids = sorted(test_ids)
    test_id_bytes = ("\n".join(test_ids) + "\n").encode("utf-8")
    test_ids_sha256 = hashlib.sha256(test_id_bytes).hexdigest()

    summary = {
        "dataset": "CH-SIMS-v2.0",
        "source_csv": "meta.csv",
        "source_csv_sha256": source_info["source_csv_sha256"],
        "manifest_row_count": len(records),
        "official_mode_split_counts": split_counts,
        "invalid_label_rows": invalid_label_rows,
        "invalid_test_label_rows": sum(
            record["mode"] == "test" and record["label_status"] == "unresolved"
            for record in records
        ),
        "source_video_groups_overlapping_splits": overlapping_source_videos,
        "normalized_transcript_groups_repeated_across_splits": duplicate_text_groups,
        "normalized_transcript_groups_with_cross_split_mapped_label_conflicts": (
            None if invalid_label_rows else mapped_conflicting_text_groups
        ),
        "normalized_transcript_cross_split_mapped_conflict_check_complete": not invalid_label_rows,
        "normalized_transcript_groups_with_cross_split_raw_score_disagreements": raw_score_disagreement_groups,
        "test_ids_count": len(test_ids),
        "test_ids_sha256": test_ids_sha256,
        "speaker_leakage_status": "unknown; no speaker ID column is available in meta.csv",
        "test_label_policy": "test rows are excluded from training, tuning, and scoring; unresolved labels are null in opted-in JSONL",
        "scoring_policy": "never treat null or unresolved labels as correct, incorrect, or silently ignored",
        "text_normalization": "Unicode NFKC, casefold, collapse whitespace, strip",
    }
    return summary, test_ids


def prepare_manifest(
    meta_csv: Path,
    output_dir: Path,
    raw_dir_name: str = "Raw",
    *,
    allow_unresolved_test_labels: bool = False,
) -> dict[str, Any]:
    """Write preflight and train/valid outputs; optionally keep null-label tests."""
    records, normalized_texts, raw_label_values, invalid_label_rows, source_info = _read_records(
        meta_csv, raw_dir_name
    )
    split_rank = {split: index for index, split in enumerate(SPLIT_ORDER)}
    ordered_triples = sorted(
        zip(records, normalized_texts, raw_label_values),
        key=lambda triple: (split_rank[triple[0]["mode"]], triple[0]["ID"]),
    )
    ordered_records = [triple[0] for triple in ordered_triples]
    ordered_texts = [triple[1] for triple in ordered_triples]
    ordered_raw_labels = [triple[2] for triple in ordered_triples]
    summary, test_ids = _audit(
        ordered_records,
        ordered_texts,
        ordered_raw_labels,
        invalid_label_rows,
        source_info,
    )

    manifest_path = output_dir / "simsv2_manifest.jsonl"
    if invalid_label_rows and not allow_unresolved_test_labels and manifest_path.exists():
        raise ManifestError(
            "refusing to leave a stale full manifest beside a blocked preflight; use a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    train_valid_records = [
        record for record in ordered_records if record["mode"] in ("train", "valid")
    ]
    train_valid_path = output_dir / "simsv2_train_valid_manifest.csv"
    with train_valid_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(train_valid_records)

    manifest_generated = not invalid_label_rows or allow_unresolved_test_labels
    if manifest_generated:
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in ordered_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
    summary["train_valid_manifest_generated"] = True
    summary["manifest_generated"] = manifest_generated
    summary["allow_unresolved_test_labels"] = allow_unresolved_test_labels
    if not invalid_label_rows:
        summary["manifest_status"] = "complete"
    elif allow_unresolved_test_labels:
        summary["manifest_status"] = "contains_unresolved_test_labels"
    else:
        summary["manifest_status"] = "blocked_unresolved_test_labels"

    test_ids_path = output_dir / "simsv2_test_ids.txt"
    test_ids_bytes = ("\n".join(test_ids) + "\n").encode("utf-8")
    test_ids_path.write_bytes(test_ids_bytes)
    (output_dir / "simsv2_test_ids.sha256").write_text(
        "{}  simsv2_test_ids.txt\n".format(summary["test_ids_sha256"]),
        encoding="ascii",
        newline="\n",
    )
    (output_dir / "simsv2_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-csv", required=True, type=Path, help="Path to CH-SIMS v2 meta.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated audit outputs")
    parser.add_argument(
        "--raw-dir-name",
        default="Raw",
        help="Relative raw video directory prefix (default: Raw)",
    )
    parser.add_argument(
        "--allow-unresolved-test-labels",
        action="store_true",
        help="Write full JSONL with test Label=null for unmappable labels; never permits train/valid null labels",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = prepare_manifest(
            args.meta_csv,
            args.output_dir,
            args.raw_dir_name,
            allow_unresolved_test_labels=args.allow_unresolved_test_labels,
        )
    except (ManifestError, OSError, UnicodeError) as exc:
        print("manifest preparation failed: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["manifest_generated"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
