"""Prepare a separate author-cohort manifest for CMU-MOSEI.

This standard-library audit preserves the author ``mmsa_label.csv`` IDs, row
order, and ``mode`` splits. It emits author test IDs with their own frozen hash
and a train/valid-only transcript CSV for encoder preparation. It does not
score the test split.

A separate provisional crosswalk audit pairs rows within equal-size source
video groups by original author CSV order and competition numeric segment
order. This hypothesis is not used to join or rewrite author records and does
not prove media identity. Literal-ID overlap statistics are also audit-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
# This portable copy is intended to live at <project>/scripts/; parents[1] is
# then the project root containing the shared jevaffect package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from jevaffect.protocol import ProtocolError, map_mosei_sentiment  # noqa: E402
from prepare_mosei_manifest import parse_filename  # noqa: E402

SPLITS = ("train", "valid", "test")
AUTHOR_REQUIRED_COLUMNS = ("video_id", "clip_id", "text", "label", "annotation", "mode")
COMPETITION_REQUIRED_COLUMNS = ("FileName", "sentiment_score")
ENCODER_FIELDS = (
    "ID",
    "video_id",
    "clip_id",
    "mode",
    "text",
    "transcript_sha256",
    "author_row_number",
    "source_csv_sha256",
)


class AuthorManifestError(ValueError):
    """Raised when author cohort rows or crosswalk inputs are malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_csv_rows(path: Path, required: tuple[str, ...], name: str):
    if not path.is_file():
        raise AuthorManifestError("{} does not exist: {}".format(name, path))
    source_hash = sha256_file(path)
    rows = []
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError:
        raise
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AuthorManifestError("{} is empty".format(name))
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise AuthorManifestError("{} contains duplicate headers".format(name))
        missing = sorted(set(required) - set(reader.fieldnames))
        if missing:
            raise AuthorManifestError(
                "{} is missing columns: {}".format(name, ", ".join(missing))
            )
        for row_number, row in enumerate(reader, start=1):
            if row is None or all(value is None for value in row.values()):
                continue
            if None in row:
                raise AuthorManifestError(
                    "{} record {} has more values than its header".format(name, row_number)
                )
            rows.append((row_number, row))
    return rows, source_hash


def _nonempty_unpadded(value: Any, field: str, row_number: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise AuthorManifestError(
            "author row {} has invalid {}".format(row_number, field)
        )
    return value


def _score_and_class(raw_score: Any, context: str) -> tuple[float, str]:
    try:
        numeric = float(raw_score)
        if not math.isfinite(numeric):
            raise ValueError
        class_name = map_mosei_sentiment(numeric)
    except (TypeError, ValueError, ProtocolError) as exc:
        raise AuthorManifestError("{} has an unmappable sentiment label".format(context)) from exc
    return numeric, class_name


def _build_author_records(author_csv: Path):
    input_rows, source_hash = _read_csv_rows(
        author_csv, AUTHOR_REQUIRED_COLUMNS, "author label CSV"
    )
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_split_map: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()

    for row_number, row in input_rows:
        video_id = _nonempty_unpadded(row.get("video_id"), "video_id", row_number)
        clip_id = _nonempty_unpadded(row.get("clip_id"), "clip_id", row_number)
        text = row.get("text")
        if text is None:
            raise AuthorManifestError("author row {} has no transcript".format(row_number))
        mode = _nonempty_unpadded(row.get("mode"), "mode", row_number)
        if mode not in SPLITS:
            raise AuthorManifestError(
                "author row {} has unknown mode {!r}".format(row_number, mode)
            )
        raw_label = _nonempty_unpadded(row.get("label"), "label", row_number)
        annotation = row.get("annotation") or ""

        identifier = "{}_{}".format(video_id, clip_id)
        try:
            parsed_video, parsed_segment, source_type = parse_filename(identifier)
        except ValueError as exc:
            raise AuthorManifestError(
                "author row {} has an invalid video_id/clip_id ID: {}".format(row_number, exc)
            ) from exc
        if parsed_video != video_id or parsed_segment != clip_id:
            raise AuthorManifestError(
                "author row {} ID parsing changed source fields".format(row_number)
            )
        if identifier in seen_ids:
            raise AuthorManifestError("duplicate author ID: {}".format(identifier))
        seen_ids.add(identifier)

        numeric_label, class_name = _score_and_class(
            raw_label, "author row {} label".format(row_number)
        )
        transcript_hash = _sha256_text(text)
        record = {
            "ID": identifier,
            "author_id": identifier,
            "dataset": "CMU-MOSEI",
            "cohort": "mmsa_author",
            "split": mode,
            "mode": mode,
            "language": "en",
            "video_id": video_id,
            "clip_id": clip_id,
            "source_video": parsed_video,
            "source_video_id_type": source_type,
            "segment_id": parsed_segment,
            "author_row_number": row_number,
            "text": text,
            "transcript_sha256": transcript_hash,
            "raw_label": raw_label,
            "label_numeric": numeric_label,
            "class": class_name,
            "annotation": annotation,
            "label_status": "mapped_by_project_adopted_mosei_mapping",
            "source_csv": author_csv.name,
            "source_csv_sha256": source_hash,
        }
        records.append(record)
        source_split_map[parsed_video].add(mode)
        split_counts[mode] += 1

    counts = {split: split_counts.get(split, 0) for split in SPLITS}
    cross_split_source_groups = sum(len(splits) > 1 for splits in source_split_map.values())
    return records, source_hash, counts, cross_split_source_groups


def _read_competition_rows(csv_dir: Path):
    all_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    for split in SPLITS:
        path = csv_dir / "label_file_{}.csv".format(split)
        input_rows, source_hash = _read_csv_rows(
            path, COMPETITION_REQUIRED_COLUMNS, "competition {} CSV".format(split)
        )
        split_counts[split] = len(input_rows)
        split_hashes[split] = source_hash
        for row_number, row in input_rows:
            file_name = _nonempty_unpadded(
                row.get("FileName"), "FileName", row_number
            )
            try:
                source_video, segment_id, source_type = parse_filename(file_name)
                score, class_name = _score_and_class(
                    row.get("sentiment_score"),
                    "competition {} row {}".format(split, row_number),
                )
                segment_order = int(segment_id)
            except (ValueError, ProtocolError, AuthorManifestError) as exc:
                raise AuthorManifestError(
                    "competition {} row {} is invalid: {}".format(split, row_number, exc)
                ) from exc
            if file_name in seen_ids:
                raise AuthorManifestError("duplicate competition ID: {}".format(file_name))
            seen_ids.add(file_name)
            all_rows.append(
                {
                    "ID": file_name,
                    "source_video": source_video,
                    "segment_id": segment_id,
                    "segment_order": segment_order,
                    "source_video_id_type": source_type,
                    "split": split,
                    "row_number": row_number,
                    "raw_label": row["sentiment_score"],
                    "label_numeric": score,
                    "class": class_name,
                    "source_csv": path.name,
                    "source_csv_sha256": source_hash,
                }
            )
    return all_rows, split_counts, split_hashes


def _literal_id_overlap_audit(
    author_rows: list[dict[str, Any]], competition_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    author_by_id = {row["ID"]: row for row in author_rows}
    comp_by_id = {row["ID"]: row for row in competition_rows}
    common_ids = set(author_by_id) & set(comp_by_id)
    raw_mismatches = 0
    class_mismatches = 0
    split_mismatches = 0
    for identifier in common_ids:
        author_row = author_by_id[identifier]
        comp_row = comp_by_id[identifier]
        if author_row["label_numeric"] != comp_row["label_numeric"]:
            raw_mismatches += 1
        if author_row["class"] != comp_row["class"]:
            class_mismatches += 1
        if author_row["split"] != comp_row["split"]:
            split_mismatches += 1
    return {
        "literal_id_intersection_count": len(common_ids),
        "literal_id_raw_label_mismatches": raw_mismatches,
        "literal_id_class_mismatches": class_mismatches,
        "literal_id_split_mismatches": split_mismatches,
        "literal_id_join_used": False,
        "note": "Audit-only comparison; literal IDs do not define the provisional crosswalk or author cohort.",
    }


def _provisional_order_crosswalk(
    author_rows: list[dict[str, Any]], competition_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    author_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    comp_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in author_rows:
        author_by_video[row["source_video"]].append(row)
    for row in competition_rows:
        comp_by_video[row["source_video"]].append(row)

    author_videos = set(author_by_video)
    comp_videos = set(comp_by_video)
    shared_videos = author_videos & comp_videos
    equal_group_count = 0
    unequal_group_count = 0
    author_only_group_count = len(author_videos - comp_videos)
    comp_only_group_count = len(comp_videos - author_videos)
    ambiguous_numeric_order_group_count = 0
    paired_rows: list[dict[str, Any]] = []
    unpaired_author_rows = 0
    unpaired_comp_rows = 0
    raw_label_mismatches = 0
    class_mismatches = 0
    split_mismatches = 0

    for video_id in sorted(author_videos | comp_videos):
        author_group = author_by_video.get(video_id, [])
        comp_group = comp_by_video.get(video_id, [])
        if not author_group or not comp_group:
            unpaired_author_rows += len(author_group)
            unpaired_comp_rows += len(comp_group)
            continue
        if len(author_group) != len(comp_group):
            unequal_group_count += 1
            unpaired_author_rows += len(author_group)
            unpaired_comp_rows += len(comp_group)
            continue

        comp_orders = [row["segment_order"] for row in comp_group]
        if len(comp_orders) != len(set(comp_orders)):
            ambiguous_numeric_order_group_count += 1
            unpaired_author_rows += len(author_group)
            unpaired_comp_rows += len(comp_group)
            continue

        equal_group_count += 1
        ordered_comp = sorted(
            comp_group,
            key=lambda row: (row["segment_order"], row["row_number"]),
        )
        # author_group remains in literal source-CSV row order. Do not sort or
        # join these rows by clip_id or literal ID.
        for rank, (author_row, comp_row) in enumerate(
            zip(author_group, ordered_comp), start=1
        ):
            raw_equal = author_row["label_numeric"] == comp_row["label_numeric"]
            class_equal = author_row["class"] == comp_row["class"]
            split_equal = author_row["split"] == comp_row["split"]
            raw_label_mismatches += not raw_equal
            class_mismatches += not class_equal
            split_mismatches += not split_equal
            paired_rows.append(
                {
                    "crosswalk_status": "provisional_order_hypothesis_only",
                    "used_for_author_manifest": False,
                    "source_video": video_id,
                    "pair_rank_within_video": rank,
                    "author_id": author_row["ID"],
                    "competition_id": comp_row["ID"],
                    "author_row_number": author_row["author_row_number"],
                    "competition_split_row_number": comp_row["row_number"],
                    "competition_numeric_segment_order": comp_row["segment_order"],
                    "author_mode": author_row["split"],
                    "competition_split": comp_row["split"],
                    "author_raw_label": author_row["raw_label"],
                    "competition_raw_label": comp_row["raw_label"],
                    "author_class": author_row["class"],
                    "competition_class": comp_row["class"],
                    "raw_label_equal": raw_equal,
                    "class_equal": class_equal,
                    "split_equal": split_equal,
                    "pairing_basis": "author source CSV order within video paired with numeric competition segment order",
                }
            )

    metrics = {
        "status": "provisional_order_hypothesis_only",
        "author_order": "source_csv_order_within_source_video",
        "competition_order": "numeric_segment_suffix_within_source_video",
        "equal_count_source_video_groups": equal_group_count,
        "unequal_count_source_video_groups": unequal_group_count,
        "author_only_source_video_groups": author_only_group_count,
        "competition_only_source_video_groups": comp_only_group_count,
        "ambiguous_competition_numeric_order_groups": ambiguous_numeric_order_group_count,
        "paired_rows": len(paired_rows),
        "unpaired_author_rows": unpaired_author_rows,
        "unpaired_competition_rows": unpaired_comp_rows,
        "raw_label_mismatches": int(raw_label_mismatches),
        "class_mismatches": int(class_mismatches),
        "split_mismatches": int(split_mismatches),
        "author_literal_ids_modified": False,
        "media_identity_proven": False,
        "used_to_join_author_manifest": False,
        "model_scoring_performed": False,
    }
    return paired_rows, metrics


def _read_order_hypothesis(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise AuthorManifestError("order hypothesis JSON does not exist: {}".format(path))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise AuthorManifestError("order hypothesis file is not valid JSON") from exc
    if not isinstance(data, list):
        raise AuthorManifestError("order hypothesis JSON must be a list")
    selected = next(
        (
            row
            for row in data
            if isinstance(row, dict)
            and row.get("author_order") == "file_order"
            and row.get("competition_order") == "numeric"
        ),
        None,
    )
    if selected is None:
        raise AuthorManifestError(
            "order hypothesis JSON lacks author file_order / competition numeric case"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "declared_author_order": selected["author_order"],
        "declared_competition_order": selected["competition_order"],
        "declared_pairs": selected.get("pairs"),
        "declared_mismatches": selected.get("mismatches"),
        "note": "Reference evidence only; this script recomputes the pairing audit and does not adopt the crosswalk.",
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _write_frozen_ids(output_dir: Path, split_rows: dict[str, list[dict[str, Any]]]):
    details = {}
    for split in SPLITS:
        ids = sorted(row["ID"] for row in split_rows[split])
        payload = ("\n".join(ids) + ("\n" if ids else "")).encode("utf-8")
        base = "mosei_author_{}_ids".format(split)
        (output_dir / (base + ".txt")).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        (output_dir / (base + ".sha256")).write_text(
            "{}  {}.txt\n".format(digest, base), encoding="ascii", newline="\n"
        )
        details[split] = {"count": len(ids), "sha256": digest}
    return details


def _write_encoder_transcripts(output_dir: Path, train_valid_rows: list[dict[str, Any]]) -> str:
    path = output_dir / "mosei_author_train_valid_encoder.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENCODER_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in train_valid_rows:
            writer.writerow({field: row[field] for field in ENCODER_FIELDS})
    return sha256_file(path)


def prepare_mosei_author_manifest(
    author_csv: Path,
    competition_csv_dir: Path,
    output_dir: Path,
    order_hypothesis_json: Path | None = None,
) -> dict[str, Any]:
    """Write author manifests and isolated, explicitly provisional crosswalk audit."""
    author_rows, author_csv_sha, split_counts, cross_split_sources = _build_author_records(
        author_csv
    )
    competition_rows, competition_counts, competition_hashes = _read_competition_rows(
        competition_csv_dir
    )

    split_rows = {
        split: [row for row in author_rows if row["split"] == split]
        for split in SPLITS
    }
    # Preserve original author file order in each author split manifest.
    crosswalk_rows, crosswalk_summary = _provisional_order_crosswalk(
        author_rows, competition_rows
    )
    literal_overlap = _literal_id_overlap_audit(author_rows, competition_rows)
    reference_comparison = _read_order_hypothesis(order_hypothesis_json)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(
            output_dir / "mosei_author_{}_manifest.jsonl".format(split),
            split_rows[split],
        )
    frozen_ids = _write_frozen_ids(output_dir, split_rows)
    train_valid_rows = [row for row in author_rows if row["split"] in ("train", "valid")]
    encoder_sha = _write_encoder_transcripts(output_dir, train_valid_rows)
    _write_jsonl(output_dir / "mosei_provisional_order_crosswalk.jsonl", crosswalk_rows)

    summary = {
        "dataset": "CMU-MOSEI",
        "cohort": "mmsa_author",
        "author_source_csv": str(author_csv),
        "author_source_csv_sha256": author_csv_sha,
        "competition_csv_dir": str(competition_csv_dir),
        "competition_split_counts": competition_counts,
        "competition_split_csv_sha256": competition_hashes,
        "split_counts": split_counts,
        "author_total_rows": len(author_rows),
        "duplicate_author_ids": 0,
        "source_video_groups_cross_author_splits": cross_split_sources,
        "frozen_author_ids": frozen_ids,
        "train_valid_encoder_csv": "mosei_author_train_valid_encoder.csv",
        "train_valid_encoder_rows": len(train_valid_rows),
        "train_valid_encoder_csv_sha256": encoder_sha,
        "encoder_csv_contains_labels": False,
        "model_scoring_performed": False,
        "test_class_distribution_emitted": False,
        "literal_id_overlap_audit": literal_overlap,
        "provisional_crosswalk_file": "mosei_provisional_order_crosswalk.jsonl",
        "provisional_crosswalk": crosswalk_summary,
        "order_hypothesis_reference": reference_comparison,
        "crosswalk_warning": (
            "The order hypothesis is provisional, is excluded from the author manifests, "
            "and does not prove media identity. Do not use it as an arbitrary ID join."
        ),
        "label_mapping_note": "Project-adopted application of the competition's one possible MOSEI mapping.",
    }
    (output_dir / "mosei_author_cohort_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-csv", required=True, type=Path)
    parser.add_argument("--competition-csv-dir", required=True, type=Path)
    parser.add_argument("--order-hypothesis-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = prepare_mosei_author_manifest(
            args.author_csv,
            args.competition_csv_dir,
            args.output_dir,
            args.order_hypothesis_json,
        )
    except (AuthorManifestError, OSError, UnicodeError, ProtocolError) as exc:
        print("Author MOSEI manifest preparation failed: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
