#!/usr/bin/env python3
"""Join verified per-record raw NPZ features to SIMS v2 train/valid rows.

This writes the small 382-row raw-prefix pilot only. It does not read or create
test cache data, run encoders, train models, or claim a native MMSA benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np


TARGET_SHAPES = {
    "text": (64, 768),
    "audio": (400, 768),
    "vision": (55, 384),
}
ARRAY_KEYS = {
    "text": "text",
    "text_mask": "text_mask",
    "text_token_ids": "text_token_ids",
    "audio": "audio",
    "audio_mask": "audio_mask",
    "audio_times": "audio_times",
    "vision": "vision",
    "vision_mask": "vision_mask",
    "vision_times": "vision_times",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CacheBuildError(ValueError):
    """Input manifests, features, or trainer cache output are inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def transcript_sha256(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheBuildError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CacheBuildError(f"JSON object expected at {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CacheBuildError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise CacheBuildError(f"JSONL row must be an object at {path}:{line_number}")
                rows.append(row)
    except OSError as exc:
        raise CacheBuildError(f"cannot read JSONL {path}: {exc}") from exc
    if not rows:
        raise CacheBuildError(f"JSONL has no records: {path}")
    return rows


def _exact_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise CacheBuildError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _resolve_under(root: Path, relative: str, name: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise CacheBuildError(f"{name} must be a safe relative POSIX path")
    path = (root / Path(*candidate.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise CacheBuildError(f"{name} escapes its bundle") from exc
    return path


def import_project_protocol(project_root: Path) -> tuple[tuple[str, ...], Callable[[float], str], Path]:
    """Import the project's shared five-class mapping from its source tree."""
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from jevaffect.protocol import CLASSES, map_simsv2_label
    except ImportError as exc:
        raise CacheBuildError(f"cannot import jevaffect.protocol from {project_root}") from exc
    protocol_path = project_root / "jevaffect" / "protocol.py"
    if not protocol_path.is_file():
        raise CacheBuildError(f"shared protocol source is missing: {protocol_path}")
    return tuple(CLASSES), map_simsv2_label, protocol_path


def read_official_rows(
    csv_path: Path,
    manifest_rows: list[dict[str, Any]],
    class_names: tuple[str, ...],
    label_mapper: Callable[[float], str],
) -> tuple[dict[str, dict[str, Any]], str]:
    import csv

    csv_sha = sha256_file(csv_path)
    lookup: dict[str, dict[str, Any]] = {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for source in csv.DictReader(stream):
                split = (source.get("mode") or "").strip().lower()
                if split not in ("train", "valid"):
                    continue
                sample_id = f"{source.get('video_id', '')}/{source.get('clip_id', '')}"
                if sample_id in lookup:
                    raise CacheBuildError(f"duplicate official train/valid ID: {sample_id}")
                raw_label = source.get("label", "").strip()
                try:
                    mapped = label_mapper(float(raw_label))
                except (ValueError, TypeError) as exc:
                    raise CacheBuildError(f"invalid official label {raw_label!r} for {sample_id}") from exc
                if mapped not in class_names:
                    raise CacheBuildError(f"shared protocol returned unknown class {mapped!r}")
                lookup[sample_id] = {
                    "ID": sample_id,
                    "split": split,
                    "video_id": source["video_id"],
                    "clip_id": source["clip_id"],
                    "text": source.get("text", ""),
                    "raw_label": raw_label,
                    "mapped_label": mapped,
                    "class_index": class_names.index(mapped),
                }
    except OSError as exc:
        raise CacheBuildError(f"cannot read official metadata CSV {csv_path}: {exc}") from exc

    manifest_ids = [row.get("ID") for row in manifest_rows]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise CacheBuildError("manifest contains duplicate IDs")
    if set(manifest_ids) - set(lookup):
        raise CacheBuildError(f"manifest IDs absent from official train/valid CSV: {sorted(set(manifest_ids)-set(lookup))[:5]}")
    if set(lookup) < set(manifest_ids):
        raise CacheBuildError("manifest ID check failed")
    manifest_sources = {row.get("source_csv_sha256") for row in manifest_rows}
    if manifest_sources != {csv_sha}:
        raise CacheBuildError("manifest source_csv_sha256 does not match the official CSV file")

    for row in manifest_rows:
        sample_id = row["ID"]
        official = lookup[sample_id]
        if row.get("split") != official["split"]:
            raise CacheBuildError(f"manifest/official split mismatch for {sample_id}")
        if row.get("source_video") != official["video_id"]:
            raise CacheBuildError(f"manifest source_video mismatch for {sample_id}")
        if row.get("text") != official["text"]:
            raise CacheBuildError(f"manifest transcript mismatch for {sample_id}")
        try:
            manifest_raw = float(row["raw_label"])
            official_raw = float(official["raw_label"])
        except (ValueError, TypeError, KeyError) as exc:
            raise CacheBuildError(f"missing or invalid manifest/official raw label for {sample_id}") from exc
        if not math.isclose(manifest_raw, official_raw, rel_tol=0.0, abs_tol=1e-6):
            raise CacheBuildError(f"manifest/official raw-label mismatch for {sample_id}")
        mapped = label_mapper(manifest_raw)
        if mapped != official["mapped_label"]:
            raise CacheBuildError(f"manifest/official mapped-label mismatch for {sample_id}")
        if row.get("label_status") not in (None, "known", "resolved"):
            raise CacheBuildError(f"unresolved label status in train/valid for {sample_id}")
        row["mapped_label"] = mapped
        row["class_index"] = class_names.index(mapped)
    return lookup, csv_sha


def read_and_validate_manifest(
    path: Path,
    official_csv: Path,
    class_names: tuple[str, ...],
    label_mapper: Callable[[float], str],
    expected_counts: dict[str, int] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl(path)
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    by_split: dict[str, set[str]] = {"train": set(), "valid": set()}
    for row in rows:
        sample_id = row.get("ID")
        if not isinstance(sample_id, str) or not sample_id or sample_id.isspace():
            raise CacheBuildError("manifest ID must be a non-empty exact canonical string")
        if sample_id in seen:
            raise CacheBuildError(f"duplicate manifest ID: {sample_id}")
        seen.add(sample_id)
        split = row.get("split")
        if split not in ("train", "valid"):
            raise CacheBuildError(f"manifest split must be train or valid; got {split!r}")
        counts[split] += 1
        by_split[split].add(sample_id)
        if row.get("dataset") not in ("CH-SIMS-v2.0", "CH-SIMS v2.0"):
            raise CacheBuildError(f"unexpected dataset for {sample_id}: {row.get('dataset')!r}")
        if row.get("language") != "zh":
            raise CacheBuildError(f"pilot language must be zh for {sample_id}")
        if row.get("transcript_condition") != "author_provided":
            raise CacheBuildError(f"pilot transcript must be author-provided for {sample_id}")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise CacheBuildError(f"missing provided transcript for {sample_id}")
        row["sha256"] = _exact_hash(row.get("sha256"), f"source video hash for {sample_id}")
        if not isinstance(row.get("source_video"), str) or not row["source_video"]:
            raise CacheBuildError(f"missing source_video for {sample_id}")
    if by_split["train"] & by_split["valid"]:
        raise CacheBuildError("train and valid manifest IDs overlap")
    if set(counts) != {"train", "valid"}:
        raise CacheBuildError("manifest must contain both non-empty train and valid splits")
    if expected_counts is not None and dict(counts) != expected_counts:
        raise CacheBuildError(f"manifest split counts {dict(counts)} differ from expected {expected_counts}")

    official, csv_sha = read_official_rows(pathlib_resolve(official_csv), rows, class_names, label_mapper)
    return rows, {
        "manifest_sha256": sha256_file(path),
        "official_csv_sha256": csv_sha,
        "split_counts": dict(counts),
        "class_order": list(class_names),
        "official_train_valid_rows": len(official),
        "manifest_rows_joined": len(rows),
    }


def pathlib_resolve(path: Path) -> Path:
    return Path(path).resolve()


def read_raw_bundle(bundle_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    bundle = bundle_dir.resolve()
    meta_path = bundle / "meta.json"
    index_path = bundle / "index.jsonl"
    meta = _read_json(meta_path)
    if meta.get("cache_kind") != "raw_video_feature_records" or meta.get("schema_version") != 1:
        raise CacheBuildError("unsupported raw feature bundle type/schema")
    if meta.get("labels_included") is not False or meta.get("canonical_ids_and_splits_joined") is not False:
        raise CacheBuildError("raw bundle must not contain prejoined labels or canonical split contract")
    if meta.get("fixture_results") is not False or meta.get("all_records_are_fixtures") is not False:
        raise CacheBuildError("fixture/mixed raw feature output is not accepted")
    if meta.get("model_results") is not True:
        raise CacheBuildError("raw bundle is not marked as verified model results")
    provenance = meta.get("encoder_provenance")
    if not isinstance(provenance, dict) or provenance.get("local_files_only") is not True:
        raise CacheBuildError("raw bundle lacks verified local encoder provenance")
    dims = provenance.get("dimensions", {})
    expected_dims = {name: values[1] for name, values in TARGET_SHAPES.items()}
    if dims != expected_dims:
        raise CacheBuildError(f"unexpected encoder dimensions: {dims!r}")
    files = meta.get("files")
    if not isinstance(files, dict) or files.get("index.jsonl") != sha256_file(index_path):
        raise CacheBuildError("raw index.jsonl hash does not match raw bundle meta.json")
    sidecar = bundle / "meta.sha256"
    if sidecar.is_file():
        parts = sidecar.read_text(encoding="ascii").split()
        if not parts or parts[0] != sha256_file(meta_path):
            raise CacheBuildError("raw meta.sha256 sidecar mismatch")
    index_rows = _read_jsonl(index_path)
    if meta.get("sample_count") != len(index_rows):
        raise CacheBuildError("raw sample_count disagrees with index rows")
    listed = files.get("records")
    actual_paths = [row.get("artifact") for row in index_rows]
    if not isinstance(listed, list) or listed != actual_paths:
        raise CacheBuildError("raw bundle records list disagrees with index.jsonl")
    seen: set[str] = set()
    for index_row in index_rows:
        sample_id = index_row.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise CacheBuildError("raw bundle IDs must be unique non-empty exact strings")
        seen.add(sample_id)
        artifact = _resolve_under(bundle, index_row.get("artifact", ""), "raw artifact path")
        expected_sha = _exact_hash(index_row.get("artifact_sha256"), f"raw artifact hash for {sample_id}")
        if not artifact.is_file() or sha256_file(artifact) != expected_sha:
            raise CacheBuildError(f"raw feature artifact SHA-256 mismatch for {sample_id}")
        metadata = index_row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("id") != sample_id:
            raise CacheBuildError(f"raw record metadata ID mismatch for {sample_id}")
        if metadata.get("fixture") is not False or metadata.get("model_results") is not True:
            raise CacheBuildError(f"raw record {sample_id} is a fixture or not model-verified")
        if metadata.get("encoder_provenance") != provenance:
            raise CacheBuildError(f"encoder provenance differs within raw bundle at {sample_id}")
    raw_input_sha = meta.get("input_manifest_sha256")
    if raw_input_sha is not None:
        _exact_hash(raw_input_sha, "raw bundle input_manifest_sha256")
    bundle_summary = {
        "raw_bundle_meta_sha256": sha256_file(meta_path),
        "raw_bundle_index_sha256": sha256_file(index_path),
        "raw_bundle_sample_count": len(index_rows),
        "raw_bundle_encoder_provenance": provenance,
        "raw_bundle_input_manifest_sha256": meta.get("input_manifest_sha256"),
    }
    return meta, index_rows, bundle_summary


def _load_record_arrays(bundle: Path, row: dict[str, Any]) -> dict[str, np.ndarray]:
    artifact = _resolve_under(bundle.resolve(), row["artifact"], "raw artifact path")
    if sha256_file(artifact) != row["artifact_sha256"]:
        raise CacheBuildError(f"raw feature artifact changed after preflight: {row['id']}")
    try:
        with np.load(artifact, allow_pickle=False) as archive:
            missing = set(ARRAY_KEYS.values()) - set(archive.files)
            if missing:
                raise CacheBuildError(f"raw NPZ missing arrays {sorted(missing)} for {row['id']}")
            arrays = {name: np.asarray(archive[key]) for name, key in ARRAY_KEYS.items()}
    except (OSError, ValueError, EOFError) as exc:
        raise CacheBuildError(f"cannot load verified NPZ for {row['id']}: {exc}") from exc
    return arrays


def validate_record_arrays(
    arrays: dict[str, np.ndarray],
    row: dict[str, Any],
) -> dict[str, Any]:
    lengths: dict[str, int] = {}
    for modality, (max_steps, dim) in TARGET_SHAPES.items():
        feature = arrays[modality]
        mask = arrays[modality + "_mask"]
        if feature.ndim != 2 or feature.shape[1] != dim or feature.dtype.kind != "f":
            raise CacheBuildError(f"{row['id']} {modality} must be floating [T,{dim}]")
        if not feature.shape[0] or not np.isfinite(feature).all():
            raise CacheBuildError(f"{row['id']} {modality} is empty or non-finite")
        if mask.dtype.kind != "b" or mask.shape != (feature.shape[0],):
            raise CacheBuildError(f"{row['id']} {modality}_mask must be bool [T]")
        if not mask.any() or not mask.all():
            raise CacheBuildError(
                f"{row['id']} {modality} has a missing/padded position in its raw sequence; "
                "the initial ALMT pilot requires every modality to be fully present"
            )
        lengths[modality] = int(mask.sum())
        if modality == "text":
            token_ids = arrays["text_token_ids"]
            if token_ids.ndim != 1 or token_ids.shape[0] != feature.shape[0] or token_ids.dtype.kind not in "iu":
                raise CacheBuildError(f"{row['id']} text_token_ids must be integer [text T]")
        else:
            times = arrays[modality + "_times"]
            if times.ndim != 1 or times.shape[0] != feature.shape[0] or times.dtype.kind != "f":
                raise CacheBuildError(f"{row['id']} {modality}_times must be floating [T]")
            if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                raise CacheBuildError(f"{row['id']} {modality}_times must be finite and nondecreasing")
        if lengths[modality] < 1:
            raise CacheBuildError(f"{row['id']} {modality} has no valid frames/tokens")
    return {
        "raw_lengths": lengths,
        "kept_lengths": {name: min(length, TARGET_SHAPES[name][0]) for name, length in lengths.items()},
        "truncated": {name: max(0, length - TARGET_SHAPES[name][0]) for name, length in lengths.items()},
    }


def preflight_join(
    manifest_rows: list[dict[str, Any]],
    raw_meta: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    bundle_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_id = {row["id"]: row for row in raw_rows}
    manifest_by_id = {row["ID"]: row for row in manifest_rows}
    missing = set(manifest_by_id) - set(raw_by_id)
    unexpected = set(raw_by_id) - set(manifest_by_id)
    if missing or unexpected:
        raise CacheBuildError(
            "raw and official pilot IDs must match exactly; "
            f"missing_raw={sorted(missing)[:5]}, unexpected_raw={sorted(unexpected)[:5]}"
        )
    record_audits: list[dict[str, Any]] = []
    preprocessing_signatures: dict[str, dict[str, Any]] = {}
    truncated_counts = Counter()
    dropped_totals = Counter()
    max_raw_lengths = Counter()
    for sample_id, manifest in manifest_by_id.items():
        raw = raw_by_id[sample_id]
        metadata = raw["metadata"]
        if manifest["sha256"] != metadata.get("source_video_sha256"):
            raise CacheBuildError(f"video SHA-256 mismatch for {sample_id}")
        if manifest["sha256"] != metadata.get("decode", {}).get("source_sha256"):
            raise CacheBuildError(f"decode video SHA-256 mismatch for {sample_id}")
        if manifest["text"] != metadata.get("transcript") or metadata.get("transcript_source") != "provided":
            raise CacheBuildError(f"provided transcript mismatch/source mismatch for {sample_id}")
        if metadata.get("prediction") is not None:
            raise CacheBuildError(f"raw feature stage unexpectedly contains a prediction for {sample_id}")
        modality_status = metadata.get("modalities")
        if not isinstance(modality_status, dict):
            raise CacheBuildError(f"raw modality status is missing for {sample_id}")
        for modality, (_, dim) in TARGET_SHAPES.items():
            status = modality_status.get(modality)
            if (not isinstance(status, dict) or status.get("valid") is not True
                    or status.get("status") != "present" or status.get("dim") != dim):
                raise CacheBuildError(f"{sample_id} lacks a complete {modality} modality")
        preprocessing = metadata.get("preprocessing")
        if not isinstance(preprocessing, dict):
            raise CacheBuildError(f"raw preprocessing config missing for {sample_id}")
        stable_preprocessing = {
            key: value for key, value in preprocessing.items() if key != "text_truncated"
        }
        preprocessing_signatures[canonical_json_sha256(stable_preprocessing)] = stable_preprocessing
        arrays = _load_record_arrays(bundle_dir, raw)
        shape_audit = validate_record_arrays(arrays, raw)
        row_audit = {
            "ID": sample_id,
            "split": manifest["split"],
            "source_video": manifest["source_video"],
            "source_video_path": manifest.get("video_path"),
            "source_video_sha256": manifest["sha256"],
            "raw_artifact": raw["artifact"],
            "raw_artifact_sha256": raw["artifact_sha256"],
            "transcript_sha256_nfkc_casefold_whitespace": transcript_sha256(manifest["text"]),
            "label": manifest["mapped_label"],
            "class_index": manifest["class_index"],
            "raw_label": manifest["raw_label"],
            "preprocessing": preprocessing,
            "text_truncated_by_encoder": bool(preprocessing.get("text_truncated", False)),
            **shape_audit,
        }
        record_audits.append(row_audit)
        for modality, count in shape_audit["truncated"].items():
            if count:
                truncated_counts[modality] += 1
                dropped_totals[modality] += count
            max_raw_lengths[modality] = max(max_raw_lengths[modality], shape_audit["raw_lengths"][modality])
    if len(preprocessing_signatures) != 1:
        raise CacheBuildError("preprocessing settings differ between raw records")
    if len(record_audits) != int(raw_meta["sample_count"]):
        raise CacheBuildError("joined manifest count differs from raw bundle sample count")
    return record_audits, {
        "preprocessing_sha256": next(iter(preprocessing_signatures)),
        "encoder_preprocessing": next(iter(preprocessing_signatures.values())),
        "sequence_truncation": {
            modality: {
                "target_steps": TARGET_SHAPES[modality][0],
                "observed_max_raw_steps": int(max_raw_lengths[modality]),
                "truncated_records": int(truncated_counts[modality]),
                "total_dropped_frames_or_tokens": int(dropped_totals[modality]),
                "policy": "keep the first target_steps in original order; no resampling/interpolation",
            }
            for modality in TARGET_SHAPES
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _pad_to(array: np.ndarray, target_steps: int, *, dtype: Any, fill: Any = 0) -> np.ndarray:
    result = np.zeros((target_steps, *array.shape[1:]), dtype=dtype)
    if fill != 0:
        result[...] = fill
    copied = min(target_steps, array.shape[0])
    if copied:
        result[:copied] = array[:copied]
    return result


def _write_split(
    stage: Path,
    split: str,
    manifests: list[dict[str, Any]],
    raw_by_id: dict[str, dict[str, Any]],
    bundle_dir: Path,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    split_dir = stage / "splits" / split
    split_dir.mkdir(parents=True)
    count = len(manifests)
    if not count:
        raise CacheBuildError(f"{split} split cannot be empty")
    ids = [row["ID"] for row in manifests]
    _write_json(split_dir / "ids.json", ids)
    _write_json(split_dir / "id_mapping.json", {"canonical_ids": ids, "source_ids": ids})
    labels = np.lib.format.open_memmap(split_dir / "labels.npy", mode="w+", dtype=np.int64, shape=(count,))
    raw_labels = np.lib.format.open_memmap(split_dir / "raw_labels.npy", mode="w+", dtype=np.float64, shape=(count,))
    label_valid = np.lib.format.open_memmap(split_dir / "label_valid.npy", mode="w+", dtype=np.bool_, shape=(count,))
    output_arrays: dict[str, np.memmap] = {}
    for modality, (steps, dim) in TARGET_SHAPES.items():
        output_arrays[modality] = np.lib.format.open_memmap(
            split_dir / f"{modality}.npy", mode="w+", dtype=np.float32, shape=(count, steps, dim)
        )
        output_arrays[modality + "_mask"] = np.lib.format.open_memmap(
            split_dir / f"{modality}_mask.npy", mode="w+", dtype=np.bool_, shape=(count, steps)
        )
        output_arrays[modality + "_lengths"] = np.lib.format.open_memmap(
            split_dir / f"{modality}_lengths.npy", mode="w+", dtype=np.int64, shape=(count,)
        )
    for modality in ("audio", "vision"):
        steps = TARGET_SHAPES[modality][0]
        output_arrays[modality + "_times"] = np.lib.format.open_memmap(
            split_dir / f"{modality}_times.npy", mode="w+", dtype=np.float64, shape=(count, steps)
        )
    row_audits: list[dict[str, Any]] = []
    for index, manifest in enumerate(manifests):
        raw = raw_by_id[manifest["ID"]]
        arrays = _load_record_arrays(bundle_dir, raw)
        shape_audit = validate_record_arrays(arrays, raw)
        labels[index] = int(manifest["class_index"])
        raw_labels[index] = float(manifest["raw_label"])
        label_valid[index] = True
        for modality, (target_steps, _) in TARGET_SHAPES.items():
            feature = arrays[modality]
            kept = min(target_steps, feature.shape[0])
            output_arrays[modality][index] = _pad_to(feature, target_steps, dtype=np.float32)
            mask = np.zeros(target_steps, dtype=np.bool_)
            mask[:kept] = arrays[modality + "_mask"][:kept]
            output_arrays[modality + "_mask"][index] = mask
            output_arrays[modality + "_lengths"][index] = kept
            if modality in ("audio", "vision"):
                times = arrays[modality + "_times"]
                padded_times = np.zeros(target_steps, dtype=np.float64)
                padded_times[:kept] = times[:kept]
                output_arrays[modality + "_times"][index] = padded_times
        row_audits.append({
            "ID": manifest["ID"],
            "split": split,
            "raw_label": manifest["raw_label"],
            "mapped_label": manifest["mapped_label"],
            "class_index": manifest["class_index"],
            "source_video_sha256": manifest["sha256"],
            "source_video_path": manifest.get("video_path"),
            "transcript_sha256_nfkc_casefold_whitespace": transcript_sha256(manifest["text"]),
            "raw_artifact": raw["artifact"],
            "raw_artifact_sha256": raw["artifact_sha256"],
            "raw_lengths": shape_audit["raw_lengths"],
            "kept_lengths": shape_audit["kept_lengths"],
            "truncated": shape_audit["truncated"],
        })
    for array in (labels, raw_labels, label_valid, *output_arrays.values()):
        array.flush()
    del labels, raw_labels, label_valid, output_arrays
    _write_json(split_dir / "raw_labels.json", [row["raw_label"] for row in manifests])
    _write_json(split_dir / "manifest_labels.json", [
        {"ID": row["ID"], "label": row["mapped_label"], "class_index": row["class_index"]}
        for row in manifests
    ])
    _write_jsonl(split_dir / "record_audit.jsonl", row_audits)
    return {"sample_count": count, "relative_path": f"splits/{split}"}


def verify_with_training_loader(cache_dir: Path, project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    try:
        from jevaffect.training import FeatureSplit, _cache_metadata, _move_features
    except ImportError as exc:
        raise CacheBuildError("could not import project trainer for CPU cache verification") from exc
    required = ("text", "audio", "vision")
    meta, fingerprint = _cache_metadata(cache_dir, required)
    check: dict[str, Any] = {"meta_fingerprint": fingerprint, "splits": {}}
    for split in ("train", "valid"):
        dataset = FeatureSplit(cache_dir, split, meta, modalities=required)
        if not dataset.ids:
            raise CacheBuildError(f"trainer loader found no {split} IDs")
        sample = dataset[0]
        moved = _move_features(sample["features"], "cpu")
        if sample["label"].device.type != "cpu" or any(
            moved[name].device.type != "cpu" for name in required
        ):
            raise CacheBuildError("trainer cache smoke check unexpectedly left CPU")
        expected_shapes = {name: (TARGET_SHAPES[name][0], TARGET_SHAPES[name][1]) for name in required}
        actual_shapes = {name: tuple(moved[name].shape) for name in required}
        if actual_shapes != expected_shapes:
            raise CacheBuildError(f"trainer loaded unexpected shapes: {actual_shapes}")
        check["splits"][split] = {
            "sample_count": len(dataset),
            "first_id": sample["id"],
            "first_label_index": int(sample["label"].item()),
            "first_shapes": {name: list(shape) for name, shape in actual_shapes.items()},
            "device": "cpu",
        }
    return check


def build_cache(
    *,
    project_root: Path,
    manifest_path: Path,
    official_csv: Path,
    raw_bundle: Path,
    output_dir: Path,
    expected_counts: dict[str, int] | None = None,
    cohort: str = "pilot",
    verify_training_loader_fn: Callable[[Path, Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if cohort not in ("pilot", "full"):
        raise CacheBuildError("cohort must be pilot or full")
    if cohort == "full" and expected_counts != {"train": 2722, "valid": 647}:
        raise CacheBuildError("full cohort requires exact official train/valid counts")
    target = output_dir.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {target}")
    project_root = project_root.resolve()
    class_names, label_mapper, protocol_path = import_project_protocol(project_root)
    manifest_rows, source_summary = read_and_validate_manifest(
        manifest_path.resolve(), official_csv.resolve(), class_names, label_mapper, expected_counts
    )
    raw_meta, raw_rows, raw_summary = read_raw_bundle(raw_bundle)
    record_audits, truncation_summary = preflight_join(manifest_rows, raw_meta, raw_rows, raw_bundle)
    raw_by_id = {row["id"]: row for row in raw_rows}
    split_rows = {
        split: [row for row in manifest_rows if row["split"] == split]
        for split in ("train", "valid")
    }
    bundle_preproc = truncation_summary["encoder_preprocessing"]
    source_descriptor = {
        "source_manifest_sha256": source_summary["manifest_sha256"],
        "official_csv_sha256": source_summary["official_csv_sha256"],
        "raw_bundle_meta_sha256": raw_summary["raw_bundle_meta_sha256"],
        "raw_bundle_index_sha256": raw_summary["raw_bundle_index_sha256"],
        "raw_bundle_input_manifest_sha256": raw_summary["raw_bundle_input_manifest_sha256"],
        "protocol_sha256": sha256_file(protocol_path),
        "encoder_provenance": raw_summary["raw_bundle_encoder_provenance"],
        "encoder_preprocessing": bundle_preproc,
        "encoder_preprocessing_sha256": truncation_summary["preprocessing_sha256"],
        "target_shapes": TARGET_SHAPES,
        "truncation_policy": "keep first target steps in original order; no resampling/interpolation",
    }
    source_hash = canonical_json_sha256(source_descriptor)
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name + ".staging-", dir=str(target.parent)))
    try:
        split_metadata = {}
        for split in ("train", "valid"):
            split_metadata[split] = _write_split(
                stage, split, split_rows[split], raw_by_id, raw_bundle, class_names
            )
        contract = {
            "schema_version": 1,
            "cache_kind": "jevaffect_raw_npz_train_valid_full" if cohort == "full" else "jevaffect_raw_npz_train_valid_pilot",
            "status": "full_official_train_valid_raw_video" if cohort == "full" else "small_train_valid_pilot_not_formal_benchmark",
            "dataset": "CH-SIMS-v2.0",
            "language": "zh",
            "class_order": list(class_names),
            "source_sha256": source_hash,
            "source_descriptor": source_descriptor,
            "text_features_ready": True,
            "feature_dims": {name: values[1] for name, values in TARGET_SHAPES.items()},
            "sequence_lengths": {name: values[0] for name, values in TARGET_SHAPES.items()},
            "split_paths": {"train": "splits/train", "valid": "splits/valid"},
            "split_counts": source_summary["split_counts"],
            "requires_all_modalities_present": True,
            "padding": "zero features; false mask and zero time for padding; lengths store kept raw sequence steps",
            "auxiliary_arrays": {
                "masks_and_lengths": "preserved for all modalities; trainer reads lengths, not mask sidecars",
                "timestamps": "audio/vision timestamps are original raw seconds in valid positions; zero padding is disambiguated by masks",
            },
            "almt_note": "Current jevaffect.almt forward does not apply padding masks; fixed zero-padded positions remain model inputs.",
            "truncation": truncation_summary["sequence_truncation"],
            "split_metadata": split_metadata,
        }
        _write_json(stage / "meta.json", contract)
        detail = {
            "schema_version": 1,
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            **source_summary,
            **raw_summary,
            "source_sha256": source_hash,
            "source_descriptor": source_descriptor,
            "class_order": list(class_names),
            "label_mapping": "jevaffect.protocol.map_simsv2_label; labels copied from official author meta.csv",
            "split_counts": source_summary["split_counts"],
            "sample_audits": record_audits,
            "truncation": truncation_summary["sequence_truncation"],
            "missing_modality_policy": "reject any sample with a missing/invalid text, audio, or vision sequence",
            "test_data": "not included in manifest join, cache arrays, or metadata counts",
            "formal_benchmark": cohort == "full",
            "formal_benchmark_note": "cohort coverage only; no trained model or test score is implied",
        }
        _write_json(stage / "metadata.json", detail)
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name not in ("meta.json", "metadata.json", "bundle.sha256"):
                relative = path.relative_to(stage).as_posix()
                files[relative] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        detail["files"] = files
        _write_json(stage / "metadata.json", detail)
        contract_hash = sha256_file(stage / "meta.json")
        detail_hash = sha256_file(stage / "metadata.json")
        (stage / "bundle.sha256").write_text(
            f"{contract_hash}  meta.json\n{detail_hash}  metadata.json\n", encoding="ascii"
        )
        if verify_training_loader_fn is None:
            verify_training_loader_fn = verify_with_training_loader
        check = verify_training_loader_fn(stage, project_root)
        _write_json(stage / "trainer_cpu_smoke_check.json", check)
        if target.exists():
            raise FileExistsError(f"cache output appeared during build: {target}")
        os.replace(stage, target)
        return {
            "output_dir": str(target),
            "source_sha256": source_hash,
            "split_counts": source_summary["split_counts"],
            "truncation": truncation_summary["sequence_truncation"],
            "trainer_cpu_smoke_check": check,
        }
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical-like SIMS v2 train/valid cache from verified raw NPZs")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--official-csv", type=Path, required=True)
    parser.add_argument("--raw-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-train", type=int, default=339)
    parser.add_argument("--expected-valid", type=int, default=43)
    parser.add_argument("--cohort", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--skip-training-loader-check", action="store_true")
    args = parser.parse_args()
    verify_fn = None if not args.skip_training_loader_check else (lambda *_: {"skipped": True})
    result = build_cache(
        project_root=args.project_root,
        manifest_path=args.manifest,
        official_csv=args.official_csv,
        raw_bundle=args.raw_bundle,
        output_dir=args.output_dir,
        expected_counts={"train": args.expected_train, "valid": args.expected_valid},
        cohort=args.cohort,
        verify_training_loader_fn=verify_fn,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
