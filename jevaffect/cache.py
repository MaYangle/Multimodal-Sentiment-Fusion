"""Trusted-source pickle to memory-mappable NumPy feature cache.

This module imports NumPy and the Python standard library only. Pickle input is
code-executable: conversion requires both an explicit trust flag and a caller
supplied SHA-256 that is checked before ``pickle.load`` is called.

The unified feature contract is ``text/audio/vision: [N,T,D]``. Source MMSA
loaders commonly consume ``text_bert: [N,3,T]`` (IDs, attention mask, segment
IDs), which is a BERT input bundle, not contextual text features. Such a source
is saved as ``text_bert.npy`` and marked ``embedding-stage-required``; it is
never relabeled as ``text.npy``. A float ``text`` array is ready only when it
matches the configured ``[N,T,expected_text_dim]`` schema in every split.

The trainer's ``labels.npy`` contract is five-class int64. Class indices come
only from the manifest's already mapped ``Label`` using
``SNEG,WNEG,NEUT,WPOS,SPOS -> 0,1,2,3,4``; this module never bins raw scores.
Source continuous values are preserved separately in ``raw_labels.npy`` and
``raw_labels.json``. If the manifest marks a test record unresolved,
``labels.npy`` stores -1 and ``label_valid.npy`` is false. Train and validation
reject unknown mapped labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np


SPLITS = ("train", "valid", "test")
TEXT_TOKEN_CHANNELS = 3
SCHEMA_VERSION = 1
try:
    # In the integrated project, use the single protocol source of truth.
    from jevaffect.protocol import CLASSES as CLASS_NAMES
except ModuleNotFoundError as exc:  # standalone fixture work area
    if exc.name != "jevaffect.protocol":
        raise
    CLASS_NAMES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")


class CachePreparationError(ValueError):
    """Raised when a source dataset cannot be converted without ambiguity."""


class CacheNotReadyError(RuntimeError):
    """Raised when a training reader is given a token-only text cache."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    raise CachePreparationError(
        "raw label must be a scalar; got {}".format(type(value).__name__)
    )


def _numeric_label(value: Any) -> tuple[float, bool]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None:
        return math.nan, False
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return math.nan, False
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return math.nan, False
        try:
            number = float(value)
        except ValueError:
            return math.nan, False
    elif isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        return math.nan, False
    return (number, math.isfinite(number))


def _scalar_label_values(raw_labels: Any, sample_count: int, field: str) -> list[Any]:
    if isinstance(raw_labels, np.ndarray):
        if raw_labels.ndim == 1:
            values = raw_labels.tolist()
        elif raw_labels.ndim == 2 and raw_labels.shape == (sample_count, 1):
            values = raw_labels[:, 0].tolist()
        else:
            raise CachePreparationError(
                "label field {!r} must be [N] or [N,1], got {}".format(
                    field, raw_labels.shape
                )
            )
    elif isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
        values = list(raw_labels)
        if values and isinstance(values[0], (list, tuple, np.ndarray)):
            flattened = []
            for index, item in enumerate(values):
                array = np.asarray(item)
                if array.size != 1:
                    raise CachePreparationError(
                        "label field {!r} sample {} is not scalar".format(field, index)
                    )
                flattened.append(array.reshape(-1)[0].item())
            values = flattened
    else:
        raise CachePreparationError("label field {!r} must be a sequence".format(field))
    if len(values) != sample_count:
        raise CachePreparationError(
            "label field {!r} has {} rows; expected {}".format(
                field, len(values), sample_count
            )
        )
    return values


def _sample_ids(raw_ids: Any, sample_count: int) -> list[str]:
    if isinstance(raw_ids, np.ndarray):
        if raw_ids.ndim != 1:
            raise CachePreparationError("id field must be a one-dimensional sequence")
        values = raw_ids.tolist()
    elif isinstance(raw_ids, Sequence) and not isinstance(raw_ids, (str, bytes)):
        values = list(raw_ids)
    else:
        raise CachePreparationError("id field must be a sequence of strings")
    if len(values) != sample_count:
        raise CachePreparationError(
            "id field has {} rows; expected {}".format(len(values), sample_count)
        )
    output = []
    for index, value in enumerate(values):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value or value.isspace():
            raise CachePreparationError(
                "id at row {} is not a non-empty string; refusing to stringify it".format(index)
            )
        output.append(value)
    if len(set(output)) != len(output):
        raise CachePreparationError("source split contains duplicate IDs")
    return output


def _sample_raw_text(raw_text: Any, sample_count: int) -> Optional[list[str]]:
    if raw_text is None:
        return None
    if isinstance(raw_text, np.ndarray):
        if raw_text.ndim != 1:
            raise CachePreparationError("raw_text must be a one-dimensional sequence")
        values = raw_text.tolist()
    elif isinstance(raw_text, Sequence) and not isinstance(raw_text, (str, bytes)):
        values = list(raw_text)
    else:
        raise CachePreparationError("raw_text must be a sequence of strings")
    if len(values) != sample_count:
        raise CachePreparationError(
            "raw_text has {} rows; expected {}".format(len(values), sample_count)
        )
    output = []
    for index, value in enumerate(values):
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, str):
            raise CachePreparationError(
                "raw_text row {} is not a string; refusing to stringify it".format(index)
            )
        output.append(value)
    return output


def _transcript_sha256(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_id(sample_id: str, id_format: str) -> str:
    """Normalize an explicitly declared ID format for manifest comparison.

    ``video_dollar_clip`` parses exactly one literal ``$_$`` separator and
    returns ``video/clip``. ``video_clip_path`` validates an existing single
    slash. ``verbatim`` leaves the string untouched. No automatic guessing is
    done, so ambiguous IDs cannot silently join the wrong manifest row.
    """
    if not isinstance(sample_id, str) or not sample_id or sample_id.isspace():
        raise CachePreparationError("manifest/source IDs must be non-empty strings")
    if id_format == "verbatim":
        return sample_id
    if id_format == "video_clip_path":
        if sample_id.count("/") != 1:
            raise CachePreparationError(
                "ID {!r} is not video/clip; check the declared ID format".format(sample_id)
            )
        video, clip = sample_id.split("/", 1)
    elif id_format == "video_dollar_clip":
        separator = "$_$"
        if sample_id.count(separator) != 1:
            raise CachePreparationError(
                "ID {!r} is not video$_$clip; check the declared ID format".format(sample_id)
            )
        video, clip = sample_id.split(separator, 1)
    else:
        raise CachePreparationError("unsupported ID format: {}".format(id_format))
    if not video or not clip or video in (".", "..") or clip in (".", ".."):
        raise CachePreparationError("unsafe video/clip ID component: {!r}".format(sample_id))
    if "/" in video or "/" in clip or "\\" in video or "\\" in clip:
        raise CachePreparationError("ID component contains a path separator: {!r}".format(sample_id))
    return video + "/" + clip


def _manifest_label_state(row: Mapping[str, Any]) -> tuple[bool, Any, Any]:
    status = str(row.get("label_status", "") or "").strip().casefold()
    mapped = row.get("Label", row.get("label"))
    raw = row.get("raw_label")
    mapped_missing = mapped is None or (
        isinstance(mapped, str) and mapped.strip().casefold() in ("", "null", "none", "nan")
    )
    if status in ("unresolved", "unknown", "invalid", "missing"):
        if not mapped_missing:
            raise CachePreparationError(
                "manifest marks a label unresolved but also supplies a mapped label"
            )
        return False, raw, mapped
    if status in ("mapped", "resolved", "valid") and mapped_missing:
        raise CachePreparationError(
            "manifest marks a label resolved but its mapped label is empty"
        )
    if not status:
        if "Label" in row or "label" in row:
            return not mapped_missing, raw, mapped
        # Raw continuous scores are provenance only. A mapped manifest Label
        # is required for the unified classifier's target.
        return False, raw, mapped
    return not mapped_missing, raw, mapped


def _class_index(mapped_label: Any, sample_id: str) -> int:
    if isinstance(mapped_label, np.generic):
        mapped_label = mapped_label.item()
    if not isinstance(mapped_label, str):
        raise CachePreparationError(
            "manifest Label for {!r} must be one of {}".format(sample_id, CLASS_NAMES)
        )
    label = mapped_label.strip()
    try:
        return CLASS_NAMES.index(label)
    except ValueError as exc:
        raise CachePreparationError(
            "manifest Label {!r} for {!r} is outside {}".format(label, sample_id, CLASS_NAMES)
        ) from exc


def _read_manifest(path: Path, id_format: str) -> tuple[dict[str, dict[str, Any]], str]:
    path = Path(path)
    if not path.is_file():
        raise CachePreparationError("author manifest does not exist: {}".format(path))
    digest = sha256_file(path)
    records: list[dict[str, Any]] = []
    suffix = path.suffix.casefold()
    try:
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise CachePreparationError(
                            "manifest JSONL line {} is not an object".format(line_number)
                        )
                    records.append(row)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    raise CachePreparationError("manifest CSV has no header")
                records.extend(reader)
        else:
            raise CachePreparationError("manifest must be .jsonl or .csv")
    except (OSError, UnicodeError, json.JSONDecodeError, csv.Error) as exc:
        raise CachePreparationError("failed to read author manifest: {}".format(exc)) from exc

    manifest: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(records, start=1):
        sample_id = row.get("ID", row.get("id"))
        split = str(row.get("mode", row.get("split", "")) or "").strip()
        if split not in SPLITS:
            raise CachePreparationError(
                "manifest row {} has unsupported split {!r}".format(row_number, split)
            )
        if isinstance(sample_id, bytes):
            sample_id = sample_id.decode("utf-8")
        canonical = canonicalize_id(sample_id, id_format)
        if canonical in manifest:
            raise CachePreparationError("author manifest contains duplicate ID {!r}".format(canonical))
        known, raw_label, mapped_label = _manifest_label_state(row)
        manifest[canonical] = {
            "id": sample_id,
            "canonical_id": canonical,
            "split": split,
            "label_known": known,
            "raw_label": raw_label,
            "mapped_label": mapped_label,
            "label_status": row.get("label_status"),
            "transcript_sha256": row.get("transcript_sha256"),
        }
    if not manifest:
        raise CachePreparationError("author manifest contains no records")
    return manifest, digest


def _manifest_raw_label_matches(source_value: Any, manifest_value: Any, tolerance: float) -> bool:
    manifest_number, manifest_valid = _numeric_label(manifest_value)
    source_number, source_valid = _numeric_label(source_value)
    if not manifest_valid or not source_valid:
        return True
    return math.isclose(source_number, manifest_number, rel_tol=0.0, abs_tol=tolerance)


def _numeric_feature(
    value: Any,
    *,
    name: str,
    sample_count: int,
    ndim: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(value)
    source_dtype = str(array.dtype)
    source_shape = list(array.shape)
    if array.dtype.kind not in "biuf":
        raise CachePreparationError(
            "{} must be a numeric array; got dtype {}".format(name, array.dtype)
        )
    if array.ndim != ndim or array.shape[0] != sample_count:
        raise CachePreparationError(
            "{} must have shape [N,...] with ndim {}; got {}".format(
                name, ndim, array.shape
            )
        )
    # Match the ALMT/CMC loaders' float32 feature convention. Full time axes
    # and zero padding are retained; only non-finite values are replaced.
    clean = np.asarray(array, dtype=np.float32)
    invalid_mask = ~np.isfinite(clean)
    replacement_count = int(np.count_nonzero(invalid_mask))
    if replacement_count:
        clean = clean.copy()
        clean[invalid_mask] = 0.0
    return clean, {
        "source_shape": source_shape,
        "source_dtype": source_dtype,
        "cache_dtype": str(clean.dtype),
        "nonfinite_replaced_with_zero": replacement_count,
    }


def _numeric_vector(value: Any, name: str, sample_count: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or array.size != sample_count:
        raise CachePreparationError(
            "{} must contain one numeric value per sample".format(name)
        )
    flattened = np.asarray(array, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(flattened)):
        raise CachePreparationError("{} contains a non-finite value".format(name))
    if np.any(flattened != np.floor(flattened)):
        raise CachePreparationError("{} must contain integer lengths".format(name))
    return flattened.astype(np.int64)


def _boolean_mask(value: Any, name: str, shape: tuple[int, int]) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf":
        raise CachePreparationError("{} must be a numeric mask with shape {}".format(name, shape))
    if not np.all(np.isfinite(array)) or not np.all(np.isin(array, (0, 1))):
        raise CachePreparationError("{} must contain only 0/1 values".format(name))
    return array.astype(np.bool_)


def _infer_trailing_zero_lengths(features: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    row_nonzero = np.any(features != 0.0, axis=-1)
    sample_count, time_steps = row_nonzero.shape
    lengths = np.zeros(sample_count, dtype=np.int64)
    interior_zero_rows = 0
    inferred_empty = 0
    for index in range(sample_count):
        positions = np.flatnonzero(row_nonzero[index])
        if positions.size == 0:
            inferred_empty += 1
            lengths[index] = 0
        else:
            lengths[index] = int(positions[-1]) + 1
            interior_zero_rows += int(np.count_nonzero(~row_nonzero[index, : lengths[index]]))
    return lengths, {
        "inferred_samples": sample_count,
        "inferred_empty_samples": inferred_empty,
        "interior_all_zero_rows_before_inferred_end": interior_zero_rows,
        "max_sequence_length": int(time_steps),
    }


def _length_and_mask(
    split_data: Mapping[str, Any],
    modality: str,
    features: np.ndarray,
    *,
    text_mask_from_tokens: Optional[np.ndarray] = None,
    infer_zero_padding: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    count, steps = features.shape[:2]
    length_field = modality + "_lengths"
    mask_field = modality + "_mask"
    source_lengths = _numeric_vector(split_data.get(length_field), length_field, count)
    source_mask = _boolean_mask(split_data.get(mask_field), mask_field, (count, steps))
    text_token_mask_used = False

    if modality == "text" and text_mask_from_tokens is not None:
        if text_mask_from_tokens.shape != (count, steps):
            text_mask_from_tokens = None
        else:
            if source_mask is not None and not np.array_equal(source_mask, text_mask_from_tokens):
                raise CachePreparationError("text_mask disagrees with text_bert attention_mask")
            source_mask = text_mask_from_tokens
            text_token_mask_used = True

    if source_lengths is not None:
        lengths = source_lengths
        source = length_field
        if source_mask is not None and not np.array_equal(source_mask.sum(axis=1), lengths):
            raise CachePreparationError(
                "{} counts disagree with {}".format(mask_field, length_field)
            )
        mask = source_mask if source_mask is not None else (
            np.arange(steps)[None, :] < lengths[:, None]
        )
    elif source_mask is not None:
        mask = source_mask
        lengths = mask.sum(axis=1).astype(np.int64)
        source = "text_bert attention_mask" if text_token_mask_used else mask_field
    elif modality == "text" and text_mask_from_tokens is not None:
        mask = text_mask_from_tokens
        lengths = mask.sum(axis=1).astype(np.int64)
        source = "text_bert attention_mask"
    elif infer_zero_padding and modality in ("audio", "vision"):
        lengths, inference = _infer_trailing_zero_lengths(features)
        mask = np.arange(steps)[None, :] < lengths[:, None]
        return lengths, mask.astype(np.bool_), {
            "source": "inferred_trailing_all_zero_rows",
            "risk": (
                "valid all-zero frames can be mistaken for padding; inference is heuristic"
            ),
            **inference,
        }
    else:
        lengths = np.full(count, steps, dtype=np.int64)
        mask = np.ones((count, steps), dtype=np.bool_)
        source = "assumed_full_length_no_source_length_or_mask"

    if np.any(lengths < 0) or np.any(lengths > steps):
        raise CachePreparationError(
            "{} must be within [0, {}]".format(length_field, steps)
        )
    return np.asarray(lengths, dtype=np.int64), np.asarray(mask, dtype=np.bool_), {
        "source": source,
        "risk": (
            "padding may be marked valid because no source length/mask was available"
            if source == "assumed_full_length_no_source_length_or_mask"
            else None
        ),
    }


def _text_plan(
    split_data: Mapping[str, Mapping[str, Any]], expected_text_dim: int
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Choose only a verified shape family; never treat token triplets as features."""
    candidates: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        data = split_data[split]
        count = _array_sample_count(data)
        context = None
        raw_text = data.get("text")
        if raw_text is not None:
            array = np.asarray(raw_text)
            token_bundle = data.get("text_bert")
            same_as_token_bundle = False
            if token_bundle is not None:
                token_array = np.asarray(token_bundle)
                same_as_token_bundle = (
                    array.ndim == 3
                    and array.shape[1] == TEXT_TOKEN_CHANNELS
                    and array.shape == token_array.shape
                )
            if (
                array.ndim == 3
                and array.shape[0] == count
                and array.shape[2] == expected_text_dim
                and array.dtype.kind == "f"
                and array.shape[1] != TEXT_TOKEN_CHANNELS
                and not same_as_token_bundle
            ):
                context = array
        token = None
        raw_tokens = data.get("text_bert")
        if raw_tokens is not None:
            array = np.asarray(raw_tokens)
            if (
                array.ndim == 3
                and array.shape[0] == count
                and array.shape[1] == TEXT_TOKEN_CHANNELS
                and array.dtype.kind in "biuf"
            ):
                token = array
        candidates[split] = {
            "context": context,
            "tokens": token,
            "text_shape": list(np.asarray(raw_text).shape) if raw_text is not None else None,
            "text_dtype": str(np.asarray(raw_text).dtype) if raw_text is not None else None,
            "text_bert_shape": list(np.asarray(raw_tokens).shape) if raw_tokens is not None else None,
            "text_bert_dtype": str(np.asarray(raw_tokens).dtype) if raw_tokens is not None else None,
        }

    has_context = [candidates[split]["context"] is not None for split in SPLITS]
    if any(has_context) and not all(has_context):
        raise CachePreparationError(
            "text feature representation is inconsistent across train/valid/test: {}".format(
                {split: candidates[split]["text_shape"] for split in SPLITS}
            )
        )
    if all(has_context):
        shapes = {tuple(candidates[split]["context"].shape[1:]) for split in SPLITS}
        if len(shapes) != 1:
            raise CachePreparationError("text contextual feature shape differs across splits")
        return "contextual", candidates
    if all(candidates[split]["tokens"] is not None for split in SPLITS):
        shapes = {tuple(candidates[split]["tokens"].shape[1:]) for split in SPLITS}
        if len(shapes) != 1:
            raise CachePreparationError("text_bert token shape differs across splits")
        return "bert_tokens", candidates
    raise CachePreparationError(
        "text schema differs across splits or is unsupported; expected float text [N,T,{}] "
        "or text_bert [N,3,T]. Observed {}".format(
            expected_text_dim,
            {split: {k: v for k, v in info.items() if k.endswith("shape") or k.endswith("dtype")}
             for split, info in candidates.items()},
        )
    )


def _array_sample_count(split_data: Mapping[str, Any]) -> int:
    ids = split_data.get("id")
    if ids is None:
        raise CachePreparationError("source split is missing required 'id' field")
    if isinstance(ids, np.ndarray):
        if ids.ndim != 1:
            raise CachePreparationError("source id field must be one-dimensional")
        return int(ids.shape[0])
    if isinstance(ids, Sequence) and not isinstance(ids, (str, bytes)):
        return len(ids)
    raise CachePreparationError("source id field must be a sequence")


def _manifest_match_split(
    split: str,
    source_ids: list[str],
    source_raw_labels: list[Any],
    manifest: Mapping[str, Mapping[str, Any]],
    source_id_format: str,
    *,
    raw_label_tolerance: float,
) -> tuple[list[str], list[dict[str, Any]], np.ndarray, np.ndarray]:
    canonical_ids = [canonicalize_id(value, source_id_format) for value in source_ids]
    if len(set(canonical_ids)) != len(canonical_ids):
        raise CachePreparationError(
            "source IDs collide after explicit normalization in {}".format(split)
        )

    class_targets = np.full(len(source_ids), -1, dtype=np.int64)
    label_valid = np.zeros(len(source_ids), dtype=np.bool_)
    manifest_rows = []
    for index, (source_id, canonical_id, raw_label) in enumerate(
        zip(source_ids, canonical_ids, source_raw_labels)
    ):
        row = manifest.get(canonical_id)
        if row is None:
            raise CachePreparationError(
                "source ID {!r} is missing from the author manifest".format(source_id)
            )
        if row["split"] != split:
            raise CachePreparationError(
                "ID {!r} is in source split {!r} but manifest split {!r}".format(
                    source_id, split, row["split"]
                )
            )
        source_number, source_label_valid = _numeric_label(raw_label)
        manifest_number, manifest_raw_valid = _numeric_label(row["raw_label"])
        if source_label_valid and manifest_raw_valid and row["label_known"] and not math.isclose(
            source_number, manifest_number, rel_tol=0.0, abs_tol=raw_label_tolerance
        ):
            raise CachePreparationError(
                "raw label mismatch for ID {!r} between source pickle and manifest".format(
                    source_id
                )
            )
        known = bool(row["label_known"])
        if split in ("train", "valid") and not known:
            raise CachePreparationError(
                "unresolved source/manifest label in {}; ID={!r}".format(split, source_id)
            )
        if known:
            class_targets[index] = _class_index(row["mapped_label"], source_id)
            label_valid[index] = True
        manifest_rows.append(
            {
                "source_id": source_id,
                "canonical_id": canonical_id,
                "split": split,
                "raw_label": _json_safe_scalar(row["raw_label"]),
                "mapped_label": _json_safe_scalar(row["mapped_label"]),
                "class_index": int(class_targets[index]),
                "label_status": row["label_status"],
                "label_known": known,
            }
        )
    return canonical_ids, manifest_rows, class_targets, label_valid


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_npy(path: Path, value: np.ndarray) -> None:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CachePreparationError("refusing to write object-dtype NPY: {}".format(path.name))
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _saved_npy_shape(path: Path) -> tuple[tuple[int, ...], str]:
    """Inspect the actual serialized NPY header without loading its payload."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(value) for value in array.shape)
    dtype = str(array.dtype)
    if isinstance(array, np.memmap) and getattr(array, "_mmap", None) is not None:
        array._mmap.close()
    return shape, dtype


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "path": path.relative_to(root).as_posix(),
    }


def convert_source_pickle(
    source_pickle: Path,
    output_dir: Path,
    *,
    expected_sha256: str,
    trust_source_pickle: bool,
    manifest_path: Path,
    source_id_format: str = "verbatim",
    manifest_id_format: str = "video_clip_path",
    label_field: str = "regression_labels",
    expected_text_dim: int = 768,
    raw_label_tolerance: float = 1e-6,
    infer_zero_padding_lengths: bool = False,
    source_reference: Optional[str] = None,
) -> dict[str, Any]:
    """Convert a hash-verified source pickle without mapping labels.

    The manifest is mandatory and must cover the exact IDs and splits in the
    source pickle. The test manifest may contain unresolved labels; such rows
    remain invalid in the training cache. ``text_features_ready`` is true only
    when every split has a float ``text`` sequence of the configured feature
    dimension. Token triplets are preserved separately and require a later
    embedding stage before training.
    """
    if not trust_source_pickle:
        raise CachePreparationError(
            "pickle was not loaded: pass --trust-source-pickle only for a trusted, verified source"
        )
    source_path = Path(source_pickle).resolve()
    target = Path(output_dir).resolve()
    if not source_path.is_file():
        raise CachePreparationError("source pickle does not exist: {}".format(source_path))
    if source_path == target or source_path.is_relative_to(target):
        raise CachePreparationError("output directory must not contain the source pickle")
    if target.exists():
        raise CachePreparationError("refusing to overwrite existing cache: {}".format(target))

    expected = (expected_sha256 or "").strip().casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CachePreparationError("expected_sha256 must be 64 hexadecimal characters")
    try:
        with source_path.open("rb") as handle:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            actual_source_hash = digest.hexdigest()
            if actual_source_hash != expected:
                raise CachePreparationError(
                    "source SHA-256 mismatch; pickle was not loaded (actual {})".format(
                        actual_source_hash
                    )
                )
            manifest, manifest_hash = _read_manifest(Path(manifest_path), manifest_id_format)
            handle.seek(0)
            source_data = pickle.load(handle)
    except CachePreparationError:
        raise
    except Exception as exc:
        raise CachePreparationError("trusted pickle could not be decoded: {}".format(exc)) from exc
    if not isinstance(source_data, Mapping):
        raise CachePreparationError("source pickle top level must map split names to records")
    missing_splits = [split for split in SPLITS if split not in source_data]
    if missing_splits:
        raise CachePreparationError(
            "source pickle missing official splits: {}".format(", ".join(missing_splits))
        )

    split_data: dict[str, Mapping[str, Any]] = {}
    ids_by_split: dict[str, list[str]] = {}
    raw_label_values: dict[str, list[Any]] = {}
    for split in SPLITS:
        entry = source_data[split]
        if not isinstance(entry, Mapping):
            raise CachePreparationError("source split {!r} must be a mapping".format(split))
        split_data[split] = entry
        count = _array_sample_count(entry)
        ids_by_split[split] = _sample_ids(entry["id"], count)
        if label_field not in entry or entry[label_field] is None:
            raw_label_values[split] = [None] * count
        else:
            raw_label_values[split] = _scalar_label_values(entry[label_field], count, label_field)

    manifest_split_counts = {split: 0 for split in SPLITS}
    for row in manifest.values():
        manifest_split_counts[row["split"]] += 1
    source_split_counts = {split: len(ids_by_split[split]) for split in SPLITS}
    if manifest_split_counts != source_split_counts:
        raise CachePreparationError(
            "manifest and pickle split counts differ: manifest={}, source={}".format(
                manifest_split_counts, source_split_counts
            )
        )

    canonical_ids: dict[str, list[str]] = {}
    manifest_label_rows: dict[str, list[dict[str, Any]]] = {}
    valid_labels: dict[str, np.ndarray] = {}
    output_labels: dict[str, np.ndarray] = {}
    raw_numeric_labels: dict[str, np.ndarray] = {}
    safe_raw_labels: dict[str, list[Any]] = {}
    for split in SPLITS:
        canonical, rows, class_targets, valid = _manifest_match_split(
            split,
            ids_by_split[split],
            raw_label_values[split],
            manifest,
            source_id_format,
            raw_label_tolerance=raw_label_tolerance,
        )
        numeric = np.full(len(raw_label_values[split]), np.nan, dtype=np.float64)
        for index, value in enumerate(raw_label_values[split]):
            number, is_valid = _numeric_label(value)
            if is_valid:
                numeric[index] = number
        canonical_ids[split] = canonical
        manifest_label_rows[split] = rows
        valid_labels[split] = valid
        output_labels[split] = class_targets
        raw_numeric_labels[split] = numeric
        safe_raw_labels[split] = [_json_safe_scalar(value) for value in raw_label_values[split]]

    text_mode, text_schema = _text_plan(split_data, expected_text_dim)

    output_parent = target.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".feature-cache-stage-", dir=str(output_parent)))
    try:
        split_metadata: dict[str, Any] = {}
        raw_text_available_by_split: dict[str, bool] = {}
        for split in SPLITS:
            entry = split_data[split]
            count = len(ids_by_split[split])
            split_root = stage / "splits" / split
            split_root.mkdir(parents=True)
            raw_text_values = _sample_raw_text(entry.get("raw_text"), count)
            raw_text_available_by_split[split] = raw_text_values is not None

            features: dict[str, np.ndarray] = {}
            feature_info: dict[str, Any] = {}
            for modality, source_key in (("audio", "audio"), ("vision", "vision")):
                if source_key not in entry:
                    raise CachePreparationError(
                        "source split {!r} is missing feature field {!r}".format(split, source_key)
                    )
                array, info = _numeric_feature(
                    entry[source_key], name="{}.{}".format(split, source_key),
                    sample_count=count, ndim=3,
                )
                features[modality] = array
                feature_info[modality] = info
                _write_npy(split_root / (modality + ".npy"), array)

            if text_mode == "contextual":
                context_raw = text_schema[split]["context"]
                text_features, info = _numeric_feature(
                    context_raw, name="{}.text".format(split), sample_count=count, ndim=3
                )
                if text_features.shape[2] != expected_text_dim:
                    raise CachePreparationError("contextual text feature dimension changed during conversion")
                features["text"] = text_features
                feature_info["text"] = info
                _write_npy(split_root / "text.npy", text_features)
            else:
                token_raw = text_schema[split]["tokens"]
                token_features, info = _numeric_feature(
                    token_raw, name="{}.text_bert".format(split), sample_count=count, ndim=3
                )
                if token_features.shape[1] != TEXT_TOKEN_CHANNELS:
                    raise CachePreparationError("text_bert must preserve its [N,3,T] token channels")
                features["text_bert"] = token_features
                feature_info["text_bert"] = info
                _write_npy(split_root / "text_bert.npy", token_features)

            count, text_steps = (
                features["text"].shape[:2]
                if text_mode == "contextual"
                else (features["text_bert"].shape[0], features["text_bert"].shape[2])
            )
            token_mask = None
            if "text_bert" in entry and text_mode == "contextual":
                token_array = np.asarray(entry["text_bert"])
                if (
                    token_array.ndim == 3
                    and token_array.shape == (count, TEXT_TOKEN_CHANNELS, text_steps)
                ):
                    token_mask = _boolean_mask(
                        np.asarray(token_array[:, 1, :]),
                        "text_bert attention_mask",
                        (count, text_steps),
                    )
            elif text_mode == "bert_tokens":
                token_mask = _boolean_mask(
                    features["text_bert"][:, 1, :],
                    "text_bert attention_mask",
                    (count, text_steps),
                )

            # Length/mask generation is explicit and recorded. Text uses the
            # BERT attention mask when available; audio/vision use CMC's source
            # lengths, or the opt-in zero-tail heuristic if absent.
            modality_lengths: dict[str, np.ndarray] = {}
            modality_masks: dict[str, np.ndarray] = {}
            length_info: dict[str, Any] = {}
            for modality in ("text", "audio", "vision"):
                source_features = (
                    features["text"] if modality == "text" and text_mode == "contextual"
                    else features["text_bert"].transpose(0, 2, 1) if modality == "text"
                    else features[modality]
                )
                lengths, mask, info = _length_and_mask(
                    entry,
                    modality,
                    source_features,
                    text_mask_from_tokens=token_mask if modality == "text" else None,
                    infer_zero_padding=infer_zero_padding_lengths,
                )
                modality_lengths[modality] = lengths
                modality_masks[modality] = mask
                length_info[modality] = info
                _write_npy(split_root / (modality + "_lengths.npy"), lengths)
                _write_npy(split_root / (modality + "_mask.npy"), mask)

            _write_npy(split_root / "labels.npy", output_labels[split])
            _write_npy(split_root / "label_valid.npy", valid_labels[split])
            _write_npy(split_root / "raw_labels.npy", raw_numeric_labels[split])
            _write_json(
                split_root / "ids.json",
                canonical_ids[split],
            )
            _write_json(
                split_root / "id_mapping.json",
                {
                    "source_id_format": source_id_format,
                    "canonical_id_format": "video/clip when parsed, otherwise source string",
                    "source_ids": ids_by_split[split],
                    "canonical_ids": canonical_ids[split],
                },
            )
            _write_json(split_root / "raw_labels.json", safe_raw_labels[split])
            _write_json(split_root / "manifest_labels.json", manifest_label_rows[split])

            raw_text_audit = None
            raw_text_audit_summary = {
                "compared": 0,
                "matches": 0,
                "mismatches": 0,
                "manifest_hash_missing": 0,
            }
            if raw_text_values is not None:
                # Store the source strings verbatim. Normalization is used only
                # to compare hashes with the manifest and never mutates this file.
                _write_json(split_root / "raw_text.json", raw_text_values)
                raw_text_audit = []
                for source_id, canonical_id, text in zip(
                    ids_by_split[split], canonical_ids[split], raw_text_values
                ):
                    manifest_hash_value = manifest[canonical_id].get("transcript_sha256")
                    if manifest_hash_value in (None, ""):
                        raw_text_audit_summary["manifest_hash_missing"] += 1
                        raw_text_audit.append(
                            {
                                "ID": canonical_id,
                                "source_id": source_id,
                                "source_normalized_sha256": _transcript_sha256(text),
                                "manifest_sha256": None,
                                "status": "manifest_hash_missing",
                            }
                        )
                        continue
                    if not isinstance(manifest_hash_value, str) or not re.fullmatch(
                        r"[0-9a-fA-F]{64}", manifest_hash_value.strip()
                    ):
                        raise CachePreparationError(
                            "manifest transcript_sha256 for {!r} is not a 64-hex digest".format(
                                canonical_id
                            )
                        )
                    source_text_hash = _transcript_sha256(text)
                    manifest_text_hash = manifest_hash_value.strip().casefold()
                    matches = source_text_hash == manifest_text_hash
                    raw_text_audit_summary["compared"] += 1
                    raw_text_audit_summary["matches" if matches else "mismatches"] += 1
                    raw_text_audit.append(
                        {
                            "ID": canonical_id,
                            "source_id": source_id,
                            "source_normalized_sha256": source_text_hash,
                            "manifest_sha256": manifest_text_hash,
                            "status": "match" if matches else "mismatch",
                        }
                    )
                _write_json(split_root / "raw_text_hash_audit.json", raw_text_audit)

            zero_rows = {}
            for modality in ("audio", "vision"):
                array = features[modality]
                all_zero = np.all(array == 0.0, axis=-1)
                zero_rows[modality] = {
                    "all_zero_rows": int(np.count_nonzero(all_zero)),
                    "all_zero_rows_marked_valid": int(
                        np.count_nonzero(all_zero & modality_masks[modality])
                    ),
                    "policy": "feature values retained; mask/lengths are stored separately",
                }

            split_metadata[split] = {
                "sample_count": count,
                "source_id_sha256": hashlib.sha256(
                    json.dumps(ids_by_split[split], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "canonical_id_set_sha256": hashlib.sha256(
                    "\n".join(sorted(canonical_ids[split])).encode("utf-8")
                ).hexdigest(),
                "feature_arrays": feature_info,
                "raw_label_field_present": label_field in entry and entry[label_field] is not None,
                "raw_text_available": raw_text_values is not None,
                "raw_text_hash_audit": raw_text_audit_summary,
                "lengths": length_info,
                "zero_padding_audit": zero_rows,
                "unresolved_label_rows": int(np.count_nonzero(~valid_labels[split])),
            }

        # Derive the public training contract from the files actually written,
        # and reject split-specific fixed sequence/feature shapes before publish.
        cache_shapes: dict[str, dict[str, list[int]]] = {}
        cache_dtypes: dict[str, dict[str, str]] = {}
        for split in SPLITS:
            split_root = stage / "splits" / split
            cache_shapes[split] = {}
            cache_dtypes[split] = {}
            for modality in ("audio", "vision"):
                shape, dtype = _saved_npy_shape(split_root / (modality + ".npy"))
                cache_shapes[split][modality] = list(shape)
                cache_dtypes[split][modality] = dtype
                split_metadata[split]["feature_arrays"][modality]["cache_shape"] = list(shape)
                split_metadata[split]["feature_arrays"][modality]["cache_dtype"] = dtype
            if text_mode == "contextual":
                shape, dtype = _saved_npy_shape(split_root / "text.npy")
                cache_shapes[split]["text"] = list(shape)
                cache_dtypes[split]["text"] = dtype
                split_metadata[split]["feature_arrays"]["text"]["cache_shape"] = list(shape)
                split_metadata[split]["feature_arrays"]["text"]["cache_dtype"] = dtype
            else:
                shape, dtype = _saved_npy_shape(split_root / "text_bert.npy")
                cache_shapes[split]["text_bert"] = list(shape)
                cache_dtypes[split]["text_bert"] = dtype
                split_metadata[split]["feature_arrays"]["text_bert"]["cache_shape"] = list(shape)
                split_metadata[split]["feature_arrays"]["text_bert"]["cache_dtype"] = dtype

        shape_modalities = ("audio", "vision", "text") if text_mode == "contextual" else ("audio", "vision", "text_bert")
        reference_split = "train"
        for modality in shape_modalities:
            reference_shape = cache_shapes[reference_split][modality][1:]
            for split in ("valid", "test"):
                observed_shape = cache_shapes[split][modality][1:]
                if observed_shape != reference_shape:
                    raise CachePreparationError(
                        "fixed {} shape mismatch: train={} but {}={} (expected identical [T,D] axes)".format(
                            modality,
                            tuple(reference_shape),
                            split,
                            tuple(observed_shape),
                        )
                    )

        feature_dims = {
            "text": cache_shapes[reference_split]["text"][2]
            if text_mode == "contextual" else None,
            "audio": cache_shapes[reference_split]["audio"][2],
            "vision": cache_shapes[reference_split]["vision"][2],
        }
        sequence_lengths = {
            "text": cache_shapes[reference_split]["text"][1]
            if text_mode == "contextual" else None,
            "audio": cache_shapes[reference_split]["audio"][1],
            "vision": cache_shapes[reference_split]["vision"][1],
        }
        split_paths = {split: "splits/" + split for split in SPLITS}
        raw_text_available = all(raw_text_available_by_split.values())

        # Compare complete manifest and source split sets after per-split checks.
        source_canonical_set = {
            sample_id for split in SPLITS for sample_id in canonical_ids[split]
        }
        if source_canonical_set != set(manifest):
            missing = len(set(manifest) - source_canonical_set)
            extra = len(source_canonical_set - set(manifest))
            raise CachePreparationError(
                "manifest/source ID sets differ after split checks (missing={}, extra={})".format(
                    missing, extra
                )
            )

        file_index = {}
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                file_index[path.relative_to(stage).as_posix()] = _artifact_record(path, stage)

        ready = text_mode == "contextual"
        status = "ready" if ready else "embedding-stage-required"
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": actual_source_hash,
            "status": status,
            "text_features_ready": ready,
            "raw_text_available": raw_text_available,
            "feature_dims": feature_dims,
            "sequence_lengths": sequence_lengths,
            "split_paths": split_paths,
            "serialized_feature_shapes": cache_shapes,
            "serialized_feature_dtypes": cache_dtypes,
            "text_representation": (
                "source float text sequence [N,T,D] validated against expected feature dimension; "
                "confirm the source field's contextual-feature provenance before training"
                if ready
                else "MMSA BERT token triplet [N,3,T]; contextual embedding stage required"
            ),
            "text_source_key": "text" if ready else "text_bert",
            "expected_text_dim": expected_text_dim,
            "source_text_schema": {
                split: {
                    key: value
                    for key, value in text_schema[split].items()
                    if key.endswith("shape") or key.endswith("dtype")
                }
                for split in SPLITS
            },
            "source_pickle": {
                "path_basename": source_path.name,
                "sha256": actual_source_hash,
                "sha256_checked_before_unpickle": True,
                "source_reference": source_reference,
                "pickle_trust_explicit": True,
            },
            "author_manifest": {
                "path_basename": Path(manifest_path).name,
                "sha256": manifest_hash,
                "manifest_id_format": manifest_id_format,
                "source_id_format": source_id_format,
                "exact_ids_and_splits_verified": True,
            },
            "splits": split_metadata,
            "labels": {
                "source_field": label_field,
                "representation": "int64 index from manifest Label; no score-to-class mapping in this converter",
                "class_order": list(CLASS_NAMES),
                "raw_source_values": "separate raw_labels.npy and raw_labels.json; never mapped",
                "unknown_test_policy": (
                    "unresolved manifest test labels are -1 in labels.npy and false in label_valid.npy; "
                    "source raw value remains in raw_labels.npy/raw_labels.json"
                ),
                "train_valid_unknown_labels": "rejected",
            },
            "padding_and_cleaning": {
                "feature_arrays": "preserve fixed source time axis and zero values",
                "nonfinite_features": "replace with zero; per-split counts are recorded in feature_arrays",
                "masks": "True means valid; generated from source masks/lengths or recorded fallback policy",
                "zero_row_risk": (
                    "an all-zero feature row may be legitimate; zero-row counts are audit-only"
                ),
                "length_inference_opt_in": infer_zero_padding_lengths,
            },
            "files": file_index,
        }
        metadata_path = stage / "metadata.json"
        _write_json(metadata_path, metadata)
        metadata_hash = sha256_file(metadata_path)
        contract = {
            "schema_version": SCHEMA_VERSION,
            "source_sha256": actual_source_hash,
            "text_features_ready": ready,
            "raw_text_available": raw_text_available,
            "feature_dims": feature_dims,
            "sequence_lengths": sequence_lengths,
            "split_paths": split_paths,
            "audit_metadata_path": "metadata.json",
            "audit_metadata_sha256": metadata_hash,
            "label_contract": {
                "class_order": list(CLASS_NAMES),
                "unresolved_test_value": -1,
                "validity_file": "label_valid.npy",
                "raw_labels_file": "raw_labels.npy",
            },
        }
        contract_path = stage / "meta.json"
        _write_json(contract_path, contract)
        contract_hash = sha256_file(contract_path)
        (stage / "metadata.sha256").write_text(
            "{}  metadata.json\n{}  meta.json\n".format(metadata_hash, contract_hash),
            encoding="ascii",
            newline="\n",
        )
        if target.exists():
            raise CachePreparationError("cache output appeared during conversion; refusing overwrite")
        os.replace(stage, target)
        return {
            "status": status,
            "text_features_ready": ready,
            "raw_text_available": raw_text_available,
            "metadata_sha256": metadata_hash,
            "meta_sha256": contract_hash,
            "source_sha256": actual_source_hash,
            "manifest_sha256": manifest_hash,
            "split_samples": source_split_counts,
            "text_schema": {
                split: {
                    "text_shape": text_schema[split]["text_shape"],
                    "text_dtype": text_schema[split]["text_dtype"],
                    "text_bert_shape": text_schema[split]["text_bert_shape"],
                    "text_bert_dtype": text_schema[split]["text_bert_dtype"],
                }
                for split in SPLITS
            },
        }
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def load_cached_split(
    cache_dir: Path,
    split: str,
    *,
    mmap_mode: Optional[str] = "r",
    require_text_features: bool = True,
    require_raw_text: bool = False,
    verify_artifact_hashes: bool = False,
) -> dict[str, Any]:
    """Load the standard ``meta.json`` contract and one safe split.

    ``meta.json`` and ``metadata.json`` are jointly checked through
    ``metadata.sha256``; ``meta.json`` also points to the detailed audit hash.
    Split directories are resolved only through the contract's confined
    relative ``split_paths``. Arrays use ``allow_pickle=False`` and are
    memory-mapped by default. Token-only caches require
    ``require_text_features=False`` for inspection and cannot enter training.
    GPT/transcript-based readers should pass ``require_raw_text=True``.
    """
    if split not in SPLITS:
        raise CachePreparationError("split must be one of {}".format(SPLITS))
    root = Path(cache_dir).resolve()
    contract_path = root / "meta.json"
    metadata_path = root / "metadata.json"
    if not contract_path.is_file() or not metadata_path.is_file():
        raise CachePreparationError("canonical meta.json and detailed metadata.json are required")
    hash_path = root / "metadata.sha256"
    if not hash_path.is_file():
        raise CachePreparationError("shared metadata hash index not found")
    hash_index = {}
    for line_number, line in enumerate(hash_path.read_text(encoding="ascii").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 2 or parts[1] in hash_index:
            raise CachePreparationError("invalid metadata.sha256 line {}".format(line_number))
        if not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise CachePreparationError("invalid SHA-256 in metadata.sha256")
        hash_index[parts[1]] = parts[0]
    if set(hash_index) != {"metadata.json", "meta.json"}:
        raise CachePreparationError("metadata.sha256 must checksum both meta.json files")
    actual_detail_hash = sha256_file(metadata_path)
    actual_contract_hash = sha256_file(contract_path)
    if actual_detail_hash != hash_index["metadata.json"]:
        raise CachePreparationError("metadata.json SHA-256 mismatch")
    if actual_contract_hash != hash_index["meta.json"]:
        raise CachePreparationError("meta.json SHA-256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA_VERSION or metadata.get("schema_version") != SCHEMA_VERSION:
        raise CachePreparationError("unsupported feature cache schema version")
    if contract.get("audit_metadata_path") != "metadata.json":
        raise CachePreparationError("meta.json has an unsupported audit metadata path")
    if contract.get("audit_metadata_sha256") != actual_detail_hash:
        raise CachePreparationError("meta.json audit metadata hash mismatch")
    if contract.get("source_sha256") != metadata.get("source_sha256"):
        raise CachePreparationError("source_sha256 differs between meta.json files")
    for key in (
        "text_features_ready", "raw_text_available", "feature_dims",
        "sequence_lengths", "split_paths",
    ):
        if contract.get(key) != metadata.get(key):
            raise CachePreparationError("{} differs between meta.json files".format(key))
    if contract.get("label_contract", {}).get("class_order") != list(CLASS_NAMES):
        raise CachePreparationError("meta.json label class order differs from jevaffect.protocol.CLASSES")
    if require_text_features and not contract.get("text_features_ready", False):
        raise CacheNotReadyError(
            "embedding-stage-required: cache has no contextual text.npy; it cannot enter training"
        )
    if require_raw_text and not contract.get("raw_text_available", False):
        raise CacheNotReadyError(
            "raw-text-unavailable: this cache cannot supply GPT or transcript-based input"
        )
    split_paths = contract.get("split_paths")
    if not isinstance(split_paths, dict) or set(split_paths) != set(SPLITS):
        raise CachePreparationError("meta.json split_paths must map train/valid/test")
    relative_split_path = Path(split_paths[split])
    if relative_split_path.is_absolute() or ".." in relative_split_path.parts:
        raise CachePreparationError("split path escapes cache root")
    root_resolved = root.resolve()
    split_root = (root / relative_split_path).resolve()
    if not split_root.is_relative_to(root_resolved):
        raise CachePreparationError("split path escapes cache root")
    if split not in metadata.get("splits", {}):
        raise CachePreparationError("detailed metadata has no split {!r}".format(split))
    if not split_root.is_dir():
        raise CachePreparationError("split path does not exist: {}".format(split_paths[split]))

    arrays: dict[str, Any] = {
        "meta": contract,
        "metadata": metadata,
        "split": split,
        "split_path": split_paths[split],
        "feature_dims": contract["feature_dims"],
        "sequence_lengths": contract["sequence_lengths"],
        "raw_text_available": contract["raw_text_available"],
    }
    if verify_artifact_hashes:
        for relative, record in metadata.get("files", {}).items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise CachePreparationError("artifact path escapes cache root: {}".format(relative))
            path = root / relative_path
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise CachePreparationError("cache artifact hash mismatch: {}".format(relative))

    for name in (
        "text", "text_bert", "audio", "vision", "labels", "label_valid", "raw_labels",
        "text_mask", "text_lengths", "audio_mask", "audio_lengths",
        "vision_mask", "vision_lengths",
    ):
        path = split_root / (name + ".npy")
        if path.is_file():
            arrays[name] = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    ids_path = split_root / "ids.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
        raise CachePreparationError("ids.json must be a list[str] of canonical manifest IDs")
    mapping = json.loads((split_root / "id_mapping.json").read_text(encoding="utf-8"))
    if mapping.get("canonical_ids") != ids:
        raise CachePreparationError("ids.json and id_mapping.json canonical IDs disagree")
    source_ids = mapping.get("source_ids")
    if not isinstance(source_ids, list) or len(source_ids) != len(ids):
        raise CachePreparationError("id_mapping.json source_ids must align with ids.json")
    if any(not isinstance(value, str) or not value for value in source_ids):
        raise CachePreparationError("id_mapping.json source_ids must be list[str]")
    if len(ids) != metadata["splits"][split]["sample_count"]:
        raise CachePreparationError("IDs and detailed split sample count disagree")
    arrays["ids"] = ids
    arrays["canonical_ids"] = ids
    arrays["source_ids"] = source_ids
    if "labels" not in arrays or "label_valid" not in arrays or "raw_labels" not in arrays:
        raise CachePreparationError("split is missing labels.npy/label_valid.npy/raw_labels.npy")
    labels = arrays["labels"]
    label_valid = arrays["label_valid"]
    raw_labels = arrays["raw_labels"]
    if labels.dtype != np.dtype(np.int64) or label_valid.dtype != np.dtype(np.bool_):
        raise CachePreparationError("labels.npy must be int64 and label_valid.npy must be bool")
    if labels.shape != (len(ids),) or label_valid.shape != (len(ids),) or raw_labels.shape != (len(ids),):
        raise CachePreparationError("label arrays and IDs have different sample counts")
    if np.any(labels[label_valid] < 0) or np.any(labels[label_valid] >= len(CLASS_NAMES)):
        raise CachePreparationError("valid labels.npy entries must be class indices 0..4")
    if np.any(labels[~label_valid] != -1):
        raise CachePreparationError("invalid labels.npy entries must use -1")
    if split in ("train", "valid") and not bool(np.all(label_valid)):
        raise CachePreparationError("train/valid cache contains unresolved labels")

    # Validate the canonical fixed-shape contract against actual NPY headers.
    modalities = ("text", "audio", "vision") if contract["text_features_ready"] else ("audio", "vision")
    for modality in modalities:
        if modality not in arrays or arrays[modality].ndim != 3:
            raise CachePreparationError("ready feature {} must be a 3-D NPY array".format(modality))
        feature = arrays[modality]
        if feature.shape[2] != contract["feature_dims"][modality]:
            raise CachePreparationError("{} feature dimension disagrees with meta.json".format(modality))
        if feature.shape[1] != contract["sequence_lengths"][modality]:
            raise CachePreparationError("{} sequence length disagrees with meta.json".format(modality))
        if feature.shape[0] != len(ids):
            raise CachePreparationError("{} features and IDs have different sample counts".format(modality))
        mask = arrays.get(modality + "_mask")
        lengths = arrays.get(modality + "_lengths")
        if mask is None or lengths is None or mask.shape != feature.shape[:2] or lengths.shape != (len(ids),):
            raise CachePreparationError("{} masks/lengths do not match feature sequence".format(modality))
        if not np.array_equal(mask.sum(axis=1), lengths):
            raise CachePreparationError("{} masks and lengths disagree".format(modality))
    if not contract["text_features_ready"] and "text_bert" in arrays:
        token_features = arrays["text_bert"]
        if token_features.ndim != 3 or token_features.shape[1] != TEXT_TOKEN_CHANNELS:
            raise CachePreparationError("text_bert.npy must retain [N,3,T] BERT input channels")
        if token_features.shape[0] != len(ids):
            raise CachePreparationError("text_bert features and IDs have different sample counts")
        if arrays["text_mask"].shape != (len(ids), token_features.shape[2]):
            raise CachePreparationError("text_bert attention mask shape mismatch")
    raw_text_path = split_root / "raw_text.json"
    raw_text_audit_path = split_root / "raw_text_hash_audit.json"
    if raw_text_path.is_file():
        raw_text = json.loads(raw_text_path.read_text(encoding="utf-8"))
        if not isinstance(raw_text, list) or len(raw_text) != len(ids) or any(
            not isinstance(value, str) for value in raw_text
        ):
            raise CachePreparationError("raw_text.json must be list[str] aligned to ids.json")
        arrays["raw_text"] = raw_text
        if not raw_text_audit_path.is_file():
            raise CachePreparationError("raw_text.json has no transcript hash audit")
        raw_text_audit = json.loads(
            raw_text_audit_path.read_text(encoding="utf-8")
        )
        if not isinstance(raw_text_audit, list) or len(raw_text_audit) != len(ids):
            raise CachePreparationError("raw_text_hash_audit.json must align with ids.json")
        if any(row.get("ID") != ids[index] for index, row in enumerate(raw_text_audit)):
            raise CachePreparationError("raw text audit IDs do not align with ids.json")
        arrays["raw_text_hash_audit"] = raw_text_audit
    elif contract["raw_text_available"]:
        raise CachePreparationError("meta.json claims raw_text_available but raw_text.json is missing")
    arrays["split_raw_text_available"] = metadata["splits"][split]["raw_text_available"]
    arrays["raw_labels_json"] = json.loads(
        (split_root / "raw_labels.json").read_text(encoding="utf-8")
    )
    arrays["manifest_labels"] = json.loads(
        (split_root / "manifest_labels.json").read_text(encoding="utf-8")
    )
    return arrays

