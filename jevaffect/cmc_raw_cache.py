"""Strict adapter for the project's distinct raw-feature pilot cache.

This reads only ``meta.json``, ``metadata.json``, ``bundle.sha256`` and the
train/valid split files produced by ``work/raw_cache_impl/scripts/build_cache.py``.
It does not use the canonical MMSA cache reader and does not inspect or create a
test split. The feature arrays are already contextual frozen-mBERT sequences,
so they are exposed to CMC as ``cached_contextual_text``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CLASS_ORDER = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
SPLITS = ("train", "valid")
REQUIRED_SPLIT_FILES = {
    "ids.json", "labels.npy", "label_valid.npy", "raw_labels.npy",
    "raw_labels.json", "manifest_labels.json", "record_audit.jsonl",
    "text.npy", "text_mask.npy", "text_lengths.npy",
    "audio.npy", "audio_mask.npy", "audio_lengths.npy", "audio_times.npy",
    "vision.npy", "vision_mask.npy", "vision_lengths.npy", "vision_times.npy",
}


class RawCacheError(ValueError):
    """Raised when raw-pilot provenance or array contracts fail."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RawCacheError("cannot read {}: {}".format(path, exc)) from exc


def _resolve_under(root: Path, relative: str, *, name: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RawCacheError("{} path is unsafe".format(name))
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RawCacheError("{} path escapes raw-cache root".format(name)) from exc
    return resolved


def _load_sidecar(root: Path) -> tuple[str, dict[str, str]]:
    path = root / "bundle.sha256"
    if not path.is_file():
        raise RawCacheError("raw-cache bundle.sha256 is missing")
    parsed: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 2 or parts[1] not in ("meta.json", "metadata.json"):
            raise RawCacheError("bundle.sha256 line {} is malformed".format(line_number))
        if parts[1] in parsed or len(parts[0]) != 64:
            raise RawCacheError("bundle.sha256 has duplicate or invalid entries")
        parsed[parts[1]] = parts[0]
    if set(parsed) != {"meta.json", "metadata.json"}:
        raise RawCacheError("bundle.sha256 must pin meta.json and metadata.json")
    for name, expected in parsed.items():
        if _sha256_file(root / name) != expected:
            raise RawCacheError("{} SHA-256 does not match bundle.sha256".format(name))
    return _sha256_file(path), parsed


def _load_array(path: Path, *, name: str) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RawCacheError("cannot load {}: {}".format(name, exc)) from exc


def _validate_split_arrays(split: str, data: dict[str, Any], expected_meta: dict[str, Any]) -> None:
    ids = data["ids"]
    count = len(ids)
    if not count or len(set(ids)) != count:
        raise RawCacheError("{} IDs must be non-empty and unique".format(split))
    if any(not isinstance(sample_id, str) or sample_id.count("/") != 1 for sample_id in ids):
        raise RawCacheError("{} IDs must be canonical video/clip strings".format(split))
    labels = data["labels"]
    label_valid = data["label_valid"]
    raw_labels = data["raw_labels"]
    if labels.dtype != np.dtype(np.int64) or labels.shape != (count,):
        raise RawCacheError("{} labels.npy must be int64 [N]".format(split))
    if label_valid.dtype != np.dtype(np.bool_) or label_valid.shape != (count,) or not np.all(label_valid):
        raise RawCacheError("{} label_valid.npy must be bool [N] and all true".format(split))
    if np.any(labels < 0) or np.any(labels >= len(CLASS_ORDER)):
        raise RawCacheError("{} class labels must be mapped indices 0..4".format(split))
    if raw_labels.shape != (count,) or raw_labels.dtype.kind not in "fiu" or not np.isfinite(raw_labels).all():
        raise RawCacheError("{} raw_labels.npy must be finite numeric [N] provenance".format(split))

    lengths_by_modality = {}
    for modality in ("text", "audio", "vision"):
        feature = data[modality]
        mask = data[modality + "_mask"]
        lengths = data[modality + "_lengths"]
        steps = int(expected_meta["sequence_lengths"][modality])
        width = int(expected_meta["feature_dims"][modality])
        if feature.shape != (count, steps, width) or feature.dtype.kind != "f":
            raise RawCacheError("{} {} must be floating [N,{},{}]".format(split, modality, steps, width))
        if not np.isfinite(feature).all():
            raise RawCacheError("{} {} has non-finite features".format(split, modality))
        if mask.shape != (count, steps) or mask.dtype != np.dtype(np.bool_):
            raise RawCacheError("{} {}_mask must be bool [N,T]".format(split, modality))
        if lengths.shape != (count,) or lengths.dtype.kind not in "iu":
            raise RawCacheError("{} {}_lengths must be integer [N]".format(split, modality))
        if np.any(lengths < 1) or np.any(lengths > steps):
            raise RawCacheError("{} {}_lengths fall outside [1,T]".format(split, modality))
        expected_mask = np.arange(steps)[None, :] < np.asarray(lengths)[:, None]
        if not np.array_equal(mask, expected_mask):
            raise RawCacheError("{} {} mask must mark the unpadded prefix exactly".format(split, modality))
        for index, length in enumerate(lengths):
            if np.any(feature[index, int(length):] != 0):
                raise RawCacheError("{} {} padding must be zero".format(split, modality))
        lengths_by_modality[modality] = lengths

    for modality in ("audio", "vision"):
        times = data[modality + "_times"]
        steps = int(expected_meta["sequence_lengths"][modality])
        if times.shape != (count, steps) or times.dtype.kind != "f" or not np.isfinite(times).all():
            raise RawCacheError("{} {}_times must be finite float [N,T]".format(split, modality))
        for index, length in enumerate(lengths_by_modality[modality]):
            valid_times = times[index, :int(length)]
            if np.any(np.diff(valid_times) < 0) or np.any(times[index, int(length):] != 0):
                raise RawCacheError("{} {} timestamps violate prefix/padding policy".format(split, modality))

    label_rows = data["manifest_labels"]
    if not isinstance(label_rows, list) or len(label_rows) != count:
        raise RawCacheError("{} manifest_labels.json must align with IDs".format(split))
    raw_label_rows = data["raw_labels_json"]
    if not isinstance(raw_label_rows, list) or len(raw_label_rows) != count:
        raise RawCacheError("{} raw_labels.json must align with IDs".format(split))
    for index, sample_id in enumerate(ids):
        expected = label_rows[index]
        if (not isinstance(expected, dict) or expected.get("ID") != sample_id
                or expected.get("class_index") != int(labels[index])
                or expected.get("label") != CLASS_ORDER[int(labels[index])]):
            raise RawCacheError("{} manifest label provenance disagrees at {}".format(split, sample_id))
        try:
            raw_value = float(raw_label_rows[index])
        except (TypeError, ValueError) as exc:
            raise RawCacheError("{} raw label JSON is invalid for {}".format(split, sample_id)) from exc
        if not np.isclose(raw_value, float(raw_labels[index]), rtol=0.0, atol=1e-12):
            raise RawCacheError("{} raw label provenance disagrees for {}".format(split, sample_id))


def load_raw_feature_pilot_train_valid(cache_dir: str | Path) -> tuple[Any, Any, dict[str, Any]]:
    """Read and verify the train/valid raw-pilot cache, without requesting test."""
    try:
        from vendor.cmc.cache_adapter import CMCFeatureSplit
    except ImportError:
        try:
            from cmc_port_impl.cache_adapter import CMCFeatureSplit
        except ImportError as exc:
            raise RawCacheError("CMCFeatureSplit source adapter is unavailable") from exc

    root = Path(cache_dir).expanduser().resolve()
    if not root.is_dir():
        raise RawCacheError("raw feature cache does not exist: {}".format(root))
    meta_path = root / "meta.json"
    detail_path = root / "metadata.json"
    if not meta_path.is_file() or not detail_path.is_file():
        raise RawCacheError("raw pilot requires both meta.json and metadata.json")
    sidecar_sha, pinned_hashes = _load_sidecar(root)
    meta = _read_json(meta_path)
    detail = _read_json(detail_path)
    cohorts = {
        "jevaffect_raw_npz_train_valid_pilot": ("small_train_valid_pilot_not_formal_benchmark", {"train": 339, "valid": 43}),
        "jevaffect_raw_npz_train_valid_full": ("full_official_train_valid_raw_video", {"train": 2722, "valid": 647}),
    }
    kind = meta.get("cache_kind")
    if kind not in cohorts or meta.get("status") != cohorts[kind][0]:
        raise RawCacheError("unsupported raw-video cohort kind/status")
    if meta.get("split_counts") != cohorts[kind][1]:
        raise RawCacheError("cohort has unexpected official split counts")
    if meta.get("class_order") != list(CLASS_ORDER) or detail.get("class_order") != list(CLASS_ORDER):
        raise RawCacheError("raw cache class order must be SNEG,WNEG,NEUT,WPOS,SPOS")
    if meta.get("dataset") != "CH-SIMS-v2.0" or meta.get("language") != "zh":
        raise RawCacheError("raw cache is not the CH-SIMS v2 Chinese pilot")
    if meta.get("text_features_ready") is not True:
        raise RawCacheError("CMC contextual-text mode requires ready frozen mBERT sequences")
    if set(meta.get("split_paths", {})) != set(SPLITS):
        raise RawCacheError("raw pilot must expose only train and valid split paths")
    if set(meta.get("split_counts", {})) != set(SPLITS):
        raise RawCacheError("raw pilot metadata must contain train/valid counts only")
    if detail.get("test_data") != "not included in manifest join, cache arrays, or metadata counts":
        raise RawCacheError("raw cache metadata does not confirm test exclusion")
    if detail.get("source_sha256") != meta.get("source_sha256"):
        raise RawCacheError("raw cache detailed metadata source hash disagrees")
    source_sha = meta.get("source_sha256")
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise RawCacheError("raw cache source_sha256 is invalid")
    if meta.get("feature_dims") != {"text": 768, "audio": 768, "vision": 384}:
        raise RawCacheError("raw feature dimensions differ from the validated mBERT/WavLM/DINO schema")
    if meta.get("sequence_lengths") != {"text": 64, "audio": 400, "vision": 55}:
        raise RawCacheError("raw sequence lengths differ from the 64/400/55 pilot adapter contract")

    files = detail.get("files")
    if not isinstance(files, dict):
        raise RawCacheError("metadata.json must list SHA-256 values for raw pilot files")
    if any(path.startswith("splits/test/") for path in files):
        raise RawCacheError("raw pilot file inventory unexpectedly contains a test split")
    if pinned_hashes["meta.json"] != _sha256_file(meta_path) or pinned_hashes["metadata.json"] != _sha256_file(detail_path):
        raise RawCacheError("raw cache sidecar changed during preflight")
    split_file_hashes: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        split_relative = meta["split_paths"][split]
        expected_split_path = "splits/" + split
        if split_relative != expected_split_path:
            raise RawCacheError("{} raw split path must be {}".format(split, expected_split_path))
        listed = {
            name[len(expected_split_path) + 1:]: item
            for name, item in files.items() if name.startswith(expected_split_path + "/")
        }
        missing = REQUIRED_SPLIT_FILES - set(listed)
        if missing:
            raise RawCacheError("{} raw split file inventory lacks {}".format(split, sorted(missing)))
        split_file_hashes[split] = {}
        for relative_name, record in listed.items():
            if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
                raise RawCacheError("{} file hash entry is malformed: {}".format(split, relative_name))
            path = _resolve_under(root, expected_split_path + "/" + relative_name, name="split file")
            if not path.is_file() or path.stat().st_size != record.get("size_bytes"):
                raise RawCacheError("{} split file missing or size changed: {}".format(split, relative_name))
            actual_hash = _sha256_file(path)
            if actual_hash != record["sha256"]:
                raise RawCacheError("{} split file SHA-256 mismatch: {}".format(split, relative_name))
            split_file_hashes[split][relative_name] = actual_hash

    datasets = {}
    try:
        for split in SPLITS:
            split_dir = _resolve_under(root, meta["split_paths"][split], name="split directory")
            ids = _read_json(split_dir / "ids.json")
            if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
                raise RawCacheError("{} ids.json must be a list of non-empty strings".format(split))
            if len(ids) != int(meta["split_counts"][split]):
                raise RawCacheError("{} split count disagrees with meta.json".format(split))
            data: dict[str, Any] = {
                "ids": ids,
                "meta": meta,
                "labels": _load_array(split_dir / "labels.npy", name=split + " labels"),
                "label_valid": _load_array(split_dir / "label_valid.npy", name=split + " label_valid"),
                "raw_labels": _load_array(split_dir / "raw_labels.npy", name=split + " raw_labels"),
                "manifest_labels": _read_json(split_dir / "manifest_labels.json"),
                "raw_labels_json": _read_json(split_dir / "raw_labels.json"),
            }
            for modality in ("text", "audio", "vision"):
                data[modality] = _load_array(split_dir / (modality + ".npy"), name=split + " " + modality)
                data[modality + "_mask"] = _load_array(split_dir / (modality + "_mask.npy"), name=split + " " + modality + " mask")
                data[modality + "_lengths"] = _load_array(split_dir / (modality + "_lengths.npy"), name=split + " " + modality + " lengths")
            for modality in ("audio", "vision"):
                data[modality + "_times"] = _load_array(split_dir / (modality + "_times.npy"), name=split + " " + modality + " times")
            _validate_split_arrays(split, data, meta)
            # No text_bert tokens are fabricated: text.npy is contextual and
            # ContextualTextEncoder will freeze/forward it without BERT tuning.
            datasets[split] = CMCFeatureSplit(data, split)
    except Exception:
        for dataset in datasets.values():
            dataset.close()
        raise

    train, valid = datasets["train"], datasets["valid"]
    identity = {
        "cache_kind": meta["cache_kind"],
        "source_sha256": source_sha,
        "meta_sha256": pinned_hashes["meta.json"],
        "metadata_sha256": pinned_hashes["metadata.json"],
        "bundle_sidecar_sha256": sidecar_sha,
        "split_counts": dict(meta["split_counts"]),
        "split_paths": dict(meta["split_paths"]),
        "split_files_sha256": split_file_hashes,
        "train_ids_sha256": hashlib.sha256(json.dumps(list(train.ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "valid_ids_sha256": hashlib.sha256(json.dumps(list(valid.ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "text_input_mode": "cached_contextual_text",
        "contextual_text_finetuned": False,
        "test_split_loaded": False,
        "padding_policy": "validate masks/lengths/zero padding; source CMC forward has no attention-mask argument",
    }
    return train, valid, identity
