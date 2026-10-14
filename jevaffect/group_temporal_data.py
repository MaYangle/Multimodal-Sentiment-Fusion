"""Strict train-only source-group holdout reader for temporal experiments.

This reader does not open official validation or test. The local diagnostic
cache contains two subsets of official TRAIN, partitioned by source video.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jevaffect.cmc_raw_cache import (
    REQUIRED_SPLIT_FILES, _load_array, _load_sidecar, _read_json,
    _resolve_under, _sha256_file, _validate_split_arrays,
)
from jevaffect.raw_temporal_data import TemporalRawSplit
from vendor.cmc.cache_adapter import CMCFeatureSplit


EXPECTED = {"train": 2182, "valid": 540}


def load_group_temporal_train_valid(cache_dir: str | Path):
    root = Path(cache_dir).resolve()
    if not root.is_dir():
        raise ValueError("Group cache directory is missing")
    _, pinned = _load_sidecar(root)
    meta = _read_json(root / "meta.json")
    detail = _read_json(root / "metadata.json")
    manifest = _read_json(root / "group_holdout_manifest.json")
    if (meta.get("cache_kind") != "jevaffect_group_disjoint_train_holdout_v1" or
            meta.get("split_counts") != EXPECTED or
            meta.get("status") != "train_only_diagnostic_not_official_validation" or
            manifest.get("official_valid_rows_used") != 0 or
            manifest.get("official_test_rows_used") != 0 or
            manifest.get("source_group_cross_split_overlap") != 0):
        raise ValueError("Not the pre-registered train-only group holdout")
    if (detail.get("official_valid_or_test_data_read") is not False or
            detail.get("source_sha256") != meta.get("source_sha256") or
            detail.get("group_manifest_sha256") != _sha256_file(root / "group_holdout_manifest.json") or
            meta["source_descriptor"]["group_manifest_sha256"] != detail["group_manifest_sha256"]):
        raise ValueError("Group holdout provenance mismatch")
    if pinned["meta.json"] != _sha256_file(root / "meta.json") or pinned["metadata.json"] != _sha256_file(root / "metadata.json"):
        raise ValueError("Group cache sidecar mismatch")
    if meta.get("class_order") != ["SNEG", "WNEG", "NEUT", "WPOS", "SPOS"]:
        raise ValueError("Group cache class order changed")
    files = detail.get("files")
    if not isinstance(files, dict) or any(key.startswith("splits/test/") for key in files):
        raise ValueError("Group cache file inventory invalid")
    cohorts = {}
    for split in ("train", "valid"):
        if meta["split_paths"][split] != "splits/" + split:
            raise ValueError("Group cache split path invalid")
        relative = "splits/" + split + "/"
        listed = {key[len(relative):]: value for key, value in files.items() if key.startswith(relative)}
        if REQUIRED_SPLIT_FILES - set(listed):
            raise ValueError(split + " group cache lacks required arrays")
        for name, entry in listed.items():
            file_path = _resolve_under(root, relative + name, name="group cache file")
            if not file_path.is_file() or file_path.stat().st_size != entry["size_bytes"] or _sha256_file(file_path) != entry["sha256"]:
                raise ValueError(split + " group cache file hash mismatch: " + name)
        folder = root / "splits" / split
        data = {"ids": _read_json(folder / "ids.json"), "meta": meta,
                "labels": _load_array(folder / "labels.npy", name=split + " labels"),
                "label_valid": _load_array(folder / "label_valid.npy", name=split + " label_valid"),
                "raw_labels": _load_array(folder / "raw_labels.npy", name=split + " raw labels"),
                "manifest_labels": _read_json(folder / "manifest_labels.json"),
                "raw_labels_json": _read_json(folder / "raw_labels.json")}
        for modality in ("text", "audio", "vision"):
            data[modality] = _load_array(folder / (modality + ".npy"), name=split + " " + modality)
            data[modality + "_mask"] = _load_array(folder / (modality + "_mask.npy"), name=split + " " + modality + " mask")
            data[modality + "_lengths"] = _load_array(folder / (modality + "_lengths.npy"), name=split + " " + modality + " lengths")
        for modality in ("audio", "vision"):
            data[modality + "_times"] = _load_array(folder / (modality + "_times.npy"), name=split + " " + modality + " times")
        _validate_split_arrays(split, data, meta)
        base = CMCFeatureSplit(data, split)
        view = TemporalRawSplit(base, root, split)
        if len(view) != EXPECTED[split] or list(view.ids) != manifest["splits"][split]["ids"]:
            raise ValueError(split + " group manifest ID mismatch")
        cohorts[split] = view
    train, valid = cohorts["train"], cohorts["valid"]
    if {sid.split("/")[0] for sid in train.ids} & {sid.split("/")[0] for sid in valid.ids}:
        raise ValueError("Source-video group leakage")
    identity = {"source_sha256": meta["source_sha256"],
                "meta_sha256": pinned["meta.json"],
                "group_manifest_sha256": detail["group_manifest_sha256"],
                "train_count": len(train), "valid_count": len(valid),
                "source_group_cross_split_overlap": 0,
                "official_validation_or_test_loaded": False}
    return train, valid, identity
