"""Competition-ID CMU-MOSEI raw-feature reader with true missing-video masks.

This reader joins only train/valid feature artifacts to the competition's own
label manifests. It never applies the provisional author-ID crosswalk. Long
audio/video sequences are time-bin averaged across the *whole* clip to fixed
caps; valid timestamps are pooled with the same bins. Empty vision stays empty.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
CAPS = {"text": (64, 768), "audio": (400, 768), "vision": (55, 384)}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pad_or_pool(sequence: np.ndarray, times: np.ndarray | None,
                 cap: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != width or not np.isfinite(values).all():
        raise ValueError("Raw feature width/shape/nonfinite mismatch")
    length = len(values)
    if times is not None:
        timestamps = np.asarray(times, dtype=np.float32)
        if (timestamps.shape != (length,) or not np.isfinite(timestamps).all() or
                np.any(timestamps < 0) or np.any(np.diff(timestamps) < 0)):
            raise ValueError("Raw feature timestamps invalid")
    else:
        timestamps = None
    output = np.zeros((cap, width), dtype=np.float32)
    mask = np.zeros(cap, dtype=np.bool_)
    padded_times = np.zeros(cap, dtype=np.float32) if times is not None else None
    if length > cap:
        if times is None:
            raise ValueError("Text sequence unexpectedly exceeds frozen 64-token cap")
        edges = np.linspace(0, length, cap + 1, dtype=np.int64)
        if np.any(np.diff(edges) < 1):
            raise ValueError("Time-bin pooling produced an empty interval")
        for index in range(cap):
            lo, hi = int(edges[index]), int(edges[index + 1])
            output[index] = values[lo:hi].mean(axis=0)
            padded_times[index] = float(timestamps[lo:hi].mean())
        mask[:] = True
        return output, mask, padded_times, length - cap
    output[:length] = values
    mask[:length] = True
    if padded_times is not None:
        padded_times[:length] = timestamps
    return output, mask, padded_times, 0


class MOSEIRawSplit(Dataset):
    def __init__(self, *, feature_dir: str | Path, input_manifest: str | Path,
                 label_manifest: str | Path, split: str,
                 verify_artifact_hashes: bool = True) -> None:
        if split not in ("train", "valid"):
            raise ValueError("MOSEI raw reader supports only train/valid")
        self.split = split
        self.feature_dir = Path(feature_dir).resolve()
        input_path = Path(input_manifest).resolve()
        label_path = Path(label_manifest).resolve()
        index_path = self.feature_dir / "index.jsonl"
        meta_path = self.feature_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("cache_kind") != "raw_video_feature_records" or meta.get("model_results") is not True:
            raise ValueError("Not a verified real raw-feature record cache")
        all_inputs = load_jsonl(input_path)
        indexed = load_jsonl(index_path)
        if len(all_inputs) != len(indexed) or len(indexed) != meta["sample_count"]:
            raise ValueError("Feature input/index row count mismatch")
        if [row["id"] for row in all_inputs] != [row["id"] for row in indexed]:
            raise ValueError("Feature input/index ID order mismatch")
        labels = load_jsonl(label_path)
        label_by_id = {row["ID"]: row for row in labels}
        if len(labels) != len(label_by_id) or any(row["split"] != split for row in labels):
            raise ValueError("Competition label manifest duplicate/split mismatch")
        self.rows: list[dict[str, Any]] = []
        for input_row, indexed_row in zip(all_inputs, indexed, strict=True):
            if input_row["split"] != split:
                continue
            sample_id = input_row["id"]
            label = label_by_id.get(sample_id)
            if label is None or label["class"] not in CLASSES:
                raise ValueError("Competition ID/class missing from own split manifest")
            metadata = indexed_row["metadata"]
            missing_video = bool(input_row.get("allow_missing_video", False))
            if (metadata["id"] != sample_id or
                    metadata["decode"]["video_status"] != ("missing_file" if missing_video else "present") or
                    metadata["decode"]["audio_status"] != "present" or
                    metadata["transcript_source"] not in ("provided", "missing")):
                raise ValueError("Input/feature media presence or transcript mismatch")
            artifact = (self.feature_dir / indexed_row["artifact"]).resolve()
            artifact.relative_to(self.feature_dir)
            if not artifact.is_file() or (verify_artifact_hashes and sha(artifact) != indexed_row["artifact_sha256"]):
                raise ValueError("Feature artifact missing or SHA-256 mismatch")
            self.rows.append({"ID": sample_id, "label": CLASSES.index(label["class"]),
                              "source_video": label["source_video"],
                              "missing_video": missing_video,
                              "artifact": artifact})
        if not self.rows or len({row["ID"] for row in self.rows}) != len(self.rows):
            raise ValueError("Empty or duplicate MOSEI split")
        self.ids = tuple(row["ID"] for row in self.rows)
        self.source_groups = frozenset(row["source_video"] for row in self.rows)
        self.identity = {"feature_index_sha256": sha(index_path),
                         "feature_meta_sha256": sha(meta_path),
                         "input_manifest_sha256": sha(input_path),
                         "label_manifest_sha256": sha(label_path),
                         "split": split, "count": len(self.rows),
                         "class_order": list(CLASSES),
                         "missing_video_count": sum(row["missing_video"] for row in self.rows),
                         "resampling": "time-bin mean across whole audio/vision clip when above cap; empty vision remains masked"}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with np.load(row["artifact"], allow_pickle=False) as raw:
            values = {}
            for kind, (cap, width) in CAPS.items():
                times = raw[kind + "_times"] if kind != "text" else None
                feature, mask, timeline, _ = _pad_or_pool(raw[kind], times, cap, width)
                raw_mask = np.asarray(raw[kind + "_mask"], dtype=np.bool_)
                if raw_mask.shape != (len(raw[kind]),) or not np.all(raw_mask):
                    raise ValueError("Expected true-for-real-prefix raw feature mask")
                if kind == "vision" and row["missing_video"] and mask.any():
                    raise ValueError("Absent MP4 created nonempty visual features")
                values[kind] = torch.from_numpy(feature)
                values[kind + "_mask"] = torch.from_numpy(mask)
                if timeline is not None:
                    values[kind + "_times"] = torch.from_numpy(timeline)
        return {"id": row["ID"], "label": torch.tensor(row["label"], dtype=torch.long), **values}


def load_mosei_raw_train_valid(*, feature_dir: str | Path,
                               input_manifest: str | Path,
                               manifest_dir: str | Path,
                               expected_counts: tuple[int, int] | None = None):
    train = MOSEIRawSplit(feature_dir=feature_dir, input_manifest=input_manifest,
                          label_manifest=Path(manifest_dir) / "mosei_train_manifest.jsonl", split="train")
    valid = MOSEIRawSplit(feature_dir=feature_dir, input_manifest=input_manifest,
                          label_manifest=Path(manifest_dir) / "mosei_valid_manifest.jsonl", split="valid")
    if expected_counts and (len(train), len(valid)) != expected_counts:
        raise ValueError("Expected MOSEI cohort counts differ")
    if train.source_groups & valid.source_groups:
        raise ValueError("MOSEI train/valid source videos overlap")
    return train, valid, {"train": train.identity, "valid": valid.identity,
                          "source_group_overlap": 0, "official_test_loaded": False}
