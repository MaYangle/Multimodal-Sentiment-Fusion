"""Audit 76 real MOSEI train/valid feature records, including missing video."""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    media_root = ROOT / "data/mosei/pilot_media"
    feature_root = ROOT / "data/mosei/pilot_feature_cache"
    selection_path = media_root / "selection.json"
    input_path = media_root / "input_records.jsonl"
    download_path = media_root / "download_manifest.json"
    index_path = feature_root / "index.jsonl"
    meta_path = feature_root / "meta.json"
    inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line]
    downloads = json.loads(download_path.read_text(encoding="utf-8"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (len(inputs) != 76 or len(rows) != 76 or [x["id"] for x in inputs] != [x["id"] for x in rows] or
            meta["sample_count"] != 76):
        raise ValueError("English raw-feature index/manifest count or order mismatch")
    if downloads["complete_count"] != downloads["expected_count"] or downloads["expected_count"] != 146:
        raise ValueError("Authenticated media inventory incomplete")
    if meta.get("trained_sentiment_head_included") is not False or meta.get("model_results") is not True:
        raise ValueError("Pilot raw feature status invalid")
    statuses = Counter()
    lengths = {name: [] for name in ("text", "audio", "vision")}
    wall = []
    for input_row, row in zip(inputs, rows, strict=True):
        metadata = row["metadata"]
        artifact = feature_root / row["artifact"]
        if not artifact.is_file() or sha(artifact) != row["artifact_sha256"]:
            raise ValueError("Raw feature artifact hash mismatch")
        video_missing = bool(input_row["allow_missing_video"])
        decode = metadata["decode"]
        modalities = metadata["modalities"]
        if decode["video_status"] != ("missing_file" if video_missing else "present"):
            raise ValueError("Video presence was not represented exactly")
        if decode["audio_status"] != "present" or metadata["transcript_source"] != "provided":
            raise ValueError("English separate audio/provided transcript did not run")
        if modalities["vision"]["valid"] is video_missing or modalities["audio"]["valid"] is not True or modalities["text"]["valid"] is not True:
            raise ValueError("Modality mask does not match true availability")
        with np.load(artifact, allow_pickle=False) as arrays:
            for kind, dim in (("text", 768), ("audio", 768), ("vision", 384)):
                value = arrays[kind]
                mask = arrays[kind + "_mask"]
                if value.ndim != 2 or value.shape[1] != dim or mask.shape != (len(value),) or not np.isfinite(value).all():
                    raise ValueError("Bad raw feature sequence or mask")
                if len(value) != modalities[kind]["length"]:
                    raise ValueError("Raw index/NPZ feature length mismatch")
                lengths[kind].append(len(value))
        statuses["video_missing" if video_missing else "video_present"] += 1
        statuses[input_row["split"]] += 1
        wall.append(float(metadata["performance"]["record_wall_seconds"]))
    report = {"scope": "small real English train/valid raw-media flow check; not a model benchmark",
              "sample_count": 76, "split_counts": {key: statuses[key] for key in ("train", "valid")},
              "video_present": statuses["video_present"], "video_missing": statuses["video_missing"],
              "separate_wav_present": 76, "provided_transcript_present": 76,
              "feature_lengths": {kind: {"min": min(values), "median": statistics.median(values),
                                         "max": max(values)} for kind, values in lengths.items()},
              "per_video_wall_seconds": {"median": statistics.median(wall), "max": max(wall)},
              "selection_sha256": sha(selection_path), "input_manifest_sha256": sha(input_path),
              "download_manifest_sha256": sha(download_path), "feature_index_sha256": sha(index_path),
              "feature_meta_sha256": sha(meta_path),
              "all_npz_sha256_verified": True, "test_assets_or_labels_used": False,
              "sentiment_checkpoint_loaded": False}
    output = ROOT / "docs/mosei_pilot_feature_flow.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
