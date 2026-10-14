#!/usr/bin/env python
"""Audit frozen TRAIN/VALID A/V timestamp grids, retained overlap, and provenance.

This script intentionally reads only cache metadata, IDs, timestamps, and
sequence lengths. Optional media checks are limited to the TRAIN clips named
by docs/native_raw_demo_source_20261013.json. It does not load labels, feature
tensors, TEST data, or learned models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW_CACHE_DEFAULT = ROOT / "data/feature_cache/simsv2_full_macbert_raw_av"
NORMALIZED_CACHE_DEFAULT = ROOT / "data/feature_cache/simsv2_native_sequence_standardized_v1"
MEDIA_MANIFEST_DEFAULT = ROOT / "docs/native_raw_demo_source_20261013.json"
SPLITS = ("train", "valid")
PAIR_TOLERANCE_SECONDS = 0.010
SAMPLE_RATE = 16_000
WAVLM_CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
WAVLM_CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
VIDEO_FPS = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def wavlm_center_and_hop_seconds(
    kernels: tuple[int, ...] = WAVLM_CONV_KERNELS,
    strides: tuple[int, ...] = WAVLM_CONV_STRIDES,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[float, float, int, int]:
    if not kernels or len(kernels) != len(strides):
        raise ValueError("WavLM kernel and stride lists must be nonempty and aligned")
    receptive_field = 1
    jump = 1
    for kernel, stride in zip(kernels, strides):
        if kernel <= 0 or stride <= 0:
            raise ValueError("WavLM kernels and strides must be positive")
        receptive_field += (kernel - 1) * jump
        jump *= stride
    center = (receptive_field - 1) / (2.0 * sample_rate)
    hop = jump / float(sample_rate)
    return center, hop, receptive_field, jump


def validate_time_array(
    times: np.ndarray,
    lengths: np.ndarray,
    *,
    name: str,
    expected_step_seconds: float | None = None,
    expected_start_seconds: float = 0.0,
    grid_atol_seconds: float = 1e-9,
) -> dict[str, Any]:
    """Check a padded timestamp matrix and return small integrity statistics."""
    times = np.asarray(times)
    lengths = np.asarray(lengths)
    if times.ndim != 2 or times.dtype.kind != "f":
        raise ValueError("{} times must be a floating [N,T] array".format(name))
    if lengths.ndim != 1 or lengths.shape[0] != times.shape[0] or lengths.dtype.kind not in "iu":
        raise ValueError("{} lengths must be an integer [N] array".format(name))
    if not np.isfinite(times).all():
        raise ValueError("{} times contain a nonfinite value".format(name))
    if np.any(lengths < 1) or np.any(lengths > times.shape[1]):
        raise ValueError("{} lengths fall outside [1,T]".format(name))

    valid_count = 0
    padded_count = 0
    duplicate_steps = 0
    negative_steps = 0
    grid_max_abs_residual = 0.0
    grid_violations = 0
    valid_first: list[float] = []
    valid_last: list[float] = []
    valid_spans: list[float] = []
    valid_steps: list[int] = []
    for row, raw_length in zip(times, lengths):
        length = int(raw_length)
        valid = row[:length]
        padded = row[length:]
        if np.any(padded != 0):
            raise ValueError("{} time padding must be zero".format(name))
        padded_count += int(padded.size)
        valid_count += length
        valid_steps.append(length)
        valid_first.append(float(valid[0]))
        valid_last.append(float(valid[-1]))
        valid_spans.append(float(valid[-1] - valid[0]))

        delta = np.diff(valid)
        negative_steps += int(np.count_nonzero(delta < 0))
        duplicate_steps += int(np.count_nonzero(delta == 0))
        if np.any(delta < 0):
            raise ValueError("{} valid timestamps must be monotonic nondecreasing".format(name))

        if expected_step_seconds is not None:
            expected = expected_start_seconds + np.arange(length, dtype=np.float64) * expected_step_seconds
            residual = np.abs(valid.astype(np.float64, copy=False) - expected)
            if residual.size:
                grid_max_abs_residual = max(grid_max_abs_residual, float(residual.max()))
                grid_violations += int(np.count_nonzero(residual > grid_atol_seconds))

    return {
        "shape": [int(times.shape[0]), int(times.shape[1])],
        "dtype": str(times.dtype),
        "valid_steps_total": valid_count,
        "lengths": _distribution(valid_steps),
        "valid_time_first_seconds": _distribution(valid_first),
        "valid_time_last_seconds": _distribution(valid_last),
        "valid_span_seconds": _distribution(valid_spans),
        "finite_values": int(times.size),
        "nonmonotonic_steps": negative_steps,
        "duplicate_valid_steps": duplicate_steps,
        "padded_time_values": padded_count,
        "zero_padded_time_values": padded_count,
        "grid_check": (
            {
                "expected_start_seconds": expected_start_seconds,
                "expected_step_seconds": expected_step_seconds,
                "atol_seconds": grid_atol_seconds,
                "max_abs_residual_seconds": grid_max_abs_residual,
                "violating_valid_values": grid_violations,
            }
            if expected_step_seconds is not None
            else None
        ),
    }


def _distribution(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def interval_overlap(audio_times: np.ndarray, vision_times: np.ndarray) -> dict[str, Any]:
    """Summarize support intersection and each stream's tails outside it."""
    audio_times = np.asarray(audio_times, dtype=np.float64)
    vision_times = np.asarray(vision_times, dtype=np.float64)
    if audio_times.ndim != 1 or vision_times.ndim != 1 or not len(audio_times) or not len(vision_times):
        raise ValueError("overlap requires nonempty one-dimensional valid timestamp arrays")
    common_start = max(float(audio_times[0]), float(vision_times[0]))
    common_end = min(float(audio_times[-1]), float(vision_times[-1]))
    intersects = common_end >= common_start
    if not intersects:
        return {
            "intersects": False,
            "start_seconds": None,
            "end_seconds": None,
            "seconds": 0.0,
            "audio_anchors_outside": int(len(audio_times)),
            "vision_anchors_outside": int(len(vision_times)),
            "audio_support_seconds_outside": float(audio_times[-1] - audio_times[0]),
            "vision_support_seconds_outside": float(vision_times[-1] - vision_times[0]),
        }
    return {
        "intersects": True,
        "start_seconds": common_start,
        "end_seconds": common_end,
        "seconds": float(common_end - common_start),
        "audio_anchors_outside": int(np.count_nonzero((audio_times < common_start) | (audio_times > common_end))),
        "vision_anchors_outside": int(np.count_nonzero((vision_times < common_start) | (vision_times > common_end))),
        "audio_support_seconds_outside": float(
            max(0.0, common_start - float(audio_times[0])) + max(0.0, float(audio_times[-1]) - common_end)
        ),
        "vision_support_seconds_outside": float(
            max(0.0, common_start - float(vision_times[0])) + max(0.0, float(vision_times[-1]) - common_end)
        ),
    }


def nearest_time_pairs(
    audio_times: np.ndarray,
    vision_times: np.ndarray,
    *,
    tolerance_seconds: float = PAIR_TOLERANCE_SECONDS,
    audio_to_video_offset_seconds: float = 0.0,
    require_common_support: bool = False,
) -> list[dict[str, float | int]]:
    """One-to-one nearest timestamp join; offset is added to audio times.

    Set require_common_support only after calculating the two streams' numeric
    support intersection. This helper does not establish physical synchrony.
    """
    audio = np.asarray(audio_times, dtype=np.float64) + float(audio_to_video_offset_seconds)
    vision = np.asarray(vision_times, dtype=np.float64)
    if audio.ndim != 1 or vision.ndim != 1 or not len(audio) or not len(vision):
        raise ValueError("pairing requires nonempty one-dimensional timestamp arrays")
    if np.any(np.diff(audio) < 0) or np.any(np.diff(vision) < 0):
        raise ValueError("pairing timestamps must be monotonic nondecreasing")
    if tolerance_seconds < 0:
        raise ValueError("pair tolerance must be nonnegative")

    overlap = interval_overlap(audio, vision) if require_common_support else None
    used_audio: set[int] = set()
    pairs: list[dict[str, float | int]] = []
    for vision_index, vision_time in enumerate(vision):
        if overlap is not None:
            if not overlap["intersects"] or not overlap["start_seconds"] <= vision_time <= overlap["end_seconds"]:
                continue
        right = int(np.searchsorted(audio, vision_time, side="left"))
        candidates = [index for index in (right - 1, right) if 0 <= index < len(audio) and index not in used_audio]
        if overlap is not None:
            candidates = [
                index for index in candidates
                if overlap["start_seconds"] <= audio[index] <= overlap["end_seconds"]
            ]
        if not candidates:
            continue
        audio_index = min(candidates, key=lambda index: abs(float(audio[index] - vision_time)))
        signed_delta = float(audio[audio_index] - vision_time)
        if abs(signed_delta) <= tolerance_seconds + 1e-12:
            pairs.append(
                {
                    "vision_index": vision_index,
                    "audio_index": audio_index,
                    "signed_audio_minus_vision_seconds": signed_delta,
                    "absolute_delta_seconds": abs(signed_delta),
                }
            )
            used_audio.add(audio_index)
    return pairs


def _check_metadata_hash(path: Path, expected: str | None) -> dict[str, Any]:
    actual = sha256_file(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": actual, "matches_expected": actual == expected}


def _summarize_split(raw_root: Path, raw_meta: dict[str, Any], raw_detail: dict[str, Any], split: str) -> dict[str, Any]:
    split_root = raw_root / "splits" / split
    ids = read_json(split_root / "ids.json")
    if not isinstance(ids, list) or len(ids) != int(raw_meta["split_counts"][split]) or len(set(ids)) != len(ids):
        raise ValueError("{} IDs are missing, duplicated, or disagree with cache metadata".format(split))
    if any(not isinstance(sample_id, str) or sample_id.count("/") != 1 for sample_id in ids):
        raise ValueError("{} IDs are not canonical video/clip IDs".format(split))

    arrays: dict[str, np.ndarray] = {}
    hashes: dict[str, dict[str, Any]] = {}
    for name in ("audio_times", "audio_lengths", "vision_times", "vision_lengths"):
        path = split_root / (name + ".npy")
        metadata_entry = raw_detail["files"]["splits/{}/{}.npy".format(split, name)]
        hashes[name + ".npy"] = _check_metadata_hash(path, metadata_entry.get("sha256"))
        if not hashes[name + ".npy"]["matches_expected"]:
            raise ValueError("{} {} file hash differs from frozen metadata".format(split, name))
        arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)

    center, audio_hop, rf_samples, stride_samples = wavlm_center_and_hop_seconds()
    video_hop = 1.0 / VIDEO_FPS
    audio_checks = validate_time_array(
        arrays["audio_times"],
        arrays["audio_lengths"],
        name=split + " audio",
        expected_step_seconds=audio_hop,
        expected_start_seconds=center,
    )
    vision_checks = validate_time_array(
        arrays["vision_times"],
        arrays["vision_lengths"],
        name=split + " vision",
        expected_step_seconds=video_hop,
        expected_start_seconds=0.0,
    )
    overlaps: list[float] = []
    audio_outside = 0
    vision_outside = 0
    audio_outside_seconds = 0.0
    vision_outside_seconds = 0.0
    no_overlap_ids: list[str] = []
    pairs_per_record: list[int] = []
    pair_residuals: list[float] = []
    for row, sample_id in enumerate(ids):
        audio_length = int(arrays["audio_lengths"][row])
        vision_length = int(arrays["vision_lengths"][row])
        audio = np.asarray(arrays["audio_times"][row, :audio_length], dtype=np.float64)
        vision = np.asarray(arrays["vision_times"][row, :vision_length], dtype=np.float64)
        coverage = interval_overlap(audio, vision)
        if coverage["intersects"]:
            overlaps.append(float(coverage["seconds"]))
        else:
            no_overlap_ids.append(sample_id)
        audio_outside += int(coverage["audio_anchors_outside"])
        vision_outside += int(coverage["vision_anchors_outside"])
        audio_outside_seconds += float(coverage["audio_support_seconds_outside"])
        vision_outside_seconds += float(coverage["vision_support_seconds_outside"])
        pairs = nearest_time_pairs(
            audio,
            vision,
            tolerance_seconds=PAIR_TOLERANCE_SECONDS,
            require_common_support=True,
        )
        pairs_per_record.append(len(pairs))
        pair_residuals.extend(float(pair["absolute_delta_seconds"]) for pair in pairs)

    return {
        "record_count": len(ids),
        "time_and_length_file_sha256": hashes,
        "audio": audio_checks,
        "vision": vision_checks,
        "numeric_support_overlap": {
            "records_with_intersection": len(overlaps),
            "records_without_intersection": len(no_overlap_ids),
            "no_intersection_ids": no_overlap_ids,
            "seconds": _distribution(overlaps),
            "audio_anchors_outside_intersection_total": audio_outside,
            "vision_anchors_outside_intersection_total": vision_outside,
            "audio_support_seconds_outside_intersection_total": audio_outside_seconds,
            "vision_support_seconds_outside_intersection_total": vision_outside_seconds,
            "outside_seconds_definition": (
                "sum of each valid timestamp support's start/end tail length beyond the per-record "
                "closed interval intersection; counts are valid anchors outside that interval"
            ),
        },
        "candidate_numeric_time_pairs": {
            "tolerance_seconds": PAIR_TOLERANCE_SECONDS,
            "method": "one-to-one nearest audio anchor for each vision anchor, both anchors within numeric support intersection",
            "physical_synchrony_established": False,
            "pairs_total": int(sum(pairs_per_record)),
            "records_with_pairs": int(sum(count > 0 for count in pairs_per_record)),
            "pairs_per_record": _distribution(pairs_per_record),
            "absolute_time_delta_seconds": _distribution(pair_residuals),
        },
    }


def _check_bundle_sidecar(raw_root: Path) -> dict[str, Any]:
    sidecar = raw_root / "bundle.sha256"
    parsed: dict[str, str] = {}
    for line in sidecar.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) == 2:
            parsed[parts[1]] = parts[0]
    actuals = {
        "meta.json": sha256_file(raw_root / "meta.json"),
        "metadata.json": sha256_file(raw_root / "metadata.json"),
    }
    return {
        "path": str(sidecar.relative_to(ROOT)),
        "sha256": sha256_file(sidecar),
        "expected_hashes": parsed,
        "actual_hashes": actuals,
        "matches": parsed == actuals,
    }


def _normalized_sequence_audit(normalized_root: Path, raw_metadata_sha256: str) -> dict[str, Any]:
    metadata_path = normalized_root / "metadata.json"
    metadata = read_json(metadata_path)
    metadata_sha = sha256_file(metadata_path)
    preserved: dict[str, Any] = {}
    for split in SPLITS:
        preserved[split] = {}
        for modality in ("audio", "vision"):
            key = modality + "_times.npy"
            entry = metadata["preserved_time_artifacts"][split][key]
            output_path = normalized_root / entry["output_relative_path"]
            actual = sha256_file(output_path)
            preserved[split][key] = {
                "source_relative_path": entry["source_relative_path"],
                "output_relative_path": entry["output_relative_path"],
                "source_sha256": entry["source_sha256"],
                "declared_output_sha256": entry["output_sha256"],
                "actual_output_sha256": actual,
                "source_matches_output": entry["source_sha256"] == actual,
                "declared_output_matches_actual": entry["output_sha256"] == actual,
            }
            if actual != entry["source_sha256"] or actual != entry["output_sha256"]:
                raise ValueError("normalized {} {} timestamps differ from raw source".format(split, modality))
    return {
        "path": str(metadata_path.relative_to(ROOT)),
        "metadata_sha256": metadata_sha,
        "raw_source_metadata_sha256": metadata.get("source_metadata_sha256"),
        "raw_source_metadata_matches_actual": metadata.get("source_metadata_sha256") == raw_metadata_sha256,
        "source_cache": metadata.get("source_cache"),
        "input_form": metadata.get("input_form"),
        "normalization": metadata.get("normalization"),
        "sequence_shapes": metadata.get("sequence_shapes"),
        "counts": metadata.get("counts"),
        "test_exported": metadata.get("test_exported"),
        "preserved_time_artifacts": preserved,
    }


def _parse_timestamp_line(line: str, key: str) -> float | None:
    match = re.search(r"\b{}\s*:\s*([-+0-9.eE]+)".format(re.escape(key)), line)
    return float(match.group(1)) if match else None


def _first_demux_line(lines: list[str], stream_type: str) -> str | None:
    for line in lines:
        if "demuxer ->" in line and "type:{} ".format(stream_type) in line:
            return line
    return None


def _inspect_representative_media(media_manifest_path: Path, train_ids: set[str]) -> dict[str, Any]:
    try:
        import imageio_ffmpeg
    except ImportError:
        return {"available": False, "reason": "imageio_ffmpeg is not installed"}

    sys.path.insert(0, str(ROOT))
    from jevaffect.media import decode_video

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    version_result = subprocess.run(
        [ffmpeg, "-version"], capture_output=True, text=True, check=False, timeout=30
    )
    version_line = (version_result.stdout or version_result.stderr).splitlines()[0].strip()
    manifest = read_json(media_manifest_path)
    rows: list[dict[str, Any]] = []
    for record in manifest.get("records", []):
        sample_id = record["id"]
        if sample_id not in train_ids:
            raise ValueError("representative media record is not in raw TRAIN IDs: {}".format(sample_id))
        path = Path(record["path"]).resolve()
        try:
            relative_path = path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise ValueError("representative media path must remain inside the project") from exc
        source_sha = sha256_file(path)
        if source_sha != record.get("sha256"):
            raise ValueError("representative TRAIN source SHA mismatch: {}".format(sample_id))

        debug = subprocess.run(
            [
                ffmpeg, "-nostdin", "-hide_banner", "-debug_ts", "-i", str(path),
                "-map", "0", "-t", "0.10", "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        lines = debug.stderr.splitlines()
        header = next((line for line in lines if "Duration:" in line), "")
        start_match = re.search(r"\bstart:\s*([-+0-9.eE]+)", header)
        audio_demux = _first_demux_line(lines, "audio")
        video_demux = _first_demux_line(lines, "video")

        audio_info = subprocess.run(
            [
                ffmpeg, "-nostdin", "-hide_banner", "-i", str(path), "-map", "0:a:0",
                "-vn", "-t", "0.10", "-ac", "1", "-ar", "16000",
                "-af", "ashowinfo", "-f", "f32le", "NUL",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        audio_info_line = next(
            (line.strip() for line in audio_info.stderr.splitlines() if "ashowinfo" in line),
            None,
        )
        audio_decode = decode_video(path, fps=VIDEO_FPS, size=224, max_seconds=120.0)
        rows.append(
            {
                "id": sample_id,
                "split": "train",
                "path": relative_path,
                "source_sha256": source_sha,
                "source_sha_matches_manifest": True,
                "ffmpeg_input_header": header.strip() or None,
                "container_start_seconds": float(start_match.group(1)) if start_match else None,
                "first_audio_demux_packet_pts_seconds": (
                    _parse_timestamp_line(audio_demux, "pkt_pts_time") if audio_demux else None
                ),
                "first_video_demux_packet_pts_seconds": (
                    _parse_timestamp_line(video_demux, "pkt_pts_time") if video_demux else None
                ),
                "first_video_demux_packet_dts_seconds": (
                    _parse_timestamp_line(video_demux, "pkt_dts_time") if video_demux else None
                ),
                "first_decoded_audio_frame_pts_seconds": (
                    _parse_timestamp_line(audio_info_line or "", "pts_time")
                ),
                "audio_ashowinfo_first_line": audio_info_line,
                "decoded_waveform_sample_count_16000hz": int(audio_decode["metadata"]["audio_samples"]),
                "decoded_video_frame_count": int(audio_decode["metadata"]["frame_count"]),
                "decoded_video_frame_timestamps_first": audio_decode["frame_times"][:6].tolist(),
                "decoded_video_timestamp_kind": audio_decode["metadata"]["frame_timestamps_kind"],
                "audio_waveform_first_sample_pts_saved_by_media_py": False,
            }
        )
    return {
        "available": True,
        "manifest_path": str(media_manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(media_manifest_path),
        "ffmpeg_version": version_line,
        "sample_count": len(rows),
        "records": rows,
        "interpretation": (
            "Both sampled TRAIN MP4s report container start 0 and first audio/video packet PTS 0; "
            "the filtered video output starts at 0. These two samples do not establish full-corpus "
            "audio/video origin or drift."
        ),
    }


def build_audit(
    *,
    raw_root: Path = RAW_CACHE_DEFAULT,
    normalized_root: Path = NORMALIZED_CACHE_DEFAULT,
    media_manifest_path: Path = MEDIA_MANIFEST_DEFAULT,
    inspect_media: bool = True,
) -> dict[str, Any]:
    raw_root = raw_root.resolve()
    normalized_root = normalized_root.resolve()
    raw_meta_path = raw_root / "meta.json"
    raw_detail_path = raw_root / "metadata.json"
    raw_meta = read_json(raw_meta_path)
    raw_detail = read_json(raw_detail_path)
    meta_sha = sha256_file(raw_meta_path)
    detail_sha = sha256_file(raw_detail_path)
    bundle = _check_bundle_sidecar(raw_root)
    if not bundle["matches"]:
        raise ValueError("raw cache metadata does not match bundle.sha256")
    if raw_meta.get("source_sha256") != raw_detail.get("source_sha256"):
        raise ValueError("raw cache metadata source_sha256 fields disagree")
    if raw_meta.get("split_counts") != raw_detail.get("split_counts"):
        raise ValueError("raw cache split counts disagree")

    split_metrics = {
        split: _summarize_split(raw_root, raw_meta, raw_detail, split)
        for split in SPLITS
    }
    normalized = _normalized_sequence_audit(normalized_root, detail_sha)
    if not normalized["raw_source_metadata_matches_actual"]:
        raise ValueError("normalized sequence cache does not point to current raw metadata")

    media = (
        _inspect_representative_media(media_manifest_path, set(read_json(raw_root / "splits/train/ids.json")))
        if inspect_media
        else {"available": False, "reason": "disabled by --skip-media"}
    )
    center, audio_hop, rf_samples, stride_samples = wavlm_center_and_hop_seconds()
    truncation = raw_meta.get("truncation", {})
    return {
        "audit_name": "native_av_time_origin_and_retained_coverage",
        "audit_date": "2026-10-14",
        "scope": {
            "dataset": raw_meta.get("dataset"),
            "splits_read": list(SPLITS),
            "split_counts": raw_meta.get("split_counts"),
            "test_arrays_or_labels_read": False,
            "feature_tensors_read": False,
            "learned_models_or_predictions_run": False,
        },
        "conclusion": {
            "full_corpus_physical_av_synchrony": "UNVERIFIED",
            "reason": (
                "audio is decoded to a raw PCM pipe without persisting first audio PTS/sample-origin metadata; "
                "video timestamps come from an FFmpeg fps=2:start_time=0 output timeline. The saved per-stream "
                "seconds and zero padding do not prove a shared source-time origin or drift."
            ),
            "representative_media_scope": "two SHA-verified local TRAIN clips only",
            "allowed_interpretation": (
                "a numerical paired-sequence diagnostic using a declared timestamp join; do not call it "
                "physically synchronous or use it to claim physical-time cross-covariation until full-corpus "
                "stream origins/offsets are supported by source/decode records"
            ),
        },
        "raw_cache": {
            "path": str(raw_root.relative_to(ROOT)),
            "cache_kind": raw_meta.get("cache_kind"),
            "source_sha256": raw_meta.get("source_sha256"),
            "manifest_sha256": raw_detail.get("manifest_sha256"),
            "metadata_hashes": {
                "meta.json": {"sha256": meta_sha, "bundle_sidecar_match": bundle["actual_hashes"]["meta.json"] == bundle["expected_hashes"].get("meta.json")},
                "metadata.json": {"sha256": detail_sha, "bundle_sidecar_match": bundle["actual_hashes"]["metadata.json"] == bundle["expected_hashes"].get("metadata.json")},
                "bundle.sha256": bundle["sha256"],
            },
            "bundle_sidecar_verified": bundle["matches"],
            "preprocessing": raw_meta.get("source_descriptor", {}).get("base_raw_cache_descriptor", {}).get("encoder_preprocessing"),
            "timestamp_storage": raw_meta.get("auxiliary_arrays", {}).get("timestamps"),
            "padding_policy": raw_meta.get("padding"),
            "truncation": {
                "metadata": truncation,
                "audio_dropped_step_equivalent_seconds_at_20ms": (
                    int(truncation.get("audio", {}).get("total_dropped_frames_or_tokens", 0)) * audio_hop
                ),
                "vision_dropped_step_equivalent_seconds_at_2fps": (
                    int(truncation.get("vision", {}).get("total_dropped_frames_or_tokens", 0)) / VIDEO_FPS
                ),
                "interpretation": (
                    "step-equivalent time is dropped anchor count times nominal hop; it is not a per-record "
                    "source-duration reconstruction because the dropped timestamps are absent from this cache"
                ),
            },
        },
        "normalized_sequence_cache": normalized,
        "expected_time_grids": {
            "audio": {
                "feature": "WavLM convolution receptive-field center on waveform-relative 16 kHz samples",
                "sample_rate_hz": SAMPLE_RATE,
                "convolution_kernels": list(WAVLM_CONV_KERNELS),
                "convolution_strides": list(WAVLM_CONV_STRIDES),
                "receptive_field_samples": rf_samples,
                "output_stride_samples": stride_samples,
                "first_center_seconds": center,
                "hop_seconds": audio_hop,
            },
            "vision": {
                "feature": "FFmpeg fps=2:start_time=0 output timeline",
                "hop_seconds": 1.0 / VIDEO_FPS,
                "expected_first_timestamp_seconds": 0.0,
            },
        },
        "split_metrics": split_metrics,
        "representative_media": media,
        "minimal_pairing_contract": [
            "Join rows by canonical record ID and use only the valid prefixes from audio_lengths and vision_lengths; never treat zero padding as a real timestamp.",
            "Establish and record the source audio-to-video offset for each applicable media pipeline before making physical-time claims; if it is unavailable, explicitly label the join as a numerical paired-sequence diagnostic.",
            "After applying any verified offset, restrict candidate pairs to the closed intersection of the two retained timestamp supports. For this cache, nearest audio-to-vision joins within 10 ms give a reproducible diagnostic on the observed 20 ms/500 ms grids; use one-to-one pairs and report residuals and pair counts.",
            "Do not interpolate, repeat rows, stretch either sequence to the other's length, or extrapolate across an unobserved tail. Report the retained support overlap and outside-overlap anchors per split.",
            "The audio states are WavLM last_hidden_state rows with a local convolution-center timestamp; transformer context can span the clip, so a matched anchor does not mean the vector contains only local audio."
        ],
        "artifact_scope": {
            "script_path": "scripts/audit_native_av_times.py",
            "report_path": "docs/NATIVE_AV_TIME_AUDIT_20261014.md",
            "machine_report_path": "docs/native_av_time_audit_20261014.json",
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache", type=Path, default=RAW_CACHE_DEFAULT)
    parser.add_argument("--normalized-cache", type=Path, default=NORMALIZED_CACHE_DEFAULT)
    parser.add_argument("--media-manifest", type=Path, default=MEDIA_MANIFEST_DEFAULT)
    parser.add_argument("--skip-media", action="store_true", help="skip the two documented local TRAIN media samples")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/native_av_time_audit_20261014.json",
        help="JSON output path",
    )
    args = parser.parse_args()
    result = build_audit(
        raw_root=args.raw_cache,
        normalized_root=args.normalized_cache,
        media_manifest_path=args.media_manifest,
        inspect_media=not args.skip_media,
    )
    result["artifact_scope"]["script_sha256"] = sha256_file(Path(__file__).resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.out),
        "script_sha256": result["artifact_scope"]["script_sha256"],
        "conclusion": result["conclusion"]["full_corpus_physical_av_synchrony"],
        "splits": {name: {
            "records": rows["record_count"],
            "numeric_overlap_records": rows["numeric_support_overlap"]["records_with_intersection"],
            "candidate_pairs": rows["candidate_numeric_time_pairs"]["pairs_total"],
        } for name, rows in result["split_metrics"].items()},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
