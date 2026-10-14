"""Acquire the complete official CH-SIMS v2 author TRAIN/VALID raw cohort.

Only CSV split/ID fields and exact author Drive inventory paths are used for
selection. Each new object receives one public bytes=0-0 preflight followed by
one finite whole-object Range request (plus at most one transport retry).
No TEST media or score values are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests


ROOT = Path(__file__).resolve().parents[1]
META_CSV = ROOT / "data/author/simsv2/meta.csv"
RAW_LISTING = ROOT / "data/simsv2_drive_listing.json"
SOURCE_INDEX = ROOT / "data/raw_feature_cache/simsv2_full_mixed/index.jsonl"
OUTPUT_DIR = ROOT / "data/author/simsv2/raw_cohort_v1"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
DOC_REPORT = ROOT / "docs/author_raw_cohort_20261014.json"
RECENT_RECHECK = ROOT / "docs/author_train_raw_recheck_20261014.json"
LEGACY_RAW = ROOT / "data/author/simsv2/raw"

EXPECTED_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
EXPECTED_LISTING_SHA256 = "6f17ce5f376cdff46694c52973be636a0ca1a858516b9222120590f803de3ae1"
STOPPED_RUN_SOURCE_SHA256 = "2efb0a7f0008983064dd9dca8fe00f22a9859f517fbf5a04936d8804a6b2998e"
EXPECTED_SPLIT_COUNTS = {"train": 2722, "valid": 647}
BASE_URL = "https://drive.usercontent.google.com/download?"
MAX_FILE_BYTES = 32 * 1024 * 1024
TIMEOUT = (15, 60)
STREAM_BLOCK_BYTES = 256 * 1024
MAX_PAYLOAD_TRANSPORT_ATTEMPTS = 2  # one initial attempt plus at most one retry
PROGRESS_FILES = 32
PROGRESS_BYTES = 64 * 1024 * 1024


class StopQueue(RuntimeError):
    def __init__(self, reason: str, record_status: str):
        super().__init__(reason)
        self.reason = reason
        self.record_status = record_status


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical_id(video_id: str, clip_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", clip_id):
        raise ValueError(f"Unsafe author ID components: {video_id!r}, {clip_id!r}")
    return f"{video_id}/{clip_id}"


def expected_remote_url(file_id: str) -> str:
    return BASE_URL + urlencode({"id": file_id, "export": "download", "confirm": "t"})


def read_targets(
    meta_path: Path,
    listing_path: Path,
    expected_split_counts: dict[str, int] | None = None,
    expected_csv_sha256: str | None = EXPECTED_CSV_SHA256,
    expected_listing_sha256: str | None = EXPECTED_LISTING_SHA256,
) -> tuple[list[dict], dict]:
    """Select exact TRAIN/VALID IDs without accessing score columns or TEST IDs."""
    inventory = json.loads(listing_path.read_text(encoding="utf-8"))
    listing = inventory if isinstance(inventory, list) else inventory.get("files", [])
    by_path: dict[str, list[dict]] = {}
    for item in listing:
        normalized = item.get("path", "").replace("/", "\\")
        by_path.setdefault(normalized, []).append(item)

    selected: list[dict] = []
    split_counts = {"train": 0, "valid": 0}
    seen_ids: set[str] = set()
    with meta_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        positions = {field: header.index(field) for field in ("mode", "video_id", "clip_id")}
        for row in reader:
            mode = row[positions["mode"]]
            if mode == "test":
                # Read only the split marker to exclude this row. Do not access
                # its identifiers or any label/score columns.
                continue
            if mode not in EXPECTED_SPLIT_COUNTS:
                raise ValueError(f"Unexpected author split label: {mode!r}")
            video_id = row[positions["video_id"]]
            clip_id = row[positions["clip_id"]]
            cid = canonical_id(video_id, clip_id)
            if cid in seen_ids:
                raise ValueError(f"Duplicate author TRAIN/VALID ID: {cid}")
            seen_ids.add(cid)
            published_path = f"Raw\\{video_id}\\{clip_id}.mp4"
            matches = by_path.get(published_path, [])
            if len(matches) != 1 or not matches[0].get("id"):
                raise ValueError(f"Raw listing does not have one exact path/ID match for {cid}")
            selected.append({
                "canonical_id": cid,
                "mode": mode,
                "video_id": video_id,
                "clip_id": clip_id,
                "file_id": matches[0]["id"],
                "published_path": published_path,
                "output_path": f"{video_id}/{clip_id}.mp4",
            })
            split_counts[mode] += 1

    expected_counts = expected_split_counts or EXPECTED_SPLIT_COUNTS
    if split_counts != expected_counts:
        raise ValueError(f"Author CSV TRAIN/VALID counts differ: {split_counts}")
    file_ids = [item["file_id"] for item in selected]
    if len(file_ids) != len(set(file_ids)):
        raise ValueError("A publisher Drive file ID maps to multiple selected canonical IDs")

    csv_hash = sha256_file(meta_path)
    listing_hash = sha256_file(listing_path)
    if expected_csv_sha256 is not None and csv_hash != expected_csv_sha256:
        raise ValueError(f"Author CSV pin mismatch: {csv_hash}")
    if expected_listing_sha256 is not None and listing_hash != expected_listing_sha256:
        raise ValueError(f"Raw listing pin mismatch: {listing_hash}")
    plan_bytes = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    pins = {
        "author_csv_sha256": csv_hash,
        "raw_listing_sha256": listing_hash,
        "selection_plan_sha256": sha256_bytes(plan_bytes),
        "selected_rows": len(selected),
        "split_counts": split_counts,
        "expected_split_counts": expected_counts,
        "fields_used": ["mode", "video_id", "clip_id"],
        "csv_structure_scanned": True,
        "test_rows_selected_or_used": False,
        "score_values_inspected_or_numerically_parsed": False,
    }
    return selected, pins


def read_competition_source_hashes(targets: list[dict]) -> dict[str, str]:
    wanted = {item["canonical_id"] for item in targets}
    result: dict[str, str] = {}
    if not SOURCE_INDEX.exists():
        return result
    with SOURCE_INDEX.open(encoding="utf-8") as stream:
        for line in stream:
            if not any(cid in line for cid in wanted):
                continue
            row = json.loads(line)
            cid = row.get("id")
            if cid in wanted:
                result[cid] = row["metadata"]["decode"]["source_sha256"]
    return result


def request_range_ok(
    status: int,
    content_range: str | None,
    expected_range: str,
    last_modified: str | None,
    expected_last_modified: str | None = None,
) -> bool:
    return (
        status == 206
        and content_range == expected_range
        and bool(last_modified)
        and (expected_last_modified is None or last_modified == expected_last_modified)
    )


def html_diagnostic(response: requests.Response) -> dict:
    sample = next(response.iter_content(4096), b"")
    lower = sample.lower()
    return {
        "error_response_bytes": len(sample),
        "error_response_prefix_sha256": sha256_bytes(sample),
        "html_prefix": lower.lstrip().startswith((b"<!doctype html", b"<html")),
        "quota_phrase_observed": any(
            term in lower for term in (b"quota exceeded", b"download quota", b"too many users")
        ),
    }


def write_progress(manifest: dict, print_line: bool = False) -> None:
    manifest["updated_at_utc"] = now_utc()
    atomic_json(MANIFEST_PATH, manifest)
    if print_line:
        print(json.dumps(progress_summary(manifest), ensure_ascii=False), flush=True)


def progress_summary(manifest: dict) -> dict:
    records = manifest["records"]
    complete = [r for r in records if r.get("status") in ("complete", "reused")]
    downloaded = sum(int(r.get("downloaded_payload_bytes", 0)) for r in records)
    reused = sum(int(r.get("reused_payload_bytes", 0)) for r in records)
    available = sum(int(r.get("bytes", 0)) for r in complete)
    partial = sum(int(r.get("partial_bytes", 0)) for r in records)
    return {
        "event": "cohort_progress",
        "status": manifest.get("status"),
        "processed_files": sum(r.get("status") != "pending" for r in records),
        "complete_files": len(complete),
        "expected_files": len(records),
        "downloaded_payload_bytes": downloaded,
        "reused_payload_bytes": reused,
        "available_payload_bytes": available,
        "preserved_partial_bytes": partial,
        "quota_stop": manifest.get("stop_reason"),
        "updated_at_utc": manifest.get("updated_at_utc"),
    }


def load_source_index() -> dict[str, str]:
    targets, _ = read_targets(META_CSV, RAW_LISTING)
    return read_competition_source_hashes(targets)


def init_manifest(targets: list[dict], pins: dict, source_hashes: dict[str, str]) -> dict:
    runner_sha = sha256_file(Path(__file__))
    config = {
        "endpoint": "https://drive.usercontent.google.com/download",
        "timeout_seconds": {"connect": TIMEOUT[0], "read": TIMEOUT[1]},
        "max_file_bytes": MAX_FILE_BYTES,
        "max_transport_attempts": MAX_PAYLOAD_TRANSPORT_ATTEMPTS,
        "workers": 1,
        "whole_file_range_only": True,
        "csv_score_values_used": False,
        "gpu_or_feature_extraction": False,
        "runner_source_sha256": runner_sha,
    }
    config_sha = sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return {
        "schema_version": 1,
        "dataset": "CH-SIMS v2",
        "status": "planned",
        "created_at_utc": now_utc(),
        "updated_at_utc": now_utc(),
        "source_pins": pins,
        "config": config,
        "config_sha256": config_sha,
        "runner_source_sha256_current": runner_sha,
        "runner_source_hash_was_runtime_pinned_at_initial_launch": True,
        "runner_source_history": [{"sha256": runner_sha, "recorded_at_utc": now_utc(), "method": "manifest creation"}],
        "source_index_matches_available": len(source_hashes),
        "stop_reason": None,
        "records": [
            {
                **item,
                "competition_source_sha256": source_hashes.get(item["canonical_id"]),
                "status": "pending",
                "preflight_attempts": 0,
                "full_payload_attempts": 0,
                "downloaded_payload_bytes": 0,
                "reused_payload_bytes": 0,
            }
            for item in targets
        ],
    }


def load_or_create_manifest(targets: list[dict], pins: dict, source_hashes: dict[str, str]) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("source_pins") != pins:
            raise RuntimeError("Existing cohort manifest source/selection pins differ; preserve and inspect it")
        if manifest.get("config", {}).get("max_file_bytes") != MAX_FILE_BYTES:
            raise RuntimeError("Existing cohort manifest budget differs; refuse silent resume")
        runner_sha = sha256_file(Path(__file__))
        config = dict(manifest.get("config", {}))
        expected_config = {
            "endpoint": "https://drive.usercontent.google.com/download",
            "timeout_seconds": {"connect": TIMEOUT[0], "read": TIMEOUT[1]},
            "max_file_bytes": MAX_FILE_BYTES,
            "max_transport_attempts": MAX_PAYLOAD_TRANSPORT_ATTEMPTS,
            "workers": 1,
            "whole_file_range_only": True,
            "csv_score_values_used": False,
            "gpu_or_feature_extraction": False,
        }
        for key, value in expected_config.items():
            if config.get(key) != value:
                raise RuntimeError(f"Existing cohort transfer config differs at {key}")
        prior_runner_sha = manifest.get("runner_source_sha256_current") or config.get("runner_source_sha256")
        history = manifest.setdefault("runner_source_history", [])
        if prior_runner_sha:
            history.append({"sha256": prior_runner_sha, "recorded_at_utc": manifest.get("updated_at_utc"), "method": "prior manifest"})
        elif STOPPED_RUN_SOURCE_SHA256 not in [item.get("sha256") for item in history]:
            history.append({
                "sha256": STOPPED_RUN_SOURCE_SHA256,
                "recorded_at_utc": "post-stop source snapshot; launch-time hash was not recorded",
                "method": "observed source snapshot and file-mtime evidence",
            })
        history.append({"sha256": runner_sha, "recorded_at_utc": now_utc(), "method": "corrected runner loaded for resume/staging"})
        manifest["runner_source_sha256_current"] = runner_sha
        manifest["runner_source_hash_was_runtime_pinned_at_initial_launch"] = manifest.get(
            "runner_source_hash_was_runtime_pinned_at_initial_launch", False
        )
        config["runner_source_sha256"] = runner_sha
        manifest["config"] = config
        manifest["config_sha256"] = sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if len(manifest.get("records", [])) != len(targets):
            raise RuntimeError("Existing manifest target count differs")
        for current, wanted in zip(manifest["records"], targets, strict=True):
            for field in ("canonical_id", "mode", "file_id", "published_path", "output_path"):
                if current.get(field) != wanted.get(field):
                    raise RuntimeError(f"Existing manifest plan mismatch at {field}")
            current_source_sha = source_hashes.get(current["canonical_id"])
            if current.get("competition_source_sha256") != current_source_sha:
                raise RuntimeError(f"Competition source SHA pin changed for {current['canonical_id']}")
        return manifest
    manifest = init_manifest(targets, pins, source_hashes)
    write_progress(manifest)
    return manifest


def output_paths(record: dict) -> tuple[Path, Path, Path, Path]:
    output = OUTPUT_DIR / record["output_path"]
    partial = output.with_suffix(output.suffix + ".partial")
    partial_state = partial.with_suffix(partial.suffix + ".json")
    receipt = output.with_suffix(output.suffix + ".provenance.json")
    return output, partial, partial_state, receipt


def validate_complete_file(record: dict) -> tuple[bool, dict | None]:
    output, _, _, receipt_path = output_paths(record)
    if not output.exists() or not receipt_path.exists():
        return False, None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    size = output.stat().st_size
    digest = sha256_file(output)
    valid = (
        receipt.get("transfer_complete") is True
        and receipt.get("canonical_id") == record["canonical_id"]
        and receipt.get("file_id") == record["file_id"]
        and receipt.get("bytes") == size
        and receipt.get("sha256") == digest
        and size <= MAX_FILE_BYTES
    )
    return valid, receipt if valid else None


def build_donor_map(targets: list[dict]) -> dict[str, list[dict]]:
    wanted = {item["canonical_id"]: item for item in targets}
    donors: dict[str, list[dict]] = {}
    if RECENT_RECHECK.exists():
        report = json.loads(RECENT_RECHECK.read_text(encoding="utf-8"))
        for row in report.get("records", []):
            cid = row.get("canonical_id") or row.get("id")
            target = wanted.get(cid)
            if not target or not row.get("transfer_complete"):
                continue
            if row.get("file_id", row.get("publisher_file_id")) != target["file_id"]:
                continue
            if row.get("source_url") != expected_remote_url(target["file_id"]):
                continue
            rel = row.get("payload_path")
            if not rel:
                continue
            source = (ROOT / rel).resolve()
            try:
                source.relative_to(ROOT.resolve())
            except ValueError:
                continue
            if not source.exists():
                continue
            expected_size = row.get("expected_bytes", row.get("actual_payload_bytes"))
            expected_sha = row.get("sha256")
            if expected_size != source.stat().st_size or expected_sha != sha256_file(source):
                continue
            if row.get("full_payload", {}).get("status") != 206 or not row.get("decoded_cpu"):
                continue
            donors.setdefault(cid, []).append({
                "path": source,
                "bytes": expected_size,
                "sha256": expected_sha,
                "last_modified": row.get("full_payload", {}).get("last_modified"),
                "source": "data/author/simsv2/raw_recheck_20261014",
            })

    # Look only at target-shaped paths; unrelated/test media are never walked.
    for target in targets:
        candidate = LEGACY_RAW / target["video_id"] / f"{target['clip_id']}.mp4"
        receipt_path = candidate.with_suffix(candidate.suffix + ".provenance.json")
        if not candidate.exists() or not receipt_path.exists():
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            source = receipt.get("source", {})
            recorded_file_id = source.get("file_id", receipt.get("file_id"))
            recorded_input_form = source.get("input_form", receipt.get("input_form"))
            recorded_url = source.get("url", receipt.get("source_url"))
            valid = (
                receipt.get("transfer_complete") is True
                and recorded_file_id == target["file_id"]
                and recorded_input_form == "author original MP4 media"
                and recorded_url == expected_remote_url(target["file_id"])
                and receipt.get("bytes") == candidate.stat().st_size
                and source.get("size", receipt.get("bytes")) == receipt.get("bytes")
                and receipt.get("sha256") == sha256_file(candidate)
            )
            if valid:
                donors.setdefault(target["canonical_id"], []).append({
                    "path": candidate,
                    "bytes": receipt["bytes"],
                    "sha256": receipt["sha256"],
                    "last_modified": source.get("Last-Modified"),
                    "source": str(candidate.relative_to(ROOT)).replace("\\", "/"),
                })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return donors


def use_donor(record: dict, donor: dict) -> dict:
    output, partial, partial_state, receipt_path = output_paths(record)
    if output.exists():
        raise RuntimeError(f"Unreceipted cohort target already exists; preserved: {output}")
    if partial.exists() or partial_state.exists():
        raise RuntimeError(f"Unexpected partial conflicts with verified cache reuse: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(donor["path"], output)
    digest = sha256_file(output)
    if output.stat().st_size != donor["bytes"] or digest != donor["sha256"]:
        raise RuntimeError(f"Copied cache changed during reuse: {record['canonical_id']}")
    receipt = {
        "canonical_id": record["canonical_id"],
        "mode": record["mode"],
        "file_id": record["file_id"],
        "published_path": record["published_path"],
        "source_url": expected_remote_url(record["file_id"]),
        "transfer_complete": True,
        "reused_from": donor["source"],
        "last_modified": donor["last_modified"],
        "bytes": donor["bytes"],
        "sha256": digest,
        "competition_source_sha256": record.get("competition_source_sha256"),
        "sha256_matches_competition_source": (
            digest == record.get("competition_source_sha256") if record.get("competition_source_sha256") else None
        ),
        "recorded_at_utc": now_utc(),
    }
    atomic_json(receipt_path, receipt)
    record.update(
        status="reused",
        reused_from=donor["source"],
        bytes=donor["bytes"],
        sha256=digest,
        reused_payload_bytes=donor["bytes"],
        transfer_complete=True,
        sha256_matches_competition_source=receipt["sha256_matches_competition_source"],
        last_modified=donor["last_modified"],
    )
    return receipt


def quota_status(response: requests.Response) -> tuple[bool, dict]:
    if "text/html" not in (response.headers.get("Content-Type") or "").lower():
        return False, {}
    diagnostic = html_diagnostic(response)
    return diagnostic["quota_phrase_observed"], diagnostic


def load_partial_state(record: dict, partial: Path, state_path: Path, preflight_prefix: bytes, total: int, last_modified: str, etag: str | None) -> dict:
    if partial.exists() and not state_path.exists():
        raise StopQueue("partial payload/state pair is incomplete; payload preserved", "partial_state_mismatch")
    if not partial.exists() and not state_path.exists():
        state = {
            "canonical_id": record["canonical_id"],
            "file_id": record["file_id"],
            "source_url": expected_remote_url(record["file_id"]),
            "expected_bytes": total,
            "last_modified": last_modified,
            "etag": etag,
            "preflight_prefix_sha256": sha256_bytes(preflight_prefix),
            "partial_bytes": 0,
            "partial_sha256": hashlib.sha256(b"").hexdigest(),
            "ranges": [],
            "updated_at_utc": now_utc(),
        }
        atomic_json(state_path, state)
        return state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    size = partial.stat().st_size if partial.exists() else 0
    digest = sha256_file(partial) if partial.exists() else hashlib.sha256(b"").hexdigest()
    partial_prefix = b""
    if size:
        with partial.open("rb") as stream:
            partial_prefix = stream.read(1)
    valid = (
        state.get("canonical_id") == record["canonical_id"]
        and state.get("file_id") == record["file_id"]
        and state.get("source_url") == expected_remote_url(record["file_id"])
        and state.get("expected_bytes") == total
        and state.get("last_modified") == last_modified
        and state.get("etag") == etag
        and state.get("preflight_prefix_sha256") == sha256_bytes(preflight_prefix)
        and state.get("partial_bytes") == size
        and state.get("partial_sha256") == digest
        and size < total
        and (size == 0 or partial_prefix == preflight_prefix)
    )
    if not valid:
        raise StopQueue("local partial does not match the pinned author source/version; preserved", "partial_integrity_mismatch")
    return state


def save_partial_state(state_path: Path, partial: Path, state: dict) -> None:
    state["partial_bytes"] = partial.stat().st_size if partial.exists() else 0
    state["partial_sha256"] = sha256_file(partial) if partial.exists() else hashlib.sha256(b"").hexdigest()
    state["updated_at_utc"] = now_utc()
    atomic_json(state_path, state)


def fetch_target(record: dict, manifest: dict) -> tuple[str, bool, int]:
    """Download one target; return (status, stop_global, actual_bytes_written)."""
    output, partial, state_path, receipt_path = output_paths(record)
    written_this_run = 0
    if output.exists():
        valid, receipt = validate_complete_file(record)
        if valid and receipt:
            record.update(
                status="complete",
                bytes=receipt["bytes"],
                sha256=receipt["sha256"],
                transfer_complete=True,
                sha256_matches_competition_source=receipt.get("sha256_matches_competition_source"),
                downloaded_payload_bytes=0,
            )
            return "complete", False, 0
        raise StopQueue("existing final output is unreceipted or fails SHA/size validation; preserved", "invalid_existing_output")

    url = expected_remote_url(record["file_id"])
    if int(record.get("preflight_attempts", 0)) >= 2:
        record.update(status="preflight_retry_budget_exhausted", transfer_complete=False)
        return record["status"], False, 0
    record["status"] = "preflight_in_progress"
    record["preflight_attempts"] = int(record.get("preflight_attempts", 0)) + 1
    write_progress(manifest)
    try:
        with requests.get(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            stream=True,
            timeout=TIMEOUT,
        ) as response:
            preflight_meta = {
                "status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "content_range": response.headers.get("Content-Range"),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
            }
            record["preflight"] = preflight_meta
            if response.status_code == 200 and "text/html" in (response.headers.get("Content-Type") or "").lower():
                quota, diagnostic = quota_status(response)
                preflight_meta.update(diagnostic)
                record.update(status="quota_html_at_preflight" if quota else "html_at_preflight", error_response_bytes=diagnostic.get("error_response_bytes", 0))
                return record["status"], True, 0
            match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match:
                raise StopQueue("author preflight did not return exact HTTP 206 bytes 0-0/total", "preflight_range_mismatch")
            total = int(match.group(1))
            last_modified = response.headers.get("Last-Modified")
            if not request_range_ok(response.status_code, response.headers.get("Content-Range"), f"bytes 0-0/{total}", last_modified):
                raise StopQueue("author preflight lacks exact range or stable Last-Modified version", "preflight_version_mismatch")
            if response.headers.get("Content-Length") not in (None, "1"):
                raise StopQueue("author preflight Content-Length is not one byte", "preflight_length_mismatch")
            prefix = bytearray()
            for block in response.iter_content(1):
                prefix.extend(block)
                if len(prefix) > 1:
                    raise StopQueue("author preflight body exceeded one byte", "preflight_body_oversized")
            if len(prefix) != 1:
                raise StopQueue("author preflight body was short", "preflight_body_short")
            preflight_prefix = bytes(prefix)
            preflight_meta["prefix_sha256"] = sha256_bytes(preflight_prefix)
            record["expected_bytes"] = total

        if total > MAX_FILE_BYTES:
            record.update(status="blocked_per_file_budget", transfer_complete=False)
            return record["status"], False, 0
        free_bytes = shutil.disk_usage(OUTPUT_DIR).free
        record["free_disk_bytes_before_payload"] = free_bytes
        if free_bytes < total + 512 * 1024 * 1024:
            record.update(status="blocked_disk_space", transfer_complete=False)
            return record["status"], False, 0

        state = load_partial_state(record, partial, state_path, preflight_prefix, total, last_modified, preflight_meta.get("etag"))
        if partial.exists() and partial.stat().st_size > 0:
            with partial.open("rb") as stream:
                existing_prefix = stream.read(1)
            if existing_prefix != preflight_prefix:
                raise StopQueue("local partial first byte differs from fresh author preflight", "partial_prefix_mismatch")
        prior_retry_count = int(record.get("transport_retry_count", 0))
        attempt_limit = 1 if prior_retry_count >= 1 else MAX_PAYLOAD_TRANSPORT_ATTEMPTS
        for attempt_number in range(1, attempt_limit + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= total:
                break
            end = total - 1
            expected_range = f"bytes {offset}-{end}/{total}"
            record["full_payload_attempts"] = int(record.get("full_payload_attempts", 0)) + 1
            record["full_payload_range"] = expected_range
            record["status"] = "full_payload_in_progress"
            write_progress(manifest)
            request_headers = {
                "Range": f"bytes={offset}-{end}",
                "If-Range": last_modified,
                "Accept-Encoding": "identity",
            }
            try:
                with requests.get(url, headers=request_headers, stream=True, timeout=TIMEOUT) as response:
                    payload_meta = {
                        "status": response.status_code,
                        "content_type": response.headers.get("Content-Type"),
                        "content_length": response.headers.get("Content-Length"),
                        "content_range": response.headers.get("Content-Range"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "etag": response.headers.get("ETag"),
                        "expected_content_range": expected_range,
                    }
                    record.setdefault("payload_responses", []).append(payload_meta)
                    if response.status_code == 200 and "text/html" in (response.headers.get("Content-Type") or "").lower():
                        quota, diagnostic = quota_status(response)
                        payload_meta.update(diagnostic)
                        record.update(status="quota_html_on_payload" if quota else "html_on_payload", error_response_bytes=diagnostic.get("error_response_bytes", 0))
                        return record["status"], True, written_this_run
                    version_ok = request_range_ok(
                        response.status_code,
                        response.headers.get("Content-Range"),
                        expected_range,
                        response.headers.get("Last-Modified"),
                        last_modified,
                    )
                    if preflight_meta.get("etag") is not None and response.headers.get("ETag") != preflight_meta["etag"]:
                        version_ok = False
                    if not version_ok:
                        raise StopQueue("full payload range or publisher version differs from preflight", "payload_range_or_version_mismatch")
                    suffix_bytes = total - offset
                    if response.headers.get("Content-Length") not in (None, str(suffix_bytes)):
                        raise StopQueue("full payload Content-Length differs from declared suffix", "payload_length_mismatch")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not partial.exists():
                        with partial.open("xb"):
                            pass
                    digest = hashlib.sha256()
                    with partial.open("rb") as existing:
                        for old_block in iter(lambda: existing.read(1024 * 1024), b""):
                            digest.update(old_block)
                    appended = 0
                    with partial.open("ab") as destination:
                        for block in response.iter_content(STREAM_BLOCK_BYTES):
                            if not block:
                                continue
                            if offset + appended + len(block) > total or partial.stat().st_size + len(block) > MAX_FILE_BYTES:
                                raise StopQueue("payload body exceeded declared object size or per-file byte budget", "payload_oversized")
                            if offset + appended == 0 and block[:1] != preflight_prefix:
                                raise StopQueue("whole payload prefix differs from fresh preflight", "payload_prefix_mismatch")
                            destination.write(block)
                            destination.flush()
                            digest.update(block)
                            appended += len(block)
                            written_this_run += len(block)
                            record["downloaded_payload_bytes"] = int(record.get("downloaded_payload_bytes", 0)) + len(block)
                            state["ranges"].append({"start": offset + appended - len(block), "end": offset + appended - 1,
                                                     "sha256": sha256_bytes(block)})
                            state["partial_bytes"] = offset + appended
                            state["partial_sha256"] = digest.hexdigest()
                            state["updated_at_utc"] = now_utc()
                            atomic_json(state_path, state)
                    if partial.stat().st_size != total:
                        raise requests.ConnectionError("Short finite object response")
                    break
            except StopQueue:
                raise
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                retry_count = int(record.get("transport_retry_count", 0))
                record.setdefault("transport_errors", []).append({
                    "type": type(exc).__name__,
                    "attempt": attempt_number,
                    "partial_bytes": partial.stat().st_size if partial.exists() else 0,
                    "time_utc": now_utc(),
                })
                if partial.exists():
                    save_partial_state(state_path, partial, state)
                    record["partial_bytes"] = partial.stat().st_size
                    record["partial_sha256"] = sha256_file(partial)
                if retry_count >= 1 or attempt_number >= attempt_limit:
                    record.update(status="transport_failed_after_one_retry", transfer_complete=False)
                    return record["status"], False, written_this_run
                record["transport_retry_count"] = retry_count + 1
                continue

        if not partial.exists() or partial.stat().st_size != total:
            record.update(status="payload_incomplete", transfer_complete=False)
            if partial.exists():
                record["partial_bytes"] = partial.stat().st_size
                record["partial_sha256"] = sha256_file(partial)
            return record["status"], False, written_this_run
        digest = sha256_file(partial)
        with partial.open("rb") as stream:
            completed_prefix = stream.read(1)
        if completed_prefix != preflight_prefix:
            raise StopQueue("completed object prefix differs from fresh preflight", "payload_prefix_mismatch")
        os.replace(partial, output)
        size = output.stat().st_size
        if size != total:
            raise StopQueue("completed output size differs from publisher total", "completed_size_mismatch")
        receipt = {
            "canonical_id": record["canonical_id"],
            "mode": record["mode"],
            "file_id": record["file_id"],
            "published_path": record["published_path"],
            "source_url": url,
            "transfer_complete": True,
            "preflight": preflight_meta,
            "payload_responses": record.get("payload_responses", []),
            "last_modified": last_modified,
            "etag": preflight_meta.get("etag"),
            "bytes": size,
            "sha256": digest,
            "competition_source_sha256": record.get("competition_source_sha256"),
            "sha256_matches_competition_source": (
                digest == record.get("competition_source_sha256") if record.get("competition_source_sha256") else None
            ),
            "payload_attempts": record.get("full_payload_attempts", 0),
            "transport_errors": record.get("transport_errors", []),
            "recorded_at_utc": now_utc(),
        }
        atomic_json(receipt_path, receipt)
        if state_path.exists():
            state_path.unlink()
        record.update(
            status="complete",
            transfer_complete=True,
            expected_bytes=total,
            bytes=size,
            sha256=digest,
            last_modified=last_modified,
            sha256_matches_competition_source=receipt["sha256_matches_competition_source"],
            completed_at_utc=now_utc(),
        )
        return "complete", False, written_this_run
    except StopQueue as exc:
        record.update(status=exc.record_status, stop_detail=str(exc), transfer_complete=False, failed_at_utc=now_utc())
        if partial.exists() and state_path.exists():
            # Preserve all bytes and persist their local digest for safe inspection.
            save_partial_state(state_path, partial, json.loads(state_path.read_text(encoding="utf-8")))
            record["partial_bytes"] = partial.stat().st_size
            record["partial_sha256"] = sha256_file(partial)
        return record["status"], True, written_this_run
    except requests.RequestException as exc:
        record.update(status="preflight_transport_error", error_type=type(exc).__name__, transfer_complete=False, failed_at_utc=now_utc())
        return record["status"], False, 0


def make_doc_report(manifest: dict) -> dict:
    records = manifest["records"]
    summary = progress_summary(manifest)
    return {
        "report_date": "2026-10-14",
        "dataset": "CH-SIMS v2",
        "author_source": "official public CH-SIMS v2 Google Drive Raw listing via drive.usercontent.google.com",
        "status": manifest.get("status"),
        "source_pins": manifest["source_pins"],
        "config_sha256": manifest.get("config_sha256"),
        "runner_source_sha256_current": manifest.get("runner_source_sha256_current"),
        "runner_source_hash_was_runtime_pinned_at_initial_launch": manifest.get("runner_source_hash_was_runtime_pinned_at_initial_launch"),
        "runner_source_history": manifest.get("runner_source_history", []),
        "selection_method": {
            "author_csv_fields_used": ["mode", "video_id", "clip_id"],
            "csv_structure_scanned": True,
            "test_rows_selected_or_used": False,
            "score_values_inspected_or_numerically_parsed": False,
            "expected_split_counts": EXPECTED_SPLIT_COUNTS,
        },
        "transfer_policy": manifest["config"],
        "progress": summary,
        "stop_reason": manifest.get("stop_reason"),
        "records": [
            {
                "canonical_id": row["canonical_id"],
                "mode": row["mode"],
                "file_id": row["file_id"],
                "published_path": row["published_path"],
                "status": row.get("status"),
                "expected_bytes": row.get("expected_bytes"),
                "bytes": row.get("bytes", row.get("downloaded_payload_bytes", 0)),
                "sha256": row.get("sha256"),
                "competition_source_sha256": row.get("competition_source_sha256"),
                "sha256_matches_competition_source": row.get("sha256_matches_competition_source"),
                "last_modified": row.get("last_modified", row.get("preflight", {}).get("last_modified")),
                "preflight": row.get("preflight"),
                "payload_responses": row.get("payload_responses"),
                "partial_bytes": row.get("partial_bytes"),
                "partial_sha256": row.get("partial_sha256"),
                "error_response_bytes": row.get("error_response_bytes"),
                "output_path": row.get("output_path"),
                "reused_from": row.get("reused_from"),
                "error_type": row.get("error_type"),
                "stop_detail": row.get("stop_detail"),
            }
            for row in records
        ],
        "updated_at_utc": manifest.get("updated_at_utc"),
    }


def persist_all(manifest: dict, write_docs: bool = True) -> None:
    write_progress(manifest)
    if write_docs:
        atomic_json(DOC_REPORT, make_doc_report(manifest))


def check_only(targets: list[dict], pins: dict, source_hashes: dict[str, str]) -> int:
    if not OUTPUT_DIR.exists() or not MANIFEST_PATH.exists():
        print(json.dumps({"status": "not_started", "expected_files": len(targets), "source_pins": pins}, ensure_ascii=False))
        return 0
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("source_pins") != pins:
        raise RuntimeError("Existing manifest source pins differ")
    bad: list[str] = []
    for row in manifest.get("records", []):
        if row.get("status") in ("complete", "reused"):
            ok, _ = validate_complete_file(row)
            if not ok:
                bad.append(row["canonical_id"])
    summary = progress_summary(manifest)
    summary["invalid_complete_files"] = bad
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if bad else 0


def stage_verified_caches(targets: list[dict], pins: dict, source_hashes: dict[str, str]) -> int:
    """Copy only exact, receipt- and SHA-verified local author MP4s; no network."""
    manifest = load_or_create_manifest(targets, pins, source_hashes)
    prior_status = manifest.get("status")
    prior_stop = manifest.get("stop_reason")
    donors = build_donor_map(targets)
    staged = 0
    for record in manifest["records"]:
        if record.get("status") in ("complete", "reused"):
            valid, _ = validate_complete_file(record)
            if not valid:
                raise RuntimeError(f"Complete cohort file failed local receipt/SHA check: {record['canonical_id']}")
            continue
        donor_list = donors.get(record["canonical_id"], [])
        output, partial, partial_state, _ = output_paths(record)
        if not donor_list or output.exists() or partial.exists() or partial_state.exists():
            continue
        use_donor(record, donor_list[0])
        staged += 1
    manifest["status"] = prior_status or "staged_local_caches"
    manifest["stop_reason"] = prior_stop
    manifest["local_cache_staging_at_utc"] = now_utc()
    manifest["local_cache_files_staged"] = int(manifest.get("local_cache_files_staged", 0)) + staged
    persist_all(manifest)
    print(json.dumps({"event": "local_cache_staging_complete", "staged_files": staged, **progress_summary(manifest)}, ensure_ascii=False), flush=True)
    return 0


def run_queue(targets: list[dict], pins: dict, source_hashes: dict[str, str], allow_quota_resume: bool) -> int:
    manifest = load_or_create_manifest(targets, pins, source_hashes)
    if manifest.get("runner_source_sha256_current") != sha256_file(Path(__file__)):
        raise RuntimeError("Loaded runner source SHA does not match the pinned manifest runner SHA")
    if manifest.get("status") == "stopped_quota" and not allow_quota_resume:
        print(json.dumps({"event": "refusing_unchanged_quota_retry", **progress_summary(manifest)}, ensure_ascii=False), flush=True)
        return 2
    if allow_quota_resume:
        manifest["stop_reason"] = None
        manifest["status"] = "running"
    elif manifest.get("status") != "complete":
        manifest["status"] = "running"
    donors = build_donor_map(targets)
    processed_since_checkpoint = 0
    bytes_since_checkpoint = 0
    manifest["started_at_utc"] = manifest.get("started_at_utc") or now_utc()
    persist_all(manifest)

    for record in manifest["records"]:
        if record.get("status") in ("complete", "reused"):
            valid, _ = validate_complete_file(record)
            if valid:
                continue
            raise RuntimeError(f"Previously completed cohort payload failed local receipt/SHA check: {record['canonical_id']}")
        if record.get("status") in (
            "blocked_per_file_budget",
            "blocked_disk_space",
            "preflight_retry_budget_exhausted",
        ):
            continue
        if record.get("status") in ("blocked_per_file_budget", "blocked_disk_space", "preflight_retry_budget_exhausted"):
            continue
        output, partial, partial_state, receipt_path = output_paths(record)
        if output.exists() and not receipt_path.exists():
            record.update(status="invalid_existing_output", stop_detail="unreceipted output preserved")
            manifest.update(status="stopped_integrity", stop_reason=f"unreceipted output: {record['canonical_id']}")
            persist_all(manifest)
            return 3
        donor_list = donors.get(record["canonical_id"], [])
        if donor_list and not partial.exists() and not partial_state.exists() and not output.exists():
            status = "reused"
            try:
                use_donor(record, donor_list[0])
                persist_all(manifest)
                processed_since_checkpoint += 1
            except Exception as exc:
                record.update(status="cache_reuse_error", error_type=type(exc).__name__, stop_detail=str(exc))
                manifest.update(status="stopped_integrity", stop_reason=f"cache reuse error: {record['canonical_id']}")
                persist_all(manifest)
                return 3
        else:
            status = str(record.get("status", "pending"))
            status, stop_global, actual = fetch_target(record, manifest)
            bytes_since_checkpoint += actual
            persist_all(manifest, write_docs=False)
            processed_since_checkpoint += 1
            if status == "complete":
                summary = progress_summary(manifest)
                print(json.dumps(summary, ensure_ascii=False), flush=True)
            if stop_global:
                is_quota = status.startswith("quota_html")
                manifest.update(
                    status="stopped_quota" if is_quota else "stopped_integrity",
                    stop_reason=f"{status}: {record['canonical_id']}",
                    stopped_at_utc=now_utc(),
                )
                persist_all(manifest)
                return 4 if is_quota else 3
        if processed_since_checkpoint >= PROGRESS_FILES or bytes_since_checkpoint >= PROGRESS_BYTES:
            print(json.dumps(progress_summary(manifest), ensure_ascii=False), flush=True)
            persist_all(manifest)
            processed_since_checkpoint = 0
            bytes_since_checkpoint = 0
        elif processed_since_checkpoint % 1 == 0:
            # Durable local manifest after each object; the docs copy is batched.
            write_progress(manifest)
        if status in ("preflight_transport_error", "transport_failed_after_one_retry", "preflight_retry_budget_exhausted", "blocked_per_file_budget", "blocked_disk_space"):
            continue

    statuses = {row.get("status") for row in manifest["records"]}
    failed = [row for row in manifest["records"] if row.get("status") not in ("complete", "reused")]
    manifest.update(
        status="complete" if not failed else "finished_with_failures",
        stop_reason=None if not failed else f"{len(failed)} selected files remain unavailable or failed",
        completed_at_utc=now_utc(),
    )
    persist_all(manifest)
    print(json.dumps({**progress_summary(manifest), "final_status": manifest["status"], "failed_count": len(failed)}, ensure_ascii=False), flush=True)
    return 0 if not failed else 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="validate source pins and local cache/receipt state without network")
    group.add_argument("--stage-caches", action="store_true", help="copy verified local author MP4 donors into the cohort tree without network")
    group.add_argument("--run", action="store_true", help="run serial official-source acquisition")
    parser.add_argument("--retry-after-quota", action="store_true", help="explicitly allow a later run after a recorded quota stop")
    args = parser.parse_args()

    targets, pins = read_targets(META_CSV, RAW_LISTING)
    source_hashes = read_competition_source_hashes(targets)
    if args.check:
        return check_only(targets, pins, source_hashes)
    if args.stage_caches:
        return stage_verified_caches(targets, pins, source_hashes)
    return run_queue(targets, pins, source_hashes, args.retry_after_quota)


if __name__ == "__main__":
    raise SystemExit(main())
