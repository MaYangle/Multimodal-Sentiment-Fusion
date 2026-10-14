"""One-shot, bounded public-author preflight and optional whole-MP4 attempt.

This script checks only two exact CH-SIMS v2 TRAIN IDs. It never reads label
values, TEST rows, or competition prediction outputs, and it makes no retries.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data/simsv2_drive_listing.json"
META = ROOT / "data/author/simsv2/meta.csv"
SOURCE_INDEX = ROOT / "data/raw_feature_cache/simsv2_full_mixed/index.jsonl"
OUT_DIR = ROOT / "data/author/simsv2/raw_recheck_20261014"
REPORT_JSON = ROOT / "docs/author_train_raw_recheck_20261014.json"
MAX_BYTES = 32 * 1024 * 1024
TIMEOUT = (15, 60)
CHUNK_BYTES = 64 * 1024
TARGETS = (
    {
        "canonical_id": "video_0003/0002",
        "mode": "train",
        "video_id": "video_0003",
        "clip_id": "0002",
        "file_id": "1knNivNIvYHVVMp20ET7fEugrqCcjlSBC",
        "published_path": r"Raw\video_0003\0002.mp4",
        "prior_competition_source_sha256": "40cddabf922ef30c67e2be2fb49fb5023a89c57b9c03674ba51fcca87c9f3e1e",
    },
    {
        "canonical_id": "video_0006/0006",
        "mode": "train",
        "video_id": "video_0006",
        "clip_id": "0006",
        "file_id": "1d4I2zWU4DnDZpf4I7Cxc64iNiSXAH1Hn",
        "published_path": r"Raw\video_0006\0006.mp4",
        "prior_competition_source_sha256": "b37619116720abfc82ffae00355db07b0a38260123db9733a5ce8d8f6ab27933",
    },
)
URL_PREFIX = "https://drive.usercontent.google.com/download?"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_selection_and_source() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    listing = inventory if isinstance(inventory, list) else inventory.get("files", [])
    for target in TARGETS:
        matches = [
            item for item in listing
            if item.get("path", "").replace("/", "\\") == target["published_path"]
        ]
        if len(matches) != 1 or matches[0].get("id") != target["file_id"]:
            raise ValueError(f"Raw inventory mismatch for {target['canonical_id']}")

    wanted = {target["canonical_id"] for target in TARGETS}
    observed: dict[str, tuple[str, str]] = {}
    with META.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        positions = {name: header.index(name) for name in ("mode", "video_id", "clip_id")}
        for values in reader:
            # The author CSV is split-ordered. Stop as soon as the requested
            # training IDs have both been found, before reading validation or
            # test records. No label columns are retained or converted.
            mode = values[positions["mode"]]
            if mode != "train":
                break
            canonical_id = f"{values[positions['video_id']]}/{values[positions['clip_id']]}"
            if canonical_id in wanted:
                observed[canonical_id] = (mode, canonical_id)
                if set(observed) == wanted:
                    break
    if set(observed) != wanted or any(mode != "train" for mode, _ in observed.values()):
        raise ValueError("Author metadata mode/ID selection mismatch")

    source_hashes: dict[str, str] = {}
    with SOURCE_INDEX.open(encoding="utf-8") as stream:
        for line in stream:
            if not any(canonical_id in line for canonical_id in wanted):
                continue
            row = json.loads(line)
            canonical_id = row.get("id")
            if canonical_id in wanted:
                # Extract only the competition-source media digest.
                source_hashes[canonical_id] = row["metadata"]["decode"]["source_sha256"]
    for target in TARGETS:
        if source_hashes.get(target["canonical_id"]) != target["prior_competition_source_sha256"]:
            raise ValueError(f"Competition-source SHA changed for {target['canonical_id']}")


def quota_observation(response: requests.Response) -> dict:
    sample = next(response.iter_content(4096), b"")
    lower = sample.lower()
    return {
        "diagnostic_body_bytes": len(sample),
        "diagnostic_prefix_sha256": sha256_bytes(sample),
        "html_prefix": lower.lstrip().startswith((b"<!doctype html", b"<html")),
        "quota_phrase_observed": any(
            phrase in lower for phrase in (b"quota exceeded", b"download quota", b"too many users")
        ),
    }


def validate_range(response: requests.Response, expected: str, expected_lm: str | None = None) -> tuple[bool, str | None]:
    actual = response.headers.get("Content-Range")
    last_modified = response.headers.get("Last-Modified")
    return (
        response.status_code == 206 and actual == expected
        and bool(last_modified) and (expected_lm is None or last_modified == expected_lm),
        last_modified,
    )


def attempt_one(target: dict) -> dict:
    started = utc_now()
    url = URL_PREFIX + urlencode({"id": target["file_id"], "export": "download", "confirm": "t"})
    record = {
        "canonical_id": target["canonical_id"],
        "mode": "train",
        "file_id": target["file_id"],
        "published_path": target["published_path"],
        "source_url": url,
        "started_at_utc": started,
        "preflight_attempts": 1,
        "full_payload_attempts": 0,
        "actual_payload_bytes": 0,
        "max_payload_bytes": MAX_BYTES,
        "timeout_seconds": {"connect": TIMEOUT[0], "read": TIMEOUT[1]},
        "prior_competition_source_sha256": target["prior_competition_source_sha256"],
        "transfer_complete": False,
        "decoded_cpu": False,
    }
    output = OUT_DIR / target["video_id"] / f"{target['clip_id']}.mp4"
    partial = output.with_suffix(output.suffix + ".part")
    if output.exists() or partial.exists():
        record.update(status="preserved_existing_recheck_output", completed_at_utc=utc_now())
        return record

    try:
        with requests.get(
            url,
            headers={"Range": "bytes=0-0", "Accept-Encoding": "identity"},
            stream=True,
            timeout=TIMEOUT,
        ) as response:
            record["preflight"] = {
                "status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "content_range": response.headers.get("Content-Range"),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
            }
            if response.status_code == 200 and "text/html" in (response.headers.get("Content-Type") or "").lower():
                record["preflight"].update(quota_observation(response))
                record["status"] = "quota_html_at_preflight" if record["preflight"]["quota_phrase_observed"] else "html_at_preflight"
                record["completed_at_utc"] = utc_now()
                return record
            match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match:
                record["status"] = "preflight_range_mismatch"
                record["completed_at_utc"] = utc_now()
                return record
            total = int(match.group(1))
            record["expected_bytes"] = total
            valid, last_modified = validate_range(response, f"bytes 0-0/{total}")
            if not valid or total < 1:
                record["status"] = "preflight_version_or_range_mismatch"
                record["completed_at_utc"] = utc_now()
                return record
            prefix = bytearray()
            for block in response.iter_content(1):
                prefix.extend(block)
                if len(prefix) > 1:
                    record["status"] = "preflight_body_exceeded_one_byte"
                    record["completed_at_utc"] = utc_now()
                    return record
            if len(prefix) != 1:
                record["status"] = "preflight_short_body"
                record["completed_at_utc"] = utc_now()
                return record
            preflight_prefix_sha = sha256_bytes(bytes(prefix))
            record["preflight"].update(
                expected_content_range=f"bytes 0-0/{total}",
                prefix_bytes_observed=1,
                prefix_sha256=preflight_prefix_sha,
                version_valid=True,
            )
            etag = response.headers.get("ETag")

        if total > MAX_BYTES:
            record["status"] = "valid_preflight_exceeds_per_file_budget"
            record["completed_at_utc"] = utc_now()
            return record

        record["full_payload_attempts"] = 1
        record["full_payload_started_at_utc"] = utc_now()
        output.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        received = 0
        full_expected = f"bytes 0-{total - 1}/{total}"
        with requests.get(
            url,
            headers={
                "Range": f"bytes=0-{total - 1}",
                "If-Range": last_modified,
                "Accept-Encoding": "identity",
            },
            stream=True,
            timeout=TIMEOUT,
        ) as response:
            full_meta = {
                "status": response.status_code,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
                "content_range": response.headers.get("Content-Range"),
                "last_modified": response.headers.get("Last-Modified"),
                "etag": response.headers.get("ETag"),
                "expected_content_range": full_expected,
            }
            record["full_payload"] = full_meta
            if response.status_code == 200 and "text/html" in (response.headers.get("Content-Type") or "").lower():
                full_meta.update(quota_observation(response))
                record["status"] = "quota_html_on_full_attempt" if full_meta["quota_phrase_observed"] else "html_on_full_attempt"
                record["completed_at_utc"] = utc_now()
                return record
            valid, full_lm = validate_range(response, full_expected, last_modified)
            if etag is not None and response.headers.get("ETag") != etag:
                valid = False
                full_meta["etag_mismatch"] = True
            if not valid:
                record["status"] = "full_payload_range_or_version_mismatch"
                record["completed_at_utc"] = utc_now()
                return record
            if response.headers.get("Content-Length") not in (None, str(total)):
                record["status"] = "full_payload_content_length_mismatch"
                record["completed_at_utc"] = utc_now()
                return record
            with partial.open("xb") as stream:
                for block in response.iter_content(CHUNK_BYTES):
                    if not block:
                        continue
                    if received + len(block) > total or received + len(block) > MAX_BYTES:
                        record["status"] = "full_payload_exceeded_declared_or_budget_bytes"
                        break
                    if received == 0 and block[:1] != prefix:
                        record["status"] = "payload_prefix_mismatch"
                        break
                    stream.write(block)
                    digest.update(block)
                    received += len(block)
            record["actual_payload_bytes"] = received
            record["partial_path"] = str(partial.relative_to(ROOT)).replace("\\", "/") if partial.exists() else None
            if received != total:
                record.setdefault("status", "full_payload_short_body")
                record["completed_at_utc"] = utc_now()
                return record

        record["sha256"] = digest.hexdigest()
        record["payload_prefix_matches_preflight"] = True
        if partial.stat().st_size != total:
            record["status"] = "full_payload_size_mismatch"
            record["completed_at_utc"] = utc_now()
            return record
        partial.replace(output)
        record.pop("partial_path", None)
        record["payload_path"] = str(output.relative_to(ROOT)).replace("\\", "/")
        record["transfer_complete"] = True
        record["sha256_matches_competition_source"] = record["sha256"] == target["prior_competition_source_sha256"]
        record["byte_identity"] = "match" if record["sha256_matches_competition_source"] else "different"

        ffmpeg = None
        try:
            import imageio_ffmpeg  # type: ignore
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
        if ffmpeg is None:
            from shutil import which
            ffmpeg = which("ffmpeg")
        if ffmpeg:
            decoded = subprocess.run(
                [ffmpeg, "-nostdin", "-v", "error", "-i", str(output), "-f", "null", "-"],
                capture_output=True,
                timeout=60,
                check=False,
            )
            record["decode"] = {
                "engine": "ffmpeg CPU default",
                "returncode": decoded.returncode,
                "stderr_sha256": sha256_bytes(decoded.stderr),
                "stderr_bytes": len(decoded.stderr),
            }
            record["decoded_cpu"] = decoded.returncode == 0
        else:
            record["decode"] = {"status": "ffmpeg_unavailable"}
        record["status"] = "complete" if record["decoded_cpu"] else "complete_decode_unavailable_or_failed"
    except requests.RequestException as exc:
        record["status"] = "request_error"
        record["error_type"] = type(exc).__name__
        if partial.exists():
            record["actual_payload_bytes"] = partial.stat().st_size
            record["partial_path"] = str(partial.relative_to(ROOT)).replace("\\", "/")
    except Exception as exc:
        record["status"] = "local_error"
        record["error_type"] = type(exc).__name__
        if partial.exists():
            record["actual_payload_bytes"] = partial.stat().st_size
            record["partial_path"] = str(partial.relative_to(ROOT)).replace("\\", "/")
    record["completed_at_utc"] = utc_now()
    return record


def main() -> None:
    check_selection_and_source()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise RuntimeError(f"Refusing to modify nonempty fresh output directory: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "report_date": "2026-10-14",
        "dataset": "CH-SIMS v2",
        "purpose": "one fresh availability recheck of two exact official TRAIN raw MP4 targets",
        "author_publication_channel": "CH-SIMS v2 author public Google Drive Raw folder via drive.usercontent.google.com",
        "selection": {
            "inventory_path": "data/simsv2_drive_listing.json",
            "metadata_path": "data/author/simsv2/meta.csv",
            "fields_used": ["mode", "video_id", "clip_id"],
            "sentiment_labels_parsed": False,
            "test_rows_selected_or_used": False,
            "selected_ids": [target["canonical_id"] for target in TARGETS],
        },
        "method": {
            "requests_per_target_max": 2,
            "preflight": "one bytes=0-0 request; require exact206 Content-Range with total and Last-Modified version",
            "full_attempt": "at most one whole-object finite Range request after valid preflight and only if total <=32MiB",
            "retries": 0,
            "workers": 1,
            "request_timeout_seconds": {"connect": TIMEOUT[0], "read": TIMEOUT[1]},
            "per_file_payload_budget_bytes": MAX_BYTES,
            "range_chunking": False,
            "accounts_cookies_or_alternate_endpoints": False,
            "gpu_or_feature_extraction": False,
        },
        "started_at_utc": utc_now(),
        "records": [],
    }
    for target in TARGETS:
        report["records"].append(attempt_one(target))
    report["completed_at_utc"] = utc_now()
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
