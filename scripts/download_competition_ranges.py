"""Complete the authorized competition ZIP with parallel verified HTTP ranges.

The official client's partial file is an immutable prefix. Every subsequent
128-MiB range is fetched from the same ETag/size, checked for exact 206 range,
length and SHA-256, and stored separately. Assembly never modifies the prefix.
No data is uploaded; no test labels or archive members are read during transfer.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFilesRequest

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data/competition_bulk"
PREFIX = BULK / "msa-challenge-at-the-4th-pazhou-ai-competition.zip"
MARKER = BULK / (PREFIX.name + ".kaggle-partial")
RANGES = BULK / "ranged_transfer"
CHUNK = 128 * 1024 * 1024


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def official_response(api: KaggleApi):
    client = api.build_kaggle_client()
    client.__enter__()
    try:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = "msa-challenge-at-the-4th-pazhou-ai-competition"
        response = client.competitions.competition_api_client.download_data_files(request)
    finally:
        client.__exit__(None, None, None)
    return response


def fetch_range(spec: tuple[int, int, int], url: str, validator: str,
                total_bytes: int) -> dict:
    index, start, end = spec
    expected = end - start + 1
    path = RANGES / f"range_{index:05d}_{start}_{end}.bin"
    sidecar = path.with_suffix(".json")
    for attempt in range(1, 5):
        descriptor, temp_name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=RANGES)
        temporary = Path(temp_name)
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}",
                                            "If-Range": validator},
                              stream=True, timeout=(15, 45)) as response:
                expected_range = f"bytes {start}-{end}/{total_bytes}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Official server returned wrong byte range/status")
                current_validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
                if current_validator and current_validator != validator:
                    raise ValueError("Official object validator changed during range download")
                digest = hashlib.sha256()
                count = 0
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
                            digest.update(block)
                            count += len(block)
                    handle.flush()
                    os.fsync(handle.fileno())
            if count != expected:
                raise ValueError("Downloaded range length differs from Content-Range")
            temporary.replace(path)
            record = {"index": index, "start": start, "end": end,
                      "bytes": count, "sha256": digest.hexdigest(),
                      "validator": validator, "remote_bytes": total_bytes}
            atomic_json(sidecar, record)
            return {"index": index, "bytes": count, "attempts": attempt}
        except (requests.RequestException, OSError, ValueError) as exc:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
            if attempt == 4:
                return {"index": index, "error_type": type(exc).__name__}
            time.sleep(2)
    raise AssertionError("unreachable")


def verify_chunk(spec: tuple[int, int, int], validator: str, total: int) -> bool:
    index, start, end = spec
    path = RANGES / f"range_{index:05d}_{start}_{end}.bin"
    sidecar = path.with_suffix(".json")
    if not path.exists() and not sidecar.exists():
        return False
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("Incomplete range/sidecar pair")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if (record["index"] != index or record["start"] != start or record["end"] != end or
            record["bytes"] != end - start + 1 or record["validator"] != validator or
            record["remote_bytes"] != total or path.stat().st_size != record["bytes"] or
            sha(path) != record["sha256"]):
        raise ValueError("Stored range changed identity or SHA-256")
    return True


def assemble(specs: list[tuple[int, int, int]], total: int, validator: str,
             prefix_size: int, prefix_sha: str) -> dict:
    if PREFIX.stat().st_size != prefix_size or sha(PREFIX) != prefix_sha:
        raise ValueError("Immutable official-client prefix changed before assembly")
    target = BULK / "msa-challenge-at-the-4th-pazhou-ai-competition.complete.zip"
    if target.exists():
        raise FileExistsError("A complete archive already exists; inspect it")
    temporary = target.with_suffix(".zip.tmp")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as output:
        for source in [PREFIX] + [RANGES / f"range_{i:05d}_{start}_{end}.bin"
                                  for i, start, end in specs]:
            with source.open("rb") as handle:
                while block := handle.read(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
        output.flush()
        os.fsync(output.fileno())
    if size != total:
        raise ValueError("Assembled archive byte count mismatch")
    with zipfile.ZipFile(temporary, "r") as archive:
        members = archive.infolist()
        if not members or not any("CMU-MOSEI/cmumosei/train/audio/" in info.filename for info in members):
            raise ValueError("Assembled ZIP lacks expected official MOSEI members")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError("Assembled ZIP member CRC failed: " + corrupt)
    temporary.replace(target)
    report = {"source": "official Kaggle object, ETag-verified parallel local ranges",
              "archive": target.name, "bytes": size, "sha256": digest.hexdigest(),
              "etag": validator, "prefix_bytes": prefix_size,
              "prefix_sha256": prefix_sha, "range_chunk_bytes": CHUNK,
              "range_chunk_count": len(specs), "zip_member_count": len(members),
              "all_zip_member_crcs_verified": True,
              "uploaded_data": False, "test_labels_read": False}
    atomic_json(BULK / "download_provenance.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-chunks-this-call", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8 or (args.max_new_chunks_this_call is not None and args.max_new_chunks_this_call < 1):
        raise ValueError("Invalid concurrency or chunk limit")
    if not PREFIX.is_file() or not MARKER.is_file():
        raise FileNotFoundError("Official Kaggle partial and ETag marker are required")
    marker = json.loads(MARKER.read_text(encoding="utf-8"))
    total = int(marker["size"])
    validator = marker["validator"]
    prefix_size = PREFIX.stat().st_size
    if prefix_size <= 0 or prefix_size >= total:
        raise ValueError("Official client prefix is not partial")
    RANGES.mkdir(parents=True, exist_ok=True)
    transfer_manifest = RANGES / "transfer_manifest.json"
    if transfer_manifest.exists():
        saved = json.loads(transfer_manifest.read_text(encoding="utf-8"))
        if saved["remote_bytes"] != total or saved["validator"] != validator or saved["prefix_bytes"] != prefix_size or sha(PREFIX) != saved["prefix_sha256"]:
            raise ValueError("Immutable prefix/object changed since range transfer began")
        prefix_sha = saved["prefix_sha256"]
    else:
        prefix_sha = sha(PREFIX)
        atomic_json(transfer_manifest, {"remote_bytes": total, "validator": validator,
                                        "prefix_bytes": prefix_size, "prefix_sha256": prefix_sha,
                                        "chunk_bytes": CHUNK, "uploaded_data": False})
    specs = [(i, start, min(total - 1, start + CHUNK - 1))
             for i, start in enumerate(range(prefix_size, total, CHUNK))]
    pending = [spec for spec in specs if not verify_chunk(spec, validator, total)]
    if args.max_new_chunks_this_call is not None:
        pending = pending[:args.max_new_chunks_this_call]
    token_path = Path.home() / ".kaggle/access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_path.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    api = KaggleApi()
    api.authenticate()
    if not api.config_values.get("username"):
        raise RuntimeError("Kaggle authentication has no account identity")
    new_done = 0
    failures = 0
    started = time.perf_counter()
    for offset in range(0, len(pending), 16):
        batch = pending[offset:offset + 16]
        response = official_response(api)
        if (int(response.headers["Content-Length"]) != total or
                (response.headers.get("ETag") or response.headers.get("Last-Modified")) != validator or
                response.headers.get("Accept-Ranges") != "bytes"):
            raise ValueError("Official Kaggle object changed before range batch")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_range, spec, response.url, validator, total) for spec in batch]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if "error_type" in result:
                    failures += 1
                else:
                    new_done += 1
                print(json.dumps({"new_done": new_done, "failures": failures,
                                  "batch_offset": offset, "chunk": result,
                                  "total_chunks": len(specs),
                                  "wall_seconds": time.perf_counter() - started}), flush=True)
    complete = all(verify_chunk(spec, validator, total) for spec in specs)
    summary = {"total_chunks": len(specs), "new_done": new_done, "failures": failures,
               "complete_ranges": sum(verify_chunk(spec, validator, total) for spec in specs),
               "all_ranges_complete": complete,
               "prefix_bytes": prefix_size, "remote_bytes": total,
               "uploaded_data": False}
    atomic_json(RANGES / "progress.json", summary)
    if complete:
        summary["assembled_archive"] = assemble(specs, total, validator, prefix_size, prefix_sha)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
