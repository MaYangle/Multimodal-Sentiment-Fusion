"""Finite HTTP ranges with immutable preserved prefix and publisher version checks.

Some gateways reject multi-GB unbounded Drive responses. Each finite range is
bound to the observed Last-Modified, exact total length and recorded file ID.
Hashes verify cached bytes; no unpublished publisher checksum is invented.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
import requests


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--prefix", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk-mib", type=int, default=8)
    a = p.parse_args()
    if Path(a.name).name != a.name or a.name in (".", "..") or not 1 <= a.workers <= 8 or not 1 <= a.chunk_mib <= 64 or a.size < 1:
        raise ValueError("Invalid payload name or transfer budget")
    inventory = json.loads(a.inventory.read_text())
    selected = [r for r in inventory["files"] if r["path"] == a.name]
    if len(selected) != 1:
        raise ValueError("Expected one recorded publisher payload")
    file = selected[0]
    url = "https://drive.usercontent.google.com/download?" + urlencode({"id": file["id"], "export": "download", "confirm": "t"})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    target = a.output_dir / a.name
    receipt = target.with_suffix(target.suffix + ".provenance.json")
    if target.exists():
        if receipt.exists() and json.loads(receipt.read_text()).get("sha256") == sha(target) and target.stat().st_size == a.size:
            print("Already complete with matching recorded checksum", flush=True)
            return
        raise ValueError("Existing completed-looking file must be inspected, not overwritten")
    prefix_size = a.prefix.stat().st_size if a.prefix else 0
    if prefix_size > a.size:
        raise ValueError("Prefix exceeds publisher object size")
    prefix_hash = sha(a.prefix) if a.prefix else None
    parts = a.output_dir / (a.name + ".range_parts")
    parts.mkdir(exist_ok=True)
    check_size = min(prefix_size, 65536) if prefix_size else 1
    last_modified = None
    for attempt in range(3):
        try:
            with requests.get(url, headers={"Range": f"bytes=0-{check_size-1}"}, stream=True, timeout=(15, 45)) as response:
                if response.status_code in (429, 500, 502, 503, 504):
                    response.raise_for_status()
                expected = f"bytes 0-{check_size-1}/{a.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Publisher prefix range/version mismatch: " + json.dumps({"status": response.status_code,
                        "Content-Range": response.headers.get("Content-Range"), "expected": expected,
                        "Content-Type": response.headers.get("Content-Type")}))
                prefix_bytes = bytearray()
                for chunk in response.iter_content(check_size):
                    prefix_bytes.extend(chunk)
                    if len(prefix_bytes) > check_size:
                        raise ValueError("Prefix body exceeds requested finite range")
                if len(prefix_bytes) != check_size:
                    raise requests.ConnectionError("Short publisher prefix response")
                last_modified = response.headers.get("Last-Modified")
                if not last_modified:
                    raise ValueError("A publisher object version validator is required")
                if a.prefix:
                    with a.prefix.open("rb") as f:
                        if f.read(check_size) != prefix_bytes:
                            raise ValueError("Preserved prefix belongs to a different object")
            break
        except requests.RequestException:
            if attempt == 2:
                raise
    source = {"publisher": inventory["publisher"], "file_id": file["id"], "url": url,
              "size": a.size, "Last-Modified": last_modified, "prefix_size": prefix_size,
              "prefix_sha256": prefix_hash, "chunk_bytes": a.chunk_mib * 1024 * 1024}
    meta = parts / "source.json"
    if meta.exists() and json.loads(meta.read_text()) != source:
        raise ValueError("Cached range source/version/prefix differs")
    save(meta, source)
    ranges = [(start, min(start + source["chunk_bytes"], a.size) - 1) for start in range(prefix_size, a.size, source["chunk_bytes"])]

    def fetch(interval):
        start, end = interval
        block = parts / f"{start}-{end}.block"
        checksum = block.with_suffix(".sha256")
        length = end - start + 1
        if block.exists():
            if not checksum.exists() or block.stat().st_size != length or checksum.read_text().strip() != sha(block):
                raise ValueError("Cached block integrity failed")
            return length
        partial = block.with_suffix(".partial")
        for attempt in range(3):
            offset = partial.stat().st_size if partial.exists() else 0
            if not 0 <= offset <= length:
                raise ValueError("Invalid partial range size")
            if offset == length:
                checksum.write_text(sha(partial), encoding="ascii")
                partial.replace(block)
                return length
            requested = start + offset
            try:
                headers = {"Range": f"bytes={requested}-{end}", "If-Range": last_modified, "Accept-Encoding": "identity"}
                with requests.get(url, headers=headers, stream=True, timeout=(15, 45)) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        response.raise_for_status()
                    expected = f"bytes {requested}-{end}/{a.size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected or response.headers.get("Last-Modified") != last_modified:
                        # Read at most a small diagnostic prefix, never an ignored-range multi-GB body.
                        diagnostic = {"Content-Type": response.headers.get("Content-Type")}
                        try:
                            sample = next(response.iter_content(4096), b"")
                            lower = sample.lower()
                            diagnostic.update(body_prefix_bytes=len(sample), body_prefix_sha256=hashlib.sha256(sample).hexdigest(),
                                html_prefix=lower.lstrip().startswith((b"<!doctype html", b"<html")),
                                quota_phrase_observed=any(term in lower for term in (b"too many users", b"quota exceeded", b"download quota")))
                        except requests.RequestException as error:
                            diagnostic["body_observation_error"] = type(error).__name__
                        raise ValueError("Publisher range/version mismatch: " + json.dumps({"status": response.status_code,
                            "Content-Range": response.headers.get("Content-Range"), "Last-Modified": response.headers.get("Last-Modified"),
                            "expected_range": expected, "expected_version": last_modified, "diagnostic": diagnostic}))
                    written = offset
                    with partial.open("ab") as f:
                        for chunk in response.iter_content(64 * 1024):
                            if written + len(chunk) > length:
                                raise ValueError("Range body exceeds declared length")
                            f.write(chunk); written += len(chunk)
                if partial.stat().st_size != length:
                    raise requests.ConnectionError("Short finite range body")
                checksum.write_text(sha(partial), encoding="ascii")
                partial.replace(block)
                return length
            except requests.RequestException:
                if attempt == 2:
                    raise
        raise RuntimeError("Range retry limit reached")

    completed = prefix_size
    started = time.monotonic()
    report = {"source": source, "transfer_complete": False, "content_schema_validated": False,
              "publisher_checksum_available": False, "ranges_total": len(ranges), "completed_bytes": completed}
    progress = parts / "progress.json"
    save(progress, report)
    print(json.dumps({"preflight": source, "range_count": len(ranges)}), flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=a.workers)
    futures = [pool.submit(fetch, r) for r in ranges]
    try:
        for future in concurrent.futures.as_completed(futures):
            completed += future.result()
            report["completed_bytes"] = completed
            report["seconds"] = time.monotonic() - started
            save(progress, report)
            print(json.dumps({"completed_bytes": completed, "total_bytes": a.size, "seconds": round(report["seconds"], 2)}), flush=True)
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "time_utc": datetime.now(timezone.utc).isoformat()}
        save(progress, report)
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if a.prefix and sha(a.prefix) != prefix_hash:
        raise ValueError("Original immutable prefix changed during transfer")
    assembling = target.with_suffix(target.suffix + ".assembling")
    combined_hash = hashlib.sha256()
    sources = ([a.prefix] if a.prefix else []) + [parts / f"{start}-{end}.block" for start, end in ranges]
    with assembling.open("wb") as out:
        for path in sources:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    out.write(chunk); combined_hash.update(chunk)
    if assembling.stat().st_size != a.size:
        raise ValueError("Whole-payload length mismatch")
    assembling.replace(target)
    report.update(transfer_complete=True, sha256=combined_hash.hexdigest(), bytes=a.size,
                  input_form=inventory.get("input_form", "author-linked processed features; not raw video"), time_utc=datetime.now(timezone.utc).isoformat())
    save(receipt, report); save(progress, report)
    print(json.dumps({"transfer_complete": True, "bytes": a.size, "sha256": report["sha256"]}), flush=True)


if __name__ == "__main__":
    main()
