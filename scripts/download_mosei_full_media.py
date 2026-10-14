"""Resume only official MOSEI TRAIN/VALID media to local disk.

Downloads 18,198 separate WAVs and the 17,949 actually present MP4s from the
authenticated competition inventory. It never requests test assets or label
files. Completed files are SHA-256 indexed append-only for interruption reuse.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
from pathlib import Path, PurePosixPath

import requests

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mosei/cloud_input_records.jsonl"
OUT = ROOT / "data/mosei/full_media"
INDEX = OUT / "download_index.jsonl"
COMPETITION = "msa-challenge-at-the-4th-pazhou-ai-competition"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-new-files-this-call", type=int)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.workers > 8 or (args.max_new_files_this_call is not None and args.max_new_files_this_call < 1):
        raise ValueError("Download concurrency or limit invalid")
    source_sha = sha(SOURCE)
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 18198 or any(row["split"] not in ("train", "valid") for row in rows):
        raise ValueError("Frozen label-free source cohort changed")
    jobs = {}
    for row in rows:
        jobs[row["audio_relative_path"]] = row["audio_expected_bytes"]
        if row["video_present"]:
            jobs[row["video_relative_path"]] = row["video_expected_bytes"]
    if len(jobs) != 36147:
        raise ValueError("Expected 36,147 train/valid WAV/MP4 members")
    OUT.mkdir(parents=True, exist_ok=True)
    completed = {}
    if INDEX.is_file():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            if line:
                item = json.loads(line)
                name = item["relative"]
                if name not in jobs or name in completed or item["source_manifest_sha256"] != source_sha:
                    raise ValueError("Local download index changed source identity")
                completed[name] = item
    remaining = []
    for relative, expected_bytes in jobs.items():
        posix = PurePosixPath(relative)
        if posix.is_absolute() or ".." in posix.parts or "\\" in relative:
            raise ValueError("Unsafe competition media path")
        path = (OUT / Path(*posix.parts)).resolve()
        path.relative_to(OUT.resolve())
        old = completed.get(relative)
        if old:
            if path.is_file() and path.stat().st_size == expected_bytes and sha(path) == old["sha256"]:
                continue
            raise ValueError("Indexed local media changed: " + relative)
        remaining.append((relative, expected_bytes, path))
    if args.max_new_files_this_call is not None:
        remaining = remaining[:args.max_new_files_this_call]
    token_path = Path.home() / ".kaggle/access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_path.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if not api.config_values.get("username"):
        raise RuntimeError("Kaggle authentication has no account identity")
    original_send = requests.Session.send
    def bounded(self, request, **kwargs):
        kwargs["timeout"] = (20, 60)
        return original_send(self, request, **kwargs)
    requests.Session.send = bounded
    lock = threading.Lock()
    def fetch(job):
        relative, expected_bytes, target = job
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                api.competition_download_file(COMPETITION, relative, path=str(target.parent),
                                              force=target.exists(), quiet=True)
                if not target.is_file() or target.stat().st_size != expected_bytes:
                    raise RuntimeError("Official inventory size mismatch")
                return {"relative": relative, "bytes": expected_bytes, "sha256": sha(target),
                        "source_manifest_sha256": source_sha}
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == 4:
                    return {"relative": relative, "error_type": type(exc).__name__}
        raise AssertionError("unreachable")
    new_count = 0
    errors = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, job): job for job in remaining}
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            with lock:
                if "sha256" in item:
                    with INDEX.open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    completed[item["relative"]] = item
                    new_count += 1
                else:
                    with (OUT / "download_errors.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    errors += 1
                if (new_count + errors) % 25 == 0 or new_count + errors == len(remaining):
                    print(json.dumps({"new_verified": new_count, "errors": errors,
                                      "completed_total": len(completed), "expected_total": len(jobs)}), flush=True)
    summary = {"source_manifest_sha256": source_sha, "expected_total": len(jobs),
               "completed_total": len(completed), "new_verified": new_count,
               "errors_this_call": errors, "complete": len(completed) == len(jobs),
               "test_or_labels_downloaded": False}
    (OUT / "download_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
