"""Resume the official competition ZIP locally with bounded read timeouts.

The official Kaggle client validates ETag and object length before appending.
On a transient stall, refresh the signed URL and retry the same local partial.
This does not upload any data or inspect hidden-test labels.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFilesRequest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/competition_bulk"
COMPETITION = "msa-challenge-at-the-4th-pazhou-ai-competition"
ARCHIVE = OUT / (COMPETITION + ".zip")
PARTIAL = OUT / (COMPETITION + ".zip.kaggle-partial")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    token_path = Path.home() / ".kaggle/access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_path.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    api = KaggleApi()
    api.authenticate()
    if not api.config_values.get("username"):
        raise RuntimeError("Kaggle authentication has no account identity")
    previous = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    if previous and not PARTIAL.is_file():
        raise ValueError("Nonempty local archive lacks official resume identity marker")
    attempts = 0
    complete_size = None
    started = time.perf_counter()
    while attempts < 500:
        attempts += 1
        with api.build_kaggle_client() as kaggle:
            request = ApiDownloadDataFilesRequest()
            request.competition_name = COMPETITION
            response = kaggle.competitions.competition_api_client.download_data_files(request)
            remote_size = int(response.headers["Content-Length"])
            validator = response.headers.get("ETag") or response.headers.get("Last-Modified")
            if remote_size < 10_000_000_000 or not validator or response.headers.get("Accept-Ranges") != "bytes":
                raise ValueError("Official bulk object is not large/resumable as expected")
            if PARTIAL.is_file():
                marker = json.loads(PARTIAL.read_text(encoding="utf-8"))
                if marker.get("size") != remote_size or marker.get("validator") != validator:
                    raise ValueError("Official object changed; refusing to append to old bytes")
            before = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
            if before > remote_size:
                raise ValueError("Local partial is larger than official object")
            print(json.dumps({"attempt": attempts, "bytes_present": before,
                              "remote_bytes": remote_size, "resuming": before > 0}), flush=True)
            try:
                api.download_file(response, str(ARCHIVE), kaggle.http_client(),
                                  quiet=True, resume=True, chunk_size=4 * 1024 * 1024,
                                  max_retries=2, timeout=25)
            except Exception as exc:
                after = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
                print(json.dumps({"attempt": attempts, "event": "retry_after_transfer_error",
                                  "error_type": type(exc).__name__, "bytes_present": after,
                                  "bytes_added": after - before}), flush=True)
                time.sleep(3)
                continue
            after = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
            print(json.dumps({"attempt": attempts, "event": "transfer_returned",
                              "bytes_present": after, "remote_bytes": remote_size}), flush=True)
            if after == remote_size:
                complete_size = remote_size
                break
            time.sleep(3)
    if complete_size is None:
        raise RuntimeError("Local bulk transfer exhausted bounded retry attempts")
    with ARCHIVE.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {"source": "authenticated official Kaggle competition ZIP",
              "archive": ARCHIVE.name, "bytes": complete_size, "sha256": digest,
              "resume_attempts": attempts, "wall_seconds": time.perf_counter() - started,
              "uploaded_data": False, "test_labels_read": False}
    (OUT / "download_provenance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
