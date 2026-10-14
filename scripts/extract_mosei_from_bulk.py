"""Extract only MOSEI train/valid WAV and MP4 files from the local ZIP.

The official test tree and all label CSV members are never opened. Each file
is verified against the authenticated inventory's byte count and given a
local SHA-256; completed files survive interruption via an append-only index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BULK = ROOT / "data/competition_bulk"
OUTPUT = ROOT / "data/mosei/full_media"
INDEX = OUTPUT / "extraction_index.jsonl"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-files-this-call", type=int,
                   help="optional bounded pilot/restart; omission extracts all train/valid media")
    args = p.parse_args()
    if args.max_files_this_call is not None and args.max_files_this_call < 1:
        raise ValueError("Bounded extraction count must be positive")
    provenance = json.loads((BULK / "download_provenance.json").read_text(encoding="utf-8"))
    archive_name = provenance["archive"]
    if Path(archive_name).name != archive_name or not archive_name.endswith(".zip"):
        raise ValueError("Unsafe or invalid verified competition archive name")
    archive_path = BULK / archive_name
    if archive_path.stat().st_size != provenance["bytes"] or sha(archive_path) != provenance["sha256"]:
        raise ValueError("Complete competition ZIP does not match verified provenance")
    source = ROOT / "data/mosei/cloud_input_records.jsonl"
    source_sha = sha(source)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 18198 or any(row["split"] not in ("train", "valid") for row in records):
        raise ValueError("Expected frozen label-free train/valid input manifest")
    expected = {}
    for row in records:
        expected[row["audio_relative_path"]] = row["audio_expected_bytes"]
        if row["video_present"]:
            expected[row["video_relative_path"]] = row["video_expected_bytes"]
    if len(expected) != 36147:
        raise ValueError("Expected 18,198 WAVs and 17,949 MP4s")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    completed = {}
    if INDEX.exists():
        for line in INDEX.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                relative = row["relative"]
                if relative not in expected or relative in completed or row["source_manifest_sha256"] != source_sha:
                    raise ValueError("Extraction resume index changed source identity")
                completed[relative] = row
    with zipfile.ZipFile(archive_path, "r") as archive:
        members = {}
        for info in archive.infolist():
            normalized = info.filename.replace("\\", "/").lstrip("/")
            if normalized in expected:
                key = normalized
            elif "/CMU-MOSEI/cmumosei/" in normalized:
                key = "CMU-MOSEI/cmumosei/" + normalized.split("/CMU-MOSEI/cmumosei/", 1)[1]
            else:
                continue
            if key in expected:
                if key in members:
                    raise ValueError("Duplicate media member in official archive: " + key)
                members[key] = info
        if set(members) != set(expected):
            absent = list(set(expected) - set(members))[:5]
            raise ValueError("Official archive media inventory mismatch; examples " + repr(absent))
        processed = 0
        for relative, bytes_expected in expected.items():
            parts = PurePosixPath(relative)
            if parts.is_absolute() or ".." in parts.parts or "\\" in relative:
                raise ValueError("Unsafe media path in frozen input")
            target = (OUTPUT / Path(*parts.parts)).resolve()
            target.relative_to(OUTPUT.resolve())
            old = completed.get(relative)
            if old is not None:
                if target.is_file() and target.stat().st_size == bytes_expected and sha(target) == old["sha256"]:
                    continue
                raise ValueError("Indexed local media changed or disappeared: " + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=target.parent)
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as destination, archive.open(members[relative], "r") as origin:
                    digest = hashlib.sha256()
                    size = 0
                    while chunk := origin.read(1024 * 1024):
                        destination.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    destination.flush()
                    os.fsync(destination.fileno())
                if size != bytes_expected:
                    raise ValueError("Official inventory byte count mismatch: " + relative)
                temporary.replace(target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            row = {"relative": relative, "bytes": bytes_expected, "sha256": digest.hexdigest(),
                   "source_manifest_sha256": source_sha, "official_archive_sha256": provenance["sha256"]}
            with INDEX.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            completed[relative] = row
            processed += 1
            if processed % 100 == 0 or processed == len(expected):
                print(json.dumps({"new_files": processed, "completed_total": len(completed),
                                  "expected": len(expected), "test_or_labels_opened": False}), flush=True)
            if args.max_files_this_call is not None and processed >= args.max_files_this_call:
                break
    summary = {"source_archive_sha256": provenance["sha256"],
               "source_manifest_sha256": source_sha, "expected_train_valid_media": len(expected),
               "verified_extracted": len(completed), "complete": len(completed) == len(expected),
               "test_or_labels_opened": False}
    (OUTPUT / "extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
