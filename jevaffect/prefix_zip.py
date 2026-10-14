"""Safely extract complete members from a truncated ZIP local-header prefix.

This parser intentionally does not require the ZIP central directory. It reads
local file headers in order, validates Zip64 sizes against an external
``files[].name,totalBytes`` inventory, and publishes only members whose
uncompressed size and CRC-32 are verified. The archive is always marked
incomplete because the central directory/end records are not validated.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional


LOCAL_FILE_SIGNATURE = 0x04034B50
CENTRAL_FILE_SIGNATURE = 0x02014B50
END_CENTRAL_SIGNATURE = 0x06054B50
ZIP64_END_SIGNATURE = 0x06064B50
ZIP64_LOCATOR_SIGNATURE = 0x07064B50
ZIP64_EXTRA_ID = 0x0001
LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
CHUNK_BYTES = 64 * 1024
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *("COM{}".format(index) for index in range(1, 10)),
    *("LPT{}".format(index) for index in range(1, 10)),
}


class PrefixZipError(ValueError):
    """Raised for invalid inputs or archive-prefix failures before extraction."""


class _MemberStop(Exception):
    def __init__(self, reason: str, message: str, member_name: Optional[str] = None):
        super().__init__(message)
        self.reason = reason
        self.member_name = member_name


@dataclass(frozen=True)
class InventoryEntry:
    name: str
    total_bytes: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_prefix(path: Path, length: int, chunk_bytes: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    remaining = length
    with Path(path).open("rb") as handle:
        while remaining:
            block = handle.read(min(chunk_bytes, remaining))
            if not block:
                raise PrefixZipError("archive shortened while hashing the verified prefix")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _safe_name(raw_name: str, *, directory: bool = False) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise _MemberStop("rejected_unsafe_path", "empty or NUL-containing ZIP path", raw_name)
    # ZIP member names use slash separators. Backslashes are rejected to avoid
    # Windows path traversal and ambiguous archive/inventory spellings.
    if "\\" in raw_name:
        raise _MemberStop("rejected_unsafe_path", "backslash in ZIP path", raw_name)
    if raw_name.startswith("/") or re.match(r"^[A-Za-z]:", raw_name):
        raise _MemberStop("rejected_unsafe_path", "absolute/drive ZIP path", raw_name)
    path_text = raw_name[:-1] if directory and raw_name.endswith("/") else raw_name
    parts = path_text.split("/")
    if not path_text or any(part in ("", ".", "..") for part in parts):
        raise _MemberStop("rejected_unsafe_path", "dot/empty path component", raw_name)
    for part in parts:
        if any(char in part for char in '<>:"|?*'):
            raise _MemberStop("rejected_unsafe_path", "Windows-invalid path character", raw_name)
        if part.endswith((" ", ".")):
            raise _MemberStop("rejected_unsafe_path", "trailing dot/space in path component", raw_name)
        if part.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
            raise _MemberStop("rejected_unsafe_path", "Windows reserved device name", raw_name)
    return "/".join(parts)


def read_inventory(path: Path) -> tuple[dict[str, InventoryEntry], str]:
    path = Path(path)
    if not path.is_file():
        raise PrefixZipError("inventory file does not exist: {}".format(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrefixZipError("cannot read inventory JSON: {}".format(exc)) from exc
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, list):
        raise PrefixZipError("inventory must be an object with a files[] list")

    inventory: dict[str, InventoryEntry] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise PrefixZipError("inventory files[{}] must be an object".format(index))
        name = item.get("name")
        total_bytes = item.get("totalBytes")
        if not isinstance(name, str):
            raise PrefixZipError("inventory files[{}].name must be a string".format(index))
        safe_name = _safe_name(name)
        if safe_name != name:
            raise PrefixZipError("inventory path is not canonical: {!r}".format(name))
        if isinstance(total_bytes, bool) or not isinstance(total_bytes, int) or total_bytes < 0:
            raise PrefixZipError("inventory files[{}].totalBytes must be a non-negative integer".format(index))
        if name in inventory:
            raise PrefixZipError("duplicate inventory path: {}".format(name))
        inventory[name] = InventoryEntry(name=name, total_bytes=total_bytes)
    if not inventory:
        raise PrefixZipError("inventory contains no files")
    return inventory, sha256_file(path)


def _zip64_sizes(extra: bytes, compressed32: int, uncompressed32: int, name: str) -> tuple[int, int]:
    cursor = 0
    uncompressed64: Optional[int] = None
    compressed64: Optional[int] = None
    while cursor < len(extra):
        if len(extra) - cursor < 4:
            raise _MemberStop("rejected_malformed_extra", "truncated ZIP extra-field header", name)
        field_id, field_size = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if field_size > len(extra) - cursor:
            raise _MemberStop("rejected_malformed_extra", "truncated ZIP extra-field payload", name)
        payload = extra[cursor : cursor + field_size]
        cursor += field_size
        if field_id != ZIP64_EXTRA_ID:
            continue
        pos = 0
        if uncompressed32 == 0xFFFFFFFF:
            if len(payload) - pos < 8:
                raise _MemberStop("rejected_malformed_zip64", "missing Zip64 uncompressed size", name)
            uncompressed64 = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
        if compressed32 == 0xFFFFFFFF:
            if len(payload) - pos < 8:
                raise _MemberStop("rejected_malformed_zip64", "missing Zip64 compressed size", name)
            compressed64 = struct.unpack_from("<Q", payload, pos)[0]
            pos += 8
        break

    if uncompressed32 == 0xFFFFFFFF and uncompressed64 is None:
        raise _MemberStop("rejected_malformed_zip64", "Zip64 uncompressed size missing", name)
    if compressed32 == 0xFFFFFFFF and compressed64 is None:
        raise _MemberStop("rejected_malformed_zip64", "Zip64 compressed size missing", name)
    return (
        uncompressed64 if uncompressed64 is not None else uncompressed32,
        compressed64 if compressed64 is not None else compressed32,
    )


def _read_member_header(handle, offset: int, file_size: int) -> dict[str, Any]:
    handle.seek(offset)
    fixed = handle.read(LOCAL_HEADER.size)
    if len(fixed) < LOCAL_HEADER.size:
        raise _MemberStop("incomplete_local_header", "archive ends inside local file header")
    (
        signature,
        version_needed,
        flags,
        method,
        _mod_time,
        _mod_date,
        crc32,
        compressed32,
        uncompressed32,
        name_length,
        extra_length,
    ) = LOCAL_HEADER.unpack(fixed)
    if signature != LOCAL_FILE_SIGNATURE:
        raise _MemberStop("rejected_signature", "unexpected local-header signature")

    name_bytes = handle.read(name_length)
    extra = handle.read(extra_length)
    if len(name_bytes) != name_length or len(extra) != extra_length:
        raise _MemberStop("incomplete_local_header", "archive ends inside name/extra fields")
    try:
        raw_name = name_bytes.decode("utf-8" if flags & 0x0800 else "cp437", errors="strict")
    except UnicodeDecodeError as exc:
        raise _MemberStop("rejected_member_name_encoding", "invalid ZIP filename encoding") from exc
    is_directory = raw_name.endswith("/")
    name = _safe_name(raw_name, directory=is_directory)

    if flags & 0x0001:
        raise _MemberStop("rejected_encrypted_member", "encrypted members are not supported", name)
    if flags & 0x0008:
        raise _MemberStop("rejected_data_descriptor", "data-descriptor members are not supported", name)
    allowed_flags = 0x0800 | (0x0006 if method == 8 else 0)
    if flags & ~allowed_flags:
        raise _MemberStop("rejected_unsupported_flags", "unsupported ZIP general-purpose flags", name)
    if method not in (0, 8):
        raise _MemberStop("rejected_unsupported_method", "only stored/deflate ZIP members are supported", name)
    if version_needed > 63:
        raise _MemberStop("rejected_unsupported_version", "unsupported ZIP version-needed", name)

    uncompressed_size, compressed_size = _zip64_sizes(extra, compressed32, uncompressed32, name)
    data_offset = offset + LOCAL_HEADER.size + name_length + extra_length
    if data_offset > file_size:
        raise _MemberStop("incomplete_local_header", "member data offset exceeds archive size", name)
    return {
        "name": name,
        "raw_name": raw_name,
        "directory": is_directory,
        "version_needed": version_needed,
        "flags": flags,
        "method": method,
        "crc32": crc32,
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "data_offset": data_offset,
        "local_header_offset": offset,
    }


def _write_member_payload(
    archive_handle,
    member: dict[str, Any],
    temp_path: Path,
    *,
    max_member_bytes: Optional[int],
    total_output_bytes: int,
    max_total_uncompressed_bytes: Optional[int],
    chunk_bytes: int,
) -> dict[str, Any]:
    name = member["name"]
    expected_size = member["uncompressed_size"]
    compressed_size = member["compressed_size"]
    if max_member_bytes is not None and expected_size > max_member_bytes:
        raise _MemberStop("member_size_limit", "member exceeds max_member_uncompressed_bytes", name)
    if max_total_uncompressed_bytes is not None and total_output_bytes + expected_size > max_total_uncompressed_bytes:
        raise _MemberStop("total_size_limit", "member exceeds max_total_uncompressed_bytes", name)
    if member["method"] == 0 and compressed_size != expected_size:
        raise _MemberStop("rejected_stored_size_mismatch", "stored member compressed/uncompressed sizes differ", name)

    actual_archive_size = os.fstat(archive_handle.fileno()).st_size
    bytes_after_data = actual_archive_size - member["data_offset"]
    if compressed_size > bytes_after_data:
        raise _MemberStop("incomplete_member_payload", "compressed member data extends beyond archive prefix", name)

    output_count = 0
    running_crc = 0
    digest = hashlib.sha256()
    archive_handle.seek(member["data_offset"])

    def emit(out: bytes, target) -> None:
        nonlocal output_count, running_crc
        if not out:
            return
        if output_count + len(out) > expected_size:
            raise _MemberStop("rejected_uncompressed_size_overrun", "decompressed bytes exceed declared size", name)
        if max_member_bytes is not None and output_count + len(out) > max_member_bytes:
            raise _MemberStop("member_size_limit", "decompressed bytes exceed member limit", name)
        if (
            max_total_uncompressed_bytes is not None
            and total_output_bytes + output_count + len(out) > max_total_uncompressed_bytes
        ):
            raise _MemberStop("total_size_limit", "decompressed bytes exceed total limit", name)
        target.write(out)
        digest.update(out)
        running_crc = binascii.crc32(out, running_crc) & 0xFFFFFFFF
        output_count += len(out)

    try:
        with temp_path.open("xb") as target:
            remaining = compressed_size
            if member["method"] == 0:
                while remaining:
                    chunk = archive_handle.read(min(chunk_bytes, remaining))
                    if not chunk:
                        raise _MemberStop("incomplete_member_payload", "stored member payload is truncated", name)
                    remaining -= len(chunk)
                    emit(chunk, target)
            else:
                inflater = zlib.decompressobj(-15)
                while remaining:
                    chunk = archive_handle.read(min(chunk_bytes, remaining))
                    if not chunk:
                        raise _MemberStop("incomplete_member_payload", "deflate member payload is truncated", name)
                    remaining -= len(chunk)
                    pending = chunk
                    while pending:
                        max_output = max(1, expected_size - output_count + 1)
                        try:
                            output = inflater.decompress(pending, max_output)
                        except zlib.error as exc:
                            raise _MemberStop("rejected_corrupt_deflate", str(exc), name) from exc
                        tail = inflater.unconsumed_tail
                        if inflater.unused_data:
                            raise _MemberStop("rejected_deflate_trailing_data", "unused bytes inside compressed size", name)
                        emit(output, target)
                        if tail and tail == pending and not output:
                            raise _MemberStop("rejected_decompression_stall", "deflate decoder made no progress", name)
                        pending = tail
                    if inflater.eof and remaining:
                        raise _MemberStop("rejected_deflate_trailing_data", "compressed size includes bytes after stream end", name)
                if not inflater.eof:
                    raise _MemberStop("incomplete_deflate_stream", "deflate stream has no end marker", name)

            target.flush()
            os.fsync(target.fileno())
    except _MemberStop:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise
    except OSError:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    if output_count != expected_size:
        temp_path.unlink(missing_ok=True)
        raise _MemberStop(
            "rejected_uncompressed_size_mismatch",
            "actual uncompressed bytes {} != declared {}".format(output_count, expected_size),
            name,
        )
    actual_crc = running_crc & 0xFFFFFFFF
    if actual_crc != member["crc32"]:
        temp_path.unlink(missing_ok=True)
        raise _MemberStop(
            "rejected_crc_mismatch",
            "actual CRC-32 {:08x} != header {:08x}".format(actual_crc, member["crc32"]),
            name,
        )
    return {
        "compressed_size": compressed_size,
        "uncompressed_size": output_count,
        "crc32_expected": "{:08x}".format(member["crc32"]),
        "crc32_verified": "{:08x}".format(actual_crc),
        "sha256": digest.hexdigest(),
    }


def _read_source_stability(archive_path: Path, before: os.stat_result) -> None:
    after = archive_path.stat()
    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise PrefixZipError("source archive changed during prefix extraction")


def extract_verified_prefix(
    archive_path: Path,
    inventory_path: Path,
    output_dir: Path,
    *,
    max_members: Optional[int] = None,
    max_total_uncompressed_bytes: Optional[int] = None,
    max_member_uncompressed_bytes: Optional[int] = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict[str, Any]:
    """Extract verified local members until archive truncation, rejection, or a pilot limit.

    The destination must not exist. Each file is written to a temporary file,
    then renamed into the staging tree only after decompression, exact size,
    and CRC-32 verification succeed. The final manifest hashes the archive
    prefix ending immediately before the first unprocessed/incomplete member.
    """
    archive_path = Path(archive_path).resolve()
    inventory_source_path = Path(inventory_path).resolve()
    target = Path(output_dir).resolve()
    if not archive_path.is_file():
        raise PrefixZipError("archive does not exist: {}".format(archive_path))
    if max_members is not None and max_members <= 0:
        raise PrefixZipError("max_members must be positive")
    if max_total_uncompressed_bytes is not None and max_total_uncompressed_bytes < 0:
        raise PrefixZipError("max_total_uncompressed_bytes must be non-negative")
    if max_member_uncompressed_bytes is not None and max_member_uncompressed_bytes < 0:
        raise PrefixZipError("max_member_uncompressed_bytes must be non-negative")
    if chunk_bytes <= 0:
        raise PrefixZipError("chunk_bytes must be positive")
    if target.exists():
        raise PrefixZipError("refusing to overwrite existing output directory: {}".format(target))
    if archive_path == target or archive_path.is_relative_to(target):
        raise PrefixZipError("output directory must not contain the source archive")
    if inventory_source_path == target or inventory_source_path.is_relative_to(target):
        raise PrefixZipError("output directory must not contain the inventory file")

    inventory, inventory_hash = read_inventory(inventory_path)
    archive_stat = archive_path.stat()
    archive_size = archive_stat.st_size
    output_parent = target.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".prefix-zip-stage-", dir=str(output_parent)))

    records: list[dict[str, Any]] = []
    ignored_directories: list[dict[str, Any]] = []
    extracted_names: set[str] = set()
    total_uncompressed = 0
    offset = 0
    local_headers_seen = 0
    stop_reason = "eof_before_central_directory"
    stop_message = "archive ended without a validated central directory"
    stopped_member: Optional[str] = None
    stop_offset = archive_size
    try:
        with archive_path.open("rb") as archive_handle:
            while offset < archive_size:
                if max_members is not None and local_headers_seen >= max_members:
                    stop_reason = "pilot_member_limit"
                    stop_message = "maximum complete member count reached"
                    stop_offset = offset
                    break
                archive_handle.seek(offset)
                signature_bytes = archive_handle.read(4)
                if not signature_bytes:
                    stop_reason = "eof_before_central_directory"
                    stop_message = "archive ended without a validated central directory"
                    stop_offset = offset
                    break
                if len(signature_bytes) < 4:
                    stop_reason = "incomplete_signature"
                    stop_message = "archive ends in a partial ZIP signature"
                    stop_offset = offset
                    break
                signature = struct.unpack("<I", signature_bytes)[0]
                if signature == CENTRAL_FILE_SIGNATURE:
                    stop_reason = "central_directory_reached_unvalidated"
                    stop_message = "local members parsed; central directory is not validated"
                    stop_offset = offset
                    break
                if signature in (END_CENTRAL_SIGNATURE, ZIP64_END_SIGNATURE, ZIP64_LOCATOR_SIGNATURE):
                    stop_reason = "end_record_reached_unvalidated"
                    stop_message = "ZIP end record reached; archive completeness is not claimed"
                    stop_offset = offset
                    break
                if signature != LOCAL_FILE_SIGNATURE:
                    stop_reason = "rejected_unexpected_signature"
                    stop_message = "unexpected signature {:08x}".format(signature)
                    stop_offset = offset
                    break

                local_headers_seen += 1
                member_offset = offset
                try:
                    member = _read_member_header(archive_handle, member_offset, archive_size)
                    stopped_member = member["name"]
                    if member["directory"]:
                        if member["compressed_size"] != 0 or member["uncompressed_size"] != 0 or member["crc32"] != 0:
                            raise _MemberStop(
                                "rejected_nonempty_directory",
                                "directory entry has payload bytes",
                                member["name"],
                            )
                        ignored_directories.append(
                            {
                                "name": member["raw_name"],
                                "local_header_offset": member_offset,
                                "verified_empty": True,
                            }
                        )
                        offset = member["data_offset"]
                        stop_offset = offset
                        continue
                    if member["name"] not in inventory:
                        raise _MemberStop(
                            "rejected_unlisted_member",
                            "ZIP file is absent from inventory files[]",
                            member["name"],
                        )
                    expected_inventory_bytes = inventory[member["name"]].total_bytes
                    if member["uncompressed_size"] != expected_inventory_bytes:
                        raise _MemberStop(
                            "rejected_inventory_size_mismatch",
                            "local uncompressed size {} != inventory totalBytes {}".format(
                                member["uncompressed_size"], expected_inventory_bytes
                            ),
                            member["name"],
                        )
                    if member["name"] in extracted_names:
                        raise _MemberStop(
                            "rejected_duplicate_member",
                            "duplicate ZIP member name",
                            member["name"],
                        )
                    if max_member_uncompressed_bytes is not None and expected_inventory_bytes > max_member_uncompressed_bytes:
                        raise _MemberStop(
                            "pilot_member_byte_limit",
                            "inventory size exceeds max_member_uncompressed_bytes",
                            member["name"],
                        )
                    if (
                        max_total_uncompressed_bytes is not None
                        and total_uncompressed + expected_inventory_bytes > max_total_uncompressed_bytes
                    ):
                        raise _MemberStop(
                            "pilot_total_byte_limit",
                            "inventory size exceeds remaining total uncompressed byte budget",
                            member["name"],
                        )

                    compressed_end = member["data_offset"] + member["compressed_size"]
                    if compressed_end > archive_size:
                        raise _MemberStop(
                            "incomplete_member_payload",
                            "declared compressed payload extends beyond archive prefix",
                            member["name"],
                        )

                    member_parts = member["name"].split("/")
                    final_path = stage.joinpath(*member_parts)
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    if not final_path.resolve().is_relative_to(stage.resolve()):
                        raise _MemberStop(
                            "rejected_unsafe_output_path",
                            "resolved output path escapes extraction root",
                            member["name"],
                        )
                    temp_path = final_path.with_name(final_path.name + ".part-{:08d}".format(local_headers_seen))
                    try:
                        payload_info = _write_member_payload(
                            archive_handle,
                            member,
                            temp_path,
                            max_member_bytes=max_member_uncompressed_bytes,
                            total_output_bytes=total_uncompressed,
                            max_total_uncompressed_bytes=max_total_uncompressed_bytes,
                            chunk_bytes=chunk_bytes,
                        )
                    except _MemberStop:
                        raise
                    os.replace(temp_path, final_path)
                    records.append(
                        {
                            "name": member["name"],
                            "source_path": member["raw_name"],
                            "local_header_offset": member_offset,
                            "method": member["method"],
                            "flags": member["flags"],
                            "inventory_totalBytes": expected_inventory_bytes,
                            **payload_info,
                        }
                    )
                    extracted_names.add(member["name"])
                    total_uncompressed += payload_info["uncompressed_size"]
                    offset = compressed_end
                    stop_offset = offset
                    stopped_member = None
                except _MemberStop as exc:
                    stop_reason = exc.reason
                    stop_message = str(exc)
                    stopped_member = exc.member_name or stopped_member
                    stop_offset = member_offset
                    break

            else:
                stop_reason = "eof_before_central_directory"
                stop_message = "archive ended without a validated central directory"
                stop_offset = offset

        _read_source_stability(archive_path, archive_stat)
        prefix_hash = sha256_prefix(archive_path, stop_offset, chunk_bytes)
        _read_source_stability(archive_path, archive_stat)
        rejected = stop_reason.startswith("rejected_")
        manifest = {
            "schema_version": 1,
            "archive_complete": False,
            "status": "prefix_extracted_with_rejection" if rejected else "prefix_extracted",
            "source_archive": {
                "path_basename": archive_path.name,
                "archive_size_bytes": archive_size,
                "archive_prefix_size_bytes": stop_offset,
                "archive_prefix_sha256": prefix_hash,
                "prefix_definition": "bytes [0, archive_prefix_size_bytes), ending before stopped/unprocessed member",
            },
            "inventory": {
                "path_basename": Path(inventory_path).name,
                "sha256": inventory_hash,
                "entry_count": len(inventory),
            },
            "limits": {
                "max_members": max_members,
                "max_total_uncompressed_bytes": max_total_uncompressed_bytes,
                "max_member_uncompressed_bytes": max_member_uncompressed_bytes,
                "chunk_bytes": chunk_bytes,
            },
            "scan": {
                "local_headers_seen": local_headers_seen,
                "verified_file_members": len(records),
                "ignored_empty_directories": ignored_directories,
                "total_uncompressed_bytes": total_uncompressed,
                "inventory_files_not_extracted": len(inventory) - len(records),
                "stop_reason": stop_reason,
                "stop_message": stop_message,
                "stop_offset_bytes": stop_offset,
                "stopped_member": stopped_member,
                "central_directory_validated": False,
            },
            "members": records,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        # The manifest is written after verified members. Publish this partial
        # tree atomically; its archive_complete field always remains false.
        if target.exists():
            raise PrefixZipError("output directory appeared during extraction; refusing overwrite")
        os.replace(stage, target)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path,
                        help="files.json with files[].name and files[].totalBytes")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New workspace directory; it must not already exist")
    parser.add_argument("--until-first-incomplete", action="store_true",
                        help="Explicitly scan until incomplete/rejected data (no pilot caps; still size/CRC bounded per member)")
    parser.add_argument("--max-members", type=int, default=None,
                        help="Maximum local ZIP members (default pilot cap: 10)")
    parser.add_argument("--max-total-uncompressed-bytes", type=int, default=None,
                        help="Total verified output byte cap (default pilot cap: 64 MiB)")
    parser.add_argument("--max-member-uncompressed-bytes", type=int, default=None,
                        help="Per-file uncompressed byte cap (default pilot cap: 32 MiB)")
    parser.add_argument("--chunk-bytes", type=int, default=CHUNK_BYTES)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.until_first_incomplete and any((
        args.max_members is not None,
        args.max_total_uncompressed_bytes is not None,
        args.max_member_uncompressed_bytes is not None,
    )):
        print("prefix extraction failed: --until-first-incomplete cannot be combined with explicit pilot caps")
        return 2
    limits = (
        {"max_members": None, "max_total_uncompressed_bytes": None,
         "max_member_uncompressed_bytes": None}
        if args.until_first_incomplete
        else {
            "max_members": 10 if args.max_members is None else args.max_members,
            "max_total_uncompressed_bytes": (
                64 * 1024 * 1024 if args.max_total_uncompressed_bytes is None
                else args.max_total_uncompressed_bytes
            ),
            "max_member_uncompressed_bytes": (
                32 * 1024 * 1024 if args.max_member_uncompressed_bytes is None
                else args.max_member_uncompressed_bytes
            ),
        }
    )
    try:
        manifest = extract_verified_prefix(
            args.archive,
            args.inventory,
            args.output_dir,
            **limits,
            chunk_bytes=args.chunk_bytes,
        )
    except (PrefixZipError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print("prefix extraction failed: {}".format(exc))
        return 2
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "archive_complete": manifest["archive_complete"],
        "prefix_size_bytes": manifest["source_archive"]["archive_prefix_size_bytes"],
        "prefix_sha256": manifest["source_archive"]["archive_prefix_sha256"],
        "verified_file_members": manifest["scan"]["verified_file_members"],
        "total_uncompressed_bytes": manifest["scan"]["total_uncompressed_bytes"],
        "stop_reason": manifest["scan"]["stop_reason"],
        "stopped_member": manifest["scan"]["stopped_member"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 2 if manifest["status"] == "prefix_extracted_with_rejection" else 0

