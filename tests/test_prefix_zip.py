"""Synthetic ZIP64 prefix fixtures; no Kaggle or user archive is read here."""

from __future__ import annotations

import binascii
import hashlib
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevaffect.prefix_zip import extract_verified_prefix  # noqa: E402


def raw_deflate(payload: bytes) -> bytes:
    compressor = zlib.compressobj(level=6, wbits=-15)
    return compressor.compress(payload) + compressor.flush()


def local_member(
    name: str,
    payload: bytes,
    *,
    method: int = 8,
    flags: int = 0x0800,
    crc_override: int | None = None,
    compressed_override: int | None = None,
    uncompressed_override: int | None = None,
) -> bytes:
    name_bytes = name.encode("utf-8")
    if method == 8:
        compressed = raw_deflate(payload)
    elif method == 0:
        compressed = payload
    else:
        compressed = payload
    crc = binascii.crc32(payload) & 0xFFFFFFFF
    if crc_override is not None:
        crc = crc_override
    compressed_size = len(compressed) if compressed_override is None else compressed_override
    uncompressed_size = len(payload) if uncompressed_override is None else uncompressed_override
    # Local Zip64 extra field: header ID 0x0001, payload length 16, then
    # uncompressed and compressed sizes in the APPNOTE-specified order.
    extra_payload = struct.pack("<QQ", uncompressed_size, compressed_size)
    extra = struct.pack("<HH", 0x0001, len(extra_payload)) + extra_payload
    header = struct.pack(
        "<IHHHHHIIIHH",
        0x04034B50,
        45,
        flags,
        method,
        0,
        0,
        crc,
        0xFFFFFFFF,
        0xFFFFFFFF,
        len(name_bytes),
        len(extra),
    )
    return header + name_bytes + extra + compressed


class PrefixZipFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / ".test-tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_inputs(self, archive_bytes: bytes, inventory_entries: list[dict]):
        archive_path = self.root / "prefix.zip"
        archive_path.write_bytes(archive_bytes)
        inventory_path = self.root / "files.json"
        inventory_path.write_text(
            json.dumps({"files": inventory_entries}), encoding="utf-8"
        )
        return archive_path, inventory_path

    def test_complete_zip64_local_member_verifies_crc_size_sha_and_prefix(self):
        name = "CH-SIMSv2/ch-simsv2s/Raw/video_01/0001.mp4"
        payload = b"synthetic clip bytes" * 100
        archive_bytes = local_member(name, payload)
        archive, inventory = self.write_inputs(
            archive_bytes,
            [{"name": name, "totalBytes": len(payload)}],
        )
        output = self.root / "extracted"
        manifest = extract_verified_prefix(
            archive,
            inventory,
            output,
            max_members=10,
            max_total_uncompressed_bytes=100_000,
            max_member_uncompressed_bytes=100_000,
        )
        extracted = output.joinpath(*name.split("/"))
        self.assertEqual(extracted.read_bytes(), payload)
        self.assertFalse(manifest["archive_complete"])
        self.assertEqual(manifest["scan"]["verified_file_members"], 1)
        record = manifest["members"][0]
        self.assertEqual(record["source_path"], name)
        self.assertEqual(record["local_header_offset"], 0)
        self.assertEqual(record["uncompressed_size"], len(payload))
        self.assertEqual(record["crc32_expected"], record["crc32_verified"])
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(manifest["source_archive"]["archive_prefix_size_bytes"], len(archive_bytes))
        self.assertEqual(
            manifest["source_archive"]["archive_prefix_sha256"],
            hashlib.sha256(archive_bytes).hexdigest(),
        )

    def test_stops_before_first_incomplete_member_and_keeps_verified_prefix(self):
        first_name = "CH-SIMSv2/ch-simsv2s/Raw/video_01/0001.mp4"
        second_name = "CH-SIMSv2/ch-simsv2s/Raw/video_01/0002.mp4"
        first = local_member(first_name, b"good clip" * 80)
        full_second = local_member(second_name, b"second clip" * 80)
        # Preserve the complete second local header, but cut its compressed body.
        fixed = struct.unpack_from("<IHHHHHIIIHH", full_second, 0)
        name_length, extra_length = fixed[-2:]
        data_start = 30 + name_length + extra_length
        truncated_second = full_second[: data_start + 3]
        archive_bytes = first + truncated_second
        inventory = [
            {"name": first_name, "totalBytes": len(b"good clip" * 80)},
            {"name": second_name, "totalBytes": len(b"second clip" * 80)},
        ]
        archive, inventory_path = self.write_inputs(archive_bytes, inventory)
        output = self.root / "partial"
        manifest = extract_verified_prefix(archive, inventory_path, output)
        self.assertEqual(manifest["scan"]["stop_reason"], "incomplete_member_payload")
        self.assertEqual(manifest["scan"]["stopped_member"], second_name)
        self.assertFalse(manifest["archive_complete"])
        self.assertEqual(manifest["scan"]["verified_file_members"], 1)
        self.assertTrue(output.joinpath(*first_name.split("/")).is_file())
        self.assertFalse(output.joinpath(*second_name.split("/")).exists())
        self.assertEqual(manifest["source_archive"]["archive_prefix_size_bytes"], len(first))
        self.assertEqual(
            manifest["source_archive"]["archive_prefix_sha256"], hashlib.sha256(first).hexdigest()
        )

    def test_crc_mismatch_never_publishes_corrupt_member(self):
        name = "clips/bad.mp4"
        payload = b"bad crc fixture"
        wrong_crc = (binascii.crc32(payload) + 1) & 0xFFFFFFFF
        archive, inventory = self.write_inputs(
            local_member(name, payload, crc_override=wrong_crc),
            [{"name": name, "totalBytes": len(payload)}],
        )
        output = self.root / "crc"
        manifest = extract_verified_prefix(archive, inventory, output)
        self.assertEqual(manifest["scan"]["stop_reason"], "rejected_crc_mismatch")
        self.assertEqual(manifest["scan"]["verified_file_members"], 0)
        self.assertFalse(output.joinpath(*name.split("/")).exists())

    def test_path_traversal_is_rejected_before_any_file_is_written(self):
        archive, inventory = self.write_inputs(
            local_member("../outside.mp4", b"unsafe"),
            [{"name": "safe.mp4", "totalBytes": 6}],
        )
        output = self.root / "traversal"
        manifest = extract_verified_prefix(archive, inventory, output)
        self.assertEqual(manifest["scan"]["stop_reason"], "rejected_unsafe_path")
        self.assertFalse((self.root / "outside.mp4").exists())
        self.assertEqual(manifest["scan"]["verified_file_members"], 0)

    def test_encryption_descriptor_and_unknown_compression_are_rejected(self):
        cases = (
            ("encrypted.mp4", 0x0801, 8, "rejected_encrypted_member"),
            ("descriptor.mp4", 0x0808, 8, "rejected_data_descriptor"),
            ("unsupported.mp4", 0x0800, 99, "rejected_unsupported_method"),
        )
        for name, flags, method, expected_reason in cases:
            with self.subTest(name=name):
                archive, inventory = self.write_inputs(
                    local_member(name, b"fixture", flags=flags, method=method),
                    [{"name": name, "totalBytes": len(b"fixture")}],
                )
                output = self.root / ("reject-" + name.replace(".", "-"))
                manifest = extract_verified_prefix(archive, inventory, output)
                self.assertEqual(manifest["scan"]["stop_reason"], expected_reason)
                self.assertEqual(manifest["scan"]["verified_file_members"], 0)

    def test_inventory_size_mismatch_is_rejected(self):
        name = "clips/mismatch.mp4"
        archive, inventory = self.write_inputs(
            local_member(name, b"payload"),
            [{"name": name, "totalBytes": 100}],
        )
        output = self.root / "inventory-mismatch"
        manifest = extract_verified_prefix(archive, inventory, output)
        self.assertEqual(manifest["scan"]["stop_reason"], "rejected_inventory_size_mismatch")
        self.assertEqual(manifest["scan"]["verified_file_members"], 0)

    def test_pilot_member_and_byte_limits_are_enforced(self):
        first_name = "clips/one.mp4"
        second_name = "clips/two.mp4"
        first_data, second_data = b"one" * 50, b"two" * 50
        archive_bytes = local_member(first_name, first_data) + local_member(second_name, second_data)
        archive, inventory = self.write_inputs(
            archive_bytes,
            [
                {"name": first_name, "totalBytes": len(first_data)},
                {"name": second_name, "totalBytes": len(second_data)},
            ],
        )
        member_limited = self.root / "member-limit"
        result = extract_verified_prefix(
            archive, inventory, member_limited, max_members=1, max_total_uncompressed_bytes=1000
        )
        self.assertEqual(result["scan"]["stop_reason"], "pilot_member_limit")
        self.assertEqual(result["scan"]["verified_file_members"], 1)

        byte_limited = self.root / "byte-limit"
        result = extract_verified_prefix(
            archive,
            inventory,
            byte_limited,
            max_members=10,
            max_total_uncompressed_bytes=len(first_data) - 1,
        )
        self.assertEqual(result["scan"]["stop_reason"], "pilot_total_byte_limit")
        self.assertEqual(result["scan"]["verified_file_members"], 0)

        per_member_limited = self.root / "per-member-limit"
        result = extract_verified_prefix(
            archive,
            inventory,
            per_member_limited,
            max_members=10,
            max_total_uncompressed_bytes=1000,
            max_member_uncompressed_bytes=len(first_data) - 1,
        )
        self.assertEqual(result["scan"]["stop_reason"], "pilot_member_byte_limit")
        self.assertEqual(result["scan"]["verified_file_members"], 0)


if __name__ == "__main__":
    unittest.main()
