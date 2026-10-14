"""Standard-library tests for the CH-SIMS v2 manifest audit script."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_simsv2_manifest import ManifestError, prepare_manifest  # noqa: E402


FIELDS = ("video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode")


def write_meta(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row(video_id, clip_id, text, label, mode):
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "text": text,
        "label": label,
        "label_T": label,
        "label_A": label,
        "label_V": label,
        "annotation": "unused-for-mapping",
        "mode": mode,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class ManifestPreparationTests(unittest.TestCase):
    def test_preserves_leading_zeroes_official_modes_and_complete_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("video_001", "0007", " Hi  THERE ", "0.2", "test"),
                    row("video_002", "0003", "你好", "-1", "train"),
                    row("video_003", "0012", "中  文", "0", "valid"),
                ),
            )

            summary = prepare_manifest(meta_csv, output_dir)
            records = read_jsonl(output_dir / "simsv2_manifest.jsonl")

            self.assertEqual([record["ID"] for record in records], [
                "video_002/0003", "video_003/0012", "video_001/0007"
            ])
            self.assertEqual([record["mode"] for record in records], ["train", "valid", "test"])
            test_record = records[-1]
            self.assertEqual(test_record["raw_relative_path"], "Raw/video_001/0007.mp4")
            self.assertEqual(test_record["language"], "zh")
            self.assertEqual(test_record["dataset"], "CH-SIMS-v2.0")
            self.assertEqual(test_record["source_video"], "video_001")
            self.assertEqual(test_record["speaker_known"], "false")
            self.assertEqual(test_record["Label"], "WPOS")
            self.assertEqual(
                test_record["transcript_sha256"],
                hashlib.sha256(b"hi there").hexdigest(),
            )
            self.assertEqual(summary["official_mode_split_counts"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(summary["speaker_leakage_status"], "unknown; no speaker ID column is available in meta.csv")

    def test_writes_fixed_sorted_test_ids_and_matching_sha256(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("video_010", "0002", "a", "0", "test"),
                    row("video_002", "0012", "b", "0.2", "test"),
                ),
            )
            summary = prepare_manifest(meta_csv, output_dir)
            ids_path = output_dir / "simsv2_test_ids.txt"
            id_bytes = ids_path.read_bytes()
            self.assertEqual(id_bytes.decode("utf-8"), "video_002/0012\nvideo_010/0002\n")
            expected_hash = hashlib.sha256(id_bytes).hexdigest()
            self.assertEqual(summary["test_ids_sha256"], expected_hash)
            self.assertIn(expected_hash, (output_dir / "simsv2_test_ids.sha256").read_text(encoding="ascii"))

    def test_invalid_label_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(meta_csv, (row("video_001", "0001", "text", "-0.1", "train"),))
            with self.assertRaises(ManifestError):
                prepare_manifest(meta_csv, output_dir, allow_unresolved_test_labels=True)
            self.assertFalse(output_dir.exists())

    def test_duplicate_sample_id_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("video_001", "0001", "first", "0", "train"),
                    row("video_001", "0001", "second", "0.2", "test"),
                ),
            )
            with self.assertRaisesRegex(ManifestError, "duplicate ID"):
                prepare_manifest(meta_csv, output_dir)
            self.assertFalse(output_dir.exists())

    def test_cross_split_repetitions_are_reported_without_deleting_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("source_1", "0001", "Same   TEXT", "0.2", "train"),
                    row("source_1", "0002", "same text", "0.4", "test"),
                    row("source_2", "0003", "same text", "-0.4", "valid"),
                ),
            )
            summary = prepare_manifest(meta_csv, output_dir)
            manifest_rows = read_jsonl(output_dir / "simsv2_manifest.jsonl")
            self.assertEqual(len(manifest_rows), 3)
            self.assertEqual(summary["source_video_groups_overlapping_splits"], 1)
            self.assertEqual(summary["normalized_transcript_groups_repeated_across_splits"], 1)
            self.assertEqual(
                summary["normalized_transcript_groups_with_cross_split_mapped_label_conflicts"],
                1,
            )

    def test_summary_does_not_publish_per_class_or_test_label_distribution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("video_001", "0001", "train", "-1", "train"),
                    row("video_002", "0002", "test", "1", "test"),
                ),
            )
            prepare_manifest(meta_csv, output_dir)
            summary_text = (output_dir / "simsv2_audit_summary.json").read_text(encoding="utf-8")
            summary = json.loads(summary_text)
            self.assertNotIn("label_distribution", summary_text.casefold())
            self.assertNotIn("label_counts", summary_text.casefold())
            self.assertNotIn("test_labels", summary)
            self.assertIn("test rows are excluded", summary["test_label_policy"])

    def test_default_preflight_freezes_test_ids_and_writes_train_valid_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("video_001", "0001", "train", "-1", "train"),
                    row("video_002", "0002", "valid", "0", "valid"),
                    row("video_003", "0003", "unresolved", "-0.1", "test"),
                ),
            )

            summary = prepare_manifest(meta_csv, output_dir)
            self.assertEqual(summary["manifest_status"], "blocked_unresolved_test_labels")
            self.assertFalse(summary["manifest_generated"])
            self.assertEqual(summary["invalid_test_label_rows"], 1)
            self.assertFalse((output_dir / "simsv2_manifest.jsonl").exists())
            with (output_dir / "simsv2_train_valid_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                train_valid_rows = list(csv.DictReader(handle))
            self.assertEqual([row["mode"] for row in train_valid_rows], ["train", "valid"])
            self.assertEqual(
                (output_dir / "simsv2_test_ids.txt").read_text(encoding="utf-8"),
                "video_003/0003\n",
            )

    def test_explicit_option_writes_null_for_unresolved_test_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            meta_csv = root / "meta.csv"
            output_dir = root / "generated"
            write_meta(
                meta_csv,
                (
                    row("video_001", "0001", "train", "-1", "train"),
                    row("video_002", "0002", "valid", "0", "valid"),
                    row("video_003", "0003", "unresolved", "-0.1", "test"),
                ),
            )

            summary = prepare_manifest(
                meta_csv,
                output_dir,
                allow_unresolved_test_labels=True,
            )
            records = read_jsonl(output_dir / "simsv2_manifest.jsonl")
            self.assertEqual(summary["manifest_status"], "contains_unresolved_test_labels")
            self.assertTrue(summary["manifest_generated"])
            self.assertEqual(records[-1]["Label"], None)
            self.assertEqual(records[-1]["label_status"], "unresolved")
            self.assertEqual(records[-1]["raw_label"], "-0.1")
            self.assertIn("not in the competition's discrete", records[-1]["label_reason"])
            self.assertTrue(all(record["Label"] is not None for record in records[:2]))


if __name__ == "__main__":
    unittest.main()
