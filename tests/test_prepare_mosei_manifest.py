"""Standard-library tests for the CMU-MOSEI manifest audit script."""

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_mosei_manifest import ManifestError, parse_filename, prepare_mosei_manifest  # noqa: E402


SPLITS = ("train", "valid", "test")


def write_split_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("FileName", "sentiment_score"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def media_paths(split, file_name):
    root = "CMU-MOSEI/cmumosei/{}/".format(split)
    return (
        root + "video/{}.mp4".format(file_name),
        root + "audio/{}.wav".format(file_name),
        root + "text/{}.txt".format(file_name),
    )


def create_inputs(root, split_rows, include_media=True):
    labels_dir = root / "kaggle"
    labels_dir.mkdir(parents=True)
    inventory = []
    for split in SPLITS:
        rows = split_rows.get(split, [])
        write_split_csv(labels_dir / "label_file_{}.csv".format(split), rows)
        if include_media:
            for row in rows:
                inventory.extend({"name": name} for name in media_paths(split, row["FileName"]))
    inventory_path = root / "files.json"
    inventory_path.write_text(
        json.dumps({"complete": True, "files": inventory}), encoding="utf-8"
    )
    return labels_dir, inventory_path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class FileNameParsingTests(unittest.TestCase):
    def test_splits_at_final_underscore_and_keeps_youtube_underscores(self):
        self.assertEqual(
            parse_filename("5Fs_7A_V2kA_0007"),
            ("5Fs_7A_V2kA", "0007", "youtube_id_11"),
        )

    def test_recognizes_numeric_source_alias_without_calling_it_youtube(self):
        self.assertEqual(parse_filename("47472_0"), ("47472", "0", "numeric_alias"))

    def test_rejects_invalid_segment_or_source_prefix_format(self):
        for value in ("5Fs_7A_V2kA_x", "short_0", "bad/name_0", "_0", "no-separator"):
            with self.subTest(value=value):
                with self.assertRaises(ManifestError):
                    parse_filename(value)


class MoseiManifestTests(unittest.TestCase):
    def test_preserves_splits_ids_source_group_paths_and_score_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {
                "train": [{"FileName": "5Fs_7A_V2kA_0000", "sentiment_score": "-2.0"}],
                "valid": [{"FileName": "Abcdefghijk_1", "sentiment_score": "-0.5"}],
                "test": [{"FileName": "47472_0", "sentiment_score": "0.5"}],
            }
            labels_dir, inventory_path = create_inputs(root, rows)
            output_dir = root / "generated"
            summary = prepare_mosei_manifest(labels_dir, inventory_path, output_dir)
            manifest = {split: read_jsonl(output_dir / "mosei_{}_manifest.jsonl".format(split))[0] for split in SPLITS}

            self.assertEqual([summary["split_row_counts"][split] for split in SPLITS], [1, 1, 1])
            self.assertEqual(manifest["train"]["ID"], "5Fs_7A_V2kA_0000")
            self.assertEqual(manifest["train"]["source_video"], "5Fs_7A_V2kA")
            self.assertEqual(manifest["train"]["segment_id"], "0000")
            self.assertEqual(manifest["train"]["source_video_id_type"], "youtube_id_11")
            self.assertEqual(manifest["train"]["class"], "SNEG")
            self.assertEqual(manifest["valid"]["class"], "WNEG")
            self.assertEqual(manifest["test"]["class"], "WPOS")
            self.assertEqual(manifest["train"]["language"], "en")
            self.assertEqual(manifest["train"]["dataset"], "CMU-MOSEI")
            self.assertEqual(manifest["train"]["raw_label"], "-2.0")
            self.assertEqual(
                manifest["train"]["video_path"],
                "CMU-MOSEI/cmumosei/train/video/5Fs_7A_V2kA_0000.mp4",
            )
            self.assertTrue(manifest["train"]["video_present_in_inventory"])
            self.assertEqual(manifest["valid"]["source_csv_sha256"], summary["split_csv_sha256"]["valid"])
            self.assertEqual(summary["source_video_groups_cross_split"], 0)
            self.assertEqual(summary["test_ids_count"], 1)

    def test_media_gaps_are_counted_and_paths_keep_presence_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {
                "train": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "0"}],
                "valid": [],
                "test": [],
            }
            labels_dir, inventory_path = create_inputs(root, rows, include_media=False)
            summary = prepare_mosei_manifest(labels_dir, inventory_path, root / "generated")
            record = read_jsonl(root / "generated" / "mosei_train_manifest.jsonl")[0]
            self.assertEqual(
                summary["media_inventory_missing_by_split"]["train"],
                {"video": 1, "audio": 1, "text": 1},
            )
            self.assertFalse(record["video_present_in_inventory"])
            self.assertFalse(record["audio_present_in_inventory"])
            self.assertFalse(record["text_present_in_inventory"])

    def test_test_ids_are_sorted_and_hash_matches_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {
                "train": [],
                "valid": [],
                "test": [
                    {"FileName": "Abcdefghijk_1", "sentiment_score": "0.2"},
                    {"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "0.4"},
                ],
            }
            labels_dir, inventory_path = create_inputs(root, rows)
            summary = prepare_mosei_manifest(labels_dir, inventory_path, root / "generated")
            ids_path = root / "generated" / "mosei_test_ids.txt"
            id_bytes = ids_path.read_bytes()
            self.assertEqual(id_bytes.decode("utf-8"), "5Fs_7A_V2kA_0\nAbcdefghijk_1\n")
            self.assertEqual(summary["test_ids_sha256"], hashlib.sha256(id_bytes).hexdigest())
            self.assertNotIn("class_counts", summary)
            self.assertNotIn("test_class_distribution", summary)

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {
                "train": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "0"}],
                "valid": [],
                "test": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "1"}],
            }
            labels_dir, inventory_path = create_inputs(root, rows)
            with self.assertRaisesRegex(ManifestError, "overlaps splits"):
                prepare_mosei_manifest(labels_dir, inventory_path, root / "generated")

    def test_source_video_cross_split_overlap_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {
                "train": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "0"}],
                "valid": [],
                "test": [{"FileName": "5Fs_7A_V2kA_1", "sentiment_score": "1"}],
            }
            labels_dir, inventory_path = create_inputs(root, rows)
            with self.assertRaisesRegex(ManifestError, "source-video groups overlap"):
                prepare_mosei_manifest(labels_dir, inventory_path, root / "generated")

    def test_out_of_range_sentiment_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {
                "train": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "3.1"}],
                "valid": [],
                "test": [],
            }
            labels_dir, inventory_path = create_inputs(root, rows)
            with self.assertRaisesRegex(ManifestError, "unmappable sentiment_score"):
                prepare_mosei_manifest(labels_dir, inventory_path, root / "generated")

    def test_duplicate_inventory_paths_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = {"train": [], "valid": [], "test": []}
            labels_dir, inventory_path = create_inputs(root, rows)
            inventory_path.write_text(
                json.dumps({"complete": True, "files": [{"name": "same"}, {"name": "same"}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "duplicate paths"):
                prepare_mosei_manifest(labels_dir, inventory_path, root / "generated")


if __name__ == "__main__":
    unittest.main()
