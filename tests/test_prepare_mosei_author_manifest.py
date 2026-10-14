"""Tests for author-cohort preservation and the separate provisional crosswalk."""

import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
INTEGRATION_ROOT = HERE.parents[1]
PROJECT_ROOT_ENV = os.environ.get("JEVAFFECT_PROJECT_ROOT")
PROJECT_ROOT_AVAILABLE = False
if PROJECT_ROOT_ENV:
    project_root = Path(PROJECT_ROOT_ENV).expanduser().resolve()
    if (project_root / "jevaffect" / "protocol.py").is_file():
        sys.path.insert(0, str(project_root))
        PROJECT_ROOT_AVAILABLE = True
        project_scripts = project_root / "scripts"
        if (project_scripts / "prepare_mosei_manifest.py").is_file():
            sys.path.insert(0, str(project_scripts))

# When copied into <project>/tests this locates <project>/scripts and
# <project>/jevaffect. In this workspace draft it locates the shared portable
# dependencies under work/ without hardcoding a drive or user path.
for ancestor in (() if PROJECT_ROOT_AVAILABLE else HERE.parents):
    parser_dirs = (
        ancestor / "scripts",
        ancestor / "mosei_impl" / "scripts",
        ancestor / "work" / "mosei_impl" / "scripts",
    )
    for parser_dir in parser_dirs:
        if (parser_dir / "prepare_mosei_manifest.py").is_file():
            if str(parser_dir) not in sys.path:
                sys.path.insert(0, str(parser_dir))
            break
    protocol_roots = (
        ancestor,
        ancestor / "protocol_impl",
        ancestor / "work" / "protocol_impl",
    )
    for protocol_root in protocol_roots:
        if (protocol_root / "jevaffect" / "protocol.py").is_file():
            if str(protocol_root) not in sys.path:
                sys.path.insert(0, str(protocol_root))
            break

if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from scripts import prepare_mosei_author_manifest as author_manifest_script  # noqa: E402

AuthorManifestError = author_manifest_script.AuthorManifestError
prepare_mosei_author_manifest = author_manifest_script.prepare_mosei_author_manifest


class PortableImportTests(unittest.TestCase):
    def test_script_uses_project_relative_roots_and_no_drive_specific_path(self):
        source = (INTEGRATION_ROOT / "scripts" / "prepare_mosei_author_manifest.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Path(__file__).resolve().parents[1]", source)
        self.assertIn("from prepare_mosei_manifest import parse_filename", source)
        self.assertIsNone(re.search(r"\b[A-Za-z]:[/\\]", source))


AUTHOR_FIELDS = (
    "video_id",
    "clip_id",
    "text",
    "label",
    "annotation",
    "mode",
    "label_T",
    "label_A",
    "label_V",
)
SPLITS = ("train", "valid", "test")


def write_author_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUTHOR_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_competition_csvs(directory, split_rows):
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        path = directory / "label_file_{}.csv".format(split)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("FileName", "sentiment_score"),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(split_rows.get(split, []))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_row(video_id, clip_id, text, label, mode):
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "text": text,
        "label": str(label),
        "annotation": "source annotation",
        "mode": mode,
        "label_T": "",
        "label_A": "",
        "label_V": "",
    }


class AuthorCohortManifestTests(unittest.TestCase):
    def test_author_splits_ids_and_encoder_transcripts_are_isolated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            author_csv = root / "mmsa_label.csv"
            author_rows = [
                make_row("5Fs_7A_V2kA", "0009", "train text, quoted", -2.0, "train"),
                make_row("Abcdefghijk", "0002", "valid text", 0.0, "valid"),
                make_row("ViyQVpGW05w", "0004", "test-only text", 2.0, "test"),
            ]
            write_author_csv(author_csv, author_rows)
            comp_dir = root / "competition"
            write_competition_csvs(comp_dir, {
                "train": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "-2.0"}],
                "valid": [{"FileName": "Abcdefghijk_0", "sentiment_score": "0.0"}],
                "test": [{"FileName": "ViyQVpGW05w_0", "sentiment_score": "2.0"}],
            })
            output = root / "generated"

            summary = prepare_mosei_author_manifest(author_csv, comp_dir, output)
            train = read_jsonl(output / "mosei_author_train_manifest.jsonl")
            test_ids = (output / "mosei_author_test_ids.txt").read_bytes()
            with (output / "mosei_author_train_valid_encoder.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                encoder_rows = list(csv.DictReader(handle))

            self.assertEqual(summary["cohort"], "mmsa_author")
            self.assertEqual(summary["split_counts"], {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(train[0]["ID"], "5Fs_7A_V2kA_0009")
            self.assertEqual(train[0]["clip_id"], "0009")
            self.assertEqual(train[0]["author_row_number"], 1)
            self.assertEqual(test_ids.decode("utf-8"), "ViyQVpGW05w_0004\n")
            self.assertEqual(
                summary["frozen_author_ids"]["test"]["sha256"],
                hashlib.sha256(test_ids).hexdigest(),
            )
            self.assertEqual([row["mode"] for row in encoder_rows], ["train", "valid"])
            self.assertNotIn("test-only text", [row["text"] for row in encoder_rows])
            self.assertNotIn("label", encoder_rows[0])
            self.assertNotIn("annotation", encoder_rows[0])
            self.assertFalse(summary["encoder_csv_contains_labels"])
            self.assertFalse(summary["model_scoring_performed"])
            self.assertFalse(summary["test_class_distribution_emitted"])

    def test_order_crosswalk_is_provisional_and_does_not_join_on_literal_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            author_csv = root / "mmsa_label.csv"
            # Deliberately different clip IDs: only within-video source order
            # aligns with competition numeric segment order in this fixture.
            write_author_csv(author_csv, [
                make_row("5Fs_7A_V2kA", "9", "author first", -2.0, "train"),
                make_row("5Fs_7A_V2kA", "8", "author second", 1.0, "valid"),
            ])
            comp_dir = root / "competition"
            write_competition_csvs(comp_dir, {
                "train": [{"FileName": "5Fs_7A_V2kA_0", "sentiment_score": "-2.0"}],
                "valid": [{"FileName": "5Fs_7A_V2kA_1", "sentiment_score": "1.0"}],
            })
            output = root / "generated"

            summary = prepare_mosei_author_manifest(author_csv, comp_dir, output)
            pairs = read_jsonl(output / "mosei_provisional_order_crosswalk.jsonl")

            self.assertEqual(len(pairs), 2)
            self.assertEqual(pairs[0]["author_id"], "5Fs_7A_V2kA_9")
            self.assertEqual(pairs[0]["competition_id"], "5Fs_7A_V2kA_0")
            self.assertTrue(all(pair["raw_label_equal"] for pair in pairs))
            self.assertTrue(all(pair["split_equal"] for pair in pairs))
            self.assertEqual(summary["literal_id_overlap_audit"]["literal_id_intersection_count"], 0)
            self.assertFalse(summary["provisional_crosswalk"]["used_to_join_author_manifest"])
            self.assertFalse(summary["provisional_crosswalk"]["media_identity_proven"])

    def test_unequal_source_video_group_is_not_crosswalked(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            author_csv = root / "mmsa_label.csv"
            write_author_csv(author_csv, [
                make_row("Abcdefghijk", "9", "first", 1.0, "train"),
                make_row("Abcdefghijk", "8", "second", 0.0, "train"),
            ])
            comp_dir = root / "competition"
            write_competition_csvs(comp_dir, {
                "train": [{"FileName": "Abcdefghijk_0", "sentiment_score": "1.0"}],
            })
            output = root / "generated"

            summary = prepare_mosei_author_manifest(author_csv, comp_dir, output)

            self.assertEqual(summary["provisional_crosswalk"]["paired_rows"], 0)
            self.assertEqual(summary["provisional_crosswalk"]["unequal_count_source_video_groups"], 1)
            self.assertEqual(summary["provisional_crosswalk"]["unpaired_author_rows"], 2)
            self.assertEqual(summary["provisional_crosswalk"]["unpaired_competition_rows"], 1)

    def test_literal_id_label_mismatch_is_reported_but_not_used_as_join(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            author_csv = root / "mmsa_label.csv"
            write_author_csv(author_csv, [
                make_row("ViyQVpGW05w", "0", "author text", -2.0, "test"),
            ])
            comp_dir = root / "competition"
            write_competition_csvs(comp_dir, {
                "test": [{"FileName": "ViyQVpGW05w_0", "sentiment_score": "2.0"}],
            })
            output = root / "generated"

            summary = prepare_mosei_author_manifest(author_csv, comp_dir, output)
            overlap = summary["literal_id_overlap_audit"]

            self.assertEqual(overlap["literal_id_intersection_count"], 1)
            self.assertEqual(overlap["literal_id_raw_label_mismatches"], 1)
            self.assertEqual(overlap["literal_id_class_mismatches"], 1)
            self.assertFalse(overlap["literal_id_join_used"])

    def test_duplicate_author_id_bad_split_and_bad_label_are_rejected(self):
        cases = [
            [
                make_row("5Fs_7A_V2kA", "0", "a", 0.0, "train"),
                make_row("5Fs_7A_V2kA", "0", "b", 1.0, "valid"),
            ],
            [make_row("5Fs_7A_V2kA", "0", "a", 0.0, "unknown")],
            [make_row("5Fs_7A_V2kA", "0", "a", 4.0, "train")],
        ]
        for rows in cases:
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                author_csv = root / "mmsa_label.csv"
                write_author_csv(author_csv, rows)
                comp_dir = root / "competition"
                write_competition_csvs(comp_dir, {})
                with self.assertRaises(AuthorManifestError):
                    prepare_mosei_author_manifest(author_csv, comp_dir, root / "generated")


if __name__ == "__main__":
    unittest.main()
