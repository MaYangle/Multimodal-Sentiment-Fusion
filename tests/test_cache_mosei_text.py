"""Metadata-only tests for the portable author MOSEI text-cache entry point."""

import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


HERE = Path(__file__).resolve()
INTEGRATION_ROOT = HERE.parents[1]
PROJECT_ROOT_ENV = os.environ.get("JEVAFFECT_PROJECT_ROOT")
PROJECT_ROOT_AVAILABLE = False
if PROJECT_ROOT_ENV:
    project_root = Path(PROJECT_ROOT_ENV).expanduser().resolve()
    if (project_root / "jevaffect" / "training.py").is_file():
        sys.path.insert(0, str(project_root))
        PROJECT_ROOT_AVAILABLE = True
for ancestor in (() if PROJECT_ROOT_AVAILABLE else HERE.parents):
    for protocol_root in (
        ancestor,
        ancestor / "protocol_impl",
        ancestor / "work" / "protocol_impl",
    ):
        if (protocol_root / "jevaffect" / "protocol.py").is_file():
            if str(protocol_root) not in sys.path:
                sys.path.insert(0, str(protocol_root))
            break
if str(INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_ROOT))

from scripts.cache_mosei_text import (  # noqa: E402
    AuthorTextCacheError,
    TextFeatureMemmapWriter,
    build_text_mlp_cache_metadata,
    load_author_train_valid,
    write_json,
    write_split_sidecars,
)


def author_row(identifier, split, text, class_name="NEUT"):
    return {
        "ID": identifier,
        "author_id": identifier,
        "dataset": "CMU-MOSEI",
        "cohort": "mmsa_author",
        "split": split,
        "text": text,
        "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "raw_label": "0.0",
        "class": class_name,
        "source_csv_sha256": "a" * 64,
    }


def write_manifest(directory, split, rows):
    path = directory / "mosei_author_{}_manifest.jsonl".format(split)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class AuthorTextManifestTests(unittest.TestCase):
    def test_reads_only_author_train_valid_and_preserves_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp)
            write_manifest(
                manifest_dir,
                "train",
                [author_row("video_with_under_0007", "train", "train transcript", "WPOS")],
            )
            write_manifest(
                manifest_dir,
                "valid",
                [author_row("another_id_3", "valid", "valid transcript", "SNEG")],
            )
            # Deliberately malformed test data. The embedding input loader must
            # not open the test manifest at all.
            (manifest_dir / "mosei_author_test_manifest.jsonl").write_text(
                "not valid JSON\n", encoding="utf-8"
            )

            records, manifest_hashes, source_hash = load_author_train_valid(manifest_dir)

        self.assertEqual(records["train"][0]["ID"], "video_with_under_0007")
        self.assertEqual(records["valid"][0]["ID"], "another_id_3")
        self.assertEqual(records["train"][0]["class"], "WPOS")
        self.assertEqual(set(manifest_hashes), {"train", "valid"})
        self.assertEqual(source_hash, "a" * 64)

    def test_rejects_transcript_hash_mismatch_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp)
            bad = author_row("video_1", "train", "some text")
            bad["transcript_sha256"] = "b" * 64
            write_manifest(manifest_dir, "train", [bad])
            write_manifest(manifest_dir, "valid", [author_row("video_2", "valid", "other")])
            with self.assertRaisesRegex(AuthorTextCacheError, "transcript hash mismatch"):
                load_author_train_valid(manifest_dir)

        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp)
            write_manifest(manifest_dir, "train", [author_row("same_1", "train", "one")])
            write_manifest(manifest_dir, "valid", [author_row("same_1", "valid", "two")])
            with self.assertRaisesRegex(AuthorTextCacheError, "duplicate author ID"):
                load_author_train_valid(manifest_dir)

    def test_rejects_competition_or_non_author_cohort_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = Path(tmp)
            row = author_row("video_1", "train", "text")
            row["cohort"] = "competition"
            write_manifest(manifest_dir, "train", [row])
            write_manifest(manifest_dir, "valid", [author_row("video_2", "valid", "text")])
            with self.assertRaisesRegex(AuthorTextCacheError, "not from the author MOSEI cohort"):
                load_author_train_valid(manifest_dir)


class PortableTextCacheTests(unittest.TestCase):
    def test_script_uses_repo_relative_imports_local_encoder_and_cpu_default(self):
        source_path = INTEGRATION_ROOT / "scripts" / "cache_mosei_text.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertIn("Path(__file__).resolve().parents[1]", source)
        self.assertIn("download_encoders.py", source)
        self.assertIn("local_files_only=True", source)
        self.assertIn('default="cpu"', source)
        self.assertIsNone(re.search(r"\b[A-Za-z]:[/\\]", source))


class TextMlpTrainingContractTests(unittest.TestCase):
    def test_metadata_contains_text_mlp_training_contract_and_combined_hash(self):
        from scripts.cache_mosei_text import combined_source_sha256

        source_hashes = {"author": "a" * 64, "encoder": "b" * 64}
        metadata = build_text_mlp_cache_metadata(
            source_hashes,
            feature_dim=768,
            max_length=64,
            split_counts={"train": 3, "valid": 2},
            encoder={"repo": "test"},
            complete=True,
        )
        self.assertEqual(metadata["source_sha256"], combined_source_sha256(source_hashes))
        self.assertEqual(metadata["split_paths"], {"train": "train", "valid": "valid"})
        self.assertEqual(metadata["sequence_lengths"], {"text": 64})
        self.assertEqual(metadata["feature_dims"], {"text": 768})
        self.assertTrue(metadata["text_features_ready"])

    def test_writer_cache_runs_actual_text_mlp_training_on_cpu(self):
        project_root = None
        if PROJECT_ROOT_ENV:
            candidate = Path(PROJECT_ROOT_ENV).expanduser().resolve()
            if (candidate / "jevaffect" / "training.py").is_file():
                project_root = candidate
        if project_root is None:
            for ancestor in HERE.parents:
                if (ancestor / "jevaffect" / "training.py").is_file():
                    project_root = ancestor
                    break
        if project_root is None:
            self.skipTest("set JEVAFFECT_PROJECT_ROOT to run the actual text_mlp CPU contract")
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import numpy as np
        from jevaffect.training import train_run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir()
            records = {
                "train": [
                    author_row("author_train_1", "train", "train text one", "SNEG"),
                    author_row("author_train_2", "train", "train text two", "WPOS"),
                    author_row("author_train_3", "train", "train text three", "NEUT"),
                    author_row("author_train_4", "train", "train text four", "SPOS"),
                ],
                "valid": [
                    author_row("author_valid_1", "valid", "valid text one", "SNEG"),
                    author_row("author_valid_2", "valid", "valid text two", "WPOS"),
                ],
            }
            source_hashes = {
                "author_csv_sha256": "a" * 64,
                "author_manifest_sha256": {"train": "b" * 64, "valid": "c" * 64},
                "encoder_sha256": "d" * 64,
                "encoder_revision": "test-only-pinned-revision",
                "cohort": "mmsa_author",
            }
            feature_dim = 4
            max_length = 3
            metadata = build_text_mlp_cache_metadata(
                source_hashes,
                feature_dim=feature_dim,
                max_length=max_length,
                split_counts={split: len(rows) for split, rows in records.items()},
                encoder={"repo": "synthetic contract fixture; no encoder inference"},
                complete=True,
            )
            for split, split_rows in records.items():
                split_dir = cache_dir / split
                split_dir.mkdir()
                write_split_sidecars(split_dir, split_rows, np)
                writer = TextFeatureMemmapWriter(
                    split_dir,
                    row_count=len(split_rows),
                    max_length=max_length,
                    hidden_size=feature_dim,
                    np_module=np,
                )
                values = np.arange(
                    len(split_rows) * max_length * feature_dim,
                    dtype=np.float32,
                ).reshape(len(split_rows), max_length, feature_dim)
                lengths = np.full(len(split_rows), max_length, dtype=np.int64)
                writer.write_batch(0, values, lengths)
                writer.close()
            write_json(cache_dir / "meta.json", metadata)

            with patch("torch.backends.cudnn.is_available", return_value=False):
                result = train_run(
                    {
                        "dataset": "CMU-MOSEI-author",
                        "language": "en",
                        "cache_dir": str(cache_dir),
                        "run_root": str(root / "runs"),
                        "seeds": [13, 42, 2026],
                        "model": {
                            "name": "text_mlp",
                            "params": {
                                "hidden_dim": 8,
                                "dropout": 0.0,
                                "pooling": "masked_mean",
                            },
                        },
                        "training": {
                            "max_epochs": 1,
                            "batch_size": 2,
                            "patience": 1,
                            "learning_rate": 0.001,
                            "device": "cpu",
                            "amp": False,
                        },
                    },
                    13,
                    root / "run",
                )
            self.assertTrue(result["completed"])
            self.assertEqual(result["last_epoch"], 1)
            self.assertTrue((root / "run" / "best.pt").is_file())
            history = [
                json.loads(line)
                for line in (root / "run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]["train_loss"]))
            self.assertTrue(np.isfinite(history[0]["valid_nll"]))


if __name__ == "__main__":
    unittest.main()
