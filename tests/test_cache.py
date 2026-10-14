"""NumPy-only fixture tests; these contain no official or experimental data."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import tempfile
import unittest
import unicodedata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevaffect.cache import (  # noqa: E402
    CacheNotReadyError,
    CachePreparationError,
    convert_source_pickle,
    load_cached_split,
)


class FeatureCacheFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(__file__).resolve().parents[1] / ".test-tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.root = Path(self.temp.name)
        self.loaded_splits = []

    def tearDown(self) -> None:
        for loaded in self.loaded_splits:
            for value in loaded.values():
                if isinstance(value, np.memmap) and getattr(value, "_mmap", None) is not None:
                    value._mmap.close()
        self.temp.cleanup()

    def _load(self, *args, **kwargs):
        loaded = load_cached_split(*args, **kwargs)
        self.loaded_splits.append(loaded)
        return loaded

    @staticmethod
    def _source(*, with_context: bool = True, with_lengths: bool = True):
        result = {}
        split_specs = {
            "train": ("video_01", "0001", -1.0, "Hi  THERE "),
            "valid": ("video_02", "0002", -0.4, "valid utterance"),
            "test": ("video_03", "0003", -0.1, "unknown test utterance"),
        }
        for split, (video, clip, label, raw_text) in split_specs.items():
            count, steps = 1, 4
            token_bundle = np.zeros((count, 3, steps), dtype=np.float32)
            token_bundle[:, 0, :] = [[11, 12, 13, 0]]
            token_bundle[:, 1, :] = [[1, 1, 1, 0]]
            token_bundle[:, 2, :] = [[0, 0, 0, 0]]
            audio = np.zeros((count, 4, 2), dtype=np.float32)
            vision = np.zeros((count, 4, 3), dtype=np.float32)
            audio[:, :2, :] = 0.25
            vision[:, :3, :] = 0.5
            record = {
                "id": [video + "$_$" + clip],
                "text_bert": token_bundle,
                "raw_text": [raw_text],
                "audio": audio,
                "vision": vision,
                "regression_labels": np.asarray([label], dtype=np.float32),
            }
            if with_context:
                record["text"] = np.ones((count, steps, 768), dtype=np.float32)
            if with_lengths:
                record["audio_lengths"] = np.asarray([2], dtype=np.int16)
                record["vision_lengths"] = np.asarray([3], dtype=np.int16)
            result[split] = record
        return result

    @staticmethod
    def _manifest_rows(unresolved_test: bool = True):
        rows = []
        values = (
            ("video_01/0001", "train", "-1.0", "SNEG", "mapped", "Hi  THERE "),
            ("video_02/0002", "valid", "-0.4", "WNEG", "mapped", "valid utterance"),
            (
                "video_03/0003",
                "test",
                "-0.1",
                None if unresolved_test else "WNEG",
                "unresolved" if unresolved_test else "mapped",
                "unknown test utterance",
            ),
        )
        for sample_id, split, raw_label, mapped, status, raw_text in values:
            normalized = " ".join(unicodedata.normalize("NFKC", raw_text).casefold().split())
            rows.append(
                {
                    "ID": sample_id,
                    "mode": split,
                    "raw_label": raw_label,
                    "Label": mapped,
                    "label_status": status,
                    "transcript_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                }
            )
        return rows

    def _write_inputs(self, source, manifest_rows=None):
        source_path = self.root / "synthetic_source.pkl"
        with source_path.open("wb") as handle:
            pickle.dump(source, handle, protocol=pickle.HIGHEST_PROTOCOL)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        manifest_path = self.root / "synthetic_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in manifest_rows or self._manifest_rows():
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return source_path, source_hash, manifest_path

    def _convert(self, source, *, manifest_rows=None, infer=False, expected_text_dim=768):
        source_path, source_hash, manifest_path = self._write_inputs(source, manifest_rows)
        output = self.root / "cache"
        summary = convert_source_pickle(
            source_path,
            output,
            expected_sha256=source_hash,
            trust_source_pickle=True,
            manifest_path=manifest_path,
            source_id_format="video_dollar_clip",
            manifest_id_format="video_clip_path",
            expected_text_dim=expected_text_dim,
            source_reference="synthetic fixture only",
            infer_zero_padding_lengths=infer,
        )
        return output, summary

    def test_context_sequence_is_ready_and_unknown_test_stays_unknown(self):
        output, summary = self._convert(self._source())
        self.assertEqual(summary["status"], "ready")
        self.assertTrue(summary["text_features_ready"])
        self.assertTrue(summary["raw_text_available"])

        train = self._load(output, "train", verify_artifact_hashes=True)
        self.assertEqual(train["text"].shape, (1, 4, 768))
        self.assertIsInstance(train["text"], np.memmap)
        self.assertEqual(train["text_lengths"].tolist(), [3])
        self.assertEqual(train["audio_lengths"].tolist(), [2])
        self.assertEqual(train["vision_lengths"].tolist(), [3])
        self.assertEqual(train["canonical_ids"], ["video_01/0001"])
        self.assertEqual(train["labels"].tolist(), [0])
        self.assertEqual(train["raw_labels"].tolist(), [-1.0])
        self.assertTrue(train["label_valid"].tolist()[0])
        self.assertEqual(train["raw_text"], ["Hi  THERE "])
        self.assertEqual(train["raw_text_hash_audit"][0]["status"], "match")
        self.assertEqual(train["ids"], ["video_01/0001"])
        self.assertEqual(train["source_ids"], ["video_01$_$0001"])

        valid = self._load(output, "valid")
        self.assertEqual(valid["labels"].tolist(), [1])
        self.assertAlmostEqual(valid["raw_labels"][0], float(np.float32(-0.4)))

        test = self._load(output, "test")
        self.assertEqual(test["labels"][0], -1)
        self.assertFalse(test["label_valid"][0])
        self.assertAlmostEqual(test["raw_labels"][0], float(np.float32(-0.1)))
        self.assertEqual(test["manifest_labels"][0]["label_status"], "unresolved")

        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        contract = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "ready")
        self.assertEqual(contract["source_sha256"], summary["source_sha256"])
        self.assertTrue(contract["text_features_ready"])
        self.assertEqual(contract["feature_dims"], {"text": 768, "audio": 2, "vision": 3})
        self.assertEqual(contract["sequence_lengths"], {"text": 4, "audio": 4, "vision": 4})
        self.assertEqual(
            contract["split_paths"],
            {"train": "splits/train", "valid": "splits/valid", "test": "splits/test"},
        )
        self.assertTrue(contract["raw_text_available"])
        self.assertEqual(
            json.loads((output / "splits" / "train" / "ids.json").read_text(encoding="utf-8")),
            ["video_01/0001"],
        )
        hash_lines = (output / "metadata.sha256").read_text(encoding="ascii").splitlines()
        self.assertEqual({line.split()[1] for line in hash_lines}, {"metadata.json", "meta.json"})
        self.assertEqual(contract["audit_metadata_sha256"], summary["metadata_sha256"])
        self.assertEqual(
            metadata["splits"]["train"]["lengths"]["text"]["source"],
            "text_bert attention_mask",
        )

    def test_token_triplet_is_never_written_as_text_features(self):
        source = self._source(with_context=False)
        output, summary = self._convert(source)
        self.assertEqual(summary["status"], "embedding-stage-required")
        self.assertFalse(summary["text_features_ready"])
        self.assertFalse((output / "splits" / "train" / "text.npy").exists())
        self.assertTrue((output / "splits" / "train" / "text_bert.npy").exists())
        with self.assertRaises(CacheNotReadyError):
            load_cached_split(output, "train")
        token_cache = self._load(output, "train", require_text_features=False)
        self.assertEqual(token_cache["text_bert"].shape, (1, 3, 4))
        self.assertIsNone(token_cache["feature_dims"]["text"])
        self.assertIsNone(token_cache["sequence_lengths"]["text"])

    def test_channel_first_three_channel_text_is_not_accepted_as_context(self):
        source = self._source(with_context=False)
        for split in source:
            source[split]["text"] = source[split]["text_bert"].copy()
        output, summary = self._convert(source, expected_text_dim=4)
        self.assertEqual(summary["status"], "embedding-stage-required")
        self.assertFalse((output / "splits" / "train" / "text.npy").exists())
        self.assertTrue((output / "splits" / "train" / "text_bert.npy").exists())

    def test_train_or_valid_unresolved_manifest_label_is_rejected(self):
        source = self._source()
        rows = self._manifest_rows()
        rows[0]["Label"] = None
        rows[0]["label_status"] = "unresolved"
        source_path, source_hash, manifest_path = self._write_inputs(source, rows)
        output = self.root / "should_not_exist"
        with self.assertRaisesRegex(CachePreparationError, "unresolved.*train"):
            convert_source_pickle(
                source_path,
                output,
                expected_sha256=source_hash,
                trust_source_pickle=True,
                manifest_path=manifest_path,
                source_id_format="video_dollar_clip",
            )
        self.assertFalse(output.exists())

    def test_missing_test_target_remains_unknown_when_manifest_unresolved(self):
        source = self._source()
        source["test"].pop("regression_labels")
        output, _summary = self._convert(source)
        test = self._load(output, "test")
        self.assertEqual(test["labels"][0], -1)
        self.assertFalse(test["label_valid"][0])
        self.assertTrue(np.isnan(test["raw_labels"][0]))
        self.assertIsNone(test["raw_labels_json"][0])

    def test_raw_text_hash_mismatch_is_recorded_without_modifying_text(self):
        source = self._source()
        source["train"]["raw_text"] = ["Changed text must remain intact"]
        output, summary = self._convert(source)
        train = self._load(output, "train")
        self.assertEqual(train["raw_text"], ["Changed text must remain intact"])
        self.assertEqual(train["raw_text_hash_audit"][0]["status"], "mismatch")
        self.assertTrue(summary["raw_text_available"])

    def test_absent_raw_text_cannot_claim_gpt_input_available(self):
        source = self._source()
        source["test"].pop("raw_text")
        output, summary = self._convert(source)
        self.assertFalse(summary["raw_text_available"])
        contract = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        self.assertFalse(contract["raw_text_available"])
        test = self._load(output, "test")
        self.assertFalse(test["split_raw_text_available"])
        self.assertNotIn("raw_text", test)
        with self.assertRaises(CacheNotReadyError):
            load_cached_split(output, "test", require_raw_text=True)

    def test_source_split_and_manifest_ids_must_match(self):
        source = self._source()
        rows = self._manifest_rows()
        rows[1]["ID"] = "video_99/0099"
        source_path, source_hash, manifest_path = self._write_inputs(source, rows)
        with self.assertRaisesRegex(CachePreparationError, "missing from the author manifest"):
            convert_source_pickle(
                source_path,
                self.root / "cache",
                expected_sha256=source_hash,
                trust_source_pickle=True,
                manifest_path=manifest_path,
                source_id_format="video_dollar_clip",
            )

    def test_fixed_train_valid_feature_shapes_must_match(self):
        source = self._source()
        source["valid"]["audio"] = np.pad(
            source["valid"]["audio"], ((0, 0), (0, 1), (0, 0))
        )
        source_path, source_hash, manifest_path = self._write_inputs(source)
        output = self.root / "bad-shape-cache"
        with self.assertRaisesRegex(CachePreparationError, "fixed audio shape mismatch"):
            convert_source_pickle(
                source_path,
                output,
                expected_sha256=source_hash,
                trust_source_pickle=True,
                manifest_path=manifest_path,
                source_id_format="video_dollar_clip",
            )
        self.assertFalse(output.exists())

    def test_pickle_requires_explicit_trust_and_matching_hash_before_load(self):
        source_path = self.root / "not_a_pickle.pkl"
        source_path.write_bytes(b"not a pickle")
        manifest_path = self.root / "manifest.jsonl"
        manifest_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(CachePreparationError, "trust-source-pickle"):
            convert_source_pickle(
                source_path,
                self.root / "cache1",
                expected_sha256="0" * 64,
                trust_source_pickle=False,
                manifest_path=manifest_path,
            )
        with self.assertRaisesRegex(CachePreparationError, "SHA-256 mismatch"):
            convert_source_pickle(
                source_path,
                self.root / "cache2",
                expected_sha256="0" * 64,
                trust_source_pickle=True,
                manifest_path=manifest_path,
            )

    def test_zero_tail_length_inference_is_opt_in_and_recorded(self):
        source = self._source(with_context=True, with_lengths=False)
        output, _summary = self._convert(source, infer=True)
        loaded = self._load(output, "train")
        self.assertEqual(loaded["audio_lengths"].tolist(), [2])
        self.assertEqual(loaded["vision_lengths"].tolist(), [3])
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["splits"]["train"]["lengths"]["audio"]["source"],
            "inferred_trailing_all_zero_rows",
        )
        self.assertIn(
            "valid all-zero frames",
            metadata["splits"]["train"]["lengths"]["audio"]["risk"],
        )

    def test_nonfinite_features_are_zeroed_and_counted(self):
        source = self._source()
        source["train"]["audio"][0, 0, 0] = np.nan
        output, _summary = self._convert(source)
        loaded = self._load(output, "train")
        self.assertEqual(float(loaded["audio"][0, 0, 0]), 0.0)
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(
            metadata["splits"]["train"]["feature_arrays"]["audio"][
                "nonfinite_replaced_with_zero"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
