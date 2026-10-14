import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jevaffect.anchor_calibration import normalizer, normalize
from jevaffect.native_training import NativeSplit, author_rows, digest
from scripts.prepare_native_sequence_standardization import (
    MODALITIES,
    POOLING_TOLERANCE,
    _close_dataset,
    prepare_cache,
)


ROOT = Path(__file__).resolve().parents[1]
SHAPES = {"text": (64, 3), "audio": (400, 2), "vision": (55, 4)}
LENGTHS = {
    "train": {
        "text": [3, 0, 2, 5, 1],
        "audio": [2, 4, 0, 5, 1],
        "vision": [1, 0, 4, 2, 3],
    },
    "valid": {
        "text": [2, 0, 4],
        "audio": [3, 0, 1],
        "vision": [0, 2, 4],
    },
}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _sha_manifest(root):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file() and path.name != "metadata.json"
    }


def _sequence_fixture(root):
    """Small deterministic, formally shaped raw and pooled caches for CPU tests."""
    root = Path(root)
    raw_cache = root / "raw"
    pooled_cache = root / "pooled"
    author_csv = root / "author.csv"
    raw_cache.mkdir(parents=True)
    pooled_cache.mkdir(parents=True)

    ids = {
        "train": [f"train{i}/clip{i}" for i in range(5)],
        "valid": [f"valid{i}/clip{i}" for i in range(3)],
    }
    labels = {
        "train": [-0.72, -0.31, 0.02, 0.44, 0.86],
        "valid": [-0.55, 0.13, 0.68],
    }
    with author_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_id", "clip_id", "mode", "label"])
        writer.writeheader()
        for split in ("train", "valid"):
            for sid, target in zip(ids[split], labels[split]):
                video_id, clip_id = sid.split("/", 1)
                writer.writerow({"video_id": video_id, "clip_id": clip_id, "mode": split, "label": target})
        writer.writerow({"video_id": "test-only", "clip_id": "clip", "mode": "test", "label": "must not parse this TEST label"})
    author_sha = digest(author_csv)

    raw_files = {}
    for split in ("train", "valid"):
        directory = raw_cache / "splits" / split
        directory.mkdir(parents=True)
        _write_json(directory / "ids.json", ids[split])
        _write_json(directory / "raw_labels.json", labels[split])
        for rel in (f"splits/{split}/ids.json", f"splits/{split}/raw_labels.json"):
            path = raw_cache / rel
            raw_files[rel] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        for modality in MODALITIES:
            steps, width = SHAPES[modality]
            values = np.zeros((len(ids[split]), steps, width), dtype=np.float32)
            lens = np.asarray(LENGTHS[split][modality], dtype=np.int64)
            for row, count in enumerate(lens):
                values[row] = np.float32(901.25 + row)
                for step in range(int(count)):
                    for coordinate in range(width):
                        # A nontrivial, modality-specific trend keeps TRAIN scaler coordinates finite.
                        values[row, step, coordinate] = (
                            0.19 * row + 0.07 * step + 0.11 * coordinate
                            + 0.013 * (MODALITIES.index(modality) + 1) * row * coordinate
                        )
            np.save(directory / (modality + ".npy"), values)
            np.save(directory / (modality + "_lengths.npy"), lens)
            for suffix in (modality + ".npy", modality + "_lengths.npy"):
                rel = f"splits/{split}/{suffix}"
                path = raw_cache / rel
                raw_files[rel] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        timestamp_names = ("text_times.npy", "audio_times.npy", "vision_times.npy") if split == "train" else ("audio_times.npy", "vision_times.npy")
        for name in timestamp_names:
            modality = name.removesuffix("_times.npy")
            steps = SHAPES[modality][0]
            times = np.full((len(ids[split]), steps), -812.5, dtype=np.float64)
            for row, count in enumerate(LENGTHS[split][modality]):
                times[row, :int(count)] = np.arange(int(count), dtype=np.float64) + 0.125 * row
            np.save(directory / name, times)
            relative = f"splits/{split}/{name}"
            path = raw_cache / relative
            raw_files[relative] = {"sha256": digest(path), "size_bytes": path.stat().st_size}

    pins = {
        "wavlm": {
            "repo": "microsoft/wavlm-base-plus",
            "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
            "weight_sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
        },
        "dinov2-small": {
            "repo": "facebook/dinov2-small",
            "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "weight_sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        },
    }
    preprocessing = {
        "audio_sample_rate": 16000,
        "max_seconds": 120.0,
        "mbert_max_tokens": 64,
        "video_fps": 2.0,
        "video_frame_size": 224,
        "video_frame_timestamps_kind": "ffmpeg_resampled_output_timeline",
        "audio_feature_times": "centers of WavLM convolutional receptive fields",
    }
    encoder_provenance = {
        "dimensions": {"text": 3, "audio": 2, "vision": 4},
        "encoders": pins,
    }
    descriptor = {
        "official_csv_sha256": author_sha,
        "text_condition": "provided transcripts, not ASR",
        "test_data_encoded": False,
        "base_raw_cache_descriptor": {
            "encoder_preprocessing": preprocessing,
            "truncation_policy": "keep first target steps in original order; no resampling/interpolation",
            "target_shapes": {m: list(SHAPES[m]) for m in MODALITIES},
            "encoder_provenance": encoder_provenance,
        },
        "replacement_text_encoder": {
            "model": "hfl/chinese-macbert-base",
            "revision": "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1",
            "files": {"pytorch_model.bin": {"sha256": "db0506d985574b80c33eec1cf13bd4c130585568753175871095ca13dbad9e23"}},
        },
    }
    raw_metadata = {
        "schema_version": 1,
        "files": raw_files,
        "official_csv_sha256": author_sha,
        "split_counts": {split: len(ids[split]) for split in ("train", "valid")},
        "source_descriptor": descriptor,
        "raw_bundle_encoder_provenance": encoder_provenance,
    }
    _write_json(raw_cache / "metadata.json", raw_metadata)

    pooled_vectors = {}
    for split in ("train", "valid"):
        directory = raw_cache / "splits" / split
        pooled_vectors[split] = {}
        for modality in MODALITIES:
            values = np.load(directory / (modality + ".npy"))
            lengths = np.load(directory / (modality + "_lengths.npy"))
            pooled = np.zeros((len(values), values.shape[-1]), dtype=np.float32)
            for row, count in enumerate(lengths):
                if count:
                    pooled[row] = values[row, 0] if modality == "text" else values[row, :int(count)].mean(0)
            pooled_vectors[split][modality] = pooled
    statistics = {m: normalizer(pooled_vectors["train"][m], np.arange(len(ids["train"]))) for m in MODALITIES}
    _write_json(pooled_cache / "normalization.json", statistics)

    pooled_artifacts = {"normalization.json": digest(pooled_cache / "normalization.json")}
    source_feature_hashes = {}
    for split in ("train", "valid"):
        directory = pooled_cache / split
        directory.mkdir()
        shutil.copyfile(raw_cache / "splits" / split / "ids.json", directory / "ids.json")
        shutil.copyfile(raw_cache / "splits" / split / "raw_labels.json", directory / "raw_labels.json")
        pooled_artifacts[split + "/ids.json"] = digest(directory / "ids.json")
        pooled_artifacts[split + "/raw_labels.json"] = digest(directory / "raw_labels.json")
        raw_dir = raw_cache / "splits" / split
        source_feature_hashes[split] = {"metadata.json": digest(raw_cache / "metadata.json")}
        source_feature_hashes[split]["splits/" + split + "/ids.json"] = digest(raw_dir / "ids.json")
        for modality in MODALITIES:
            values = pooled_vectors[split][modality]
            standardized = normalize(values, statistics[modality])
            present = np.asarray(LENGTHS[split][modality]) > 0
            standardized[~present] = 0
            np.save(directory / (modality + ".npy"), standardized[:, None, :])
            np.save(directory / (modality + "_lengths.npy"), present.astype(np.int64))
            for suffix in (modality + ".npy", modality + "_lengths.npy"):
                relative = split + "/" + suffix
                pooled_artifacts[relative] = digest(directory / suffix)
                source_feature_hashes[split]["splits/" + split + "/" + suffix] = digest(raw_dir / suffix)

    pooled_metadata = {
        "dataset": "simsv2",
        "input_form": "fixture pooled vectors",
        "source_form": "fixture MacBERT/WavLM/DINOv2 source",
        "source_cache": str(raw_cache.resolve()),
        "source_feature_sha256": source_feature_hashes,
        "author_csv_sha256": author_sha,
        "preparation_sha256": digest(ROOT / "scripts/prepare_native_pooled_standardization.py"),
        "normalization_implementation_sha256": digest(ROOT / "jevaffect/anchor_calibration.py"),
        "normalization": {
            "kind": "TRAIN clip-pooled coordinate mean/population STD; tiny STD<1e-6 uses1",
            "fit_split": "train",
            "fit_count": len(ids["train"]),
            "fit_ids_sha256": pooled_artifacts["train/ids.json"],
        },
        "counts": {split: len(ids[split]) for split in ("train", "valid")},
        "dimensions": {m: SHAPES[m][1] for m in MODALITIES},
        "artifact_sha256": pooled_artifacts,
        "test_exported": False,
    }
    _write_json(pooled_cache / "metadata.json", pooled_metadata)
    return raw_cache, pooled_cache, author_csv


class NativeSequenceStandardizationTests(unittest.TestCase):
    def test_valid_steps_keep_shapes_labels_lengths_and_pooling_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_cache, pooled_cache, author_csv = _sequence_fixture(root)
            output = root / "sequence-standardized"
            result = prepare_cache(raw_cache, pooled_cache, output, author_csv, allow_fixture=True)

            self.assertEqual(result["counts"], {"train": 5, "valid": 3})
            self.assertEqual(result["sequence_shapes"], {m: list(SHAPES[m]) for m in MODALITIES})
            self.assertFalse(result["test_exported"])
            self.assertFalse((output / "test").exists())
            self.assertTrue(result["padding_audit"]["extra_padding_cleanup_intervention"])
            self.assertTrue(all(result["padding_audit"]["original_nonzero_padding_entries"][m] > 0 for m in MODALITIES))
            self.assertEqual(set(result["preserved_time_artifacts"]["train"]), {"text_times.npy", "audio_times.npy", "vision_times.npy"})
            self.assertEqual(set(result["preserved_time_artifacts"]["valid"]), {"audio_times.npy", "vision_times.npy"})
            self.assertTrue(all(
                value <= POOLING_TOLERANCE
                for split_errors in result["pooling_identity"]["maximum_absolute_error"].values()
                for value in split_errors.values()
            ))

            rows = author_rows(author_csv, "simsv2")
            stats = json.loads((pooled_cache / "normalization.json").read_text(encoding="utf-8"))
            output_metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(output_metadata["normalization"]["fit_count"], 5)
            self.assertEqual(output_metadata["normalization"]["statistics_sha256"], digest(pooled_cache / "normalization.json"))
            self.assertEqual((output / "normalization.json").read_bytes(), (pooled_cache / "normalization.json").read_bytes())
            self.assertEqual(set(output_metadata["artifact_sha256"]), set(_sha_manifest(output)))
            for relative, expected in output_metadata["artifact_sha256"].items():
                self.assertEqual(digest(output / relative), expected)
            for split, artifacts in result["preserved_time_artifacts"].items():
                for name, record in artifacts.items():
                    source_time = raw_cache / record["source_relative_path"]
                    output_time = output / record["output_relative_path"]
                    self.assertEqual(source_time.read_bytes(), output_time.read_bytes())
                    self.assertEqual(digest(source_time), record["source_sha256"])
                    self.assertEqual(digest(output_time), record["output_sha256"])
                    self.assertEqual(record["output_sha256"], output_metadata["artifact_sha256"][record["output_relative_path"]])

            for split in ("train", "valid"):
                raw = NativeSplit(raw_cache, split, rows, MODALITIES, pooled=False)
                normalized_pool = NativeSplit(pooled_cache, split, rows, MODALITIES, pooled=True)
                normalized_sequence = NativeSplit(output, split, rows, MODALITIES, pooled=True)
                self.assertEqual((raw_cache / "splits" / split / "ids.json").read_bytes(), (output / split / "ids.json").read_bytes())
                self.assertEqual((raw_cache / "splits" / split / "raw_labels.json").read_bytes(), (output / split / "raw_labels.json").read_bytes())
                self.assertEqual(raw.ids, normalized_sequence.ids)
                np.testing.assert_array_equal(raw.targets, normalized_sequence.targets)
                for modality in MODALITIES:
                    original_lengths = np.load(raw_cache / "splits" / split / (modality + "_lengths.npy"))
                    saved_lengths = np.load(output / split / (modality + "_lengths.npy"))
                    np.testing.assert_array_equal(saved_lengths, original_lengths)
                    original = np.load(raw_cache / "splits" / split / (modality + ".npy"))
                    transformed = np.load(output / split / (modality + ".npy"))
                    expected = np.zeros_like(original)
                    for row, count in enumerate(original_lengths):
                        if count:
                            expected[row, :int(count)] = (
                                original[row, :int(count)] - np.asarray(stats[modality]["mean"], np.float32)
                            ) / np.asarray(stats[modality]["scale"], np.float32)
                    np.testing.assert_allclose(transformed, expected, atol=0, rtol=0)
                    for row, count in enumerate(saved_lengths):
                        self.assertTrue(np.all(transformed[row, int(count):] == 0))
                    np.testing.assert_allclose(normalized_sequence.features[modality], normalized_pool.features[modality], atol=POOLING_TOLERANCE, rtol=0)
                _close_dataset(raw)
                _close_dataset(normalized_pool)
                _close_dataset(normalized_sequence)

            self.assertEqual(result["cache_reused"], False)
            before = digest(output / "metadata.json")
            reused = prepare_cache(raw_cache, pooled_cache, output, author_csv, allow_fixture=True)
            self.assertTrue(reused["cache_reused"])
            self.assertEqual(before, digest(output / "metadata.json"))
            damaged = output / "train/audio.npy"
            damaged_values = np.load(damaged)
            damaged_values[0, 0, 0] += 3
            np.save(damaged, damaged_values)
            with self.assertRaisesRegex(ValueError, "modified"):
                prepare_cache(raw_cache, pooled_cache, output, author_csv, allow_fixture=True)
            self.assertEqual(np.load(damaged)[0, 0, 0], damaged_values[0, 0, 0])

    def test_corrupt_statistics_and_raw_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_cache, pooled_cache, author_csv = _sequence_fixture(root)
            statistics_path = pooled_cache / "normalization.json"
            statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
            statistics["text"]["mean"][0] += 0.25
            _write_json(statistics_path, statistics)
            with self.assertRaisesRegex(ValueError, "artifact"):
                prepare_cache(raw_cache, pooled_cache, root / "bad-stats", author_csv, allow_fixture=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_cache, pooled_cache, author_csv = _sequence_fixture(root)
            feature_path = raw_cache / "splits/train/audio.npy"
            values = np.load(feature_path)
            values[0, 0, 0] += 0.5
            np.save(feature_path, values)
            with self.assertRaisesRegex(ValueError, "hash"):
                prepare_cache(raw_cache, pooled_cache, root / "bad-raw", author_csv, allow_fixture=True)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_cache, pooled_cache, author_csv = _sequence_fixture(root)
            timestamp_path = raw_cache / "splits/train/audio_times.npy"
            timestamps = np.load(timestamp_path)
            timestamps[0, 0] += 0.5
            np.save(timestamp_path, timestamps)
            with self.assertRaisesRegex(ValueError, "timestamp"):
                prepare_cache(raw_cache, pooled_cache, root / "bad-timestamp", author_csv, allow_fixture=True)

    def test_fixture_opt_in_is_python_api_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_cache, pooled_cache, author_csv = _sequence_fixture(root)
            with self.assertRaisesRegex(ValueError, "Pinned author CSV"):
                prepare_cache(raw_cache, pooled_cache, root / "formal-cli-contract", author_csv)


if __name__ == "__main__":
    unittest.main()
