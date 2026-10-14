from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "scripts" / "build_raw_feature_cache.py"
PROJECT_ROOT = HERE
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SPEC = importlib.util.spec_from_file_location("raw_cache_builder", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RawCacheBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "raw_bundle"
        (self.bundle / "records").mkdir(parents=True)
        self.csv_path = self.root / "meta.csv"
        self.manifest_path = self.root / "manifest.jsonl"
        self.output = self.root / "cache"
        self.ids = ["vid_01/00001", "vid_01/00002", "vid_02/00003"]
        self.splits = ["train", "train", "valid"]
        self.labels = ["-1.0", "0.0", "1.0"]
        self.texts = ["负向文本", "中性文本", "正向文本"]
        self.hashes = [hashlib.sha256(f"video-{i}".encode()).hexdigest() for i in range(3)]
        with self.csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video_id", "clip_id", "text", "label", "mode"])
            writer.writeheader()
            for sample_id, split, label, text in zip(self.ids, self.splits, self.labels, self.texts):
                video, clip = sample_id.split("/")
                writer.writerow({"video_id": video, "clip_id": clip, "text": text, "label": label, "mode": split})
            # Invalid test-only row proves that only train/valid labels are mapped.
            writer.writerow({"video_id": "vid_test", "clip_id": "00000", "text": "test", "label": "-0.1", "mode": "test"})
        csv_hash = sha256(self.csv_path)
        manifest = []
        for sample_id, split, label, text, video_hash in zip(self.ids, self.splits, self.labels, self.texts, self.hashes):
            video, _ = sample_id.split("/")
            manifest.append({
                "ID": sample_id,
                "split": split,
                "dataset": "CH-SIMS-v2.0",
                "language": "zh",
                "video_path": f"raw/{video}.mp4",
                "sha256": video_hash,
                "text": text,
                "raw_label": label,
                "source_video": video,
                "transcript_condition": "author_provided",
                "source_csv_sha256": csv_hash,
            })
        self.manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8")

        provenance = {
            "local_files_only": True,
            "dimensions": {"text": 768, "audio": 768, "vision": 384},
            "encoders": {"mbert": {"revision": "test"}},
        }
        index_rows = []
        lengths = [(65, 401, 56), (3, 4, 2), (2, 2, 3)]
        for index, sample_id in enumerate(self.ids):
            video_hash = self.hashes[index]
            text_len, audio_len, vision_len = lengths[index]
            rng = np.random.default_rng(index)
            arrays = {
                "text": rng.normal(size=(text_len, 768)).astype(np.float32),
                "text_mask": np.ones(text_len, dtype=np.bool_),
                "text_token_ids": np.arange(text_len, dtype=np.int64) + 100,
                "audio": rng.normal(size=(audio_len, 768)).astype(np.float32),
                "audio_mask": np.ones(audio_len, dtype=np.bool_),
                "audio_times": np.arange(audio_len, dtype=np.float64) * 0.02,
                "vision": rng.normal(size=(vision_len, 384)).astype(np.float32),
                "vision_mask": np.ones(vision_len, dtype=np.bool_),
                "vision_times": np.arange(vision_len, dtype=np.float64) * 0.5,
            }
            artifact_rel = f"records/{index:04d}.npz"
            artifact_path = self.bundle / artifact_rel
            np.savez_compressed(artifact_path, **arrays)
            modalities = {
                "text": {"valid": True, "status": "present", "length": text_len, "dim": 768},
                "audio": {"valid": True, "status": "present", "length": audio_len, "dim": 768},
                "vision": {"valid": True, "status": "present", "length": vision_len, "dim": 384},
            }
            metadata = {
                "id": sample_id,
                "source_video_sha256": video_hash,
                "decode": {"source_sha256": video_hash},
                "transcript": self.texts[index],
                "transcript_source": "provided",
                "preprocessing": {
                    "audio_sample_rate": 16000,
                    "video_fps": 2.0,
                    "mbert_max_tokens": 64,
                    "text_truncated": text_len > 64,
                    "audio_feature_times": "centers of WavLM convolutional receptive fields",
                    "video_frame_timestamps_kind": "decoder_timeline",
                },
                "modalities": modalities,
                "encoder_provenance": provenance,
                "prediction": None,
                "fixture": False,
                "model_results": True,
            }
            index_rows.append({
                "id": sample_id,
                "artifact": artifact_rel,
                "artifact_sha256": sha256(artifact_path),
                "metadata": metadata,
            })
        index_path = self.bundle / "index.jsonl"
        index_path.write_text("".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8")
        raw_meta = {
            "schema_version": 1,
            "cache_kind": "raw_video_feature_records",
            "status": "ready_for_canonical_cache_join",
            "sample_count": len(index_rows),
            "input_manifest_sha256": hashlib.sha256(b"raw extraction manifest").hexdigest(),
            "encoder_provenance": provenance,
            "files": {
                "index.jsonl": sha256(index_path),
                "records": [row["artifact"] for row in index_rows],
            },
            "labels_included": False,
            "canonical_ids_and_splits_joined": False,
            "fixture_results": False,
            "all_records_are_fixtures": False,
            "model_results": True,
        }
        meta_path = self.bundle / "meta.json"
        meta_path.write_text(json.dumps(raw_meta, indent=2) + "\n", encoding="utf-8")
        (self.bundle / "meta.sha256").write_text(f"{sha256(meta_path)}  meta.json\n", encoding="ascii")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def build(self, **kwargs):
        return builder.build_cache(
            project_root=PROJECT_ROOT,
            manifest_path=self.manifest_path,
            official_csv=self.csv_path,
            raw_bundle=self.bundle,
            output_dir=self.output,
            expected_counts=None,
            verify_training_loader_fn=lambda cache, root: {"device": "cpu", "checked": True},
            **kwargs,
        )

    def test_build_joins_official_mapping_preserves_times_and_reports_truncation(self):
        result = self.build()
        self.assertEqual(result["split_counts"], {"train": 2, "valid": 1})
        self.assertEqual(result["truncation"]["text"]["truncated_records"], 1)
        self.assertEqual(result["truncation"]["audio"]["total_dropped_frames_or_tokens"], 1)
        self.assertEqual(result["truncation"]["vision"]["total_dropped_frames_or_tokens"], 1)
        train = self.output / "splits" / "train"
        self.assertEqual(json.loads((train / "ids.json").read_text(encoding="utf-8")), self.ids[:2])
        labels = np.load(train / "labels.npy", allow_pickle=False)
        self.assertEqual(labels.tolist(), [0, 2])
        text = np.load(train / "text.npy", allow_pickle=False)
        audio = np.load(train / "audio.npy", allow_pickle=False)
        vision = np.load(train / "vision.npy", allow_pickle=False)
        self.assertEqual(text.shape, (2, 64, 768))
        self.assertEqual(audio.shape, (2, 400, 768))
        self.assertEqual(vision.shape, (2, 55, 384))
        self.assertTrue(np.all(text[1, 3:] == 0))
        self.assertTrue(np.all(audio[1, 4:] == 0))
        self.assertTrue(np.all(vision[1, 2:] == 0))
        self.assertEqual(np.load(train / "audio_lengths.npy", allow_pickle=False).tolist(), [400, 4])
        self.assertEqual(np.load(train / "vision_lengths.npy", allow_pickle=False).tolist(), [55, 2])
        masks = np.load(train / "audio_mask.npy", allow_pickle=False)
        self.assertTrue(masks[0].all())
        self.assertTrue(not masks[1, 4:].any())
        times = np.load(train / "audio_times.npy", allow_pickle=False)
        self.assertEqual(times[0, 399], 399 * 0.02)
        self.assertTrue(np.all(times[1, 4:] == 0))
        meta = json.loads((self.output / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["split_paths"], {"train": "splits/train", "valid": "splits/valid"})
        self.assertFalse((self.output / "splits" / "test").exists())
        self.assertTrue((self.output / "bundle.sha256").is_file())

    def test_missing_or_noncanonical_raw_id_is_rejected_without_output(self):
        index_path = self.bundle / "index.jsonl"
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["id"] += "__provided"
        rows[0]["metadata"]["id"] = rows[0]["id"]
        index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        meta = json.loads((self.bundle / "meta.json").read_text(encoding="utf-8"))
        meta["files"]["index.jsonl"] = sha256(index_path)
        (self.bundle / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (self.bundle / "meta.sha256").write_text(f"{sha256(self.bundle / 'meta.json')}  meta.json\n", encoding="ascii")
        with self.assertRaisesRegex(builder.CacheBuildError, "IDs must match exactly"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_invalid_raw_modality_mask_is_rejected(self):
        index_path = self.bundle / "index.jsonl"
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        artifact_path = self.bundle / rows[1]["artifact"]
        with np.load(artifact_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["vision_mask"][0] = False
        np.savez_compressed(artifact_path, **arrays)
        rows[1]["artifact_sha256"] = sha256(artifact_path)
        index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        meta = json.loads((self.bundle / "meta.json").read_text(encoding="utf-8"))
        meta["files"]["index.jsonl"] = sha256(index_path)
        (self.bundle / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (self.bundle / "meta.sha256").write_text(f"{sha256(self.bundle / 'meta.json')}  meta.json\n", encoding="ascii")
        with self.assertRaisesRegex(builder.CacheBuildError, "missing/padded position"):
            self.build()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
