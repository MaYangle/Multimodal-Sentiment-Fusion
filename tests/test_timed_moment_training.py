import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from jevaffect.native_training import digest
from jevaffect.timed_moment_training import (
    CASES,
    EXPECTED_DIMS,
    PARAMETER_COUNT,
    TimedMomentSplit,
    _load_and_validate_data,
    train,
)
from jevaffect.timed_moments import TimedMomentRegressor


class TimedMomentTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / "author.csv"
        rows = [
            {"video_id": "train_a", "clip_id": "0001", "mode": "train", "label": "-0.5"},
            {"video_id": "train_b", "clip_id": "0001", "mode": "train", "label": "0.4"},
            {"video_id": "valid_a", "clip_id": "0001", "mode": "valid", "label": "0.2"},
            {"video_id": "test_a", "clip_id": "0001", "mode": "test", "label": "must-not-be-parsed"},
        ]
        with self.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("video_id", "clip_id", "mode", "label"))
            writer.writeheader()
            writer.writerows(rows)
        self.csv_sha = digest(self.csv)
        self.cache = self.root / "timed-grid"
        self.norm = self.root / "pooled-normalization"
        self.cache.mkdir()
        self.norm.mkdir()
        self.ids = {"train": ["train_a/0001", "train_b/0001"], "valid": ["valid_a/0001"]}
        rng = np.random.default_rng(91)
        artifact_hashes = {}
        for split, ids in self.ids.items():
            directory = self.cache / split
            directory.mkdir()
            (directory / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
            targets = {"train": [-.5, .4], "valid": [.2]}[split]
            (directory / "raw_labels.json").write_text(json.dumps(targets), encoding="utf-8")
            n = len(ids)
            np.save(directory / "text.npy", rng.normal(size=(n, 768)).astype(np.float32))
            np.save(directory / "audio_mean.npy", rng.normal(size=(n, 768)).astype(np.float32))
            np.save(directory / "vision_mean.npy", rng.normal(size=(n, 384)).astype(np.float32))
            np.save(directory / "audio_grid.npy", rng.normal(size=(n, 4, 768)).astype(np.float32))
            np.save(directory / "vision_grid.npy", rng.normal(size=(n, 4, 384)).astype(np.float32))
            pair_valid = np.ones(n, dtype=bool)
            presence = np.ones((n, 3), dtype=bool)
            if split == "train":
                pair_valid[1] = False
                presence[1, 1] = False
            np.save(directory / "pair_valid.npy", pair_valid)
            np.save(directory / "presence.npy", presence)
            grid_times = np.tile(np.asarray([0.0, .25, .5, .75], dtype=np.float64), (n, 1))
            np.save(directory / "grid_times.npy", grid_times)
            for path in directory.iterdir():
                if path.is_file():
                    artifact_hashes[path.relative_to(self.cache).as_posix()] = digest(path)
        cache_meta = {
            "format": "native-timed-grid-v1",
            "author_csv_sha256": self.csv_sha,
            "counts": {"train": 2, "valid": 1},
            "test_exported": False,
            "artifact_sha256": artifact_hashes,
            "pairing_assumptions": "fixture only; timestamps do not assert physical synchronization",
        }
        (self.cache / "metadata.json").write_text(json.dumps(cache_meta), encoding="utf-8")

        stats = {}
        for modality, dim in EXPECTED_DIMS.items():
            stats[modality] = {"mean": np.zeros(dim, dtype=np.float32).tolist(),
                               "scale": np.ones(dim, dtype=np.float32).tolist(),
                               "fit_indices": [0, 1]}
        stats_path = self.norm / "normalization.json"
        stats_path.write_text(json.dumps(stats), encoding="utf-8")
        norm_hashes = {"normalization.json": digest(stats_path)}
        for split, ids in self.ids.items():
            directory = self.norm / split
            directory.mkdir()
            (directory / "ids.json").write_text(json.dumps(ids), encoding="utf-8")
            norm_hashes[f"{split}/ids.json"] = digest(directory / "ids.json")
        norm_meta = {
            "author_csv_sha256": self.csv_sha,
            "counts": {"train": 2, "valid": 1},
            "normalization": {"fit_split": "train", "fit_count": 2,
                              "fit_ids_sha256": norm_hashes["train/ids.json"]},
            "artifact_sha256": norm_hashes,
        }
        (self.norm / "metadata.json").write_text(json.dumps(norm_meta), encoding="utf-8")
        self.config = {
            "dataset": "simsv2", "author_csv": str(self.csv), "cache_dir": str(self.cache),
            "normalization_cache_dir": str(self.norm), "run_root": str(self.root / "runs"),
            "seeds": [13, 42, 2026], "device": "cpu", "batch_size": 32,
            "learning_rate": .0001, "weight_decay": .0001, "max_epochs": 2, "patience": 12,
            "rank": 32, "hidden": 234, "dropout": .1,
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_normalization_and_invalid_pair_zeroing(self):
        _, datasets, stats_identity, _, counts = _load_and_validate_data(self.config, allow_fixture=True)
        train_split = datasets["train"]
        self.assertEqual(counts, {"train": 2, "valid": 1})
        self.assertEqual(stats_identity["sha256"], digest(self.norm / "normalization.json"))
        self.assertTrue(train_split.pair_valid[0])
        self.assertFalse(train_split.pair_valid[1])
        np.testing.assert_array_equal(train_split.features["audio_mean"][1], np.zeros(768, np.float32))
        np.testing.assert_array_equal(train_split.features["audio_grid"][1], np.zeros((4, 768), np.float32))
        np.testing.assert_array_equal(train_split.features["vision_grid"][1], np.zeros((4, 384), np.float32))
        self.assertTrue(np.isfinite(train_split.features["audio_grid"][0]).all())

    def test_model_budget_and_paired_permutation_invariance(self):
        dims = {"text": 768, "audio": 768, "vision": 384}
        for case in CASES:
            model = TimedMomentRegressor(dims, case, rank=32, hidden=234, dropout=.1)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), PARAMETER_COUNT)
        model = TimedMomentRegressor(dims, "paired_cov", rank=32, hidden=234, dropout=.1).eval()
        torch.manual_seed(17)
        features = {
            "text": torch.randn(3, 768),
            "audio_mean": torch.randn(3, 768),
            "vision_mean": torch.randn(3, 384),
            "audio_grid": torch.randn(3, 4, 768),
            "vision_grid": torch.randn(3, 4, 384),
            "pair_valid": torch.tensor([True, False, True]),
        }
        with torch.no_grad():
            original = model(features)
            permuted = dict(features)
            permutation = torch.tensor([2, 0, 3, 1])
            permuted["audio_grid"] = features["audio_grid"][:, permutation]
            permuted["vision_grid"] = features["vision_grid"][:, permutation]
            reordered = model(permuted)
        torch.testing.assert_close(original, reordered, rtol=1e-6, atol=1e-6)
        for case in CASES:
            candidate = TimedMomentRegressor(dims, case, rank=32, hidden=234, dropout=.1).eval()
            with torch.no_grad():
                output = candidate(features)
            self.assertEqual(output.shape, (3,))
            self.assertTrue(torch.isfinite(output).all())
        unpaired = TimedMomentRegressor(dims, "unpaired_cov", rank=32, hidden=234, dropout=.1).eval()
        with torch.no_grad():
            audio = unpaired.audio_projection(features["audio_grid"])
            vision = unpaired.vision_projection(features["vision_grid"])
            audio = audio - audio.mean(dim=1, keepdim=True)
            shifted_vision = torch.roll(vision, shifts=1, dims=1)
            shifted_vision = shifted_vision - shifted_vision.mean(dim=1, keepdim=True)
            expected = (audio * shifted_vision).mean(dim=1)
            expected = torch.where(features["pair_valid"][:, None], expected, torch.zeros_like(expected))
            torch.testing.assert_close(unpaired.moment(features), expected)

    def test_exact_resume_and_partial_run_not_complete(self):
        whole = dict(self.config, run_root=str(self.root / "whole"))
        train(whole, "paired_cov", 13, allow_fixture=True)
        resumed = dict(self.config, run_root=str(self.root / "resumed"))
        partial = train(resumed, "paired_cov", 13, stop_after=1, allow_fixture=True)
        self.assertFalse(partial["complete"])
        finished = train(resumed, "paired_cov", 13, resume=True, allow_fixture=True)
        self.assertTrue(finished["complete"])
        whole_state = torch.load(self.root / "whole/formal/paired_cov/seed_13/last.pt", map_location="cpu", weights_only=False)
        resume_state = torch.load(self.root / "resumed/formal/paired_cov/seed_13/last.pt", map_location="cpu", weights_only=False)
        for key in whole_state["model"]:
            self.assertTrue(torch.equal(whole_state["model"][key], resume_state["model"][key]), key)
        for expected, actual in zip(whole_state["history"], resume_state["history"]):
            self.assertEqual(expected["epoch"], actual["epoch"])
            self.assertEqual(expected["train_L1"], actual["train_L1"])
            self.assertEqual(expected["valid"], actual["valid"])
            self.assertEqual(expected["learning_rate"], actual["learning_rate"])
        self.assertEqual(json.loads((self.root / "whole/formal/paired_cov/seed_13/valid_predictions.json").read_text()),
                         json.loads((self.root / "resumed/formal/paired_cov/seed_13/valid_predictions.json").read_text()))

    def test_cache_hash_tampering_is_rejected(self):
        feature = self.cache / "train/audio_grid.npy"
        array = np.load(feature)
        array[0, 0, 0] += 1
        np.save(feature, array)
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            _load_and_validate_data(self.config, allow_fixture=True)


if __name__ == "__main__":
    unittest.main()
