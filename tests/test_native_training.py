import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from jevaffect.native_training import NativeSplit, author_rows, digest, train


class NativeTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / "author.csv"
        with self.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "clip_id", "mode", "label", "text"])
            writer.writeheader()
            for sid, split, label in [("a", "train", -.6), ("b", "train", .4), ("c", "valid", -.1), ("d", "valid", .2), ("e", "test", "must-not-be-parsed")]:
                writer.writerow(dict(video_id=sid, clip_id="1", mode=split, label=label, text="fixture"))
        self.cache = self.root / "cache"
        self.cache.mkdir()
        hashes = {}
        rng = np.random.default_rng(4)
        for split, ids in (("train", ["a/1", "b/1"]), ("valid", ["c/1", "d/1"])):
            directory = self.cache / split
            directory.mkdir()
            (directory / "ids.json").write_text(json.dumps(ids))
            for m in ("text", "audio", "vision"):
                np.save(directory / (m + ".npy"), rng.normal(size=(2, 3, 4)).astype(np.float32))
                np.save(directory / (m + "_lengths.npy"), np.array([1, 0], np.int64))
                for file in (m + ".npy", m + "_lengths.npy"):
                    hashes[split + "/" + file] = digest(directory / file)
        (self.cache / "meta.json").write_text(json.dumps({"artifact_sha256": hashes}))
        self.rows = author_rows(self.csv, "simsv2")

    def tearDown(self):
        self.temp.cleanup()

    def test_original_scores_and_missing_modality_pooling(self):
        ds = NativeSplit(self.cache, "valid", self.rows, ("text", "audio", "vision"))
        self.assertAlmostEqual(ds.targets[0], -.1)
        for m in ds.features:
            self.assertTrue(np.all(ds.features[m][1] == 0))
        for m in ("audio", "vision"):
            raw = np.load(self.cache / "valid" / (m + ".npy"))
            np.testing.assert_array_equal(ds.features[m][0], raw[0, 0])
        with self.assertRaises(ValueError):
            NativeSplit(self.cache, "test", self.rows, ("text",))

    def test_cohort_and_corrupt_hash_rejected(self):
        path = self.cache / "train" / "ids.json"
        path.write_text(json.dumps(["a/1", "c/1"]))
        with self.assertRaises(ValueError):
            NativeSplit(self.cache, "train", self.rows, ("text",))
        path.write_text(json.dumps(["a/1", "b/1"]))
        np.save(self.cache / "train" / "text_lengths.npy", np.array([2, 2], np.int64))
        with self.assertRaises(ValueError):
            NativeSplit(self.cache, "train", self.rows, ("text",))

    def test_exact_epoch_resume_and_identity_guard(self):
        config = {"dataset": "simsv2", "model": "text", "author_csv": str(self.csv), "cache_dir": str(self.cache),
                  "run_root": str(self.root / "whole"), "seeds": [13, 42, 2026], "device": "cpu", "hidden": 8,
                  "batch_size": 2, "learning_rate": .001, "weight_decay": .0001, "max_epochs": 3, "patience": 12}
        train(config, 13)
        separate = dict(config, run_root=str(self.root / "resumed"))
        train(separate, 13, stop_after=1)
        train(separate, 13, resume=True)
        states = [torch.load(Path(c["run_root"]) / "text/seed_13/last.pt", weights_only=False) for c in (config, separate)]
        for key in states[0]["model"]:
            self.assertTrue(torch.equal(states[0]["model"][key], states[1]["model"][key]), key)
        for a, b in zip(states[0]["history"], states[1]["history"]):
            self.assertEqual(a["train_L1"], b["train_L1"])
            self.assertEqual(a["valid"], b["valid"])
        with self.assertRaises(ValueError):
            train(dict(separate, learning_rate=.002), 13, resume=True)
