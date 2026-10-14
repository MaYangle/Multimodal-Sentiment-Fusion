import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from jevaffect.author_features import aligned_suffix, temporal_ids, AuthorFeatureSplit, COHORT, MODALITIES
from jevaffect.author_training import train_author
from jevaffect.native_training import digest


class AuthorFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.cache = self.root / "cache"; self.cache.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def make_cache(self):
        rng = np.random.default_rng(23); meta = {"cohort": COHORT, "text_pooling": "valid_mean", "source_sha256": "fixture",
            "input_form": "synthetic aligned fixture", "artifact_sha256": {}, "split_counts": {"train": 2, "valid": 2}, "feature_shapes": {m: [4, 3] for m in MODALITIES}}
        for split, videos in (("train", ["a", "b"]), ("valid", ["c", "d"])):
            directory = self.cache / split; directory.mkdir()
            source = {m: rng.normal(size=(2, 4, 3)) for m in MODALITIES}
            for x in source.values():
                x[0, :2] = 0; x[1, :1] = 0
            prepared, lengths, _, _ = aligned_suffix(source)
            (directory / "ids.json").write_text(json.dumps([v + "|0.0|1.0" for v in videos])); np.save(directory / "targets.npy", np.array([-.5, .75], np.float32))
            for m in MODALITIES:
                np.save(directory / (m + ".npy"), prepared[m]); np.save(directory / (m + "_lengths.npy"), lengths)
            for file in directory.iterdir():
                meta["artifact_sha256"][file.relative_to(self.cache).as_posix()] = digest(file)
        (self.cache / "metadata.json").write_text(json.dumps(meta))

    def test_shared_alignment_interior_zeros_and_audio_rule(self):
        text = np.array([[[0., 0.], [2., 4.], [0., 0.], [6., 8.]]])
        audio = np.array([[[9., 9.], [1., -np.inf], [3., 4.], [5., 6.]]]); vision = np.arange(8.).reshape(1, 4, 2)
        prepared, lengths, starts, counters = aligned_suffix(dict(text=text, audio=audio, vision=vision))
        np.testing.assert_array_equal(lengths, [3]); np.testing.assert_array_equal(starts, [1])
        np.testing.assert_array_equal(prepared["text"][0], [[2, 4], [0, 0], [6, 8], [0, 0]])
        np.testing.assert_array_equal(prepared["audio"][0], [[1, 0], [3, 4], [5, 6], [0, 0]])
        np.testing.assert_array_equal(prepared["vision"][0, :3], vision[0, 1:])
        self.assertEqual(counters["audio"]["audio_negative_inf_replaced"], 1)
        with self.assertRaises(ValueError):
            aligned_suffix(dict(text=text * 0, audio=audio, vision=vision))
        with self.assertRaises(ValueError):
            aligned_suffix(dict(text=text, audio=audio, vision=np.full_like(vision, np.inf)))

    def test_identity_pooling_labels_masks_and_corruption_guard(self):
        self.make_cache()
        ids = temporal_ids(np.array([["a", "-0.48", "1.00"], ["b", "0", "2"]]))
        self.assertEqual(ids[0], "a|-0.48|1.00")
        with self.assertRaises(ValueError):
            temporal_ids(np.array([["a", "0", "1"], ["a", "0", "1"]]))
        ds = AuthorFeatureSplit(self.cache, "valid")
        try:
            raw = np.load(self.cache / "valid/text.npy")
            np.testing.assert_array_equal(ds.features["text"][0], raw[0, :2].mean(0))
            self.assertEqual(ds.targets.tolist(), [-.5, .75])
            self.assertEqual(ds.lengths["text"].tolist(), [2, 3])
        finally:
            ds.close()
        with self.assertRaises(ValueError):
            AuthorFeatureSplit(self.cache, "test")
        target_path = self.cache / "valid/targets.npy"; np.save(target_path, np.zeros(2, np.float32))
        with self.assertRaises(ValueError):
            AuthorFeatureSplit(self.cache, "valid")

    def test_exact_epoch_resume_and_version_guard(self):
        self.make_cache()
        config = {"dataset": "mosei", "cohort": COHORT, "model": "early_fusion", "cache_dir": str(self.cache),
                  "expected_source_sha256": "fixture", "run_root": str(self.root / "whole"), "seeds": [13, 42, 2026],
                  "device": "cpu", "hidden": 8, "dropout": .1, "batch_size": 2, "learning_rate": .001,
                  "weight_decay": .0001, "max_epochs": 3, "patience": 12}
        train_author(config, 13)
        resumed = dict(config, run_root=str(self.root / "resumed"))
        train_author(resumed, 13, stop_after=1); train_author(resumed, 13, resume=True)
        states = [torch.load(Path(c["run_root"]) / "formal/early_fusion/seed_13/last.pt", weights_only=False) for c in (config, resumed)]
        for key in states[0]["model"]:
            self.assertTrue(torch.equal(states[0]["model"][key], states[1]["model"][key]), key)
        for a, b in zip(states[0]["history"], states[1]["history"]):
            self.assertEqual(a["train_L1"], b["train_L1"]); self.assertEqual(a["valid"], b["valid"])
        predictions = [json.loads((Path(c["run_root"]) / "formal/early_fusion/seed_13/valid_predictions.json").read_text()) for c in (config, resumed)]
        self.assertEqual(*predictions)
        with self.assertRaises(ValueError):
            train_author(dict(resumed, expected_source_sha256="different"), 13, resume=True)
