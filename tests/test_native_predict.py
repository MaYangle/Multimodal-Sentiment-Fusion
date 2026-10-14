import json
import unittest
from pathlib import Path
import numpy as np
import test_native_training as fixtures
from jevaffect.native_training import train, digest
from jevaffect.native_predict import read_features, predict_features


class NativePredictTests(unittest.TestCase):
    def test_glove_valid_mean_differs_from_cls_pooling(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            path = fixture.root / "glove.npz"
            np.savez(path, text=np.array([[1., 1.], [5., 5.], [0., 0.]], np.float32), text_length=np.array(2, np.int64))
            spec = {"modalities": ("text",), "shapes": {"text": (3, 2)}, "pooled": True, "case": None, "text_pooling": "valid_mean"}
            np.testing.assert_array_equal(read_features(path, spec)["text"].numpy(), [[3., 3.]])
            spec["text_pooling"] = "cls"
            np.testing.assert_array_equal(read_features(path, spec)["text"].numpy(), [[1., 1.]])
        finally:
            fixture.tearDown()

    def test_masks_shape_and_all_missing(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            path = fixture.root / "input.npz"
            spec = {"modalities": ("text", "audio", "vision"), "shapes": {m: (3, 4) for m in ("text", "audio", "vision")}, "pooled": True, "case": "joint_dynamic"}
            values = {m: np.ones((3, 4), np.float32) for m in spec["modalities"]}
            values["audio"][1:] = np.nan
            values.update({m + "_length": np.array(1, np.int64) for m in spec["modalities"]})
            np.savez(path, **values)
            features = read_features(path, spec)
            self.assertTrue(np.isfinite(features["audio"].numpy()).all())
            for m in spec["modalities"]:
                values[m + "_length"] = np.array(0, np.int64)
            np.savez(path, **values)
            with self.assertRaises(ValueError):
                read_features(path, spec)
            values["text_length"] = np.array(4, np.int64); np.savez(path, **values)
            with self.assertRaises(ValueError):
                read_features(path, spec)
        finally:
            fixture.tearDown()

    def test_native_checkpoint_prediction_and_descriptor_guard(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            config = {"dataset": "simsv2", "model": "early_fusion", "author_csv": str(fixture.csv), "cache_dir": str(fixture.cache),
                      "run_root": str(fixture.root / "run"), "seeds": [13, 42, 2026], "device": "cpu", "hidden": 8,
                      "batch_size": 2, "learning_rate": .001, "weight_decay": .0001, "max_epochs": 1, "patience": 12}
            train(config, 13); root = Path(config["run_root"]) / "early_fusion/seed_13"
            arrays = {}
            for m in ("text", "audio", "vision"):
                arrays[m] = np.load(fixture.cache / "valid" / (m + ".npy"))[0]
                arrays[m + "_length"] = np.load(fixture.cache / "valid" / (m + "_lengths.npy"))[0]
            file = fixture.root / "input.npz"; np.savez(file, **arrays)
            records = [{"id": "c/1", "features_npz": str(file), "metadata_sha256": digest(fixture.cache / "meta.json")}]
            outputs = predict_features(records, root / "best.pt", fixture.cache)
            known = json.loads((root / "valid_predictions.json").read_text())[0]["prediction"]
            self.assertAlmostEqual(outputs[0]["score_raw"], known, places=6)
            with self.assertRaises(ValueError):
                predict_features(records * 2, root / "best.pt", fixture.cache)
            with self.assertRaises(ValueError):
                predict_features([{**records[0], "metadata_sha256": "wrong"}], root / "best.pt", fixture.cache)
        finally:
            fixture.tearDown()
