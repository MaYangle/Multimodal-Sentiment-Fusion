import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from jevaffect.native_training import author_rows, digest
from jevaffect.reliability_training import ReliabilitySplit, objective, train_case


class ReliabilityTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "author.csv"
        with self.source.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id", "clip_id", "mode", "label", "text", "label_T", "label_A", "label_V"])
            w.writeheader()
            for sid, split, score in [("a", "train", -.6), ("b", "train", .4), ("c", "valid", -.1), ("d", "valid", .2)]:
                extra = score if split == "train" else "validation-auxiliary-must-not-be-read"
                w.writerow(dict(video_id=sid, clip_id="1", mode=split, label=score, text="fixture", label_T=extra, label_A=extra, label_V=extra))
        self.cache = self.root / "cache"; self.cache.mkdir()
        hashes = {}; rng = np.random.default_rng(3)
        for split, ids in (("train", ["a/1", "b/1"]), ("valid", ["c/1", "d/1"])):
            directory = self.cache / split; directory.mkdir()
            (directory / "ids.json").write_text(json.dumps(ids))
            for m in ("text", "audio", "vision"):
                np.save(directory / (m + ".npy"), rng.normal(size=(2, 3, 4)).astype(np.float32))
                np.save(directory / (m + "_lengths.npy"), np.array([1, 2], np.int64))
                for filename in (m + ".npy", m + "_lengths.npy"):
                    hashes[split + "/" + filename] = digest(directory / filename)
        (self.cache / "meta.json").write_text(json.dumps({"artifact_sha256": hashes}))
        self.rows = author_rows(self.source, "simsv2")

    def tearDown(self):
        self.temp.cleanup()

    def test_validation_auxiliary_never_parsed(self):
        valid = ReliabilitySplit(self.cache, "valid", self.rows, self.source)
        self.assertFalse(valid.auxiliary_valid.any())
        self.assertTrue(np.all(valid.auxiliary == 0))
        train = ReliabilitySplit(self.cache, "train", self.rows, self.source)
        self.assertTrue(train.auxiliary_valid.all())
        self.assertNotIn("auxiliary", train[0]["features"])

    def test_disabled_auxiliary_has_no_loss_or_gradient(self):
        from jevaffect.reliability import ReliabilityRegressor
        model = ReliabilityRegressor({m: 4 for m in ("text", "audio", "vision")}, hidden=8, dropout=0, fusion="dynamic")
        x = {m: torch.randn(2, 4) for m in ("text", "audio", "vision")}
        x["availability"] = torch.ones(2, 3, dtype=torch.bool)
        out = model(x); y = torch.tensor([-.6, .4]); valid = x["availability"]
        loss, parts = objective(out, y, torch.zeros(2, 3), valid, False, .01, .5)
        other, _ = objective(out, y, torch.ones(2, 3), valid, False, .01, .5)
        self.assertEqual(loss.item(), other.item())
        self.assertEqual(parts["auxiliary_L1"], 0)
        loss.backward()
        for parameter in model.unimodal_heads.parameters():
            self.assertIsNone(parameter.grad)
        model.zero_grad(set_to_none=True)
        enabled, _ = objective(model(x), y, torch.ones(2, 3), valid, True, .01, .5)
        self.assertNotEqual(enabled.item(), loss.item())
        enabled.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.unimodal_heads.parameters()))

    def test_exact_distribution_checkpoint_resume(self):
        config = {"dataset": "simsv2", "author_csv": str(self.source), "cache_dir": str(self.cache),
                  "run_root": str(self.root / "whole"), "seeds": [13, 42, 2026], "device": "cpu", "hidden": 8,
                  "dropout": .1, "batch_size": 2, "learning_rate": .0001, "weight_decay": .0001,
                  "max_epochs": 3, "patience": 12, "regularization": .01, "auxiliary_weight": .5}
        train_case(config, "S1G1", 13)
        separate = dict(config, run_root=str(self.root / "split"))
        train_case(separate, "S1G1", 13, stop_after=1)
        train_case(separate, "S1G1", 13, resume=True)
        states = [torch.load(Path(c["run_root"]) / "S1G1/formal/seed_13/last.pt", weights_only=False) for c in (config, separate)]
        for key in states[0]["model"]:
            self.assertTrue(torch.equal(states[0]["model"][key], states[1]["model"][key]), key)
        for a, b in zip(states[0]["history"], states[1]["history"]):
            self.assertEqual(a["train_loss"], b["train_loss"])
            self.assertEqual(a["valid"], b["valid"])
        with self.assertRaises(ValueError):
            train_case(dict(separate, auxiliary_weight=.9), "S1G1", 13, resume=True)
