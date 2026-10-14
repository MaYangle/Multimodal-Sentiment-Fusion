import csv
import json
import unittest
from pathlib import Path
import numpy as np
import torch
import test_native_training as fixtures
from jevaffect.reference_residual import build_reference_model, REFERENCE_CASES
from jevaffect.reference_training import ReferenceSplit, reference_objective, train_reference
from jevaffect.native_training import author_rows


class ReferenceResidualTests(unittest.TestCase):
    def test_same_initialization_capacity_and_decoder_gradients(self):
        config = {"text_hidden": 128, "context_hidden": 304, "dropout": 0.}
        models = {}
        for case in REFERENCE_CASES:
            torch.manual_seed(13); models[case] = build_reference_model(case, dict(text=768, audio=768, vision=384), config)
        self.assertEqual(sum(p.numel() for p in models["sum1"].parameters()), 492691)
        for case in REFERENCE_CASES:
            for key, value in models[case].state_dict().items():
                self.assertTrue(torch.equal(value, models["flat0"].state_dict()[key]), key)
        x = {m: torch.randn(2, d) for m, d in dict(text=768, audio=768, vision=384).items()}
        for case in ("flat0", "sum0"):
            model = models[case]; output = model(x)
            torch.testing.assert_close(output["additive_joint"], output["text_reference"] + output["annotation_correction"])
            loss, _ = reference_objective(output, torch.tensor([-.4, .6]), torch.tensor([.2, -.2]), torch.ones(2, dtype=torch.bool), False, .3)
            loss.backward()
            self.assertIsNotNone(model.text[1].weight.grad)  # text pathway active even flat
            if case == "flat0":
                self.assertIsNone(model.text_head.weight.grad); self.assertIsNone(model.correction_head.weight.grad)
            else:
                self.assertIsNone(model.joint_head.weight.grad)

    def fixture_reference_labels(self, fixture):
        with fixture.csv.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream); fields = list(reader.fieldnames) + ["label_T"]; rows = list(reader)
        for r in rows:
            r["label_T"] = ".2" if r["mode"] == "train" else "must-not-be-parsed"
        with fixture.csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    def test_validation_annotation_isolation_and_exact_target_difference(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            self.fixture_reference_labels(fixture)
            rows = author_rows(fixture.csv, "simsv2"); ds = ReferenceSplit(fixture.cache, "valid", rows, fixture.csv)
            self.assertFalse(ds.reference_valid.any())
            for x in ds.lengths.values():
                x._mmap.close()
            out = {"prediction": torch.tensor([.3, -.4]), "text_reference": torch.tensor([.1, .2]), "annotation_correction": torch.tensor([.4, -.5])}
            y, t = torch.tensor([.6, -.2]), torch.tensor([.2, .1])
            loss, _ = reference_objective(out, y, t, torch.ones(2, dtype=torch.bool), True, .3)
            expected = (out["prediction"] - y).abs().mean() + .3 * ((out["text_reference"] - t).abs().mean() + (out["annotation_correction"] - (y - t)).abs().mean())
            torch.testing.assert_close(loss, expected)
        finally:
            fixture.tearDown()

    def test_exact_reference_epoch_resume(self):
        fixture = fixtures.NativeTrainingTests(); fixture.setUp()
        try:
            self.fixture_reference_labels(fixture)
            config = {"dataset": "simsv2", "author_csv": str(fixture.csv), "cache_dir": str(fixture.cache),
                      "run_root": str(fixture.root / "whole"), "seeds": [13, 42, 2026], "device": "cpu",
                      "text_hidden": 8, "context_hidden": 8, "dropout": .1, "auxiliary_weight": .3,
                      "batch_size": 2, "learning_rate": .001, "weight_decay": .0001, "max_epochs": 3, "patience": 12}
            train_reference(config, "sum1", 13)
            resumed = dict(config, run_root=str(fixture.root / "resumed"))
            train_reference(resumed, "sum1", 13, stop_after=1); train_reference(resumed, "sum1", 13, resume=True)
            states = [torch.load(Path(c["run_root"]) / "sum1/formal/seed_13/last.pt", weights_only=False) for c in (config, resumed)]
            for key in states[0]["model"]:
                self.assertTrue(torch.equal(states[0]["model"][key], states[1]["model"][key]), key)
            for a, b in zip(states[0]["history"], states[1]["history"]):
                self.assertEqual(a["train_loss"], b["train_loss"]); self.assertEqual(a["valid"], b["valid"])
            with self.assertRaises(ValueError):
                train_reference(dict(resumed, auxiliary_weight=.4), "sum1", 13, resume=True)
        finally:
            fixture.tearDown()
