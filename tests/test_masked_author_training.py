import json
import unittest
from pathlib import Path
import torch
from tests import test_author_features
from jevaffect.masked_author_training import train_masked_author


class MaskedAuthorTrainingTests(unittest.TestCase):
    def test_exact_epoch_resume_and_budget(self):
        fixture = test_author_features.AuthorFeatureTests()
        fixture.setUp()
        try:
            fixture.make_cache()
            config = {"dataset": "mosei", "cohort": "mosei_multibench_aligned50", "model": "almt_masked",
                "cache_dir": str(fixture.cache), "expected_source_sha256": "fixture",
                "run_root": str(fixture.root / "whole"), "seeds": [13, 42, 2026], "device": "cpu",
                "hidden": 256, "dropout": .1, "batch_size": 2, "learning_rate": .0001,
                "weight_decay": .0001, "max_epochs": 2, "patience": 12}
            first = train_masked_author(config, 13)
            resumed = dict(config, run_root=str(fixture.root / "resumed"))
            train_masked_author(resumed, 13, stop_after=1)
            second = train_masked_author(resumed, 13, resume=True)
            # Positional tables depend on this tiny fixture's sequence lengths.
            self.assertEqual(first["parameters"], second["parameters"])
            states = [torch.load(Path(c["run_root"]) / "formal/almt_masked/seed_13/last.pt", weights_only=False)
                      for c in (config, resumed)]
            for name, value in states[0]["model"].items():
                self.assertTrue(torch.equal(value, states[1]["model"][name]), name)
            for a, b in zip(states[0]["history"], states[1]["history"]):
                self.assertEqual(a["train_L1"], b["train_L1"])
                self.assertEqual(a["valid"], b["valid"])
            self.assertEqual(first["valid_metrics"], second["valid_metrics"])
            with self.assertRaises(ValueError):
                train_masked_author(dict(resumed, learning_rate=.002), 13, resume=True)
        finally:
            fixture.tearDown()
