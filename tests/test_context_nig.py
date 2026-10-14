import json
import unittest
from pathlib import Path
import numpy as np
import torch
import test_native_training as fixtures
from jevaffect.context_nig import ContextNIGRegressor, build_context_model
from jevaffect.context_training import summed_objective, train_case
from jevaffect.native_training import digest
from jevaffect.nig import nig_loss


class ContextNIGTests(unittest.TestCase):
    def test_missing_input_cannot_contaminate_joint_expert(self):
        model = ContextNIGRegressor({m: 4 for m in ("text", "audio", "vision")}, hidden=8, joint_hidden=4, dropout=0).eval()
        x = {m: torch.randn(2, 4) for m in ("text", "audio", "vision")}
        x["availability"] = torch.tensor([[True, False, True], [True, False, False]])
        y = {**x, "audio": torch.full((2, 4), float("nan")), "vision": x["vision"].clone()}
        y["vision"][1] = float("nan")
        a, b = model(x), model(y)
        torch.testing.assert_close(a["fused"], b["fused"])
        self.assertEqual(a["branches"].shape, (2, 4, 4))
        self.assertTrue(torch.all(a["weights"][~a["availability"]] == 0))
        with self.assertRaises(ValueError):
            model({**x, "availability": torch.zeros(2, 3, dtype=torch.bool)})

    def test_source_sum_and_joint_removal_gradient(self):
        model = ContextNIGRegressor({m: 4 for m in ("text", "audio", "vision")}, hidden=8, joint_hidden=4, dropout=0, enable_joint=False)
        x = {m: torch.randn(2, 4) for m in ("text", "audio", "vision")}
        x["availability"] = torch.ones(2, 3, dtype=torch.bool)
        out = model(x); target = torch.tensor([-.6, .4])
        loss, _ = summed_objective(out, target, .01)
        expected = nig_loss(out["fused"], target, .01)
        for i in range(3):
            expected = expected + nig_loss(out["branches"][:, i], target, .01)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.all(p.grad == 0) for p in model.joint.parameters()))
        self.assertFalse(out["availability"][:, 3].any())

    def test_capacity_reference_and_same_initialization_for_direct_ablation(self):
        dims = dict(text=768, audio=768, vision=384)
        config = {"three_hidden": 256, "joint_branch_hidden": 192, "joint_hidden": 64, "dropout": .1}
        models = {}
        for name in ("three_dynamic", "joint_dynamic", "joint_off_dynamic"):
            torch.manual_seed(13)
            models[name] = build_context_model(name, dims, config)
        count = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(abs(count(models["joint_dynamic"]) / count(models["three_dynamic"]) - 1), .002)
        for key, value in models["joint_dynamic"].state_dict().items():
            self.assertTrue(torch.equal(value, models["joint_off_dynamic"].state_dict()[key]), key)

    def test_exact_context_resume(self):
        fixture = fixtures.NativeTrainingTests()
        fixture.setUp()
        try:
            meta = json.loads((fixture.cache / "meta.json").read_text())
            for split in ("train", "valid"):
                for m in ("text", "audio", "vision"):
                    path = fixture.cache / split / (m + "_lengths.npy")
                    np.save(path, np.array([1, 2], np.int64))
                    meta["artifact_sha256"][split + "/" + path.name] = digest(path)
            (fixture.cache / "meta.json").write_text(json.dumps(meta))
            config = {"dataset": "simsv2", "author_csv": str(fixture.csv), "cache_dir": str(fixture.cache),
                      "run_root": str(fixture.root / "whole"), "seeds": [13, 42, 2026], "device": "cpu",
                      "three_hidden": 8, "joint_branch_hidden": 8, "joint_hidden": 4, "dropout": .1,
                      "batch_size": 2, "learning_rate": .0001, "weight_decay": .0001, "max_epochs": 3, "patience": 12, "regularization": .01}
            train_case(config, "joint_dynamic", 13)
            resumed = dict(config, run_root=str(fixture.root / "resumed"))
            train_case(resumed, "joint_dynamic", 13, stop_after=1)
            train_case(resumed, "joint_dynamic", 13, resume=True)
            states = [torch.load(Path(c["run_root"]) / "joint_dynamic/formal/seed_13/last.pt", weights_only=False) for c in (config, resumed)]
            for key in states[0]["model"]:
                self.assertTrue(torch.equal(states[0]["model"][key], states[1]["model"][key]), key)
            for a, b in zip(states[0]["history"], states[1]["history"]):
                self.assertEqual(a["train_loss"], b["train_loss"])
                self.assertEqual(a["valid"], b["valid"])
        finally:
            fixture.tearDown()
