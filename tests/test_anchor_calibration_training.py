import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from jevaffect.native_training import digest
from jevaffect.anchor_calibration_training import fit


class AnchorCalibrationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, name, alter_valid=False):
        directory = self.root / name; part = directory / "seed_13"; part.mkdir(parents=True)
        rng = np.random.default_rng(7); meta = {"dataset": "simsv2", "seeds": [13,42,2026], "artifact_sha256": {},
            "calibration_split": {"fit_indices": list(range(8)), "select_indices": list(range(8,12))}}
        for split, n in (("train",12),("valid",4)):
            y = rng.uniform(-.8,.8,n).astype(np.float32); m = np.clip(y + rng.normal(0,.15,n),-1,1).astype(np.float32)
            if split == "valid" and alter_valid:
                y = -y
            values = {"features": rng.normal(size=(n,5)).astype(np.float32), "anchor": m, "target": y,
                      "views": rng.uniform(-1,1,(n,3)).astype(np.float32)}
            np.savez(part / (split + ".npz"), **values)
            ids = [f"source_{i//2}/clip_{i}" for i in range(n)]
            (part / (split+"_ids.json")).write_text(json.dumps(ids))
        for path in part.iterdir():
            meta["artifact_sha256"][path.relative_to(directory).as_posix()] = digest(path)
        (directory / "metadata.json").write_text(json.dumps(meta))
        return directory

    def config(self, cache, output):
        return {"dataset":"simsv2","table_dir":str(cache),"run_root":str(output),"seeds":[13,42,2026],
            "device":"cpu","hidden":8,"dropout":.1,"max_epochs":3,"patience":12,"batch_size":4,
            "learning_rate":.001,"weight_decay":.0001,"gradient_norm":1.}

    def test_exact_resume_and_identity_guard(self):
        cache = self.cache("cache"); cfg = self.config(cache,self.root/"whole")
        whole = fit(cfg,"disagreement",13,smoke=True)
        separate = dict(cfg,run_root=str(self.root/"resumed"))
        fit(separate,"disagreement",13,stop_after=1,smoke=True)
        restored = fit(separate,"disagreement",13,resume=True,smoke=True)
        states = [torch.load(Path(c["run_root"])/"smoke/disagreement/seed_13/last.pt",weights_only=False) for c in (cfg,separate)]
        for name,tensor in states[0]["model"].items():
            self.assertTrue(torch.equal(tensor,states[1]["model"][name]),name)
        self.assertEqual(whole["valid"],restored["valid"])
        for a,b in zip(states[0]["history"],states[1]["history"]):
            self.assertEqual(a["fit_objective"],b["fit_objective"])
            self.assertEqual(a["selection_objective"],b["selection_objective"])
        with self.assertRaises(ValueError):
            fit(dict(separate,learning_rate=.02),"disagreement",13,resume=True,smoke=True)

    def test_official_valid_targets_do_not_fit_or_select(self):
        configs = [self.config(self.cache(name,alter),self.root/(name+"_runs")) for name,alter in (("one",False),("two",True))]
        results = [fit(c,"mixture",13,smoke=True) for c in configs]
        states = [torch.load(Path(c["run_root"])/"smoke/mixture/seed_13/last.pt",weights_only=False) for c in configs]
        for name,tensor in states[0]["model"].items():
            self.assertTrue(torch.equal(tensor,states[1]["model"][name]),name)
        self.assertEqual(results[0]["best_epoch"],results[1]["best_epoch"])
        self.assertNotEqual(results[0]["valid"],results[1]["valid"])
