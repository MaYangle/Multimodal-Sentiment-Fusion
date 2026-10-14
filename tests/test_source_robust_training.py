import json
import unittest
from pathlib import Path
import torch
from tests import test_anchor_crossfit as anchor_fixtures
from jevaffect.anchor_crossfit import prepare
from jevaffect.source_robust import CASES
from jevaffect.source_robust_training import train_fold,run_suite


class SourceRobustTrainingTests(unittest.TestCase):
    def setUp(self):
        # Reuse the TRAIN/invalid-VALID/invalid-TEST fixture's data preparation.
        self.fixture=anchor_fixtures.AnchorCrossfitTests(methodName="test_source_cohorts_are_disjoint_and_every_train_id_is_held_once")
        self.fixture.setUp();self.root=self.fixture.root
        self.config={**self.fixture.config,"predict_valid":False,"cases":list(CASES),"dro_step_size":.01,"dann_gradient_scale":.1}
        self.data=prepare(self.config,allow_fixture=True)

    def tearDown(self):
        self.fixture.tearDown()

    def test_exact_group_weight_sampler_and_adversarial_epoch_resume(self):
        for case in ("group_dro","dann"):
            cfg={**self.config,"run_root":str(self.root/f"{case}-uninterrupted")}
            a=train_fold(cfg,self.data,case,13,0)
            cfg2={**cfg,"run_root":str(self.root/f"{case}-resume")}
            train_fold(cfg2,self.data,case,13,0,stop_after=1)
            b=train_fold(cfg2,self.data,case,13,0,resume=True)
            first=torch.load(Path(a["root"])/"best.pt",weights_only=False)
            second=torch.load(Path(b["root"])/"best.pt",weights_only=False)
            for name,value in first["model"].items():
                self.assertTrue(torch.equal(value,second["model"][name]),name)
            self.assertEqual(a["oof_native"],b["oof_native"])
            h=[json.loads((Path(r["root"])/"history.json").read_text()) for r in (a,b)]
            self.assertEqual([{k:v for k,v in row.items() if k!="seconds"} for row in h[0]],
                             [{k:v for k,v in row.items() if k!="seconds"} for row in h[1]])
            c=train_fold(cfg2,self.data,case,13,0,resume=True)
            self.assertEqual(c["optimizer_updates"],b["optimizer_updates"])
            with self.assertRaises(ValueError):
                train_fold({**cfg2,"dro_step_size":.02},self.data,case,13,0,resume=True)

    def test_detached_domain_probe_point_path_matches_balanced_erm(self):
        a=train_fold(self.config,self.data,"balanced_erm",42,1)
        b=train_fold(self.config,self.data,"domain_probe",42,1)
        self.assertEqual(a["oof_native"],b["oof_native"])
        self.assertEqual(a["best_epoch"],b["best_epoch"])
        for checkpoint in ("best.pt","last.pt"):
            states=[torch.load(Path(r["root"])/checkpoint,weights_only=False)["model"] for r in (a,b)]
            for name in states[0]:
                if name.startswith("regressor."):
                    self.assertTrue(torch.equal(states[0][name],states[1][name]),name)

    def test_train_only_and_partial_matrix_scope(self):
        report=run_suite(self.config,self.root/"partial.json",case="clip_erm",seed=13,fold=0,stop_after=1,allow_fixture=True)
        self.assertFalse(report["suite_complete"])
        self.assertFalse(report["official_VALID_used"])
        self.assertFalse(report["test_evaluated"])
        identity=json.loads((Path(report["results"][0]["root"])/"identity.json").read_text())
        groups=[{s.split("/")[0] for s in identity["cohort"][k]} for k in ("fit_ids","select_ids","oof_ids")]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])


if __name__=="__main__":
    unittest.main()
