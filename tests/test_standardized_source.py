import unittest
import numpy as np
from tests import test_anchor_crossfit as anchor_fixtures
from jevaffect.anchor_crossfit import prepare
from jevaffect.standardized_source import transform_fold,run


class StandardizedSourceTests(unittest.TestCase):
    def setUp(self):
        self.fixture=anchor_fixtures.AnchorCrossfitTests(methodName="test_source_cohorts_are_disjoint_and_every_train_id_is_held_once")
        self.fixture.setUp();self.root=self.fixture.root;self.config={**self.fixture.config,"predict_valid":False}
        self.data=prepare(self.config,allow_fixture=True)

    def tearDown(self):self.fixture.tearDown()

    def test_scaler_fits_only_optimization_sources_and_preserves_raw(self):
        cfg,data=transform_fold(self.config,self.data,0)
        indices=cfg["preprocessing"]["fit_indices_in_cache_order"]
        original={m:x.copy() for m,x in self.data["train"].features.items()}
        for m,x in original.items():
            np.testing.assert_allclose(data["train"].features[m][indices].mean(0),0,atol=1e-6)
            np.testing.assert_allclose(data["train"].features[m][indices].std(0),1,atol=1e-6)
            held=[i for i in range(len(x)) if i not in indices]
            self.data["train"].features[m][held]+=100
        new_cfg,_=transform_fold(self.config,self.data,0)
        self.assertEqual(cfg["preprocessing"]["statistics"],new_cfg["preprocessing"]["statistics"])
        self.assertEqual(data["train"].ids,self.data["train"].ids)
        np.testing.assert_array_equal(data["train"].targets,self.data["train"].targets)
        for m,x in original.items():np.testing.assert_array_equal(self.data["train"].features[m][indices],x[indices])

    def test_real_wrapper_fixture_train_resume_and_no_valid_input(self):
        a=run(self.config,self.root/"first.json",seed=13,fold=0,stop_after=1,allow_fixture=True)
        b=run(self.config,self.root/"resume.json",seed=13,fold=0,resume=True,allow_fixture=True)
        self.assertFalse(a["suite_complete"]);self.assertFalse(b["official_VALID_used"])
        self.assertEqual(b["results"][0]["epochs_executed"],3)
        self.assertFalse(b["test_evaluated"])


if __name__=="__main__":unittest.main()
