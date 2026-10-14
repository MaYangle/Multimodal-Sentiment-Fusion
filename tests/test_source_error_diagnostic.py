import unittest
import numpy as np
from scripts.diagnose_source_regression_errors import decompose


class SourceErrorDiagnosticTests(unittest.TestCase):
    def test_constant_source_offsets_cancel_only_in_pairs(self):
        ids=["a/0","a/1","b/0","b/1","b/2"]
        y=np.array([-.5,.5,-.4,0,.4])
        p=y+np.array([.2,.2,-.1,-.1,-.1])
        r,b,w=decompose(p,y,ids)
        self.assertAlmostEqual(r["MSE"],.022)
        self.assertAlmostEqual(r["source_bias_MSE"],r["MSE"])
        self.assertAlmostEqual(r["within_source_MSE"],0)
        self.assertAlmostEqual(r["within_source_pair_MAE_equal_source"],0)
        np.testing.assert_allclose(b,[.04,.04,.01,.01,.01])
        np.testing.assert_allclose(w,0,atol=1e-16)

    def test_within_source_variation_and_clip_weighting(self):
        ids=["a/0","a/1","b/0","b/1","b/2"]
        r,b,w=decompose(np.array([-.2,.2,-.1,0,.1]),np.zeros(5),ids)
        self.assertAlmostEqual(r["source_bias_MSE"],0)
        self.assertAlmostEqual(r["MSE"],.02)
        self.assertAlmostEqual(r["within_source_MSE"],.02)
        self.assertAlmostEqual(r["within_source_pair_MAE_equal_source"],(.4+.4/3)/2)
        self.assertAlmostEqual(r["within_source_pair_MAE_pair_weighted"],.2)
        np.testing.assert_allclose(b,0)
        np.testing.assert_allclose(w,[.04,.04,.01,0,.01])


if __name__=="__main__":
    unittest.main()
