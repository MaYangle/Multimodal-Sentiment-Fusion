import unittest
import numpy as np
from scipy.stats import t
from jevaffect.student_pool import components, pool_quantile


class StudentPoolTests(unittest.TestCase):
    def test_identical_distribution_quantile_and_duplication_invariance(self):
        p = np.array([[[.2, 1., 2., .8], [.2, 1., 2., .8]]])
        a = np.ones((1, 2), bool)
        q = pool_quantile(p, a, .95)
        expected = t.ppf(.95, df=4, loc=.2, scale=np.sqrt(.8))
        self.assertAlmostEqual(float(q[0]), expected)
        duplicated = np.tile(p, (1, 2, 1))
        np.testing.assert_allclose(pool_quantile(duplicated, np.ones((1, 4), bool), .95), q, rtol=0, atol=1e-10)

    def test_masks_and_preserved_weighted_mean(self):
        p = np.array([[[0., 1., 2., .8], [1., 3., 2., .8], [99., 4., 2., .8]]])
        a = np.array([[True, True, False]])
        mu, _, _, w, _ = components(p, a)
        self.assertAlmostEqual(float((mu * w).sum()), .75)
        np.testing.assert_allclose(w, [[.25, .75, 0]])
        p[:, 2, 0] = -99
        self.assertTrue(np.isfinite(pool_quantile(p, a, .5)).all())
        with self.assertRaises(ValueError):
            components(p, np.zeros_like(a))
