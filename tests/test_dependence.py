import unittest
import numpy as np
from jevaffect.dependence import covariance_variance, replace_variance, nig_variance, fit_variance_scale, fit_correlation


class DependenceTests(unittest.TestCase):
    def test_known_covariance_and_duplicate_invariance(self):
        p = np.array([[[0., 2., 3., 4.], [1., 1., 4., 2.]]]); a = np.ones((1, 2), bool)
        r = np.array([[1., .7], [.7, 1.]])
        var = covariance_variance(p, a, r)
        std = np.sqrt(nig_variance(p[0])); weights = np.array([2 / 3, 1 / 3])
        self.assertAlmostEqual(var[0], (weights * std) @ r @ (weights * std))
        duplicated = np.tile(p, (1, 2, 1)); dr = np.tile(r, (2, 2))
        np.testing.assert_allclose(covariance_variance(duplicated, np.ones((1, 4), bool), dr), var)
        absent = np.array([[True, False]])
        self.assertAlmostEqual(covariance_variance(p, absent, r)[0], std[0] ** 2)

    def test_variance_change_preserves_location_and_degrees_of_freedom(self):
        p = np.array([[.2, 2., 3., .1], [-.4, 3., 4., .2]])
        q = replace_variance(p, np.array([.5, .3]), 2)
        np.testing.assert_array_equal(q[:, :3], p[:, :3])
        np.testing.assert_allclose(nig_variance(q), [1., .6])

    def test_training_fit_recovers_scale_and_rejects_invalid_correlation(self):
        rng = np.random.default_rng(35); n = 20000
        p = np.tile([0., 1., 4., 3.], (n, 1))
        from scipy.stats import t
        y = t.rvs(8, size=n, random_state=rng) * np.sqrt(1.5 * 2.5)
        fit = fit_variance_scale(p, nig_variance(p), y)
        import json
        json.dumps(fit, allow_nan=False)
        self.assertLess(abs(fit["variance_scale"] - 2.5), .15)
        experts = np.stack([p, p], 1); experts[:, 1, 0] = .1 * rng.normal(size=n)
        shrunk, raw = fit_correlation(experts, np.ones((n, 2), bool), y)
        self.assertGreater(raw[0, 1], .99); self.assertGreater(np.linalg.eigvalsh(shrunk).min(), 0)
        with self.assertRaises(ValueError):
            covariance_variance(experts, np.ones((n, 2), bool), np.array([[1., 2.], [2., 1.]]))
