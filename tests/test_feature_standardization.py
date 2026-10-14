import unittest
import numpy as np
from jevaffect.feature_standardization import fit_pooled_standardization, standardize_sequence


class FeatureStandardizationTests(unittest.TestCase):
    def test_train_statistics_constant_dimension_and_prefix_mask(self):
        stats = fit_pooled_standardization([[1., 2.], [3., 2.]])
        np.testing.assert_allclose(stats["mean"], [2., 2.]); np.testing.assert_allclose(stats["scale"], [1., 1.])
        self.assertEqual(stats["constant_or_tiny_coordinates"], 1)
        normalized = standardize_sequence([[1., 2.], [3., 2.], [1e10, -1e10]], 2, stats)
        np.testing.assert_array_equal(normalized, [[-1., 0.], [1., 0.], [0., 0.]])
        np.testing.assert_array_equal(normalized[:2].mean(0), [0., 0.])

    def test_validation_shift_cannot_refit_statistics(self):
        stats = fit_pooled_standardization([[1., 2.], [3., 2.]])
        held = standardize_sequence([[100., 7.], [102., 7.]], 2, stats)
        np.testing.assert_array_equal(held.mean(0), [99., 5.])
        self.assertEqual(stats["mean"], [2., 2.])
