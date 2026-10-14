import unittest
import numpy as np
from jevaffect.regression_uncertainty import uncertainty_metrics


class RegressionUncertaintyTests(unittest.TestCase):
    def test_t_density_closed_form_and_intervals(self):
        # v=1, a=2, b=2 => df4, scale sqrt2, predictive variance4.
        p = np.tile([0., 1., 2., 2.], (4, 1))
        r = uncertainty_metrics(p, np.zeros(4), [-1, 1])
        self.assertAlmostEqual(r["mean_predictive_variance"], 4)
        self.assertAlmostEqual(r["density_NLL"], -np.log(3 / (8 * np.sqrt(2))))
        self.assertIsNone(r["error_vs_variance_spearman"])
        for metrics in r["intervals"].values():
            self.assertEqual(metrics["coverage"], 1)
            self.assertLessEqual(metrics["range_intersection_mean_width"], 2)

    def test_variance_ranks_error_and_invalid_input(self):
        params = np.array([[0., 1., 3., .01], [0., 1., 3., .5], [0., 1., 3., 2.], [0., 1., 3., 4.]])
        r = uncertainty_metrics(params, [.01, .1, .5, .9], [-1, 1])
        self.assertAlmostEqual(r["error_vs_variance_spearman"], 1)
        self.assertLess(r["selective_MAE"]["0.25"]["MAE"], r["selective_MAE"]["1.0"]["MAE"])
        for bad in (params[:2], np.full((4, 4), np.nan), np.array([[0., 0., 2., 1.]] * 4)):
            with self.assertRaises(ValueError):
                uncertainty_metrics(bad, [0, 0, 0, 0], [-1, 1])
