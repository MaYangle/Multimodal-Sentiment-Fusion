import unittest
import numpy as np
from jevaffect.native_metrics import native_metrics


class NativeMetricTests(unittest.TestCase):
    def test_sims_closed_right_boundaries(self):
        # -0.1 is negative under author Acc3/Acc5; neutral under old task.
        y = np.array([-.7, -.1, 0, .1, .7], np.float32)
        p = np.array([-.8, -.2, 0, 0, .2], np.float32)
        m = native_metrics(p, y, "simsv2")
        self.assertEqual(m["Mult_acc_5"], 1)
        self.assertEqual(m["Mult_acc_2"], .8)
        self.assertEqual(m["weak_count"], 3)

    def test_simsv2_clip_applies_to_regression(self):
        m = native_metrics([4., -5.], [1., -1.], "simsv2")
        self.assertEqual(m["MAE"], 0)
        self.assertAlmostEqual(m["Corr"], 1)

    def test_mosei_neutral_and_round_to_even(self):
        m = native_metrics([0., 0., 1.5, 9.], [0., -1., 2., 3.], "mosei")
        self.assertEqual(m["nonzero_count"], 3)
        self.assertEqual(m["Has0_acc_2"], .75)
        self.assertEqual(m["Non0_acc_2"], 1)
        self.assertEqual(m["Mult_acc_7"], .75)
        self.assertEqual(m["MAE"], 1.875)

    def test_undefined_and_rejected(self):
        m = native_metrics([0., 0.], [0., 0.], "mosei")
        self.assertIsNone(m["Corr"])
        self.assertIsNone(m["Non0_acc_2"])
        for p, y in (([], []), ([np.nan], [0]), ([0], [4]), ([0, 1], [0])):
            with self.assertRaises(ValueError):
                native_metrics(p, y, "mosei")
