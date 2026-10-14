import unittest
import numpy as np
import torch
from jevaffect.anchor_calibration import CalibrationHead, objective, disagreement, bin_targets, normalizer, normalize, diagnostics, describe


class AnchorCalibrationTests(unittest.TestCase):
    def test_native_edges_and_train_only_normalization(self):
        y = np.array([-.7, -.1, .1, .7], np.float32)
        np.testing.assert_array_equal(bin_targets(y), [0, 1, 2, 3])
        x = np.array([[1., 0.], [3., 0.], [1000., 1000.]], np.float32)
        stats = normalizer(x, [0, 1])
        self.assertEqual(stats["mean"], [2., 0.]); self.assertEqual(stats["scale"], [1., 1.])
        np.testing.assert_array_equal(normalize(x, stats)[:2], [[-1., 0.], [1., 0.]])

    def test_gate_ablation_and_active_gradients(self):
        torch.manual_seed(1); model = CalibrationHead(5, hidden=8, dropout=0).eval()
        x = torch.randn(4, 5); m = torch.tensor([-.8, -.3, .2, .8]); y = torch.tensor([-1., -.2, 0., 1.])
        d = torch.randn(4, 6)
        for case in ("scale", "mixture"):
            first = model(x, d, m, case); second = model(x, d * 100, m, case)
            torch.testing.assert_close(first["bins"], second["bins"], rtol=0, atol=0)
        first = model(x, d, m, "disagreement"); second = model(x, d*100, m, "disagreement")
        self.assertGreater(float((first["bins"] - second["bins"]).abs().max().detach()), 1e-6)
        loss = objective(first, y, m, "disagreement"); loss.backward()
        self.assertTrue(torch.isfinite(model.weight_head.weight.grad).all())
        self.assertGreater(float(model.weight_head.weight.grad[:, -6:].abs().sum()), 0)

    def test_common_point_and_event_reporting(self):
        torch.manual_seed(2); model = CalibrationHead(3, dropout=0).eval()
        x = torch.randn(3, 3); m = torch.tensor([-1., 0., 1.]); y = np.array([-1., .2, 1.], np.float32)
        d = torch.from_numpy(disagreement(np.array([[-1., 1., 0.], [0., 0., 0.], [1., 0., 1.]])))
        scores = []
        for case in ("scale", "mixture", "disagreement", "direct_hit"):
            out = model(x, d, m, case); described = describe(out)
            report = diagnostics(m.numpy(), y, described); scores.append(report["native"])
            self.assertEqual(report["hit_count"], 2)
            if case == "direct_hit":
                self.assertNotIn("bin_log_score", report); self.assertNotIn("intervals", report)
            else:
                np.testing.assert_allclose(described["median"], m.numpy(), atol=1e-8)
        self.assertTrue(all(s == scores[0] for s in scores))

    def test_known_mixture_density_reference(self):
        torch.manual_seed(3); model = CalibrationHead(4, hidden=8, dropout=0).eval()
        x = torch.randn(3, 4); m = torch.tensor([-1., 0., 1.]); d = torch.randn(3, 6)
        out = model(x, d, m, "mdn"); described = describe(out)
        report = diagnostics(m.numpy(), np.array([-1., .2, 1.], np.float32), described)
        self.assertTrue(np.all(described["bins"] >= 0))
        np.testing.assert_allclose(described["bins"].sum(1), 1.)
        self.assertEqual(report["hit_count"], 2)
        objective(out, torch.tensor([-1., .2, 1.]), m, "mdn").backward()
        self.assertGreater(float(model.mdn_head.weight.grad.abs().sum()), 0)
