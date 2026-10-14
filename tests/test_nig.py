import unittest

import torch

from jevaffect.nig import fuse_nig, nig_loss, predictive_distribution, predictive_variance
from jevaffect.reliability import ReliabilityRegressor


class TestNIG(unittest.TestCase):
    def test_extreme_raw_evidence_preserves_strict_variance_domain(self):
        model = ReliabilityRegressor({"text": 2, "audio": 2, "vision": 2}, hidden=4, dropout=0.)
        for head in model.distribution_heads.values():
            torch.nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(torch.tensor([0., -100., -100., -100.]))
        x = {m: torch.zeros(1, 2) for m in ("text", "audio", "vision")}
        x["availability"] = torch.tensor([[True, False, False]])
        output = model(x)
        self.assertTrue(torch.all(output["fused"][:, 2] > 1))
        self.assertTrue(torch.isfinite(predictive_variance(output["fused"])).all())

    def test_dynamic_fusion_matches_author_pairwise_nig_sum_and_masks_absent_branch(self):
        parameters = torch.tensor([[
            [0.0, 1.0, 2.0, 0.5],
            [2.0, 3.0, 3.0, 0.7],
            [100.0, 20.0, 5.0, 1.0],
        ]])
        available = torch.tensor([[True, True, False]])
        fused, weights = fuse_nig(parameters, available, mode="dynamic")

        mu = (1.0 * 0.0 + 3.0 * 2.0) / 4.0
        expected = torch.tensor([[
            mu,
            4.0,
            2.0 + 3.0 + 0.5,
            0.5 + 0.7 + 0.5 * (1.0 * (0.0 - mu) ** 2 + 3.0 * (2.0 - mu) ** 2),
        ]])
        torch.testing.assert_close(fused, expected)
        torch.testing.assert_close(weights, torch.tensor([[0.25, 0.75, 0.0]]))

    def test_uniform_control_uses_equal_mean_but_preserves_total_gamma(self):
        parameters = torch.tensor([[
            [-1.0, 1.0, 2.0, 0.5],
            [3.0, 3.0, 3.0, 0.7],
            [40.0, 8.0, 4.0, 1.0],
        ]])
        available = torch.tensor([[True, True, False]])
        uniform, weights = fuse_nig(parameters, available, mode="uniform")
        dynamic, _ = fuse_nig(parameters, available, mode="dynamic")

        self.assertAlmostEqual(uniform[0, 0].item(), 1.0)
        self.assertAlmostEqual(uniform[0, 1].item(), dynamic[0, 1].item())
        self.assertAlmostEqual(uniform[0, 2].item(), dynamic[0, 2].item())
        expected_beta = 0.5 + 0.7 + 0.5 * 2.0 * ((-1.0 - 1.0) ** 2 + (3.0 - 1.0) ** 2)
        self.assertAlmostEqual(uniform[0, 3].item(), expected_beta, places=6)
        torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5, 0.0]]))

    def test_nig_loss_matches_student_t_nll_and_regularizer(self):
        parameters = torch.tensor(
            [[0.2, 1.5, 2.0, 0.8], [-0.1, 0.7, 3.0, 0.4]], dtype=torch.float64
        )
        target = torch.tensor([0.9, 0.4], dtype=torch.float64)
        coefficient = 0.02
        actual = nig_loss(parameters, target, regularization=coefficient, reduction="none")
        distribution = predictive_distribution(parameters)
        regularizer = coefficient * (target - parameters[:, 0]).abs() * (
            2 * parameters[:, 1] + parameters[:, 2]
        )
        torch.testing.assert_close(actual, -distribution.log_prob(target) + regularizer)

        expected_variance = parameters[:, 3] * (1 + parameters[:, 1]) / (
            parameters[:, 1] * (parameters[:, 2] - 1)
        )
        torch.testing.assert_close(predictive_variance(parameters), expected_variance)
        torch.testing.assert_close(distribution.variance, expected_variance)

    def test_loss_backward_is_finite_at_exact_mean_for_branch_batches(self):
        parameters = torch.tensor(
            [[[0.0, 1.0, 2.0, 0.5], [0.5, 2.0, 1.5, 0.7]]], requires_grad=True
        )
        target = torch.tensor([[0.0, 0.5]])
        loss = nig_loss(parameters, target, reduction="mean")
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parameters.grad).all())

    def test_model_shapes_masks_and_does_not_infer_missing_from_zero(self):
        torch.manual_seed(3)
        model = ReliabilityRegressor({"text": 2, "audio": 3, "vision": 1}, hidden=4, dropout=0.0)
        inputs = {
            "text": torch.zeros(2, 2),
            "audio": torch.ones(2, 3),
            "vision": torch.ones(2, 1),
            "availability": torch.tensor([[True, False, True], [False, True, True]]),
        }
        output = model(inputs)
        self.assertEqual(output["fused"].shape, (2, 4))
        self.assertEqual(output["branches"].shape, (2, 3, 4))
        self.assertEqual(output["unimodal_prediction"].shape, (2, 3))
        self.assertEqual(output["weights"].shape, (2, 3))
        self.assertEqual(output["prediction"].shape, (2,))
        self.assertTrue(torch.equal(output["prediction"], output["fused"][:, 0]))
        self.assertTrue(torch.all(output["branches"][..., 1:] > 0))
        self.assertTrue(torch.equal(output["weights"] > 0, inputs["availability"]))

        no_mask = model({key: value for key, value in inputs.items() if key != "availability"})
        self.assertTrue(torch.all(no_mask["availability"]))
        with self.assertRaisesRegex(ValueError, "at least one available"):
            model({**inputs, "availability": torch.zeros(2, 3, dtype=torch.bool)})
