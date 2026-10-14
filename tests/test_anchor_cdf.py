import unittest
import torch
from jevaffect.anchor_cdf import parameters, cdf, bin_probabilities, error_band_probability, quantile


class AnchorCDFTests(unittest.TestCase):
    def test_anchor_generalized_median_and_probability_law(self):
        m = torch.tensor([-1., -.9999, -.5, 0., .5, .9999, 1.], dtype=torch.float64)
        for z in (-40., 0., 40.):
            p = parameters(m, torch.full_like(m, -.8), torch.full_like(m, z))
            mid = quantile(p, .5)
            torch.testing.assert_close(mid, m, atol=1e-8, rtol=0)
            torch.testing.assert_close(cdf(p, m)[1:-1], torch.full_like(m[1:-1], .5), atol=1e-12, rtol=0)
            grid = torch.linspace(-1.1, 1.1, 101)[None].expand(len(m), -1)
            values = cdf(p, grid)
            self.assertTrue(torch.all(values[:, 1:] >= values[:, :-1]))
            self.assertTrue(torch.all(values[:, 0] == 0)); self.assertTrue(torch.all(values[:, -1] == 1))
            probabilities = bin_probabilities(p, [-.7, -.1, .1, .7])
            self.assertTrue(torch.all(probabilities >= 0))
            torch.testing.assert_close(probabilities.sum(1), torch.ones_like(m), atol=1e-12, rtol=0)

    def test_finite_gradients_and_frozen_point(self):
        m = torch.tensor([-.8, 0., .6], requires_grad=True)
        scale = torch.tensor([-10., -.5, 1.], requires_grad=True)
        weight = torch.tensor([-20., 0., 20.], requires_grad=True)
        p = parameters(m, scale, weight)
        probabilities = bin_probabilities(p, [-.7, -.1, .1, .7])
        loss = -probabilities[torch.arange(3), torch.tensor([0, 2, 4])].clamp_min(1e-12).log().mean()
        loss.backward()
        self.assertIsNone(m.grad)
        self.assertTrue(torch.isfinite(scale.grad).all()); self.assertTrue(torch.isfinite(weight.grad).all())

    def test_no_mixture_is_centered_logistic_and_endpoints_retained(self):
        m = torch.tensor([-.4, .4], dtype=torch.float64)
        p = parameters(m, torch.zeros_like(m), torch.ones_like(m), mixture=False)
        torch.testing.assert_close(p["location"], m, rtol=0, atol=0)
        at_lower = cdf(p, -1.)
        self.assertTrue(torch.all(at_lower > 0))
        all_mass = error_band_probability(p, 2.)
        torch.testing.assert_close(all_mass, torch.ones_like(m), rtol=0, atol=0)
        self.assertTrue(torch.all(error_band_probability(p, .1) >= 0))
        torch.testing.assert_close(quantile(p, [0., 1.]), torch.tensor([[-1., 1.], [-1., 1.]], dtype=torch.float64), rtol=0, atol=0)

    def test_native_units_and_invalid_contracts(self):
        m = torch.tensor([-2., 0., 2.])
        p = parameters(m, torch.zeros_like(m), torch.zeros_like(m), bound=3., kernel_scale=.25)
        torch.testing.assert_close(quantile(p, .5), m.double(), rtol=0, atol=1e-8)
        with self.assertRaises(ValueError):
            parameters(m, torch.zeros_like(m), torch.zeros_like(m))
        with self.assertRaises(ValueError):
            bin_probabilities(p, [1., -1.])
        with self.assertRaises(ValueError):
            error_band_probability(p, 0.)
