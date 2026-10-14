import unittest
import torch
from jevaffect import shape_controls as s


class ShapeControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.m = torch.tensor([-1., -.6, 0., .37, 1.], dtype=torch.float64)

    def test_all_cases_probability_law_medians_and_active_gradients(self):
        x = torch.randn(5, 257)
        for case in s.CASES:
            model = s.ShapeHead(case, dropout=0).eval()
            out = model(x, self.m)
            grid = torch.linspace(-1.1, 1.1, 101, dtype=torch.float64)[None].expand(5, -1)
            f = s.cdf(out, grid)
            self.assertTrue(torch.all(f[:, 1:] >= f[:, :-1]-1e-12), case)
            p = s.probabilities(out, [-.7, -.1, .1, .7])
            torch.testing.assert_close(p.sum(-1), torch.ones(5, dtype=torch.float64))
            self.assertTrue(torch.all(p >= 0), case)
            if case != 'mdn':
                torch.testing.assert_close(s.quantiles(out, [.5])[:, 0], self.m)
                torch.testing.assert_close(s.cdf(out, self.m)[1:4], torch.full((3,), .5, dtype=torch.float64), atol=1e-12, rtol=0)
            loss = -p[torch.arange(5), torch.tensor([0, 1, 2, 3, 4])].clamp_min(1e-12).log().mean()
            loss.backward()
            self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters()), case)
            self.assertTrue(any(torch.count_nonzero(t.grad) > 0 for t in model.parameters()), case)

    def test_implicit_mdn_gradient_matches_actual_final_cdf_finite_difference(self):
        m = torch.tensor([.23], dtype=torch.float64, requires_grad=True)
        scale = torch.tensor([-.9], dtype=torch.float64, requires_grad=True)
        logits = torch.linspace(-.5, .8, 15, dtype=torch.float64)[None].requires_grad_()
        def score(a, b):
            return s.mdn_cdf(s.mdn_parameters(m, a, b), torch.tensor([.61], dtype=torch.float64)).sum()
        score(scale, logits).backward()
        self.assertIsNone(m.grad)
        eps = 1e-5
        finite_scale = (score(scale.detach()+eps, logits.detach())-score(scale.detach()-eps, logits.detach()))/(2*eps)
        torch.testing.assert_close(scale.grad[0], finite_scale, rtol=2e-5, atol=1e-8)
        for k in (0, 7, 14):
            plus, minus = logits.detach().clone(), logits.detach().clone()
            plus[0, k] += eps; minus[0, k] -= eps
            fd = (score(scale.detach(), plus)-score(scale.detach(), minus))/(2*eps)
            torch.testing.assert_close(logits.grad[0, k], fd, rtol=2e-5, atol=1e-8)

    def test_hit_event_uses_raw_point_and_includes_endpoint_atoms(self):
        p = s.mdn_parameters(self.m, torch.zeros(5), torch.zeros(5, 15))
        out = dict(kind='mdn', parameters=p)
        raw = torch.tensor([-1.2, -.6, 0., .37, 1.2], dtype=torch.float64)
        hit = s.hit_probability(out, raw, .1)
        self.assertEqual(float(hit[0]), 0.)
        self.assertEqual(float(hit[-1]), 0.)
        at_bound = s.hit_probability(out, self.m, .1)
        self.assertGreater(float(at_bound[0]), .5)
        self.assertGreater(float(at_bound[-1]), .5)


if __name__ == '__main__':
    unittest.main()
