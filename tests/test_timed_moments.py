import unittest
import torch
from jevaffect.timed_moments import CASES, TimedMomentRegressor


class TimedMomentsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.x = {"text": torch.randn(3, 5), "audio_mean": torch.randn(3, 7),
                  "vision_mean": torch.randn(3, 3), "audio_grid": torch.randn(3, 4, 7),
                  "vision_grid": torch.randn(3, 4, 3), "pair_valid": torch.tensor([True, True, False])}

    def model(self, case):
        return TimedMomentRegressor({"text": 5, "audio": 7, "vision": 3}, case, rank=2, hidden=6, dropout=0)

    def test_equal_parameters_and_real_gradients(self):
        sizes = []
        for case in CASES:
            model = self.model(case)
            output = model(self.x)
            self.assertEqual(output.shape, (3,))
            output.square().sum().backward()
            for p in model.parameters():
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
            sizes.append(sum(p.numel() for p in model.parameters()))
            formal = TimedMomentRegressor({"text": 768, "audio": 768, "vision": 384}, case)
            self.assertEqual(sum(p.numel() for p in formal.parameters()), 494101)
        self.assertEqual(len(set(sizes)), 1)

    def test_common_permutation_and_offset_invariant_interaction(self):
        model = self.model("paired_cov")
        original = model.moment(self.x)
        changed = dict(self.x)
        changed["audio_grid"] = self.x["audio_grid"][:, [2, 0, 3, 1]] + torch.randn(3, 1, 7)
        changed["vision_grid"] = self.x["vision_grid"][:, [2, 0, 3, 1]] + torch.randn(3, 1, 3)
        torch.testing.assert_close(model.moment(changed), original, atol=1e-6, rtol=1e-5)
        changed["audio_mean"] = self.x["audio_mean"] + 2
        self.assertFalse(torch.allclose(model(changed), model(self.x)))

    def test_pair_break_changes_covariance_but_preserves_variance(self):
        changed = dict(self.x)
        changed["vision_grid"] = torch.roll(self.x["vision_grid"], 1, 1)
        paired = self.model("paired_cov")
        self.assertFalse(torch.allclose(paired.moment(changed), paired.moment(self.x)))
        unpaired = self.model("unpaired_cov")
        unpaired.load_state_dict(paired.state_dict())
        torch.testing.assert_close(unpaired.moment(self.x), paired.moment(changed))
        marginal = self.model("marginal_variance")
        torch.testing.assert_close(marginal.moment(changed), marginal.moment(self.x))

    def test_missing_pair_cannot_change_moment_or_global_fallback(self):
        for case in CASES:
            model = self.model(case)
            self.assertEqual(model.moment(self.x)[2].abs().sum().item(), 0)
            changed = {k: v.clone() for k, v in self.x.items()}
            changed["audio_grid"][2] = 1e6
            changed["vision_grid"][2] = -1e6
            torch.testing.assert_close(model(changed), model(self.x))


if __name__ == "__main__":
    unittest.main()
