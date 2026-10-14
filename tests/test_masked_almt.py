import unittest
import torch
from jevaffect.almt import ALMTClassifier
from jevaffect.masked_almt import MaskedALMTRegressor


class MaskedALMTTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(17)
        kwargs = dict(feature_dims={m: 4 for m in ("text", "audio", "vision")},
                      lengths={m: 5 for m in ("text", "audio", "vision")}, dim=8,
                      token_len=2, proj_heads=2, proj_mlp_dim=8, l_enc_heads=2,
                      l_enc_mlp_dim=8, ahl_heads=2, ahl_dim_head=4,
                      fusion_heads=2, fusion_mlp_dim=8, fusion_depth=1)
        return MaskedALMTRegressor(**kwargs).eval(), kwargs

    def test_fully_valid_upstream_parity(self):
        model, kwargs = self.model()
        original = ALMTClassifier(**kwargs)
        original.classifier = torch.nn.Linear(8, 1)
        original.load_state_dict(model.state_dict()); original.eval()
        features = {m: torch.randn(2, 5, 4) for m in ("text", "audio", "vision")}
        features["lengths"] = {m: torch.tensor([5, 5]) for m in ("text", "audio", "vision")}
        torch.testing.assert_close(model(features), original(features), rtol=0, atol=0)

    def test_padding_values_layout_and_gradients(self):
        model, _ = self.model()
        mods = ("text", "audio", "vision")
        features = {m: torch.randn(2, 5, 4, requires_grad=True) for m in mods}
        features["lengths"] = {m: torch.tensor([3, 2]) for m in mods}
        # An interior zero is a valid observation; it must not be dropped.
        with torch.no_grad():
            features["text"][0, 1] = 0
        baseline = model(features)
        baseline.sum().backward()
        moved = {}; masks = {}
        for m in mods:
            moved[m] = torch.full_like(features[m], float("nan"))
            masks[m] = torch.zeros(2, 5, dtype=torch.bool)
            for i, n in enumerate((3, 2)):
                moved[m][i, 5-n:] = features[m].detach()[i, :n]
                masks[m][i, 5-n:] = True
            moved[m].requires_grad_()
        moved["valid_masks"] = masks
        shifted = model(moved); shifted.sum().backward()
        torch.testing.assert_close(shifted, baseline, rtol=1e-5, atol=1e-6)
        for m in mods:
            for i, n in enumerate((3, 2)):
                torch.testing.assert_close(moved[m].grad[i, 5-n:], features[m].grad[i, :n], rtol=1e-4, atol=1e-6)
                self.assertTrue(torch.all(features[m].grad[i, n:] == 0))
                self.assertTrue(torch.all(moved[m].grad[i, :5-n] == 0))

    def test_mask_contract(self):
        model, _ = self.model()
        features = {m: torch.randn(2, 5, 4) for m in ("text", "audio", "vision")}
        features["lengths"] = {m: torch.tensor([0, 6]) for m in ("text", "audio", "vision")}
        with self.assertRaises(ValueError):
            model(features)
        features["valid_masks"] = {m: torch.ones(2, 5) for m in ("text", "audio", "vision")}
        with self.assertRaises(ValueError):
            model(features)
