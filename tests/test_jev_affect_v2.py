import torch

from jevaffect.jev_affect_v2 import JevAffectV2


def batch():
    torch.manual_seed(9)
    return {
        "text": torch.randn(2, 7, 768),
        "audio": torch.randn(2, 11, 768),
        "vision": torch.randn(2, 5, 384),
        "text_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0]] * 2, dtype=torch.bool),
        "audio_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool),
        "vision_mask": torch.tensor([[1, 1, 1, 0, 0]] * 2, dtype=torch.bool),
        "audio_times": torch.arange(11).float()[None, :].repeat(2, 1) * 0.02,
        "vision_times": torch.arange(5).float()[None, :].repeat(2, 1) * 0.5,
    }


def test_utility_attention_and_padding_contract():
    model = JevAffectV2(dropout=0).eval()
    source = batch()
    baseline = model(source, return_attention=True)
    assert baseline["logits"].shape == (2, 5)
    assert baseline["text_to_av_attention"].shape == (2, 7, 16)
    assert torch.isfinite(baseline["logits"]).all()
    changed = {name: value.clone() for name, value in source.items()}
    for kind in ("text", "audio", "vision"):
        changed[kind][~changed[kind + "_mask"]] = 999
    assert torch.allclose(baseline["logits"], model(changed)["logits"], atol=1e-5)
    targets = torch.full((2, 8, 2), float("nan"))
    targets[baseline["valid_windows"]] = 0.4
    supervised = model(source, utility_targets=targets, utility_weight=0.2)
    assert supervised["utility_loss"] > 0
    loss = torch.nn.functional.cross_entropy(supervised["logits"], torch.tensor([0, 4]))
    loss.backward()
    assert model.cross.in_proj_weight.grad is not None


def test_missing_av_and_no_cross_are_finite():
    model = JevAffectV2(dropout=0).eval()
    source = batch()
    source["audio_mask"][1] = False
    source["vision_mask"][1] = False
    with torch.no_grad():
        cross = model(source)
        assert torch.isfinite(cross["logits"]).all()
        assert torch.all(cross["cross_delta_logits"][1] == 0)
        model.cross_attention = False
        removed = model(source)
        assert torch.isfinite(removed["logits"]).all()
        assert torch.all(removed["cross_delta_logits"] == 0)
