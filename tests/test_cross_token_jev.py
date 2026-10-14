import torch

from jevaffect.cross_token_jev import CrossTokenJev


def sample(batch_size=2):
    torch.manual_seed(8)
    return {
        "text": torch.randn(batch_size, 7, 768),
        "audio": torch.randn(batch_size, 11, 768),
        "vision": torch.randn(batch_size, 5, 384),
        "text_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0]] * batch_size, dtype=torch.bool),
        "audio_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]] * batch_size, dtype=torch.bool),
        "vision_mask": torch.tensor([[1, 1, 1, 0, 0]] * batch_size, dtype=torch.bool),
        "audio_times": torch.arange(11).float()[None, :].repeat(batch_size, 1) * 0.02,
        "vision_times": torch.arange(5).float()[None, :].repeat(batch_size, 1) * 0.5,
    }


def test_shapes_mask_invariance_and_gradients():
    model = CrossTokenJev(dropout=0).eval()
    batch = sample()
    first = model(batch, return_attention=True)
    assert first["logits"].shape == (2, 5)
    assert first["text_to_av_attention"].shape == (2, 7, 16)
    assert torch.allclose(first["modality_weight"].sum(-1), torch.ones(2))
    modified = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    for modality in ("text", "audio", "vision"):
        modified[modality][~modified[modality + "_mask"]] = 999.0
    second = model(modified)
    assert torch.allclose(first["logits"], second["logits"], atol=1e-5)
    model.train()
    loss = torch.nn.functional.cross_entropy(model(batch)["logits"], torch.tensor([0, 4]))
    loss.backward()
    assert model.text_from_av.in_proj_weight.grad is not None
    assert torch.isfinite(model.text_from_av.in_proj_weight.grad).all()


def test_missing_modalities_and_controls_are_finite():
    batch = sample()
    batch["vision_mask"][0] = False
    batch["audio_mask"][1] = False
    batch["vision_mask"][1] = False
    for cross in (True, False):
        for gate in ("learned", "uniform"):
            model = CrossTokenJev(cross_attention=cross, use_av_time=False, gate_kind=gate).eval()
            output = model(batch)
            assert torch.isfinite(output["logits"]).all()
            assert output["modality_weight"][0, 2] == 0
            assert output["modality_weight"][1, 1] == 0
            assert output["modality_weight"][1, 2] == 0
    batch["text_mask"][1] = False
    assert torch.isfinite(CrossTokenJev().eval()(batch)["logits"]).all()


def test_audio_time_changes_output_only_when_enabled():
    batch = sample()
    shifted = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    shifted["audio_times"] *= 3
    with_time = CrossTokenJev(use_av_time=True, dropout=0).eval()
    no_time = CrossTokenJev(use_av_time=False, dropout=0).eval()
    assert not torch.allclose(with_time(batch)["logits"], with_time(shifted)["logits"])
    assert torch.allclose(no_time(batch)["logits"], no_time(shifted)["logits"], atol=1e-5)
