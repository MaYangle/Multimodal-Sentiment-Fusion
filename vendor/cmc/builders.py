"""Build CMC pseudo-pretrain/fusion models from canonical cache schema."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .cache_adapter import CMCFeatureSplit
from .models import (
    CMCClassifier,
    CMCModelError,
    CMCUnimodalModel,
    ContextualTextEncoder,
    LocalBertTokenEncoder,
    MODALITIES,
    transfer_unimodal_to_cmc,
)


def _make_text_encoder(
    split: CMCFeatureSplit,
    *,
    bert_model_dir: str | None,
    injected_text_encoder: nn.Module | None,
) -> nn.Module:
    if injected_text_encoder is not None:
        return injected_text_encoder
    if split.text_mode == "bert_token_triplets":
        if not bert_model_dir:
            raise CMCModelError(
                "canonical text_bert tokens require an explicitly prepared local bert_model_dir; no download is performed"
            )
        return LocalBertTokenEncoder.from_local(bert_model_dir)
    if split.text_mode == "cached_contextual_text":
        return ContextualTextEncoder(int(split.text_width))
    raise CMCModelError("unsupported canonical cache text input mode {!r}".format(split.text_mode))


def _model_options(config: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"embed_dim", "nhead", "transformer_layers", "out_dropout", "top_k", "temperature"}
    unknown = set(config) - allowed
    if unknown:
        raise CMCModelError("unknown CMC model setting(s): {}".format(sorted(unknown)))
    return dict(config)


def build_unimodal_pseudo_models(
    split: CMCFeatureSplit,
    *,
    model_config: Mapping[str, Any],
    bert_model_dir: str | None,
    finetune_text: bool = False,
    device: torch.device | str = "cpu",
    text_encoder_factory=None,
) -> tuple[dict[str, CMCUnimodalModel], dict[str, CMCUnimodalModel]]:
    """Build source Text/Audio/Vision encoders and parameter-only EMA copies."""
    options = _model_options(model_config)
    if split.text_mode == "cached_contextual_text" and finetune_text:
        raise CMCModelError(
            "cached contextual text cannot fine-tune the source BERT; use token-triplet cache input"
        )
    text_encoder = (
        text_encoder_factory(split)
        if text_encoder_factory is not None
        else _make_text_encoder(split, bert_model_dir=bert_model_dir, injected_text_encoder=None)
    )
    if not hasattr(text_encoder, "output_dim"):
        raise CMCModelError("text encoder must expose output_dim")
    text_dim = int(text_encoder.output_dim)
    model_args = {
        "embed_dim": int(options.get("embed_dim", 32)),
        "nhead": int(options.get("nhead", 2)),
        "transformer_layers": int(options.get("transformer_layers", 4)),
        "out_dropout": float(options.get("out_dropout", 0.3)),
    }
    models = {
        "text": CMCUnimodalModel(
            "text", text_dim, text_encoder=text_encoder,
            finetune_text=finetune_text, **model_args,
        ),
        "audio": CMCUnimodalModel(
            "audio", int(split.features["audio"].shape[2]), **model_args,
        ),
        "vision": CMCUnimodalModel(
            "vision", int(split.features["vision"].shape[2]), **model_args,
        ),
    }
    target_device = torch.device(device)
    models = {name: model.to(target_device) for name, model in models.items()}
    ema_models = {name: copy.deepcopy(model).eval() for name, model in models.items()}
    for model in ema_models.values():
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return models, ema_models


def _source_optimizer(
    model: nn.Module,
    *,
    lr: float,
    lr_bert: float,
    weight_decay_bert: float,
) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
    text_parameters = list(model.text_model.named_parameters()) if hasattr(model, "text_model") else []
    bert_decay = [parameter for name, parameter in text_parameters if not any(tag in name for tag in no_decay)]
    bert_no_decay = [parameter for name, parameter in text_parameters if any(tag in name for tag in no_decay)]
    text_ids = {id(parameter) for _, parameter in text_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in text_ids]
    groups = []
    if bert_decay:
        groups.append({"params": bert_decay, "weight_decay": weight_decay_bert, "lr": lr_bert})
    if bert_no_decay:
        groups.append({"params": bert_no_decay, "weight_decay": 0.0, "lr": lr_bert})
    if other_parameters:
        groups.append({"params": other_parameters, "weight_decay": 0.0, "lr": lr})
    if not groups:
        raise CMCModelError("source CMC optimizer has no parameters")
    return torch.optim.Adam(groups)


def build_pseudo_pretraining_bundle(
    split: CMCFeatureSplit,
    *,
    model_config: Mapping[str, Any],
    optimizer_config: Mapping[str, float],
    bert_model_dir: str | None,
    finetune_text: bool = False,
    device: torch.device | str = "cpu",
    text_encoder_factory=None,
) -> dict[str, Any]:
    models, ema_models = build_unimodal_pseudo_models(
        split,
        model_config=model_config,
        bert_model_dir=bert_model_dir,
        finetune_text=finetune_text,
        device=device,
        text_encoder_factory=text_encoder_factory,
    )
    lr = float(optimizer_config.get("lr", 1e-3))
    lr_bert = float(optimizer_config.get("lr_bert", 5e-5))
    weight_decay_bert = float(optimizer_config.get("weight_decay_bert", 1e-3))
    optimizers = {
        modality: _source_optimizer(
            model, lr=lr, lr_bert=lr_bert, weight_decay_bert=weight_decay_bert
        ) if modality == "text" else torch.optim.Adam(model.parameters(), lr=lr)
        for modality, model in models.items()
    }
    return {"models": models, "ema_models": ema_models, "optimizers": optimizers}


def build_fusion_model(
    split: CMCFeatureSplit,
    pseudo_models: Mapping[str, CMCUnimodalModel],
    *,
    model_config: Mapping[str, Any],
    bert_model_dir: str | None,
    finetune_text: bool = True,
    device: torch.device | str = "cpu",
    text_encoder_factory=None,
) -> tuple[CMCClassifier, dict[str, Any]]:
    options = _model_options(model_config)
    if split.text_mode == "cached_contextual_text" and finetune_text:
        raise CMCModelError(
            "source CMC Stage 2 BERT finetuning requires canonical text_bert token triplets"
        )
    text_encoder = (
        text_encoder_factory(split)
        if text_encoder_factory is not None
        else _make_text_encoder(split, bert_model_dir=bert_model_dir, injected_text_encoder=None)
    )
    model = CMCClassifier(
        text_dim=int(text_encoder.output_dim),
        audio_dim=int(split.features["audio"].shape[2]),
        vision_dim=int(split.features["vision"].shape[2]),
        text_encoder=text_encoder,
        embed_dim=int(options.get("embed_dim", 32)),
        num_classes=5,
        nhead=int(options.get("nhead", 2)),
        transformer_layers=int(options.get("transformer_layers", 4)),
        top_k=int(options.get("top_k", 3)),
        out_dropout=float(options.get("out_dropout", 0.3)),
        temperature=float(options.get("temperature", 0.07)),
        finetune_text=finetune_text,
        text_input_mode=split.text_mode,
    ).to(torch.device(device))
    transfer = transfer_unimodal_to_cmc(model, pseudo_models)
    return model, transfer


def build_fusion_optimizer(
    model: CMCClassifier, *, lr: float = 1e-3, lr_bert: float = 5e-5, weight_decay_bert: float = 1e-3
) -> torch.optim.Optimizer:
    """Original CMC grouped Adam with local BERT separated by decay policy."""
    return _source_optimizer(
        model,
        lr=float(lr),
        lr_bert=float(lr_bert),
        weight_decay_bert=float(weight_decay_bert),
    )
