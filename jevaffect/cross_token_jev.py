"""Cross-token multimodal ordinal fusion with explicit missing-token masks.

MacBERT text, WavLM audio and DINOv2 vision are projected to a shared width.
Text uses token-order positions; audio/video use measured clip timestamps.
Cross attention learns associations rather than assuming word/frame alignment.
The five-class branch distributions expose ordinal disagreement to a learned
modality gate. This is an independent Jev-inspired experiment, not Jev weights.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CrossTokenInputError(ValueError):
    pass


def _positions(values: Tensor, width: int) -> Tensor:
    if width % 2:
        raise CrossTokenInputError("shared width must be even")
    frequencies = torch.exp(torch.arange(0, width, 2, device=values.device, dtype=values.dtype)
                            * (-math.log(10000.0) / width))
    phase = values.unsqueeze(-1) * frequencies * (2 * math.pi)
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


def _masked_attention_pool(sequence: Tensor, mask: Tensor, scorer: nn.Module) -> Tensor:
    score = scorer(sequence).squeeze(-1)
    safe_mask = mask.clone()
    safe_mask[:, 0] |= ~mask.any(dim=1)
    score = score.masked_fill(~safe_mask, torch.finfo(score.dtype).min)
    weights = score.softmax(dim=1) * mask.to(score.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bt,btd->bd", weights, sequence)


class CrossTokenJev(nn.Module):
    """Token↔AV cross-attention plus structured five-class modality gating."""

    def __init__(self, *, text_dim: int = 768, audio_dim: int = 768,
                 vision_dim: int = 384, width: int = 128, heads: int = 4,
                 dropout: float = 0.15, cross_attention: bool = True,
                 use_av_time: bool = True, gate_kind: str = "learned") -> None:
        super().__init__()
        if width < 32 or width % heads or gate_kind not in ("learned", "uniform"):
            raise CrossTokenInputError("invalid shared width, head count or gate kind")
        self.width = width
        self.input_dims = {"text": text_dim, "audio": audio_dim, "vision": vision_dim}
        self.cross_attention = cross_attention
        self.use_av_time = use_av_time
        self.gate_kind = gate_kind
        self.projection = nn.ModuleDict({
            "text": nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width)),
            "audio": nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, width)),
            "vision": nn.Sequential(nn.LayerNorm(vision_dim), nn.Linear(vision_dim, width)),
        })
        self.modality_embedding = nn.Parameter(torch.zeros(3, width))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.null_text = nn.Parameter(torch.zeros(width))
        self.null_av = nn.Parameter(torch.zeros(width))
        self.text_from_av = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.audio_from_text = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.vision_from_text = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.pre_norm = nn.ModuleDict({key: nn.LayerNorm(width) for key in ("text", "audio", "vision")})
        self.post_norm = nn.ModuleDict({key: nn.LayerNorm(width) for key in ("text", "audio", "vision")})
        self.feedforward = nn.ModuleDict({key: nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(),
                                                              nn.Dropout(dropout), nn.Linear(2 * width, width))
                                          for key in ("text", "audio", "vision")})
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.ModuleDict({key: nn.Linear(width, 1) for key in ("text", "audio", "vision")})
        self.branch = nn.ModuleDict({key: nn.Linear(width, 5) for key in ("text", "audio", "vision")})
        # Three pooled vectors + three entropy, three ordinal conflict and
        # three token-coverage statistics. Missing streams receive zero weight.
        self.gate = nn.Sequential(nn.LayerNorm(3 * width + 9),
                                  nn.Linear(3 * width + 9, width), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(width, 3))
        self.fusion_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                         nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 5))
        self.branch_scale = nn.Parameter(torch.tensor(0.3))
        self.prior_logits = nn.Parameter(torch.zeros(5))

    def _validate(self, batch: Mapping[str, Tensor]) -> None:
        for key, dim in self.input_dims.items():
            value = batch.get(key)
            mask = batch.get(key + "_mask")
            if (not isinstance(value, Tensor) or value.ndim != 3 or value.shape[-1] != dim or
                    not value.is_floating_point() or not isinstance(mask, Tensor) or
                    mask.dtype != torch.bool or mask.shape != value.shape[:2] or value.shape[1] < 1):
                raise CrossTokenInputError(key + " feature/mask contract failed")
            if not torch.isfinite(value[mask]).all():
                raise CrossTokenInputError(key + " valid features are nonfinite")
        for key in ("audio", "vision"):
            times = batch.get(key + "_times")
            if (not isinstance(times, Tensor) or times.shape != batch[key].shape[:2] or
                    not torch.isfinite(times).all() or torch.any(times[batch[key + "_mask"]] < 0)):
                raise CrossTokenInputError(key + " timestamps invalid")
        if len({batch[key].shape[0] for key in ("text", "audio", "vision")}) != 1:
            raise CrossTokenInputError("modalities have different batch sizes")

    def _initial_tokens(self, batch: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        masks = {key: batch[key + "_mask"] for key in ("text", "audio", "vision")}
        at = batch["audio_times"].masked_fill(~masks["audio"], 0)
        vt = batch["vision_times"].masked_fill(~masks["vision"], 0)
        duration = torch.maximum(at.max(1).values, vt.max(1).values + 0.25 * masks["vision"].any(1)).clamp_min(0.25)
        length = masks["text"].sum(1).clamp_min(1).to(batch["text"].dtype)
        text_position = torch.arange(batch["text"].shape[1], device=length.device,
                                     dtype=length.dtype)[None, :] / length[:, None]
        positions = {"text": text_position,
                     "audio": at / duration[:, None] if self.use_av_time else torch.zeros_like(at),
                     "vision": vt / duration[:, None] if self.use_av_time else torch.zeros_like(vt)}
        tokens = {}
        for index, key in enumerate(("text", "audio", "vision")):
            features = batch[key].masked_fill(~masks[key].unsqueeze(-1), 0)
            projected = self.projection[key](features)
            embedded = projected + self.modality_embedding[index] + _positions(positions[key].to(projected.dtype), self.width)
            tokens[key] = embedded * masks[key].unsqueeze(-1)
        return tokens, masks

    def forward(self, batch: Mapping[str, Tensor], *, return_attention: bool = False) -> dict[str, Tensor]:
        self._validate(batch)
        tokens, masks = self._initial_tokens(batch)
        t, a, v = tokens["text"], tokens["audio"], tokens["vision"]
        original_t = masks["text"]
        safe_t = original_t.clone()
        missing_t = ~safe_t.any(1)
        if missing_t.any():
            t = t.clone()
            t[missing_t, 0] = self.null_text
            safe_t[missing_t, 0] = True
        av = torch.cat((a, v), dim=1)
        safe_av = torch.cat((masks["audio"], masks["vision"]), dim=1)
        missing_av = ~safe_av.any(1)
        if missing_av.any():
            av = av.clone()
            av[missing_av, 0] = self.null_av
            safe_av[missing_av, 0] = True
        attention_map = None
        if self.cross_attention:
            new_t, attention_map = self.text_from_av(
                self.pre_norm["text"](t), av, av,
                key_padding_mask=~safe_av, need_weights=return_attention,
                average_attn_weights=True)
            new_a, _ = self.audio_from_text(
                self.pre_norm["audio"](a), t, t,
                key_padding_mask=~safe_t, need_weights=False)
            new_v, _ = self.vision_from_text(
                self.pre_norm["vision"](v), t, t,
                key_padding_mask=~safe_t, need_weights=False)
            t = self.post_norm["text"](t + self.dropout(new_t))
            a = self.post_norm["audio"](a + self.dropout(new_a))
            v = self.post_norm["vision"](v + self.dropout(new_v))
        for key, sequence in (("text", t), ("audio", a), ("vision", v)):
            sequence = sequence + self.dropout(self.feedforward[key](sequence))
            tokens[key] = sequence * masks[key].unsqueeze(-1)
        pooled = {key: _masked_attention_pool(tokens[key], masks[key], self.pool[key])
                  for key in ("text", "audio", "vision")}
        valid = torch.stack([masks[key].any(1) for key in ("text", "audio", "vision")], dim=-1)
        branches = {key: self.branch[key](pooled[key]) for key in ("text", "audio", "vision")}
        probabilities = torch.stack([branches[key].softmax(-1) for key in ("text", "audio", "vision")], dim=1)
        cdf = probabilities.cumsum(-1)[..., :4]
        conflict = torch.stack([(cdf[:, index:index + 1] - cdf).abs().mean(dim=(1, 2)) * 1.5
                                for index in range(3)], dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum(-1) / math.log(5)
        coverage = torch.stack([masks[key].float().mean(1) for key in ("text", "audio", "vision")], dim=-1)
        descriptor = torch.cat((pooled["text"], pooled["audio"], pooled["vision"],
                                entropy, conflict, coverage), dim=-1)
        if self.gate_kind == "learned":
            gate_logits = self.gate(descriptor).masked_fill(~valid, -1e4)
            weight = gate_logits.softmax(-1) * valid
        else:
            weight = valid.to(pooled["text"].dtype)
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1)
        stack = torch.stack([pooled[key] for key in ("text", "audio", "vision")], dim=1)
        fused = (stack * weight.unsqueeze(-1)).sum(1)
        branch_logits = torch.stack([branches[key] for key in ("text", "audio", "vision")], dim=1)
        logits = self.fusion_head(fused) + self.branch_scale * (branch_logits * weight.unsqueeze(-1)).sum(1)
        logits = torch.where(valid.any(1, keepdim=True), logits, self.prior_logits)
        result = {"logits": logits, "text_logits": branches["text"],
                  "audio_logits": branches["audio"], "vision_logits": branches["vision"],
                  "branch_probabilities": probabilities, "modality_weight": weight,
                  "ordinal_conflict": conflict, "valid_modalities": valid}
        if return_attention and attention_map is not None:
            result["text_to_av_attention"] = attention_map * original_t.unsqueeze(-1)
        return result
