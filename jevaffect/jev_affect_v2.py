"""Jev-inspired intervention-calibrated temporal cross-token sentiment model.

The independent base network predicts five-class text/audio/vision evidence on
physical AV windows. A bounded text-to-AV cross-token residual receives a
window-utility attention bias. Utility supervision, when enabled by the
trainer, comes exclusively from source-group out-of-fold training examples.
No TypeSafe Jev architecture or weights are used.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from jevaffect.cross_token_jev import _positions
from jevaffect.jev_temporal import JevTemporalOrdinal, _masked_mean


class JevAffectV2(nn.Module):
    """Bounded reliability-biased attention plus structured ordinal branches."""

    def __init__(self, *, hidden: int = 96, windows: int = 8,
                 cross_attention: bool = True, temporal: bool = True,
                 gate_kind: str = "learned", dropout: float = 0.15) -> None:
        super().__init__()
        self.base = JevTemporalOrdinal(hidden=hidden, windows=windows,
                                       gate_kind=gate_kind, temporal=temporal,
                                       dropout=dropout)
        self.windows = windows
        self.hidden = hidden
        self.cross_attention = cross_attention
        self.temporal = temporal
        self.modality_embedding = nn.Parameter(torch.zeros(3, hidden))
        nn.init.normal_(self.modality_embedding, std=0.02)
        self.null_av = nn.Parameter(torch.zeros(hidden))
        self.cross = nn.MultiheadAttention(hidden, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.text_norm = nn.LayerNorm(hidden)
        self.delta_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 5))
        self.cross_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, batch: Mapping[str, Tensor], *,
                utility_targets: Tensor | None = None, utility_weight: float = 0.0,
                return_attention: bool = False) -> dict[str, Tensor]:
        output = self.base(batch, utility_targets=utility_targets,
                           utility_weight=utility_weight)
        if not self.cross_attention:
            output["cross_delta_logits"] = torch.zeros_like(output["logits"])
            return output
        masks = {kind: batch[kind + "_mask"] for kind in ("text", "audio", "vision")}
        projected = {}
        for index, kind in enumerate(("text", "audio", "vision")):
            values = batch[kind].masked_fill(~masks[kind].unsqueeze(-1), 0)
            tokens = getattr(self.base, kind + "_projection")(values)
            projected[kind] = tokens * masks[kind].unsqueeze(-1)
        duration = output["time_windows_seconds"][:, -1, 1].clamp_min(0.25)
        text_length = masks["text"].sum(1).clamp_min(1).to(projected["text"].dtype)
        text_index = torch.arange(projected["text"].shape[1], device=text_length.device,
                                  dtype=text_length.dtype)[None, :] / text_length[:, None]
        text_tokens = projected["text"] + self.modality_embedding[0] + _positions(text_index, self.hidden)
        text_tokens = text_tokens * masks["text"].unsqueeze(-1)
        av_tokens = []
        av_masks = []
        av_biases = []
        for index, kind in enumerate(("audio", "vision"), start=1):
            times = batch[kind + "_times"]
            normalized = times / duration[:, None] if self.temporal else torch.zeros_like(times)
            tokens = projected[kind] + self.modality_embedding[index] + _positions(normalized.to(projected[kind].dtype), self.hidden)
            tokens = tokens * masks[kind].unsqueeze(-1)
            av_tokens.append(tokens)
            av_masks.append(masks[kind])
            position = (times / duration[:, None] * self.windows).floor().long().clamp(0, self.windows - 1)
            gate = output["gate"][:, :, index - 1].gather(1, position)
            av_biases.append(gate.clamp_min(0.02).log())
        audiovisual = torch.cat(av_tokens, dim=1)
        valid_av = torch.cat(av_masks, dim=1)
        bias = torch.cat(av_biases, dim=1)
        missing = ~valid_av.any(1)
        if missing.any():
            audiovisual = audiovisual.clone()
            valid_av = valid_av.clone()
            bias = bias.clone()
            audiovisual[missing, 0] = self.null_av
            valid_av[missing, 0] = True
            bias[missing, 0] = 0
        bias = bias.masked_fill(~valid_av, -1e4)
        query_count = text_tokens.shape[1]
        heads = self.cross.num_heads
        attention_bias = bias[:, None, None, :].expand(-1, heads, query_count, -1).reshape(
            -1, query_count, audiovisual.shape[1])
        attended, weights = self.cross(self.text_norm(text_tokens), audiovisual, audiovisual,
                                       attn_mask=attention_bias,
                                       need_weights=return_attention, average_attn_weights=True)
        pooled = _masked_mean(self.cross_norm(attended), masks["text"])
        correction = self.cross_scale.tanh() * self.delta_head(pooled)
        correction = torch.where(masks["text"].any(1, keepdim=True) &
                                 (masks["audio"].any(1, keepdim=True) | masks["vision"].any(1, keepdim=True)),
                                 correction, torch.zeros_like(correction))
        output["cross_delta_logits"] = correction
        output["logits"] = output["logits"] + correction
        if return_attention and weights is not None:
            output["text_to_av_attention"] = weights * masks["text"].unsqueeze(-1)
        return output
