"""ALMT scalar baseline with valid-rank positions and masked projection keys.

Reuses the attributed upstream weights without editing the frozen adapter.
This is a common padding correction, not an original research contribution.
"""
import torch
from torch import nn
from einops import rearrange, repeat
from jevaffect.almt import ALMTClassifier


class MaskedALMTRegressor(ALMTClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classifier = nn.Linear(self.classifier.in_features, 1)

    def _mask(self, features, modality):
        x = features[modality]
        if "valid_masks" in features:
            mask = features["valid_masks"][modality]
            if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
                raise ValueError("Explicit valid masks must be boolean [batch,steps]")
        else:
            lengths = features["lengths"][modality]
            if lengths.shape != (len(x),) or lengths.dtype not in (torch.int32, torch.int64) or torch.any(lengths < 0) or torch.any(lengths > x.shape[1]):
                raise ValueError("Invalid declared feature lengths")
            mask = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        if mask.device != x.device:
            raise ValueError("Feature and mask devices differ")
        return mask

    @staticmethod
    def _attention(layer, x, valid):
        norm = layer
        core = norm.fn
        q = core.to_q(norm.norm_q(x))
        k = core.to_k(norm.norm_k(x))
        v = core.to_v(norm.norm_v(x))
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=core.heads) for t in (q, k, v))
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * core.scale
        dots = dots.masked_fill(~valid[:, None, None, :], -torch.inf)
        out = torch.einsum("b h i j, b h j d -> b h i d", core.attend(dots), v)
        return core.to_out(rearrange(out, "b h n d -> b n (h d)"))

    def _project(self, projection, x, mask):
        linear, transformer = projection
        # Invalid values are excluded before linear projection, including NaNs.
        x = linear(x.masked_fill(~mask[:, :, None], 0.))
        b = x.shape[0]
        extra = repeat(transformer.extra_token, "1 n d -> b n d", b=b)
        ranks = (mask.long().cumsum(1) - 1).clamp_min(0) + self.token_len
        positions = transformer.pos_embedding[0][ranks]
        x = torch.cat((extra, x), 1)
        x = x + torch.cat((transformer.pos_embedding[:, :self.token_len].expand(b, -1, -1), positions), 1)
        x = transformer.dropout(x)
        valid = torch.cat((torch.ones(b, self.token_len, device=x.device, dtype=torch.bool), mask), 1)
        for attention, ff in transformer.encoder.layers:
            x = self._attention(attention, x, valid) + x
            x = ff(x) + x
        return x[:, :self.token_len]

    def forward(self, features):
        self._validate_inputs(features)
        tokens = {m: self._project(proj, features[m], self._mask(features, m))
                  for m, proj in (("text", self.proj_l), ("audio", self.proj_a), ("vision", self.proj_v))}
        hyper = repeat(self.h_hyper, "1 n d -> b n d", b=len(features["text"]))
        language = self.l_encoder(tokens["text"])
        hyper = self.h_hyper_layer(language, tokens["audio"], tokens["vision"], hyper)
        return self.classifier(self.fusion_layer(hyper, language[-1])[:, 0])
