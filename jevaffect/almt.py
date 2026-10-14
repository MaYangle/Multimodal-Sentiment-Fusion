"""Five-class ALMT adapter for cached contextual text feature sequences.

This is an ALMT structural baseline for a shared cached-feature pipeline. It
keeps the author's projection Transformers, language-guided encoder,
hyper-modality learning layers, and cross-modal Transformer. It intentionally
does not construct or fine-tune the author's internal BERT; ``features['text']``
must already contain contextual per-token vectors such as cached BERT outputs.

Inputs use fixed-size sequences:

* text: ``[B, text_length, text_dim]`` (MOSEI defaults: ``[B, 50, 768]``)
* audio: ``[B, audio_length, audio_dim]`` (defaults: ``[B, 500, 74]``)
* vision: ``[B, vision_length, vision_dim]`` (defaults: ``[B, 500, 35]``)

``lengths`` in the constructor are the configured maximum sequence lengths
that size the learned positional embeddings. The upstream ALMT attention
layers do not accept padding masks. This adapter preserves that behavior:
every supplied position, including padding, participates in the projection
Transformers and subsequent attention. Do not infer that batch padding masks
are honored; callers must supply the same fixed-length/padding convention for
all compared models and report it in experiment metadata.
"""

from collections.abc import Mapping
from typing import Dict, Optional

import torch
from torch import Tensor, nn
from einops import repeat

from vendor.almt.almt_layer import (
    CrossTransformer,
    HhyperLearningEncoder,
    Transformer,
)


_DEFAULT_FEATURE_DIMS = {"text": 768, "audio": 74, "vision": 35}
_DEFAULT_LENGTHS = {"text": 50, "audio": 500, "vision": 500}
_MODALITIES = ("text", "audio", "vision")


def _complete_positive_ints(
    values: Optional[Mapping[str, int]], defaults: Dict[str, int], name: str
) -> Dict[str, int]:
    result = dict(defaults)
    if values is not None:
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
        result.update(values)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}[{key!r}] must be a positive integer")
    return result


class ALMTClassifier(nn.Module):
    """ALMT feature-sequence model with a five-class output head.

    Args:
        feature_dims: Optional per-token widths, keyed by ``text``, ``audio``,
            and ``vision``. Defaults match the upstream MOSEI YAML.
        lengths: Optional fixed sequence lengths with the same keys. These
            determine positional-embedding sizes and are checked on each
            forward pass.
        dim: Shared projection/hyper token width (upstream default: 128).
        token_len: Number of adaptive hyper-modality tokens (default: 8).
        num_classes: Output classes. Must remain five for this adapter.

    Other layer defaults mirror ``configs/mosei.yaml`` and the upstream
    ``ALMT`` constructor. No BERT module or additional mask operation is added.
    """

    def __init__(
        self,
        feature_dims: Optional[Mapping[str, int]] = None,
        lengths: Optional[Mapping[str, int]] = None,
        dim: int = 128,
        token_len: int = 8,
        num_classes: int = 5,
        proj_depth: int = 1,
        proj_heads: int = 8,
        proj_mlp_dim: int = 128,
        l_enc_heads: int = 8,
        l_enc_mlp_dim: int = 128,
        ahl_depth: int = 3,
        ahl_heads: int = 8,
        ahl_dim_head: int = 16,
        ahl_dropout: float = 0.0,
        fusion_depth: int = 4,
        fusion_heads: int = 8,
        fusion_mlp_dim: int = 128,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes != 5:
            raise ValueError("ALMTClassifier is a fixed five-class head")
        if dim <= 0 or token_len <= 0:
            raise ValueError("dim and token_len must be positive")
        if ahl_depth < 2:
            raise ValueError("ahl_depth must be at least 2 for the language encoder")

        self.feature_dims = _complete_positive_ints(
            feature_dims, _DEFAULT_FEATURE_DIMS, "feature_dims"
        )
        self.lengths = _complete_positive_ints(lengths, _DEFAULT_LENGTHS, "lengths")
        self.dim = dim
        self.token_len = token_len
        self.h_hyper = nn.Parameter(torch.ones(1, token_len, dim))

        def make_projection(modality: str) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.feature_dims[modality], dim),
                Transformer(
                    num_frames=self.lengths[modality],
                    save_hidden=False,
                    token_len=token_len,
                    dim=dim,
                    depth=proj_depth,
                    heads=proj_heads,
                    mlp_dim=proj_mlp_dim,
                    dropout=dropout,
                    emb_dropout=emb_dropout,
                ),
            )

        self.proj_l = make_projection("text")
        self.proj_a = make_projection("audio")
        self.proj_v = make_projection("vision")

        self.l_encoder = Transformer(
            num_frames=token_len,
            save_hidden=True,
            token_len=None,
            dim=dim,
            depth=ahl_depth - 1,
            heads=l_enc_heads,
            mlp_dim=l_enc_mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )
        self.h_hyper_layer = HhyperLearningEncoder(
            dim=dim,
            depth=ahl_depth,
            heads=ahl_heads,
            dim_head=ahl_dim_head,
            dropout=ahl_dropout,
        )
        self.fusion_layer = CrossTransformer(
            source_num_frames=token_len,
            tgt_num_frames=token_len,
            dim=dim,
            depth=fusion_depth,
            heads=fusion_heads,
            mlp_dim=fusion_mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
        )

        # The upstream model uses Linear(token_dim, 1) for regression.
        # This is the sole task-head change for the shared five-class protocol.
        self.classifier = nn.Linear(dim, num_classes)

    def _validate_inputs(self, features: Mapping[str, Tensor]) -> None:
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping with text/audio/vision tensors")
        batch_size = None
        for modality in _MODALITIES:
            value = features.get(modality)
            if not isinstance(value, Tensor):
                raise TypeError(f"features[{modality!r}] must be a torch.Tensor")
            expected = (self.lengths[modality], self.feature_dims[modality])
            if value.ndim != 3 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"features[{modality!r}] must have shape [B,{expected[0]},{expected[1]}], "
                    f"got {tuple(value.shape)}"
                )
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("text, audio, and vision batch sizes must match")

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        """Return five logits per example; padding positions are not masked."""
        self._validate_inputs(features)
        x_text = features["text"]
        x_audio = features["audio"]
        x_visual = features["vision"]
        batch_size = x_text.size(0)

        # Repeat, matching upstream ALMT.py. HhyperLearningLayer updates this
        # batch tensor in place, so a view/expand of the learned parameter is
        # not equivalent here.
        h_hyper = repeat(self.h_hyper, "1 n d -> b n d", b=batch_size)

        # Text inputs are cached contextual features; the original internal
        # BertTextEncoder and its fine-tuning path are intentionally omitted.
        h_v = self.proj_v(x_visual)[:, : self.token_len]
        h_a = self.proj_a(x_audio)[:, : self.token_len]
        h_l = self.proj_l(x_text)[:, : self.token_len]

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        return self.classifier(feat)
