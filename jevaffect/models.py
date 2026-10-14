"""Small model factory for cached text and ALMT baseline comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class ModelConfigError(ValueError):
    """Raised when a baseline config does not match the cached feature schema."""


class TextClassifier(nn.Module):
    """Two-layer MLP over cached contextual text features.

    ``pooling`` is explicit: ``cls`` uses token zero, while ``masked_mean``
    averages positions below ``features['lengths']['text']``. The first-layer
    width is inferred from the cache metadata. This is a text-only baseline,
    not the JevAffect fusion method.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelConfigError("{} must be a positive integer".format(name))
        if pooling not in ("cls", "masked_mean"):
            raise ModelConfigError("pooling must be 'cls' or 'masked_mean'")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ModelConfigError("dropout must be a number in [0, 1)")
        if not 0.0 <= float(dropout) < 1.0:
            raise ModelConfigError("dropout must be a number in [0, 1)")

        self.input_dim = input_dim
        self.pooling = pooling
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 5),
        )

    def _pool(self, features: Mapping[str, Any]) -> Tensor:
        text = features.get("text")
        if not isinstance(text, Tensor) or text.ndim != 3:
            raise ValueError("features['text'] must have shape [B, T, D]")
        batch_size, time_steps, feature_dim = text.shape
        if time_steps <= 0 or feature_dim != self.input_dim:
            raise ValueError("text sequence shape does not match the configured input width")

        length_map = features.get("lengths")
        if not isinstance(length_map, Mapping) or "text" not in length_map:
            raise ValueError("features['lengths']['text'] is required")
        lengths = torch.as_tensor(length_map["text"], device=text.device)
        if lengths.ndim != 1 or lengths.numel() != batch_size:
            raise ValueError("text lengths must have shape [B]")
        if lengths.dtype == torch.bool or lengths.dtype.is_floating_point:
            raise ValueError("text lengths must contain integer values")
        lengths = lengths.to(dtype=torch.long)
        if torch.any(lengths < 1) or torch.any(lengths > time_steps):
            raise ValueError("text lengths must be in [1, T]")

        if self.pooling == "cls":
            return text[:, 0, :]

        positions = torch.arange(time_steps, device=text.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        weights = valid.to(dtype=text.dtype).unsqueeze(-1)
        return (text * weights).sum(dim=1) / lengths.to(text.dtype).unsqueeze(1)

    def forward(self, features: Mapping[str, Any]) -> Tensor:
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping")
        return self.classifier(self._pool(features))


def build_model(
    name: str,
    feature_dims: Mapping[str, int],
    sequence_lengths: Mapping[str, int],
    params: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build a text or ALMT baseline from the cache schema.

    Cache dimensions and sequence lengths are injected through the existing
    ALMT adapter constructor fields; conflicting overrides fail instead of
    silently changing the model's expected input contract.
    """
    if not isinstance(feature_dims, Mapping) or not isinstance(sequence_lengths, Mapping):
        raise ModelConfigError("feature_dims and sequence_lengths must be mappings")
    if name not in ("text_mlp", "almt"):
        raise ModelConfigError("model name must be one of: text_mlp, almt")
    required = ("text",) if name == "text_mlp" else ("text", "audio", "vision")
    if any(key not in feature_dims for key in required):
        raise ModelConfigError("feature_dims lacks a modality required by {}".format(name))
    if any(key not in sequence_lengths for key in required):
        raise ModelConfigError("sequence_lengths lacks a modality required by {}".format(name))
    options = dict(params or {})

    if name == "text_mlp":
        unknown = set(options) - {"hidden_dim", "dropout", "pooling"}
        if unknown:
            raise ModelConfigError("unknown text_mlp params: {}".format(sorted(unknown)))
        return TextClassifier(input_dim=int(feature_dims["text"]), **options)

    if name == "almt":
        from jevaffect.almt import ALMTClassifier

        expected_dims = {key: int(feature_dims[key]) for key in required}
        expected_lengths = {key: int(sequence_lengths[key]) for key in required}
        if "feature_dims" in options and options["feature_dims"] != expected_dims:
            raise ModelConfigError("feature_dims conflicts with cache metadata")
        if "lengths" in options and options["lengths"] != expected_lengths:
            raise ModelConfigError("lengths conflicts with cache metadata")
        options["feature_dims"] = expected_dims
        options["lengths"] = expected_lengths
        return ALMTClassifier(**options)

    raise AssertionError("unreachable model dispatch")
