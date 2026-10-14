"""Adapter for the vendored, vector-level legacy MSA fusion model.

The original model consumes one vector per modality. This adapter accepts
``text``, ``audio``, and ``vision`` as either ``[B, D]`` vectors or
``[B, T, D]`` sequences. Text sequences are reduced to their first token to
match the legacy BERT-CLS representation. Audio and vision sequences use a
masked mean to match the legacy clip-level temporal averaging. The result is
then passed unchanged to the original TransformerFusionModel, whose encoders
expand each vector to one token. This deliberately preserves the legacy
single-token self/cross-attention behavior; it is not a temporal model.

Optional validity metadata can be supplied in either form:

``lengths={"text": [B], "audio": [B], "vision": [B]}``

``masks={"text": [B,T], "audio": [B,T], "vision": [B,T]}``

Masks use ``True`` for valid (non-padding) positions. Both ``lengths`` and
``masks`` may be supplied; a position must satisfy both to contribute.
For convenience, ``length``/``mask`` aliases and per-modality keys such as
``audio_lengths``/``audio_mask`` are accepted. Metadata is not passed into the
legacy network.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from vendor.legacy_msa.src.models.fusion import TransformerFusionModel


_MODALITIES = ("text", "audio", "vision")


class LegacyMSAFusion(TransformerFusionModel):
    """Run the vendored five-class fusion model behind a sequence adapter.

    Args:
        text_dim: Per-token text feature width (legacy default: 768).
        audio_dim: Per-frame audio feature width (legacy default: 40).
        visual_dim: Per-frame visual feature width (legacy default: 35).
        hidden_dim: Legacy fusion hidden width.
        num_heads: Legacy attention-head count.
        num_layers: Legacy encoder-layer count.
        dropout_rate: Legacy dropout rate.

    The constructor fixes the output width at five classes. It adds no
    parameters and preserves the base model's state-dict keys.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 40,
        visual_dim: int = 35,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            num_classes=5,
        )

    @staticmethod
    def _metadata(features: Mapping, plural: str, singular: str, modality: str):
        """Read a modality entry from mapped or flat batch metadata."""
        source = features.get(plural, features.get(singular))
        if isinstance(source, Mapping):
            return source.get(modality)
        if source is not None:
            raise TypeError(f"'{plural}'/'{singular}' must map modality names to values")
        return features.get(f"{modality}_{plural}", features.get(f"{modality}_{singular}"))

    @classmethod
    def _reduce_one(cls, features: Mapping, modality: str) -> Tensor:
        value = features.get(modality)
        if not isinstance(value, Tensor):
            raise TypeError(f"features['{modality}'] must be a torch.Tensor")
        if value.ndim == 2:
            return value
        if value.ndim != 3:
            raise ValueError(
                f"features['{modality}'] must have shape [B,D] or [B,T,D]; "
                f"got {tuple(value.shape)}"
            )

        batch_size, time_steps, _ = value.shape
        if time_steps == 0:
            raise ValueError(f"features['{modality}'] has no time steps")

        lengths = cls._metadata(features, "lengths", "length", modality)
        mask = cls._metadata(features, "masks", "mask", modality)

        valid = torch.ones((batch_size, time_steps), dtype=torch.bool, device=value.device)
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=value.device).reshape(-1)
            if lengths.numel() != batch_size:
                raise ValueError(
                    f"lengths for '{modality}' must have one value per batch item"
                )
            if torch.any(lengths < 0) or torch.any(lengths > time_steps):
                raise ValueError(f"lengths for '{modality}' must be in [0, T]")
            positions = torch.arange(time_steps, device=value.device).unsqueeze(0)
            valid &= positions < lengths.to(dtype=torch.long).unsqueeze(1)

        if mask is not None:
            mask = torch.as_tensor(mask, device=value.device, dtype=torch.bool)
            if tuple(mask.shape) != (batch_size, time_steps):
                raise ValueError(
                    f"mask for '{modality}' must have shape {(batch_size, time_steps)}"
                )
            valid &= mask

        if modality == "text":
            # Historical preprocessing fed BERT's [CLS] vector, not a pooled
            # sequence. The sequence adapter therefore keeps position zero.
            if torch.any(~valid[:, 0]):
                raise ValueError(
                    "the first text position must be valid (legacy CLS convention)"
                )
            return value[:, 0, :]

        # Historical video/audio preprocessing averaged frames across the clip.
        weights = valid.to(dtype=value.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError(f"'{modality}' has a batch item with no valid time steps")
        return (value * weights).sum(dim=1) / counts

    def forward(self, features: Mapping) -> Tensor:
        """Reduce the input sequences as the old preprocessing did, then infer.

        Returns the original model's logits with shape ``[B, 5]``.
        """
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping containing text/audio/vision")
        vectors = {name: self._reduce_one(features, name) for name in _MODALITIES}
        return super().forward(vectors)
