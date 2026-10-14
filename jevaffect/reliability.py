"""Independent unimodal experts with Normal--Inverse-Gamma fusion."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from jevaffect.nig import fuse_nig


MODALITIES = ("text", "audio", "vision")
_PARAMETER_FLOOR = 1e-4


class ReliabilityRegressor(nn.Module):
    """Predict shared-target NIG experts and fuse them at prediction time.

    Each branch has its own MLP and two output heads: a four-parameter NIG for
    the common multimodal target and a scalar auxiliary prediction. Branch MLPs
    have the same configurable hidden width but do not share parameters.
    """

    def __init__(
        self,
        dims: Mapping[str, int],
        hidden: int = 128,
        dropout: float = 0.1,
        fusion: str = "dynamic",
    ) -> None:
        super().__init__()
        if tuple(dims.keys()) != MODALITIES:
            # Keep tensor branch order unambiguous regardless of dict insertion order.
            if set(dims.keys()) != set(MODALITIES):
                raise ValueError(f"dims must contain exactly {MODALITIES}")
            dims = {name: dims[name] for name in MODALITIES}
        if any(not isinstance(dims[name], int) or dims[name] < 1 for name in MODALITIES):
            raise ValueError("each modality dimension must be a positive integer")
        if not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if fusion not in ("dynamic", "uniform"):
            raise ValueError("fusion must be 'dynamic' or 'uniform'")

        self.modalities = MODALITIES
        self.dims = dict(dims)
        self.hidden = hidden
        self.dropout = float(dropout)
        self.fusion = fusion
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.dims[name], hidden),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )
            for name in MODALITIES
        })
        self.distribution_heads = nn.ModuleDict({name: nn.Linear(hidden, 4) for name in MODALITIES})
        self.unimodal_heads = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in MODALITIES})

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not isinstance(features, Mapping):
            raise TypeError("features must be a mapping containing text, audio, and vision tensors")
        missing = [name for name in MODALITIES if name not in features]
        if missing:
            raise ValueError("missing modality feature tensors: " + ", ".join(missing))

        raw_parameters: list[torch.Tensor] = []
        unimodal_predictions: list[torch.Tensor] = []
        batch_size: int | None = None
        for name in MODALITIES:
            value = features[name]
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                raise ValueError(f"{name} features must have shape [B, D]")
            if value.shape[1] != self.dims[name]:
                raise ValueError(f"{name} feature width {value.shape[1]} != configured width {self.dims[name]}")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("all modality feature batches must have the same size")
            hidden = self.encoders[name](value)
            raw_parameters.append(self.distribution_heads[name](hidden))
            unimodal_predictions.append(self.unimodal_heads[name](hidden).squeeze(-1))

        if batch_size is None or batch_size < 1:
            raise ValueError("features must contain a nonempty batch")
        branches = torch.stack(raw_parameters, dim=1)
        mu, gamma_raw, alpha_raw, beta_raw = branches.unbind(dim=-1)
        branches = torch.stack((
            mu,
            F.softplus(gamma_raw) + _PARAMETER_FLOOR,
            F.softplus(alpha_raw) + 1.0 + _PARAMETER_FLOOR,
            F.softplus(beta_raw) + _PARAMETER_FLOOR,
        ), dim=-1)
        unimodal_prediction = torch.stack(unimodal_predictions, dim=1)

        availability = features.get("availability")
        if availability is None:
            # With pooled inputs and no explicit mask, all supplied modalities
            # are treated as available. All-zero vectors are never inferred as missing.
            availability = torch.ones((batch_size, len(MODALITIES)), dtype=torch.bool, device=branches.device)
        else:
            availability = torch.as_tensor(availability, device=branches.device)
            if availability.shape != (batch_size, len(MODALITIES)):
                raise ValueError(f"availability must have shape [{batch_size}, {len(MODALITIES)}]")
            if availability.dtype != torch.bool:
                raise TypeError("availability must be a boolean tensor")

        fused, weights = fuse_nig(branches, availability, mode=self.fusion)
        return {
            "prediction": fused[:, 0],
            "fused": fused,
            "branches": branches,
            "unimodal_prediction": unimodal_prediction,
            "weights": weights,
            "availability": availability,
        }


__all__ = ["MODALITIES", "ReliabilityRegressor"]
