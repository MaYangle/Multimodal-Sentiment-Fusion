"""Controlled projected A/V moments; established operators, not a novelty claim."""
from __future__ import annotations

import torch
from torch import nn


CASES = ("static_product", "marginal_variance", "paired_cov", "unpaired_cov")


class TimedMomentRegressor(nn.Module):
    """Preserve global vectors and append one equally sized A/V statistic.

    Grid timestamps are decoder-reported anchors. Their physical cross-stream
    origin is unverified for the corpus. Common row permutations leave the
    paired statistic unchanged: it encodes pairing, not chronological direction.
    """

    def __init__(self, dims, case, rank=32, hidden=234, dropout=.1):
        super().__init__()
        if set(dims) != {"text", "audio", "vision"} or case not in CASES:
            raise ValueError("Expected T/A/V dimensions and a registered moment case")
        if rank < 1 or hidden < 1 or any(d <= 0 for d in dims.values()):
            raise ValueError("Dimensions must be positive")
        self.case = case
        self.audio_projection = nn.Linear(dims["audio"], rank, bias=False)
        self.vision_projection = nn.Linear(dims["vision"], rank, bias=False)
        self.head = nn.Sequential(nn.Linear(sum(dims.values()) + rank, hidden),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def moment(self, features):
        valid = features["pair_valid"]
        if valid.ndim != 1 or valid.dtype != torch.bool:
            raise ValueError("pair_valid must be a boolean batch vector")
        a_grid, v_grid = features["audio_grid"], features["vision_grid"]
        if a_grid.ndim != 3 or v_grid.ndim != 3 or a_grid.shape[:2] != v_grid.shape[:2] or a_grid.shape[1] != 4 or a_grid.shape[0] != len(valid):
            raise ValueError("Expected four jointly valid query rows per example")
        if self.case == "static_product":
            # Same pair eligibility as temporal controls, but use original
            # full retained-clip means rather than grid means.
            moment = self.audio_projection(features["audio_mean"]) * self.vision_projection(features["vision_mean"])
        else:
            a = self.audio_projection(a_grid)
            v = self.vision_projection(v_grid)
            if self.case == "unpaired_cov":
                v = torch.roll(v, shifts=1, dims=1)
            a = a - a.mean(dim=1, keepdim=True)
            v = v - v.mean(dim=1, keepdim=True)
            if self.case == "marginal_variance":
                moment = .5 * (a.square().mean(dim=1) + v.square().mean(dim=1))
            else:
                moment = (a * v).mean(dim=1)
        return torch.where(valid[:, None], moment, torch.zeros_like(moment))

    def forward(self, features):
        global_vectors = torch.cat([features["text"], features["audio_mean"], features["vision_mean"]], dim=-1)
        return self.head(torch.cat([global_vectors, self.moment(features)], dim=-1)).flatten()
