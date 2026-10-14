"""Pooled common-encoder MoNIG adaptation with a joint/pseudo-modality expert."""
import torch
from torch import nn
from torch.nn import functional as F
from jevaffect.reliability import ReliabilityRegressor, MODALITIES
from jevaffect.nig import fuse_nig

CASES = ("three_uniform", "three_dynamic", "joint_uniform", "joint_dynamic", "joint_off_uniform", "joint_off_dynamic")


class ContextNIGRegressor(nn.Module):
    def __init__(self, dims, hidden=192, joint_hidden=64, dropout=.1, fusion="dynamic", enable_joint=True):
        super().__init__()
        self.base = ReliabilityRegressor(dims, hidden=hidden, dropout=dropout, fusion=fusion)
        for parameter in self.base.unimodal_heads.parameters():
            parameter.requires_grad_(False)
        self.joint = nn.Sequential(nn.Linear(sum(dims.values()), joint_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(joint_hidden, 4))
        self.fusion, self.enable_joint = fusion, enable_joint

    def forward(self, features):
        # Explicitly remove unavailable values before both independent and joint
        # encoding; a missing input cannot contaminate the joint expert.
        availability = features.get("availability")
        if availability is None:
            availability = torch.ones((features["text"].shape[0], 3), dtype=torch.bool, device=features["text"].device)
        if availability.dtype != torch.bool or availability.shape != (features["text"].shape[0], 3):
            raise ValueError("Input availability must be boolean [B,3]")
        masked = {m: torch.where(availability[:, i, None], features[m], torch.zeros_like(features[m])) for i, m in enumerate(MODALITIES)}
        masked["availability"] = availability
        base = self.base(masked)
        raw = self.joint(torch.cat([masked[m] for m in MODALITIES], -1))
        mu, v, alpha, beta = raw.unbind(-1)
        joint = torch.stack((mu, F.softplus(v) + 1e-4, F.softplus(alpha) + 1 + 1e-4, F.softplus(beta) + 1e-4), -1)
        parameters = torch.cat((base["branches"], joint[:, None]), 1)
        joint_available = availability.any(1) if self.enable_joint else torch.zeros_like(availability[:, 0])
        expert_availability = torch.cat((availability, joint_available[:, None]), 1)
        fused, weights = fuse_nig(parameters, expert_availability, self.fusion)
        return {"prediction": fused[:, 0], "fused": fused, "branches": parameters, "weights": weights,
                "availability": expert_availability, "input_availability": availability}


def build_context_model(case, dims, config):
    if case not in CASES:
        raise ValueError("Unregistered joint-context case")
    fusion = "dynamic" if case.endswith("dynamic") else "uniform"
    if case.startswith("three_"):
        model = ReliabilityRegressor(dims, hidden=config["three_hidden"], dropout=config["dropout"], fusion=fusion)
        for parameter in model.unimodal_heads.parameters():
            parameter.requires_grad_(False)
        return model
    return ContextNIGRegressor(dims, hidden=config["joint_branch_hidden"], joint_hidden=config["joint_hidden"],
                               dropout=config["dropout"], fusion=fusion, enable_joint=not case.startswith("joint_off_"))
