"""Annotation-reference regression ablation; no causal/reliability claims."""
import torch
from torch import nn

REFERENCE_CASES = ("flat0", "flat1", "sum0", "sum1")


class ReferenceResidualRegressor(nn.Module):
    def __init__(self, dims, text_hidden=128, context_hidden=304, dropout=.1, additive=True):
        super().__init__(); self.additive = additive
        self.text = nn.Sequential(nn.LayerNorm(dims["text"]), nn.Linear(dims["text"], text_hidden), nn.GELU(), nn.Dropout(dropout))
        self.text_head = nn.Linear(text_hidden, 1)
        context_dim = text_hidden + dims["audio"] + dims["vision"]
        self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, context_hidden), nn.GELU(), nn.Dropout(dropout))
        self.correction_head = nn.Linear(context_hidden, 1); self.joint_head = nn.Linear(context_hidden, 1)

    def forward(self, features):
        tvec = self.text(features["text"])
        context = self.context(torch.cat((tvec, features["audio"], features["vision"]), -1))
        reference, correction, direct = self.text_head(tvec).flatten(), self.correction_head(context).flatten(), self.joint_head(context).flatten()
        additive = reference + correction
        return {"prediction": additive if self.additive else direct, "text_reference": reference,
                "annotation_correction": correction, "direct_joint": direct, "additive_joint": additive}


def build_reference_model(case, dims, config):
    if case not in REFERENCE_CASES:
        raise ValueError("Unknown annotation-reference case")
    return ReferenceResidualRegressor(dims, config["text_hidden"], config["context_hidden"], config["dropout"], case.startswith("sum"))
