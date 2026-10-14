"""Matched AnchorCDF calibration heads and native probability diagnostics."""
import math
import numpy as np
import torch
from torch import nn
from jevaffect.anchor_cdf import parameters, bin_probabilities, error_band_probability, quantile
from jevaffect.native_metrics import native_metrics

CASES = ("scale", "mixture", "views", "disagreement", "direct_hit", "mdn")
BOUNDARIES = np.array([-.7, -.1, .1, .7], np.float32)
TOLERANCE = float(np.float32(.1))


def disagreement(views):
    """Differences of common-M model forecasts, not human modality labels."""
    values = np.asarray(views, np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Finite [samples,three views] forecasts required")
    values = np.clip(values, -1., 1.)
    signed = np.stack((values[:, 0] - values[:, 1], values[:, 0] - values[:, 2], values[:, 1] - values[:, 2]), 1)
    return np.concatenate((signed, np.abs(signed)), 1).astype(np.float32)


def bin_targets(target):
    y = np.asarray(target, np.float32)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or np.any(np.abs(y) > 1):
        raise ValueError("Finite native CH-SIMS targets required")
    return np.searchsorted(BOUNDARIES, y, side="left").astype(np.int64)


def normalizer(values, fit_indices):
    """Fit only on optimization rows; tiny-variance coordinates keep scale1."""
    x = np.asarray(values, np.float32); indices = np.asarray(fit_indices, np.int64)
    if x.ndim != 2 or not np.isfinite(x).all() or not len(indices) or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(x):
        raise ValueError("Invalid normalization optimization cohort")
    mean = x[indices].astype(np.float64).mean(0)
    scale = x[indices].astype(np.float64).std(0)
    scale[scale < 1e-6] = 1.
    return {"mean": mean.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist(), "fit_indices": indices.tolist()}


def normalize(values, stats):
    result = (np.asarray(values, np.float32) - np.asarray(stats["mean"], np.float32)) / np.asarray(stats["scale"], np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite normalized confidence features")
    return result.astype(np.float32)


class CalibrationHead(nn.Module):
    def __init__(self, feature_dim, hidden=32, dropout=.1):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.scale_head = nn.Linear(hidden, 1)
        self.weight_head = nn.Linear(hidden + 6, 1)
        self.mdn_head = nn.Linear(hidden, 15)
        nn.init.constant_(self.scale_head.bias, math.log(math.expm1(.35 - 1e-4)))
        nn.init.constant_(self.weight_head.bias, -2.)

    def forward(self, features, differences, anchor, case):
        if case not in CASES or differences.shape != (len(features), 6) or anchor.shape != (len(features),):
            raise ValueError("Unknown calibration case or input schema")
        context = self.context(features)
        d = differences if case in ("views", "disagreement", "direct_hit") else torch.zeros_like(differences)
        raw_scale = self.scale_head(context).flatten()
        z = self.weight_head(torch.cat((context, d), -1)).flatten()
        if case == "direct_hit":
            return {"p_hit": torch.sigmoid(z), "hit_logit": z}
        if case == "mdn":
            distribution = {"anchor": anchor.detach().double(), "scale": nn.functional.softplus(raw_scale.double()) + 1e-4,
                "weights": self.mdn_head(context).double().softmax(1), "centers": torch.linspace(-1.4, 1.4, 15, device=anchor.device, dtype=torch.float64)}
            edges = torch.as_tensor(BOUNDARIES.astype(np.float64), device=anchor.device)
            probabilities = _mdn_cdf(distribution, edges[None].expand(len(anchor), -1))
            ends = torch.cat((torch.zeros_like(probabilities[:, :1]), probabilities, torch.ones_like(probabilities[:, :1])), 1)
            lo, hi = anchor.double() - TOLERANCE, anchor.double() + TOLERANCE
            before = torch.where(lo <= -1, 0., _mdn_cdf(distribution, lo))
            return {"parameters": distribution, "bins": ends[:, 1:] - ends[:, :-1],
                    "p_hit": _mdn_cdf(distribution, hi) - before}
        distribution = parameters(anchor, raw_scale, z, mixture=case != "scale")
        return {"parameters": distribution,
                "bins": bin_probabilities(distribution, BOUNDARIES.astype(np.float64).tolist()),
                "p_hit": error_band_probability(distribution, TOLERANCE)}


def _mdn_cdf(params, value):
    x = torch.as_tensor(value, dtype=torch.float64, device=params["anchor"].device)
    if x.ndim == 1:
        raw = torch.sigmoid((x[:, None] - params["centers"][None]) / params["scale"][:, None])
        output = (params["weights"] * raw).sum(-1)
    else:
        raw = torch.sigmoid((x[:, :, None] - params["centers"][None, None]) / params["scale"][:, None, None])
        output = (params["weights"][:, None] * raw).sum(-1)
    return torch.where(x < -1, 0., torch.where(x >= 1, 1., output))


@torch.no_grad()
def _mdn_quantile(params, probabilities):
    p = torch.as_tensor(probabilities, dtype=torch.float64, device=params["anchor"].device).reshape(-1)
    target = p[None].expand(len(params["anchor"]), -1)
    lo, hi = torch.full_like(target, -1), torch.full_like(target, 1)
    for _ in range(60):
        mid = (lo + hi) / 2; lower = _mdn_cdf(params, mid) < target
        lo = torch.where(lower, mid, lo); hi = torch.where(lower, hi, mid)
    return torch.where(target == 0, -1., torch.where(target == 1, 1., (lo + hi) / 2))


def objective(output, target, anchor, case):
    if case == "direct_hit":
        hit = ((anchor.float() - target.float()).abs() <= .1).float()
        return nn.functional.binary_cross_entropy_with_logits(output["hit_logit"], hit)
    classes = torch.from_numpy(bin_targets(target.detach().cpu().numpy())).to(target.device)
    assigned = output["bins"][torch.arange(len(target), device=target.device), classes]
    return -assigned.clamp_min(1e-12).log().mean()


def event_metrics(probability, hit, errors):
    p, y, e = np.asarray(probability, np.float64), np.asarray(hit, np.float64), np.asarray(errors, np.float64)
    if p.ndim != 1 or p.shape != y.shape or p.shape != e.shape or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("Invalid event probability diagnostic")
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    groups = np.minimum((p * 15).astype(int), 14)
    reliability = []
    for group in range(15):
        mask = groups == group
        if mask.any():
            reliability.append({"bin": group, "count": int(mask.sum()), "mean_probability": float(p[mask].mean()), "observed_hit_rate": float(y[mask].mean())})
    order = np.argsort(-p, kind="stable")
    risk = np.cumsum(e[order]) / np.arange(1, len(e) + 1)
    return {"hit_count": int(y.sum()), "count": len(y), "hit_Brier": float(np.mean((p - y) ** 2)),
        "hit_log_score": float(-np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped))),
        "hit_ECE15": float(sum(r["count"] / len(y) * abs(r["mean_probability"] - r["observed_hit_rate"]) for r in reliability)),
        "risk_AURC": float(risk.mean()), "reliability": reliability,
        "risk_at_coverage": {str(c): float(risk[max(0, math.ceil(len(y) * c) - 1)]) for c in (.1, .25, .5, .75, 1.)}}


def diagnostics(anchor, target, output):
    m, y = np.asarray(anchor, np.float32), np.asarray(target, np.float32)
    errors = np.abs(m - y); hit = errors <= np.float32(.1)
    result = {"native": native_metrics(m, y, "simsv2"), **event_metrics(output["p_hit"], hit, errors)}
    if "bins" in output:
        probabilities = np.asarray(output["bins"], np.float64)
        classes = bin_targets(y)
        if probabilities.shape != (len(y), 5) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1., atol=1e-10):
            raise ValueError("Invalid common native-bin forecasts")
        onehot = np.eye(5)[classes]
        result.update(bin_log_score=float(-np.log(np.maximum(probabilities[np.arange(len(y)), classes], 1e-12)).mean()),
                      bin_Brier=float(np.mean(np.sum((probabilities - onehot) ** 2, 1))))
    if "intervals" in output:
        result["intervals"] = {}
        for nominal, pair in output["intervals"].items():
            lo, hi = np.asarray(pair[0]), np.asarray(pair[1])
            if lo.shape != y.shape or hi.shape != y.shape or np.any(lo > hi) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
                raise ValueError("Invalid native predictive intervals")
            result["intervals"][str(nominal)] = {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean_width": float((hi-lo).mean())}
    return result


@torch.no_grad()
def describe(output):
    values = {"p_hit": output["p_hit"].detach().cpu().numpy()}
    if "bins" in output:
        values["bins"] = output["bins"].detach().cpu().numpy()
        params = output["parameters"]
        get_quantile = _mdn_quantile if "weights" in params else quantile
        probabilities = [.25, .75, .1, .9, .05, .95, .025, .975]
        qs = get_quantile(params, probabilities).cpu().numpy()
        values["intervals"] = {nominal: (qs[:, 2*i], qs[:, 2*i + 1]) for i, nominal in enumerate((.5, .8, .9, .95))}
        values["median"] = get_quantile(params, .5).reshape(-1).cpu().numpy()
        if "weight" in params:
            values["mixture_weight"] = params["weight"].cpu().numpy()
        values["scale"] = params["scale"].cpu().numpy()
    return values
