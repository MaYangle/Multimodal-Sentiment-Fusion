"""Median-anchored logistic shape prototype; not a validated contribution.

The center-kernel mixture preserves a frozen native point as a median.
Probabilities concern observed score intervals, never latent human neutrality.
"""
import math
import torch
from torch.nn import functional as F


def parameters(anchor, raw_scale, weight_logit, *, bound=1., kernel_scale=.05, mixture=True):
    values = (anchor, raw_scale, weight_logit)
    if any(not torch.is_tensor(v) or not v.is_floating_point() or v.ndim != 1 for v in values) or not len(anchor) or any(v.shape != anchor.shape or v.device != anchor.device for v in values):
        raise ValueError("Equal nonempty floating [batch] tensors on one device required")
    if not math.isfinite(bound) or bound <= 0 or not math.isfinite(kernel_scale) or kernel_scale <= 0:
        raise ValueError("Positive finite native bound and kernel scale required")
    if any(not torch.isfinite(v).all() for v in values) or torch.any(anchor.abs() > bound):
        raise ValueError("Finite parameters and an explicitly bounded native point anchor required")
    m = anchor.detach().double()
    scale = F.softplus(raw_scale.double()) + 1e-4
    u = torch.sigmoid(m / kernel_scale)
    maximum_weight = .5 / torch.maximum(u, 1 - u)
    weight = (1 - 1e-4) * maximum_weight * torch.sigmoid(weight_logit.double()) if mixture else torch.zeros_like(m)
    q = (.5 - weight * u) / (1 - weight)
    if torch.any(q <= 0) or torch.any(q >= 1):
        raise ValueError("Median constraint produced an invalid CDF quantile")
    location = m - scale * (torch.log(q) - torch.log1p(-q))
    if not torch.isfinite(location).all():
        raise ValueError("Nonfinite median-preserving location")
    return {"anchor": m, "scale": scale, "weight": weight, "location": location,
            "bound": float(bound), "kernel_scale": float(kernel_scale)}


def cdf(params, value):
    m = params["anchor"]
    x = torch.as_tensor(value, device=m.device, dtype=torch.float64)
    if x.ndim == 0:
        x = x.expand_as(m)
    if x.ndim not in (1, 2) or x.shape[0] != len(m) or torch.isnan(x).any():
        raise ValueError("CDF value must be scalar, [batch] or [batch,queries]")
    def expand(v):
        return v[:, None] if x.ndim == 2 else v
    base = torch.sigmoid((x - expand(params["location"])) / expand(params["scale"]))
    kernel = torch.sigmoid(x / params["kernel_scale"])
    weight = expand(params["weight"])
    result = weight * kernel + (1 - weight) * base
    # Censoring retains endpoint atoms, including the jump at the lower bound.
    return torch.where(x < -params["bound"], 0., torch.where(x >= params["bound"], 1., result))


def bin_probabilities(params, boundaries):
    edges = torch.as_tensor(boundaries, dtype=torch.float64, device=params["anchor"].device)
    if edges.ndim != 1 or not len(edges) or not torch.isfinite(edges).all() or torch.any(edges[1:] <= edges[:-1]) or torch.any(edges.abs() >= params["bound"]):
        raise ValueError("Sorted finite interior native bin boundaries required")
    values = cdf(params, edges[None].expand(len(params["anchor"]), -1))
    endpoints = torch.cat((torch.zeros_like(values[:, :1]), values, torch.ones_like(values[:, :1])), 1)
    probabilities = endpoints[:, 1:] - endpoints[:, :-1]
    if torch.any(probabilities < 0) or not torch.isfinite(probabilities).all():
        raise ValueError("Invalid native bin probabilities")
    return probabilities


def error_band_probability(params, tolerance):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive native-unit error tolerance required")
    m = params["anchor"]
    lo, hi = m - tolerance, m + tolerance
    before_left = torch.where(lo <= -params["bound"], 0., cdf(params, lo))
    return cdf(params, hi) - before_left


@torch.no_grad()
def quantile(params, probabilities):
    p = torch.as_tensor(probabilities, dtype=torch.float64, device=params["anchor"].device)
    if p.ndim > 1 or not torch.isfinite(p).all() or torch.any(p < 0) or torch.any(p > 1):
        raise ValueError("Finite scalar/vector quantiles in[0,1] required")
    scalar = p.ndim == 0; p = p.reshape(-1)
    target = p[None].expand(len(params["anchor"]), -1)
    lo = torch.full_like(target, -params["bound"]); hi = torch.full_like(target, params["bound"])
    for _ in range(60):
        mid = (lo + hi) / 2
        lower = cdf(params, mid) < target
        lo = torch.where(lower, mid, lo); hi = torch.where(lower, hi, mid)
    value = (lo + hi) / 2
    value = torch.where(target == 0, -params["bound"], torch.where(target == 1, params["bound"], value))
    return value[:, 0] if scalar else value
