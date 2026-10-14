"""Differentiable bounded-CDF CRPS quadrature on original scalar targets.

Gauss-Legendre nodes split at the frozen median and the observation. This
scores the final censored predictive law, including its endpoint masses.
It assumes no discrete target support and makes no novelty claim.
"""
from functools import lru_cache
import numpy as np
import torch
from jevaffect.shape_controls import cdf


@lru_cache(maxsize=16)
def _rule(nodes):
    if not isinstance(nodes, int) or isinstance(nodes, bool) or not 8 <= nodes <= 512:
        raise ValueError('Integer Gauss-Legendre order8..512 required')
    x, w = np.polynomial.legendre.leggauss(nodes)
    return x.copy(), w.copy()


def per_sample(output, target, nodes=48):
    p = output['parameters']
    m = p['anchor'].detach().double()
    bound = p['bound']
    if (target.shape != m.shape or not len(m) or not torch.isfinite(target).all()
            or not np.isfinite(bound) or bound <= 0 or torch.any(target.abs() > bound)
            or not torch.isfinite(m).all() or torch.any(m.abs() > bound)
            or target.device != m.device):
        raise ValueError('Original finite scalar targets and anchors in the native bound required')
    y = target.detach().double()
    breaks = torch.stack((torch.full_like(m, -bound), torch.minimum(m,y),
                          torch.maximum(m,y), torch.full_like(m,bound)), -1)
    left, right = breaks[:, :-1], breaks[:, 1:]
    centers, half_width = (left+right)/2, (right-left)/2
    rule_x, rule_w = _rule(nodes)
    x = torch.as_tensor(rule_x, device=m.device, dtype=torch.float64)
    w = torch.as_tensor(rule_w, device=m.device, dtype=torch.float64)
    queries = centers[:, :, None]+half_width[:, :, None]*x
    flat = queries.reshape(len(m), -1)
    probabilities = cdf(output, flat)
    truth = (flat >= y[:, None]).double()
    integrand = (probabilities-truth).square().reshape(len(m),3,nodes)
    result = (integrand*w[None,None]*half_width[:,:,None]).sum((1,2))
    if not torch.isfinite(result).all() or torch.any(result < 0):
        raise ValueError('Invalid final-CDF native CRPS integral')
    return result
