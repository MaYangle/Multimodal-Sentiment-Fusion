"""Censored shape references with a training-consistent MDN median.

Classical constructions only; no novelty or performance claim. The implicit
median gradient is dq/dtheta = -(partial F(q)/partial theta)/density(q).
"""
import math
import torch
from torch import nn
from jevaffect import anchor_cdf, anchored_bernstein

CASES = ('scale', 'mixture', 'bern_uniform', 'bern_tied', 'bern_d3',
         'bern_d6', 'bern_d12', 'median_mdn', 'mdn', 'spline')
OUTPUTS = dict(scale=1, mixture=2, bern_uniform=2, bern_tied=9,
               bern_d3=10, bern_d6=16, bern_d12=28, median_mdn=16,
               mdn=16, spline=9)


def mdn_parameters(anchor, raw_scale, logits, *, recenter=True, bound=1.):
    if (anchor.ndim != 1 or not len(anchor) or raw_scale.shape != anchor.shape
            or logits.shape != (len(anchor), 15) or not math.isfinite(bound)
            or bound <= 0 or torch.any(anchor.abs() > bound)
            or any(not torch.isfinite(v).all() for v in (anchor, raw_scale, logits))):
        raise ValueError('Finite bounded anchors and 15-kernel parameters required')
    m = anchor.detach().double()
    scale = nn.functional.softplus(raw_scale.double()) + 1e-4
    weights = logits.double().softmax(-1)
    centers = torch.linspace(-1.4, 1.4, 15, dtype=torch.float64, device=m.device) * bound
    def latent(x):
        z = torch.sigmoid((x[:, None] - centers[None]) / scale[:, None])
        return (weights*z).sum(-1), (weights*z*(1-z)/scale[:, None]).sum(-1)
    if recenter:
        with torch.no_grad():
            lo, hi = centers[0]-40*scale, centers[-1]+40*scale
            for _ in range(64):
                mid = (lo+hi)/2
                lower = latent(mid)[0] < .5
                lo, hi = torch.where(lower, mid, lo), torch.where(lower, hi, mid)
            root = ((lo+hi)/2).detach()
        value, density = latent(root)
        # Zero-valued correction preserves the solved forward root exactly,
        # while attaching the implicit derivative to the final scored CDF.
        correction = (value-value.detach())/density.detach().clamp_min(1e-300)
        median = root-correction
        shift = m-median
    else:
        shift = torch.zeros_like(m)
    return dict(anchor=m, scale=scale, weights=weights, centers=centers,
                shift=shift, bound=float(bound), median_fixed=bool(recenter))


def mdn_cdf(params, value):
    m = params['anchor']
    x = _queries(m, value)
    shift = params['shift'][:, None] if x.ndim == 2 else params['shift']
    scale = params['scale'][:, None] if x.ndim == 2 else params['scale']
    weights = params['weights'][:, None] if x.ndim == 2 else params['weights']
    z = torch.sigmoid((x[..., None]-shift[..., None]-params['centers'])/scale[..., None])
    result = (weights*z).sum(-1)
    return torch.where(x < -params['bound'], 0., torch.where(x >= params['bound'], 1., result))


def _queries(anchor, value):
    x = torch.as_tensor(value, dtype=torch.float64, device=anchor.device)
    if x.ndim == 0:
        x = x.expand_as(anchor)
    if x.ndim not in (1, 2) or x.shape[0] != len(anchor) or not torch.isfinite(x).all():
        raise ValueError('Finite scalar/per-sample/grid queries required')
    return x


def spline_cdf(params, value):
    x = _queries(params['anchor'], value)
    u = anchor_cdf.cdf(params['base'], x)
    increments = .5*params['logits'].double().reshape(-1, 2, 4).softmax(-1)
    knots = torch.cat((torch.zeros_like(increments[:, :1, 0]),
                       increments[:, 0].cumsum(-1),
                       .5+increments[:, 1].cumsum(-1)), -1)
    index = (u*8).floor().long().clamp(0, 7)
    if x.ndim == 2:
        knots = knots[:, None].expand(-1, x.shape[1], -1)
    left = knots.gather(-1, index[..., None]).squeeze(-1)
    right = knots.gather(-1, (index+1)[..., None]).squeeze(-1)
    result = left+(right-left)*(u*8-index)
    return result.clamp(0., 1.)


class ShapeHead(nn.Module):
    def __init__(self, case, feature_dim=257, hidden=32, dropout=.1):
        super().__init__()
        if case not in CASES:
            raise ValueError('Unregistered probability shape case')
        self.case = case
        self.context = nn.Sequential(nn.Linear(feature_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, OUTPUTS[case])
        nn.init.zeros_(self.head.bias)
        # Match initial native logistic scale/margins without extra teacher inputs.
        with torch.no_grad():
            self.head.bias[0] = math.log(math.expm1(.35-1e-4))
            if case.startswith('bern_'):
                self.head.bias[1] = self.head.bias[0]
            if case == 'mixture':
                self.head.bias[1] = -2.

    def forward(self, context, anchor):
        raw = self.head(self.context(context))
        case = self.case
        if case in ('scale', 'mixture'):
            p = anchor_cdf.parameters(anchor, raw[:, 0], raw[:, 1] if case == 'mixture'
                                      else torch.zeros_like(anchor), mixture=case == 'mixture')
            return dict(kind='logistic', parameters=p)
        if case.startswith('bern_'):
            degree = int(case[6:]) if case[5:6] == 'd' else 6
            if case == 'bern_uniform':
                logits = torch.zeros((len(anchor), 2, degree+1), dtype=raw.dtype, device=raw.device)
            elif case == 'bern_tied':
                logits = raw[:, 2:, None].transpose(1, 2).expand(-1, 2, -1)
            else:
                logits = raw[:, 2:].reshape(-1, 2, degree+1)
            p = anchored_bernstein.parameters(anchor, raw[:, :2], logits, degree=degree)
            return dict(kind='bernstein', parameters=p)
        if case in ('median_mdn', 'mdn'):
            return dict(kind='mdn', parameters=mdn_parameters(anchor, raw[:, 0], raw[:, 1:],
                                                            recenter=case == 'median_mdn'))
        p = dict(anchor=anchor.detach().double(), bound=1., logits=raw[:, 1:],
                 base=anchor_cdf.parameters(anchor, raw[:, 0], torch.zeros_like(anchor), mixture=False))
        return dict(kind='spline', parameters=p)


def cdf(output, value):
    kind, p = output['kind'], output['parameters']
    functions = dict(logistic=anchor_cdf.cdf, bernstein=anchored_bernstein.cdf,
                     mdn=mdn_cdf, spline=spline_cdf)
    return functions[kind](p, value)


def probabilities(output, boundaries):
    p = output['parameters']
    edges = torch.as_tensor(boundaries, dtype=torch.float64, device=p['anchor'].device)
    if (edges.ndim != 1 or not len(edges) or not torch.isfinite(edges).all()
            or torch.any(edges[1:] <= edges[:-1]) or torch.any(edges.abs() >= p['bound'])):
        raise ValueError('Increasing interior native bin boundaries required')
    values = cdf(output, edges[None].expand(len(p['anchor']), -1))
    values = torch.cat((torch.zeros_like(values[:, :1]), values, torch.ones_like(values[:, :1])), -1)
    return values[:, 1:]-values[:, :-1]


def hit_probability(output, raw_point, tolerance):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Positive native error tolerance required')
    p = output['parameters']
    center = raw_point.detach().double()
    lo, hi = center-tolerance, center+tolerance
    before = torch.where(lo <= -p['bound'], 0., cdf(output, lo))
    return (cdf(output, hi)-before).clamp(0., 1.)


@torch.no_grad()
def quantiles(output, levels):
    p = output['parameters']
    q = torch.as_tensor(levels, dtype=torch.float64, device=p['anchor'].device).reshape(-1)
    if not len(q) or not torch.isfinite(q).all() or torch.any((q < 0) | (q > 1)):
        raise ValueError('Valid quantile levels required')
    target = q[None].expand(len(p['anchor']), -1)
    lo, hi = torch.full_like(target, -p['bound']), torch.full_like(target, p['bound'])
    for _ in range(64):
        mid = (lo+hi)/2
        lower = cdf(output, mid) < target
        lo, hi = torch.where(lower, mid, lo), torch.where(lower, hi, mid)
    result = (lo+hi)/2
    if output['kind'] != 'mdn' or p['median_fixed']:
        result = torch.where(target == .5, p['anchor'][:, None], result)
    return torch.where(target == 0, -p['bound'], torch.where(target == 1, p['bound'], result))
