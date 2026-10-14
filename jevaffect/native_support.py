"""Known CH-SIMS native-support probability controls, not a new algorithm.

These PMFs predict released aggregate scores, not individual rater states.
Native regression labels and point evaluation are unchanged. MOSEI is not
supported by this eleven-level adapter.
"""
import math
import numpy as np
import torch
from torch import nn
from jevaffect.shape_controls import ShapeHead, probabilities

SUPPORT = np.asarray([-1., -.8, -.6, -.4, -.2, 0., .2, .4, .6, .8, 1.], np.float32)
NEURAL_CASES = ('ordinal', 'ordinal_projected', 'median_mdn_cells', 'median_mdn_cells_projected')
CASES = (*NEURAL_CASES, 'frequency_prior', 'frequency_prior_projected')


def support_tensor(device=None):
    return torch.as_tensor(SUPPORT.copy(), dtype=torch.float32, device=device)


def validate_probability(p):
    if (p.ndim != 2 or p.shape[1] != 11 or not len(p) or not p.is_floating_point()
            or not torch.isfinite(p).all() or torch.any(p < 0)
            or not torch.allclose(p.sum(-1), torch.ones(len(p), dtype=p.dtype, device=p.device), atol=1e-10, rtol=0)):
        raise ValueError('Finite normalized [N,11] native-score probabilities required')


def positive_probability(p):
    if p.ndim != 2 or p.shape[1] != 11 or not torch.isfinite(p).all() or torch.any(p < -1e-12):
        raise ValueError('Finite nonnegative native masses required')
    p = p.double().clamp_min(1e-12)
    return p/p.sum(-1, keepdim=True)


def project_median(p, anchor):
    """KL I-projection onto F(m-)<=.5<=F(m), with m detached.

    An off-grid m may be a nonunique generalized median; inverse-CDF quantiles
    still return support values. The proportional rescaling is classical.
    """
    validate_probability(p)
    if anchor.shape != (len(p),) or not torch.isfinite(anchor).all() or torch.any(anchor.abs() > 1):
        raise ValueError('One finite bounded native anchor per PMF required')
    if torch.any(p <= 0):
        raise ValueError('Strictly positive base PMFs required for KL projection')
    scores = support_tensor(p.device).double()[None]
    m = anchor.detach().double()[:, None]
    below, above = scores < m, scores > m
    lower_mass = (p*below).sum(-1, keepdim=True)
    upper_mass = (p*above).sum(-1, keepdim=True)
    # Denominators for inactive branches can be zero at endpoint anchors.
    lower = p*torch.where(below, .5/lower_mass.clamp_min(1e-300), .5/(1-lower_mass).clamp_min(1e-300))
    upper = p*torch.where(above, .5/upper_mass.clamp_min(1e-300), .5/(1-upper_mass).clamp_min(1e-300))
    result = torch.where(lower_mass > .5, lower, torch.where(upper_mass > .5, upper, p))
    return result


def target_indices(target):
    if target.ndim != 1 or not len(target) or not torch.isfinite(target).all():
        raise ValueError('Original finite native score vector required')
    matches = target.float()[:, None] == support_tensor(target.device)[None]
    if not torch.all(matches.sum(-1) == 1):
        raise ValueError('Target not on the verified CH TRAIN score support; no rounding allowed')
    return matches.long().argmax(-1)


def nll(p, target):
    validate_probability(p)
    index = target_indices(target)
    if len(index) != len(p):
        raise ValueError('Target and forecast counts differ')
    return -p[torch.arange(len(p), device=p.device), index].clamp_min(1e-12).log().mean()


def crps(p, target):
    """Exact CRPS energy identity, including off-grid original scalar targets.

    On support targets this equals gap-weighted ranked probability score.
    Off-support targets remain valid regression observations; never round them.
    """
    validate_probability(p)
    if (target.shape != (len(p),) or not torch.isfinite(target).all()
            or torch.any(target.abs() > 1)):
        raise ValueError('Original finite bounded native targets required')
    scores = support_tensor(p.device).double()
    distance = (scores[:, None]-scores[None, :]).abs()
    first = (p*(scores[None]-target.double()[:, None]).abs()).sum(-1)
    second = (p[:, :, None]*p[:, None, :]*distance[None]).sum((1,2))
    return first-.5*second


def hit_probability(p, point, tolerance=.1):
    validate_probability(p)
    if point.shape != (len(p),) or not torch.isfinite(point).all() or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Finite native point and positive native tolerance required')
    # Deliberately use float32 subtraction/comparison, exactly as hit truth.
    hit = (support_tensor(p.device)[None]-point.detach().float()[:, None]).abs() <= np.float32(tolerance)
    return (p*hit).sum(-1)


def author_bins(p):
    validate_probability(p)
    index = np.searchsorted(np.asarray([-.7,-.1,.1,.7],np.float32), SUPPORT, side='left')
    return torch.stack([p[:, np.flatnonzero(index == k).tolist()].sum(-1) for k in range(5)], -1)


@torch.no_grad()
def quantiles(p, levels):
    validate_probability(p)
    q = torch.as_tensor(levels, dtype=torch.float64, device=p.device).reshape(-1)
    if not len(q) or not torch.isfinite(q).all() or torch.any((q < 0) | (q > 1)):
        raise ValueError('Valid discrete quantile levels required')
    index = (p.cumsum(-1)[:, None, :] < q[None, :, None]).sum(-1).clamp_max(10)
    return support_tensor(p.device).double()[index]


class NativeSupportHead(nn.Module):
    def __init__(self, case, feature_dim=257, hidden=32, dropout=.1):
        super().__init__()
        if case not in NEURAL_CASES:
            raise ValueError('Registered neural native-support case required')
        self.case = case
        if case.startswith('ordinal'):
            self.context = nn.Sequential(nn.Linear(feature_dim,hidden),nn.GELU(),nn.Dropout(dropout))
            self.location = nn.Linear(hidden,1)
            self.raw_gaps = nn.Parameter(torch.full((9,), math.log(math.expm1(.2-1e-4))))
        else:
            self.reference = ShapeHead('median_mdn', feature_dim, hidden, dropout)

    def forward(self, context, anchor):
        if self.case.startswith('ordinal'):
            location = self.location(self.context(context)).double()
            gaps = nn.functional.softplus(self.raw_gaps.double())+1e-4
            cuts = torch.cat((gaps.new_zeros(1),gaps.cumsum(-1)))
            cuts = cuts-cuts.mean()
            cumulative = torch.sigmoid(cuts[None]-location)
            ends = torch.cat((torch.zeros_like(cumulative[:,:1]), cumulative, torch.ones_like(cumulative[:,:1])), -1)
            p = ends[:,1:]-ends[:,:-1]
        else:
            scores = SUPPORT.astype(np.float64)
            boundaries = (scores[:-1]+scores[1:])/2
            p = probabilities(self.reference(context,anchor), boundaries)
        base = positive_probability(p)
        final = project_median(base,anchor) if self.case.endswith('_projected') else base
        return dict(probability=final, unprojected_probability=base)


def frequency_prior(target):
    index = target_indices(target)
    counts = torch.bincount(index, minlength=11).double()+.5
    return counts/counts.sum()
