"""Known covariance/scale controls for frozen scalar-expert uncertainty.

Fitting functions accept TRAIN OOF arrays only; cohort authority is enforced by
the study driver. Covariance readout is moment matched, not an exact sum law.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import t
from jevaffect.student_pool import components


def expert_variances(parameters, availability):
    mu, df, scale, weights, a = components(parameters, availability)
    return mu, scale ** 2 * df / (df - 2), weights, a


def fit_correlation(parameters, availability, targets, shrinkage=.1):
    mu, var, _, a = expert_variances(parameters, availability)
    y = np.asarray(targets, np.float64)
    if y.shape != (len(mu),) or not np.isfinite(y).all() or not a.all() or len(y) <= mu.shape[1] or not 0 <= shrinkage <= 1:
        raise ValueError("Finite full-modality TRAIN OOF cohort and valid shrinkage required")
    z = (y[:, None] - mu) / np.sqrt(var)
    if np.any(z.std(0) <= 1e-10):
        raise ValueError("Constant standardized residual column")
    raw = np.corrcoef(z, rowvar=False)
    return (1 - shrinkage) * raw + shrinkage * np.eye(raw.shape[0]), raw


def covariance_variance(parameters, availability, correlation):
    _, variance, weights, _ = expert_variances(parameters, availability)
    r = np.asarray(correlation, np.float64)
    k = variance.shape[1]
    if r.shape != (k, k) or not np.isfinite(r).all() or not np.allclose(r, r.T, atol=1e-10) or not np.allclose(np.diag(r), 1):
        raise ValueError("Symmetric unit-diagonal correlation required")
    if np.linalg.eigvalsh(r).min() < -1e-8:
        raise ValueError("Correlation must be positive semidefinite")
    weighted_std = weights * np.sqrt(variance)
    result = np.einsum("ni,ij,nj->n", weighted_std, r, weighted_std)
    if np.any(result <= 0) or not np.isfinite(result).all():
        raise ValueError("Invalid aggregate variance")
    return result


def nig_variance(parameters):
    p = np.asarray(parameters, np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all():
        raise ValueError("Finite [N,4] NIG required")
    _, v, alpha, beta = p.T
    if np.any(v <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("Invalid NIG")
    return beta * (1 + v) / (v * (alpha - 1))


def replace_variance(parameters, variance, scale=1.):
    p = np.array(parameters, np.float64, copy=True)
    requested = np.asarray(variance, np.float64) * scale
    if requested.shape != (len(p),) or np.any(requested <= 0) or not np.isfinite(requested).all():
        raise ValueError("Positive finite variance required")
    nig_variance(p)
    p[:, 3] = requested * p[:, 1] * (p[:, 2] - 1) / (1 + p[:, 1])
    return p


def fit_variance_scale(parameters, variance, targets, bounds=(-6, 6)):
    p = replace_variance(parameters, variance); y = np.asarray(targets, np.float64)
    if y.shape != (len(p),) or not np.isfinite(y).all():
        raise ValueError("Invalid calibration targets")
    mu, v, alpha, beta = p.T
    base_scale = np.sqrt(beta * (1 + v) / (alpha * v))
    def objective(log_q):
        return float(-t.logpdf(y, df=2 * alpha, loc=mu, scale=base_scale * np.exp(log_q / 2)).mean())
    return _minimize(objective, bounds)


def fit_pool_scale(parameters, availability, targets, bounds=(-6, 6)):
    mu, df, scale, weights, _ = components(parameters, availability)
    y = np.asarray(targets, np.float64)
    if y.shape != (len(mu),) or not np.isfinite(y).all():
        raise ValueError("Invalid pool calibration targets")
    log_weights = np.full_like(weights, -np.inf)
    np.log(weights, out=log_weights, where=weights > 0)
    def objective(log_q):
        return float(-logsumexp(log_weights + t.logpdf(y[:, None], df=df, loc=mu, scale=scale * np.exp(log_q / 2)), axis=1).mean())
    return _minimize(objective, bounds)


def _minimize(objective, bounds):
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("Invalid scale search bounds")
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError("Scale fit failed")
    return {"variance_scale": float(np.exp(result.x)), "train_fit_density_NLL": float(result.fun),
            "log_bounds": list(bounds), "at_boundary": bool(min(result.x - bounds[0], bounds[1] - result.x) < 1e-4),
            "fit_scope": "TRAIN source-disjoint OOF only"}
