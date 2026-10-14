"""Proper linear pooling of Student-t predictive distributions (known baseline).

This preserves the gamma-weighted point mean but does not sum correlated NIG
evidence. It is a diagnostic comparator, not a new claimed algorithm.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t, spearmanr


def components(parameters, availability):
    p = np.asarray(parameters, np.float64)
    a = np.asarray(availability, bool)
    if p.ndim != 3 or p.shape[-1] != 4 or a.shape != p.shape[:2] or not np.isfinite(p).all() or np.any(a.sum(1) == 0):
        raise ValueError("Finite [N,K,4] and nonempty availability required")
    mu, v, alpha, beta = np.moveaxis(p, -1, 0)
    if np.any(v <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("Invalid NIG component")
    weight = np.where(a, v, 0)
    weight /= weight.sum(1, keepdims=True)
    scale = np.sqrt(beta * (1 + v) / (alpha * v))
    return mu, 2 * alpha, scale, weight, a


def pool_quantile(parameters, availability, probability):
    if not 0 < probability < 1:
        raise ValueError("Quantile probability must lie in(0,1)")
    mu, df, scale, weight, a = components(parameters, availability)
    q = t.ppf(probability, df=df, loc=mu, scale=scale)
    lower = np.where(a, q, np.inf).min(1)
    upper = np.where(a, q, -np.inf).max(1)
    # Mixture quantile is bracketed by component quantiles at the same level.
    for _ in range(45):
        middle = (lower + upper) / 2
        cdf = (weight * t.cdf(middle[:, None], df=df, loc=mu, scale=scale)).sum(1)
        below = cdf < probability
        lower = np.where(below, middle, lower)
        upper = np.where(below, upper, middle)
    return (lower + upper) / 2


def pool_metrics(parameters, availability, targets, native_range):
    mu, df, scale, weight, a = components(parameters, availability)
    y = np.asarray(targets, np.float64).reshape(-1)
    if len(y) != len(mu) or not np.isfinite(y).all():
        raise ValueError("Invalid target array")
    mean = (weight * mu).sum(1)
    variance = (weight * (scale ** 2 * df / (df - 2) + (mu - mean[:, None]) ** 2)).sum(1)
    logweights = np.full_like(weight, -np.inf)
    np.log(weight, out=logweights, where=weight > 0)
    nll = -logsumexp(logweights + t.logpdf(y[:, None], df=df, loc=mu, scale=scale), axis=1)
    errors = np.abs(np.clip(mean, *native_range) - y)
    intervals = {}
    widths = None
    for level in (.5, .8, .9, .95):
        low = pool_quantile(parameters, availability, (1 - level) / 2)
        high = pool_quantile(parameters, availability, (1 + level) / 2)
        if level == .9:
            widths = high - low
        intervals[str(level)] = {"coverage": float(((y >= low) & (y <= high)).mean()), "mean_width": float((high - low).mean())}
    corr = lambda x: None if np.ptp(x) == 0 or np.ptp(errors) == 0 else float(spearmanr(x, errors).statistic)
    return {"density_NLL": float(nll.mean()), "intervals": intervals, "error_variance_spearman": corr(variance),
            "error_width90_spearman": corr(widths), "point_mean": mean.tolist(),
            "mean_predictive_variance": float(variance.mean()), "distribution": "proper linear Student-t pool, gamma weights"}
