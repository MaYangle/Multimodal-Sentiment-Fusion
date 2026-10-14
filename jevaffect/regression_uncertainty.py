"""Distribution-matched evaluation for NIG/Student-t sentiment regression."""
from __future__ import annotations
import numpy as np


def uncertainty_metrics(parameters, target, native_range):
    from scipy.stats import t, spearmanr
    params = np.asarray(parameters, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if params.shape != (len(y), 4) or not len(y) or not np.isfinite(params).all() or not np.isfinite(y).all():
        raise ValueError("Finite [N,4] NIG parameters and [N] targets required")
    mu, v, alpha, beta = params.T
    if np.any(v <= 0) or np.any(alpha <= 1) or np.any(beta <= 0):
        raise ValueError("NIG variance requires v>0, alpha>1, beta>0")
    low, high = native_range
    if not low < high or np.any(y < low - 1e-6) or np.any(y > high + 1e-6):
        raise ValueError("Targets outside declared native range")
    scale = np.sqrt(beta * (1 + v) / (alpha * v))
    variance = beta * (1 + v) / (v * (alpha - 1))
    errors = np.abs(np.clip(mu, low, high) - y)
    ranking = np.argsort(variance, kind="stable")
    rank_corr = None if np.ptp(variance) == 0 or np.ptp(errors) == 0 else float(spearmanr(variance, errors).statistic)
    nll = -t.logpdf(y, df=2 * alpha, loc=mu, scale=scale)
    if not np.isfinite(nll).all():
        raise ValueError("Nonfinite Student-t likelihood")
    intervals = {}
    for level in (.5, .8, .9, .95):
        quantile = t.ppf((1 + level) / 2, df=2 * alpha)
        left, right = mu - scale * quantile, mu + scale * quantile
        intervals[str(level)] = {"coverage": float(np.mean((y >= left) & (y <= right))),
                                 "mean_width": float(np.mean(right - left)),
                                 "range_intersection_mean_width": float(np.maximum(0, np.minimum(high, right) - np.maximum(low, left)).mean())}
    selective = {}
    for fraction in (.25, .5, .75, 1.):
        n = max(1, int(np.ceil(len(y) * fraction)))
        selective[str(fraction)] = {"count": n, "MAE": float(errors[ranking[:n]].mean())}
    return {"count": len(y), "distribution": "unbounded Student-t predictive density from NIG",
            "density_NLL": float(nll.mean()), "error_vs_variance_spearman": rank_corr,
            "mean_predictive_variance": float(variance.mean()), "intervals": intervals,
            "selective_MAE": selective,
            "density_limit": "Quantized bounded sentiment is treated as a continuous density; NLL is not class cross-entropy."}
