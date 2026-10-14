"""Native continuous-score protocols; no common sentiment label scale.

Definitions: thuiar/ch-sims-v2 utils/metricsTop.py and thuiar/MMSA
src/MMSA/utils/metricsTop.py. Full precision is retained; undefined statistics
are None with support counts rather than NaN in JSON artifacts.
"""
from __future__ import annotations
import numpy as np


def weighted_f1(truth, pred):
    if len(truth) == 0:
        return None
    total = 0.0
    for label in np.unique(truth):
        support = np.count_nonzero(truth == label)
        tp = np.count_nonzero((truth == label) & (pred == label))
        denominator = support + np.count_nonzero(pred == label)
        total += support * 2 * tp / denominator
    return float(total / len(truth))


def native_metrics(prediction, target, dataset):
    p, y = np.asarray(prediction).reshape(-1), np.asarray(target).reshape(-1)
    if len(y) == 0 or p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Predictions and targets must be finite, nonempty equal-size arrays")
    if dataset not in ("simsv2", "mosei"):
        raise ValueError("Unknown native dataset")
    limit = 1 if dataset == "simsv2" else 3
    if np.any(np.abs(y) > limit + 1e-6):
        raise ValueError("Target outside native label range")
    if dataset == "simsv2":
        p, y = np.clip(p, -1, 1), np.clip(y, -1, 1)
    corr = None if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0 else float(np.corrcoef(p, y)[0, 1])
    result = {"count": len(y), "MAE": float(np.mean(np.abs(p - y))), "Corr": corr}
    if dataset == "simsv2":
        for count, boundaries in ((2, [0]), (3, [-.1, .1]), (5, [-.7, -.1, .1, .7])):
            # Author intervals are open on the left, closed on the right.
            py = np.searchsorted(np.asarray(boundaries, dtype=p.dtype), p, side="left")
            yy = np.searchsorted(np.asarray(boundaries, dtype=y.dtype), y, side="left")
            result[f"Mult_acc_{count}"] = float(np.mean(py == yy))
            if count == 2:
                result["F1_score"] = weighted_f1(yy, py)
                weak = (y >= -.4) & (y <= .4)
                result["weak_count"] = int(weak.sum())
                result["Mult_acc_2_weak"] = float(np.mean(py[weak] == yy[weak])) if weak.any() else None
        denominator = float(np.sum((y - y.mean()) ** 2))
        numerator = float(np.sum((y - p) ** 2))
        result["R_squre"] = 1 - numerator / denominator if denominator else (1.0 if numerator == 0 else 0.0)
    else:
        for count, clip in ((5, 2), (7, 3)):
            result[f"Mult_acc_{count}"] = float(np.mean(np.round(np.clip(p, -clip, clip)) == np.round(np.clip(y, -clip, clip))))
        result["Has0_acc_2"] = float(np.mean((p >= 0) == (y >= 0)))
        result["Has0_F1_score"] = weighted_f1(y >= 0, p >= 0)
        nonzero = y != 0
        result["nonzero_count"] = int(nonzero.sum())
        result["Non0_acc_2"] = float(np.mean((p[nonzero] > 0) == (y[nonzero] > 0))) if nonzero.any() else None
        result["Non0_F1_score"] = weighted_f1(y[nonzero] > 0, p[nonzero] > 0)
    return result
