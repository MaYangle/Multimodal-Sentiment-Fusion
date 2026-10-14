"""Known train-only normalization of clip-pooled feature coordinates."""
import numpy as np


def fit_pooled_standardization(pooled):
    x = np.asarray(pooled, np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite TRAIN [N,D] means required")
    mean, std = x.mean(0), x.std(0, ddof=0)
    scale = np.where(std < 1e-6, 1., std)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_count": len(x),
            "constant_or_tiny_coordinates": int((std < 1e-6).sum()), "minimum_std_rule": "STD<1e-6 -> scale1"}


def standardize_sequence(values, length, stats):
    x = np.asarray(values, np.float32); mean, scale = np.asarray(stats["mean"]), np.asarray(stats["scale"])
    if x.ndim != 2 or not 0 < length <= len(x) or mean.shape != (x.shape[1],) or scale.shape != mean.shape or np.any(scale <= 0) or not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Invalid sequence or fitted coordinate statistics")
    output = np.zeros_like(x); output[:length] = ((x[:length].astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite normalized sequence")
    return output
