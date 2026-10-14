"""Independent CPU audit of the registered native scalar-score CRPS suite.

The audit replays selected and final checkpoints with NumPy equations, checks
the frozen point-table and source-role bindings, and reconstructs all outer
forecasts, metrics, quadrature checks, canonical merges, and paired bootstrap
comparisons. It never calls the CRPS trainer or its prediction helpers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import audit_nested_shapes as reference  # noqa: E402


CASES = ("scale", "mixture", "median_mdn", "bern_d6")
SEEDS = (13, 42, 2026)
FOLDS = (0, 1, 2)
PRIMARY = "bern_d6"
REFERENCE_CASES = ("mixture", "median_mdn")
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 5000
QUADRATURE_NODES = (48, 96, 192)
QUADRATURE_TOLERANCE = 1e-4
NUMERIC_TOLERANCE = 1e-5
INTERVAL_LEVELS = (.5, .8, .9, .95)
SOURCE_CSV_SHA256 = reference.CSV_SHA256


class AuditError(RuntimeError):
    """A concise failure with an audit check name."""


class Recorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.findings: list[dict[str, str]] = []

    def must(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            self.findings.append({"check": name, "issue": detail})
            raise AuditError(detail)


def _legendre_rule(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes not in QUADRATURE_NODES:
        raise ValueError(f"Expected registered quadrature order in {QUADRATURE_NODES}")
    x, w = np.polynomial.legendre.leggauss(nodes)
    return x.astype(np.float64), w.astype(np.float64)


def reference_native_hit_probability(params: dict[str, Any], center: np.ndarray,
                                     tolerance: float = float(np.float32(.1))) -> np.ndarray:
    """Mass in the closed hit interval, including censoring atoms at +/-1."""
    center32 = np.asarray(center, dtype=np.float32).reshape(-1)
    if center32.shape != params["anchor"].shape or not np.isfinite(center32).all():
        raise ValueError("Finite float32 event centers matching the forecast rows required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Positive finite event tolerance required")
    low = center32.astype(np.float64) - tolerance
    high = center32.astype(np.float64) + tolerance
    before = reference.reference_cdf(params, low)
    # F(low) would subtract the censored atom at -1, although the inclusive
    # lower end of the event must retain that atom.
    before = np.where(low[:, None] <= -1., 0., before)
    after = reference.reference_cdf(params, high)
    return np.clip(after[:, 0] - before[:, 0], 0., 1.)


def _split_gauss_crps(cdf_function, anchor: np.ndarray, target: np.ndarray,
                      nodes: int = 48) -> np.ndarray:
    """Generic split quadrature core, used for analytic audit fixtures."""
    x, w = _legendre_rule(nodes)
    m = np.asarray(anchor, dtype=np.float64).reshape(-1)
    y = np.asarray(target, dtype=np.float32).astype(np.float64).reshape(-1)
    if not len(m) or y.shape != m.shape or not np.isfinite(m).all() or not np.isfinite(y).all():
        raise ValueError("Finite target and anchor rows of equal nonzero length required")
    if np.any(np.abs(m) > 1.) or np.any(np.abs(y) > 1.):
        raise ValueError("Native CRPS targets and anchors must remain in [-1,1]")
    breaks = np.stack((np.full_like(m, -1.), np.minimum(m, y), np.maximum(m, y), np.full_like(m, 1.)), -1)
    left, right = breaks[:, :-1], breaks[:, 1:]
    center = (left + right) / 2.
    half = (right - left) / 2.
    query = center[:, :, None] + half[:, :, None] * x[None, None, :]
    flat = query.reshape(len(m), -1)
    probability = np.asarray(cdf_function(flat), dtype=np.float64)
    if probability.shape != flat.shape or not np.isfinite(probability).all() \
            or np.any((probability < -1e-12) | (probability > 1. + 1e-12)):
        raise ValueError("Finite bounded sample-by-grid CDF values required")
    event = (flat >= y[:, None]).astype(np.float64)
    integrand = np.square(probability - event).reshape(len(m), 3, nodes)
    score = np.sum(integrand * w[None, None, :] * half[:, :, None], axis=(1, 2))
    if not np.isfinite(score).all() or np.any(score < -1e-14):
        raise ValueError("Invalid independently integrated final-CDF CRPS")
    return np.maximum(score, 0.)


def reference_crps(params: dict[str, Any], target: np.ndarray, nodes: int = 48) -> np.ndarray:
    """Split Gauss-Legendre CRPS integral evaluated only from the NumPy CDF."""
    return _split_gauss_crps(lambda grid: reference.reference_cdf(params, grid),
                             params["anchor"], target, nodes)


def reference_crps_trapezoid_1001(params: dict[str, Any], target: np.ndarray) -> np.ndarray:
    """The older fixed grid, retained as an explicitly approximate readout."""
    grid = np.linspace(-1., 1., 1001, dtype=np.float64)
    y = np.asarray(target, dtype=np.float32).astype(np.float64).reshape(-1)
    values = reference.reference_cdf(params, np.broadcast_to(grid, (len(y), len(grid))))
    integrand = np.square(values - (grid[None, :] >= y[:, None]))
    step = grid[1] - grid[0]
    return step * (integrand[:, 0] / 2.
                   + integrand[:, 1:-1].sum(axis=1)
                   + integrand[:, -1] / 2.)


def _repair_reference_bins(raw: np.ndarray, *, negative_tolerance: float,
                           row_sum_tolerance: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate and repair floating-point edge-subtraction roundoff only."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or not len(raw) or not np.isfinite(raw).all():
        raise AuditError("Independent native-bin probabilities have an invalid shape or value")
    minimum = float(raw.min())
    sum_error = float(np.max(np.abs(raw.sum(axis=1) - 1.)))
    if minimum < -negative_tolerance or sum_error > row_sum_tolerance:
        raise AuditError("Independent native-bin probabilities exceed registered roundoff bounds")
    clipped = np.maximum(raw, 0.)
    sums = clipped.sum(axis=1, keepdims=True)
    if np.any(sums <= 0.):
        raise AuditError("Independent native-bin probabilities have a zero row sum")
    repaired = clipped / sums
    maximum_correction = float(np.max(np.abs(repaired - raw)))
    if maximum_correction > row_sum_tolerance:
        raise AuditError("Independent native-bin repair exceeds registered correction bound")
    return repaired, {
        "negative_entry_tolerance": negative_tolerance,
        "row_sum_tolerance": row_sum_tolerance,
        "minimum_before_repair": minimum,
        "maximum_absolute_row_sum_error_before_repair": sum_error,
        "maximum_absolute_correction": maximum_correction,
        "repaired_rows": int(np.count_nonzero(np.any(repaired != raw, axis=1))),
    }


def _sanitize_reference_bins(params: dict[str, Any], *, negative_tolerance: float,
                             row_sum_tolerance: float, cdf_range_tolerance: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Clamp tiny CDF range roundoff, then repair only tiny negative bins."""
    boundaries = np.asarray(reference.BOUNDARIES, dtype=np.float64)
    cdf_values = reference.reference_cdf(
        params, np.broadcast_to(boundaries[None, :], (len(params["anchor"]), len(boundaries))))
    minimum_cdf, maximum_cdf = float(cdf_values.min()), float(cdf_values.max())
    if minimum_cdf < -cdf_range_tolerance or maximum_cdf > 1. + cdf_range_tolerance:
        raise AuditError("Independent CDF values exceed the registered range roundoff tolerance")
    bounded_cdf = np.clip(cdf_values, 0., 1.)
    cdf_correction = float(np.max(np.abs(bounded_cdf - cdf_values)))
    endpoints = np.concatenate((np.zeros((len(cdf_values), 1)), bounded_cdf,
                                np.ones((len(cdf_values), 1))), axis=1)
    repaired, audit = _repair_reference_bins(np.diff(endpoints, axis=1),
                                              negative_tolerance=negative_tolerance,
                                              row_sum_tolerance=row_sum_tolerance)
    audit["cdf_range_tolerance"] = cdf_range_tolerance
    audit["maximum_CDF_clamp_correction"] = cdf_correction
    return repaired, audit


def _raw_unclipped_metrics(point: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(point, dtype=np.float32).astype(np.float64)
    y = np.asarray(target, dtype=np.float32).astype(np.float64)
    return {"count": int(len(y)), "MAE": float(np.mean(np.abs(raw - y))),
            "Corr": reference._raw_corr(raw, y)}


def _event_truth(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    y = np.asarray(target, dtype=np.float32)
    return (np.abs(c - y) <= np.float32(.1)).astype(np.float64)


def _event_metrics(probability: np.ndarray, hit: np.ndarray, errors: np.ndarray) -> dict[str, Any]:
    """Independent NumPy event metrics matching the registered 15-bin report."""
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(hit, dtype=np.float64)
    e = np.asarray(errors, dtype=np.float64)
    if (p.ndim != 1 or p.shape != y.shape or p.shape != e.shape or not len(p)
            or not np.isfinite(p).all() or np.any((p < 0.) | (p > 1.))
            or not np.isfinite(y).all() or not np.isfinite(e).all()):
        raise ValueError("Invalid independent event metric inputs")
    clipped = np.clip(p, 1e-12, 1. - 1e-12)
    groups = np.minimum((p * 15).astype(np.int64), 14)
    reliability = []
    for group in range(15):
        mask = groups == group
        if mask.any():
            reliability.append({"bin": group, "count": int(mask.sum()),
                                "mean_probability": float(p[mask].mean()),
                                "observed_hit_rate": float(y[mask].mean())})
    order = np.argsort(-p, kind="stable")
    risk = np.cumsum(e[order]) / np.arange(1, len(e) + 1)
    return {
        "hit_count": int(y.sum()), "count": len(y),
        "hit_Brier": float(np.mean(np.square(p - y))),
        "hit_log_score": float(-np.mean(y * np.log(clipped) + (1. - y) * np.log1p(-clipped))),
        "hit_ECE15": float(sum(row["count"] / len(y)
                               * abs(row["mean_probability"] - row["observed_hit_rate"])
                               for row in reliability)),
        "reliability": reliability,
        "risk_AURC": float(risk.mean()),
        "risk_at_coverage": {str(c): float(risk[max(0, math.ceil(len(y) * c) - 1)])
                              for c in (.1, .25, .5, .75, 1.)},
    }


def _make_native_records(ids: list[str], data: dict[str, Any], bins: np.ndarray,
                         raw_hit_probability: np.ndarray, native_hit_probability: np.ndarray,
                         quantiles: np.ndarray, trapz: np.ndarray,
                         scores: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    records = reference._make_records(ids, data, bins, raw_hit_probability, quantiles, trapz)
    point = np.asarray(data["native_point"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    anchor = np.asarray(data["anchor"], dtype=np.float32)
    native_truth = _event_truth(anchor, target)
    raw_truth = _event_truth(point, target)
    for i, row in enumerate(records):
        row["raw_hit_probability"] = row.pop("p_hit")
        row["raw_hit_Brier"] = row.pop("hit_Brier")
        row["raw_hit_log_score"] = row.pop("hit_log_score")
        probability = float(native_hit_probability[i])
        truth = float(native_truth[i])
        clipped = float(np.clip(probability, 1e-12, 1. - 1e-12))
        row["native_hit_probability"] = probability
        row["native_hit_truth"] = int(truth)
        row["native_hit_Brier"] = float((probability - truth) ** 2)
        row["native_hit_log_score"] = float(-truth * math.log(clipped) - (1. - truth) * math.log1p(-clipped))
        row["raw_hit_truth"] = int(raw_truth[i])
        for nodes, values in scores.items():
            row[f"CRPS_{nodes}"] = float(values[i])
    return records


def _compare_native_records(expected: list[dict[str, Any]], actual: list[dict[str, Any]], label: str) -> None:
    """Keep forecast fields tight; allow tiny tail amplification in interval scores."""
    expected_core = [{key: value for key, value in row.items() if key != "interval_scores"}
                     for row in expected]
    actual_core = [{key: value for key, value in row.items() if key != "interval_scores"}
                   for row in actual]
    reference._compare_json(expected_core, actual_core, label + ".forecast", NUMERIC_TOLERANCE)
    expected_scores = [row.get("interval_scores") for row in expected]
    actual_scores = [row.get("interval_scores") for row in actual]
    reference._compare_json(expected_scores, actual_scores, label + ".interval_scores", 2e-5)


def _metric_summary(point: np.ndarray, target: np.ndarray, anchor: np.ndarray, bins: np.ndarray,
                    native_hit_probability: np.ndarray, raw_hit_probability: np.ndarray,
                    quantiles: np.ndarray, trapz: np.ndarray,
                    scores: dict[int, np.ndarray],
                    bin_roundoff: dict[str, Any] | None = None) -> dict[str, Any]:
    point = np.asarray(point, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    anchor = np.asarray(anchor, dtype=np.float32)
    labels = reference._class_targets(target)
    one_hot = np.eye(5, dtype=np.float64)[labels]
    native_truth = _event_truth(anchor, target)
    raw_truth = _event_truth(point, target)
    native_error = np.abs(anchor.astype(np.float64) - target.astype(np.float64))
    raw_error = np.abs(point.astype(np.float64) - target.astype(np.float64))
    native_event = _event_metrics(native_hit_probability, native_truth, native_error)
    raw_event = _event_metrics(raw_hit_probability, raw_truth, raw_error)
    q = np.asarray(quantiles, dtype=np.float64)
    intervals: dict[str, Any] = {}
    interval_scores: dict[str, float] = {}
    for j, nominal in enumerate(INTERVAL_LEVELS):
        lo, hi = q[:, 2*j], q[:, 2*j+1]
        intervals[str(nominal)] = {
            "coverage": float(np.mean((target >= lo) & (target <= hi))),
            "mean_width": float(np.mean(hi - lo)),
        }
        alpha = 1. - nominal
        vals = hi - lo + 2. / alpha * (np.maximum(lo - target, 0.) + np.maximum(target - hi, 0.))
        interval_scores[str(nominal)] = float(vals.mean())
    clipped_native = np.clip(native_hit_probability, 1e-12, 1. - 1e-12)
    clipped_raw = np.clip(raw_hit_probability, 1e-12, 1. - 1e-12)
    bin_nll = -np.log(np.maximum(bins[np.arange(len(labels)), labels], 1e-12))
    native = reference._author_native_metrics(point, target)
    raw_point = _raw_unclipped_metrics(point, target)
    discrepancies: dict[str, Any] = {}
    for lower, upper in ((48, 96), (96, 192), (48, 192)):
        difference = scores[lower] - scores[upper]
        discrepancies[f"{lower}_vs_{upper}"] = {
            "mean_score_difference": float(difference.mean()),
            "mean_absolute_row_difference": float(np.abs(difference).mean()),
            "maximum_absolute_row_difference": float(np.abs(difference).max()),
        }
    max_mean_absolute = max(row["mean_absolute_row_difference"] for row in discrepancies.values())
    result = {
        "native_author_point": native,
        "raw_unclipped_point": raw_point,
        "raw_unclipped_MAE": raw_point["MAE"],
        "raw_unclipped_Corr": raw_point["Corr"],
        "count": int(len(target)),
        "native_hit_count": native_event["hit_count"],
        "native_hit_Brier": native_event["hit_Brier"],
        "native_hit_log_score": native_event["hit_log_score"],
        "native_hit_ECE15": native_event["hit_ECE15"],
        "native_hit_reliability": native_event["reliability"],
        "raw_hit_count": raw_event["hit_count"],
        "raw_hit_Brier": raw_event["hit_Brier"],
        "raw_hit_log_score": raw_event["hit_log_score"],
        "raw_hit_ECE15": raw_event["hit_ECE15"],
        "raw_hit_reliability": raw_event["reliability"],
        "bin_log_score": float(bin_nll.mean()),
        "bin_Brier": float(np.mean(np.square(bins - one_hot).sum(axis=1))),
        "CRPS_trapezoid_1001": float(np.mean(trapz)),
        "CRPS_by_quadrature_nodes": {str(nodes): float(values.mean()) for nodes, values in scores.items()},
        "CRPS_48": float(scores[48].mean()),
        "CRPS_quadrature_discrepancies": discrepancies,
        "CRPS_quadrature_gate": {
            "tolerance": QUADRATURE_TOLERANCE,
            "maximum_mean_absolute_row_difference": float(max_mean_absolute),
            "pass": bool(max_mean_absolute <= QUADRATURE_TOLERANCE),
        },
        "anchor_clipped_count": int(np.count_nonzero(point != anchor)),
        "intervals": intervals,
        "interval_scores": interval_scores,
    }
    if bin_roundoff is not None:
        result["native_bin_probability_roundoff"] = bin_roundoff
    return result


def _shape_select_crps(model_state: dict[str, Any], data: dict[str, Any], stats: dict[str, Any],
                       case: str, nodes: int = 48) -> float:
    raw = reference._linear_head_outputs(model_state, reference._normalize_context(data, stats))
    params = reference.reference_parameters(case, data["anchor"], raw)
    return float(reference_crps(params, data["target"], nodes).mean())


def _replay_history(history: list[dict[str, Any]], fit_count: int,
                    config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(history, list) or not history or len(history) > int(config["max_epochs"]):
        raise AuditError("CRPS history is empty or exceeds the registered epoch budget")
    scores: list[float] = []
    for epoch, row in enumerate(history, 1):
        score = float(row.get("selection_CRPS_48", math.nan))
        fit_score = float(row.get("fit_CRPS_48", math.nan))
        seconds = float(row.get("seconds", math.nan))
        if (row.get("epoch") != epoch or row.get("updates") != math.ceil(fit_count / int(config["batch_size"]))
                or not math.isfinite(score) or not math.isfinite(fit_score) or not math.isfinite(seconds)):
            raise AuditError(f"CRPS history epoch {epoch} has invalid score, timing, order, or update count")
        scores.append(score)
    best, stale, best_epoch = math.inf, 0, 0
    for epoch, score in enumerate(scores, 1):
        if score < best:
            best, stale, best_epoch = score, 0, epoch
        else:
            stale += 1
    by_patience = stale >= int(config["patience"])
    by_budget = len(history) >= int(config["max_epochs"])
    if not (by_patience or by_budget):
        raise AuditError("CRPS history stopped before registered patience or epoch budget")
    if by_patience and not by_budget and stale != int(config["patience"]):
        raise AuditError("CRPS history continues beyond the first registered patience stop")
    return {"best": best, "best_epoch": best_epoch, "stale": stale,
            "updates_per_epoch": math.ceil(fit_count / int(config["batch_size"]))}


def _expected_config(config: dict[str, Any], config_path: Path, plan_path: Path) -> dict[str, Any]:
    expected = dict(config)
    expected["config_sha256"] = reference.sha256_file(config_path)
    expected["plan_sha256"] = reference.sha256_file(plan_path)
    return expected


def _expected_point_binding(point: dict[str, Any], seed: int, fold: int) -> dict[str, Any]:
    stage = point["rows"][(seed, fold)]
    return {
        "suite_sha256": reference.sha256_file(point["suite_path"]),
        "manifest_sha256": point["manifest_sha256"],
        "point_best_checkpoint_sha256": stage["point_best_checkpoint_sha256"],
        "tables": {role: stage["tables"][role]["sha256"] for role in ("shape_fit", "shape_select", "outer")},
        "ids": {role: stage["tables"][role]["ids_sha256"] for role in ("shape_fit", "shape_select", "outer")},
        "source_csv_sha256": SOURCE_CSV_SHA256,
        "nested_manifest_sha256": point["manifest"].get("nested_manifest_sha256"),
        "smoke_only": False,
    }


def _expected_code_hashes(repo_root: Path) -> dict[str, str]:
    names = (
        "native_crps_training.py", "cdf_crps.py", "nested_shape_training.py",
        "shape_controls.py", "nested_shape_points.py", "anchored_bernstein.py",
        "anchor_cdf.py", "anchor_calibration.py", "native_training.py", "native_metrics.py",
    )
    return {name: reference.sha256_file(repo_root / "jevaffect" / name) for name in names}


def _initial_encoder_sha256(seed: int, case: str) -> str:
    """Recreate just the common encoder initialization from the run seed."""
    import torch
    from jevaffect import shape_controls

    torch.manual_seed(seed)
    model = shape_controls.ShapeHead(case, feature_dim=257, hidden=32, dropout=.1)
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.context.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _valid_saved_rng_state(checkpoint: dict[str, Any]) -> bool:
    """Check the persisted CPU and loader RNG states without restoring them."""
    import torch

    tensor_states = (checkpoint.get("torch_rng"), checkpoint.get("loader_rng"))
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu"
           or value.dtype != torch.uint8 or value.ndim != 1 or not value.numel()
           for value in tensor_states):
        return False
    numpy_state = checkpoint.get("numpy_rng")
    if (not isinstance(numpy_state, tuple) or len(numpy_state) != 5
            or numpy_state[0] != "MT19937" or not isinstance(numpy_state[1], np.ndarray)
            or numpy_state[1].shape != (624,) or numpy_state[1].dtype != np.uint32
            or not isinstance(numpy_state[2], (int, np.integer))
            or not isinstance(numpy_state[3], (int, np.integer))):
        return False
    python_state = checkpoint.get("python_rng")
    if (not isinstance(python_state, tuple) or len(python_state) != 3
            or not isinstance(python_state[0], int) or not isinstance(python_state[1], tuple)
            or len(python_state[1]) != 625):
        return False
    optimizer = checkpoint.get("optimizer")
    return isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict) \
        and isinstance(optimizer.get("param_groups"), list)


def _audit_fit(suite_row: dict[str, Any], *, case: str, seed: int, fold: int,
               config: dict[str, Any], expected_config: dict[str, Any],
               point: dict[str, Any], repo_root: Path, rec: Recorder
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label = f"{case} seed {seed} fold {fold}"
    rec.must(label + " result identity",
             suite_row.get("case") == case and suite_row.get("seed") == seed and suite_row.get("fold") == fold
             and suite_row.get("complete") is True and suite_row.get("smoke_only") is False
             and suite_row.get("official_VALID_used") is False and suite_row.get("official_TEST_used") is False,
             f"Formal CRPS result is incomplete, duplicated, or has invalid identity for {label}")
    model_root = reference.resolve_path(suite_row["root"], repo_root)
    expected_root = reference.resolve_path(config["run_root"], repo_root) / "formal" / case / f"seed_{seed}" / f"fold_{fold}"
    rec.must(label + " artifact root", model_root.resolve() == expected_root.resolve(),
             f"CRPS result root differs from the registered location for {label}")
    history_path, identity_path = model_root / "history.json", model_root / "identity.json"
    best_path, last_path = model_root / "best.pt", model_root / "last.pt"
    prediction_path, result_path = model_root / "outer_predictions.json", model_root / "result.json"
    for path, hash_key in ((best_path, "best_sha256"), (last_path, "last_sha256"),
                           (history_path, "history_sha256"), (prediction_path, "predictions_sha256")):
        rec.must(label + " artifact hash " + path.name,
                 path.is_file() and suite_row.get(hash_key) == reference.sha256_file(path),
                 f"Saved {path.name} is missing or differs from the suite hash for {label}")

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected_binding = _expected_point_binding(point, seed, fold)
    rec.must(label + " config/plan/point binding",
             identity.get("config") == expected_config and identity.get("binding") == expected_binding
             and identity.get("config_sha256") == expected_config["config_sha256"]
             and identity.get("plan_sha256") == expected_config["plan_sha256"]
             and suite_row.get("config_sha256") == expected_config["config_sha256"]
             and suite_row.get("plan_sha256") == expected_config["plan_sha256"]
             and suite_row.get("point_binding") == expected_binding,
             f"Identity for {label} is not bound to the registered config, plan, and frozen point artifacts")
    expected_encoder = _initial_encoder_sha256(seed, case)
    rec.must(label + " deterministic encoder initialization",
             identity.get("initial_encoder_sha256") == expected_encoder
             and suite_row.get("initial_encoder_sha256") == expected_encoder,
             f"Saved initial common encoder bytes differ from independent seed {seed} reconstruction")
    tables = {role: point["tables"][(seed, fold, role)] for role in ("shape_fit", "shape_select", "outer")}
    rec.must(label + " fit/select ID and selection semantics",
             identity.get("fit_ids") == tables["shape_fit"]["ids"]
             and identity.get("select_ids") == tables["shape_select"]["ids"]
             and identity.get("outer_used_for_selection") is False
             and identity.get("selection") == "first minimum shape_select CRPS_48"
             and identity.get("training_objective") == "final_censored_CDF_CRPS_Gauss_Legendre_48",
             f"Identity for {label} includes wrong role IDs or selection/objective semantics")
    expected_code = _expected_code_hashes(repo_root)
    rec.must(label + " frozen implementation hashes", identity.get("implementation_sha256") == expected_code
             and suite_row.get("implementation_sha256") == expected_code,
             f"Bound training or CDF implementation bytes changed for {label}")
    stats, _ = reference._normalization_for_fit(tables["shape_fit"])
    rec.must(label + " shape_fit-only normalization", identity.get("normalization") == stats,
             f"Normalization for {label} differs from independent shape_fit-only recomputation")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    stopping = _replay_history(history, len(tables["shape_fit"]["ids"]), config)
    rec.must(label + " first-minimum and stop budget",
             suite_row.get("epochs") == len(history)
             and suite_row.get("best_epoch") == stopping["best_epoch"]
             and suite_row.get("updates") == len(history) * stopping["updates_per_epoch"]
             and suite_row.get("best_selection_CRPS_48") == stopping["best"],
             f"Saved fit {label} did not retain the first shape_select CRPS minimum or registered stop budget")

    best_state, last_state = reference._torch_load(best_path), reference._torch_load(last_path)
    best_model, last_model = best_state.get("model", {}), last_state.get("model", {})
    output_dim = reference._expected_output_dim(case)
    expected_shapes = {"context.0.weight": (32, 257), "context.0.bias": (32,),
                       "head.weight": (output_dim, 32), "head.bias": (output_dim,)}
    models_valid = all(set(model) == set(expected_shapes)
                       and all(tuple(model[name].shape) == shape and bool(model[name].isfinite().all())
                               for name, shape in expected_shapes.items())
                       for model in (best_model, last_model))
    expected_parameters = 32 * 258 + output_dim * 33
    rec.must(label + " best/last tensor schemas", models_valid
             and suite_row.get("parameters") == expected_parameters
             and sum(tensor.numel() for tensor in best_model.values()) == expected_parameters,
             f"Best or last checkpoint for {label} has invalid tensor schema or parameter count")
    rec.must(label + " checkpoint identity and history state",
             best_state.get("identity") == identity and best_state.get("epoch") == stopping["best_epoch"]
             and abs(float(best_state.get("selection_CRPS_48", math.nan)) - stopping["best"]) <= 1e-12
             and last_state.get("identity") == identity and last_state.get("history") == history
             and last_state.get("epoch") == len(history) and last_state.get("stale") == stopping["stale"]
             and abs(float(last_state.get("best", math.nan)) - stopping["best"]) <= 1e-12
             and _valid_saved_rng_state(last_state),
             f"Saved best/last state or resumable RNG state for {label} differs from the first-minimum history")

    select = tables["shape_select"]
    selected_score = _shape_select_crps(best_model, select, stats, case, 48)
    last_score = _shape_select_crps(last_model, select, stats, case, 48)
    rec.must(label + " independent selected shape_select CRPS",
             abs(selected_score - stopping["best"]) <= NUMERIC_TOLERANCE,
             f"Independent best-checkpoint shape_select CRPS {selected_score:.12g} differs from history {stopping['best']:.12g}")
    rec.must(label + " independent last-checkpoint shape_select CRPS",
             abs(last_score - float(history[-1]["selection_CRPS_48"])) <= NUMERIC_TOLERANCE,
             f"Independent last-checkpoint shape_select CRPS {last_score:.12g} differs from final history score")

    outer = tables["outer"]
    raw_best = reference._linear_head_outputs(best_model, reference._normalize_context(outer, stats))
    params = reference.reference_parameters(case, outer["anchor"], raw_best)
    bins, bin_roundoff = _sanitize_reference_bins(
        params,
        negative_tolerance=float(config["bin_negative_roundoff_tolerance"]),
        row_sum_tolerance=float(config["bin_sum_roundoff_tolerance"]),
        cdf_range_tolerance=float(config["cdf_range_roundoff_tolerance"]),
    )
    if (not np.isfinite(bins).all() or np.any(bins < -1e-12)
            or not np.allclose(bins.sum(axis=1), 1., atol=1e-10, rtol=0.)):
        raise AuditError(f"Independent bins are invalid for {label}")
    quantiles = reference.reference_quantiles(params)
    quantile_order = np.asarray([6, 4, 2, 0, 8, 1, 3, 5, 7], dtype=np.int64)
    if (not np.isfinite(quantiles).all() or np.any(quantiles < -1.) or np.any(quantiles > 1.)
            or np.any(np.diff(quantiles[:, quantile_order], axis=1) < -1e-12)):
        raise AuditError(f"Independent quantiles violate native bounds/order for {label}")
    if case != "mdn" and not np.allclose(quantiles[:, 8], outer["anchor"], atol=1e-12, rtol=0.):
        raise AuditError(f"Fixed CDF median differs from the clipped point anchor for {label}")
    raw_hit = reference_native_hit_probability(params, outer["native_point"])
    native_hit = reference_native_hit_probability(params, outer["anchor"])
    score_rows = {nodes: reference_crps(params, outer["target"], nodes) for nodes in QUADRATURE_NODES}
    trapz = reference_crps_trapezoid_1001(params, outer["target"])
    quadrature_ok = (
        float(np.abs(score_rows[48] - score_rows[192]).mean()) <= QUADRATURE_TOLERANCE
        and max(float(np.abs(score_rows[a] - score_rows[b]).mean())
                for a, b in ((48, 96), (96, 192), (48, 192))) <= QUADRATURE_TOLERANCE
    )
    rec.checks.append({"name": label + " 48/96/192 outer numerical accuracy", "passed": quadrature_ok})
    if not quadrature_ok:
        rec.findings.append({"check": label + " 48/96/192 outer numerical accuracy",
                             "issue": f"Independent outer quadrature discrepancy exceeds {QUADRATURE_TOLERANCE} for {label}"})
    records = _make_native_records(outer["ids"], outer, bins, raw_hit, native_hit,
                                   quantiles, trapz, score_rows)
    saved = json.loads(prediction_path.read_text(encoding="utf-8"))
    rec.must(label + " outer ID/order", [row.get("id") for row in saved] == outer["ids"],
             f"Saved outer forecasts for {label} are not in frozen canonical fold order")
    _compare_native_records(records, saved, f"{label}.outer_records")
    computed = _metric_summary(outer["native_point"], outer["target"], outer["anchor"], bins,
                               native_hit, raw_hit, quantiles, trapz, score_rows, bin_roundoff)
    producer_metrics = suite_row.get("outer") or {}
    producer_roundoff = producer_metrics.get("native_bin_probability_roundoff", {})
    strict_roundoff_fields = (
        "negative_entry_tolerance", "row_sum_tolerance", "cdf_range_tolerance",
    )
    roundoff_metadata_ok = all(producer_roundoff.get(name) == bin_roundoff[name]
                               for name in strict_roundoff_fields)
    roundoff_differences = {}
    try:
        repaired_rows = int(producer_roundoff["repaired_rows"])
        roundoff_metadata_ok &= 0 <= repaired_rows <= len(outer["ids"])
    except (KeyError, TypeError, ValueError):
        roundoff_metadata_ok = False
    numeric_roundoff_tolerances = {
        # This field is an actual bin probability, so NumPy/Torch forward CDF
        # comparison uses the normal forecast tolerance, not a roundoff bound.
        "minimum_before_repair": NUMERIC_TOLERANCE,
        "maximum_absolute_row_sum_error_before_repair": 1e-12,
        "maximum_absolute_correction": 1e-12,
        "maximum_CDF_clamp_correction": 1e-12,
    }
    for name, comparison_tolerance in numeric_roundoff_tolerances.items():
        try:
            difference = abs(float(producer_roundoff[name]) - float(bin_roundoff[name]))
            roundoff_differences[name] = {
                "producer": float(producer_roundoff[name]),
                "independent": float(bin_roundoff[name]),
                "absolute_difference": difference,
            }
            roundoff_metadata_ok &= difference <= comparison_tolerance
        except (KeyError, TypeError, ValueError):
            roundoff_metadata_ok = False
    try:
        producer_roundoff_values = {
            name: float(producer_roundoff[name]) for name in numeric_roundoff_tolerances
        }
        roundoff_metadata_ok &= (
            math.isfinite(producer_roundoff_values["minimum_before_repair"])
            and producer_roundoff_values["minimum_before_repair"] >= -float(config["bin_negative_roundoff_tolerance"])
            and producer_roundoff_values["maximum_absolute_row_sum_error_before_repair"]
            <= float(config["bin_sum_roundoff_tolerance"])
            and producer_roundoff_values["maximum_absolute_correction"]
            <= float(config["bin_sum_roundoff_tolerance"])
            and producer_roundoff_values["maximum_CDF_clamp_correction"]
            <= float(config["cdf_range_roundoff_tolerance"])
        )
    except (KeyError, TypeError, ValueError):
        roundoff_metadata_ok = False
    rec.must(label + " native bin roundoff metadata", roundoff_metadata_ok,
             f"Saved bin correction metadata differs from the independently measured CDF/bin roundoff for {label}: "
             + json.dumps({"producer": producer_roundoff, "independent": bin_roundoff,
                           "differences": roundoff_differences}, sort_keys=True))
    computed_compare = json.loads(json.dumps(computed))
    producer_compare = json.loads(json.dumps(producer_metrics))
    computed_compare["native_bin_probability_roundoff"].pop("repaired_rows", None)
    producer_compare["native_bin_probability_roundoff"].pop("repaired_rows", None)
    reference._compare_json(computed_compare, producer_compare,
                            f"{label}.outer_metrics", NUMERIC_TOLERANCE)
    result_on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    reference._compare_json(suite_row, result_on_disk, f"{label}.result_file", NUMERIC_TOLERANCE)

    # Check the final checkpoint also produces finite, bounded forecast laws;
    # only the selected checkpoint contributes to the held-source readout.
    raw_last = reference._linear_head_outputs(last_model, reference._normalize_context(outer, stats))
    last_params = reference.reference_parameters(case, outer["anchor"], raw_last)
    last_bins, _ = _sanitize_reference_bins(
        last_params,
        negative_tolerance=float(config["bin_negative_roundoff_tolerance"]),
        row_sum_tolerance=float(config["bin_sum_roundoff_tolerance"]),
        cdf_range_tolerance=float(config["cdf_range_roundoff_tolerance"]),
    )
    last_crps = reference_crps(last_params, outer["target"], 48)
    rec.must(label + " last-checkpoint outer forecast sanity",
             np.isfinite(last_bins).all() and np.all(last_bins >= -1e-12)
             and np.allclose(last_bins.sum(axis=1), 1., atol=1e-10, rtol=0.)
             and np.isfinite(last_crps).all(),
             f"Last checkpoint produces invalid forecast values on the held source rows for {label}")
    fit_replay = {
        "case": case, "seed": seed, "fold": fold,
        "best_checkpoint_sha256": suite_row["best_sha256"],
        "last_checkpoint_sha256": suite_row["last_sha256"],
        "history_sha256": suite_row["history_sha256"],
        "outer_predictions_sha256": suite_row["predictions_sha256"],
        "best_epoch": stopping["best_epoch"], "epochs": len(history),
        "shape_select_CRPS_48_best_replayed": selected_score,
        "shape_select_CRPS_48_last_replayed": last_score,
        "outer_CRPS_quadrature_discrepancies": computed["CRPS_quadrature_discrepancies"],
        "outer_quadrature_verified": bool(computed["CRPS_quadrature_gate"]["pass"]),
        "native_bin_probability_roundoff": {
            **bin_roundoff,
            "independent_repaired_rows": bin_roundoff["repaired_rows"],
            "producer_repaired_rows": repaired_rows,
            "producer_minimum_before_repair": producer_roundoff.get("minimum_before_repair"),
        },
    }
    return records, fit_replay


def _row_metrics(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    names = ("CRPS_48", "native_hit_Brier", "native_hit_log_score", "raw_hit_Brier",
             "raw_hit_log_score", "bin_NLL", "bin_Brier", "CRPS_trapezoid_1001")
    return {name: np.asarray([float(row[name]) for row in records], dtype=np.float64) for name in names}


def _merged_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bins = np.asarray([row["bin_probabilities"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target"] for row in rows], dtype=np.float32)
    point = np.asarray([row["native_point"] for row in rows], dtype=np.float32)
    anchor = np.asarray([row["anchor"] for row in rows], dtype=np.float32)
    native_hit = np.asarray([row["native_hit_probability"] for row in rows], dtype=np.float64)
    raw_hit = np.asarray([row["raw_hit_probability"] for row in rows], dtype=np.float64)
    q = np.asarray([[edge for level in INTERVAL_LEVELS for edge in row["intervals"][str(level)]]
                    for row in rows], dtype=np.float64)
    trapz = np.asarray([row["CRPS_trapezoid_1001"] for row in rows], dtype=np.float64)
    score_rows = {nodes: np.asarray([row[f"CRPS_{nodes}"] for row in rows], dtype=np.float64)
                  for nodes in QUADRATURE_NODES}
    result = _metric_summary(point, target, anchor, bins, native_hit, raw_hit, q, trapz, score_rows)
    result["anchor_clipping_frequency"] = float(np.mean(point != anchor))
    return result


def _same_seed_frozen_point(merged: dict[str, dict[int, list[dict[str, Any]]]]) -> bool:
    """Point, target, and anchor bind across cases within a seed only."""
    for seed in SEEDS:
        baseline = merged[PRIMARY][seed]
        for case in CASES:
            current = merged[case][seed]
            if len(current) != len(baseline):
                return False
            if not all(a["id"] == b["id"] and a["native_point"] == b["native_point"]
                       and a["target"] == b["target"] and a["anchor"] == b["anchor"]
                       for a, b in zip(baseline, current)):
                return False
    return True


def _paired_source_bootstrap(merged: dict[str, dict[int, list[dict[str, Any]]]],
                             source_by_id: dict[str, str], source_names: list[str]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(len(source_names), size=(BOOTSTRAP_REPLICATES, len(source_names)), replace=True)
    draws = np.stack([np.bincount(row, minlength=len(source_names)) for row in sampled]).astype(np.float64)
    source_index = {name: i for i, name in enumerate(source_names)}
    per_case_seed: dict[str, dict[int, dict[str, np.ndarray]]] = {case: {} for case in CASES}
    source_counts: np.ndarray | None = None
    interval_sums: dict[str, dict[int, dict[str, dict[str, np.ndarray]]]] = {case: {} for case in CASES}
    for case in CASES:
        for seed in SEEDS:
            rows = merged[case][seed]
            metrics = _row_metrics(rows)
            sums = {name: np.zeros(len(source_names), dtype=np.float64) for name in metrics}
            counts = np.zeros(len(source_names), dtype=np.float64)
            i_sums = {str(level): np.zeros(len(source_names), dtype=np.float64) for level in INTERVAL_LEVELS}
            for i, record in enumerate(rows):
                source_i = source_index[source_by_id[record["id"]]]
                counts[source_i] += 1.
                for name, values in metrics.items():
                    sums[name][source_i] += values[i]
                for level in INTERVAL_LEVELS:
                    lo, hi = record["intervals"][str(level)]
                    target = float(record["target"])
                    alpha = 1. - level
                    score = hi - lo + 2. / alpha * (max(lo - target, 0.) + max(target - hi, 0.))
                    i_sums[str(level)][source_i] += score
            if source_counts is None:
                source_counts = counts
            elif not np.array_equal(counts, source_counts):
                raise AuditError("Canonical merged source counts differ across case/seed bootstrap cohorts")
            per_case_seed[case][seed] = {**sums, "_count": counts}
            interval_sums[case][seed] = i_sums

    assert source_counts is not None
    output: dict[str, Any] = {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "sampled_source_count_per_draw": len(source_names),
        "source_videos": len(source_names),
        "paired_source_draws_shared_across_cases_and_seeds": True,
        "delta_definition": "control metric minus bern_d6 metric; positive favors bern_d6 for proper scores",
        "comparisons": {},
    }
    metric_names = tuple(name for name in next(iter(per_case_seed[PRIMARY].values())).keys()
                         if name != "_count")
    for control in (case for case in CASES if case != PRIMARY):
        output["comparisons"][control] = {}
        for metric in metric_names:
            output["comparisons"][control][metric] = {}
            for seed_key in (*map(str, SEEDS), "pooled"):
                if seed_key == "pooled":
                    control_sum = sum(per_case_seed[control][seed][metric] for seed in SEEDS)
                    primary_sum = sum(per_case_seed[PRIMARY][seed][metric] for seed in SEEDS)
                    count = source_counts * len(SEEDS)
                    control_all = np.concatenate([_row_metrics(merged[control][seed])[metric] for seed in SEEDS])
                    primary_all = np.concatenate([_row_metrics(merged[PRIMARY][seed])[metric] for seed in SEEDS])
                else:
                    seed = int(seed_key)
                    control_sum = per_case_seed[control][seed][metric]
                    primary_sum = per_case_seed[PRIMARY][seed][metric]
                    count = source_counts
                    control_all = _row_metrics(merged[control][seed])[metric]
                    primary_all = _row_metrics(merged[PRIMARY][seed])[metric]
                delta = (draws @ (control_sum - primary_sum)) / (draws @ count)
                lo, hi = np.quantile(delta, [.025, .975])
                output["comparisons"][control][metric][seed_key] = {
                    "primary_mean": float(primary_all.mean()),
                    "control_mean": float(control_all.mean()),
                    "observed_delta_control_minus_primary": float(control_all.mean() - primary_all.mean()),
                    "paired_source_bootstrap_95_ci": [float(lo), float(hi)],
                    "bootstrap_probability_primary_better": float(np.mean(delta > 0.)),
                    "primary_better_direction": bool(control_all.mean() > primary_all.mean()),
                }
        output["comparisons"][control]["interval_scores"] = {"0.9": {}}
        for seed_key in (*map(str, SEEDS), "pooled"):
            level = "0.9"
            if seed_key == "pooled":
                control_sum = sum(interval_sums[control][seed][level] for seed in SEEDS)
                primary_sum = sum(interval_sums[PRIMARY][seed][level] for seed in SEEDS)
                count = source_counts * len(SEEDS)
                control_all = np.concatenate([np.asarray([r["interval_scores"][level] for r in merged[control][seed]]) for seed in SEEDS])
                primary_all = np.concatenate([np.asarray([r["interval_scores"][level] for r in merged[PRIMARY][seed]]) for seed in SEEDS])
            else:
                seed = int(seed_key)
                control_sum = interval_sums[control][seed][level]
                primary_sum = interval_sums[PRIMARY][seed][level]
                count = source_counts
                control_all = np.asarray([r["interval_scores"][level] for r in merged[control][seed]])
                primary_all = np.asarray([r["interval_scores"][level] for r in merged[PRIMARY][seed]])
            delta = (draws @ (control_sum - primary_sum)) / (draws @ count)
            lo, hi = np.quantile(delta, [.025, .975])
            output["comparisons"][control]["interval_scores"][level][seed_key] = {
                "primary_mean": float(primary_all.mean()),
                "control_mean": float(control_all.mean()),
                "observed_delta_control_minus_primary": float(control_all.mean() - primary_all.mean()),
                "paired_source_bootstrap_95_ci": [float(lo), float(hi)],
                "primary_better_direction": bool(control_all.mean() > primary_all.mean()),
            }
    return output


def _promotion_gate(bootstrap: dict[str, Any], *, quadrature_verified: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add("independent quadrature verified for all 36 selected outer forecasts", quadrature_verified,
        "Every independent mean absolute 48/96/192 score difference is within tolerance"
        if quadrature_verified else "At least one selected outer forecast failed the registered quadrature tolerance")

    required = ("CRPS_48", "native_hit_Brier", "native_hit_log_score")
    for control in REFERENCE_CASES:
        for metric in required:
            entries = bootstrap["comparisons"][control][metric]
            all_seed_directions = all(entries[str(seed)]["primary_better_direction"] for seed in SEEDS)
            pooled_ci = entries["pooled"]["paired_source_bootstrap_95_ci"]
            add(f"{metric} versus {control}", all_seed_directions and pooled_ci[0] > 0.,
                f"all three seed directions favor bern_d6={all_seed_directions}; pooled 95% CI lower bound > 0={pooled_ci[0] > 0.}")
        entry = bootstrap["comparisons"][control]["interval_scores"]["0.9"]["pooled"]
        add(f"90% interval score versus {control}", entry["observed_delta_control_minus_primary"] >= 0.,
            "No positive pooled 90% interval-score deterioration" if entry["observed_delta_control_minus_primary"] >= 0.
            else "bernoulli degree-6 has a positive pooled 90% interval-score deterioration")
    return {"passed": all(item["passed"] for item in checks), "checks": checks,
            "interpretation": "Any failed required comparison rejects promotion; other cases/readouts are supportive."}


def audit(suite_path: Path, point_suite_path: Path, config_path: Path, repo_root: Path) -> dict[str, Any]:
    rec = Recorder()
    report: dict[str, Any] = {
        "audit": "independent-cpu-native-scalar-crps-audit-v1",
        "suite": str(suite_path), "point_suite": str(point_suite_path), "registered_config": str(config_path),
        "checks": rec.checks, "findings": rec.findings,
        "quadrature_tolerance": QUADRATURE_TOLERANCE,
        "complete": False, "promotion_gate_passed": False,
    }
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_hash = reference.sha256_file(config_path)
        plan_path = reference.resolve_path(config["plan"], repo_root)
        plan_hash = reference.sha256_file(plan_path)
        expected = _expected_config(config, config_path, plan_path)
        rec.must("registered CRPS config", config.get("dataset") == "simsv2"
                 and config.get("point_suite") == "runs/native_nested_shape_points_v1/suite.json"
                 and config.get("seeds") == list(SEEDS) and config.get("cases") == list(CASES)
                 and config.get("device") == "cpu" and config.get("hidden") == 32
                 and config.get("dropout") == .1 and config.get("batch_size") == 64
                 and config.get("torch_threads") == 4 and config.get("learning_rate") == .001
                 and config.get("weight_decay") == .0001 and config.get("gradient_norm") == 1.
                 and config.get("max_epochs") == 100 and config.get("patience") == 12
                 and config.get("quadrature_nodes") == 48 and config.get("readout_nodes") == list(QUADRATURE_NODES)
                 and config.get("quadrature_node_tolerance") == QUADRATURE_TOLERANCE
                 and config.get("bin_negative_roundoff_tolerance") == 1e-12
                 and config.get("bin_sum_roundoff_tolerance") == 1e-10
                 and config.get("cdf_range_roundoff_tolerance") == 1e-12
                 and config.get("primary_case") == PRIMARY
                 and config.get("selection") == "first minimum shape_select CRPS_48"
                 and config.get("official_VALID_used") is False and config.get("official_TEST_used") is False,
                 "Registered config differs from the frozen CPU/seed/case/CRPS training contract")
        plan_flat = " ".join(plan_path.read_text(encoding="utf-8").lower().split())
        rec.must("registered protocol statements",
                 "5000 paired129-source bootstrap draws,seed20261013" in plan_flat
                 and "first minimum shape_select crps" in plan_flat
                 and "48/96/192" in plan_flat
                 and "no discrete support" in plan_flat,
                 "Study plan does not retain the frozen objective, no-support, numerical, and bootstrap rules")

        point = reference.audit_point_tables(point_suite_path, repo_root, rec)
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        suite_identity = suite.get("suite_identity", {})
        rec.must("formal suite markers",
                 suite.get("formal_fit_count") == 36
                 and suite.get("expected_formal_fit_count") == 36
                 and suite.get("official_VALID_used") is False and suite.get("official_TEST_used") is False
                 and suite_identity.get("config_sha256") == config_hash
                 and suite_identity.get("plan_sha256") == plan_hash
                 and suite_identity.get("point_suite_sha256") == reference.sha256_file(point_suite_path)
                 and suite_identity.get("smoke_only") is False
                 and suite_identity.get("official_VALID_used") is False
                 and suite_identity.get("official_TEST_used") is False,
                 "CRPS suite is incomplete, smoke-only, or not bound to registered inputs")
        rows = suite.get("results", [])
        expected_keys = {(case, seed, fold) for case in CASES for seed in SEEDS for fold in FOLDS}
        actual_keys = {(row.get("case"), row.get("seed"), row.get("fold")) for row in rows}
        rec.must("36-fit formal matrix", len(rows) == 36 and actual_keys == expected_keys and len(actual_keys) == len(rows),
                 f"CRPS suite has {len(rows)} unique fits; exactly 36 registered case/seed/fold fits required")
        producer_quadrature_gate = all(row.get("quadrature_gate_pass") is True for row in rows)
        expected_suite_complete = all(row.get("complete") is True and row.get("smoke_only") is False
                                      and row.get("quadrature_gate_pass") is True for row in rows)
        rec.must("suite completion and quadrature aggregate markers",
                 suite.get("quadrature_gate_pass") is producer_quadrature_gate
                 and suite.get("suite_complete") is expected_suite_complete,
                 "Suite-level completion/quadrature flags do not match the formal fit results")
        by_key = {(row["case"], int(row["seed"]), int(row["fold"])): row for row in rows}
        records_by_fit: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        fit_replays: list[dict[str, Any]] = []
        for case, seed, fold in sorted(expected_keys):
            key = (case, seed, fold)
            try:
                fit_records, fit_replay = _audit_fit(
                    by_key[key], case=case, seed=seed, fold=fold,
                    config=config, expected_config=expected,
                    point=point, repo_root=repo_root, rec=rec)
                records_by_fit[key] = fit_records
                fit_replays.append(fit_replay)
            except (AuditError, AssertionError, KeyError, OSError, ValueError, TypeError, IndexError) as exc:
                if not any(item.get("issue") == str(exc) for item in rec.findings):
                    rec.findings.append({"check": f"{case} seed {seed} fold {fold} audit", "issue": str(exc)})
                records_by_fit.pop(key, None)

        # Identical encoder bytes are expected across the four cases in a
        # seed/fold; frozen point predictions are expected within each seed,
        # while different seeds may train different point teachers.
        encoder_identity_ok = True
        for seed in SEEDS:
            for fold in FOLDS:
                values = {by_key[(case, seed, fold)].get("initial_encoder_sha256") for case in CASES}
                encoder_identity_ok &= len(values) == 1 and None not in values
        rec.must("common encoder initialization per seed/fold", encoder_identity_ok,
                 "CRPS cases for a seed/fold do not share identical common encoder bytes")

        canonical_ids = point["canonical_ids"]
        source_by_id = {sample_id: reference.source_of(sample_id) for sample_id in canonical_ids}
        merged: dict[str, dict[int, list[dict[str, Any]]]] = {case: {} for case in CASES}
        for case in CASES:
            for seed in SEEDS:
                by_id: dict[str, dict[str, Any]] = {}
                for fold in FOLDS:
                    for row in records_by_fit.get((case, seed, fold), []):
                        if row["id"] in by_id:
                            raise AuditError(f"Duplicate outer row for {case}, seed {seed}: {row['id']}")
                        by_id[row["id"]] = row
                rec.must(f"canonical merge {case}/{seed}", set(by_id) == set(canonical_ids)
                         and len(by_id) == len(canonical_ids) == 2722,
                         f"Merged {case}, seed {seed} predictions do not cover canonical TRAIN IDs once")
                merged[case][seed] = [by_id[sample_id] for sample_id in canonical_ids]
                rec.must(f"129 source merge {case}/{seed}",
                         len({source_by_id[row["id"]] for row in merged[case][seed]}) == 129,
                         f"Merged {case}, seed {seed} does not cover exactly 129 source videos")
        rec.must("within-seed point invariance", _same_seed_frozen_point(merged),
                 "A case changed the frozen raw point, target, or anchor within the same seed")

        bootstrap = _paired_source_bootstrap(merged, source_by_id, point["source_ids"])
        quadrature_verified = (len(fit_replays) == 36
                               and all(row["outer_quadrature_verified"] for row in fit_replays))
        gate = _promotion_gate(bootstrap, quadrature_verified=quadrature_verified)
        canonical_id_order_sha256 = hashlib.sha256(
            json.dumps(canonical_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        report.update({
            "audited_fits": len(records_by_fit),
            "audited_outer_rows": sum(len(rows_for_fit) for rows_for_fit in records_by_fit.values()),
            "canonical_train_ids": len(canonical_ids),
            "source_videos": len(point["source_ids"]),
            "per_fit_replays": fit_replays,
            "quadrature_verified": quadrature_verified,
            "canonical_id_order_sha256": canonical_id_order_sha256,
            "per_case_seed_metrics": {
                case: {str(seed): _merged_summary(merged[case][seed]) for seed in SEEDS} for case in CASES
            },
            "merged_canonical_outputs": {
                "case_seed_merges": len(CASES) * len(SEEDS),
                "rows_per_merge": len(canonical_ids),
                "total_merged_rows": len(CASES) * len(SEEDS) * len(canonical_ids),
                "source_videos_per_merge": len(point["source_ids"]),
                "outer_source_once_per_merge": True,
                "canonical_id_order_sha256": canonical_id_order_sha256,
            },
            "paired_source_bootstrap": bootstrap,
            "promotion_gate": gate,
            "promotion_gate_passed": gate["passed"],
            "checks_passed": sum(bool(check["passed"]) for check in rec.checks),
            "checks_failed": sum(not bool(check["passed"]) for check in rec.checks),
        })
        report["complete"] = not rec.findings and len(records_by_fit) == 36
        report["recommendation"] = ("retain the registered CRPS evidence without promotion" if not gate["passed"]
                                    else "registered empirical gate passed; this does not establish originality")
    except (AuditError, AssertionError, KeyError, OSError, ValueError, TypeError, IndexError,
            json.JSONDecodeError) as exc:
        if not any(item.get("issue") == str(exc) for item in rec.findings):
            rec.findings.append({"check": "audit setup", "issue": str(exc)})
        report["error"] = str(exc)
    report["checks_passed"] = sum(bool(check["passed"]) for check in rec.checks)
    report["checks_failed"] = sum(not bool(check["passed"]) for check in rec.checks)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--suite", type=Path, help="Formal native CRPS suite.json")
    parser.add_argument("--point-suite", type=Path, help="Frozen nine-fit point suite.json")
    parser.add_argument("--config", type=Path, help="Registered native CRPS config JSON")
    parser.add_argument("--report", type=Path, help="Output audit JSON")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    suite_path = reference.resolve_path(args.suite or "runs/native_cdf_crps_v1/suite.json", repo_root)
    point_suite = reference.resolve_path(args.point_suite or "runs/native_nested_shape_points_v1/suite.json", repo_root)
    config = reference.resolve_path(args.config or "configs/native_cdf_crps_v1.json", repo_root)
    report_path = reference.resolve_path(args.report or "docs/native_cdf_crps_audit_20261013.json", repo_root)
    result = audit(suite_path, point_suite, config, repo_root)
    result["report_path"] = str(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result.get("complete", False),
                      "audited_fits": result.get("audited_fits", 0),
                      "audited_outer_rows": result.get("audited_outer_rows", 0),
                      "promotion_gate_passed": result.get("promotion_gate_passed", False),
                      "checks_failed": result.get("checks_failed", 0),
                      "findings": result.get("findings", [])[:20],
                      "error": result.get("error"), "report": str(report_path)},
                     indent=2, allow_nan=False))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
