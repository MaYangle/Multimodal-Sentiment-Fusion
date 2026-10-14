"""Independent CPU audit of the registered nested native shape suite.

This script does not use the shape producer's CDF, metric, prediction, or
completion helpers as audit evidence. It checks source/table/checkpoint
bindings, reconstructs each selected model from its saved tensors, evaluates
all outer rows with separate NumPy equations, and rebuilds the paired-source
bootstrap and promotion gate from canonical per-row records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (13, 42, 2026)
FOLDS = (0, 1, 2)
CASES = ("scale", "mixture", "bern_uniform", "bern_tied", "bern_d3",
         "bern_d6", "bern_d12", "median_mdn", "mdn", "spline")
PRIMARY = "bern_d6"
CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
BOUNDARIES = np.asarray([-.7, -.1, .1, .7], dtype=np.float32).astype(np.float64)
TOLERANCE = float(np.float32(.1))
QUANTILE_LEVELS = (.25, .75, .1, .9, .05, .95, .025, .975, .5)
INTERVAL_LEVELS = (.5, .8, .9, .95)
BOOTSTRAP_SEED = 20261013
BOOTSTRAP_REPLICATES = 5000
NUMERIC_TOLERANCE = 1e-5


class AuditError(RuntimeError):
    """An audit failure with a concise reviewer-actionable message."""


class Recorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.findings: list[dict[str, str]] = []

    def must(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            self.findings.append({"check": name, "issue": detail})
            raise AuditError(detail)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def source_of(sample_id: str) -> str:
    if not isinstance(sample_id, str) or sample_id.count("/") != 1:
        raise ValueError(f"Malformed CH-SIMS source/clip ID: {sample_id!r}")
    source, clip = sample_id.split("/", 1)
    if not source or not clip:
        raise ValueError(f"Malformed CH-SIMS source/clip ID: {sample_id!r}")
    return source


def parse_author_train(csv_path: Path) -> tuple[list[str], dict[str, np.float32]]:
    ids: list[str] = []
    target_by_id: dict[str, np.float32] = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("mode") != "train":
                continue
            sample_id = f"{row['video_id']}/{row['clip_id']}"
            target = np.float32(float(row["label"]))
            if sample_id in target_by_id or not np.isfinite(target) or abs(float(target)) > 1:
                raise AuditError("Author TRAIN CSV has duplicate IDs or invalid native targets")
            ids.append(sample_id)
            target_by_id[sample_id] = target
    if not ids or len(ids) != len(set(ids)):
        raise AuditError("Author TRAIN CSV did not yield a nonempty unique ID cohort")
    return ids, target_by_id


def _assert_roles(manifest: dict[str, Any], canonical_ids: list[str]) -> None:
    folds = manifest.get("folds")
    roles = ("point_fit", "point_select", "shape_fit", "shape_select", "outer")
    universe = set(canonical_ids)
    if not isinstance(folds, list) or len(folds) != 3:
        raise AuditError("Point manifest must contain exactly three outer folds")
    outer_seen: list[str] = []
    outer_sources: list[str] = []
    for fold_index, fold in enumerate(folds):
        if fold.get("fold") != fold_index:
            raise AuditError("Point role manifest fold order differs from 0, 1, 2")
        role_ids = {name: fold.get(name + "_ids") for name in roles}
        if any(not isinstance(values, list) or not values or len(values) != len(set(values))
               for values in role_ids.values()):
            raise AuditError(f"Fold {fold_index} has an empty or duplicate source-held role")
        role_sets = {name: set(values) for name, values in role_ids.items()}
        if set.union(*role_sets.values()) != universe:
            raise AuditError(f"Fold {fold_index} role IDs do not partition all author TRAIN IDs")
        source_sets = {name: {source_of(sid) for sid in values} for name, values in role_ids.items()}
        for i, left in enumerate(roles):
            for right in roles[i + 1:]:
                if source_sets[left] & source_sets[right]:
                    raise AuditError(f"Fold {fold_index} roles {left}/{right} share source videos")
        if role_sets["outer"] != set(fold.get("outer_ids", [])):
            raise AuditError(f"Fold {fold_index} outer IDs differ from its explicit outer list")
        point_holdout = set(fold.get("point_oof_ids", []))
        expected_holdout = role_sets["shape_fit"] | role_sets["shape_select"] | role_sets["outer"]
        if point_holdout != expected_holdout or set(fold.get("oof_ids", [])) != expected_holdout:
            raise AuditError(f"Fold {fold_index} point holdout is not shape_fit+shape_select+outer")
        if set(fold.get("fit_ids", [])) != role_sets["point_fit"] or set(fold.get("select_ids", [])) != role_sets["point_select"]:
            raise AuditError(f"Fold {fold_index} point fit/select aliases differ from the declared roles")
        if set(fold.get("point_scaler_fit_ids", [])) != role_sets["point_fit"]:
            raise AuditError(f"Fold {fold_index} point scaler includes rows outside point_fit")
        outer_seen.extend(role_ids["outer"])
        outer_sources.extend(sorted(source_sets["outer"]))
    if len(outer_seen) != len(universe) or set(outer_seen) != universe or len(set(outer_seen)) != len(universe):
        raise AuditError("Outer source folds do not hold every author TRAIN ID exactly once")
    if len(outer_sources) != len(set(outer_sources)):
        raise AuditError("An outer source video appears in more than one fold")


def audit_point_tables(point_suite_path: Path, repo_root: Path, rec: Recorder) -> dict[str, Any]:
    suite = json.loads(point_suite_path.read_text(encoding="utf-8"))
    manifest_path = resolve_path(suite["manifest_path"], repo_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rec.must("point suite manifest digest",
             suite.get("manifest_sha256") == sha256_file(manifest_path),
             "Point suite manifest SHA256 does not match suite.json")
    rec.must("point suite manifest path",
             manifest_path.resolve() == point_suite_path.with_suffix(point_suite_path.suffix + ".manifest.json").resolve(),
             "Point suite does not use its adjacent manifest sidecar")
    rec.must("formal point stage markers",
             suite.get("suite_complete") is True and suite.get("smoke_only") is False
             and suite.get("full_matrix_requested") is True
             and suite.get("formal_point_matrix_count") == 9
             and suite.get("expected_formal_point_matrix_count") == 9
             and suite.get("table_count") == 27 and suite.get("expected_formal_table_count") == 27
             and manifest.get("smoke_only") is False
             and manifest.get("official_VALID_used") is False
             and manifest.get("official_TEST_used") is False,
             "Point suite is incomplete, smoke-only, or not TRAIN-only")
    rec.must("point suite source hash",
             manifest.get("author_csv_sha256") == CSV_SHA256
             and suite.get("source_csv_sha256", CSV_SHA256) == CSV_SHA256,
             "Point stage is not bound to the registered author TRAIN CSV SHA256")
    canonical_ids = manifest.get("canonical_author_train_ids", [])
    rec.must("point canonical cohort", len(canonical_ids) == 2722 and len(set(canonical_ids)) == 2722
             and manifest.get("author_train_count") == 2722 and manifest.get("author_source_count") == 129,
             "Point manifest does not bind 2,722 unique author TRAIN IDs from 129 sources")
    config_path = resolve_path("configs/native_nested_shape_points_v1.json", repo_root)
    config_sha = sha256_file(config_path)
    rec.must("point registered config binding",
             manifest.get("config_sha256") == config_sha and suite.get("config_sha256") in (None, "", config_sha),
             "Point manifest config SHA differs from the registered point config")
    author_csv = resolve_path("data/author/simsv2/meta.csv", repo_root)
    rec.must("author CSV file hash", sha256_file(author_csv) == CSV_SHA256,
             "Current author CSV bytes differ from the registered source hash")
    author_ids, targets = parse_author_train(author_csv)
    rec.must("TRAIN-only author cohort/order", author_ids == canonical_ids,
             "Point canonical IDs differ from an independent TRAIN-only CSV parse")
    sources = {source_of(sid) for sid in canonical_ids}
    rec.must("author source count", len(sources) == 129,
             f"Independent author CSV parse found {len(sources)} sources, expected 129")
    nested = manifest.get("nested_manifest", {})
    rec.must("nested manifest content hash", manifest.get("nested_manifest_sha256") == canonical_hash(nested),
             "Nested role manifest SHA does not match the serialized role manifest")
    _assert_roles(nested, canonical_ids)

    expected_matrix = {(seed, fold) for seed in SEEDS for fold in FOLDS}
    rows = suite.get("results", [])
    keys = {(r.get("seed"), r.get("fold")) for r in rows}
    rec.must("point model matrix", len(rows) == 9 and keys == expected_matrix and len(keys) == len(rows),
             "Point suite lacks exactly one complete model per registered seed/fold")
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    table_cache: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        seed, fold = int(row["seed"]), int(row["fold"])
        by_key[(seed, fold)] = row
        rec.must(f"point identity {seed}/{fold} complete",
                 row.get("complete") is True and row.get("point_result", {}).get("smoke_only") is False
                 and row.get("official_VALID_used") is False and row.get("official_TEST_used") is False,
                 f"Point identity seed {seed}, fold {fold} is incomplete, smoke-only, or reads official held data")
        point_root = resolve_path(row["point_model_root"], repo_root)
        point_hashes = {
            "best.pt": row.get("point_best_checkpoint_sha256"),
            "last.pt": row.get("point_last_checkpoint_sha256"),
            "history.json": row.get("point_history_sha256"),
            "identity.json": row.get("point_identity_sha256"),
        }
        for basename, expected_sha in point_hashes.items():
            path = point_root / basename
            rec.must(f"point {seed}/{fold} {basename} binding",
                     path.is_file() and expected_sha == sha256_file(path),
                     f"Point seed {seed}, fold {fold} {basename} is missing or hash-mismatched")
        rec.must(f"point {seed}/{fold} point checkpoint cross-binding",
                 row.get("point_result", {}).get("checkpoint_sha256") == row.get("point_best_checkpoint_sha256"),
                 f"Point seed {seed}, fold {fold} suite and model result bind different best checkpoints")
        fold_manifest = nested["folds"][fold]
        table_entries = row.get("tables", {})
        rec.must(f"point {seed}/{fold} table role set", set(table_entries) == {"shape_fit", "shape_select", "outer"},
                 f"Point seed {seed}, fold {fold} must export exactly its three shape roles")
        for role in ("shape_fit", "shape_select", "outer"):
            entry = table_entries[role]
            path = resolve_path(entry["path"], repo_root)
            ids_path = resolve_path(entry["ids_path"], repo_root)
            expected_ids = fold_manifest[role + "_ids"]
            expected_sources = sorted({source_of(sid) for sid in expected_ids})
            rec.must(f"point {seed}/{fold}/{role} file hashes",
                     path.is_file() and entry.get("sha256") == sha256_file(path)
                     and ids_path.is_file() and entry.get("ids_sha256") == sha256_file(ids_path),
                     f"Point table or ID file hash mismatch for {seed}/{fold}/{role}")
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            rec.must(f"point {seed}/{fold}/{role} ID role/order",
                     ids == expected_ids and entry.get("ids_in_canonical_author_train_order") is True,
                     f"Point table IDs for {seed}/{fold}/{role} differ from canonical role order")
            rec.must(f"point {seed}/{fold}/{role} metadata binding",
                     entry.get("count") == len(expected_ids)
                     and entry.get("source_groups") == expected_sources
                     and entry.get("source_csv_sha256") == CSV_SHA256
                     and entry.get("config_sha256") == manifest.get("config_sha256")
                     and isinstance(entry.get("implementation_sha256"), dict)
                     and all(sha256_file(repo_root / "jevaffect" / name) == digest
                             for name, digest in entry.get("implementation_sha256", {}).items()),
                     f"Point table metadata is not fully bound for {seed}/{fold}/{role}")
            with np.load(path, allow_pickle=False) as archive:
                rec.must(f"point {seed}/{fold}/{role} array schema",
                         set(archive.files) == {"context", "anchor", "native_point", "target", "ids"},
                         f"Unexpected arrays in point table {seed}/{fold}/{role}")
                data = {name: np.array(archive[name], copy=True) for name in archive.files}
            n = len(expected_ids)
            arrays_valid = (data["context"].shape == (n, 256)
                            and all(data[name].shape == (n,) for name in ("anchor", "native_point", "target", "ids"))
                            and all(data[name].dtype == np.float32 and np.isfinite(data[name]).all()
                                    for name in ("context", "anchor", "native_point", "target")))
            rec.must(f"point {seed}/{fold}/{role} array shapes/dtypes", arrays_valid,
                     f"Invalid context, point, anchor, target shape/dtype in {seed}/{fold}/{role}")
            ids_np = data["ids"].astype(str).tolist()
            target_expected = np.asarray([targets[sid] for sid in expected_ids], dtype=np.float32)
            rec.must(f"point {seed}/{fold}/{role} NPZ IDs/targets/anchors",
                     ids_np == expected_ids
                     and np.array_equal(data["target"], target_expected)
                     and np.array_equal(data["anchor"], np.clip(data["native_point"], -1., 1.).astype(np.float32)),
                     f"Point table {seed}/{fold}/{role} IDs, native targets, or clipped anchor semantics differ")
            table_cache[(seed, fold, role)] = {**data, "ids": ids, "entry": entry}
    return {"suite": suite, "manifest": manifest, "manifest_sha256": suite["manifest_sha256"],
            "canonical_ids": canonical_ids, "target_by_id": targets, "nested": nested,
            "rows": by_key, "tables": table_cache, "source_ids": sorted(sources),
            "suite_path": point_suite_path}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1. / (1. + np.exp(-x[positive]))
    expx = np.exp(x[~positive])
    out[~positive] = expx / (1. + expx)
    return out


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0., np.asarray(x, dtype=np.float64))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    return np.log(p) - np.log1p(-p)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _beta_basis_cdf(t: np.ndarray, degree: int) -> np.ndarray:
    """Integer Beta CDFs as independent binomial survival polynomials."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0., 1.)
    order = degree + 1
    # For Beta(k+1,n-k), the CDF is P[Binomial(n,t) >= k+1].
    # Build the lower binomial tail in ascending order, then take its complement.
    lower_terms = []
    for j in range(order + 1):
        lower_terms.append(math.comb(order, j) * np.power(t, j) * np.power(1. - t, order - j))
    lower = np.stack(lower_terms, axis=-1)
    return np.stack([1. - lower[..., :k + 1].sum(axis=-1) for k in range(degree + 1)], axis=-1)


def _latent_mdn_median(scale: np.ndarray, weights: np.ndarray, centers: np.ndarray) -> np.ndarray:
    lo = np.broadcast_to(centers[0] - 40. * scale, scale.shape).copy()
    hi = np.broadcast_to(centers[-1] + 40. * scale, scale.shape).copy()
    for _ in range(64):
        mid = (lo + hi) / 2.
        cdf = (_sigmoid((mid[:, None] - centers[None, :]) / scale[:, None]) * weights).sum(-1)
        lower = cdf < .5
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    return (lo + hi) / 2.


def reference_parameters(case: str, anchor: np.ndarray, raw: np.ndarray) -> dict[str, Any]:
    """Reconstruct each registered conditional distribution from head outputs."""
    m = np.asarray(anchor, dtype=np.float64).reshape(-1)
    r = np.asarray(raw, dtype=np.float64)
    if r.ndim != 2 or r.shape[0] != len(m):
        raise ValueError("Raw head output and anchor row counts differ")
    if case in ("scale", "mixture"):
        scale = _softplus(r[:, 0]) + 1e-4
        if case == "mixture":
            center_cdf = _sigmoid(m / .05)
            max_weight = .5 / np.maximum(center_cdf, 1. - center_cdf)
            weight = (1. - 1e-4) * max_weight * _sigmoid(r[:, 1])
        else:
            weight = np.zeros_like(m)
        q = (.5 - weight * _sigmoid(m / .05)) / (1. - weight)
        return {"kind": "logistic", "anchor": m, "scale": scale,
                "weight": weight, "location": m - scale * _logit(q), "bound": 1.}
    if case.startswith("bern_"):
        degree = 3 if case == "bern_d3" else 12 if case == "bern_d12" else 6
        margins = _softplus(r[:, :2]) + 1e-4
        widths = np.stack((m + 1., 1. - m), axis=1) + margins
        if case == "bern_uniform":
            logits = np.zeros((len(m), 2, degree + 1), dtype=np.float64)
        elif case == "bern_tied":
            shared = np.broadcast_to(r[:, 2:2 + degree + 1][:, None, :], (len(m), 2, degree + 1))
            logits = shared
        else:
            logits = r[:, 2:].reshape(len(m), 2, degree + 1)
        return {"kind": "bernstein", "anchor": m, "widths": widths,
                "weights": _softmax(logits), "degree": degree, "bound": 1.}
    if case in ("median_mdn", "mdn"):
        scale = _softplus(r[:, 0]) + 1e-4
        weights = _softmax(r[:, 1:])
        centers = np.linspace(-1.4, 1.4, 15, dtype=np.float64)
        median = _latent_mdn_median(scale, weights, centers) if case == "median_mdn" else None
        return {"kind": "mdn", "anchor": m, "scale": scale, "weights": weights,
                "centers": centers, "shift": m - median if median is not None else np.zeros_like(m),
                "median_fixed": case == "median_mdn", "bound": 1.}
    if case == "spline":
        scale = _softplus(r[:, 0]) + 1e-4
        increments = .5 * _softmax(r[:, 1:].reshape(len(m), 2, 4))
        knots = np.concatenate((np.zeros((len(m), 1)), increments[:, 0].cumsum(-1),
                                .5 + increments[:, 1].cumsum(-1)), axis=1)
        return {"kind": "spline", "anchor": m, "scale": scale, "knots": knots, "bound": 1.}
    raise ValueError(f"Unregistered case: {case}")


def reference_cdf(params: dict[str, Any], value: np.ndarray) -> np.ndarray:
    """Evaluate an independently coded censored CDF for sample-by-grid input."""
    m = params["anchor"]
    x = np.asarray(value, dtype=np.float64)
    if x.ndim == 1:
        if x.shape[0] != len(m):
            raise ValueError("Vector query length differs from sample count")
        x = x[:, None]
    if x.ndim != 2 or x.shape[0] != len(m) or not np.isfinite(x).all():
        raise ValueError("Finite sample-by-grid query required")
    if params["kind"] == "logistic":
        base = _sigmoid((x - params["location"][:, None]) / params["scale"][:, None])
        kernel = _sigmoid(x / .05)
        result = params["weight"][:, None] * kernel + (1. - params["weight"][:, None]) * base
    elif params["kind"] == "bernstein":
        degree = params["degree"]
        t_left = (m[:, None] - x) / params["widths"][:, 0, None]
        t_right = (x - m[:, None]) / params["widths"][:, 1, None]
        left_basis = _beta_basis_cdf(t_left, degree)
        right_basis = _beta_basis_cdf(t_right, degree)
        left = 1. - (left_basis * params["weights"][:, None, 0, :]).sum(-1)
        right = (right_basis * params["weights"][:, None, 1, :]).sum(-1)
        result = np.where(x < m[:, None], .5 * left, .5 + .5 * right)
    elif params["kind"] == "mdn":
        latent = _sigmoid((x[:, :, None] - params["shift"][:, None, None]
                           - params["centers"][None, None, :]) / params["scale"][:, None, None])
        result = (params["weights"][:, None, :] * latent).sum(-1)
    elif params["kind"] == "spline":
        u = _sigmoid((x - m[:, None]) / params["scale"][:, None])
        u = np.where(x < -1., 0., np.where(x >= 1., 1., u))
        index = np.floor(u * 8.).astype(np.int64).clip(0, 7)
        left = np.take_along_axis(params["knots"][:, None, :], index[:, :, None], axis=2)[:, :, 0]
        right = np.take_along_axis(params["knots"][:, None, :], (index + 1)[:, :, None], axis=2)[:, :, 0]
        result = left + (right - left) * (u * 8. - index)
        result = np.clip(result, 0., 1.)
    else:
        raise ValueError(f"Unknown reconstructed distribution: {params['kind']}")
    return np.where(x < -1., 0., np.where(x >= 1., 1., result))


def reference_bins(params: dict[str, Any]) -> np.ndarray:
    values = reference_cdf(params, np.broadcast_to(BOUNDARIES[None, :], (len(params["anchor"]), 4)))
    endpoints = np.concatenate((np.zeros((len(values), 1)), values, np.ones((len(values), 1))), axis=1)
    return np.diff(endpoints, axis=1)


def reference_quantiles(params: dict[str, Any]) -> np.ndarray:
    target = np.broadcast_to(np.asarray(QUANTILE_LEVELS, dtype=np.float64)[None, :],
                             (len(params["anchor"]), len(QUANTILE_LEVELS)))
    lo, hi = np.full_like(target, -1.), np.full_like(target, 1.)
    for _ in range(64):
        mid = (lo + hi) / 2.
        lower = reference_cdf(params, mid) < target
        lo = np.where(lower, mid, lo)
        hi = np.where(lower, hi, mid)
    q = (lo + hi) / 2.
    q = np.where(target == 0., -1., np.where(target == 1., 1., q))
    if params["kind"] != "mdn" or params["median_fixed"]:
        q[:, 8] = params["anchor"]
    return q


def reference_hit_probability(params: dict[str, Any], raw_point: np.ndarray) -> np.ndarray:
    point = np.asarray(raw_point, dtype=np.float32).astype(np.float64)
    low = point - TOLERANCE
    high = point + TOLERANCE
    before = reference_cdf(params, low)
    before = np.where(low <= -1., 0., before)
    after = reference_cdf(params, high)
    return np.clip(after - before, 0., 1.)[:, 0]


def reference_crps(params: dict[str, Any], target: np.ndarray) -> np.ndarray:
    grid = np.linspace(-1., 1., 1001, dtype=np.float64)
    values = np.broadcast_to(grid[None, :], (len(params["anchor"]), len(grid)))
    cdf = reference_cdf(params, values)
    truth = values >= np.asarray(target, dtype=np.float32).astype(np.float64)[:, None]
    return np.trapezoid(np.square(cdf - truth), grid, axis=1)


def _linear_head_outputs(state: dict[str, Any], normalized: np.ndarray) -> np.ndarray:
    """Run saved Linear/GELU/Linear tensors without importing ShapeHead."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(np.asarray(normalized, dtype=np.float32))
    with torch.inference_mode():
        hidden = F.gelu(F.linear(x, state["context.0.weight"].cpu(), state["context.0.bias"].cpu()))
        raw = F.linear(hidden, state["head.weight"].cpu(), state["head.bias"].cpu())
    return raw.cpu().numpy()


def _normalization_for_fit(fit: dict[str, Any]) -> dict[str, Any]:
    x = np.concatenate((fit["context"], fit["anchor"][:, None]), axis=1).astype(np.float32)
    mean = x.astype(np.float64).mean(axis=0)
    scale = x.astype(np.float64).std(axis=0)
    scale[scale < 1e-6] = 1.
    return {"mean": mean.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist(),
            "fit_indices": list(range(len(fit["ids"])))}, x


def _normalize_context(data: dict[str, Any], stats: dict[str, Any]) -> np.ndarray:
    x = np.concatenate((data["context"], data["anchor"][:, None]), axis=1).astype(np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    scale = np.asarray(stats["scale"], dtype=np.float32)
    return ((x - mean) / scale).astype(np.float32)


def _class_targets(target: np.ndarray) -> np.ndarray:
    return np.searchsorted(BOUNDARIES.astype(np.float32), np.asarray(target, dtype=np.float32), side="left")


def _raw_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _weighted_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    total = 0.
    for label in np.unique(truth):
        support = np.count_nonzero(truth == label)
        denominator = support + np.count_nonzero(prediction == label)
        tp = np.count_nonzero((truth == label) & (prediction == label))
        total += support * 2. * tp / denominator
    return total / len(truth)


def _author_native_metrics(point: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    raw_p = np.asarray(point, dtype=np.float32).reshape(-1)
    raw_y = np.asarray(target, dtype=np.float32).reshape(-1)
    p = np.clip(raw_p, -1., 1.)
    y = np.clip(raw_y, -1., 1.)
    result: dict[str, Any] = {"count": len(y), "MAE": float(np.mean(np.abs(p - y))),
                              "Corr": _raw_corr(p, y)}
    for count, edges in ((2, [0.]), (3, [-.1, .1]), (5, [-.7, -.1, .1, .7])):
        edge_array = np.asarray(edges, dtype=p.dtype)
        pred_cls = np.searchsorted(edge_array, p, side="left")
        truth_cls = np.searchsorted(edge_array, y, side="left")
        result[f"Mult_acc_{count}"] = float(np.mean(pred_cls == truth_cls))
        if count == 2:
            result["F1_score"] = float(_weighted_f1(truth_cls, pred_cls))
            weak = (y >= -.4) & (y <= .4)
            result["weak_count"] = int(weak.sum())
            result["Mult_acc_2_weak"] = float(np.mean(pred_cls[weak] == truth_cls[weak])) if weak.any() else None
    denominator = float(np.sum((y.astype(np.float64) - y.mean()) ** 2))
    numerator = float(np.sum((y.astype(np.float64) - p.astype(np.float64)) ** 2))
    result["R_squre"] = 1. - numerator / denominator if denominator else (1. if numerator == 0 else 0.)
    return result


def _aggregate_metrics(point: np.ndarray, target: np.ndarray, bins: np.ndarray,
                       hit_p: np.ndarray, q: np.ndarray, crps: np.ndarray) -> dict[str, Any]:
    point = np.asarray(point, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    labels = _class_targets(target)
    hit_truth = (np.abs(point - target) <= np.float32(.1)).astype(np.float64)
    hit_p = np.asarray(hit_p, dtype=np.float64)
    clipped_hit = np.clip(hit_p, 1e-12, 1. - 1e-12)
    errors = np.abs(point - target)
    groups = np.minimum((hit_p * 15).astype(int), 14)
    reliability = []
    for group in range(15):
        mask = groups == group
        if mask.any():
            reliability.append({"bin": group, "count": int(mask.sum()),
                                "mean_probability": float(hit_p[mask].mean()),
                                "observed_hit_rate": float(hit_truth[mask].mean())})
    order = np.argsort(-hit_p, kind="stable")
    risk = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    native = _author_native_metrics(point, target)
    one_hot = np.eye(5, dtype=np.float64)[labels]
    result: dict[str, Any] = {
        "native": native,
        "native_raw_point": native,
        "count": len(target), "hit_count": int(hit_truth.sum()),
        "hit_Brier": float(np.mean(np.square(hit_p - hit_truth))),
        "hit_log_score": float(-np.mean(hit_truth * np.log(clipped_hit)
                                         + (1. - hit_truth) * np.log1p(-clipped_hit))),
        "hit_ECE15": float(sum(row["count"] / len(target)
                                * abs(row["mean_probability"] - row["observed_hit_rate"])
                                for row in reliability)),
        "risk_AURC": float(risk.mean()), "reliability": reliability,
        "risk_at_coverage": {str(c): float(risk[max(0, math.ceil(len(target) * c) - 1)])
                             for c in (.1, .25, .5, .75, 1.)},
        "bin_log_score": float(-np.log(np.maximum(bins[np.arange(len(target)), labels], 1e-12)).mean()),
        "bin_Brier": float(np.mean(np.square(bins - one_hot).sum(axis=1))),
        "CRPS_trapezoid_1001": float(np.mean(crps)),
        "anchor_clipped_count": int(np.count_nonzero(point != np.clip(point, -1., 1.))),
        "intervals": {}, "interval_scores": {},
    }
    for j, nominal in enumerate(INTERVAL_LEVELS):
        lo, hi = q[:, 2 * j], q[:, 2 * j + 1]
        result["intervals"][str(nominal)] = {
            "coverage": float(np.mean((target >= lo) & (target <= hi))),
            "mean_width": float(np.mean(hi - lo)),
        }
        alpha = 1. - nominal
        scores = (hi - lo + 2. / alpha
                  * (np.maximum(lo - target, 0.) + np.maximum(target - hi, 0.)))
        result["interval_scores"][str(nominal)] = float(np.mean(scores))
    result["raw_unclipped_point"] = {
        "MAE": float(np.mean(np.abs(point.astype(np.float64) - target.astype(np.float64)))),
        "Corr": _raw_corr(point, target),
    }
    result["native_raw_point_semantics"] = "producer key is author native protocol; CH-SIMS metric clips raw point to [-1,1]"
    return result


def _make_records(ids: list[str], data: dict[str, Any], bins: np.ndarray, hit_p: np.ndarray,
                  q: np.ndarray, crps: np.ndarray) -> list[dict[str, Any]]:
    point = np.asarray(data["native_point"], dtype=np.float32)
    anchor = np.asarray(data["anchor"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    labels = _class_targets(target)
    hit_truth = (np.abs(point - target) <= np.float32(.1)).astype(np.float64)
    clipped = np.clip(hit_p, 1e-12, 1. - 1e-12)
    result = []
    for i, sid in enumerate(ids):
        interval_values, scores = {}, {}
        for j, nominal in enumerate(INTERVAL_LEVELS):
            lo, hi = q[i, 2 * j:2 * j + 2]
            alpha = 1. - nominal
            interval_values[str(nominal)] = [float(lo), float(hi)]
            scores[str(nominal)] = float(hi - lo + 2. / alpha
                                         * (max(lo - float(target[i]), 0.)
                                            + max(float(target[i]) - hi, 0.)))
        nll = -math.log(max(float(bins[i, labels[i]]), 1e-12))
        onehot = np.eye(5, dtype=np.float64)[labels[i]]
        hit = float(hit_p[i])
        result.append({
            "id": sid, "target": float(target[i]), "native_point": float(point[i]),
            "anchor": float(anchor[i]), "median": float(q[i, 8]),
            "bin_probabilities": bins[i].tolist(), "p_hit": hit,
            "intervals": interval_values, "interval_scores": scores,
            "bin_NLL": float(nll), "bin_Brier": float(np.square(bins[i] - onehot).sum()),
            "hit_Brier": float((hit - hit_truth[i]) ** 2),
            "hit_log_score": float(-hit_truth[i] * math.log(float(clipped[i]))
                                    - (1. - hit_truth[i]) * math.log1p(-float(clipped[i]))),
            "CRPS_trapezoid_1001": float(crps[i]),
        })
    return result


def _compare_json(expected: Any, actual: Any, path: str = "$", tol: float = NUMERIC_TOLERANCE) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AuditError(f"Serialized value keys differ at {path}")
        for key in expected:
            _compare_json(expected[key], actual[key], f"{path}.{key}", tol)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AuditError(f"Serialized list length differs at {path}")
        for i, (left, right) in enumerate(zip(expected, actual)):
            _compare_json(left, right, f"{path}[{i}]", tol)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isfinite(float(actual)):
            raise AuditError(f"Serialized number is invalid at {path}")
        if abs(float(expected) - float(actual)) > tol:
            raise AuditError(f"Serialized number differs at {path}: {expected!r} versus {actual!r}")
    elif expected != actual:
        raise AuditError(f"Serialized value differs at {path}: {expected!r} versus {actual!r}")


def _torch_load(path: Path) -> dict[str, Any]:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _expected_output_dim(case: str) -> int:
    return {"scale": 1, "mixture": 2, "bern_uniform": 2, "bern_tied": 9,
            "bern_d3": 10, "bern_d6": 16, "bern_d12": 28,
            "median_mdn": 16, "mdn": 16, "spline": 9}[case]


def _replay_stopping(history: list[dict[str, Any]], fit_count: int, config: dict[str, Any]) -> dict[str, Any]:
    if not history or len(history) > int(config["max_epochs"]):
        raise AuditError("Shape history is empty or exceeds its registered epoch budget")
    scores = [float(row["selection_NLL"]) for row in history]
    if any(not math.isfinite(value) for value in scores):
        raise AuditError("Shape selection history contains a nonfinite NLL")
    best = math.inf
    stale = 0
    best_epoch = None
    for i, score in enumerate(scores):
        if score < best:
            best, stale, best_epoch = score, 0, i + 1
        else:
            stale += 1
    complete_by_patience = stale >= int(config["patience"])
    complete_by_budget = len(history) >= int(config["max_epochs"])
    if not (complete_by_patience or complete_by_budget):
        raise AuditError("Shape history stopped before patience or the maximum epoch budget")
    if complete_by_patience and len(history) < int(config["max_epochs"]) and stale != int(config["patience"]):
        raise AuditError("Shape history continues past its first patience stop")
    expected_updates = math.ceil(fit_count / int(config["batch_size"]))
    for epoch, row in enumerate(history, start=1):
        if row.get("epoch") != epoch or row.get("updates") != expected_updates:
            raise AuditError(f"Shape epoch {epoch} has an invalid number of optimizer updates")
        if not math.isfinite(float(row.get("fit_NLL", math.nan))) or not math.isfinite(float(row.get("seconds", math.nan))):
            raise AuditError(f"Shape epoch {epoch} has nonfinite fit loss or timing")
    return {"best": best, "best_epoch": int(best_epoch), "stale": stale,
            "complete": complete_by_patience or complete_by_budget,
            "updates_per_epoch": expected_updates}


def audit_one_fit(shape_row: dict[str, Any], *, case: str, seed: int, fold: int,
                  config: dict[str, Any], config_hash: str, plan_hash: str,
                  repo_root: Path, point: dict[str, Any], rec: Recorder) -> list[dict[str, Any]]:
    label = f"{case} seed {seed} fold {fold}"
    rec.must(label + " result identity",
             shape_row.get("case") == case and shape_row.get("seed") == seed and shape_row.get("fold") == fold
             and shape_row.get("complete") is True and shape_row.get("smoke_only") is False
             and shape_row.get("official_VALID_used") is False and shape_row.get("official_TEST_used") is False,
             f"Formal shape result is incomplete, duplicated, or has an invalid identity for {label}")
    if not shape_row.get("complete") or shape_row.get("smoke_only"):
        raise AuditError(f"Cannot audit outer forecasts for incomplete fit {label}")
    model_root = resolve_path(shape_row["root"], repo_root)
    expected_root = resolve_path(config["run_root"], repo_root) / "formal" / case / f"seed_{seed}" / f"fold_{fold}"
    rec.must(label + " artifact root", model_root.resolve() == expected_root.resolve(),
             f"Shape result root differs from the registered {case}/{seed}/{fold} location")
    history_path = model_root / "history.json"
    identity_path = model_root / "identity.json"
    best_path = model_root / "best.pt"
    last_path = model_root / "last.pt"
    prediction_path = model_root / "outer_predictions.json"
    result_path = model_root / "result.json"
    for path, hash_key in ((best_path, "best_sha256"), (last_path, "last_sha256"),
                           (history_path, "history_sha256"), (prediction_path, "predictions_sha256")):
        rec.must(label + " artifact hash " + path.name,
                 path.is_file() and shape_row.get(hash_key) == sha256_file(path),
                 f"Saved {path.name} is missing or differs from the suite hash for {label}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    point_stage = point["rows"][(seed, fold)]
    expected_binding = {
        "suite_sha256": sha256_file(point["suite_path"]),
        "manifest_sha256": point["manifest_sha256"],
        "point_best_checkpoint_sha256": point_stage["point_best_checkpoint_sha256"],
        "tables": {role: point_stage["tables"][role]["sha256"] for role in ("shape_fit", "shape_select", "outer")},
        "ids": {role: point_stage["tables"][role]["ids_sha256"] for role in ("shape_fit", "shape_select", "outer")},
        "source_csv_sha256": CSV_SHA256,
        "nested_manifest_sha256": point["manifest"].get("nested_manifest_sha256"),
        "smoke_only": False,
    }
    expected_config = dict(config, config_sha256=config_hash, plan_sha256=plan_hash)
    rec.must(label + " config and point table binding",
             identity.get("config") == expected_config and identity.get("binding") == expected_binding,
             f"Checkpoint identity for {label} is not bound to registered config, point suite, hashes, and IDs")
    rec.must(label + " fit/select ID bindings",
             identity.get("fit_ids") == point["tables"][(seed, fold, "shape_fit")]["ids"]
             and identity.get("select_ids") == point["tables"][(seed, fold, "shape_select")]["ids"]
             and identity.get("outer_used_for_selection") is False
             and identity.get("selection") == "first minimum shape_select native-bin NLL",
             f"Checkpoint identity for {label} includes wrong fit/select IDs or selection semantics")
    expected_impl_files = ("nested_shape_training.py", "shape_controls.py", "anchored_bernstein.py",
                           "anchor_cdf.py", "anchor_calibration.py", "native_training.py", "native_metrics.py")
    expected_impl = {name: sha256_file(repo_root / "jevaffect" / name) for name in expected_impl_files}
    rec.must(label + " frozen code hashes", identity.get("implementation") == expected_impl,
             f"Current shape training/control/metric code differs from the code bound into {label}")

    tables = {role: point["tables"][(seed, fold, role)] for role in ("shape_fit", "shape_select", "outer")}
    stats, _ = _normalization_for_fit(tables["shape_fit"])
    rec.must(label + " shape_fit-only normalization",
             identity.get("normalization") == stats,
             f"Shape normalization for {label} differs from an independent shape_fit-only recomputation")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    stopping = _replay_stopping(history, len(tables["shape_fit"]["ids"]), config)
    rec.must(label + " first-minimum/early-stop selection",
             shape_row.get("epochs") == len(history) and shape_row.get("best_epoch") == stopping["best_epoch"]
             and shape_row.get("updates") == len(history) * stopping["updates_per_epoch"]
             and history[stopping["best_epoch"] - 1]["selection_NLL"] == stopping["best"],
             f"Saved fit {label} does not select the first minimum within the registered patience/budget")
    output_dim = _expected_output_dim(case)
    expected_shapes = {
        "context.0.weight": (32, 257), "context.0.bias": (32,),
        "head.weight": (output_dim, 32), "head.bias": (output_dim,),
    }
    best_state = _torch_load(best_path)
    last_state = _torch_load(last_path)
    best_model = best_state.get("model", {})
    last_model = last_state.get("model", {})
    rec.must(label + " selected checkpoint tensor schema",
             set(best_model) == set(expected_shapes)
             and all(tuple(best_model[name].shape) == shape and bool(best_model[name].isfinite().all())
                     for name, shape in expected_shapes.items()),
             f"Selected checkpoint {label} has missing, unexpected, or nonfinite model parameters")
    expected_params = 32 * 258 + output_dim * 33
    rec.must(label + " active parameter count",
             shape_row.get("parameters") == expected_params
             and sum(t.numel() for t in best_model.values()) == expected_params,
             f"Trainable parameter count for {label} differs from its registered architecture")
    rec.must(label + " checkpoint reload and history state",
             best_state.get("identity") == identity and best_state.get("epoch") == stopping["best_epoch"]
             and abs(float(best_state.get("selection_NLL", math.nan)) - stopping["best"]) <= 1e-12
             and last_state.get("identity") == identity and last_state.get("history") == history
             and last_state.get("epoch") == len(history) and last_state.get("stale") == stopping["stale"]
             and abs(float(last_state.get("best", math.nan)) - stopping["best"]) <= 1e-12,
             f"Saved selected/last checkpoint reload state differs from first-minimum history for {label}")

    # Recompute the chosen shape-select score from the serialized best tensors.
    selected_data = tables["shape_select"]
    selected_raw = _linear_head_outputs(best_model, _normalize_context(selected_data, stats))
    selected_params = reference_parameters(case, selected_data["anchor"], selected_raw)
    selected_probs = reference_bins(selected_params)
    selected_labels = _class_targets(selected_data["target"])
    replay_score = float(-np.log(np.maximum(selected_probs[np.arange(len(selected_labels)), selected_labels], 1e-12)).mean())
    rec.must(label + " independent selected NLL replay",
             abs(replay_score - stopping["best"]) <= NUMERIC_TOLERANCE,
             f"Independent best checkpoint shape_select NLL {replay_score:.12g} differs from history {stopping['best']:.12g}")

    outer = tables["outer"]
    raw = _linear_head_outputs(best_model, _normalize_context(outer, stats))
    params = reference_parameters(case, outer["anchor"], raw)
    bins = reference_bins(params)
    hit = reference_hit_probability(params, outer["native_point"])
    quantiles = reference_quantiles(params)
    crps = reference_crps(params, outer["target"])
    records = _make_records(outer["ids"], outer, bins, hit, quantiles, crps)
    saved_records = json.loads(prediction_path.read_text(encoding="utf-8"))
    rec.must(label + " independent outer per-row replay",
             [r.get("id") for r in saved_records] == outer["ids"],
             f"Saved outer prediction rows for {label} are not in canonical fold order")
    _compare_json(records, saved_records, f"{label}.outer_records")
    computed = _aggregate_metrics(outer["native_point"], outer["target"], bins, hit, quantiles, crps)
    producer_metrics = dict(shape_row.get("outer") or {})
    producer_metrics.pop("native_raw_point_semantics", None)
    producer_metrics.pop("raw_unclipped_point", None)
    computed_compare = dict(computed)
    computed_compare.pop("native_raw_point_semantics", None)
    computed_compare.pop("raw_unclipped_point", None)
    _compare_json(computed_compare, producer_metrics, f"{label}.outer_aggregate")
    result_on_disk = json.loads(result_path.read_text(encoding="utf-8"))
    _compare_json(shape_row, result_on_disk, f"{label}.result_file")
    return records


def _metric_rows(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {name: np.asarray([float(row[name]) for row in records], dtype=np.float64)
            for name in ("bin_NLL", "bin_Brier", "hit_Brier", "hit_log_score", "CRPS_trapezoid_1001")}


def _merged_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    bins = np.asarray([row["bin_probabilities"] for row in records], dtype=np.float64)
    point = np.asarray([row["native_point"] for row in records], dtype=np.float32)
    anchor = np.asarray([row["anchor"] for row in records], dtype=np.float32)
    target = np.asarray([row["target"] for row in records], dtype=np.float32)
    hit = np.asarray([row["p_hit"] for row in records], dtype=np.float64)
    q = np.asarray([[edge for nominal in INTERVAL_LEVELS for edge in row["intervals"][str(nominal)]]
                    for row in records], dtype=np.float64)
    crps = np.asarray([row["CRPS_trapezoid_1001"] for row in records], dtype=np.float64)
    summary = _aggregate_metrics(point, target, bins, hit, q, crps)
    summary["anchor_clipped_count"] = int(np.count_nonzero(point != anchor))
    summary["anchor_clipping_frequency"] = float(np.mean(point != anchor))
    return summary


def same_point_target_anchor(candidate: list[dict[str, Any]], reference: list[dict[str, Any]]) -> bool:
    """Compare the frozen point table fields for two shape cases in one seed."""
    if len(candidate) != len(reference):
        return False
    return all(left.get("id") == right.get("id")
               and left.get("native_point") == right.get("native_point")
               and left.get("target") == right.get("target")
               and left.get("anchor") == right.get("anchor")
               for left, right in zip(candidate, reference))


def same_seed_raw_point_invariant(merged: dict[str, dict[int, list[dict[str, Any]]]],
                                  cases: tuple[str, ...] = CASES,
                                  seeds: tuple[int, ...] = SEEDS) -> bool:
    """Ensure cases share a point within each seed while seeds may differ."""
    for seed in seeds:
        reference = merged[PRIMARY][seed]
        for case in cases:
            if not same_point_target_anchor(merged[case][seed], reference):
                return False
    return True


def _paired_source_bootstrap(merged: dict[str, dict[int, list[dict[str, Any]]]],
                             source_by_id: dict[str, str], source_names: list[str],
                             replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(len(source_names), size=(replicates, len(source_names)), replace=True)
    draws = np.stack([np.bincount(row, minlength=len(source_names)) for row in samples]).astype(np.float64)
    source_index = {source: i for i, source in enumerate(source_names)}
    case_source_sums: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    source_counts: np.ndarray | None = None
    interval_case_sums: dict[str, dict[str, dict[int, dict[str, np.ndarray]]]] = {}
    for case in CASES:
        case_source_sums[case] = {}
        interval_case_sums[case] = {}
        for seed in SEEDS:
            records = merged[case][seed]
            per_row = _metric_rows(records)
            sums = {metric: np.zeros(len(source_names), dtype=np.float64) for metric in per_row}
            counts = np.zeros(len(source_names), dtype=np.float64)
            interval_sums = {str(level): {"coverage": np.zeros(len(source_names)),
                                          "width": np.zeros(len(source_names))}
                             for level in INTERVAL_LEVELS}
            for i, row in enumerate(records):
                sidx = source_index[source_by_id[row["id"]]]
                counts[sidx] += 1.
                for metric, values in per_row.items():
                    sums[metric][sidx] += values[i]
                target = float(row["target"])
                for level in INTERVAL_LEVELS:
                    lo, hi = row["intervals"][str(level)]
                    interval_sums[str(level)]["coverage"][sidx] += float(lo <= target <= hi)
                    interval_sums[str(level)]["width"][sidx] += float(hi - lo)
            if source_counts is None:
                source_counts = counts
            elif not np.array_equal(counts, source_counts):
                raise AuditError("Canonical rows differ across seed/case bootstrap cohorts")
            case_source_sums[case][str(seed)] = {**sums, "_count": counts}
            interval_case_sums[case][str(seed)] = interval_sums

    output: dict[str, Any] = {"seed": BOOTSTRAP_SEED, "replicates": replicates,
                              "sampled_source_count_per_draw": len(source_names),
                              "source_videos": len(source_names),
                              "paired_source_draws_shared_across_cases_and_seeds": True,
                              "delta_definition": "control metric minus bern_d6 metric; positive favors bern_d6",
                              "comparisons": {}}
    compare_metrics = ("bin_NLL", "bin_Brier", "hit_Brier", "hit_log_score", "CRPS_trapezoid_1001")
    controls = [case for case in CASES if case != PRIMARY]
    pooled_source_count = source_counts * len(SEEDS)
    for control in controls:
        output["comparisons"][control] = {}
        for metric in compare_metrics:
            output["comparisons"][control][metric] = {}
            for seed_key in [*(str(seed) for seed in SEEDS), "pooled"]:
                if seed_key == "pooled":
                    control_sum = sum(case_source_sums[control][str(seed)][metric] for seed in SEEDS)
                    primary_sum = sum(case_source_sums[PRIMARY][str(seed)][metric] for seed in SEEDS)
                    count = pooled_source_count
                    control_all = np.concatenate([_metric_rows(merged[control][seed])[metric] for seed in SEEDS])
                    primary_all = np.concatenate([_metric_rows(merged[PRIMARY][seed])[metric] for seed in SEEDS])
                else:
                    control_sum = case_source_sums[control][seed_key][metric]
                    primary_sum = case_source_sums[PRIMARY][seed_key][metric]
                    count = source_counts
                    seed = int(seed_key)
                    control_all = _metric_rows(merged[control][seed])[metric]
                    primary_all = _metric_rows(merged[PRIMARY][seed])[metric]
                denominator = draws @ count
                delta_draws = (draws @ (control_sum - primary_sum)) / denominator
                low, high = np.quantile(delta_draws, [.025, .975])
                output["comparisons"][control][metric][seed_key] = {
                    "primary_mean": float(primary_all.mean()), "control_mean": float(control_all.mean()),
                    "observed_delta_control_minus_primary": float(control_all.mean() - primary_all.mean()),
                    "paired_source_bootstrap_95_ci": [float(low), float(high)],
                    "bootstrap_probability_primary_better": float(np.mean(delta_draws > 0.)),
                    "primary_better_direction": bool(control_all.mean() > primary_all.mean()),
                }
        output["comparisons"][control]["intervals"] = {}
        for level in INTERVAL_LEVELS:
            output["comparisons"][control]["intervals"][str(level)] = {}
            for metric in ("coverage", "width"):
                output["comparisons"][control]["intervals"][str(level)][metric] = {}
                for seed_key in [*(str(seed) for seed in SEEDS), "pooled"]:
                    if seed_key == "pooled":
                        control_sum = sum(interval_case_sums[control][str(seed)][str(level)][metric]
                                          for seed in SEEDS)
                        primary_sum = sum(interval_case_sums[PRIMARY][str(seed)][str(level)][metric]
                                          for seed in SEEDS)
                        count = pooled_source_count
                        get_values = lambda case: np.concatenate([
                            np.asarray([float((r["target"] >= r["intervals"][str(level)][0])
                                              and (r["target"] <= r["intervals"][str(level)][1]))
                                        if metric == "coverage" else
                                        r["intervals"][str(level)][1] - r["intervals"][str(level)][0]
                                        for r in merged[case][seed]], dtype=np.float64)
                            for seed in SEEDS])
                    else:
                        control_sum = interval_case_sums[control][seed_key][str(level)][metric]
                        primary_sum = interval_case_sums[PRIMARY][seed_key][str(level)][metric]
                        count = source_counts
                        seed = int(seed_key)
                        get_values = lambda case: np.asarray([
                            float((r["target"] >= r["intervals"][str(level)][0])
                                  and (r["target"] <= r["intervals"][str(level)][1]))
                            if metric == "coverage" else
                            r["intervals"][str(level)][1] - r["intervals"][str(level)][0]
                            for r in merged[case][seed]], dtype=np.float64)
                    delta_draws = (draws @ (control_sum - primary_sum)) / (draws @ count)
                    low, high = np.quantile(delta_draws, [.025, .975])
                    cvals, pvals = get_values(control), get_values(PRIMARY)
                    output["comparisons"][control]["intervals"][str(level)][metric][seed_key] = {
                        "primary_mean": float(pvals.mean()), "control_mean": float(cvals.mean()),
                        "observed_delta_control_minus_primary": float(cvals.mean() - pvals.mean()),
                        "paired_source_bootstrap_95_ci": [float(low), float(high)],
                        "primary_better_direction": bool(cvals.mean() < pvals.mean()) if metric == "coverage"
                        else bool(cvals.mean() > pvals.mean()),
                    }
    return output


def _promotion_gate(bootstrap: dict[str, Any]) -> dict[str, Any]:
    comparisons = bootstrap["comparisons"]
    checks: list[dict[str, Any]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    for control in ("mixture", "median_mdn"):
        for metric in ("bin_NLL", "hit_Brier", "hit_log_score"):
            entries = comparisons[control][metric]
            seed_positive = all(entries[str(seed)]["primary_better_direction"] for seed in SEEDS)
            pooled_ci_positive = entries["pooled"]["paired_source_bootstrap_95_ci"][0] > 0.
            add(f"{metric} versus {control}", seed_positive and pooled_ci_positive,
                f"three seed directions positive={seed_positive}; pooled paired 95% CI lower bound > 0={pooled_ci_positive}")
    for control in ("bern_uniform", "bern_tied", "bern_d3", "bern_d12"):
        entries = comparisons[control]["bin_NLL"]
        seed_positive = all(entries[str(seed)]["primary_better_direction"] for seed in SEEDS)
        pooled_ci_positive = entries["pooled"]["paired_source_bootstrap_95_ci"][0] > 0.
        add(f"bin_NLL shape sensitivity versus {control}", seed_positive and pooled_ci_positive,
            f"three seed directions positive={seed_positive}; pooled paired 95% CI lower bound > 0={pooled_ci_positive}")
    # A Pareto worsening means the primary has both lower empirical coverage
    # and greater mean width than a same-nominal interval control.
    for control in ("mixture", "median_mdn"):
        worsened = []
        for level in map(str, INTERVAL_LEVELS):
            coverage = comparisons[control]["intervals"][level]["coverage"]["pooled"]
            width = comparisons[control]["intervals"][level]["width"]["pooled"]
            if coverage["observed_delta_control_minus_primary"] > 0 and width["observed_delta_control_minus_primary"] < 0:
                worsened.append(level)
        add(f"coverage-width Pareto screen versus {control}", not worsened,
            "primary is not simultaneously lower-coverage and wider at any interval level" if not worsened
            else "primary has lower coverage and greater width at nominal levels " + ", ".join(worsened))
    return {"passed": all(item["passed"] for item in checks), "checks": checks,
            "interpretation": "Conservative registered gate; any failed required comparison rejects promotion."}


def audit(shape_suite_path: Path, point_suite_path: Path, config_path: Path,
          repo_root: Path, bootstrap_replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    rec = Recorder()
    report: dict[str, Any] = {
        "audit": "independent-cpu-native-nested-shape-probability-audit-v1",
        "shape_suite": str(shape_suite_path), "point_suite": str(point_suite_path),
        "registered_config": str(config_path), "checks": rec.checks, "findings": rec.findings,
        "numerical_replay_tolerance": NUMERIC_TOLERANCE,
        "complete": False, "promotion_gate_passed": False,
    }
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config_hash = sha256_file(config_path)
        plan_path = resolve_path(config["plan"], repo_root)
        plan_hash = sha256_file(plan_path)
        rec.must("registered shape config", config.get("dataset") == "simsv2"
                 and config.get("point_suite") == "runs/native_nested_shape_points_v1/suite.json"
                 and config.get("seeds") == list(SEEDS) and config.get("cases") == list(CASES)
                 and config.get("device") == "cpu" and config.get("hidden") == 32
                 and config.get("batch_size") == 64 and config.get("max_epochs") == 100
                 and config.get("patience") == 12 and config.get("primary_case") == PRIMARY
                 and config.get("official_VALID_used") is False and config.get("official_TEST_used") is False,
                 "Registered shape config differs from the frozen CPU/seed/case/training contract")
        plan_text = plan_path.read_text(encoding="utf-8").lower()
        plan_flat = " ".join(plan_text.split())
        rec.must("registered protocol statements", "5000 paired source/video bootstrap draws (seed20261013)" in plan_flat
                 and "no post-hoc degree selection" in plan_flat
                 and "outer sources enter no normalization, optimization or selection" in plan_flat,
                 "Protocol text does not retain the registered source bootstrap, selection, and no-outer-fit rules")
        point = audit_point_tables(point_suite_path, repo_root, rec)
        shape_suite = json.loads(shape_suite_path.read_text(encoding="utf-8"))
        rec.must("shape suite completion marker",
                 shape_suite.get("suite_complete") is True
                 and shape_suite.get("official_VALID_used") is False
                 and shape_suite.get("official_TEST_used") is False,
                 "Shape suite is incomplete or marks use of official VALID/TEST data")
        shape_rows = shape_suite.get("results", [])
        expected_keys = {(case, seed, fold) for case in CASES for seed in SEEDS for fold in FOLDS}
        actual_keys = {(row.get("case"), row.get("seed"), row.get("fold")) for row in shape_rows}
        rec.must("90-fit formal matrix", len(shape_rows) == 90 and actual_keys == expected_keys
                 and len(actual_keys) == len(shape_rows),
                 f"Shape suite has {len(shape_rows)} rows; expected exactly 90 unique case/seed/fold fits")

        records_by_fit: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        failures_before = len(rec.findings)
        for row in shape_rows:
            key = (str(row.get("case")), int(row.get("seed")), int(row.get("fold")))
            try:
                records_by_fit[key] = audit_one_fit(
                    row, case=key[0], seed=key[1], fold=key[2], config=config,
                    config_hash=config_hash, plan_hash=plan_hash, repo_root=repo_root,
                    point=point, rec=rec)
            except (AuditError, AssertionError, KeyError, OSError, ValueError, TypeError, IndexError) as exc:
                if not any(f.get("issue") == str(exc) for f in rec.findings):
                    rec.findings.append({"check": f"{key[0]} seed {key[1]} fold {key[2]} audit", "issue": str(exc)})
                records_by_fit.pop(key, None)

        canonical_ids = point["canonical_ids"]
        source_by_id = {sid: source_of(sid) for sid in canonical_ids}
        merged: dict[str, dict[int, list[dict[str, Any]]]] = {case: {} for case in CASES}
        for case in CASES:
            for seed in SEEDS:
                by_id: dict[str, dict[str, Any]] = {}
                for fold in FOLDS:
                    for row in records_by_fit.get((case, seed, fold), []):
                        if row["id"] in by_id:
                            raise AuditError(f"Outer forecast duplicate for {case}, seed {seed}: {row['id']}")
                        by_id[row["id"]] = row
                rec.must(f"merged outer coverage {case}/{seed}", set(by_id) == set(canonical_ids)
                         and len(by_id) == 2722,
                         f"Merged {case}, seed {seed} outer predictions do not cover every TRAIN ID exactly once")
                merged[case][seed] = [by_id[sid] for sid in canonical_ids]
        report["audited_fits"] = len(records_by_fit)
        report["audited_outer_rows"] = sum(len(rows) for rows in records_by_fit.values())
        report["canonical_train_ids"] = len(canonical_ids)
        report["source_videos"] = len(point["source_ids"])
        report["canonical_id_order_sha256"] = hashlib.sha256(
            json.dumps(canonical_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        report["raw_point_metric_note"] = (
            "Producer field native_raw_point is an alias for the author CH-SIMS metric, which clips predictions to [-1,1]. "
            "This audit separately reports MAE/Corr on the exported raw point without clipping; hit truth and event centers use that raw point."
        )
        report["per_case_seed_metrics"] = {
            case: {str(seed): _merged_summary(merged[case][seed]) for seed in SEEDS}
            for case in CASES
        }
        row_by_key = {(row["case"], int(row["seed"]), int(row["fold"])): row for row in shape_rows}
        merge_sources = {
            case: {
                str(seed): [str(resolve_path(row_by_key[(case, seed, fold)]["root"], repo_root)
                                / "outer_predictions.json") for fold in FOLDS]
                for seed in SEEDS
            }
            for case in CASES
        }
        report["merged_canonical_outputs"] = {
            "case_seed_merges": len(CASES) * len(SEEDS),
            "rows_per_merge": len(canonical_ids),
            "total_merged_rows": len(CASES) * len(SEEDS) * len(canonical_ids),
            "unique_ids_per_merge": len(canonical_ids),
            "source_videos_per_merge": len(point["source_ids"]),
            "outer_source_once_per_merge": True,
            "canonical_id_order_sha256": report["canonical_id_order_sha256"],
            "canonical_order_summary": "Canonical TRAIN-order merged metrics are embedded under per_case_seed_metrics.",
            "per_fold_prediction_files": merge_sources,
        }
        # The point baseline is common to all shapes. Check the raw point and
        # target fields remain identical after the independent canonical merge.
        rec.must("raw point invariant within each seed", same_seed_raw_point_invariant(merged),
                 "Raw point, native target, or clipped anchor changed across shape cases within a seed")
        bootstrap = _paired_source_bootstrap(merged, source_by_id, point["source_ids"], bootstrap_replicates)
        report["paired_source_bootstrap"] = bootstrap
        gate = _promotion_gate(bootstrap)
        report["promotion_gate"] = gate
        report["promotion_gate_passed"] = gate["passed"]
        report["checks_passed"] = sum(bool(item["passed"]) for item in rec.checks)
        report["checks_failed"] = sum(not bool(item["passed"]) for item in rec.checks)
        report["complete"] = (not rec.findings and len(records_by_fit) == 90
                              and report["audited_outer_rows"] == 30 * sum(
                                  len(point["tables"][(13, fold, "outer")]["ids"]) for fold in FOLDS))
        report["suite_complete"] = bool(report["complete"])
        report["recommendation"] = "reject promotion; retain the registered negative study" if not gate["passed"] else "promotion gate passed"
    except (AuditError, AssertionError, KeyError, OSError, ValueError, TypeError, IndexError, json.JSONDecodeError) as exc:
        if not any(item.get("issue") == str(exc) for item in rec.findings):
            rec.findings.append({"check": "audit setup", "issue": str(exc)})
        report["complete"] = False
        report["checks_passed"] = sum(bool(item["passed"]) for item in rec.checks)
        report["checks_failed"] = sum(not bool(item["passed"]) for item in rec.checks)
        report["error"] = str(exc)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suite", type=Path, help="Formal shape suite.json")
    parser.add_argument("--point-suite", type=Path, help="Formal frozen point suite.json")
    parser.add_argument("--config", type=Path, help="Registered shape config JSON")
    parser.add_argument("--report", type=Path, help="Output JSON report")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    shape_suite = resolve_path(args.suite or "runs/native_nested_shapes_v1/suite.json", repo_root)
    point_suite = resolve_path(args.point_suite or "runs/native_nested_shape_points_v1/suite.json", repo_root)
    config_path = resolve_path(args.config or "configs/native_nested_shapes_v1.json", repo_root)
    report_path = resolve_path(args.report or "docs/native_nested_shape_probability_audit_20261013.json", repo_root)
    if args.bootstrap_replicates != BOOTSTRAP_REPLICATES:
        parser.error(f"Bootstrap replicates are frozen at {BOOTSTRAP_REPLICATES}")
    result = audit(shape_suite, point_suite, config_path, repo_root, args.bootstrap_replicates)
    if "merged_canonical_outputs" in result:
        result["merged_canonical_outputs"]["report_path"] = str(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"complete": result.get("complete", False),
                      "audited_fits": result.get("audited_fits", 0),
                      "audited_outer_rows": result.get("audited_outer_rows", 0),
                      "promotion_gate_passed": result.get("promotion_gate_passed", False),
                      "recommendation": result.get("recommendation"),
                      "checks_failed": result.get("checks_failed", 0),
                      "findings": result.get("findings", [])[:20],
                      "report": str(report_path)}, indent=2, allow_nan=False))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())

