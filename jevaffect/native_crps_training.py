"""Source-nested native scalar-score CRPS fits on frozen point tables."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from jevaffect import cdf_crps, nested_shape_training as frozen, shape_controls as shapes
from jevaffect.anchor_calibration import (
    BOUNDARIES,
    TOLERANCE,
    bin_targets,
    diagnostics,
    event_metrics,
    normalizer,
)
from jevaffect.native_training import digest, save_checkpoint, save_json


CASES = ("scale", "mixture", "median_mdn", "bern_d6")
SEEDS = (13, 42, 2026)
FOLDS = (0, 1, 2)
TRAINING_NODES = 48
READOUT_NODES = (48, 96, 192)
CRPS_NODE_TOLERANCE = 1e-4
BIN_NEGATIVE_ROUNDOFF_TOLERANCE = 1e-12
BIN_SUM_ROUNDOFF_TOLERANCE = 1e-10
CDF_RANGE_ROUNDOFF_TOLERANCE = 1e-12
PLAN_PATH = Path(__file__).resolve().parents[1] / "docs" / "NATIVE_CDF_CRPS_PLAN_V1.md"


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_state_sha256(module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(name.encode("utf-8"))
        hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "native_crps_training.py",
        "cdf_crps.py",
        "nested_shape_training.py",
        "shape_controls.py",
        "nested_shape_points.py",
        "anchored_bernstein.py",
        "anchor_cdf.py",
        "anchor_calibration.py",
        "native_training.py",
        "native_metrics.py",
    )
    return {name: digest(root / name) for name in names}


def _run_hashes(config: dict) -> dict[str, str]:
    config_sha256 = config.get("config_sha256")
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        config_sha256 = _canonical_sha256(config)
    plan_sha256 = config.get("plan_sha256")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        plan_sha256 = digest(PLAN_PATH)
    return {"config_sha256": config_sha256, "plan_sha256": plan_sha256}


def _validate_training_contract(config: dict, smoke_only: bool) -> None:
    if config.get("device") != "cpu":
        raise ValueError("Native scalar-score CRPS fits are registered for CPU only")
    fixed = {
        "dataset": "simsv2",
        "dropout": 0.1,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_norm": 1.0,
        "patience": 12,
        "quadrature_nodes": 48,
        "readout_nodes": [48, 96, 192],
        "bin_negative_roundoff_tolerance": BIN_NEGATIVE_ROUNDOFF_TOLERANCE,
        "bin_sum_roundoff_tolerance": BIN_SUM_ROUNDOFF_TOLERANCE,
        "cdf_range_roundoff_tolerance": CDF_RANGE_ROUNDOFF_TOLERANCE,
        "quadrature_node_tolerance": CRPS_NODE_TOLERANCE,
        "device": "cpu",
        "primary_case": "bern_d6",
        "plan": "docs/NATIVE_CDF_CRPS_PLAN_V1.md",
        "official_VALID_used": False,
        "official_TEST_used": False,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Registered CRPS training setting changed: {key}")
    if config.get("cases") != list(CASES) or config.get("seeds") != list(SEEDS):
        raise ValueError("Registered native CRPS case and seed matrix required")
    if config.get("selection") != "first minimum shape_select CRPS_48":
        raise ValueError("Registered first-minimum shape_select CRPS selection required")
    if _run_hashes(config)["plan_sha256"] != digest(PLAN_PATH):
        raise ValueError("Registered native CRPS plan hash changed")
    if not smoke_only:
        for key, expected in {"point_suite": "runs/native_nested_shape_points_v1/suite.json",
                              "hidden": 32, "batch_size": 64, "max_epochs": 100, "torch_threads": 4}.items():
            if config.get(key) != expected:
                raise ValueError(f"Registered CRPS training setting changed: {key}")


def _bins_from_cdf_values(values: torch.Tensor):
    if values.ndim != 2 or values.shape[1] != 6 or not torch.isfinite(values).all():
        raise ValueError("Finite native CDF values at both bounds and four bin edges required")
    minimum, maximum = float(values.min()), float(values.max())
    if minimum < -CDF_RANGE_ROUNDOFF_TOLERANCE or maximum > 1 + CDF_RANGE_ROUNDOFF_TOLERANCE:
        raise ValueError("Native CDF values exceed the roundoff clamp bound")
    bounded = values.clamp(0.0, 1.0)
    correction = float((bounded - values).abs().max())
    return bounded[:, 1:] - bounded[:, :-1], correction


def _crps_loss(output, target: torch.Tensor) -> torch.Tensor:
    values = cdf_crps.per_sample(output, target, nodes=TRAINING_NODES)
    if values.ndim != 1 or values.shape != target.shape or not torch.isfinite(values).all():
        raise ValueError("Invalid per-row native CRPS objective")
    return values.mean()


@torch.no_grad()
def selection(model, batches) -> float:
    """Mean 48-node CRPS over shape_select, with dropout disabled."""
    model.eval()
    total, count = 0.0, 0
    for x, anchor, _, target in batches:
        values = cdf_crps.per_sample(model(x, anchor), target, nodes=TRAINING_NODES)
        if values.shape != target.shape or not torch.isfinite(values).all():
            raise ValueError("Nonfinite shape_select native CRPS")
        total += float(values.sum())
        count += len(target)
    if count == 0:
        raise ValueError("shape_select must contain rows")
    return total / count


def _raw_point_metrics(point: np.ndarray, target: np.ndarray) -> dict:
    raw = np.asarray(point, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if raw.shape != y.shape or raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all() or not np.isfinite(y).all():
        raise ValueError("Finite native point and target rows required")
    corr = None
    if len(y) > 1 and np.std(raw) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(raw, y)[0, 1])
    return {"count": int(len(y)), "MAE": float(np.mean(np.abs(raw - y))), "Corr": corr}


def _common_bin_probability_readout(output):
    """Form five bins after validating then clamping only CDF roundoff."""
    parameters = output["parameters"]
    edges = torch.as_tensor(BOUNDARIES, dtype=torch.float64, device=parameters["anchor"].device)
    values = shapes.cdf(output, edges[None].expand(len(parameters["anchor"]), -1))
    values = torch.cat((torch.zeros_like(values[:, :1]), values, torch.ones_like(values[:, :1])), dim=1)
    return _bins_from_cdf_values(values)


def _common_bin_probabilities(output) -> torch.Tensor:
    return _common_bin_probability_readout(output)[0]


def _repair_bin_probabilities(probabilities: np.ndarray):
    """Correct float64 edge-subtraction roundoff, rejecting real CDF errors."""
    raw = np.asarray(probabilities, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Finite five-bin native probabilities required")
    minimum = float(raw.min())
    sum_error = float(np.max(np.abs(raw.sum(1) - 1.0)))
    if minimum < -BIN_NEGATIVE_ROUNDOFF_TOLERANCE or sum_error > BIN_SUM_ROUNDOFF_TOLERANCE:
        raise ValueError("Native bin probabilities exceed the roundoff repair bound")
    clipped = np.maximum(raw, 0.0)
    row_sums = clipped.sum(1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Native bin probabilities have a zero row sum")
    repaired = clipped / row_sums
    maximum_correction = float(np.max(np.abs(repaired - raw)))
    if maximum_correction > BIN_SUM_ROUNDOFF_TOLERANCE:
        raise ValueError("Native bin correction exceeds the registered roundoff bound")
    audit = {
        "negative_entry_tolerance": BIN_NEGATIVE_ROUNDOFF_TOLERANCE,
        "row_sum_tolerance": BIN_SUM_ROUNDOFF_TOLERANCE,
        "minimum_before_repair": minimum,
        "maximum_absolute_row_sum_error_before_repair": sum_error,
        "maximum_absolute_correction": maximum_correction,
        "repaired_rows": int(np.count_nonzero(np.any(repaired != raw, axis=1))),
    }
    return repaired, audit


@torch.no_grad()
def _quadrature_readout(model, data: dict, stats: dict, batch_size: int):
    values = {node: [] for node in READOUT_NODES}
    native_hit_probability = []
    for x, anchor, _, target in frozen.loader(data, stats, batch_size):
        output = model(x, anchor)
        native_hit_probability.append(
            shapes.hit_probability(output, anchor, TOLERANCE).detach().cpu().numpy()
        )
        for node in READOUT_NODES:
            per_row = cdf_crps.per_sample(output, target, nodes=node)
            values[node].append(per_row.detach().cpu().numpy())
    joined = {node: np.concatenate(parts).astype(np.float64, copy=False) for node, parts in values.items()}
    hit_probability = np.concatenate(native_hit_probability).astype(np.float64, copy=False)
    if any(len(score) != len(data["ids"]) for score in joined.values()) or len(hit_probability) != len(data["ids"]):
        raise ValueError("Outer CRPS readout lost frozen table rows")
    return joined, hit_probability


@torch.no_grad()
def predict(model, data: dict, stats: dict, batch_size: int):
    """Keep the common frozen readout fields and add native-CRPS diagnostics."""
    model.eval()
    probabilities, raw_hits, interval_rows, medians, trapezoid_scores = [], [], [], [], []
    cdf_clamp_corrections = []
    levels = [.25, .75, .1, .9, .05, .95, .025, .975, .5]
    grid = torch.linspace(-1.0, 1.0, 1001, dtype=torch.float64)
    for x, anchor_batch, point_batch, target_batch in frozen.loader(data, stats, batch_size):
        output = model(x, anchor_batch)
        bin_probabilities, cdf_correction = _common_bin_probability_readout(output)
        probabilities.append(bin_probabilities.cpu().numpy())
        cdf_clamp_corrections.append(cdf_correction)
        raw_hits.append(shapes.hit_probability(output, point_batch, TOLERANCE).cpu().numpy())
        quantile_values = shapes.quantiles(output, levels).cpu().numpy()
        interval_rows.append(quantile_values[:, :8])
        medians.append(quantile_values[:, 8])
        cdf_values = shapes.cdf(output, grid[None].expand(len(target_batch), -1))
        truth = (grid[None] >= target_batch.double()[:, None]).double()
        trapezoid_scores.append(torch.trapezoid((cdf_values - truth).square(), grid, dim=1).cpu().numpy())

    p, bin_roundoff = _repair_bin_probabilities(np.concatenate(probabilities))
    bin_roundoff["cdf_range_tolerance"] = CDF_RANGE_ROUNDOFF_TOLERANCE
    bin_roundoff["maximum_CDF_clamp_correction"] = max(cdf_clamp_corrections, default=0.0)
    raw_probability = np.concatenate(raw_hits)
    q = np.concatenate(interval_rows)
    median = np.concatenate(medians)
    trapezoid = np.concatenate(trapezoid_scores)
    point = np.asarray(data["native_point"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    if p.shape != (len(target), 5) or not np.isfinite(p).all():
        raise ValueError("Invalid common native-bin forecasts")
    metrics = diagnostics(point, target, {
        "bins": p,
        "p_hit": raw_probability,
        "intervals": {
            level: (q[:, 2 * index], q[:, 2 * index + 1])
            for index, level in enumerate((.5, .8, .9, .95))
        },
    })
    metrics["native_author_point"] = metrics.pop("native")
    metrics["CRPS_trapezoid_1001"] = float(trapezoid.mean())
    metrics["anchor_clipped_count"] = int(np.count_nonzero(point != data["anchor"]))
    metrics["native_bin_probability_roundoff"] = bin_roundoff
    if model.case != "mdn" and not np.array_equal(median, data["anchor"].astype(np.float64)):
        raise ValueError("Fixed probabilistic median moved")

    labels = bin_targets(target)
    hit_truth = (np.abs(point - target) <= np.float32(.1)).astype(np.float64)
    nll = -np.log(np.maximum(p[np.arange(len(p)), labels], 1e-12))
    clipped_raw_probability = np.clip(raw_probability, 1e-12, 1 - 1e-12)
    records = []
    for i, sid in enumerate(data["ids"]):
        interval_values, interval_scores = {}, {}
        for j, nominal in enumerate((.5, .8, .9, .95)):
            lo, hi = q[i, 2 * j:2 * j + 2]
            alpha = 1 - nominal
            interval_values[str(nominal)] = [float(lo), float(hi)]
            interval_scores[str(nominal)] = float(
                hi - lo + 2 / alpha * (max(lo - float(target[i]), 0) + max(float(target[i]) - hi, 0))
            )
        records.append({
            "id": sid,
            "target": float(target[i]),
            "native_point": float(point[i]),
            "anchor": float(data["anchor"][i]),
            "median": float(median[i]),
            "bin_probabilities": p[i].tolist(),
            "raw_hit_probability": float(raw_probability[i]),
            "intervals": interval_values,
            "interval_scores": interval_scores,
            "bin_NLL": float(nll[i]),
            "bin_Brier": float(np.sum((p[i] - np.eye(5)[labels[i]]) ** 2)),
            "raw_hit_Brier": float((raw_probability[i] - hit_truth[i]) ** 2),
            "raw_hit_log_score": float(-hit_truth[i] * np.log(clipped_raw_probability[i])
                                        - (1 - hit_truth[i]) * np.log1p(-clipped_raw_probability[i])),
            "CRPS_trapezoid_1001": float(trapezoid[i]),
        })
    metrics["interval_scores"] = {
        str(level): float(np.mean([row["interval_scores"][str(level)] for row in records]))
        for level in (.5, .8, .9, .95)
    }
    score_rows, native_probability = _quadrature_readout(model, data, stats, batch_size)

    point = np.asarray(data["native_point"], dtype=np.float32)
    target = np.asarray(data["target"], dtype=np.float32)
    anchor = np.asarray(data["anchor"], dtype=np.float32)
    clipped_point = np.clip(point, -1.0, 1.0).astype(np.float32)
    if not np.array_equal(anchor, clipped_point):
        raise ValueError("Frozen anchor must equal the clipped native point")
    tolerance = np.float32(TOLERANCE)
    native_hit = (np.abs(clipped_point - target) <= tolerance).astype(np.float64)
    raw_hit = (np.abs(point - target) <= tolerance).astype(np.float64)

    raw_probability = np.asarray([row["raw_hit_probability"] for row in records], dtype=np.float64)
    raw_error = np.abs(point.astype(np.float64) - target.astype(np.float64))
    native_error = np.abs(clipped_point.astype(np.float64) - target.astype(np.float64))
    native_event = event_metrics(native_probability, native_hit, native_error)
    raw_event = event_metrics(raw_probability, raw_hit, raw_error)

    metrics["raw_unclipped_point"] = _raw_point_metrics(point, target)
    metrics["raw_unclipped_MAE"] = metrics["raw_unclipped_point"]["MAE"]
    metrics["raw_unclipped_Corr"] = metrics["raw_unclipped_point"]["Corr"]

    metrics["native_hit_count"] = native_event["hit_count"]
    metrics["native_hit_Brier"] = native_event["hit_Brier"]
    metrics["native_hit_log_score"] = native_event["hit_log_score"]
    metrics["native_hit_ECE15"] = native_event["hit_ECE15"]
    metrics["native_hit_reliability"] = native_event["reliability"]
    metrics["raw_hit_count"] = raw_event["hit_count"]
    metrics["raw_hit_Brier"] = raw_event["hit_Brier"]
    metrics["raw_hit_log_score"] = raw_event["hit_log_score"]
    metrics["raw_hit_ECE15"] = raw_event["hit_ECE15"]
    metrics["raw_hit_reliability"] = raw_event["reliability"]
    for key in ("hit_count", "hit_Brier", "hit_log_score", "hit_ECE15", "reliability", "risk_AURC", "risk_at_coverage"):
        metrics.pop(key, None)

    for row_index, row in enumerate(records):
        row["native_hit_probability"] = float(native_probability[row_index])
        row["native_hit_truth"] = int(native_hit[row_index])
        row["native_hit_Brier"] = float((native_probability[row_index] - native_hit[row_index]) ** 2)
        probability = float(np.clip(native_probability[row_index], 1e-12, 1 - 1e-12))
        truth = native_hit[row_index]
        row["native_hit_log_score"] = float(-truth * np.log(probability) - (1 - truth) * np.log1p(-probability))
        row["raw_hit_truth"] = int(raw_hit[row_index])
        for node in READOUT_NODES:
            row[f"CRPS_{node}"] = float(score_rows[node][row_index])

    metrics["CRPS_by_quadrature_nodes"] = {
        str(node): float(score_rows[node].mean()) for node in READOUT_NODES
    }
    metrics["CRPS_48"] = metrics["CRPS_by_quadrature_nodes"]["48"]
    discrepancies = {}
    for lower, upper in ((48, 96), (96, 192), (48, 192)):
        difference = score_rows[lower] - score_rows[upper]
        discrepancies[f"{lower}_vs_{upper}"] = {
            "mean_score_difference": float(difference.mean()),
            "mean_absolute_row_difference": float(np.abs(difference).mean()),
            "maximum_absolute_row_difference": float(np.abs(difference).max()),
        }
    metrics["CRPS_quadrature_discrepancies"] = discrepancies
    max_mean_absolute = max(row["mean_absolute_row_difference"] for row in discrepancies.values())
    metrics["CRPS_quadrature_gate"] = {
        "tolerance": CRPS_NODE_TOLERANCE,
        "maximum_mean_absolute_row_difference": float(max_mean_absolute),
        "pass": bool(max_mean_absolute <= CRPS_NODE_TOLERANCE),
    }
    return metrics, records


def train(config: dict, tables: dict, binding: dict, *, case: str, seed: int, fold: int,
          resume: bool = False, stop_after: int | None = None):
    if case not in CASES or seed not in SEEDS or fold not in FOLDS or config.get("device") != "cpu":
        raise ValueError("Registered case, seed, fold and CPU CRPS runtime required")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if set(tables) != {"shape_fit", "shape_select", "outer"}:
        raise ValueError("Exactly the frozen shape_fit, shape_select and outer tables are required")
    smoke_only = bool(binding.get("smoke_only"))
    _validate_training_contract(config, smoke_only)

    torch.set_num_threads(int(config["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fit, select = tables["shape_fit"], tables["shape_select"]
    fit_context = frozen.context(fit)
    stats = normalizer(fit_context, np.arange(len(fit["ids"])))
    model = shapes.ShapeHead(case, fit_context.shape[1], config["hidden"], config["dropout"])
    encoder_initial_sha256 = _module_state_sha256(model.context)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    fit_loader = frozen.loader(fit, stats, config["batch_size"], generator=generator)
    select_loader = frozen.loader(select, stats, config["batch_size"])

    hashes = _run_hashes(config)
    identity = {
        "config": config,
        **hashes,
        "case": case,
        "seed": seed,
        "fold": fold,
        "binding": binding,
        "normalization": stats,
        "fit_ids": fit["ids"],
        "select_ids": select["ids"],
        "implementation_sha256": _implementation_hashes(),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__},
        "initial_encoder_sha256": encoder_initial_sha256,
        "training_objective": "final_censored_CDF_CRPS_Gauss_Legendre_48",
        "selection": "first minimum shape_select CRPS_48",
        "outer_used_for_selection": False,
    }
    root = Path(config["run_root"]) / ("smoke" if smoke_only else "formal") / case / f"seed_{seed}" / f"fold_{fold}"
    root.mkdir(parents=True, exist_ok=True)
    last = root / "last.pt"
    epoch0, best, stale, history = 0, float("inf"), 0, []
    if last.exists():
        if not resume:
            raise ValueError("CRPS checkpoint exists; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("CRPS checkpoint identity changed")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        epoch0, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        generator.set_state(state["loader_rng"])
    save_json(root / "identity.json", identity)

    cap = min(config["max_epochs"], epoch0 + stop_after) if stop_after is not None else config["max_epochs"]
    for epoch in range(epoch0, cap):
        if stale >= config["patience"]:
            break
        started = time.perf_counter()
        model.train()
        total, count, updates = 0.0, 0, 0
        for x, anchor, _, target in fit_loader:
            opt.zero_grad(set_to_none=True)
            objective = _crps_loss(model(x, anchor), target)
            if not torch.isfinite(objective):
                raise ValueError("Nonfinite native scalar-score CRPS objective")
            objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Inactive or nonfinite CRPS model gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm"])
            opt.step()
            total += float(objective.detach()) * len(target)
            count += len(target)
            updates += 1
        score = selection(model, select_loader)
        history.append({
            "epoch": epoch + 1,
            "fit_CRPS_48": total / count,
            "selection_CRPS_48": score,
            "updates": updates,
            "seconds": time.perf_counter() - started,
        })
        if score < best:
            best, stale = score, 0
            save_checkpoint(root / "best.pt", {
                "identity": identity,
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "selection_CRPS_48": score,
            })
        else:
            stale += 1
        save_checkpoint(last, {
            "identity": identity,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch + 1,
            "best": best,
            "stale": stale,
            "history": history,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "loader_rng": generator.get_state(),
        })
        save_json(root / "history.json", history)

    complete = stale >= config["patience"] or len(history) >= config["max_epochs"]
    if not (root / "best.pt").exists():
        raise ValueError("No selected CRPS checkpoint exists yet")
    selected = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"])
    reproduced = selection(model, select_loader)
    if reproduced != selected["selection_CRPS_48"]:
        raise ValueError("Selected shape_select CRPS is not reproducible")

    metrics, records = predict(model, tables["outer"], stats, config["batch_size"]) if complete else (None, None)
    if complete:
        save_json(root / "outer_predictions.json", records)
    result = {
        "case": case,
        "seed": seed,
        "fold": fold,
        "root": str(root),
        "complete": bool(complete),
        "smoke_only": smoke_only,
        "best_epoch": selected["epoch"],
        "best_selection_CRPS_48": selected["selection_CRPS_48"],
        "epochs": len(history),
        "updates": sum(row["updates"] for row in history),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loop_seconds": sum(row["seconds"] for row in history),
        "initial_encoder_sha256": encoder_initial_sha256,
        "outer": metrics,
        "quadrature_gate_pass": bool(metrics["CRPS_quadrature_gate"]["pass"]) if complete else None,
        "best_sha256": digest(root / "best.pt"),
        "last_sha256": digest(last),
        "history_sha256": digest(root / "history.json"),
        "predictions_sha256": digest(root / "outer_predictions.json") if complete else None,
        "implementation_sha256": identity["implementation_sha256"],
        "config_sha256": hashes["config_sha256"],
        "plan_sha256": hashes["plan_sha256"],
        "point_binding": binding,
        "official_VALID_used": False,
        "official_TEST_used": False,
    }
    save_json(root / "result.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result
