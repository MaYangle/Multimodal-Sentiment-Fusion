"""Audit formal source-risk baselines and paired TRAIN OOF diagnostics on CPU.

The audit does not optimize, use official VALID/TEST, or claim a new method.
It verifies all 54 source-fold teachers, reloads selected checkpoints on CPU,
reconstructs OOF forecasts, checks the two registered parity controls, and
reports source-cluster development comparisons.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.anchor_crossfit import (  # noqa: E402
    PooledView,
    _loader,
    _predict_targets,
    _read_predictions,
    prepare,
    _write_predictions,
)
from jevaffect.native_metrics import native_metrics  # noqa: E402
from jevaffect.native_training import digest, save_json  # noqa: E402
from jevaffect.source_crossfit import (  # noqa: E402
    group_of,
    make_folds,
    manifest_hash,
    subset,
    training_rows_only,
    validate_folds,
)
from jevaffect.source_robust import SourceRiskModel  # noqa: E402
from jevaffect.source_robust_training import fit_source_groups  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
CASES = ("clip_erm", "balanced_erm", "dro_uniform", "group_dro", "domain_probe", "dann")
SEEDS = (13, 42, 2026)
FOLDS = (0, 1, 2)
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261013
POINT_PARAMETERS = 492033
PREDICTION_TOLERANCE = 1e-6


def replay_scheduler(history,config):
    """Plateau scheduler tracks thresholded improvement, not exact selection min."""
    parameter=torch.nn.Parameter(torch.zeros(()))
    optimizer=torch.optim.AdamW([parameter],lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=config["scheduler_patience"],factor=config["scheduler_factor"])
    for row in history:
        scheduler.step(row["inner_select"]["MAE"])
        if optimizer.param_groups[0]["lr"]!=row["learning_rate"]:
            raise ValueError("Epoch learning rate differs from actual plateau scheduling")
    return scheduler.state_dict()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolve(path, base=PROJECT_ROOT):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (Path(base) / value).resolve()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "target", "prediction"]:
            raise ValueError(f"Unexpected source OOF columns: {path}")
        output = []
        for row in reader:
            target, prediction = float(row["target"]), float(row["prediction"])
            if not math.isfinite(target) or not math.isfinite(prediction):
                raise ValueError(f"Nonfinite OOF output: {path}")
            output.append({"id": row["id"], "target": target, "prediction": prediction})
        return output


def compare_numeric(left, right, tolerance=0.0, path="value"):
    """Recursively compare source records, histories, and model diagnostics."""
    if isinstance(right, dict):
        if not isinstance(left, dict) or set(left) != set(right):
            raise ValueError(f"Mapping keys differ at {path}")
        for key in right:
            compare_numeric(left[key], right[key], tolerance, f"{path}.{key}")
    elif isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or len(left) != len(right):
            raise ValueError(f"Sequence shape differs at {path}")
        for index, (a, b) in enumerate(zip(left, right)):
            compare_numeric(a, b, tolerance, f"{path}[{index}]")
    elif isinstance(right, bool) or right is None or isinstance(right, str):
        if left != right:
            raise ValueError(f"Value differs at {path}: {left} != {right}")
    elif isinstance(right, (int, float, np.number)):
        if not math.isfinite(float(left)) or not math.isclose(float(left), float(right), rel_tol=0,
                                                               abs_tol=tolerance):
            raise ValueError(f"Numeric value differs at {path}: {left} != {right}")
    elif left != right:
        raise ValueError(f"Value differs at {path}: {left} != {right}")


def merge_oof_records(fold_records, folds, canonical_ids, target_by_id):
    """Merge the three held-source folds and reject missing/duplicate/wrong targets."""
    ids = list(canonical_ids)
    if set(fold_records) != set(FOLDS) or len(ids) != len(set(ids)):
        raise ValueError("OOF merge requires canonical unique TRAIN IDs and folds 0/1/2")
    validate_folds(ids, folds)
    by_id = {}
    held_groups = {}
    for fold in FOLDS:
        expected = list(folds[fold]["oof_ids"])
        records = list(fold_records[fold])
        got = [row["id"] for row in records]
        if len(got) != len(set(got)) or set(got) != set(expected):
            raise ValueError(f"Fold {fold} OOF IDs are missing, extra, or duplicated")
        for row in records:
            sid = row["id"]
            source = group_of(sid)
            previous = held_groups.setdefault(source, fold)
            if previous != fold:
                raise ValueError("A source group appears in multiple held-source folds")
            if sid in by_id:
                raise ValueError(f"Duplicate OOF ID {sid}")
            if sid not in target_by_id or np.float32(row["target"]) != np.float32(target_by_id[sid]):
                raise ValueError(f"OOF target differs from original float32 M for {sid}")
            if not math.isfinite(float(row["prediction"])):
                raise ValueError(f"Nonfinite OOF prediction for {sid}")
            by_id[sid] = {"id": sid, "target": float(np.float32(target_by_id[sid])),
                          "prediction": float(row["prediction"])}
    if set(by_id) != set(ids) or len(by_id) != len(ids):
        raise ValueError("OOF rows do not cover canonical TRAIN exactly once")
    if len(held_groups) != len({group_of(sid) for sid in ids}):
        raise ValueError("Not every source group is held out exactly once")
    return [by_id[sid] for sid in ids]


def validate_epoch_accounting(history, fit_count, batch_size, case, source_counts,
                              max_epochs, patience):
    if not history:
        raise ValueError("Formal teacher has no epoch history")
    steps = math.ceil(fit_count / batch_size)
    expected_source_count = len(source_counts)
    for index, row in enumerate(history, 1):
        if row.get("epoch") != index or row.get("optimizer_updates") != steps:
            raise ValueError("Epoch/update accounting differs from ceil(fit_count / batch_size)")
        sampled = row.get("sampled_source_counts")
        if (not isinstance(sampled, list) or len(sampled) != expected_source_count
                or any(not isinstance(value, int) or value < 0 for value in sampled)
                or sum(sampled) != fit_count):
            raise ValueError("Sampled source counts do not disclose exactly one fit epoch")
        if case == "clip_erm" and sampled != list(source_counts):
            raise ValueError("Clip ERM must visit each fit clip exactly once per epoch")
        q = np.asarray(row.get("source_probability"), np.float64)
        if q.shape != (expected_source_count,) or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError("Invalid epoch source-risk probability vector")
        if not np.isclose(q.sum(), 1., rtol=0, atol=1e-10):
            raise ValueError("Epoch source-risk probabilities do not sum to one")
        if case != "group_dro" and not np.allclose(q, np.full(expected_source_count, 1 / expected_source_count),
                                                   rtol=0, atol=1e-12):
            raise ValueError(f"{case} must retain fixed uniform source weights")
        if case in ("domain_probe", "dann"):
            if not math.isfinite(float(row["domain_CE"])) or row["domain_CE"] <= 0:
                raise ValueError(f"{case} must report a positive source-classifier CE")
        elif float(row["domain_CE"]) != 0:
            raise ValueError(f"{case} must not train the source classifier")
        if case in ("clip_erm", "balanced_erm", "domain_probe", "dann") and not math.isclose(
            float(row["train_clip_L1"]), float(row["risk_objective"]), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(f"{case} point objective does not match the batch-mean L1 reference")
        expected_reversal = 0.0
        if case == "dann":
            last_fraction = (((index - 1) * steps + steps - 1)
                             / max(1, max_epochs * steps - 1))
            expected_reversal = .1 * (2 / (1 + math.exp(-10 * last_fraction)) - 1)
        if not math.isclose(float(row["reversal_scale_end"]), expected_reversal,
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{case} reversal schedule differs from the registered fixed budget")
        for metric in ("train_clip_L1", "risk_objective", "domain_CE", "learning_rate", "seconds"):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"Nonfinite epoch statistic {metric}")
    return steps


def first_minimum_mae(history):
    values = [float(row["inner_select"]["MAE"]) for row in history]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Inner-selection MAE history is empty or nonfinite")
    index = min(range(len(values)), key=values.__getitem__)
    return int(history[index]["epoch"]), values[index], history[index]


def source_diagnostics(records, *, minimum_worst_source_clips=5):
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)) or not records:
        raise ValueError("Source diagnostics require nonempty unique OOF IDs")
    target = np.asarray([row["target"] for row in records], np.float32)
    prediction = np.asarray([row["prediction"] for row in records], np.float32)
    clipped = np.clip(prediction, -1., 1.)
    error = clipped - target
    source_to_indices = {}
    for index, sid in enumerate(ids):
        source_to_indices.setdefault(group_of(sid), []).append(index)
    source_mae = {}
    source_error_mean = {}
    source_sizes = {}
    between_numerator = within_numerator = 0.0
    for source, indices_list in source_to_indices.items():
        indices = np.asarray(indices_list, np.int64)
        local_error = error[indices].astype(np.float64)
        mean = float(local_error.mean())
        source_mae[source] = float(np.abs(local_error).mean())
        source_error_mean[source] = mean
        source_sizes[source] = len(indices)
        between_numerator += len(indices) * mean * mean
        within_numerator += float(np.sum((local_error - mean) ** 2))
    eligible = [source for source, count in source_sizes.items() if count >= minimum_worst_source_clips]
    if not eligible:
        raise ValueError("No source meets the minimum clip count for worst-source MAE")
    total_mse = float(np.mean(error.astype(np.float64) ** 2))
    between = between_numerator / len(ids)
    within = within_numerator / len(ids)
    if not math.isclose(total_mse, between + within, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Clipped MSE source/within decomposition does not sum")
    return {
        "source_count": len(source_to_indices),
        "source_equal_mean_MAE": float(np.mean(list(source_mae.values()))),
        "worst_eligible_source_MAE": max(source_mae[s] for s in eligible),
        "worst_eligible_source": max(eligible, key=source_mae.__getitem__),
        "worst_source_minimum_clips": minimum_worst_source_clips,
        "worst_source_clip_count": source_sizes[max(eligible, key=source_mae.__getitem__)],
        "source_sizes_min": min(source_sizes.values()),
        "source_sizes_max": max(source_sizes.values()),
        "clipped_error_total_MSE": total_mse,
        "clipped_error_source_between_MSE": float(between),
        "clipped_error_within_source_MSE": float(within),
        "clipped_error_source_signed_mean": float(np.mean(list(source_error_mean.values()))),
    }


def paired_source_bootstrap(delta_by_seed, ids, *, replicates=BOOTSTRAP_REPS,
                            random_seed=BOOTSTRAP_SEED):
    """Paired clip-weighted bootstrap that resamples whole source groups."""
    if replicates < 2000:
        raise ValueError("At least 2,000 whole-source bootstrap draws are required")
    ids = list(ids)
    if len(ids) != len(set(ids)) or not ids or set(delta_by_seed) != set(SEEDS):
        raise ValueError("Bootstrap requires canonical IDs and all three seeds")
    sources = sorted({group_of(sid) for sid in ids})
    group_index = {source: i for i, source in enumerate(sources)}
    row_group = np.asarray([group_index[group_of(sid)] for sid in ids], np.int64)
    counts = np.bincount(row_group, minlength=len(sources)).astype(np.float64)
    values = {}
    for seed in SEEDS:
        delta = np.asarray(delta_by_seed[seed], np.float64)
        if delta.shape != (len(ids),) or not np.isfinite(delta).all():
            raise ValueError("Bootstrap loss differences must be finite and canonical-ID aligned")
        values[seed] = delta
    draws = np.random.default_rng(random_seed).integers(0, len(sources),
                                                       size=(replicates, len(sources)), dtype=np.int32)
    denom = counts[draws].sum(1)
    per_seed, distributions = {}, []
    for seed in SEEDS:
        cluster_sums = np.bincount(row_group, weights=values[seed], minlength=len(sources))
        boot = cluster_sums[draws].sum(1) / denom
        distributions.append(boot)
        lower, upper = np.percentile(boot, (2.5, 97.5))
        per_seed[str(seed)] = {"estimate": float(values[seed].mean()),
                               "ci95": [float(lower), float(upper)],
                               "replicates": replicates}
    pooled = np.mean(np.stack(distributions, axis=0), axis=0)
    lower, upper = np.percentile(pooled, (2.5, 97.5))
    return {
        "by_seed": per_seed,
        "three_seed_same_source_average": {
            "estimate": float(np.mean([values[seed].mean() for seed in SEEDS])),
            "ci95": [float(lower), float(upper)],
            "replicates": replicates,
            "shared_source_draws_across_seeds": True,
        },
        "source_count": len(sources),
        "replicates": replicates,
        "random_seed": random_seed,
        "direction": "candidate minus comparator MAE; negative favors the candidate",
    }


def compare_point_tensors(source_state, reference_state, *, point_prefix="regressor."):
    if any(key.startswith(point_prefix) for key in reference_state):
        reference_state = {key[len(point_prefix):]: value for key, value in reference_state.items()
                           if key.startswith(point_prefix)}
    point = {key[len(point_prefix):]: value for key, value in source_state.items()
             if key.startswith(point_prefix)}
    if set(point) != set(reference_state):
        raise ValueError("Point model state-dict keys differ")
    differences = {}
    for key in reference_state:
        left, right = point[key].detach().cpu(), reference_state[key].detach().cpu()
        if left.shape != right.shape:
            raise ValueError(f"Point state tensor shape differs: {key}")
        differences[key] = float((left - right).abs().max()) if left.numel() else 0.0
        if not torch.equal(left, right):
            raise ValueError(f"Point model tensor is not exactly equal: {key}")
    return {"exact": True, "max_abs_delta": max(differences.values(), default=0.0),
            "tensor_count": len(differences)}


def compare_prediction_rows(left, right, *, exact=False, tolerance=PREDICTION_TOLERANCE):
    if len(left) != len(right):
        raise ValueError("OOF prediction row counts differ")
    max_delta = 0.0
    for index, (a, b) in enumerate(zip(left, right)):
        if a["id"] != b["id"]:
            raise ValueError(f"OOF ID order differs at row {index}")
        if np.float32(a["target"]) != np.float32(b["target"]):
            raise ValueError(f"OOF targets differ at {a['id']}")
        delta = abs(float(a["prediction"]) - float(b["prediction"]))
        max_delta = max(max_delta, delta)
        prediction_differs = (float(a["prediction"]) != float(b["prediction"])) if exact else delta > tolerance
        if prediction_differs:
            raise ValueError(f"OOF predictions differ at {a['id']}: delta={delta}")
    return {"exact": max_delta == 0, "max_abs_delta": max_delta, "count": len(left)}


def _feature_hashes_equal(actual, expected):
    # NativeSplit's metadata/id keys may be absolute or cache-relative depending
    # on the caller's working directory; compare key suffix plus content hash.
    def normalized(mapping):
        out = {}
        for key, value in mapping.items():
            text = str(key).replace("\\", "/")
            base = Path(text).name
            if base in ("meta.json", "metadata.json"):
                path_key = base
            elif text.endswith("/ids.json"):
                parts = text.split("/")
                path_key = "/".join(parts[-2:])
            else:
                parts = text.split("/")
                candidates = ["/".join(parts[i:]) for i, item in enumerate(parts)
                              if item in ("train", "valid", "splits")]
                path_key = candidates[-1] if candidates else text
            if path_key in out:
                raise ValueError("Feature hash keys collide after normalization")
            out[path_key] = value
        return out
    return normalized(actual) == normalized(expected)


def _read_torch(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _source_rows(records, train_target_by_id):
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("OOF records contain duplicate IDs")
    for row in records:
        if row["id"] not in train_target_by_id or np.float32(row["target"]) != np.float32(train_target_by_id[row["id"]]):
            raise ValueError(f"OOF target differs from TRAIN-only author M for {row['id']}")


def _fit_identity_cohort(identity, data, case, seed, fold, cohort, config):
    if identity.get("config") != config or identity.get("case") != case or identity.get("seed") != seed:
        raise ValueError("Source teacher config/case/seed identity mismatch")
    if identity.get("fold") != fold or identity.get("cohort") != cohort:
        raise ValueError("Source teacher fold cohort identity mismatch")
    if identity.get("rng_seed") != seed + 10000 * fold:
        raise ValueError("Source teacher RNG seed differs from the registered schedule")
    if identity.get("fold_manifest_sha256") != manifest_hash(data["fold_manifest"]):
        raise ValueError("Source teacher fold-manifest hash mismatch")
    if identity.get("author_csv_sha256") != data["author_csv_sha256"]:
        raise ValueError("Source teacher CSV hash mismatch")
    if identity.get("train_ids") != data["ids"]:
        raise ValueError("Source teacher canonical TRAIN IDs changed")
    if not _feature_hashes_equal(identity.get("feature_hashes", {}), data["train"].hashes):
        raise ValueError("Source teacher feature hashes changed")
    if (identity.get("smoke_only") is not False or identity.get("fixture_smoke") is not False
            or identity.get("smoke_groups") is not None):
        raise ValueError("Formal source teacher is marked as smoke/fixture")
    if identity.get("official_VALID_used") is not False or identity.get("test_evaluated") is not False:
        raise ValueError("Source teacher reports forbidden VALID/TEST access")
    domains, groups, counts = fit_source_groups(cohort["fit_ids"])
    if identity.get("domains") != domains or identity.get("domain_counts") != counts.tolist():
        raise ValueError("Source teacher domain map/counts differ from the fit IDs")
    if identity.get("feature_dimensions") != {m: int(data["train"].features[m].shape[-1])
                                               for m in ("text", "audio", "vision")}:
        raise ValueError("Source teacher feature dimensions changed")
    return domains, groups, counts


def _compare_history_fields(left, right, fields, *, exact=True, tolerance=0):
    if len(left) != len(right):
        raise ValueError("Parity histories have different epoch counts")
    for a, b in zip(left, right):
        for field_left, field_right in fields:
            if field_left not in a or field_right not in b:
                raise ValueError(f"Parity history is missing {field_left}/{field_right}")
            compare_numeric(a[field_left], b[field_right], tolerance=tolerance,
                            path=f"epoch{a.get('epoch')}.{field_left}")


def _compare_optimizer_point_state(source_last, reference_last, point_parameter_count):
    left = source_last["optimizer"]["state"]
    right = reference_last["optimizer"]["state"]
    for index in range(point_parameter_count):
        a, b = left.get(index), right.get(index)
        if a is None or b is None:
            if a is not b:
                raise ValueError("Point optimizer state presence differs")
            continue
        if set(a) != set(b):
            raise ValueError("Point optimizer state fields differ")
        for name in a:
            if torch.is_tensor(a[name]) or torch.is_tensor(b[name]):
                if not torch.is_tensor(a[name]) or not torch.is_tensor(b[name]) or not torch.equal(a[name].cpu(), b[name].cpu()):
                    raise ValueError(f"Point optimizer tensor state differs: {index}/{name}")
            else:
                compare_numeric(a[name], b[name], tolerance=0,
                                path=f"optimizer.point_param_{index}.{name}")


def _audit_frozen_oof(suite_path, view, seed, canonical_ids, targets, project_root):
    suite = _read_json(suite_path)
    if not suite.get("suite_complete") or suite.get("smoke_only") or suite.get("test_evaluated"):
        raise ValueError("Frozen reference source suite is incomplete or out of scope")
    if suite.get("official_validation_labels_parsed", False) or suite.get("official_VALID_used", False):
        raise ValueError("Frozen reference source suite reports official VALID access")
    key = f"{view}/seed_{seed}"
    summary = suite.get("oof", {}).get(key)
    if not summary or not summary.get("coverage_exactly_once") or not summary.get("complete"):
        raise ValueError(f"Frozen reference lacks complete OOF for {key}")
    path = _resolve(summary["path"], project_root)
    if digest(path) != summary["sha256"] or summary["count"] != len(canonical_ids):
        raise ValueError(f"Frozen reference OOF hash/count changed for {key}")
    rows = _read_csv(path)
    if [row["id"] for row in rows] != canonical_ids:
        raise ValueError(f"Frozen reference OOF order differs for {key}")
    _source_rows(rows, targets)
    return rows, {"path": str(path), "sha256": summary["sha256"], "count": len(rows)}


def _audit_text640(source_suite_path, source_audit_path, seed, ids, targets, project_root):
    suite = _read_json(source_suite_path)
    audit = _read_json(source_audit_path)
    expected_keys = {f"text/seed_{value}" for value in SEEDS}
    if (suite.get("smoke_only") is not False or suite.get("test_evaluated") is not False
            or set(suite.get("oof", {})) != expected_keys):
        raise ValueError("Frozen text-640 OOF summary is partial or out of scope")
    if (not audit.get("complete_nine_control_teachers") or audit.get("test_evaluated")
            or audit.get("official_VALID_used")):
        raise ValueError("Frozen 640-unit text control audit is incomplete/out of scope")
    summary = suite.get("oof", {}).get(f"text/seed_{seed}")
    if not summary or not summary.get("coverage_exactly_once") or not summary.get("complete"):
        raise ValueError(f"Frozen 640-unit text OOF missing for seed {seed}")
    path = _resolve(summary["path"], project_root)
    if digest(path) != summary["sha256"] or summary.get("count") != len(ids):
        raise ValueError(f"Frozen 640-unit text OOF hash/count changed for seed {seed}")
    records = _read_csv(path)
    if [row["id"] for row in records] != ids:
        raise ValueError(f"Frozen 640-unit text OOF IDs differ for seed {seed}")
    _source_rows(records, targets)
    native = native_metrics(np.asarray([row["prediction"] for row in records], np.float32),
                            np.asarray([row["target"] for row in records], np.float32), "simsv2")
    compare_numeric(native, audit["native_OOF"][str(seed)]["wide_text"], tolerance=1e-6,
                    path=f"text640.seed_{seed}.native")
    return records, {"path": str(path), "sha256": summary["sha256"], "count": len(records),
                     "native": native, "capacity_audit_sha256": digest(source_audit_path)}


def _point_arrays(records):
    return (np.asarray([row["prediction"] for row in records], np.float32),
            np.asarray([row["target"] for row in records], np.float32))


def _source_metric_set(records):
    prediction, target = _point_arrays(records)
    return {"native": native_metrics(prediction, target, "simsv2"),
            "source_diagnostics": source_diagnostics(records)}


def _source_comparisons(predictions, ids, reference_names, reps=BOOTSTRAP_REPS):
    errors = {}
    for key, pred in predictions.items():
        errors[key] = {seed: np.abs(np.clip(pred[seed], -1, 1) - predictions["target"][seed])
                       for seed in SEEDS}
    comparisons = {}
    for candidate, references in reference_names.items():
        comparisons[candidate] = {}
        for reference in references:
            delta = {seed: errors[candidate][seed] - errors[reference][seed] for seed in SEEDS}
            comparisons[candidate][reference] = paired_source_bootstrap(delta, ids, replicates=reps)
    return comparisons


def _gain_gate(comparison):
    by_seed = comparison["by_seed"]
    pooled = comparison["three_seed_same_source_average"]["ci95"]
    lower_each_seed = all(by_seed[str(seed)]["estimate"] < 0 for seed in SEEDS)
    return {
        "pass": lower_each_seed and pooled[1] < 0,
        "all_seed_point_differences_below_zero": lower_each_seed,
        "pooled_source_ci_upper_below_zero": pooled[1] < 0,
        "criterion": "Known-baseline gain only: lower MAE in all three seeds and paired pooled source-cluster 95% CI upper bound below zero.",
        "novelty_or_test_claim": False,
    }


def _validate_config(config):
    expected = {
        "dataset": "simsv2",
        "seeds": list(SEEDS),
        "cases": list(CASES),
        "fold_count": 3,
        "fold_seed": 20261012,
        "inner_fraction": .1,
        "predict_valid": False,
        "hidden": 256,
        "dropout": .1,
        "batch_size": 32,
        "learning_rate": .0001,
        "weight_decay": .0001,
        "max_epochs": 80,
        "patience": 12,
        "gradient_norm": 1.,
        "scheduler_patience": 3,
        "scheduler_factor": .5,
        "dro_step_size": .01,
        "dann_gradient_scale": .1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Source-robust config differs from the frozen protocol: {key}")


def audit_suite(*, suite_path, config_path, output_path,
                anchor_suite_path=PROJECT_ROOT / "docs/native_anchor_crossfit_formal_20261013.json",
                text640_suite_path=PROJECT_ROOT / "docs/native_source_text_capacity_fold2_20261013.json",
                text640_audit_path=PROJECT_ROOT / "docs/native_source_text_capacity_audit_20261013.json",
                bootstrap_replicates=BOOTSTRAP_REPS, project_root=PROJECT_ROOT):
    if bootstrap_replicates < 2000:
        raise ValueError("At least 2,000 whole-source bootstrap draws are required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    suite_path = _resolve(suite_path, project_root)
    config_path = _resolve(config_path, project_root)
    output_path = _resolve(output_path, project_root)
    anchor_suite_path = _resolve(anchor_suite_path, project_root)
    text640_suite_path = _resolve(text640_suite_path, project_root)
    text640_audit_path = _resolve(text640_audit_path, project_root)
    _require(not output_path.exists(), f"Audit output exists; refusing overwrite: {output_path}")
    for path in (suite_path, config_path, anchor_suite_path, text640_suite_path, text640_audit_path):
        _require(path.is_file(), f"Required source-robust audit input is missing: {path}")
    config = _read_json(config_path)
    _validate_config(config)
    suite = _read_json(suite_path)
    _require(suite.get("suite_complete") is True and suite.get("smoke_only") is False
             and suite.get("test_evaluated") is False and suite.get("official_VALID_used") is False,
             "Only the complete TRAIN-only 54-teacher suite is in audit scope")
    expected_keys = {(case, seed, fold) for case in CASES for seed in SEEDS for fold in FOLDS}
    actual_keys = [(row.get("case"), row.get("seed"), row.get("fold")) for row in suite.get("results", [])]
    _require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys,
             "Formal suite must contain each of the 54 case/seed/fold teachers exactly once")
    expected_oof_keys = {f"{case}/seed_{seed}" for case in CASES for seed in SEEDS}
    _require(set(suite.get("oof", {})) == expected_oof_keys,
             "Formal suite must contain all 18 merged TRAIN OOF tables")
    _require(all(row.get("complete") is True and row.get("smoke_only") is False
                 and row.get("official_VALID_used") is False and row.get("test_evaluated") is False
                 for row in suite["results"]), "A formal teacher is incomplete or outside TRAIN-only scope")

    manifest_path = suite_path.with_suffix(suite_path.suffix + ".manifest.json")
    _require(manifest_path.is_file(), "Formal source-robust suite manifest is required")
    manifest = _read_json(manifest_path)
    _require(manifest.get("config") == config and manifest.get("smoke_only") is False
             and manifest.get("official_VALID_used") is False and manifest.get("test_evaluated") is False,
             "Source-robust manifest/config/scope mismatch")

    prepare_config = dict(config)
    prepare_config["author_csv"] = str(_resolve(config["author_csv"], project_root))
    prepare_config["cache_dir"] = str(_resolve(config["cache_dir"], project_root))
    data = prepare(prepare_config)
    ids = list(data["ids"])
    targets = {sid: float(data["rows"][sid]["score"]) for sid in ids}
    _require(len(ids) == 2722 and len({group_of(sid) for sid in ids}) == 129,
             "Pinned source-robust TRAIN cohort count differs")
    _require(manifest.get("ids") == ids and manifest.get("source_csv_sha256") == SOURCE_CSV_SHA256
             and data["author_csv_sha256"] == SOURCE_CSV_SHA256,
             "Canonical TRAIN IDs or source CSV pin differs")
    _require(_feature_hashes_equal(manifest.get("feature_hashes", {}), data["train"].hashes),
             "Feature-cache hash manifest differs from current TRAIN bytes")
    _require(manifest.get("fold_manifest") == data["fold_manifest"],
             "Frozen source-fold manifest differs from the registered generator")
    validate_folds(ids, data["fold_manifest"]["folds"])

    anchor_suite = _read_json(anchor_suite_path)
    _require(anchor_suite.get("suite_complete") is True and anchor_suite.get("smoke_only") is False
             and anchor_suite.get("test_evaluated") is False,
             "Frozen fusion/text source-CV reference suite is not complete")
    anchor_manifest_path = anchor_suite_path.with_suffix(anchor_suite_path.suffix + ".manifest.json")
    _require(anchor_manifest_path.is_file(), "Frozen anchor source manifest is required")
    anchor_manifest = _read_json(anchor_manifest_path)
    text640_suite = _read_json(text640_suite_path)
    _require(text640_audit := _read_json(text640_audit_path), "Text capacity audit is empty")
    if not text640_audit.get("complete_nine_control_teachers") or text640_audit.get("test_evaluated"):
        raise ValueError("Frozen text-640 control audit is incomplete or out of scope")
    text_capacity_config_path = PROJECT_ROOT / "configs/native_source_text_capacity_v1.json"
    text_capacity_config = _read_json(text_capacity_config_path)
    text640_manifest_path = text640_suite_path.with_suffix(text640_suite_path.suffix + ".manifest.json")
    text640_config_path = text640_suite_path.with_suffix(text640_suite_path.suffix + ".config.json")
    _require(text640_manifest_path.is_file() and text640_config_path.is_file(),
             "Frozen text-640 source manifest/config snapshot is required")
    text640_manifest = _read_json(text640_manifest_path)
    text640_saved_config = _read_json(text640_config_path)
    _require(text_capacity_config.get("hidden") == 640 and text_capacity_config.get("predict_valid") is False,
             "Text-640 control is not the registered source-heldout capacity control")
    _require(text640_saved_config == text_capacity_config
             and text640_audit.get("config_sha256") == digest(text_capacity_config_path),
             "Text-640 frozen config/audit identity mismatch")
    if text640_audit.get("frozen_source_suite_sha256") != digest(anchor_suite_path):
        raise ValueError("Text-640 audit was not paired with the supplied frozen source suite")
    for control_manifest, control_name in ((anchor_manifest, "fusion/text-256"),
                                           (text640_manifest, "text-640")):
        control_folds = control_manifest["fold_splits"]["fold_manifest"]
        if (control_manifest.get("ids") != ids
                or control_manifest.get("author_csv_sha256") != SOURCE_CSV_SHA256
                or control_manifest.get("smoke_groups") is not None
                or control_manifest.get("fixture_smoke") is not False
                or control_manifest.get("test_evaluated") is not False
                or control_manifest.get("official_validation_labels_parsed") is not False
                or control_folds != data["fold_manifest"]
                or not _feature_hashes_equal(control_manifest.get("train_feature_hashes", {}),
                                             data["train"].hashes)):
            raise ValueError(f"Frozen {control_name} OOF does not share the registered TRAIN IDs/features/folds")

    train_ids = ids
    results_map = {(row["case"], row["seed"], row["fold"]): row for row in suite["results"]}
    baseline_results = {(row.get("view"), row.get("seed"), row.get("fold")): row
                        for row in anchor_suite.get("results", [])}
    source_code_names = ("source_robust.py", "source_robust_training.py", "anchor_crossfit.py",
                         "native_training.py", "native_metrics.py", "source_crossfit.py")
    current_source_code = {name: digest(PROJECT_ROOT / "jevaffect" / name) for name in source_code_names}
    _require(all(("fusion", seed, fold) in baseline_results and ("text", seed, fold) in baseline_results
                 for seed in SEEDS for fold in FOLDS),
             "Frozen original fusion/text controls are missing source-fold results")

    fold_rows = {(case, seed): {} for case in CASES for seed in SEEDS}
    identity_hashes, result_hashes, history_hashes = {}, {}, {}
    checkpoint_hashes, prediction_hashes = {}, {}
    teacher_budgets, teacher_parity = {}, {}
    for case in CASES:
        for seed in SEEDS:
            for fold in FOLDS:
                key = (case, seed, fold)
                summary = results_map[key]
                root = _resolve(summary["root"], project_root)
                expected_root = (_resolve(config["run_root"], project_root) / "formal" / case
                                 / f"seed_{seed}" / f"fold_{fold}").resolve()
                _require(root == expected_root and root.is_dir(), f"Unexpected or missing run root for {key}")
                result_path, identity_path = root / "result.json", root / "identity.json"
                result, identity = _read_json(result_path), _read_json(identity_path)
                _require(result == summary, f"Suite result differs from saved result for {key}")
                result_hashes[f"{case}/seed_{seed}/fold_{fold}"] = digest(result_path)
                _require(identity.get("implementation") == current_source_code,
                         f"Teacher code hashes differ for {key}")
                _require(identity.get("selection") == "minimum source-inner native MAE; first checkpoint wins ties"
                         and identity.get("scope") == "known source-risk/domain-adversarial regression baselines",
                         f"Teacher selection/scope identity differs for {key}")
                _require(identity.get("official_VALID_used") is False and identity.get("test_evaluated") is False,
                         f"Teacher identity reports VALID/TEST access for {key}")
                cohort = data["fold_manifest"]["folds"][fold]
                domains, groups, domain_counts = _fit_identity_cohort(identity, data, case, seed, fold,
                                                                        cohort, config)
                _require(result["fit_count"] == len(cohort["fit_ids"])
                         and result["select_count"] == len(cohort["select_ids"])
                         and result["oof_count"] == len(cohort["oof_ids"])
                         and result["source_count"] == len(domains),
                         f"Saved cohort counts differ from the source manifest for {key}")
                identity_hashes[f"{case}/seed_{seed}/fold_{fold}"] = digest(identity_path)
                history_path = root / "history.json"
                history = _read_json(history_path)
                history_hashes[f"{case}/seed_{seed}/fold_{fold}"] = digest(history_path)
                steps = validate_epoch_accounting(history, len(cohort["fit_ids"]), config["batch_size"],
                                                  case, domain_counts.tolist(), config["max_epochs"],
                                                  config["patience"])
                best_epoch, best_mae, best_row = first_minimum_mae(history)
                if result.get("best_epoch") != best_epoch or result.get("epochs_executed") != len(history):
                    raise ValueError(f"Teacher did not select the first minimum inner MAE for {key}")
                total_updates = sum(int(row["optimizer_updates"]) for row in history)
                if (result.get("optimizer_updates") != total_updates
                        or total_updates != len(history) * steps):
                    raise ValueError(f"Teacher optimizer updates do not match the recorded budget for {key}")
                loop_seconds = sum(float(row["seconds"]) for row in history)
                if not math.isclose(float(result["loop_seconds"]), loop_seconds, rel_tol=1e-12, abs_tol=1e-9):
                    raise ValueError(f"Teacher loop time differs from the epoch log for {key}")
                best_path, last_path = root / "best.pt", root / "last.pt"
                _require(best_path.is_file() and last_path.is_file(), f"Best/last checkpoint missing for {key}")
                best_hash = digest(best_path)
                _require(best_hash == result["checkpoint_sha256"], f"Best checkpoint hash differs for {key}")
                best_state, last_state = _read_torch(best_path), _read_torch(last_path)
                _require(best_state.get("identity") == identity and last_state.get("identity") == identity,
                         f"Checkpoint identity differs for {key}")
                _require(best_state.get("epoch") == best_epoch and last_state.get("epoch") == len(history)
                         and last_state.get("history") == history,
                         f"Checkpoint epoch/history differs for {key}")
                _require(math.isclose(float(best_state["metrics"]["MAE"]), best_mae, rel_tol=0, abs_tol=1e-12)
                         and math.isclose(float(last_state["best"]), best_mae, rel_tol=0, abs_tol=1e-12),
                         f"Checkpoint best inner MAE differs for {key}")
                _require(best_state["metrics"]["count"] == len(cohort["select_ids"]),
                         f"Selected checkpoint metric cohort differs for {key}")
                _require(result["complete"] is (len(history) >= config["max_epochs"]
                                                  or last_state["stale"] >= config["patience"])
                         and result["complete"], f"Teacher complete/early-stop rule differs for {key}")
                _require(set(("optimizer", "scheduler", "torch_rng", "numpy_rng", "python_rng", "loader_rng"))
                         <= set(last_state), f"Last checkpoint lacks resume state for {key}")
                _require(last_state["active_gradients"] == result["first_batch_nonzero_gradients"],
                         f"First-batch gradient counts differ from last checkpoint for {key}")
                _require(set(("model", "cuda_rng")) <= set(last_state),
                         f"Last checkpoint lacks model/CUDA RNG resume state for {key}")
                _require(last_state["scheduler"] == replay_scheduler(history,config)
                         and math.isclose(float(last_state["optimizer"]["param_groups"][0]["lr"]),
                                          float(history[-1]["learning_rate"]), rel_tol=0, abs_tol=1e-15),
                         f"Optimizer/scheduler resume state differs from saved epoch history for {key}")
                last_probability = torch.softmax(last_state["model"]["log_source_probability"].double(), 0)
                if not torch.allclose(last_probability.cpu(),
                                      torch.tensor(history[-1]["source_probability"], dtype=torch.float64),
                                      rtol=0, atol=1e-12):
                    raise ValueError(f"Last source-risk q differs from the epoch history for {key}")
                point_parameter_count = result["point_parameters"]
                domain_parameter_count = result["domain_parameters"]
                _require(point_parameter_count == POINT_PARAMETERS
                         and domain_parameter_count == config["hidden"] * len(domains) + len(domains),
                         f"Point/domain parameter count differs for {key}")
                active = result["first_batch_nonzero_gradients"]
                _require(isinstance(active, dict) and 0 < active["point"] <= point_parameter_count
                         and 0 <= active["domain"] <= domain_parameter_count,
                         f"First-batch gradient count outside parameter budget for {key}")
                if case in ("domain_probe", "dann"):
                    _require(active["domain"] > 0, f"{case} source classifier did not receive gradient for {key}")
                else:
                    _require(active["domain"] == 0, f"{case} unexpectedly trained the source classifier for {key}")
                _require(sum(int(v.numel()) for k, v in best_state["model"].items()
                             if k.startswith("regressor.")) == point_parameter_count,
                         f"Point state tensor count differs for {key}")
                _require(sum(int(v.numel()) for k, v in best_state["model"].items()
                             if k.startswith("domain_classifier.")) == domain_parameter_count,
                         f"Domain state tensor count differs for {key}")
                # Every saved OOF output is re-evaluated from the CPU-reloaded checkpoint.
                view = PooledView(data["train"], ("text", "audio", "vision"))
                model = SourceRiskModel({m: int(view.features[m].shape[-1]) for m in view.features},
                                        len(domains), config["hidden"], config["dropout"]).cpu()
                model.load_state_dict(best_state["model"])
                select_loader = _loader(subset(view, cohort["select_ids"]), config)
                select_metrics, select_records = _predict_targets(model, select_loader, "cpu")
                compare_numeric(select_metrics, best_state["metrics"], tolerance=PREDICTION_TOLERANCE,
                                path=f"{case}/seed_{seed}/fold_{fold}.inner_select")
                _require(abs(select_metrics["MAE"] - best_mae) <= PREDICTION_TOLERANCE,
                         f"CPU-reloaded selection MAE differs for {key}")
                oof_loader = _loader(subset(view, cohort["oof_ids"]), config)
                oof_metrics, cpu_oof_rows = _predict_targets(model, oof_loader, "cpu")
                compare_numeric(oof_metrics, result["oof_native"], tolerance=PREDICTION_TOLERANCE,
                                path=f"{case}/seed_{seed}/fold_{fold}.oof_native")
                prediction_path = root / "oof_predictions.csv"
                _require(result["predictions_file"] and _resolve(result["predictions_file"], project_root)
                         == prediction_path.resolve(), f"Unexpected OOF path for {key}")
                _require(digest(prediction_path) == result["predictions_sha256"],
                         f"OOF CSV hash differs for {key}")
                disk_rows = _read_csv(prediction_path)
                if [r["id"] for r in disk_rows] != cohort["oof_ids"]:
                    raise ValueError(f"Fold OOF ID order differs for {key}")
                _source_rows(disk_rows, targets)
                cpu_oof_rows_by_id = {row["id"]: row for row in cpu_oof_rows}
                ordered_cpu_rows = [cpu_oof_rows_by_id[sid] for sid in cohort["oof_ids"]]
                compare_prediction_rows(disk_rows, ordered_cpu_rows, tolerance=PREDICTION_TOLERANCE)
                if len(disk_rows) != result["oof_count"]:
                    raise ValueError(f"Fold OOF count differs for {key}")
                fold_rows[(case, seed)][fold] = disk_rows
                checkpoint_hashes[f"{case}/seed_{seed}/fold_{fold}"] = {
                    "best": best_hash, "last": digest(last_path),
                }
                prediction_hashes[f"{case}/seed_{seed}/fold_{fold}"] = result["predictions_sha256"]
                teacher_budgets[f"{case}/seed_{seed}/fold_{fold}"] = {
                    "fit_count": len(cohort["fit_ids"]), "select_count": len(cohort["select_ids"]),
                    "oof_count": len(cohort["oof_ids"]), "source_count": len(domains),
                    "batch_updates_per_epoch": steps, "epochs_executed": len(history),
                    "optimizer_updates": total_updates, "best_epoch": best_epoch,
                    "best_inner_MAE": best_mae, "point_parameters": point_parameter_count,
                    "domain_parameters": domain_parameter_count,
                    "first_batch_nonzero_gradients": active, "loop_seconds": loop_seconds,
                    "sampled_source_counts_by_epoch": [row["sampled_source_counts"] for row in history],
                    "runtime": identity["runtime"],
                }
                del model, best_state, last_state

    merged_oof, oof_hashes = {}, {}
    for case in CASES:
        merged_oof[case], oof_hashes[case] = {}, {}
        for seed in SEEDS:
            canonical = merge_oof_records(fold_rows[(case, seed)], data["fold_manifest"]["folds"],
                                          ids, targets)
            summary = suite.get("oof", {}).get(f"{case}/seed_{seed}")
            _require(summary and summary.get("coverage_exactly_once") is True
                     and summary.get("count") == 2722,
                     f"Merged OOF summary incomplete for {case}/seed_{seed}")
            path = _resolve(summary["path"], project_root)
            expected = (_resolve(config["run_root"], project_root) / "formal" / case / f"seed_{seed}"
                        / "oof_predictions.csv").resolve()
            _require(path == expected and digest(path) == summary["sha256"],
                     f"Merged OOF path/hash differs for {case}/seed_{seed}")
            records = _read_csv(path)
            if [r["id"] for r in records] != ids:
                raise ValueError(f"Merged OOF canonical order differs for {case}/seed_{seed}")
            compare_prediction_rows(records, canonical, exact=True)
            native = native_metrics(np.asarray([r["prediction"] for r in records], np.float32),
                                    np.asarray([r["target"] for r in records], np.float32), "simsv2")
            compare_numeric(native, summary["native"], tolerance=PREDICTION_TOLERANCE,
                            path=f"{case}/seed_{seed}.merged_native")
            merged_oof[case][seed] = records
            oof_hashes[case][str(seed)] = summary["sha256"]

    # Recheck the frozen reference anchors and text-only OOF tables by canonical ID.
    reference_oof = {"fusion256": {}, "text256": {}, "text640": {}}
    reference_hashes = {"anchor_suite": digest(anchor_suite_path),
                        "text640_suite": digest(text640_suite_path),
                        "text640_audit": digest(text640_audit_path)}
    for seed in SEEDS:
        for name, view_name in (("fusion256", "fusion"), ("text256", "text")):
            reference_oof[name][seed], info = _audit_frozen_oof(anchor_suite_path, view_name, seed,
                                                                 ids, targets, project_root)
            reference_hashes[f"{name}/seed_{seed}"] = info["sha256"]
        reference_oof["text640"][seed], info = _audit_text640(
            text640_suite_path, text640_audit_path, seed, ids, targets, project_root
        )
        reference_hashes[f"text640/seed_{seed}"] = info["sha256"]

    parity = {"clip_erm_vs_frozen_fusion256": {}, "domain_probe_vs_balanced_erm": {}}
    point_parameter_objects = len([key for key in ("head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias")])
    for seed in SEEDS:
        for fold in FOLDS:
            clip_root = _resolve(results_map[("clip_erm", seed, fold)]["root"], project_root)
            fusion_result = baseline_results[("fusion", seed, fold)]
            fusion_root = _resolve(fusion_result["root"], project_root)
            fusion_best, fusion_last = _read_torch(fusion_root / "best.pt"), _read_torch(fusion_root / "last.pt")
            clip_best, clip_last = _read_torch(clip_root / "best.pt"), _read_torch(clip_root / "last.pt")
            clip_best_state = compare_point_tensors(clip_best["model"], fusion_best["model"], point_prefix="regressor.")
            clip_last_state = compare_point_tensors(clip_last["model"], fusion_last["model"], point_prefix="regressor.")
            _require(clip_best["epoch"] == fusion_best["epoch"], "clip_erm best epoch differs from frozen fusion")
            _require(fusion_result["complete"] and not fusion_result["smoke_only"]
                     and not fusion_result["test_evaluated"], "Frozen fusion fold is incomplete/out of scope")
            _require(digest(fusion_root / "best.pt") == fusion_result["checkpoint_sha256"],
                     "Frozen fusion checkpoint hash differs")
            _require(digest(fusion_root / "oof_predictions.csv") == fusion_result["predictions_sha256"],
                     "Frozen fusion fold OOF hash differs")
            clip_result = results_map[("clip_erm", seed, fold)]
            _require(clip_result["best_epoch"] == fusion_result["best_epoch"]
                     and clip_result["epochs_executed"] == fusion_result["epochs_executed"]
                     and clip_result["optimizer_updates"] == fusion_result["optimizer_updates"],
                     "clip_erm epoch/update budget differs from frozen fusion")
            compare_numeric(clip_best["metrics"], fusion_best["metrics"], tolerance=0,
                            path=f"clip_erm_vs_fusion.seed_{seed}.fold_{fold}.metrics")
            _compare_optimizer_point_state(clip_last, fusion_last, 4)
            _compare_history_fields(
                _read_json(clip_root / "history.json"), _read_json(fusion_root / "history.json"),
                (("epoch", "epoch"), ("train_clip_L1", "train_L1"),
                 ("inner_select", "inner_select"), ("optimizer_updates", "optimizer_updates"),
                 ("learning_rate", "learning_rate")), exact=True,
            )
            clip_csv = _read_csv(clip_root / "oof_predictions.csv")
            fusion_csv = _read_csv(fusion_root / "oof_predictions.csv")
            clip_oof = compare_prediction_rows(clip_csv, fusion_csv, exact=True)
            parity["clip_erm_vs_frozen_fusion256"][f"seed_{seed}/fold_{fold}"] = {
                "best_point_state": clip_best_state, "last_point_state": clip_last_state,
                "last_point_optimizer_state_exact": True,
                "history_loss_inner_mae_lr_epoch_exact": True, "oof_predictions": clip_oof,
            }

            probe_root = _resolve(results_map[("domain_probe", seed, fold)]["root"], project_root)
            balanced_root = _resolve(results_map[("balanced_erm", seed, fold)]["root"], project_root)
            probe_best, probe_last = _read_torch(probe_root / "best.pt"), _read_torch(probe_root / "last.pt")
            balanced_best = _read_torch(balanced_root / "best.pt")
            balanced_last = _read_torch(balanced_root / "last.pt")
            probe_best_state = compare_point_tensors(probe_best["model"], balanced_best["model"], point_prefix="regressor.")
            probe_last_state = compare_point_tensors(probe_last["model"], balanced_last["model"], point_prefix="regressor.")
            probe_result = results_map[("domain_probe", seed, fold)]
            balanced_result = results_map[("balanced_erm", seed, fold)]
            _require(probe_best["epoch"] == balanced_best["epoch"]
                     and probe_result["best_epoch"] == balanced_result["best_epoch"]
                     and probe_result["epochs_executed"] == balanced_result["epochs_executed"],
                     "domain_probe point epochs differ from balanced_erm")
            _compare_history_fields(
                _read_json(probe_root / "history.json"), _read_json(balanced_root / "history.json"),
                (("epoch", "epoch"), ("train_clip_L1", "train_clip_L1"),
                 ("risk_objective", "risk_objective"), ("inner_select", "inner_select"),
                 ("optimizer_updates", "optimizer_updates"),
                 ("sampled_source_counts", "sampled_source_counts"),
                 ("source_probability", "source_probability"), ("learning_rate", "learning_rate")),
                exact=True,
            )
            point_optimizer_objects = len([key for key in ("head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias")])
            _compare_optimizer_point_state(probe_last, balanced_last, point_optimizer_objects)
            probe_csv = _read_csv(probe_root / "oof_predictions.csv")
            balanced_csv = _read_csv(balanced_root / "oof_predictions.csv")
            probe_oof = compare_prediction_rows(probe_csv, balanced_csv, exact=True)
            parity["domain_probe_vs_balanced_erm"][f"seed_{seed}/fold_{fold}"] = {
                "best_point_state": probe_best_state, "last_point_state": probe_last_state,
                "point_optimizer_state_exact": True, "best_and_executed_epochs_exact": True,
                "point_history_exact": True, "oof_predictions": probe_oof,
                "domain_probe_head_gradients": probe_result["first_batch_nonzero_gradients"]["domain"],
                "balanced_head_gradients": balanced_result["first_batch_nonzero_gradients"]["domain"],
            }
            del fusion_best, fusion_last, clip_best, clip_last, probe_best, probe_last, balanced_best, balanced_last

    metrics_by_case_seed = {}
    for case in CASES:
        metrics_by_case_seed[case] = {}
        for seed in SEEDS:
            records = merged_oof[case][seed]
            native = native_metrics(np.asarray([r["prediction"] for r in records], np.float32),
                                    np.asarray([r["target"] for r in records], np.float32), "simsv2")
            metrics_by_case_seed[case][str(seed)] = {**native, **source_diagnostics(records)}
    reference_metrics = {}
    for name, by_seed in reference_oof.items():
        reference_metrics[name] = {}
        for seed, records in by_seed.items():
            p, y = _point_arrays(records)
            reference_metrics[name][str(seed)] = {**native_metrics(p, y, "simsv2"), **source_diagnostics(records)}

    targets_by_seed = {seed: np.asarray([targets[sid] for sid in ids], np.float32) for seed in SEEDS}
    prediction_arrays = {case: {seed: np.asarray([r["prediction"] for r in merged_oof[case][seed]], np.float32)
                                for seed in SEEDS} for case in CASES}
    prediction_arrays["fusion256"] = {seed: np.asarray([r["prediction"] for r in reference_oof["fusion256"][seed]], np.float32)
                                      for seed in SEEDS}
    prediction_arrays["text256"] = {seed: np.asarray([r["prediction"] for r in reference_oof["text256"][seed]], np.float32)
                                    for seed in SEEDS}
    prediction_arrays["text640"] = {seed: np.asarray([r["prediction"] for r in reference_oof["text640"][seed]], np.float32)
                                    for seed in SEEDS}
    prediction_arrays["target"] = targets_by_seed
    reference_list = ("fusion256", "text256", "text640")
    candidates = {
        case: list(reference_list) for case in CASES
    }
    candidates["group_dro"] = ["dro_uniform", "balanced_erm", *reference_list]
    candidates["dann"] = ["domain_probe", "balanced_erm", *reference_list]
    mae_comparisons = _source_comparisons(prediction_arrays, ids, candidates, bootstrap_replicates)
    gain_gates = {case: {reference: _gain_gate(value)
                         for reference, value in comparisons.items()}
                  for case, comparisons in mae_comparisons.items()}

    audit = {
        "scope": "Formal native CH-SIMS TRAIN source-OOF audit; no official VALID/TEST access.",
        "suite": str(suite_path),
        "suite_sha256": digest(suite_path),
        "suite_manifest": str(manifest_path),
        "suite_manifest_sha256": digest(manifest_path),
        "config": str(config_path),
        "config_sha256": digest(config_path),
        "author_csv_sha256": data["author_csv_sha256"],
        "feature_hashes": data["train"].hashes,
        "fold_manifest_sha256": manifest_hash(data["fold_manifest"]),
        "source_code_sha256": current_source_code,
        "control_inputs": {"anchor_crossfit_suite_sha256": digest(anchor_suite_path),
                            "text_capacity_suite_sha256": digest(text640_suite_path),
                            "text_capacity_audit_sha256": digest(text640_audit_path),
                            "text_capacity_config_sha256": digest(text_capacity_config_path),
                            "frozen_reference_oof_sha256": reference_hashes},
        "cases": list(CASES),
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "teacher_count": len(expected_keys),
        "merged_oof_count": len(CASES) * len(SEEDS),
        "train_count": len(ids),
        "source_count": len({group_of(sid) for sid in ids}),
        "cpu_checkpoint_reloads": True,
        "official_VALID_used": False,
        "test_evaluated": False,
        "fold_teacher_hashes": {
            "identity_json_sha256": identity_hashes,
            "result_json_sha256": result_hashes,
            "history_json_sha256": history_hashes,
            "checkpoint_sha256": checkpoint_hashes,
            "prediction_sha256": prediction_hashes,
        },
        "merged_oof_sha256": oof_hashes,
        "teacher_budgets": teacher_budgets,
        "parity": parity,
        "merged_oof_native_and_source_diagnostics": metrics_by_case_seed,
        "frozen_strong_control_metrics": reference_metrics,
        "paired_whole_source_bootstrap": {
            "replicates": bootstrap_replicates,
            "random_seed": BOOTSTRAP_SEED,
            "unit": "source video group; all clips from sampled source retained",
            "difference": "candidate minus comparator native MAE; negative favors candidate",
            "comparisons": mae_comparisons,
            "known_baseline_gain_gates": gain_gates,
        },
        "interpretation": (
            "These are known source-risk/domain-adversarial baselines. A passing MAE gate only records a TRAIN source-OOF baseline gain; it is not evidence of originality, official VALID performance, or TEST performance."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True,
                        help="Formal source-robust 54-result suite JSON")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/native_source_robust_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-suite", type=Path,
                        default=PROJECT_ROOT / "docs/native_anchor_crossfit_formal_20261013.json")
    parser.add_argument("--text640-suite", type=Path,
                        default=PROJECT_ROOT / "docs/native_source_text_capacity_fold2_20261013.json")
    parser.add_argument("--text640-audit", type=Path,
                        default=PROJECT_ROOT / "docs/native_source_text_capacity_audit_20261013.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPS)
    args = parser.parse_args()
    result = audit_suite(
        suite_path=args.suite, config_path=args.config, output_path=args.output,
        anchor_suite_path=args.anchor_suite, text640_suite_path=args.text640_suite,
        text640_audit_path=args.text640_audit,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps({"output": str(_resolve(args.output)), "teacher_count": result["teacher_count"],
                      "merged_oof_count": result["merged_oof_count"], "test_evaluated": False},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
