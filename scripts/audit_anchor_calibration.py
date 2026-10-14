"""Independently audit the formal AnchorCDF calibration suite on CPU.

The audit reloads each selected checkpoint, reproduces VALID predictions and
metrics, reconstructs the scalar Global fit, and reports paired source-cluster
bootstrap comparisons. It never trains or reads TEST labels.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.anchor_calibration import (  # noqa: E402
    BOUNDARIES,
    CASES,
    CalibrationHead,
    bin_targets,
    diagnostics,
    disagreement,
    describe,
    error_band_probability,
    objective,
    parameters,
    bin_probabilities,
)
from jevaffect.anchor_calibration_training import (  # noqa: E402
    _records,
    evaluate,
    load_table,
    prepare_training,
    tensors,
    _selection,
)
from jevaffect.native_training import digest, save_json  # noqa: E402
from jevaffect.native_training import author_rows  # noqa: E402
from jevaffect.source_crossfit import group_of  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (13, 42, 2026)
ALL_CASES = ("global", *CASES)
BOOTSTRAP_SEED = 20261013
DEFAULT_BOOTSTRAP_REPS = 5000
BOUNDARY_TOLERANCE = 1e-6
SOURCE_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolve(path, base=PROJECT_ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_raw_feature_hashes(metadata):
    declared = metadata.get("raw_feature_hashes", {})
    if set(declared) != {"train", "valid"}:
        raise ValueError("Raw-feature split hash manifest is incomplete")
    metadata_keys = [key for hashes in declared.values() for key in hashes
                     if str(key).replace("\\", "/").endswith("/metadata.json")]
    if not metadata_keys:
        raise ValueError("Raw-feature hash manifest does not locate its cache metadata")
    cache_metadata = _resolve(metadata_keys[0], PROJECT_ROOT)
    cache_root = cache_metadata.parent
    verified = {}
    for split in ("train", "valid"):
        hashes = declared[split]
        if not hashes:
            raise ValueError(f"Missing {split} raw-feature hashes")
        for key, expected in hashes.items():
            text = str(key).replace("\\", "/")
            path_obj = Path(text)
            if path_obj.is_absolute():
                path = path_obj.resolve()
            elif text.startswith("data/"):
                path = (PROJECT_ROOT / Path(text)).resolve()
            elif text.startswith(("splits/", "train/", "valid/")):
                path = (cache_root / Path(text)).resolve()
            else:
                raise ValueError(f"Unrecognized raw-feature hash path: {key}")
            actual = digest(path)
            if actual != expected:
                raise ValueError(f"Raw feature hash mismatch: {path}")
            verified[f"{split}/{path.relative_to(PROJECT_ROOT).as_posix()}"] = actual
    return verified


def _verify_author_targets(metadata, train_by_seed, valid_by_seed):
    csv_path = _resolve(metadata["author_csv"], PROJECT_ROOT)
    csv_hash = digest(csv_path)
    if csv_hash != SOURCE_CSV_SHA256 or metadata.get("author_csv_sha256") != csv_hash:
        raise ValueError("Original CH-SIMS CSV differs from its registered source hash")
    rows = author_rows(csv_path, "simsv2")
    train_ids = [sid for sid, row in rows.items() if row["split"] == "train"]
    valid_ids = [sid for sid, row in rows.items() if row["split"] == "valid"]
    if len(train_ids) != 2722 or len(valid_ids) != 647 or len({group_of(sid) for sid in train_ids}) != 129:
        raise ValueError("Original author TRAIN/VALID cohort counts differ")
    if train_ids != train_by_seed[SEEDS[0]]["ids"] or valid_ids != valid_by_seed[SEEDS[0]]["ids"]:
        raise ValueError("Calibration table IDs differ from canonical author CSV TRAIN/VALID order")
    for seed in SEEDS:
        for split_name, arrays in (("train", train_by_seed[seed]), ("valid", valid_by_seed[seed])):
            for index, sid in enumerate(arrays["ids"]):
                row = rows.get(sid)
                if row is None or row["split"] != split_name:
                    raise ValueError(f"Unexpected author split for calibration ID {sid}")
                if np.float32(arrays["target"][index]) != np.float32(row["score"]):
                    raise ValueError(f"Calibration target differs from original float32 M for {sid}")
    return csv_hash


def validate_head_split(train_ids, split):
    """Validate calibration fit/select indices, IDs, and source isolation."""
    ids = list(train_ids)
    fit_indices = list(split["fit_indices"])
    select_indices = list(split["select_indices"])
    if len(ids) != len(set(ids)) or not fit_indices or not select_indices:
        raise ValueError("Calibration TRAIN IDs and fit/select cohorts must be nonempty and unique")
    expected = set(range(len(ids)))
    fit_set, select_set = set(fit_indices), set(select_indices)
    if len(fit_set) != len(fit_indices) or len(select_set) != len(select_indices):
        raise ValueError("Calibration fit/select indices contain duplicates")
    if fit_set & select_set or fit_set | select_set != expected:
        raise ValueError("Calibration fit/select indices do not partition TRAIN")
    fit_ids = [ids[i] for i in fit_indices]
    select_ids = [ids[i] for i in select_indices]
    if split.get("fit_ids") != fit_ids or split.get("select_ids") != select_ids:
        raise ValueError("Calibration split IDs differ from their canonical indices")
    fit_sources = {group_of(sid) for sid in fit_ids}
    select_sources = {group_of(sid) for sid in select_ids}
    if fit_sources & select_sources:
        raise ValueError("Calibration head fit/select sources overlap")
    if split.get("fit_sources") != sorted(fit_sources) or split.get("select_sources") != sorted(select_sources):
        raise ValueError("Calibration split source lists do not match the ID cohorts")
    if split.get("source_count") != len(fit_sources | select_sources):
        raise ValueError("Calibration source count differs from the TRAIN IDs")
    if split.get("select_source_count") != len(select_sources):
        raise ValueError("Calibration selection source count differs from the TRAIN IDs")
    return fit_indices, select_indices


def expected_head_split(train_ids, seed=20261013, inner_fraction=.1):
    """Rebuild the preregistered source selection split from canonical IDs."""
    ids = list(train_ids)
    groups = sorted({group_of(sid) for sid in ids})
    if not groups or len(set(ids)) != len(ids):
        raise ValueError("Cannot rebuild calibration selection from invalid TRAIN IDs")
    rng = np.random.default_rng(seed)
    selected_groups = set(rng.permutation(groups)[:max(1, int(np.ceil(len(groups) * inner_fraction)))] .tolist())
    fit_groups = set(groups) - selected_groups
    positions = {sid: index for index, sid in enumerate(ids)}
    fit_ids = [sid for sid in ids if group_of(sid) in fit_groups]
    select_ids = [sid for sid in ids if group_of(sid) in selected_groups]
    return {
        "seed": seed,
        "inner_fraction": inner_fraction,
        "source_count": len(groups),
        "select_source_count": len(selected_groups),
        "fit_sources": sorted(fit_groups),
        "select_sources": sorted(selected_groups),
        "fit_ids": fit_ids,
        "select_ids": select_ids,
        "fit_indices": [positions[sid] for sid in fit_ids],
        "select_indices": [positions[sid] for sid in select_ids],
    }


def first_minimum_epoch(history, objective_key="selection_objective"):
    if not history or any(objective_key not in row for row in history):
        raise ValueError("Selection history is empty or lacks its objective")
    values = [float(row[objective_key]) for row in history]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Selection history contains nonfinite objective values")
    # min over an ordered sequence returns the first exact tie.
    best_index = min(range(len(values)), key=values.__getitem__)
    return history[best_index]["epoch"], values[best_index]


def _compare_payload(actual, expected, *, tolerance=1e-7, path="root"):
    """Recursively compare recomputed JSON-compatible predictions/metrics."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Prediction/metric keys differ at {path}")
        for key in expected:
            _compare_payload(actual[key], expected[key], tolerance=tolerance, path=f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Prediction/metric lengths differ at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_payload(left, right, tolerance=tolerance, path=f"{path}[{index}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"Prediction/metric value differs at {path}")
    elif isinstance(expected, (int, float, np.number)):
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-7, abs_tol=tolerance
        ):
            raise ValueError(f"Prediction/metric numeric value differs at {path}: {actual} != {expected}")
    else:
        if actual != expected:
            raise ValueError(f"Prediction/metric value differs at {path}")


def _verify_fit_only_statistics(train, fit_indices, stats):
    x = np.concatenate((train["features"], train["anchor"][:, None]), 1)
    d = disagreement(train["views"])
    views = np.concatenate((train["views"], np.abs(train["views"])), 1)
    expected_values = {"context": x, "differences": d, "views": views}
    if set(stats) != set(expected_values):
        raise ValueError("Calibration normalization paths differ from the registered context")
    for key, values in expected_values.items():
        stat = stats[key]
        if stat.get("fit_indices") != list(fit_indices):
            raise ValueError(f"{key} normalization includes rows outside calibration fit")
        fit = values[np.asarray(fit_indices, np.int64)].astype(np.float64)
        mean = fit.mean(0)
        scale = fit.std(0)
        scale[scale < 1e-6] = 1.
        if not np.array_equal(np.asarray(stat["mean"], np.float32), mean.astype(np.float32)):
            raise ValueError(f"{key} normalization mean does not reproduce from fit rows only")
        if not np.array_equal(np.asarray(stat["scale"], np.float32), scale.astype(np.float32)):
            raise ValueError(f"{key} normalization scale does not reproduce from fit rows only")


def _verify_probabilities_and_point(case, records, anchor):
    anchor = np.asarray(anchor, np.float32)
    _require(len(records) == len(anchor), f"{case}: VALID record count differs from anchor")
    if case == "direct_hit":
        _require(all(not ("bin_probabilities" in row or "intervals" in row or "median" in row)
                     for row in records), "Direct-hit must not be reported as a full CDF")
    else:
        _require(all({"bin_probabilities", "intervals", "median", "scale"} <= set(row)
                     for row in records), f"{case}: CDF probabilities/intervals/median are incomplete")
    max_point_delta = 0.0
    max_median_delta = 0.0
    for index, row in enumerate(records):
        prediction = float(row["prediction"])
        _require(math.isfinite(prediction), f"{case}: nonfinite original point prediction")
        point_delta = abs(prediction - float(anchor[index]))
        max_point_delta = max(max_point_delta, point_delta)
        _require(point_delta <= BOUNDARY_TOLERANCE, f"{case}: calibration changed the native point prediction")
        probability = float(row["p_hit"])
        _require(math.isfinite(probability) and 0 <= probability <= 1,
                 f"{case}: invalid error-band probability")
        if "bin_probabilities" in row:
            bins = np.asarray(row["bin_probabilities"], np.float64)
            _require(bins.shape == (5,) and np.isfinite(bins).all()
                     and np.all(bins >= 0) and np.allclose(bins.sum(), 1., atol=1e-10),
                     f"{case}: invalid native-bin probabilities")
        if "scale" in row:
            scale = float(row["scale"])
            _require(math.isfinite(scale) and scale > 0, f"{case}: invalid predictive scale")
        if "mixture_weight" in row:
            weight = float(row["mixture_weight"])
            _require(math.isfinite(weight) and 0 <= weight <= 1,
                     f"{case}: invalid mixture weight")
        if "intervals" in row:
            expected_levels = {"0.5", "0.8", "0.9", "0.95"}
            _require(set(row["intervals"]) == expected_levels, f"{case}: interval levels differ")
            for level, pair in row["intervals"].items():
                interval = np.asarray(pair, np.float64)
                _require(interval.shape == (2,) and np.isfinite(interval).all()
                         and -1 <= interval[0] <= interval[1] <= 1,
                         f"{case}: invalid {level} interval")
        if "median" in row:
            median = float(row["median"])
            _require(math.isfinite(median) and -1 <= median <= 1, f"{case}: invalid distribution median")
            median_delta = abs(median - float(anchor[index]))
            max_median_delta = max(max_median_delta, median_delta)
            # MDN is an explicit reference whose distribution median need not be the point anchor.
            if case != "mdn":
                _require(median_delta <= BOUNDARY_TOLERANCE,
                         f"{case}: median-preserving calibration moved the native anchor median")
    return {"max_point_delta": max_point_delta, "max_median_delta": max_median_delta,
            "mdn_median_may_differ": case == "mdn"}


def _global_rebuild(config, train, valid, fit_indices, root, seed):
    from scipy.optimize import minimize_scalar

    scale_json = _read_json(root / "global_scale.json")
    _require(scale_json.get("fit_rows") == list(fit_indices),
             "Global scale fit rows differ from the registered optimization cohort")
    _require(float(config["global_scale_min"]) <= float(scale_json["scale"])
             <= float(config["global_scale_max"]), "Global scale is outside its registered bounds")
    anchor_fit = torch.from_numpy(train["anchor"][np.asarray(fit_indices, np.int64)])
    target_fit = torch.from_numpy(train["target"][np.asarray(fit_indices, np.int64)])

    def fit_objective(log_scale):
        scale = math.exp(float(log_scale))
        raw_scale = math.log(math.expm1(max(scale - 1e-4, 1e-12)))
        dist = parameters(anchor_fit, torch.full_like(anchor_fit, raw_scale),
                          torch.zeros_like(anchor_fit), mixture=False)
        return float(objective({"bins": bin_probabilities(dist, BOUNDARIES.astype(np.float64).tolist())},
                               target_fit, anchor_fit, "scale"))

    optimized = minimize_scalar(
        fit_objective,
        bounds=(math.log(config["global_scale_min"]), math.log(config["global_scale_max"])),
        method="bounded", options={"xatol": 1e-8},
    )
    _require(bool(optimized.success), f"Global scale reconstruction failed for seed {seed}")
    rebuilt_scale = math.exp(float(optimized.x))
    _require(math.isclose(float(scale_json["scale"]), rebuilt_scale, rel_tol=1e-8, abs_tol=1e-9),
             f"Global scale differs from independent scalar optimization for seed {seed}")
    _require(math.isclose(float(scale_json["fit_objective"]), float(optimized.fun), rel_tol=1e-8, abs_tol=1e-10),
             f"Global fit objective differs for seed {seed}")
    _require(int(scale_json["calls"]) == int(optimized.nfev),
             f"Global optimizer call count differs for seed {seed}")

    raw = math.log(math.expm1(max(rebuilt_scale - 1e-4, 1e-12)))
    anchor = torch.from_numpy(valid["anchor"])
    dist = parameters(anchor, torch.full_like(anchor, raw), torch.zeros_like(anchor), mixture=False)
    output = {
        "parameters": dist,
        "bins": bin_probabilities(dist, BOUNDARIES.astype(np.float64).tolist()),
        "p_hit": error_band_probability(dist, float(np.float32(.1))),
    }
    values = describe(output)
    metrics = diagnostics(valid["anchor"], valid["target"], values)
    records = _records(valid, values)
    return rebuilt_scale, metrics, records, int(optimized.nfev), float(optimized.fun)


def _loss_vectors(case_records):
    target = np.asarray([row["target"] for row in case_records], np.float32)
    point = np.asarray([row["prediction"] for row in case_records], np.float32)
    hit = (np.abs(point - target) <= np.float32(.1)).astype(np.float64)
    p_hit = np.clip(np.asarray([row["p_hit"] for row in case_records], np.float64), 1e-12, 1 - 1e-12)
    result = {
        "hit_Brier": (p_hit - hit) ** 2,
        "hit_log_score": -(hit * np.log(p_hit) + (1 - hit) * np.log1p(-p_hit)),
    }
    if "bin_probabilities" in case_records[0]:
        probs = np.asarray([row["bin_probabilities"] for row in case_records], np.float64)
        classes = bin_targets(target)
        assigned = probs[np.arange(len(target)), classes]
        result["bin_log_score"] = -np.log(np.maximum(assigned, 1e-12))
    return result


def paired_source_cluster_bootstrap(loss_differences, ids_by_seed, *, reps=DEFAULT_BOOTSTRAP_REPS,
                                    random_seed=BOOTSTRAP_SEED):
    """Paired cluster bootstrap of per-ID candidate-minus-control loss vectors.

    One source resample is shared across seeds for the three-seed average.
    Negative differences favor the candidate.
    """
    if reps < 2000:
        raise ValueError("At least 2,000 fixed bootstrap replicates are required")
    if set(loss_differences) != set(SEEDS) or set(ids_by_seed) != set(SEEDS):
        raise ValueError("Bootstrap must include the three registered seeds")
    canonical_ids = list(ids_by_seed[SEEDS[0]])
    groups = [group_of(sid) for sid in canonical_ids]
    if not canonical_ids or len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("Bootstrap IDs must be nonempty and unique")
    for seed in SEEDS:
        if list(ids_by_seed[seed]) != canonical_ids:
            raise ValueError("Paired bootstrap seeds must use identical canonical VALID IDs")
        values = np.asarray(loss_differences[seed], np.float64)
        if values.shape != (len(canonical_ids),) or not np.isfinite(values).all():
            raise ValueError("Bootstrap loss vectors must be finite and aligned to VALID IDs")
    sources = sorted(set(groups))
    source_index = {source: index for index, source in enumerate(sources)}
    row_groups = np.asarray([source_index[source] for source in groups], np.int64)
    counts = np.bincount(row_groups, minlength=len(sources)).astype(np.float64)
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, len(sources), size=(reps, len(sources)), dtype=np.int32)
    denominators = counts[draws].sum(1)
    per_seed = {}
    boot_by_seed = []
    for seed in SEEDS:
        sums = np.bincount(row_groups, weights=np.asarray(loss_differences[seed], np.float64),
                           minlength=len(sources))
        boot = sums[draws].sum(1) / denominators
        boot_by_seed.append(boot)
        low, high = np.percentile(boot, [2.5, 97.5])
        per_seed[str(seed)] = {
            "estimate": float(np.mean(loss_differences[seed])),
            "ci95": [float(low), float(high)],
            "resampled_sources": len(sources),
            "resamples": reps,
        }
    pooled = np.mean(np.stack(boot_by_seed, axis=0), axis=0)
    pooled_low, pooled_high = np.percentile(pooled, [2.5, 97.5])
    return {
        "by_seed": per_seed,
        "three_seed_same_source_average": {
            "estimate": float(np.mean([np.mean(loss_differences[seed]) for seed in SEEDS])),
            "ci95": [float(pooled_low), float(pooled_high)],
            "resampled_sources": len(sources),
            "resamples": reps,
            "shared_source_resamples_across_seeds": True,
        },
        "direction": "candidate loss minus comparator loss; negative favors disagreement",
        "unit": "official VALID source/video cluster, with all clips retained per sampled source",
        "random_seed": random_seed,
    }


def _stable_bootstrap_seed(control, metric):
    label = f"{control}:{metric}"
    return BOOTSTRAP_SEED + sum((i + 1) * ord(char) for i, char in enumerate(label))


def build_bootstrap_comparisons(records_by_seed_case, ids_by_seed, reps=DEFAULT_BOOTSTRAP_REPS):
    loss_by_seed_case = {
        (seed, case): _loss_vectors(records_by_seed_case[(seed, case)])
        for seed in SEEDS for case in ALL_CASES
    }
    comparisons = {}
    controls = {
        "mixture": ("bin_log_score", "hit_Brier", "hit_log_score"),
        "views": ("bin_log_score", "hit_Brier", "hit_log_score"),
        "scale": ("bin_log_score", "hit_Brier", "hit_log_score"),
        "mdn": ("bin_log_score", "hit_Brier", "hit_log_score"),
        "direct_hit": ("hit_Brier", "hit_log_score"),
    }
    for control, metrics in controls.items():
        comparisons[control] = {}
        for metric in metrics:
            differences = {
                seed: loss_by_seed_case[(seed, "disagreement")][metric]
                       - loss_by_seed_case[(seed, control)][metric]
                for seed in SEEDS
            }
            comparisons[control][metric] = paired_source_cluster_bootstrap(
                differences, ids_by_seed, reps=reps,
                random_seed=_stable_bootstrap_seed(control, metric),
            )
    return {"replicates": reps, "candidate": "disagreement", "comparisons": comparisons}


def summarize_registered_gates(bootstrap, point_invariance):
    comparisons = bootstrap["comparisons"]

    def beats_all_seeds_and_pooled(control, metric):
        summary = comparisons[control][metric]
        seed_negative = all(summary["by_seed"][str(seed)]["estimate"] < 0 for seed in SEEDS)
        pooled_upper = summary["three_seed_same_source_average"]["ci95"][1]
        return seed_negative and pooled_upper < 0

    def no_clear_disadvantage(control, metric):
        summary = comparisons[control][metric]
        pooled_lower = summary["three_seed_same_source_average"]["ci95"][0]
        per_seed_lower = [summary["by_seed"][str(seed)]["ci95"][0] for seed in SEEDS]
        # No registered noninferiority margin exists; this only means the 95% interval
        # does not establish that disagreement is worse, not that the methods are equivalent.
        return pooled_lower <= 0 and all(value <= 0 for value in per_seed_lower)

    gates = {
        "primary_bin_score_vs_mixture": {
            "pass": beats_all_seeds_and_pooled("mixture", "bin_log_score"),
            "criterion": "Disagreement bin log loss is lower in each seed and the paired three-seed same-source 95% interval is wholly below zero.",
        },
        "error_band_brier_vs_scale_and_mixture": {
            "pass": all(beats_all_seeds_and_pooled(control, "hit_Brier") for control in ("scale", "mixture")),
            "criterion": "Disagreement hit Brier is lower in each seed and the paired three-seed interval is wholly below zero versus both Scale and Mixture.",
        },
        "view_control_consistency": {
            "pass": all(beats_all_seeds_and_pooled("views", metric)
                         for metric in ("bin_log_score", "hit_Brier")),
            "criterion": "Disagreement beats Views on bin log loss and hit Brier in each seed, with both pooled paired intervals wholly below zero.",
        },
        "mdn_competitiveness": {
            "pass": None,
            "no_clear_disadvantage_detected": all(no_clear_disadvantage("mdn", metric)
                                                  for metric in ("bin_log_score", "hit_Brier", "hit_log_score")),
            "criterion": "Not established: no noninferiority margin was registered. An interval containing zero is not evidence of competitiveness/equivalence; retain the descriptive contrasts.",
        },
        "direct_hit_competitiveness": {
            "pass": all(beats_all_seeds_and_pooled("direct_hit", metric)
                         for metric in ("hit_Brier", "hit_log_score")),
            "criterion": "Supported by superiority on both hit scores: lower in each seed and paired three-seed intervals wholly below zero. No post-hoc noninferiority margin is used.",
        },
        "point_invariance": {
            "pass": bool(point_invariance),
            "criterion": "Every case's saved/recomputed native point prediction and MAE match the fixed anchor within 1e-6; MDN distribution median is allowed to differ.",
        },
    }
    gate_results = {name: bool(value["pass"]) for name, value in gates.items()}
    return {
        "gates": gates,
        "all_score_gates_pass": all(gate_results.values()),
        "promotion_eligible": False,
        "promotion_blocker": "Official VALID results are development evidence, not source-transfer evidence; the registered nested outer-source rebuild and MOSEI replication remain outstanding.",
        "decision_scope": "No TEST fitting, tuning, scoring, or claim of nested source transfer.",
    }


def _validate_config(config):
    fixed = {
        "dataset": "simsv2",
        "seeds": list(SEEDS),
        "cases": list(ALL_CASES),
        "device": "cpu",
        "hidden": 32,
        "dropout": .1,
        "max_epochs": 100,
        "patience": 12,
        "batch_size": 64,
        "learning_rate": .001,
        "weight_decay": .0001,
        "gradient_norm": 1.,
        "kernel_scale": .05,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Calibration config {key} differs from the registered formal value")
    if not 0 < config.get("global_scale_min", 0) < config.get("global_scale_max", 0):
        raise ValueError("Invalid registered Global scale bounds")


def audit_suite(*, suite_path, config_path, output_path, bootstrap_reps=DEFAULT_BOOTSTRAP_REPS,
                project_root=PROJECT_ROOT):
    if bootstrap_reps < 2000:
        raise ValueError("At least 2,000 fixed bootstrap replicates are required")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    suite_path = _resolve(suite_path, project_root)
    config_path = _resolve(config_path, project_root)
    output_path = _resolve(output_path, project_root)
    _require(not output_path.exists(), f"Audit output already exists; refusing overwrite: {output_path}")
    _require(suite_path.is_file() and config_path.is_file(), "Formal suite and config files are required")
    config = _read_json(config_path)
    _validate_config(config)
    suite = _read_json(suite_path)
    _require(suite.get("suite_complete") is True and suite.get("smoke_only") is False,
             "Only a complete formal calibration suite can be audited")
    _require(len(suite.get("results", [])) == len(ALL_CASES) * len(SEEDS),
             "Formal suite must contain all 21 case/seed results")
    expected = {(case, seed) for case in ALL_CASES for seed in SEEDS}
    keys = [(row.get("case"), row.get("seed")) for row in suite["results"]]
    _require(len(keys) == len(set(keys)) and set(keys) == expected,
             "Formal suite is subset-selected, duplicated, or missing cases/seeds")
    _require(all(row.get("complete") is True and row.get("smoke_only") is False
                 and row.get("test_evaluated") is False for row in suite["results"]),
             "A calibration result is incomplete, smoke-only, or TEST-evaluated")

    table_dir = _resolve(config["table_dir"], project_root)
    metadata_path = table_dir / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata_hash = digest(metadata_path)
    _require(metadata.get("dataset") == "simsv2" and metadata.get("seeds") == list(SEEDS)
             and metadata.get("train_count") == 2722 and metadata.get("valid_count") == 647
             and metadata.get("source_count") == 129,
             "Calibration table metadata differs from registered CH-SIMS counts")
    _require(metadata.get("test_evaluated") is False,
             "Calibration table metadata reports TEST access")
    _require(metadata.get("official_validation_labels_parsed_for_teacher_stage") is False,
             "Teacher stage metadata reports official VALID label parsing")
    _require(metadata.get("view_order") == ["text", "audio", "vision"]
             and metadata.get("point_anchor_semantics", {}).get("train")
             == "clipped fusion-view source-CV OOF prediction"
             and metadata.get("point_anchor_semantics", {}).get("valid")
             == "clipped frozen official early_fusion VALID prediction"
             and metadata.get("point_anchor_semantics", {}).get("full_fusion_teacher_valid_excluded") is True
             and metadata.get("point_anchor_semantics", {}).get("full_train_teacher_predictions_excluded_from_calibration") is True,
             "Calibration tables do not preserve the registered point-anchor/view semantics")
    expected_artifacts = {
        f"seed_{seed}/{name}"
        for seed in SEEDS for name in ("train.npz", "valid.npz", "train_ids.json", "valid_ids.json")
    }
    _require(set(metadata.get("artifact_sha256", {})) == expected_artifacts,
             "Calibration table artifact hash manifest is incomplete or contains extras")
    _require("not a fully nested pipeline" in metadata.get("source_group_calibration_selection_limit", ""),
             "Calibration metadata lost the nested source-transfer limitation")
    split = metadata["calibration_split"]

    tables_by_seed = {}
    ids_by_seed = {seed: None for seed in SEEDS}
    canonical_valid_ids = None
    stats_by_seed = {}
    for seed in SEEDS:
        train, train_meta = load_table(table_dir, seed, "train")
        valid, valid_meta = load_table(table_dir, seed, "valid")
        _require(train_meta == metadata and valid_meta == metadata,
                 f"Calibration table metadata changed for seed {seed}")
        _require(len(train["ids"]) == 2722 and len(valid["ids"]) == 647,
                 f"Formal table counts differ for seed {seed}")
        expected_feature_width = sum(metadata["feature_dimensions"].values())
        _require(train["features"].shape == (2722, expected_feature_width)
                 and valid["features"].shape == (647, expected_feature_width),
                 f"Pooled T/A/V feature dimensions differ for seed {seed}")
        _require(np.all(np.abs(train["target"]) <= 1) and np.all(np.abs(valid["target"]) <= 1),
                 f"Native CH-SIMS target escaped its registered range for seed {seed}")
        _require(np.all(np.abs(train["anchor"]) <= 1) and np.all(np.abs(valid["anchor"]) <= 1)
                 and np.all(np.abs(train["views"]) <= 1) and np.all(np.abs(valid["views"]) <= 1),
                 f"Calibrated input anchor/view predictions are not clipped for seed {seed}")
        if seed == SEEDS[0]:
            ids_by_seed[seed] = train["ids"]
            canonical_valid_ids = valid["ids"]
        else:
            _require(train["ids"] == ids_by_seed[SEEDS[0]],
                     "Canonical TRAIN ID order differs across seeds")
            _require(valid["ids"] == canonical_valid_ids,
                     "Canonical VALID ID order differs across seeds")
            ids_by_seed[seed] = train["ids"]
        tables_by_seed[seed] = {"train": train, "valid": valid}
        fit_indices, select_indices = validate_head_split(train["ids"], split)
        _compare_payload(split, expected_head_split(train["ids"]), tolerance=0,
                         path="calibration_split")
        cohorts, stats = prepare_training(train, metadata)
        _require(np.array_equal(cohorts[0], np.asarray(fit_indices, np.int64))
                 and np.array_equal(cohorts[1], np.asarray(select_indices, np.int64)),
                 "Training helper cohorts differ from metadata fit/select indices")
        _verify_fit_only_statistics(train, fit_indices, stats)
        stats_by_seed[seed] = stats
    for seed in SEEDS[1:]:
        _require(tables_by_seed[seed]["train"]["target"].tobytes()
                 == tables_by_seed[SEEDS[0]]["train"]["target"].tobytes(),
                 "TRAIN original M targets differ across seed table copies")
        _require(tables_by_seed[seed]["valid"]["target"].tobytes()
                 == tables_by_seed[SEEDS[0]]["valid"]["target"].tobytes(),
                 "VALID original M targets differ across seed table copies")
        _require(tables_by_seed[seed]["train"]["features"].tobytes()
                 == tables_by_seed[SEEDS[0]]["train"]["features"].tobytes(),
                 "TRAIN pooled features differ across seed table copies")
        _require(tables_by_seed[seed]["valid"]["features"].tobytes()
                 == tables_by_seed[SEEDS[0]]["valid"]["features"].tobytes(),
                 "VALID pooled features differ across seed table copies")
    verified_raw_feature_hashes = _verify_raw_feature_hashes(metadata)
    original_csv_hash = _verify_author_targets(
        metadata,
        {seed: tables_by_seed[seed]["train"] for seed in SEEDS},
        {seed: tables_by_seed[seed]["valid"] for seed in SEEDS},
    )

    expected_metadata_hash = metadata_hash
    names = ("anchor_calibration_training.py", "anchor_calibration.py", "anchor_cdf.py",
             "native_metrics.py", "native_training.py")
    code_paths = {name: PROJECT_ROOT / "jevaffect" / name for name in names}
    current_code = {name: digest(path) for name, path in code_paths.items()}
    results_map = {(row["case"], row["seed"]): row for row in suite["results"]}
    records_by_seed_case = {}
    case_results = {}
    checkpoint_hashes = {}
    prediction_hashes = {}
    identity_hashes = {}
    result_hashes = {}
    history_hashes = {}
    global_fit_hashes = {}
    selected_metrics = {}
    point_invariance = True
    native_point_reference = {}
    source_fit_select = {str(seed): {
        "fit_count": len(split["fit_indices"]),
        "select_count": len(split["select_indices"]),
        "fit_source_count": len(split["fit_sources"]),
        "select_source_count": len(split["select_sources"]),
        "fit_select_source_disjoint": True,
    } for seed in SEEDS}
    model_parameters = {}

    for case in ALL_CASES:
        case_results[case] = {}
        for seed in SEEDS:
            summary = results_map[(case, seed)]
            root = _resolve(summary["root"], project_root)
            expected_root = _resolve(config["run_root"], project_root) / "formal" / case / f"seed_{seed}"
            _require(root == expected_root.resolve(), f"Unexpected calibration run root for {case}/seed_{seed}")
            result = _read_json(root / "result.json")
            _require(result == summary, f"Suite result summary changed for {case}/seed_{seed}")
            result_hashes[f"{case}/seed_{seed}"] = digest(root / "result.json")
            identity_path = root / "identity.json"
            identity = _read_json(identity_path)
            _require(identity.get("config") == config and identity.get("case") == case
                     and identity.get("seed") == seed and identity.get("smoke_only") is False,
                     f"Calibration config/case/seed identity mismatch for {case}/seed_{seed}")
            _require(identity.get("metadata_sha256") == expected_metadata_hash,
                     f"Calibration table metadata hash mismatch for {case}/seed_{seed}")
            _require(identity.get("implementation") == current_code,
                     f"Calibration code hashes differ for {case}/seed_{seed}")
            _require(identity.get("runtime", {}).get("device") == "cpu",
                     f"Calibration result was not recorded as CPU for {case}/seed_{seed}")
            _require(identity.get("test_evaluated") is False,
                     f"Calibration identity reports TEST evaluation for {case}/seed_{seed}")
            expected_selection = "TRAIN OOF source-inner Bernoulli score" if case == "direct_hit" else "TRAIN OOF source-inner native-bin score"
            _require(identity.get("selection") == expected_selection,
                     f"Calibration selection objective differs for {case}/seed_{seed}")
            _require(identity.get("scope") == "head-inner selection is not nested full-pipeline source transfer",
                     f"Calibration source-transfer scope differs for {case}/seed_{seed}")
            _compare_payload(identity.get("normalization"), stats_by_seed[seed], tolerance=1e-8,
                             path=f"{case}/seed_{seed}.normalization")
            _verify_fit_only_statistics(tables_by_seed[seed]["train"], split["fit_indices"],
                                        identity["normalization"])
            identity_hashes[f"{case}/seed_{seed}"] = digest(identity_path)

            train = tables_by_seed[seed]["train"]
            valid = tables_by_seed[seed]["valid"]
            fit_indices = np.asarray(split["fit_indices"], np.int64)
            history = []
            if case == "global":
                _require(result["parameters"] == 1 and result["complete"] is True,
                         f"Global scale fit parameters/completion mismatch for seed {seed}")
                scale, metrics, records, calls, fit_objective = _global_rebuild(
                    config, train, valid, fit_indices, root, seed
                )
                _require(result["optimizer_calls"] == calls,
                         f"Global optimizer call count mismatch for seed {seed}")
                _require(math.isclose(float(result["scale"]), scale, rel_tol=1e-8, abs_tol=1e-9),
                         f"Global result scale differs from rebuilt value for seed {seed}")
                model_parameters[f"global/seed_{seed}"] = 1
                selection_summary = {"fit_rows": len(fit_indices), "fit_objective": fit_objective,
                                     "optimizer_calls": calls, "scale": scale}
                checkpoint_hashes[f"global/seed_{seed}"] = None
                global_fit_hashes[f"global/seed_{seed}"] = digest(root / "global_scale.json")
            else:
                history = _read_json(root / "history.json")
                history_hashes[f"{case}/seed_{seed}"] = digest(root / "history.json")
                _require(history and [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
                         f"Training history epoch sequence is invalid for {case}/seed_{seed}")
                epoch, best_objective = first_minimum_epoch(history)
                _require(result["best_epoch"] == epoch and result["epochs_executed"] == len(history),
                         f"First-minimum epoch or budget differs for {case}/seed_{seed}")
                updates = sum(int(row["updates"]) for row in history)
                expected_updates = math.ceil(len(split["fit_indices"]) / config["batch_size"])
                _require(result["optimizer_updates"] == updates and updates > 0
                         and all(row["updates"] == expected_updates for row in history),
                         f"Optimizer update count differs for {case}/seed_{seed}")
                loop_seconds = sum(float(row["seconds"]) for row in history)
                _require(math.isclose(float(result["loop_seconds"]), loop_seconds, rel_tol=1e-12, abs_tol=1e-9),
                         f"Training time differs for {case}/seed_{seed}")
                best_path = root / "best.pt"
                last_path = root / "last.pt"
                _require(best_path.is_file() and last_path.is_file(),
                         f"Best/last checkpoint is missing for {case}/seed_{seed}")
                best_hash = digest(best_path)
                _require(best_hash == result["checkpoint_sha256"],
                         f"Best checkpoint hash differs for {case}/seed_{seed}")
                selected = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
                last = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
                complete_from_state = len(history) >= config["max_epochs"] or last["stale"] >= config["patience"]
                _require(result["complete"] is complete_from_state and complete_from_state,
                         f"Checkpoint completion rule differs for {case}/seed_{seed}")
                _require(selected.get("identity") == identity and last.get("identity") == identity,
                         f"Best/last checkpoint identity differs for {case}/seed_{seed}")
                _require(selected.get("epoch") == epoch and last.get("epoch") == len(history)
                         and last.get("history") == history,
                         f"Best/last checkpoint epochs or history differ for {case}/seed_{seed}")
                _require(math.isclose(float(selected["selection_objective"]), best_objective,
                                      rel_tol=1e-9, abs_tol=1e-10)
                         and math.isclose(float(last["best"]), best_objective, rel_tol=1e-9, abs_tol=1e-10),
                         f"Selected objective differs from first-minimum history for {case}/seed_{seed}")
                _require(set(("optimizer", "torch_rng", "numpy_rng", "python_rng", "loader_rng"))
                         <= set(last), f"Last checkpoint lacks resume state for {case}/seed_{seed}")
                parameter_count = sum(int(p.numel()) for p in selected["model"].values())
                _require(parameter_count == result["parameters"],
                         f"Model parameter count differs for {case}/seed_{seed}")
                active_gradients = result.get("first_batch_active_gradient_parameters")
                _require(isinstance(active_gradients, int) and 0 < active_gradients <= parameter_count
                         and last["active_gradient_parameters"] == active_gradients,
                         f"Active gradient count is missing or invalid for {case}/seed_{seed}")
                model_parameters[f"{case}/seed_{seed}"] = parameter_count
                cohorts, stats = prepare_training(train, metadata)
                model = CalibrationHead(train["features"].shape[1] + 1,
                                        config["hidden"], config["dropout"]).to("cpu")
                model.load_state_dict(selected["model"])
                if case in CASES:
                    select_ds = torch.utils.data.Subset(tensors(train, stats, case), cohorts[1].tolist())
                    select_loader = torch.utils.data.DataLoader(select_ds, batch_size=config["batch_size"],
                                                                 shuffle=False, num_workers=0)
                    selected_objective = _selection(model, select_loader, case, torch.device("cpu"))
                    _require(math.isclose(selected_objective, best_objective, rel_tol=1e-8, abs_tol=1e-9),
                             f"Selected checkpoint objective does not recompute for {case}/seed_{seed}")
                metrics, records = evaluate(model, valid, stats, case,
                                            config["batch_size"], device="cpu")
                del model, selected, last
                checkpoint_hashes[f"{case}/seed_{seed}"] = {
                    "best": best_hash, "last": digest(last_path),
                }
                selection_summary = {
                    "best_epoch": epoch,
                    "epochs_executed": len(history),
                    "best_selection_objective": best_objective,
                    "optimizer_updates": updates,
                    "trainable_parameters": parameter_count,
                    "first_batch_active_gradient_parameters": active_gradients,
                    "loop_seconds": loop_seconds,
                    "history_sha256": history_hashes[f"{case}/seed_{seed}"],
                }

            pred_path = root / "valid_predictions.json"
            pred_hash = digest(pred_path)
            _require(pred_hash == result["predictions_sha256"],
                     f"VALID prediction hash differs for {case}/seed_{seed}")
            saved_records = _read_json(pred_path)
            _require(len(saved_records) == 647 and len({row["id"] for row in saved_records}) == 647,
                     f"VALID prediction IDs/count differ for {case}/seed_{seed}")
            _require([row["id"] for row in saved_records] == valid["ids"],
                     f"VALID prediction ID order differs for {case}/seed_{seed}")
            if case == "global":
                _compare_payload(saved_records, records, tolerance=1e-7, path=f"{case}/seed_{seed}.predictions")
                _compare_payload(result["valid"], metrics, tolerance=1e-7, path=f"{case}/seed_{seed}.metrics")
            else:
                # `evaluate` is the same frozen inference helper; compare every stored
                # per-ID prediction, probability, interval, and metric after CPU reload.
                _compare_payload(saved_records, records, tolerance=1e-7, path=f"{case}/seed_{seed}.predictions")
                _compare_payload(result["valid"], metrics, tolerance=1e-7, path=f"{case}/seed_{seed}.metrics")
            point_checks = _verify_probabilities_and_point(case, records, valid["anchor"])
            point_invariance = point_invariance and point_checks["max_point_delta"] <= BOUNDARY_TOLERANCE
            native = metrics["native"]
            if seed not in native_point_reference:
                native_point_reference[seed] = native
            else:
                _compare_payload(native, native_point_reference[seed], tolerance=BOUNDARY_TOLERANCE,
                                 path=f"{case}/seed_{seed}.native_point_metrics")
            records_by_seed_case[(seed, case)] = records
            prediction_hashes[f"{case}/seed_{seed}"] = pred_hash
            selected_metrics[f"{case}/seed_{seed}"] = {
                "valid": metrics,
                "point_checks": point_checks,
                "prediction_count": len(records),
                "prediction_hash": pred_hash,
                "selection": selection_summary,
            }
            case_results[case][str(seed)] = {
                "complete": True,
                "best_epoch": selection_summary.get("best_epoch"),
                "epochs_executed": selection_summary.get("epochs_executed"),
                "optimizer_updates": selection_summary.get("optimizer_updates"),
                "parameters": model_parameters[f"{case}/seed_{seed}"],
                "first_batch_active_gradient_parameters": selection_summary.get("first_batch_active_gradient_parameters"),
                "loop_seconds": selection_summary.get("loop_seconds"),
                "prediction_count": len(records),
                "checkpoint_sha256": checkpoint_hashes[f"{case}/seed_{seed}"],
                "prediction_sha256": pred_hash,
                "valid_metrics": metrics,
                "max_point_delta": point_checks["max_point_delta"],
                "max_median_delta": point_checks["max_median_delta"],
            }

    bootstrap = build_bootstrap_comparisons(records_by_seed_case,
                                            {seed: tables_by_seed[seed]["valid"]["ids"] for seed in SEEDS},
                                            reps=bootstrap_reps)
    gates = summarize_registered_gates(bootstrap, point_invariance)
    report = {
        "scope": "Formal CH-SIMS native AnchorCDF calibration audit; official VALID is development evidence only.",
        "suite": str(suite_path),
        "suite_sha256": digest(suite_path),
        "config": str(config_path),
        "config_sha256": digest(config_path),
        "table_dir": str(table_dir),
        "table_metadata_sha256": metadata_hash,
        "table_artifact_sha256": metadata["artifact_sha256"],
        "author_csv_sha256": original_csv_hash,
        "raw_feature_hashes_verified": verified_raw_feature_hashes,
        "dataset": "simsv2",
        "cases": list(ALL_CASES),
        "seeds": list(SEEDS),
        "formal_results_audited": len(expected),
        "counts": {"train": 2722, "valid": 647, "source_groups": 129},
        "source_head_fit_select": source_fit_select,
        "selection_scope": "TRAIN OOF source-inner split; other source-CV teachers may have seen heldout calibration sources.",
        "teacher_results": case_results,
        "training_code_sha256": current_code,
        "audit_script_sha256": digest(Path(__file__)),
        "checkpoint_sha256": checkpoint_hashes,
        "identity_json_sha256": identity_hashes,
        "result_json_sha256": result_hashes,
        "history_json_sha256": history_hashes,
        "global_fit_artifact_sha256": global_fit_hashes,
        "valid_prediction_sha256": prediction_hashes,
        "recomputed_valid_metrics": selected_metrics,
        "paired_source_cluster_bootstrap": bootstrap,
        "registered_plan_gates": gates,
        "official_valid_is_development_evidence": True,
        "nested_source_transfer_verified": False,
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True,
                        help="Formal 21-result suite JSON")
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs/native_anchor_calibration_v1.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="New audit JSON path; existing files are rejected")
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS,
                        help="Fixed-seed paired source-cluster draws; minimum 2000")
    args = parser.parse_args()
    result = audit_suite(suite_path=args.suite, config_path=args.config,
                         output_path=args.output, bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({
        "output": str(_resolve(args.output)),
        "formal_results_audited": result["formal_results_audited"],
        "all_score_gates_pass": result["registered_plan_gates"]["all_score_gates_pass"],
        "promotion_eligible": result["registered_plan_gates"]["promotion_eligible"],
        "test_evaluated": False,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
