"""Audit the formal AnchorCDF teachers and package TRAIN/VALID calibration tables.

This is a CPU-only preparation step. It reads completed checkpoints and saved
predictions, never runs an optimizer, and refuses smoke, partial, or overwritten
outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import NativeSplit, author_rows, digest, save_json
from jevaffect.source_crossfit import group_of, manifest_hash


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
SEEDS = (13, 42, 2026)
VIEWS = ("fusion", "text", "audio", "vision")
VIEW_MODALITIES = {
    "fusion": ("text", "audio", "vision"),
    "text": ("text",),
    "audio": ("audio",),
    "vision": ("vision",),
}
FOLDS = (0, 1, 2, "full")
MODALITIES = ("text", "audio", "vision")
CALIBRATION_SEED = 20261013
CALIBRATION_FRACTION = .1
EXPECTED_DIMS = {"text": 768, "audio": 768, "vision": 384}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _resolve(path, base=PROJECT_ROOT):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_predictions(path, columns):
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames == list(columns), f"Unexpected prediction columns in {path}")
        records = []
        for row in reader:
            record = {"id": row["id"]}
            for key in columns[1:]:
                value = float(row[key])
                _require(math.isfinite(value), f"Nonfinite {key} in {path}")
                record[key] = value
            records.append(record)
    return records


def _target_matches(actual, expected):
    return np.isfinite(actual) and np.float32(actual) == np.float32(expected)


def _normalized_feature_hashes(hashes):
    """Drop only the cache-root prefix, preserving split and artifact names."""
    normalized = {}
    for key, value in hashes.items():
        path = str(key).replace("\\", "/")
        if path.endswith("/metadata.json") or path == "metadata.json":
            relative = "metadata.json"
        elif path.endswith("/meta.json") or path == "meta.json":
            relative = "meta.json"
        else:
            pieces = path.split("/")
            candidates = ["/".join(pieces[index:]) for index, piece in enumerate(pieces)
                          if piece in ("train", "valid")]
            if candidates:
                relative = candidates[-1]
            elif path.endswith("/ids.json") or path == "ids.json":
                relative = "ids.json"
            else:
                relative = path
        if relative in normalized:
            raise ValueError("Feature hash keys collide after cache-root normalization")
        normalized[relative] = value
    return normalized


def validate_source_folds(ids, folds):
    """Validate the exact TRAIN partition and source isolation of all three folds."""
    ids = list(ids)
    universe = set(ids)
    _require(len(ids) == len(universe), "Canonical TRAIN IDs contain duplicates")
    _require(len(folds) == 3, "Exactly three source folds are required")
    seen_oof = []
    held_source_fold = {}
    for number, fold in enumerate(folds):
        _require(fold.get("fold") == number, "Fold numbers are missing or out of order")
        roles = (fold["fit_ids"], fold["select_ids"], fold["oof_ids"])
        _require(all(roles), "Each source fold requires nonempty fit/select/oof cohorts")
        sets = [set(role) for role in roles]
        _require(all(len(values) == len(role) for values, role in zip(sets, roles)),
                 "A source fold contains duplicate IDs")
        _require(set.union(*sets) == universe, "A source fold does not partition all TRAIN IDs")
        source_sets = [{group_of(sid) for sid in role} for role in roles]
        _require(not (source_sets[0] & source_sets[1] or source_sets[0] & source_sets[2]
                      or source_sets[1] & source_sets[2]), "Fit/select/oof sources overlap")
        for sid in fold["oof_ids"]:
            source = group_of(sid)
            previous = held_source_fold.setdefault(source, number)
            _require(previous == number, "A source is held out by more than one fold")
        seen_oof.extend(fold["oof_ids"])
    _require(len(seen_oof) == len(universe) and set(seen_oof) == universe,
             "OOF source cohorts do not cover every TRAIN ID exactly once")
    _require(len(held_source_fold) == len({group_of(sid) for sid in ids}),
             "Every TRAIN source must have one held-source fold")


def make_calibration_split(ids, seed=CALIBRATION_SEED, inner_fraction=CALIBRATION_FRACTION):
    """Choose one deterministic TRAIN-only, source-disjoint calibration holdout."""
    ids = list(ids)
    if len(set(ids)) != len(ids) or not 0 < inner_fraction < 1:
        raise ValueError("Invalid calibration split request")
    groups = sorted({group_of(sid) for sid in ids})
    if len(groups) < 3:
        raise ValueError("Too few source groups for calibration selection")
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(groups).tolist()
    select_count = max(1, int(np.ceil(len(groups) * inner_fraction)))
    if select_count >= len(groups):
        raise ValueError("Calibration split leaves no fit sources")
    select_sources = set(ordered[:select_count])
    fit_sources = set(groups) - select_sources
    fit_ids = [sid for sid in ids if group_of(sid) in fit_sources]
    select_ids = [sid for sid in ids if group_of(sid) in select_sources]
    positions = {sid: index for index, sid in enumerate(ids)}
    fit_indices = [positions[sid] for sid in fit_ids]
    select_indices = [positions[sid] for sid in select_ids]
    if set(fit_indices) & set(select_indices) or set(fit_indices) | set(select_indices) != set(range(len(ids))):
        raise ValueError("Calibration indices do not partition TRAIN")
    if fit_sources & select_sources:
        raise ValueError("Calibration source groups overlap")
    return {
        "seed": seed,
        "inner_fraction": inner_fraction,
        "source_count": len(groups),
        "select_source_count": len(select_sources),
        "fit_sources": sorted(fit_sources),
        "select_sources": sorted(select_sources),
        "fit_ids": fit_ids,
        "select_ids": select_ids,
        "fit_indices": fit_indices,
        "select_indices": select_indices,
    }


def merge_oof_records(fold_records, folds, ids, targets):
    """Merge fold prediction records in canonical order, rejecting bad coverage/labels."""
    ids = list(ids)
    validate_source_folds(ids, folds)
    if set(fold_records) != {0, 1, 2}:
        raise ValueError("OOF records must contain exactly folds 0, 1, and 2")
    merged_by_id = {}
    for number in range(3):
        records = fold_records[number]
        actual_ids = [record.get("id") for record in records]
        expected_ids = folds[number]["oof_ids"]
        if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
            raise ValueError(f"Fold {number} OOF IDs are missing, extra, or duplicated")
        for record in records:
            sid = record["id"]
            if sid in merged_by_id:
                raise ValueError("An ID appears in more than one OOF fold")
            if sid not in targets or not _target_matches(float(record["target"]), targets[sid]):
                raise ValueError(f"OOF target differs from the pinned native M target for {sid}")
            prediction = float(record["prediction"])
            if not math.isfinite(prediction):
                raise ValueError(f"Nonfinite OOF prediction for {sid}")
            merged_by_id[sid] = {
                "id": sid,
                "target": float(np.float32(targets[sid])),
                "prediction": prediction,
            }
    if set(merged_by_id) != set(ids) or len(merged_by_id) != len(ids):
        raise ValueError("Merged OOF does not contain every canonical TRAIN ID exactly once")
    return [merged_by_id[sid] for sid in ids]


def _validate_full_selection(ids, cohort):
    fit_ids, select_ids = cohort["fit_ids"], cohort["select_ids"]
    _require(set(fit_ids) | set(select_ids) == set(ids), "Full teacher split does not cover TRAIN")
    _require(not (set(fit_ids) & set(select_ids)), "Full teacher IDs overlap")
    fit_sources, select_sources = set(cohort["fit_sources"]), set(cohort["select_sources"])
    _require(fit_sources == {group_of(sid) for sid in fit_ids}, "Full fit source list is inconsistent")
    _require(select_sources == {group_of(sid) for sid in select_ids}, "Full select source list is inconsistent")
    _require(not (fit_sources & select_sources), "Full teacher sources overlap")
    _require(fit_sources | select_sources == {group_of(sid) for sid in ids},
             "Full teacher sources do not cover TRAIN")


def _validate_teacher(root, summary_result, cohort, fold_manifest, full_manifest,
                      config, manifest, source_csv_hash, train_feature_hashes,
                      valid_feature_hashes, project_root):
    root = _resolve(summary_result["root"], project_root)
    expected_fold = summary_result["fold"]
    _require(root.is_dir(), f"Missing formal teacher directory: {root}")
    result = _read_json(root / "result.json")
    _require(result == summary_result, f"Suite summary and saved result differ: {root}")
    _require(result["complete"] and not result["smoke_only"] and not result["test_evaluated"],
             f"Teacher is incomplete, smoke-only, or has TEST activity: {root}")
    _require(not result.get("official_validation_labels_parsed", True),
             f"Teacher identity reports parsing official VALID labels: {root}")

    identity_path = root / "identity.json"
    identity = _read_json(identity_path)
    _require(identity["config"] == config, f"Teacher configuration identity mismatch: {root}")
    _require(identity["dataset"] == "simsv2" and identity["seed"] == summary_result["seed"],
             f"Teacher dataset/seed identity mismatch: {root}")
    _require(identity["view"] == summary_result["view"] and identity["fold"] == expected_fold,
             f"Teacher view/fold identity mismatch: {root}")
    _require(identity["modalities"] == list(VIEW_MODALITIES[summary_result["view"]]),
             f"Teacher input modalities mismatch: {root}")
    _require(identity["cohort"] == cohort, f"Teacher cohort identity mismatch: {root}")
    _require(identity["manifest_sha256"] == manifest_hash(cohort),
             f"Teacher selected-cohort manifest hash mismatch: {root}")
    _require(identity["author_csv_sha256"] == source_csv_hash,
             f"Teacher source CSV identity mismatch: {root}")
    _require(identity["train_ids"] == manifest["ids"], f"Teacher canonical TRAIN IDs differ: {root}")
    _require(_normalized_feature_hashes(identity["feature_hashes"])
             == _normalized_feature_hashes(train_feature_hashes),
             f"Teacher TRAIN feature identity mismatch: {root}")
    _require(_normalized_feature_hashes(identity["valid_feature_hashes"])
             == _normalized_feature_hashes(valid_feature_hashes),
             f"Teacher VALID feature identity mismatch: {root}")
    _require(identity["fold_manifest_sha256"] == manifest_hash(fold_manifest),
             f"Teacher source-fold manifest hash mismatch: {root}")
    _require(identity["full_selection_manifest_sha256"] == manifest_hash(full_manifest),
             f"Teacher full-selection manifest hash mismatch: {root}")
    if isinstance(expected_fold, int):
        _require(summary_result["manifest_sha256"] == manifest_hash(fold_manifest),
                 f"Teacher result source-fold manifest hash mismatch: {root}")
    _require(identity["official_validation_labels_parsed"] is False and identity["test_evaluated"] is False,
             f"Teacher identity contains forbidden split access: {root}")
    _require(identity["fixture_smoke"] is False and identity["smoke_groups"] is None,
             f"Formal teacher identity is marked as fixture/smoke: {root}")
    code_paths = {
        "anchor_crossfit.py": PROJECT_ROOT / "jevaffect/anchor_crossfit.py",
        "native_training.py": PROJECT_ROOT / "jevaffect/native_training.py",
        "source_crossfit.py": PROJECT_ROOT / "jevaffect/source_crossfit.py",
        "native_metrics.py": PROJECT_ROOT / "jevaffect/native_metrics.py",
    }
    actual_code = {name: digest(path) for name, path in code_paths.items()}
    _require(identity["implementation_sha256"] == actual_code,
             f"Teacher source-code hashes differ from the saved identity: {root}")
    _require(identity["selection"] == "minimum inner-selection native MAE; first checkpoint wins ties",
             f"Unexpected inner-selection rule: {root}")
    _require(identity["loss"] == "unclipped L1 on original CH-SIMS M scores",
             f"Unexpected point-teacher objective: {root}")

    history = _read_json(root / "history.json")
    _require(history and [row["epoch"] for row in history] == list(range(1, len(history) + 1)),
             f"Teacher history epochs are incomplete or out of order: {root}")
    maes = [float(row["inner_select"]["MAE"]) for row in history]
    _require(all(math.isfinite(value) for value in maes), f"Nonfinite inner-selection MAE: {root}")
    selected_index = min(range(len(maes)), key=lambda index: maes[index])
    selected_row = history[selected_index]
    _require(result["best_epoch"] == selected_row["epoch"],
             f"Checkpoint is not the first minimum inner-selection MAE epoch: {root}")
    _require(result["epochs_executed"] == len(history), f"Teacher epoch count mismatch: {root}")
    update_count = sum(int(row["optimizer_updates"]) for row in history)
    _require(result["optimizer_updates"] == update_count, f"Teacher optimizer update count mismatch: {root}")
    _require(math.isclose(float(result["loop_seconds"]), sum(float(row["seconds"]) for row in history),
                          rel_tol=1e-12, abs_tol=1e-9), f"Teacher loop time mismatch: {root}")

    last_path, best_path = root / "last.pt", root / "best.pt"
    _require(last_path.is_file() and best_path.is_file(), f"Missing best/last checkpoint: {root}")
    last_checkpoint_hash = digest(last_path)
    _require(digest(best_path) == result["checkpoint_sha256"], f"Best checkpoint hash mismatch: {root}")
    last = torch.load(last_path, map_location="cpu", weights_only=False, mmap=True)
    _require(last["identity"] == identity and last["history"] == history,
             f"Last checkpoint identity/history mismatch: {root}")
    _require(last["epoch"] == len(history), f"Last checkpoint epoch mismatch: {root}")
    _require(math.isclose(float(last["best"]), min(maes), rel_tol=0, abs_tol=0),
             f"Last checkpoint best score mismatch: {root}")
    _require(set(("optimizer", "scheduler", "torch_rng", "numpy_rng", "python_rng", "loader_rng")) <= set(last),
             f"Last checkpoint is missing resume state: {root}")
    complete_from_state = last["stale"] >= config["patience"] or len(history) >= config["max_epochs"]
    _require(complete_from_state and result["complete"], f"Teacher stopped before its registered budget: {root}")
    del last

    selected = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
    _require(selected["identity"] == identity, f"Selected checkpoint identity mismatch: {root}")
    _require(selected["epoch"] == selected_row["epoch"], f"Selected checkpoint epoch mismatch: {root}")
    _require(selected["metrics"] == selected_row["inner_select"],
             f"Selected checkpoint metrics differ from the first minimum history row: {root}")
    parameter_count = sum(int(value.numel()) for value in selected["model"].values())
    _require(parameter_count == result["trainable_parameters"], f"Teacher parameter count mismatch: {root}")
    budget = {
        "root": str(root),
        "view": result["view"],
        "seed": result["seed"],
        "fold": result["fold"],
        "best_epoch": result["best_epoch"],
        "epochs_executed": len(history),
        "optimizer_updates": update_count,
        "loop_seconds": float(result["loop_seconds"]),
        "trainable_parameters": parameter_count,
        "inner_select_best_mae": min(maes),
        "last_checkpoint_sha256": last_checkpoint_hash,
        "runtime": identity["runtime"],
    }
    del selected
    return root, result, identity, history, budget


def _audit_oof_file(path, expected_hash, expected_ids, targets):
    _require(path.is_file() and digest(path) == expected_hash, f"OOF prediction CSV hash mismatch: {path}")
    records = _read_predictions(path, ("id", "target", "prediction"))
    ids = [record["id"] for record in records]
    _require(len(ids) == len(set(ids)) and set(ids) == set(expected_ids),
             f"OOF prediction CSV has missing, extra, or duplicate IDs: {path}")
    for record in records:
        _require(record["id"] in targets and _target_matches(record["target"], targets[record["id"]]),
                 f"OOF prediction target differs from original float32 M: {record['id']}")
    return records


def _audit_full_train_file(path, expected_hash, ids, targets):
    """Hash/label audit only. These in-sample predictions never feed TRAIN tables."""
    _require(path.is_file() and digest(path) == expected_hash,
             f"Full-teacher TRAIN prediction CSV hash mismatch: {path}")
    records = _read_predictions(path, ("id", "target", "prediction"))
    actual_ids = [record["id"] for record in records]
    _require(len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == set(ids),
             f"Full-teacher TRAIN prediction IDs are not exact: {path}")
    for record in records:
        _require(_target_matches(record["target"], targets[record["id"]]),
                 f"Full-teacher TRAIN prediction target differs from original M: {record['id']}")
    return len(records)


def _audit_full_valid_file(path, expected_hash, ids):
    _require(path.is_file() and digest(path) == expected_hash,
             f"Full-teacher feature-only VALID prediction CSV hash mismatch: {path}")
    records = _read_predictions(path, ("id", "prediction"))
    actual_ids = [record["id"] for record in records]
    _require(len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == set(ids),
             f"Full-teacher feature-only VALID IDs are not exact: {path}")
    return {record["id"]: float(np.clip(record["prediction"], -1., 1.)) for record in records}


def _audit_baseline_anchors(baseline_suite_path, baseline_run_root, targets, valid_ids, csv_hash,
                            train_feature_hashes, valid_feature_hashes, project_root):
    suite = _read_json(baseline_suite_path)
    records = [r for r in suite.get("results", []) if r.get("model") == "early_fusion"]
    _require(len(records) == len(SEEDS), "Frozen baseline suite must contain three early_fusion results")
    _require(suite.get("suite_complete") is True
             and suite.get("test_evaluated") in (None, False),
             "Frozen baseline suite is incomplete or reports TEST evaluation")
    by_seed = {}
    audits = {}
    cp_hashes, prediction_hashes = {}, {}
    for result in records:
        seed = result["seed"]
        _require(seed in SEEDS and seed not in by_seed, "Duplicate or unregistered baseline seed")
        _require(result["dataset"] == "simsv2" and result["complete"] and not result["smoke_only"]
                 and not result["test_evaluated"], "Frozen baseline result is not formal/complete")
        _require(result["train_count"] == 2722 and result["valid_count"] == 647,
                 "Frozen baseline split counts differ from pinned CH-SIMS")
        folder = _resolve(baseline_run_root, project_root) / "early_fusion" / f"seed_{seed}"
        checkpoint_path = folder / "best.pt"
        prediction_path = folder / "valid_predictions.json"
        _require(digest(checkpoint_path) == result["checkpoint_sha256"],
                 f"Frozen early-fusion checkpoint hash mismatch for seed {seed}")
        _require(digest(prediction_path) == result["predictions_sha256"],
                 f"Frozen early-fusion VALID prediction hash mismatch for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        _require(checkpoint["epoch"] == result["best_epoch"],
                 f"Frozen baseline checkpoint epoch mismatch for seed {seed}")
        baseline_identity = checkpoint["identity"]
        _require(baseline_identity["config"]["model"] == "early_fusion"
                 and baseline_identity["seed"] == seed,
                 f"Frozen baseline model identity mismatch for seed {seed}")
        _require(baseline_identity["author_csv_sha256"] == csv_hash,
                 f"Frozen baseline CSV identity mismatch for seed {seed}")
        _require(_normalized_feature_hashes(baseline_identity["features"]["train"])
                 == _normalized_feature_hashes(train_feature_hashes)
                 and _normalized_feature_hashes(baseline_identity["features"]["valid"])
                 == _normalized_feature_hashes(valid_feature_hashes),
                 f"Frozen baseline feature identity mismatch for seed {seed}")
        _require(checkpoint["metrics"] == result["valid_metrics"],
                 f"Frozen baseline checkpoint metrics mismatch for seed {seed}")
        _require(sum(int(value.numel()) for value in checkpoint["model"].values()) == result["trainable_parameters"],
                 f"Frozen baseline parameter count mismatch for seed {seed}")
        del checkpoint
        predictions = _read_json(prediction_path)
        ids = [row["id"] for row in predictions]
        _require(len(ids) == len(set(ids)) and set(ids) == set(valid_ids),
                 f"Frozen baseline VALID IDs differ for seed {seed}")
        values = {}
        for row in predictions:
            sid = row["id"]
            _require(sid in targets and _target_matches(float(row["target"]), targets[sid]),
                     f"Frozen baseline VALID target differs from original float32 M: {sid}")
            prediction = float(row["prediction"])
            _require(math.isfinite(prediction), f"Nonfinite frozen anchor for {sid}")
            values[sid] = float(np.clip(prediction, -1., 1.))
        by_seed[seed] = values
        cp_hashes[str(seed)] = result["checkpoint_sha256"]
        prediction_hashes[str(seed)] = result["predictions_sha256"]
        audits[str(seed)] = {
            "best_epoch": result["best_epoch"],
            "trainable_parameters": result["trainable_parameters"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "predictions_sha256": result["predictions_sha256"],
            "prediction_count": len(predictions),
            "role": "frozen official early_fusion VALID point anchor",
        }
    _require(set(by_seed) == set(SEEDS), "Frozen baseline anchor is missing a seed")
    return by_seed, {
        "suite_sha256": digest(baseline_suite_path),
        "run_root": str(_resolve(baseline_run_root, project_root)),
        "checkpoint_sha256": cp_hashes,
        "predictions_sha256": prediction_hashes,
        "by_seed": audits,
    }


def prepare_tables(*, suite_path, config_path, output_dir,
                    baseline_suite_path=None, baseline_run_root="runs/native_simsv2_v1",
                    project_root=PROJECT_ROOT):
    """Audit all 48 teachers and write immutable per-seed table artifacts."""
    suite_path = _resolve(suite_path, project_root)
    config_path = _resolve(config_path, project_root)
    output_dir = _resolve(output_dir, project_root)
    if baseline_suite_path is None:
        baseline_suite_path = Path(project_root) / "docs/native_simsv2_baselines_v1.json"
    baseline_suite_path = _resolve(baseline_suite_path, project_root)
    _require(not output_dir.exists(), f"Output directory already exists; refusing overwrite: {output_dir}")
    _require(suite_path.is_file() and config_path.is_file() and baseline_suite_path.is_file(),
             "Suite, training config, and frozen baseline suite files are required")
    suite_manifest_path = suite_path.with_suffix(suite_path.suffix + ".manifest.json")
    suite_config_path = suite_path.with_suffix(suite_path.suffix + ".config.json")
    _require(suite_manifest_path.is_file() and suite_config_path.is_file(),
             "Completed suite requires adjacent manifest and config snapshots")

    config = _read_json(config_path)
    suite_config = _read_json(suite_config_path)
    suite = _read_json(suite_path)
    manifest = _read_json(suite_manifest_path)
    _require(config == suite_config, "Current training config differs from the saved suite config")
    _require(config.get("dataset") == "simsv2", "Only native CH-SIMS v2 is registered")
    _require(config.get("predict_valid") is True, "Full-teacher feature-only VALID predictions are required")
    _require(config.get("source_csv_sha256") == SOURCE_CSV_SHA256,
             "Training config does not contain the pinned author CSV hash")
    _require(suite.get("complete") is True and suite.get("suite_complete") is True,
             "The complete formal 48-teacher suite is required before table construction")
    _require(suite.get("smoke_only") is False and suite.get("test_evaluated") is False,
             "Smoke or TEST-evaluated suites cannot build calibration tables")
    _require(suite.get("selected") == {"seed": None, "view": None, "fold": None, "smoke_groups": None},
             "A subset-selected suite report cannot be promoted to a full suite")
    _require(manifest.get("dataset") == "simsv2" and manifest.get("has_full") is True
             and manifest.get("fixture_smoke") is False and manifest.get("smoke_groups") is None,
             "Suite manifest is not the full formal source-CV suite")
    _require(manifest.get("pinned_source_csv_sha256") == SOURCE_CSV_SHA256,
             "Suite manifest source CSV pin differs from the registered file")
    _require(manifest.get("seeds") == list(SEEDS) and manifest.get("selected_views") == list(VIEWS)
             and manifest.get("selected_folds") == list(FOLDS),
             "Suite manifest is missing registered seeds/views/folds")
    _require(manifest.get("official_validation_labels_parsed") is False
             and manifest.get("test_evaluated") is False,
             "Suite manifest reports forbidden split access")

    rows = author_rows(_resolve(config["author_csv"], project_root), "simsv2")
    csv_path = _resolve(config["author_csv"], project_root)
    csv_hash = digest(csv_path)
    _require(csv_hash == SOURCE_CSV_SHA256 and manifest.get("author_csv_sha256") == csv_hash,
             "Author CSV bytes differ from the registered source file")
    train_ids = [sid for sid, row in rows.items() if row["split"] == "train"]
    valid_ids = [sid for sid, row in rows.items() if row["split"] == "valid"]
    _require(len(train_ids) == 2722 and len({group_of(sid) for sid in train_ids}) == 129,
             "Pinned author TRAIN must contain 2,722 IDs from 129 sources")
    _require(len(valid_ids) == 647 and set(train_ids).isdisjoint(valid_ids),
             "Pinned author VALID must contain 647 IDs disjoint from TRAIN")
    _require(manifest.get("ids") == train_ids and manifest.get("train_ids") == train_ids,
             "Suite manifest TRAIN IDs do not match canonical author CSV order")

    fold_manifest = manifest["fold_splits"]["fold_manifest"]
    full_manifest = manifest["fold_splits"]["fold"]
    validate_source_folds(train_ids, fold_manifest["folds"])
    _validate_full_selection(train_ids, full_manifest)
    calibration_split = make_calibration_split(train_ids)
    _require(calibration_split["select_source_count"] == math.ceil(129 * .1) == 13,
             "Registered calibration selection must hold out 13 of 129 TRAIN sources")

    cache_dir = _resolve(config["cache_dir"], project_root)
    train_ds = NativeSplit(cache_dir, "train", rows, MODALITIES, pooled=True)
    valid_ds = NativeSplit(cache_dir, "valid", rows, MODALITIES, pooled=True)
    train_ds.lengths.clear()
    valid_ds.lengths.clear()
    _require(set(train_ds.ids) == set(train_ids) and set(valid_ds.ids) == set(valid_ids),
             "Native pooled feature cache IDs differ from author TRAIN/VALID IDs")
    _require(not (set(train_ds.ids) & set(valid_ds.ids)), "Native TRAIN/VALID feature IDs overlap")
    for split_name, dataset in (("train", train_ds), ("valid", valid_ds)):
        dims = {modality: int(dataset.features[modality].shape[-1]) for modality in MODALITIES}
        _require(dims == EXPECTED_DIMS, f"Unexpected pinned {split_name} T/A/V pooled feature dimensions")
        for sid, target in zip(dataset.ids, dataset.targets):
            _require(rows[sid]["split"] == split_name and _target_matches(float(target), rows[sid]["score"]),
                     f"{split_name} pooled target differs from original float32 M: {sid}")
    train_feature_hashes, valid_feature_hashes = train_ds.hashes, valid_ds.hashes
    recorded_train_feature_hashes = manifest.get("train_feature_hashes", {})
    recorded_valid_feature_hashes = manifest.get("valid_feature_hashes", {})
    _require(_normalized_feature_hashes(recorded_train_feature_hashes)
             == _normalized_feature_hashes(train_ds.hashes),
             "Current raw-derived TRAIN features do not match the suite manifest")
    _require(_normalized_feature_hashes(recorded_valid_feature_hashes)
             == _normalized_feature_hashes(valid_ds.hashes),
             "Current raw-derived VALID features do not match the suite manifest")
    train_feature_hashes = recorded_train_feature_hashes
    valid_feature_hashes = recorded_valid_feature_hashes
    train_target_by_id = {sid: float(rows[sid]["score"]) for sid in train_ids}
    valid_target_by_id = {sid: float(rows[sid]["score"]) for sid in valid_ids}

    run_root = _resolve(config["run_root"], project_root)
    _require(_resolve(manifest["root"], project_root) == run_root,
             "Suite run root differs from the registered config")
    results = suite.get("results", [])
    expected_keys = {(seed, view, fold) for seed in SEEDS for view in VIEWS for fold in FOLDS}
    actual_keys = [(row.get("seed"), row.get("view"), row.get("fold")) for row in results]
    _require(len(actual_keys) == len(set(actual_keys)) and set(actual_keys) == expected_keys,
             "Formal suite summary must contain each of the 48 seed/view/fold teachers exactly once")
    result_map = {(row["seed"], row["view"], row["fold"]): row for row in results}

    baseline_by_seed, baseline_audit = _audit_baseline_anchors(
        baseline_suite_path, baseline_run_root, valid_target_by_id, valid_ids, csv_hash,
        train_feature_hashes, valid_feature_hashes, project_root
    )
    fold_records = {(seed, view): {} for seed in SEEDS for view in VIEWS}
    full_valid_predictions = {(seed, view): {} for seed in SEEDS for view in VIEWS}
    ignored_full_train_files = {}
    ignored_fusion_valid_files = {}
    budgets, runtimes, prediction_hashes, checkpoint_hashes = {}, {}, {}, {}
    expected_cohorts = {
        (seed, view, fold): (fold_manifest["folds"][fold] if isinstance(fold, int) else full_manifest)
        for seed in SEEDS for view in VIEWS for fold in FOLDS
    }
    for key in sorted(expected_keys, key=lambda value: (value[0], VIEWS.index(value[1]), FOLDS.index(value[2]))):
        seed, view, fold = key
        result = result_map[key]
        cohort = expected_cohorts[key]
        expected_root = run_root / "formal" / view / f"seed_{seed}" / (f"fold_{fold}" if isinstance(fold, int) else "full")
        result_root = _resolve(result["root"], project_root)
        _require(result_root == expected_root.resolve(), f"Teacher result points outside expected root: {key}")
        root, checked_result, identity, history, budget = _validate_teacher(
            expected_root, result, cohort, fold_manifest, full_manifest, config, manifest,
            csv_hash, train_feature_hashes, valid_feature_hashes, project_root,
        )
        result_path = root / "result.json"
        _require(_read_json(result_path) == checked_result, f"Teacher result changed during audit: {key}")
        budget_key = f"seed_{seed}/{view}/{fold}"
        budgets[budget_key] = budget
        runtimes[budget_key] = identity["runtime"]
        checkpoint_hashes[budget_key] = {
            "best": checked_result["checkpoint_sha256"],
            "last": budget["last_checkpoint_sha256"],
        }

        if isinstance(fold, int):
            expected_prediction_path = root / "oof_predictions.csv"
            _require(_resolve(checked_result["predictions_file"], project_root) == expected_prediction_path.resolve(),
                     f"Unexpected OOF prediction path: {key}")
            records_for_fold = _audit_oof_file(expected_prediction_path, checked_result["predictions_sha256"],
                                               cohort["oof_ids"], train_target_by_id)
            _require(len(records_for_fold) == checked_result["oof_count"], f"OOF count mismatch: {key}")
            fold_records[(seed, view)][fold] = records_for_fold
            prediction_hashes[budget_key] = checked_result["predictions_sha256"]
        else:
            train_prediction_path = root / "full_train_predictions.csv"
            _require(_resolve(checked_result["full_train_predictions_file"], project_root)
                     == train_prediction_path.resolve(), f"Unexpected full-teacher TRAIN prediction path: {key}")
            train_prediction_count = _audit_full_train_file(
                train_prediction_path, checked_result["train_predictions_sha256"], train_ids, train_target_by_id
            )
            _require(train_prediction_count == checked_result["train_prediction_count"],
                     f"Full-teacher TRAIN prediction count mismatch: {key}")
            ignored_full_train_files[budget_key] = checked_result["train_predictions_sha256"]
            valid_prediction_path = root / "full_valid_predictions.csv"
            _require(_resolve(checked_result["valid_predictions_file"], project_root)
                     == valid_prediction_path.resolve(), f"Unexpected full-teacher VALID prediction path: {key}")
            view_values = _audit_full_valid_file(
                valid_prediction_path, checked_result["valid_predictions_sha256"], valid_ids
            )
            _require(len(view_values) == checked_result["valid_prediction_count"],
                     f"Full-teacher VALID prediction count mismatch: {key}")
            full_valid_predictions[(seed, view)] = view_values
            prediction_hashes[budget_key] = checked_result["valid_predictions_sha256"]
            if view == "fusion":
                # Audited for integrity, intentionally excluded from the VALID point anchor.
                ignored_fusion_valid_files[f"seed_{seed}"] = checked_result["valid_predictions_sha256"]

    canonical_train_features = _reorder_features(train_ds, train_ids)
    canonical_valid_features = _reorder_features(valid_ds, valid_ids)
    train_tables, valid_tables, merged_oof_hashes = {}, {}, {}
    oof_summary = suite.get("oof", {})
    for seed in SEEDS:
        merged = {}
        for view in VIEWS:
            merged[(seed, view)] = merge_oof_records(
                fold_records[(seed, view)], fold_manifest["folds"], train_ids, train_target_by_id
            )
            summary_key = f"{view}/seed_{seed}"
            summary = oof_summary.get(summary_key)
            _require(summary is not None and summary.get("coverage_exactly_once") is True
                     and summary.get("complete") is True and summary.get("count") == len(train_ids),
                     f"Suite merged OOF summary is incomplete for {summary_key}")
            merged_path = _resolve(summary["path"], project_root)
            expected_path = run_root / "formal" / view / f"seed_{seed}" / "oof_predictions.csv"
            _require(merged_path == expected_path.resolve(), f"Merged OOF path mismatch for {summary_key}")
            merged_records = _audit_oof_file(merged_path, summary["sha256"], train_ids, train_target_by_id)
            expected_records = merged[(seed, view)]
            _require(merged_records == expected_records, f"Merged OOF content differs from fold files for {summary_key}")
            merged_oof_hashes[summary_key] = summary["sha256"]

        fusion_anchor = {record["id"]: float(np.clip(record["prediction"], -1., 1.))
                         for record in merged[(seed, "fusion")]}
        train_view_maps = {
            view: {record["id"]: record["prediction"] for record in merged[(seed, view)]}
            for view in ("text", "audio", "vision")
        }
        train_views = np.asarray([
            [float(np.clip(train_view_maps[view][sid], -1., 1.))
             for view in ("text", "audio", "vision")]
            for sid in train_ids
        ], dtype=np.float32)
        train_anchor = np.asarray([fusion_anchor[sid] for sid in train_ids], dtype=np.float32)
        train_target = np.asarray([train_target_by_id[sid] for sid in train_ids], dtype=np.float32)
        train_tables[seed] = {
            "features": canonical_train_features.copy(),
            "anchor": train_anchor,
            "target": train_target,
            "views": train_views,
        }

        valid_anchor = np.asarray([baseline_by_seed[seed][sid] for sid in valid_ids], dtype=np.float32)
        valid_views = np.asarray([
            [full_valid_predictions[(seed, view)][sid] for view in ("text", "audio", "vision")]
            for sid in valid_ids
        ], dtype=np.float32)
        valid_target = np.asarray([valid_target_by_id[sid] for sid in valid_ids], dtype=np.float32)
        valid_tables[seed] = {
            "features": canonical_valid_features.copy(),
            "anchor": valid_anchor,
            "target": valid_target,
            "views": valid_views,
        }

    source_feature_hashes = {
        "train": train_feature_hashes,
        "valid": valid_feature_hashes,
    }
    code_hashes = {
        "prepare_anchor_calibration_tables.py": digest(Path(__file__)),
        "anchor_crossfit.py": digest(PROJECT_ROOT / "jevaffect/anchor_crossfit.py"),
        "native_training.py": digest(PROJECT_ROOT / "jevaffect/native_training.py"),
        "source_crossfit.py": digest(PROJECT_ROOT / "jevaffect/source_crossfit.py"),
        "native_metrics.py": digest(PROJECT_ROOT / "jevaffect/native_metrics.py"),
    }
    metadata = {
        "dataset": "simsv2",
        "format": "AnchorCDF calibration tables v1; float32 arrays, no object arrays",
        "seeds": list(SEEDS),
        "views": list(VIEWS),
        "train_count": len(train_ids),
        "valid_count": len(valid_ids),
        "source_count": len({group_of(sid) for sid in train_ids}),
        "author_csv": str(csv_path),
        "author_csv_sha256": csv_hash,
        "source_csv_sha256_pin": SOURCE_CSV_SHA256,
        "suite_json": str(suite_path),
        "suite_json_sha256": digest(suite_path),
        "suite_manifest": str(suite_manifest_path),
        "suite_manifest_sha256": digest(suite_manifest_path),
        "suite_config_sha256": digest(suite_config_path),
        "baseline_suite": str(baseline_suite_path),
        "baseline_anchor": baseline_audit,
        "raw_feature_hashes": source_feature_hashes,
        "code_sha256": code_hashes,
        "view_order": ["text", "audio", "vision"],
        "view_mean_semantics": "clipped native-M teacher outputs on TRAIN OOF and label-free full-teacher VALID predictions",
        "point_anchor_semantics": {
            "train": "clipped fusion-view source-CV OOF prediction",
            "valid": "clipped frozen official early_fusion VALID prediction",
            "full_fusion_teacher_valid_excluded": True,
            "full_train_teacher_predictions_excluded_from_calibration": True,
        },
        "target_semantics": "original author M score, float32 comparison semantics, never clipped",
        "feature_semantics": "NativeSplit pooled [text CLS, masked-mean audio, masked-mean vision], concatenated in that order",
        "feature_dimensions": EXPECTED_DIMS,
        "calibration_split": calibration_split,
        "source_group_calibration_selection_limit": (
            "The calibration fit/select split is deterministic and source-disjoint within TRAIN, but the reused source-CV OOF point table is not a fully nested pipeline: teachers in other folds may include calibration-selection sources. "
            "Do not call its inner head selection pipeline-independent or claim honest source transfer; a nested outer source evaluation must rebuild point/view OOF and calibration before promotion."
        ),
        "teacher_budgets": budgets,
        "teacher_runtimes": runtimes,
        "checkpoint_sha256": checkpoint_hashes,
        "prediction_sha256": prediction_hashes,
        "merged_oof_sha256": merged_oof_hashes,
        "verified_but_excluded_full_train_prediction_sha256": ignored_full_train_files,
        "verified_but_excluded_full_fusion_valid_sha256": ignored_fusion_valid_files,
        "official_validation_labels_parsed_for_teacher_stage": False,
        "test_evaluated": False,
        "artifact_sha256": {},
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent) as temp_name:
        staging = Path(temp_name)
        for seed in SEEDS:
            seed_dir = staging / f"seed_{seed}"
            seed_dir.mkdir(parents=True)
            train_npz = seed_dir / "train.npz"
            valid_npz = seed_dir / "valid.npz"
            np.savez_compressed(train_npz, **train_tables[seed])
            np.savez_compressed(valid_npz, **valid_tables[seed])
            train_id_path, valid_id_path = seed_dir / "train_ids.json", seed_dir / "valid_ids.json"
            save_json(train_id_path, train_ids)
            save_json(valid_id_path, valid_ids)
            prefix = f"seed_{seed}/"
            for path in (train_npz, valid_npz, train_id_path, valid_id_path):
                metadata["artifact_sha256"][prefix + path.name] = digest(path)
        save_json(staging / "metadata.json", metadata)
        staging.rename(output_dir)
    return metadata


def _reorder_features(dataset, ids):
    index = {sid: position for position, sid in enumerate(dataset.ids)}
    if set(ids) != set(index) or len(ids) != len(index):
        raise ValueError("Feature IDs differ from canonical IDs")
    values = [np.asarray(dataset.features[modality][[index[sid] for sid in ids]], dtype=np.float32)
              for modality in MODALITIES]
    result = np.concatenate(values, axis=1).astype(np.float32, copy=False)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Invalid pooled multimodal feature matrix")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True, help="Completed train_anchor_crossfit suite JSON")
    parser.add_argument("--config", type=Path, required=True, help="Pinned native_anchor_crossfit config JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    parser.add_argument("--baseline-suite", type=Path,
                        default=PROJECT_ROOT / "docs/native_simsv2_baselines_v1.json")
    parser.add_argument("--baseline-run-root", type=Path, default=PROJECT_ROOT / "runs/native_simsv2_v1")
    args = parser.parse_args()
    metadata = prepare_tables(
        suite_path=args.suite,
        config_path=args.config,
        output_dir=args.output_dir,
        baseline_suite_path=args.baseline_suite,
        baseline_run_root=args.baseline_run_root,
    )
    print(json.dumps({
        "output_dir": str(_resolve(args.output_dir)),
        "dataset": metadata["dataset"],
        "seeds": metadata["seeds"],
        "train_count": metadata["train_count"],
        "valid_count": metadata["valid_count"],
        "artifacts": metadata["artifact_sha256"],
        "test_evaluated": False,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
