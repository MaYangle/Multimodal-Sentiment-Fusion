"""Independent CPU audit for the source-nested native point-table stage.

The audit rebuilds the TRAIN join, source folds, nested role allocation,
fit-only feature statistics, and all exported point tables from pinned inputs
and frozen best checkpoints.  It never imports or calls producer validation
or prediction helpers as evidence.
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
ROLES = ("point_fit", "point_select", "shape_fit", "shape_select", "outer")
TABLE_ROLES = ("shape_fit", "shape_select", "outer")
MODALITIES = ("text", "audio", "vision")
EXPECTED_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
CPU_GPU_TOLERANCE = 1e-5


class AuditError(RuntimeError):
    """A concrete audit failure with a reviewer-actionable message."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_of(sample_id: str) -> str:
    if sample_id.count("/") != 1 or not all(sample_id.split("/")):
        raise ValueError(f"Malformed CH-SIMS source/clip ID: {sample_id!r}")
    return sample_id.split("/", 1)[0]


def read_author_train(csv_path: Path) -> tuple[list[str], dict[str, np.float32]]:
    """Read TRAIN labels only and retain their source CSV row order."""
    ids: list[str] = []
    targets: dict[str, np.float32] = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != "train":
                continue
            sample_id = row["video_id"] + "/" + row["clip_id"]
            score = float(row["label"])
            if sample_id in targets or not np.isfinite(score) or abs(score) > 1:
                raise AuditError("Author TRAIN CSV has a duplicate ID or invalid native score")
            ids.append(sample_id)
            targets[sample_id] = np.float32(score)
    if not ids or len(ids) != len(set(ids)):
        raise AuditError("Author TRAIN CSV did not yield a nonempty unique canonical cohort")
    return ids, targets


def pool_sequences(values: np.ndarray, lengths: np.ndarray, modality: str) -> np.ndarray:
    """Reproduce the cache's declared pooling from raw sequence arrays."""
    x = np.asarray(values)
    n = np.asarray(lengths)
    if x.ndim != 3 or n.shape != (x.shape[0],) or n.dtype.kind not in "iu":
        raise ValueError(f"Invalid {modality} sequence or length shape")
    if np.any(n < 0) or np.any(n > x.shape[1]) or not np.isfinite(x).all():
        raise ValueError(f"Invalid {modality} lengths or nonfinite cached features")
    if modality == "text":
        result = np.asarray(x[:, 0], dtype=np.float32).copy()
        result[n == 0] = 0
    elif modality in ("audio", "vision"):
        result = np.stack([
            np.asarray(x[i, :int(length)].mean(axis=0), dtype=np.float32)
            if length else np.zeros(x.shape[-1], dtype=np.float32)
            for i, length in enumerate(n)
        ])
    else:
        raise ValueError(f"Unexpected modality {modality!r}")
    if result.dtype != np.float32 or result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"Pooled {modality} features are not finite float32 rows")
    return np.ascontiguousarray(result)


def pooled_feature_hash(features: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for modality in sorted(features):
        values = np.asarray(features[modality])
        if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Pooled feature content must be finite float32 matrices")
        h.update(modality.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
        h.update(values.tobytes(order="C"))
    return h.hexdigest()


def rebuild_source_roles(ids: list[str], *, expected_source_count: int | None = 129) -> tuple[list[str], dict[str, Any]]:
    """Independently replay the registered source fold and nested RNG splits."""
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical author TRAIN IDs must be unique")
    groups: dict[str, list[str]] = {}
    for sample_id in ids:
        groups.setdefault(source_of(sample_id), []).append(sample_id)
    if expected_source_count is not None and len(groups) != expected_source_count:
        raise ValueError(f"Expected {expected_source_count} TRAIN sources, reconstructed {len(groups)}")

    rng = np.random.default_rng(20261012)
    buckets: list[list[str]] = [[], [], []]
    sizes = np.zeros(3, dtype=np.int64)
    for raw_group in rng.permutation(sorted(groups)):
        group = str(raw_group)
        bucket = int(np.argmin(sizes))
        buckets[bucket].append(group)
        sizes[bucket] += len(groups[group])

    outer_ids: list[list[str]] = []
    outer_sources: list[set[str]] = []
    for bucket in buckets:
        held = set(bucket)
        outer_sources.append(held)
        outer_ids.append([sid for sid in ids if source_of(sid) in held])

    folds: list[dict[str, Any]] = []
    all_sources = set(groups)
    for fold_index, held_sources in enumerate(outer_sources):
        remaining = sorted(all_sources - held_sources)
        rng = np.random.default_rng(20261013 + 1000 * fold_index)
        ordered = [str(group) for group in rng.permutation(remaining)]
        calibration_count = int(math.ceil(.15 * len(remaining)))
        calibration_sources = ordered[:calibration_count]
        post_calibration = len(remaining) - calibration_count
        point_select_count = int(math.ceil(.1 * post_calibration))
        point_select_sources = ordered[calibration_count:calibration_count + point_select_count]
        point_fit_sources = ordered[calibration_count + point_select_count:]
        shape_order = [str(group) for group in rng.permutation(calibration_sources)]
        shape_select_count = int(math.ceil(.3 * calibration_count))
        shape_select_sources = shape_order[:shape_select_count]
        shape_fit_sources = shape_order[shape_select_count:]

        source_roles = {
            "point_fit": point_fit_sources,
            "point_select": point_select_sources,
            "shape_fit": shape_fit_sources,
            "shape_select": shape_select_sources,
            "outer": sorted(held_sources),
        }
        role_ids = {
            role: [sid for sid in ids if source_of(sid) in set(sources)]
            for role, sources in source_roles.items()
        }
        point_holdout = set(role_ids["shape_fit"]) | set(role_ids["shape_select"]) | set(role_ids["outer"])
        folds.append({
            "fold": fold_index,
            "role_ids": role_ids,
            "role_sources": {role: sorted(set(source_roles[role])) for role in ROLES},
            "outer_ids": role_ids["outer"],
            "point_oof_ids": [sid for sid in ids if sid in point_holdout],
            "point_fit_source_count": len(set(point_fit_sources)),
            "point_select_source_count": len(set(point_select_sources)),
            "calibration_source_count": calibration_count,
            "shape_select_source_count": len(set(shape_select_sources)),
        })
    return outer_ids, {"groups": groups, "folds": folds}


def fit_only_statistics(features: dict[str, np.ndarray], indices: list[int]) -> dict[str, dict[str, Any]]:
    """Recompute coordinate means/population standard deviations in float64."""
    idx = np.asarray(indices, dtype=np.int64)
    if not len(idx) or len(np.unique(idx)) != len(idx):
        raise ValueError("Scaler fit indices must be nonempty and unique")
    result: dict[str, dict[str, Any]] = {}
    for modality, values in features.items():
        x = np.asarray(values, dtype=np.float32)
        if idx.min() < 0 or idx.max() >= len(x):
            raise ValueError("Scaler fit index is outside the feature cache")
        selected = x[idx].astype(np.float64)
        mean = selected.mean(axis=0)
        scale = selected.std(axis=0)
        scale[scale < 1e-6] = 1.
        result[modality] = {
            "mean": mean.astype(np.float32).tolist(),
            "scale": scale.astype(np.float32).tolist(),
            "fit_indices": idx.tolist(),
        }
    return result


def replay_plateau_lr(
    scores: list[float], initial_lr: float, *, patience: int = 3,
    factor: float = .5, threshold: float = 1e-4, eps: float = 1e-8,
) -> list[float]:
    """Replay PyTorch ReduceLROnPlateau(mode=min, threshold_mode=rel) defaults."""
    best = math.inf
    bad_epochs = 0
    lr = float(initial_lr)
    rates: list[float] = []
    for score in scores:
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("Selection MAE history contains a nonfinite value")
        improved = best == math.inf or value < best * (1. - threshold)
        if improved:
            best = value
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs > patience:
            candidate = max(lr * factor, 0.)
            if lr - candidate > eps:
                lr = candidate
            bad_epochs = 0
        rates.append(lr)
    return rates


def first_minimum_index(values: list[float]) -> int:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("First-minimum selection requires finite nonempty values")
    best = min(float(value) for value in values)
    return next(index for index, value in enumerate(values) if float(value) == best)


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        raise ValueError(f"Array shapes differ: {a.shape} versus {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Cannot compare nonfinite arrays")
    if not a.size:
        return 0.
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


class Recorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.findings: list[dict[str, str]] = []

    def must(self, name: str, condition: bool, detail: str) -> None:
        self.checks.append({"name": name, "passed": bool(condition)})
        if not condition:
            self.findings.append({"check": name, "issue": detail})
            raise AuditError(detail)


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path)


def _read_cache(config: dict[str, Any], repo_root: Path, author_ids: list[str],
                targets: dict[str, np.float32], manifest: dict[str, Any], rec: Recorder):
    cache_ref = Path(config["cache_dir"])
    cache_io = resolve_path(cache_ref, repo_root)
    metadata_file = cache_io / "metadata.json"
    if not metadata_file.exists():
        metadata_file = cache_io / "meta.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    split_rel = metadata.get("split_paths", {}).get("train")
    if split_rel is None:
        split_rel = "splits/train" if (cache_io / "splits").exists() else "train"
    train_dir = cache_io / split_rel
    cache_ids_file = train_dir / "ids.json"
    cache_ids = json.loads(cache_ids_file.read_text(encoding="utf-8"))
    rec.must("cache TRAIN IDs join author TRAIN", len(cache_ids) == len(set(cache_ids))
             and set(cache_ids) == set(author_ids),
             "Raw cache TRAIN IDs do not exactly match author CSV TRAIN IDs; inspect cache split metadata and row mapping.")

    expected_file_hashes: dict[str, str] = {
        str(cache_ref / metadata_file.name): sha256_file(metadata_file),
        # NativeSplit resolves the split directory before storing ids.json in
        # its source hash map, while metadata keeps its configured path spelling.
        str(cache_ids_file.resolve()): sha256_file(cache_ids_file),
    }
    recorded = metadata.get("artifact_sha256", {})
    file_records = metadata.get("files", {})
    pooled: dict[str, np.ndarray] = {}
    dimensions: dict[str, int] = {}
    cache_targets = np.asarray([targets[sid] for sid in cache_ids], dtype=np.float32)
    raw_label_file = train_dir / "raw_labels.json"
    if raw_label_file.exists():
        raw_targets = np.asarray(json.loads(raw_label_file.read_text(encoding="utf-8")), dtype=np.float32).reshape(-1)
        rec.must("cache raw labels match author TRAIN", raw_targets.shape == cache_targets.shape
                 and max_abs_difference(raw_targets, cache_targets) <= 1e-6,
                 "Cache raw_labels.json differs from the author TRAIN CSV; investigate label provenance.")

    for modality in MODALITIES:
        feature_file = train_dir / f"{modality}.npy"
        lengths_file = train_dir / f"{modality}_lengths.npy"
        for actual_file in (feature_file, lengths_file):
            key = (Path(split_rel) / actual_file.name).as_posix()
            actual_sha = sha256_file(actual_file)
            known_sha = recorded.get(key, file_records.get(key, {}).get("sha256"))
            rec.must(f"{modality} cache artifact pin", known_sha == actual_sha,
                     f"Cached {modality} file {key} is missing from metadata or its SHA256 differs.")
            expected_file_hashes[key] = actual_sha
        raw = np.load(feature_file, mmap_mode="r")
        lengths = np.load(lengths_file, mmap_mode="r")
        pooled[modality] = pool_sequences(raw, lengths, modality)
        dimensions[modality] = int(pooled[modality].shape[-1])

    # The trainer's NativeSplit hash map preserves the configured path spelling
    # for metadata/ids, while feature artifacts are cache-relative POSIX keys.
    suite_hashes = manifest.get("raw_feature_file_sha256", {})
    rec.must("raw cache hash map matches suite manifest", suite_hashes == expected_file_hashes,
             "Suite raw_feature_file_sha256 does not match independently hashed cache metadata, IDs, arrays, and lengths.")
    content_hash = pooled_feature_hash(pooled)
    rec.must("pooled raw feature content hash", content_hash == manifest.get("pooled_feature_content_sha256"),
             "Independently pooled raw features differ from pooled_feature_content_sha256 in the suite manifest.")
    rec.must("fusion input width", sum(dimensions.values()) == 1920,
             f"Pinned raw cache pooled dimensions sum to {sum(dimensions.values())}, expected 1,920.")
    cache_position = {sample_id: index for index, sample_id in enumerate(cache_ids)}
    return cache_ids, cache_position, cache_targets, pooled, dimensions, expected_file_hashes, content_hash


def _assert_expected_manifest(nested: dict[str, Any], ids: list[str], outer_ids: list[list[str]],
                              rebuilt: dict[str, Any], rec: Recorder) -> None:
    rec.must("nested manifest identity", nested.get("name") == "nested-source-honest-shape-tables-v1"
             and nested.get("role_seed") == 20261013 and nested.get("outer_fold_seed") == 20261012
             and nested.get("outer_fold_count") == 3 and nested.get("smoke") is False,
             "Nested suite manifest has an unexpected name, seed, fold count, or smoke marker.")
    rec.must("outer folds independently reconstructed", nested.get("outer_parent", {}).get("outer_oof_ids") == outer_ids,
             "Outer source folds differ from a fresh seed-20261012 greedy source-fold reconstruction.")
    actual_folds = nested.get("folds", [])
    rec.must("nested fold count", len(actual_folds) == 3,
             f"Nested manifest has {len(actual_folds)} folds; expected three.")
    outer_seen: list[str] = []
    outer_sources: list[str] = []
    for index, (actual, expected) in enumerate(zip(actual_folds, rebuilt["folds"])):
        for role in ROLES:
            rec.must(f"fold {index} {role} IDs", actual.get(role + "_ids") == expected["role_ids"][role],
                     f"Fold {index} {role} IDs differ from independent role-seed allocation or canonical author TRAIN order.")
            rec.must(f"fold {index} {role} sources", actual.get("role_sources", {}).get(role) == expected["role_sources"][role],
                     f"Fold {index} {role} source groups differ from independently reconstructed source roles.")
        rec.must(f"fold {index} point holdout", actual.get("point_oof_ids") == expected["point_oof_ids"]
                 and actual.get("oof_ids") == expected["point_oof_ids"],
                 f"Fold {index} point holdout is not shape_fit + shape_select + that fold's outer IDs in canonical order.")
        rec.must(f"fold {index} aliases and scaler scope", actual.get("fit_ids") == expected["role_ids"]["point_fit"]
                 and actual.get("select_ids") == expected["role_ids"]["point_select"]
                 and actual.get("point_scaler_fit_ids") == expected["role_ids"]["point_fit"],
                 f"Fold {index} point fit/select aliases or scaler-fit IDs are inconsistent with nested roles.")
        role_sets = {role: set(actual.get(role + "_ids", [])) for role in ROLES}
        source_sets = {role: {source_of(sid) for sid in role_sets[role]} for role in ROLES}
        rec.must(f"fold {index} full role partition", set.union(*role_sets.values()) == set(ids)
                 and all(not role_sets[a] & role_sets[b] for i, a in enumerate(ROLES) for b in ROLES[i + 1:]),
                 f"Fold {index} roles do not partition all TRAIN IDs exactly once.")
        rec.must(f"fold {index} source isolation", all(not source_sets[a] & source_sets[b]
                 for i, a in enumerate(ROLES) for b in ROLES[i + 1:]),
                 f"Fold {index} has a source shared between nested roles; calibration, selection, and outer roles must be disjoint.")
        cal_sources = source_sets["shape_fit"] | source_sets["shape_select"]
        point_sources = source_sets["point_fit"] | source_sets["point_select"]
        rec.must(f"fold {index} calibration exclusion", not cal_sources & point_sources
                 and not cal_sources & source_sets["outer"],
                 f"Fold {index} calibration source entered point fit/selection or outer evaluation.")
        outer_seen.extend(actual.get("outer_ids", []))
        outer_sources.extend(sorted({source_of(sid) for sid in actual.get("outer_ids", [])}))
        rec.must(f"fold {index} holdout semantics", "not global once-only OOF" in actual.get("point_holdout_semantics", ""),
                 f"Fold {index} point holdout is not explicitly described as repeating across folds.")
    rec.must("outer ID once-only coverage", len(outer_seen) == len(ids) and len(set(outer_seen)) == len(ids)
             and set(outer_seen) == set(ids),
             "Across outer folds, TRAIN IDs are missing or held more than once.")
    rec.must("outer source once-only coverage", len(outer_sources) == len(set(outer_sources))
             and len(set(outer_sources)) == len(rebuilt["groups"]),
             "Across outer folds, TRAIN source groups are missing or held more than once.")


def _compare_saved_stats(saved: dict[str, Any], expected: dict[str, Any]) -> bool:
    if set(saved) != set(expected):
        return False
    for modality in expected:
        for key in ("mean", "scale", "fit_indices"):
            if saved[modality].get(key) != expected[modality][key]:
                return False
    return True


def _replay_and_validate_history(model_root: Path, point_result: dict[str, Any],
                                 identity: dict[str, Any], config: dict[str, Any],
                                 fold: dict[str, Any], rec: Recorder) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history_path = model_root / "history.json"
    best_path = model_root / "best.pt"
    last_path = model_root / "last.pt"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    rec.must("nonempty bounded point history", isinstance(history, list) and 0 < len(history) <= config["max_epochs"],
             f"Point history has {len(history) if isinstance(history, list) else 'invalid'} epochs; expected 1..{config['max_epochs']}.")
    rec.must("sequential point history epochs", [row.get("epoch") for row in history] == list(range(1, len(history) + 1)),
             "Point history epochs are missing, repeated, or out of order.")
    scores = [float(row.get("inner_select", {}).get("MAE", math.nan)) for row in history]
    selected_index = first_minimum_index(scores)
    selected_epoch = selected_index + 1
    rec.must("first minimum point_select MAE", point_result.get("best_epoch") == selected_epoch,
             f"Recorded best epoch {point_result.get('best_epoch')} is not first minimum point_select MAE epoch {selected_epoch}.")
    rec.must("reported first-minimum selection rule", identity.get("selection") == "minimum inner-selection native MAE; first checkpoint wins ties",
             "Point identity does not bind first-minimum inner-selection native MAE checkpoint selection.")
    updates_per_epoch = int(math.ceil(len(fold["point_fit_ids"]) / int(config["batch_size"])))
    rec.must("history optimizer update counts", all(row.get("optimizer_updates") == updates_per_epoch for row in history),
             f"Point history optimizer updates differ from {updates_per_epoch} batches per point_fit epoch.")

    replayed = replay_plateau_lr(
        scores, float(config["learning_rate"]),
        patience=int(config["scheduler_patience"]), factor=float(config["scheduler_factor"]),
    )
    recorded = [float(row.get("learning_rate", math.nan)) for row in history]
    lr_errors = [abs(a - b) for a, b in zip(replayed, recorded)]
    rec.must("ReduceLROnPlateau learning-rate replay", len(recorded) == len(replayed)
             and all(math.isfinite(x) and x <= 1e-12 for x in lr_errors),
             "Recorded point learning rates do not replay from the configured initial LR and plateau scheduler; inspect history/checkpoint resume state.")
    rec.must("history and result epoch counts", point_result.get("epochs_executed") == len(history),
             "Point result epoch count differs from the complete history file.")
    best_stale = len(history) - selected_epoch
    early_stopped = len(history) < int(config["max_epochs"])
    rec.must("stopping budget and patience", (len(history) == int(config["max_epochs"]) or best_stale >= int(config["patience"]))
             and (not early_stopped or best_stale == int(config["patience"])),
             f"History stopped at {len(history)} epochs with {best_stale} epochs since first best; expected max_epochs={config['max_epochs']} or exactly patience={config['patience']}.")
    rec.must("formal point stage completion", point_result.get("complete") is True,
             "Point trainer result is not marked complete under the registered epoch/patience budget.")

    import torch

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    rec.must("best checkpoint selected epoch", best.get("epoch") == selected_epoch,
             "best.pt epoch does not equal the first point_select MAE minimum.")
    rec.must("best checkpoint identity", best.get("identity") == identity,
             "best.pt identity differs from identity.json.")
    rec.must("best checkpoint selection metric", abs(float(best.get("metrics", {}).get("MAE", math.nan)) - scores[selected_index]) <= 1e-12,
             "best.pt selection MAE differs from the selected history row.")
    rec.must("last checkpoint history state", last.get("epoch") == len(history) and last.get("history") == history,
             "last.pt epoch/history does not match history.json.")
    rec.must("last checkpoint stale count", last.get("stale") == best_stale,
             f"last.pt stale counter {last.get('stale')} differs from {best_stale} nonimproving epochs after first minimum.")
    rec.must("last checkpoint best MAE", abs(float(last.get("best", math.nan)) - scores[selected_index]) <= 1e-12,
             "last.pt best MAE differs from first minimum point_select MAE.")
    return best, history


def _audit_one_model(row: dict[str, Any], expected: dict[str, Any], *, seed: int, fold_index: int,
                     config: dict[str, Any], config_sha: str, suite_manifest: dict[str, Any],
                     repo_root: Path, canonical_ids: list[str], author_targets: dict[str, np.float32],
                     cache_ids: list[str], cache_position: dict[str, int], cache_targets: np.ndarray,
                     features: dict[str, np.ndarray], dimensions: dict[str, int],
                     file_hashes: dict[str, str], feature_content_sha: str,
                     rec: Recorder, max_errors: dict[str, float]) -> None:
    point_root_ref = row.get("point_model_root")
    artifact_root = resolve_path(suite_manifest["artifact_root"], repo_root)
    expected_model_root = artifact_root / "point_models" / "formal" / "fusion" / f"seed_{seed}" / f"fold_{fold_index}"
    model_root = resolve_path(point_root_ref, repo_root) if point_root_ref else expected_model_root
    rec.must(f"seed {seed} fold {fold_index} point model path", model_root.resolve() == expected_model_root.resolve()
             and model_root.is_dir(),
             f"Seed {seed} fold {fold_index} point model artifacts are missing or outside the registered formal artifact tree.")
    rec.must(f"seed {seed} fold {fold_index} stage record", row.get("seed") == seed and row.get("fold") == fold_index
             and row.get("complete") is True,
             f"Suite record for seed {seed} fold {fold_index} has incorrect identity or is incomplete.")
    for name, expected_sha in {
        "best.pt": row.get("point_best_checkpoint_sha256"),
        "last.pt": row.get("point_last_checkpoint_sha256"),
        "history.json": row.get("point_history_sha256"),
        "identity.json": row.get("point_identity_sha256"),
        "oof_predictions.csv": row.get("point_oof_predictions_sha256"),
    }.items():
        path = model_root / name
        rec.must(f"seed {seed} fold {fold_index} {name} hash", path.is_file()
                 and expected_sha == sha256_file(path),
                 f"Seed {seed} fold {fold_index} {name} is missing or its hash differs from point_stage.json.")

    actual_fold = suite_manifest["nested_manifest"]["folds"][fold_index]
    rec.must(f"seed {seed} fold {fold_index} fold role checkpoint", row.get("nested_role_manifest_sha256")
             == canonical_hash(suite_manifest["nested_manifest"]),
             f"Seed {seed} fold {fold_index} point stage is bound to a different nested role manifest.")
    scaler_path = resolve_path(row.get("point_scaler_path", ""), repo_root)
    scaler_sha = row.get("point_scaler_sha256")
    rec.must(f"seed {seed} fold {fold_index} scaler file hash", scaler_path.is_file()
             and scaler_sha == sha256_file(scaler_path),
             f"Seed {seed} fold {fold_index} point_fit scaler is missing or its file hash differs from point_stage.json.")
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    expected_fit_ids = expected["role_ids"]["point_fit"]
    expected_indices = [cache_position[sid] for sid in expected_fit_ids]
    expected_stats = fit_only_statistics(features, expected_indices)
    preprocessing = scaler.get("preprocessing", {})
    rec.must(f"seed {seed} fold {fold_index} scaler fit IDs", scaler.get("fit_ids") == expected_fit_ids
             and preprocessing.get("fit_ids") == expected_fit_ids
             and preprocessing.get("fit_indices_in_cache_order") == expected_indices,
             f"Seed {seed} fold {fold_index} scaler includes IDs outside point_fit or uses non-cache-order indices.")
    rec.must(f"seed {seed} fold {fold_index} fit-only scaler statistics", _compare_saved_stats(scaler.get("statistics", {}), expected_stats)
             and _compare_saved_stats(preprocessing.get("statistics", {}), expected_stats),
             f"Seed {seed} fold {fold_index} scaler means/scales differ from an independent float64 calculation over point_fit only.")
    rec.must(f"seed {seed} fold {fold_index} scaler provenance", scaler.get("fit_role") == "point_fit only"
             and scaler.get("fit_source_groups") == expected["role_sources"]["point_fit"]
             and preprocessing.get("raw_feature_sha256") == file_hashes
             and scaler.get("sha256") == canonical_hash(preprocessing),
             f"Seed {seed} fold {fold_index} scaler provenance/hash does not bind point_fit IDs, source groups, and raw cache hashes.")
    excluded = set(actual_fold["point_select_ids"]) | set(actual_fold["shape_fit_ids"]) | set(actual_fold["shape_select_ids"]) | set(actual_fold["outer_ids"])
    rec.must(f"seed {seed} fold {fold_index} scaler exclusion", not set(scaler.get("fit_ids", [])) & excluded,
             f"Seed {seed} fold {fold_index} point scaler contains point_select, calibration, or outer IDs.")

    identity_path = model_root / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    rec.must(f"seed {seed} fold {fold_index} identity roles and IDs", identity.get("seed") == seed
             and identity.get("rng_seed") == seed + 10000 * fold_index
             and identity.get("fold") == fold_index
             and identity.get("cohort") == actual_fold
             and identity.get("train_ids") == canonical_ids,
             f"Seed {seed} fold {fold_index} frozen identity has different RNG, cohort roles, or canonical TRAIN IDs.")
    rec.must(f"seed {seed} fold {fold_index} source/config identity hashes", identity.get("author_csv_sha256") == EXPECTED_CSV_SHA256
             and identity.get("feature_hashes") == file_hashes
             and identity.get("manifest_sha256") == canonical_hash(actual_fold)
             and identity.get("fold_manifest_sha256") == canonical_hash(suite_manifest["nested_manifest"]),
             f"Seed {seed} fold {fold_index} identity does not bind pinned CSV, raw cache, and nested source role hashes.")
    identity_config = identity.get("config", {})
    nested_config = identity_config.get("nested_pipeline", {})
    rec.must(f"seed {seed} fold {fold_index} identity config hash", nested_config.get("config_file_sha256") == config_sha
             and nested_config.get("pooled_feature_content_sha256") == feature_content_sha
             and nested_config.get("nested_role_manifest_sha256") == canonical_hash(suite_manifest["nested_manifest"])
             and identity_config.get("preprocessing") == preprocessing,
             f"Seed {seed} fold {fold_index} identity config is not bound to the registered JSON, pooled data, role manifest, and scaler.")
    current_files = {
        "nested_shape_points.py": repo_root / "jevaffect" / "nested_shape_points.py",
        "anchor_crossfit.py": repo_root / "jevaffect" / "anchor_crossfit.py",
        "standardized_source.py": repo_root / "jevaffect" / "standardized_source.py",
        "native_training.py": repo_root / "jevaffect" / "native_training.py",
        "source_crossfit.py": repo_root / "jevaffect" / "source_crossfit.py",
        "native_metrics.py": repo_root / "jevaffect" / "native_metrics.py",
    }
    expected_impl = {name: sha256_file(path) for name, path in current_files.items()}
    recorded_impl = row.get("implementation_sha256", {})
    identity_impl = identity.get("implementation_sha256", {})
    rec.must(f"seed {seed} fold {fold_index} frozen implementation hashes",
             all(recorded_impl.get(name) == digest for name, digest in expected_impl.items())
             and all(identity_impl.get(name) == expected_impl[name]
                     for name in ("native_training.py", "source_crossfit.py", "native_metrics.py")),
             f"Seed {seed} fold {fold_index} artifacts are not bound to the current frozen nested/anchor/standardized/native code hashes.")
    rec.must(f"seed {seed} fold {fold_index} no VALID/TEST use", row.get("official_VALID_used") is False
             and row.get("official_TEST_used") is False
             and identity.get("official_validation_labels_parsed") is False
             and identity.get("test_evaluated") is False,
             f"Seed {seed} fold {fold_index} identity does not certify TRAIN-only preparation.")

    point_result = row.get("point_result", {})
    rec.must(f"seed {seed} fold {fold_index} nested trainer checkpoint hash",
             point_result.get("checkpoint_sha256") == row.get("point_best_checkpoint_sha256"),
             f"Seed {seed} fold {fold_index} trainer result and point-stage record disagree on the best checkpoint SHA256.")
    rec.must(f"seed {seed} fold {fold_index} point cohort counts", point_result.get("fit_count") == len(expected_fit_ids)
             and point_result.get("select_count") == len(expected["role_ids"]["point_select"])
             and point_result.get("oof_count") == len(expected["point_oof_ids"]),
             f"Seed {seed} fold {fold_index} trainer cohort counts do not match point_fit, point_select, and nested holdout IDs.")
    expected_params = 1920 * int(config["hidden"]) + int(config["hidden"]) + int(config["hidden"]) + 1
    rec.must(f"seed {seed} fold {fold_index} trainable parameter count", point_result.get("trainable_parameters") == expected_params,
             f"Seed {seed} fold {fold_index} point model has {point_result.get('trainable_parameters')} parameters; expected {expected_params}.")
    best, history = _replay_and_validate_history(model_root, point_result, identity, config, actual_fold, rec)

    import torch
    import torch.nn.functional as F

    state = best.get("model", {})
    expected_shapes = {
        "head.0.weight": (int(config["hidden"]), 1920),
        "head.0.bias": (int(config["hidden"]),),
        "head.3.weight": (1, int(config["hidden"])),
        "head.3.bias": (1,),
    }
    rec.must(f"seed {seed} fold {fold_index} checkpoint tensor schema", set(state) == set(expected_shapes)
             and all(tuple(state[name].shape) == shape for name, shape in expected_shapes.items())
             and all(torch.isfinite(state[name]).all().item() for name in expected_shapes),
             f"Seed {seed} fold {fold_index} best.pt does not contain the registered 1,920→256→1 finite point model tensors.")

    table_rows = row.get("tables", {})
    rec.must(f"seed {seed} fold {fold_index} table role set", set(table_rows) == set(TABLE_ROLES),
             f"Seed {seed} fold {fold_index} must export exactly shape_fit, shape_select, and outer tables.")
    table_root = artifact_root / "tables" / f"seed_{seed}" / f"fold_{fold_index}"
    canonical_file = table_root / "canonical_train_ids.json"
    canonical_sha = row.get("canonical_train_ids_sha256")
    rec.must(f"seed {seed} fold {fold_index} canonical ID file", canonical_file.is_file()
             and row.get("canonical_train_ids_path")
             and resolve_path(row["canonical_train_ids_path"], repo_root).resolve() == canonical_file.resolve()
             and sha256_file(canonical_file) == canonical_sha
             and json.loads(canonical_file.read_text(encoding="utf-8")) == canonical_ids,
             f"Seed {seed} fold {fold_index} canonical TRAIN ID file/hash/order is invalid.")

    for role in TABLE_ROLES:
        meta = table_rows[role]
        expected_ids = expected["role_ids"][role]
        expected_table = table_root / f"{role}.npz"
        expected_ids_file = table_root / f"{role}_ids.json"
        table_path = resolve_path(meta.get("path", ""), repo_root)
        ids_path = resolve_path(meta.get("ids_path", ""), repo_root)
        rec.must(f"seed {seed} fold {fold_index} {role} table hashes", table_path.is_file()
                 and table_path.resolve() == expected_table.resolve()
                 and meta.get("sha256") == sha256_file(table_path)
                 and ids_path.is_file() and ids_path.resolve() == expected_ids_file.resolve()
                 and meta.get("ids_sha256") == sha256_file(ids_path),
                 f"Seed {seed} fold {fold_index} {role} table or ID sidecar is missing, outside the expected table path, or has a mismatched hash.")
        rec.must(f"seed {seed} fold {fold_index} {role} declared row metadata", meta.get("count") == len(expected_ids)
                 and meta.get("source_groups") == expected["role_sources"][role]
                 and meta.get("ids_in_canonical_author_train_order") is True
                 and meta.get("source_csv_sha256") == EXPECTED_CSV_SHA256
                 and meta.get("config_sha256") == config_sha
                 and meta.get("raw_feature_file_sha256") == file_hashes
                 and meta.get("implementation_sha256") == expected_impl,
                 f"Seed {seed} fold {fold_index} {role} metadata count, groups, TRAIN ordering, source/config/cache, or implementation binding is invalid.")
        rec.must(f"seed {seed} fold {fold_index} {role} ID sidecar", json.loads(ids_path.read_text(encoding="utf-8")) == expected_ids,
                 f"Seed {seed} fold {fold_index} {role} IDs differ from independently rebuilt canonical role order.")

        with np.load(table_path, allow_pickle=False) as archive:
            rec.must(f"seed {seed} fold {fold_index} {role} array names", set(archive.files)
                     == {"context", "anchor", "native_point", "target", "ids"},
                     f"Seed {seed} fold {fold_index} {role} NPZ has unexpected or missing arrays.")
            table = {key: archive[key].copy() for key in archive.files}
        expected_table_shapes = {
            "context": (len(expected_ids), int(config["hidden"])),
            "anchor": (len(expected_ids),),
            "native_point": (len(expected_ids),),
            "target": (len(expected_ids),),
            "ids": (len(expected_ids),),
        }
        rec.must(f"seed {seed} fold {fold_index} {role} tensor shapes", all(
            table.get(name, np.empty((0,))).shape == shape for name, shape in expected_table_shapes.items()
        ) and meta.get("arrays") == {name: list(shape) for name, shape in expected_table_shapes.items() if name != "ids"},
                 f"Seed {seed} fold {fold_index} {role} table arrays differ from the registered context/point/target schema.")
        table_ids = table["ids"].astype(str).tolist()
        rec.must(f"seed {seed} fold {fold_index} {role} canonical IDs", table_ids == expected_ids,
                 f"Seed {seed} fold {fold_index} {role} NPZ rows are not in canonical author TRAIN order.")

        indices = np.asarray([cache_position[sid] for sid in expected_ids], dtype=np.int64)
        normalized: dict[str, np.ndarray] = {}
        for modality in MODALITIES:
            stats = expected_stats[modality]
            mean = np.asarray(stats["mean"], dtype=np.float32)
            scale = np.asarray(stats["scale"], dtype=np.float32)
            normalized[modality] = ((features[modality][indices] - mean) / scale).astype(np.float32)
        joined = np.concatenate([normalized[modality] for modality in MODALITIES], axis=1)
        torch.set_num_threads(1)
        w0, b0 = state["head.0.weight"].cpu(), state["head.0.bias"].cpu()
        w1, b1 = state["head.3.weight"].cpu(), state["head.3.bias"].cpu()
        expected_contexts: list[np.ndarray] = []
        expected_points: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(joined), 512):
                batch = torch.from_numpy(joined[start:start + 512])
                hidden = F.gelu(F.linear(batch, w0, b0))
                # The selected model is evaluated; Dropout(.1) is identity.
                prediction = F.linear(hidden, w1, b1).flatten()
                expected_contexts.append(hidden.cpu().numpy())
                expected_points.append(prediction.cpu().numpy())
        cpu_context = np.concatenate(expected_contexts).astype(np.float32)
        cpu_point = np.concatenate(expected_points).astype(np.float32)
        cpu_anchor = np.clip(cpu_point, -1., 1.).astype(np.float32)
        target_expected = np.asarray([author_targets[sid] for sid in expected_ids], dtype=np.float32)
        errors = {
            "context": max_abs_difference(cpu_context, table["context"]),
            "native_point": max_abs_difference(cpu_point, table["native_point"]),
            "anchor": max_abs_difference(cpu_anchor, table["anchor"]),
            "target": max_abs_difference(target_expected, table["target"]),
        }
        for array_name, error in errors.items():
            max_errors[array_name] = max(max_errors.get(array_name, 0.), error)
        rec.must(f"seed {seed} fold {fold_index} {role} independent CPU forward", 
                 max(errors["context"], errors["native_point"], errors["anchor"]) <= CPU_GPU_TOLERANCE,
                 f"Seed {seed} fold {fold_index} {role} CPU replay differs from saved point table by more than {CPU_GPU_TOLERANCE}: {errors}.")
        rec.must(f"seed {seed} fold {fold_index} {role} target and anchor semantics",
                 errors["target"] == 0. and max_abs_difference(np.clip(table["native_point"], -1., 1.).astype(np.float32), table["anchor"]) <= 1e-7,
                 f"Seed {seed} fold {fold_index} {role} target is not the original author M score or anchor is not clipped raw native point.")


def audit(suite_path: Path, config_path: Path, repo_root: Path) -> dict[str, Any]:
    rec = Recorder()
    report: dict[str, Any] = {
        "audit": "independent-cpu-native-nested-shape-point-stage-v1",
        "suite": str(suite_path),
        "config": str(config_path),
        "cpu_gpu_tolerance": CPU_GPU_TOLERANCE,
        "checks": rec.checks,
        "findings": rec.findings,
        "complete": False,
    }
    max_errors: dict[str, float] = {}
    audited_models = 0
    try:
        config_bytes_sha = sha256_file(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        manifest_path = suite_path.with_suffix(suite_path.suffix + ".manifest.json")
        rec.must("formal suite file", suite_path.is_file(), "Formal suite.json is missing; wait for all nine point identities and 27 tables to finish.")
        rec.must("pinned config hash", config.get("source_csv_sha256") == EXPECTED_CSV_SHA256
                 and config_bytes_sha == config.get("config_sha256", config_bytes_sha),
                 "Registered config is not bound to the pinned CH-SIMS source CSV or its recorded config hash.")
        csv_path = resolve_path(config["author_csv"], repo_root)
        csv_sha = sha256_file(csv_path)
        rec.must("author CSV SHA256", csv_sha == EXPECTED_CSV_SHA256,
                 f"Author CSV SHA256 is {csv_sha}; expected pinned TRAIN source CSV SHA256 {EXPECTED_CSV_SHA256}.")
        author_ids, author_targets = read_author_train(csv_path)
        source_count = len({source_of(sid) for sid in author_ids})
        rec.must("pinned author TRAIN cohort", len(author_ids) == 2722 and source_count == 129,
                 f"Author CSV has {len(author_ids)} TRAIN IDs from {source_count} sources; expected 2,722 and 129.")

        rec.must("suite manifest file", manifest_path.is_file(), "Suite sidecar manifest is missing; the suite cannot be identity-audited.")
        suite_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rec.must("suite-to-manifest hash", suite.get("manifest_sha256") == sha256_file(manifest_path)
                 and resolve_path(suite.get("manifest_path", ""), repo_root).resolve() == manifest_path.resolve(),
                 "suite.json manifest path/hash does not point to its adjacent immutable stage manifest.")
        rec.must("formal suite completion markers", suite.get("suite_complete") is True
                 and suite.get("smoke_only") is False and suite.get("full_matrix_requested") is True
                 and suite.get("formal_point_matrix_count") == 9
                 and suite.get("expected_formal_point_matrix_count") == 9
                 and suite.get("table_count") == 27 and suite.get("expected_formal_table_count") == 27,
                 "suite.json is partial, smoke-only, or does not declare the full 9-model/27-table matrix.")
        rec.must("suite manifest source/config identity", suite_manifest.get("config_sha256") == config_bytes_sha
                 and suite_manifest.get("config") == config
                 and suite_manifest.get("author_csv_sha256") == EXPECTED_CSV_SHA256
                 and suite_manifest.get("official_VALID_used") is False
                 and suite_manifest.get("official_TEST_used") is False
                 and suite_manifest.get("smoke_only") is False,
                 "Suite manifest is not bound to the registered config, pinned author CSV, and TRAIN-only formal mode.")
        artifact_root = resolve_path(suite_manifest.get("artifact_root", ""), repo_root)
        rec.must("formal artifact root", artifact_root.is_dir()
                 and resolve_path(suite.get("artifact_root", ""), repo_root).resolve() == artifact_root.resolve(),
                 "Suite artifact_root is missing or differs between suite.json and suite manifest.")
        rec.must("producer implementation freeze", config.get("nested_pipeline", {}).get("implementation_sha256")
                 == sha256_file(repo_root / "jevaffect" / "nested_shape_points.py"),
                 "Current nested point producer differs from the implementation SHA frozen in config; audit the intended source revision before trusting tables.")

        cache_ids, cache_position, cache_targets, features, dimensions, file_hashes, feature_content_sha = _read_cache(
            config, repo_root, author_ids, author_targets, suite_manifest, rec,
        )
        rec.must("suite manifest author ID order", suite_manifest.get("canonical_author_train_ids") == author_ids
                 and suite_manifest.get("author_train_count") == len(author_ids)
                 and suite_manifest.get("author_source_count") == source_count,
                 "Suite canonical author TRAIN IDs/counts differ from an independent CSV parse.")
        rec.must("cache-row-order-safe ID join", len(cache_position) == len(author_ids)
                 and all(cache_ids[cache_position[sid]] == sid for sid in author_ids)
                 and max_abs_difference(cache_targets, np.asarray([author_targets[sid] for sid in cache_ids], dtype=np.float32)) == 0.,
                 "The cache-to-author ID index is not bijective or cache-order targets do not match the author CSV; exported tables cannot be safely reconstructed by ID.")
        outer_ids, rebuilt = rebuild_source_roles(author_ids)
        nested = suite_manifest.get("nested_manifest", {})
        rec.must("nested manifest content hash", suite_manifest.get("nested_manifest_sha256") == canonical_hash(nested),
                 "Suite nested role manifest content hash does not match its serialized manifest.")
        _assert_expected_manifest(nested, author_ids, outer_ids, rebuilt, rec)
        rec.must("manifest implementation hashes", suite_manifest.get("implementation_sha256", {}).get("nested_shape_points.py")
                 == sha256_file(repo_root / "jevaffect" / "nested_shape_points.py")
                 and suite_manifest.get("implementation_sha256", {}).get("anchor_crossfit.py")
                 == sha256_file(repo_root / "jevaffect" / "anchor_crossfit.py")
                 and suite_manifest.get("implementation_sha256", {}).get("standardized_source.py")
                 == sha256_file(repo_root / "jevaffect" / "standardized_source.py"),
                 "Suite manifest implementation hashes differ from the current nested/anchor/standardized modules.")

        rows = suite.get("results", [])
        expected_keys = {(seed, fold) for seed in SEEDS for fold in FOLDS}
        actual_keys = {(row.get("seed"), row.get("fold")) for row in rows}
        rec.must("nine unique point identities", len(rows) == 9 and actual_keys == expected_keys
                 and len(actual_keys) == len(rows),
                 f"suite.json contains {len(rows)} point results; expected one each for seeds {SEEDS} and folds {FOLDS}.")
        root_tables = suite_manifest.get("nested_manifest", {}).get("folds", [])
        if rows and len(root_tables) == 3:
            for row in rows:
                seed, fold_index = row.get("seed"), row.get("fold")
                try:
                    _audit_one_model(
                        row, rebuilt["folds"][fold_index], seed=seed, fold_index=fold_index,
                        config=config, config_sha=config_bytes_sha, suite_manifest=suite_manifest,
                        repo_root=repo_root, canonical_ids=author_ids, author_targets=author_targets,
                        cache_ids=cache_ids, cache_position=cache_position, cache_targets=cache_targets,
                        features=features, dimensions=dimensions, file_hashes=file_hashes,
                        feature_content_sha=feature_content_sha, rec=rec, max_errors=max_errors,
                    )
                    audited_models += 1
                except (AuditError, AssertionError, KeyError, OSError, ValueError, TypeError) as exc:
                    # One failed identity should not hide actionable issues in
                    # the remaining eight independent model/table artifacts.
                    report["findings"].append({
                        "check": f"seed {seed} fold {fold_index} audit continuation",
                        "issue": str(exc),
                    })
        else:
            rec.must("nested fold rows available", len(root_tables) == 3,
                     "Suite manifest does not contain all three nested folds for per-model auditing.")
        report["complete"] = not report["findings"] and audited_models == 9
        report["audited_point_identities"] = audited_models
        report["audited_tables"] = audited_models * 3
        report["max_cpu_replay_absolute_error"] = max_errors
        report["role_semantics"] = {
            "outer": "each TRAIN ID/source is outer-held once across the three folds",
            "calibration": "shape_fit and shape_select are separate source roles excluded from point fitting, selection, and scaler fitting",
            "point_holdout": "shape_fit + shape_select + this fold's outer rows; it repeats across folds and is not global once-only OOF",
        }
        report["summary"] = {
            "author_train_ids": len(author_ids),
            "author_train_sources": source_count,
            "raw_cache_order_differs_from_author_order": cache_ids != author_ids,
            "raw_cache_content_sha256": feature_content_sha,
            "raw_cache_file_count": len(file_hashes),
            "dimensions": dimensions,
            "checks_passed": sum(1 for item in rec.checks if item["passed"]),
            "checks_failed": sum(1 for item in rec.checks if not item["passed"]),
        }
    except (AuditError, AssertionError, KeyError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        if not any(item.get("issue") == str(exc) for item in report["findings"]):
            report["findings"].append({"check": "audit setup", "issue": str(exc)})
        report["complete"] = False
        report["summary"] = {
            "checks_passed": sum(1 for item in rec.checks if item["passed"]),
            "checks_failed": sum(1 for item in rec.checks if not item["passed"]),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suite", type=Path, help="Formal suite.json; defaults to runs/native_nested_shape_points_v1/suite.json")
    parser.add_argument("--config", type=Path, help="Registered config JSON; defaults to configs/native_nested_shape_points_v1.json")
    parser.add_argument("--report", type=Path, help="Output report; defaults to docs/native_nested_shape_point_audit_20261013.json")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    suite_path = (args.suite or Path("runs/native_nested_shape_points_v1/suite.json"))
    config_path = (args.config or Path("configs/native_nested_shape_points_v1.json"))
    report_path = (args.report or Path("docs/native_nested_shape_point_audit_20261013.json"))
    suite_path = resolve_path(suite_path, repo_root)
    config_path = resolve_path(config_path, repo_root)
    report_path = resolve_path(report_path, repo_root)
    result = audit(suite_path, config_path, repo_root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "complete": result.get("complete", False),
        "audited_point_identities": result.get("audited_point_identities", 0),
        "audited_tables": result.get("audited_tables", 0),
        "summary": result.get("summary", {}),
        "findings": result.get("findings", []),
        "report": str(report_path),
    }, indent=2, allow_nan=False))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
