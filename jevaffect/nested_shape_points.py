"""Nested, source-honest point-model and shape-table preparation.

This stage prepares selected point anchors and 256-wide fusion contexts for a
later censored-median Bernstein shape study.  It owns a new outer-source
partition and never reuses the older global OOF calibration protocol.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from jevaffect import anchor_crossfit, standardized_source
from jevaffect.native_training import PooledRegressor, digest, save_json
from jevaffect.source_crossfit import group_of, manifest_hash


FORMAL_SEEDS = (13, 42, 2026)
OUTER_FOLD_SEED = 20261012
ROLE_SEED = 20261013
ROLE_NAMES = ("point_fit", "point_select", "shape_fit", "shape_select", "outer")
TABLE_ROLES = ("shape_fit", "shape_select", "outer")
PIPELINE_NAME = "nested-source-honest-shape-tables-v1"


def feature_content_sha256(dataset) -> str:
    """Hash the pooled arrays actually handed to normalization/training."""
    h = hashlib.sha256()
    for modality in sorted(dataset.features):
        values = np.asarray(dataset.features[modality])
        if values.dtype != np.float32 or values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Pooled fusion features must be finite float32 matrices")
        h.update(modality.encode("utf-8")); h.update(b"\0")
        h.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
        h.update(values.tobytes(order="C"))
    return h.hexdigest()


def _role_ids(ids, sources):
    sources = set(sources)
    return [sid for sid in ids if group_of(sid) in sources]


def validate_nested_manifest(ids, manifest):
    """Validate nested roles and true outer once-only coverage ourselves."""
    ids = list(ids)
    universe = set(ids)
    if len(ids) != len(universe) or not manifest.get("folds"):
        raise ValueError("Nested manifest requires unique canonical TRAIN IDs and folds")
    outer_seen, outer_sources_seen = [], []
    for index, fold in enumerate(manifest["folds"]):
        if fold.get("fold") != index:
            raise ValueError("Nested outer folds must be ordered 0, 1, 2")
        role_lists = {role: list(fold.get(role + "_ids", [])) for role in ROLE_NAMES}
        if any(not values or len(values) != len(set(values)) for values in role_lists.values()):
            raise ValueError("Every nested role must contain unique source-held TRAIN IDs")
        role_sets = {role: set(values) for role, values in role_lists.items()}
        if set.union(*role_sets.values()) != universe:
            raise ValueError("Nested roles must partition the complete TRAIN cohort per outer fold")
        source_sets = {role: {group_of(sid) for sid in role_sets[role]} for role in ROLE_NAMES}
        role_keys = list(ROLE_NAMES)
        for left_index, left in enumerate(role_keys):
            for right in role_keys[left_index + 1:]:
                if source_sets[left] & source_sets[right]:
                    raise ValueError("Nested roles must be source-disjoint")
        if any(not source_sets[role] for role in ROLE_NAMES):
            raise ValueError("Each nested role needs at least one complete source")
        if role_sets["outer"] != set(fold.get("outer_ids", [])):
            raise ValueError("Nested outer IDs must match the parent outer OOF IDs")
        point_holdout = set(fold.get("point_oof_ids", []))
        if point_holdout != role_sets["shape_fit"] | role_sets["shape_select"] | role_sets["outer"]:
            raise ValueError("Point holdout must contain calibration roles plus this outer fold")
        if set(fold.get("fit_ids", [])) != role_sets["point_fit"]:
            raise ValueError("Frozen point trainer fit role differs from point_fit")
        if set(fold.get("select_ids", [])) != role_sets["point_select"]:
            raise ValueError("Frozen point trainer selection role differs from point_select")
        if set(fold.get("oof_ids", [])) != point_holdout:
            raise ValueError("Frozen point holdout must be explicit and nested")
        if set(fold.get("point_scaler_fit_ids", [])) != role_sets["point_fit"]:
            raise ValueError("Scaler statistics may use point_fit IDs only")
        if not (source_sets["shape_fit"] | source_sets["shape_select"]).isdisjoint(
            source_sets["point_fit"] | source_sets["point_select"]
        ):
            raise ValueError("Calibration sources cannot enter point fitting or selection")
        outer_seen.extend(role_lists["outer"])
        outer_sources_seen.extend(sorted(source_sets["outer"]))
    if (len(outer_seen) != len(universe) or len(set(outer_seen)) != len(universe)
            or set(outer_seen) != universe):
        raise ValueError("Outer OOF IDs must cover every TRAIN ID exactly once")
    if len(outer_sources_seen) != len(set(outer_sources_seen)):
        raise ValueError("Outer source groups must be held out exactly once")
    return True


def build_nested_manifest(ids, outer_folds, *, smoke=False, role_seed=ROLE_SEED):
    """Split each parent outer-training side into point and calibration roles."""
    ids = list(ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Canonical TRAIN IDs must be unique")
    if len(outer_folds) != 3:
        raise ValueError("The registered outer design has exactly three source folds")
    all_sources = {group_of(sid) for sid in ids}
    outer_sets = []
    outer_ids = []
    for index, parent in enumerate(outer_folds):
        held_ids = list(parent["oof_ids"])
        held_sources = {group_of(sid) for sid in held_ids}
        if (not held_ids or len(held_ids) != len(set(held_ids))
                or not set(held_ids) <= set(ids)
                or set(held_ids) != set(_role_ids(ids, held_sources))):
            raise ValueError("Parent outer OOF IDs must hold complete sources")
        outer_ids.append(held_ids)
        outer_sets.append(held_sources)
    if (set.union(*outer_sets) != all_sources
            or any(outer_sets[i] & outer_sets[j] for i in range(3) for j in range(i))):
        raise ValueError("Parent outer OOF sources must cover TRAIN exactly once")

    folds = []
    for fold_index, outer_source_set in enumerate(outer_sets):
        remaining = sorted(all_sources - outer_source_set)
        rng = np.random.default_rng(role_seed + 1000 * fold_index)
        ordered = [str(source) for source in rng.permutation(remaining)]
        requested_cal = int(math.ceil(.15 * len(remaining)))
        cal_count = max(requested_cal, 2) if smoke else requested_cal
        if cal_count >= len(remaining):
            raise ValueError("Nested calibration allocation leaves no point cohort")
        calibration_sources = ordered[:cal_count]
        after_calibration = len(remaining) - cal_count
        requested_point_select = int(math.ceil(.1 * after_calibration))
        point_select_count = max(requested_point_select, 1) if smoke else requested_point_select
        if point_select_count >= after_calibration:
            raise ValueError("Nested point selection allocation leaves no point-fit cohort")
        point_select_sources = ordered[cal_count:cal_count + point_select_count]
        point_fit_sources = ordered[cal_count + point_select_count:]

        shape_order = [str(source) for source in rng.permutation(calibration_sources)]
        requested_shape_select = int(math.ceil(.3 * cal_count))
        shape_select_count = min(max(requested_shape_select, 1), cal_count - 1) if smoke else requested_shape_select
        if shape_select_count < 1 or shape_select_count >= cal_count:
            raise ValueError("Calibration allocation must leave shape-fit and shape-select sources")
        shape_select_sources = shape_order[:shape_select_count]
        shape_fit_sources = shape_order[shape_select_count:]

        role_sources = {
            "point_fit": point_fit_sources,
            "point_select": point_select_sources,
            "shape_fit": shape_fit_sources,
            "shape_select": shape_select_sources,
            "outer": sorted(outer_source_set),
        }
        role_lists = {role: _role_ids(ids, sources) for role, sources in role_sources.items()}
        point_oof_ids = [sid for sid in ids if sid in (
            set(role_lists["shape_fit"]) | set(role_lists["shape_select"]) | set(role_lists["outer"])
        )]
        fold_manifest = {
            "fold": fold_index,
            **{role + "_ids": role_lists[role] for role in ROLE_NAMES},
            "fit_ids": role_lists["point_fit"],
            "select_ids": role_lists["point_select"],
            "oof_ids": point_oof_ids,
            "outer_ids": role_lists["outer"],
            "point_oof_ids": point_oof_ids,
            "point_scaler_fit_ids": role_lists["point_fit"],
            "role_sources": {role: sorted(set(role_sources[role])) for role in ROLE_NAMES},
            "requested_source_counts": {
                "calibration": requested_cal,
                "point_select": requested_point_select,
                "shape_select": requested_shape_select,
            },
            "source_counts": {role: len(set(role_sources[role])) for role in ROLE_NAMES},
            "point_holdout_semantics": (
                "point OOF is this fold's shape_fit + shape_select calibration sources plus outer-held sources; "
                "it is not global once-only OOF"
            ),
            "scaler_scope": "point_fit sources only; point_select, calibration, and outer sources excluded",
            "outer_metrics_scope": "fixed-stage final readout only; never point selection or normalization",
        }
        folds.append(fold_manifest)
    result = {
        "name": PIPELINE_NAME,
        "scope": "official CH-SIMS TRAIN only; source-honest nested outer evaluation",
        "role_seed": role_seed,
        "outer_fold_seed": OUTER_FOLD_SEED,
        "outer_fold_count": 3,
        "fractions": {"calibration_sources": .15, "point_selection_sources_after_calibration": .1,
                      "shape_selection_sources_within_calibration": .3},
        "smoke_minimum_policy": (
            "smoke only: reserve at least two calibration sources and one source for each of the four inner roles; "
            "outer remains a complete source fold"
        ),
        "smoke": bool(smoke),
        "outer_parent": {
            "source": "make_folds(seed=20261012).oof_ids",
            "fold_seed": OUTER_FOLD_SEED,
            "outer_oof_ids": outer_ids,
        },
        "folds": folds,
        "full_manifest_compatibility_interface": "unused by this nested stage; retained only for frozen trainer identity compatibility",
        "official_VALID_used": False,
        "official_TEST_used": False,
    }
    validate_nested_manifest(ids, result)
    return result


def validate_registered_config(config, *, allow_fixture=False):
    if config.get("dataset") != "simsv2":
        raise ValueError("Nested shape points are registered for CH-SIMS v2 only")
    if config.get("predict_valid") is not False:
        raise ValueError("Nested shape point stage is TRAIN-only; set predict_valid=false")
    if (config.get("fold_count") != 3 or config.get("fold_seed") != OUTER_FOLD_SEED
            or tuple(config.get("seeds", ())) != FORMAL_SEEDS):
        raise ValueError("Nested outer folds and formal seeds are frozen")
    pipeline = config.get("nested_pipeline")
    if not isinstance(pipeline, dict) or pipeline.get("name") != PIPELINE_NAME:
        raise ValueError("Missing registered nested_pipeline configuration")
    if pipeline.get("implementation_sha256") != digest(Path(__file__)):
        raise ValueError("Nested shape point implementation SHA differs from frozen config")
    expected = {
        "role_seed": ROLE_SEED,
        "calibration_source_fraction": .15,
        "point_selection_fraction": .1,
        "shape_selection_fraction": .3,
        "point_holdout_semantics": "shape_fit+shape_select+outer; not global once-only OOF",
        "scaler_fit_role": "point_fit",
        "context_definition": "selected standardized fusion Linear(1920,256) then GELU and eval-mode Dropout",
        "anchor_definition": "clip(selected raw native point output,-1,1)",
        "target_definition": "original native CH-SIMS M score, float32",
        "export_tables": ["shape_fit", "shape_select", "outer"],
        "outer_role": "fixed-stage final readout only; excluded from point selection and normalization",
        "official_VALID_used": False,
        "official_TEST_used": False,
    }
    for key, value in expected.items():
        if pipeline.get(key) != value:
            raise ValueError("Nested pipeline guard differs at " + key)
    if not allow_fixture:
        if (config.get("source_csv_sha256") != anchor_crossfit.SOURCE_CSV_SHA256
                or config.get("author_csv") != "data/author/simsv2/meta.csv"
                or config.get("cache_dir") != "data/feature_cache/simsv2_full_macbert_raw_av"):
            raise ValueError("Frozen author CSV and raw feature cache paths are pinned")
        fixed = {
            "batch_size": 32, "hidden": 256, "dropout": .1,
            "learning_rate": .0001, "weight_decay": .0001, "max_epochs": 80,
            "patience": 12, "gradient_norm": 1.0, "scheduler_patience": 3,
            "scheduler_factor": .5,
        }
        for key, value in fixed.items():
            if config.get(key) != value:
                raise ValueError("Frozen native point model differs at " + key)
        if tuple(config.get("seeds", ())) != FORMAL_SEEDS:
            raise ValueError("Formal seed contract is 13, 42, 2026")
    return True


def prepare_nested(config, *, allow_fixture=False, smoke_groups=None, config_sha256=None):
    """Load only TRAIN and build the new explicit nested role manifest."""
    validate_registered_config(config, allow_fixture=allow_fixture)
    data = anchor_crossfit.prepare(config, allow_fixture=allow_fixture, smoke_groups=smoke_groups)
    if data["valid_features"] is not None:
        raise ValueError("Official VALID features must remain unloaded in this stage")
    if not allow_fixture and smoke_groups is None and len(data["ids"]) != 2722:
        raise ValueError("Pinned author TRAIN must contain 2,722 IDs")
    dims = {m: int(x.shape[-1]) for m, x in data["train"].features.items()}
    if set(dims) != {"text", "audio", "vision"}:
        raise ValueError("Fusion point model requires text, audio, and vision pooled features")
    if not allow_fixture and sum(dims.values()) != 1920:
        raise ValueError("Pinned fusion input width must be 1,920")
    nested = build_nested_manifest(data["ids"], data["fold_manifest"]["folds"],
                                   smoke=bool(allow_fixture or smoke_groups is not None))
    source_outer = {"seed": OUTER_FOLD_SEED, "fold_count": 3,
                    "outer_oof_ids": copy.deepcopy(nested["outer_parent"]["outer_oof_ids"])}
    data["outer_fold_manifest"] = source_outer
    data["fold_manifest"] = nested
    data["nested_manifest"] = nested
    data["full_manifest"] = {
        "compatibility_only": True,
        "used_by_nested_stage": False,
        "description": "Frozen trainer compatibility field; no full-training cohort is selected or used.",
    }
    data["pooled_feature_sha256"] = feature_content_sha256(data["train"])
    data["config_sha256"] = config_sha256
    data["nested_manifest_sha256"] = manifest_hash(nested)
    data["point_pipeline_module_sha256"] = digest(Path(__file__))
    return data


def _verify_data_guard(data):
    actual = feature_content_sha256(data["train"])
    if actual != data.get("pooled_feature_sha256"):
        raise ValueError("Pooled feature content changed after nested role preparation")
    cache_ids = set(data["train"].ids)
    if (len(cache_ids) != len(data["train"].ids)
            or not set(data["ids"]) <= cache_ids
            or (data.get("smoke_groups") is None and cache_ids != set(data["ids"]))):
        raise ValueError("Pooled feature rows must align with the canonical author TRAIN IDs")
    if data.get("point_pipeline_module_sha256") != digest(Path(__file__)):
        raise ValueError("Nested shape point code changed after role preparation")
    validate_nested_manifest(data["ids"], data["nested_manifest"])


def _write_npz(path, *, context, anchor, native_point, target, ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            context=np.asarray(context, dtype=np.float32),
            anchor=np.asarray(anchor, dtype=np.float32),
            native_point=np.asarray(native_point, dtype=np.float32),
            target=np.asarray(target, dtype=np.float32),
            ids=np.asarray(ids, dtype=str),
        )
    temporary.replace(path)
    return digest(path)


def _predict_tables(config, data, fold, model_root, table_root):
    """Read the selected checkpoint once, then export each held role."""
    model_root, table_root = Path(model_root), Path(table_root)
    best_path = model_root / "best.pt"
    if not best_path.exists():
        return {}
    device = torch.device(config.get("device", "cpu"))
    dims = {m: data["train"].features[m].shape[-1] for m in ("text", "audio", "vision")}
    model = PooledRegressor(dims, hidden=config["hidden"], dropout=config["dropout"]).to(device)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    ids = data["ids"]
    position = {sid: index for index, sid in enumerate(data["train"].ids)}
    fold_roles = data["nested_manifest"]["folds"][fold]
    result = {}
    for role in TABLE_ROLES:
        role_set = set(fold_roles[role + "_ids"])
        table_ids = [sid for sid in ids if sid in role_set]
        indices = np.asarray([position[sid] for sid in table_ids], dtype=np.int64)
        contexts, native_points = [], []
        with torch.inference_mode():
            for start in range(0, len(indices), 512):
                batch_indices = indices[start:start + 512]
                features = {
                    modality: torch.as_tensor(
                        np.asarray(data["train"].features[modality][batch_indices], dtype=np.float32),
                        device=device,
                    )
                    for modality in ("text", "audio", "vision")
                }
                raw = model(features)
                joined = torch.cat([features[m] for m in model.modalities], dim=-1)
                context = model.head[0](joined)
                context = model.head[1](context)
                context = model.head[2](context)  # eval-mode dropout is identity
                contexts.append(context.float().cpu().numpy())
                native_points.append(raw.float().cpu().numpy())
        context = np.concatenate(contexts, axis=0)
        native_point = np.concatenate(native_points, axis=0)
        if context.shape != (len(table_ids), int(config["hidden"])):
            raise ValueError("Selected fusion trunk width differs from the registered hidden layer")
        if not np.isfinite(context).all() or not np.isfinite(native_point).all():
            raise ValueError("Nonfinite selected point/context output")
        target = np.asarray([data["train"].targets[position[sid]] for sid in table_ids], dtype=np.float32)
        anchor = np.clip(native_point, -1.0, 1.0).astype(np.float32)
        table_path = table_root / (role + ".npz")
        table_sha = _write_npz(table_path, context=context, anchor=anchor,
                               native_point=native_point, target=target, ids=table_ids)
        id_path = table_root / (role + "_ids.json")
        save_json(id_path, table_ids)
        result[role] = {
            "path": str(table_path), "sha256": table_sha, "count": len(table_ids),
            "ids_path": str(id_path), "ids_sha256": digest(id_path),
            "ids_in_canonical_author_train_order": True,
            "arrays": {"context": [len(table_ids), int(config["hidden"])], "anchor": [len(table_ids)],
                       "native_point": [len(table_ids)], "target": [len(table_ids)]},
            "source_groups": fold_roles["role_sources"][role],
            "source_csv_sha256": data["author_csv_sha256"],
            "raw_feature_file_sha256": data["train"].hashes,
            "config_sha256": data.get("config_sha256"),
            "implementation_sha256": {
                "nested_shape_points.py": digest(Path(__file__)),
                "anchor_crossfit.py": digest(Path(anchor_crossfit.__file__)),
                "standardized_source.py": digest(Path(standardized_source.__file__)),
                "native_training.py": digest(Path(anchor_crossfit.__file__).with_name("native_training.py")),
                "source_crossfit.py": digest(Path(anchor_crossfit.__file__).with_name("source_crossfit.py")),
                "native_metrics.py": digest(Path(anchor_crossfit.__file__).with_name("native_metrics.py")),
            },
        }
    return result


def train_point_fold(config, data, *, seed, fold, output_root, resume=False,
                     stop_after=None, config_sha256=None):
    """Train one nested point teacher through the frozen trainer and export tables."""
    if seed not in FORMAL_SEEDS or fold not in (0, 1, 2):
        raise ValueError("Nested point model requires a registered seed and outer fold")
    _verify_data_guard(data)
    if config_sha256 is not None and data.get("config_sha256") not in (None, config_sha256):
        raise ValueError("Config file changed after nested role preparation")
    cohort = data["nested_manifest"]["folds"][fold]
    cfg, transformed = standardized_source.transform_fold(config, data, fold)
    cfg["nested_pipeline"] = {
        **cfg["nested_pipeline"],
        "config_file_sha256": data.get("config_sha256"),
        "pooled_feature_content_sha256": data["pooled_feature_sha256"],
        "nested_role_manifest_sha256": data["nested_manifest_sha256"],
    }
    if config_sha256 is not None and data.get("config_sha256") != config_sha256:
        raise ValueError("Config file changed after nested role preparation")
    if cfg["preprocessing"]["fit_ids"] != cohort["point_fit_ids"]:
        raise ValueError("Scaler fit IDs differ from the nested point_fit role")
    if set(cfg["preprocessing"]["fit_ids"]) & set(
        cohort["shape_fit_ids"] + cohort["shape_select_ids"] + cohort["outer_ids"]
    ):
        raise ValueError("Calibration/outer IDs leaked into point scaler fitting")
    point_result = anchor_crossfit.train_teacher(
        cfg, transformed, view="fusion", seed=seed, fold=fold,
        resume=resume, stop_after=stop_after, output_root=output_root,
    )
    model_root = Path(point_result["root"])
    save_json(model_root / "nested_roles.json", cohort)
    scaler_path = model_root / "point_fit_scaler.json"
    save_json(scaler_path, {
        "fit_ids": cfg["preprocessing"]["fit_ids"],
        "fit_source_groups": sorted(cohort["role_sources"]["point_fit"]),
        "fit_role": "point_fit only",
        "statistics": cfg["preprocessing"]["statistics"],
        "preprocessing": cfg["preprocessing"],
        "sha256": manifest_hash(cfg["preprocessing"]),
    })
    table_root = Path(output_root).parent / "tables" / f"seed_{seed}" / f"fold_{fold}"
    table_root.mkdir(parents=True, exist_ok=True)
    canonical_ids_path = table_root / "canonical_train_ids.json"
    save_json(canonical_ids_path, data["ids"])
    tables = _predict_tables(cfg, transformed, fold, model_root, table_root)
    result = {
        "seed": seed, "fold": fold, "point_result": point_result,
        "complete": bool(point_result["complete"] and set(tables) == set(TABLE_ROLES)),
        "point_model_root": str(model_root),
        "point_best_checkpoint_sha256": digest(model_root / "best.pt"),
        "point_last_checkpoint_sha256": digest(model_root / "last.pt"),
        "point_history_sha256": digest(model_root / "history.json"),
        "point_identity_sha256": digest(model_root / "identity.json"),
        "point_oof_predictions_sha256": digest(model_root / "oof_predictions.csv"),
        "point_scaler_path": str(scaler_path), "point_scaler_sha256": digest(scaler_path),
        "canonical_train_ids_path": str(canonical_ids_path),
        "canonical_train_ids_sha256": digest(canonical_ids_path),
        "tables": tables,
        "nested_role_manifest_sha256": data["nested_manifest_sha256"],
        "raw_feature_content_sha256": data["pooled_feature_sha256"],
        "raw_feature_file_sha256": data["train"].hashes,
        "source_csv_sha256": data["author_csv_sha256"],
        "implementation_sha256": {
            "nested_shape_points.py": digest(Path(__file__)),
            "anchor_crossfit.py": digest(Path(anchor_crossfit.__file__)),
            "standardized_source.py": digest(Path(standardized_source.__file__)),
            "native_training.py": digest(Path(anchor_crossfit.__file__).with_name("native_training.py")),
            "source_crossfit.py": digest(Path(anchor_crossfit.__file__).with_name("source_crossfit.py")),
            "native_metrics.py": digest(Path(anchor_crossfit.__file__).with_name("native_metrics.py")),
        },
        "official_VALID_used": False, "official_TEST_used": False,
        "outer_data_role": "fixed-stage final readout only; excluded from point selection and scaler fit",
    }
    save_json(table_root / "point_stage.json", result)
    return result


def run_nested_suite(config, *, output, seed=None, fold=None, smoke_groups=None,
                     stop_after=None, resume=False, allow_fixture=False,
                     config_sha256=None):
    """Run selected point folds and export up to three tables per fold."""
    validate_registered_config(config, allow_fixture=allow_fixture)
    if seed is not None and seed not in FORMAL_SEEDS:
        raise ValueError("Unknown registered seed")
    if fold is not None and fold not in (0, 1, 2):
        raise ValueError("Outer fold must be 0, 1, or 2")
    if smoke_groups is not None and smoke_groups < 8:
        raise ValueError("Smoke mode must use at least eight complete source groups")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    data = prepare_nested(config, allow_fixture=allow_fixture, smoke_groups=smoke_groups,
                          config_sha256=config_sha256)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = output.parent / (output.stem + "_artifacts")
    point_root = artifact_root / "point_models"
    point_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": PIPELINE_NAME,
        "config": config,
        "config_sha256": config_sha256,
        "nested_manifest": data["nested_manifest"],
        "nested_manifest_sha256": data["nested_manifest_sha256"],
        "canonical_author_train_ids": data["ids"],
        "author_train_count": len(data["ids"]),
        "author_source_count": len({group_of(sid) for sid in data["ids"]}),
        "author_csv_sha256": data["author_csv_sha256"],
        "raw_feature_file_sha256": data["train"].hashes,
        "pooled_feature_content_sha256": data["pooled_feature_sha256"],
        "implementation_sha256": {
            "nested_shape_points.py": digest(Path(__file__)),
            "prepare_nested_shape_points.py": digest(Path(__file__).parents[1] / "scripts" / "prepare_nested_shape_points.py"),
            "anchor_crossfit.py": digest(Path(anchor_crossfit.__file__)),
            "standardized_source.py": digest(Path(standardized_source.__file__)),
        },
        "full_manifest_compatibility_interface": "present but unused; nested roles replace global-OOF calibration",
        "official_VALID_used": False, "official_TEST_used": False,
        "smoke_only": bool(allow_fixture or smoke_groups is not None),
        "smoke_groups": smoke_groups,
        "artifact_root": str(artifact_root),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    save_json(manifest_path, manifest)
    selected_seeds = [seed] if seed is not None else list(FORMAL_SEEDS)
    selected_folds = [fold] if fold is not None else [0, 1, 2]
    results = []
    for current_seed in selected_seeds:
        for current_fold in selected_folds:
            row = train_point_fold(
                config, data, seed=current_seed, fold=current_fold,
                output_root=point_root, resume=resume, stop_after=stop_after,
                config_sha256=config_sha256,
            )
            results.append(row)
            partial = {
                "results": results, "suite_complete": False,
                "smoke_only": manifest["smoke_only"],
                "full_matrix_requested": False,
                "official_VALID_used": False, "official_TEST_used": False,
                "manifest_path": str(manifest_path),
                "manifest_sha256": digest(manifest_path),
            }
            save_json(output, partial)
    full_matrix_requested = (
        seed is None and fold is None and smoke_groups is None and not allow_fixture
        and stop_after is None
    )
    expected = {(s, f) for s in FORMAL_SEEDS for f in range(3)}
    completed = {(r["seed"], r["fold"]) for r in results if r["complete"]}
    complete = full_matrix_requested and completed == expected
    report = {
        "results": results,
        "suite_complete": complete,
        "smoke_only": manifest["smoke_only"],
        "full_matrix_requested": full_matrix_requested,
        "formal_point_matrix_count": len(completed),
        "expected_formal_point_matrix_count": 9,
        "table_count": sum(len(row["tables"]) for row in results),
        "expected_formal_table_count": 27,
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest(manifest_path),
        "artifact_root": str(artifact_root),
        "official_VALID_used": False,
        "official_TEST_used": False,
    }
    save_json(output, report)
    return report

