"""Source-disjoint CH-SIMS point teachers for the AnchorCDF stage.

This module only trains native-M point regressors.  It does not fit a CDF or
replace any existing point prediction.  Labels are read from official TRAIN
rows only; optional official VALID predictions use a feature-only dataset.
"""
from __future__ import annotations

import csv
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jevaffect.native_metrics import native_metrics
from jevaffect.native_training import NativeSplit, PooledRegressor, digest, move, save_checkpoint, save_json
from jevaffect.source_crossfit import (
    group_of,
    manifest_hash,
    make_folds,
    subset,
    training_rows_only,
    validate_folds,
)


SOURCE_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
VIEWS = {
    "fusion": ("text", "audio", "vision"),
    "text": ("text",),
    "audio": ("audio",),
    "vision": ("vision",),
}
FORMAL_SEEDS = (13, 42, 2026)


class PooledView(Dataset):
    """Expose one view from an already pooled NativeSplit without rereading it."""

    def __init__(self, base, modalities):
        self.base = base
        self.modalities = tuple(modalities)
        if not self.modalities or not set(self.modalities) <= set(base.features):
            raise ValueError("View modalities are absent from the pooled base dataset")
        self.ids = base.ids
        self.targets = getattr(base, "targets", None)
        self.features = {m: base.features[m] for m in self.modalities}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        features = {m: torch.from_numpy(np.array(x[index], dtype=np.float32, copy=True))
                    for m, x in self.features.items()}
        sid = self.ids[index]
        if self.targets is None:
            return features, sid
        return features, torch.tensor(float(self.targets[index]), dtype=torch.float32), sid


class FeatureOnlyPooledSplit(Dataset):
    """Pooled split accessor that never reads a label file or target column."""

    def __init__(self, cache, split="valid", modalities=("text", "audio", "vision"), expected_count=None):
        if split != "valid":
            raise ValueError("Feature-only access is reserved for official VALID")
        cache = Path(cache)
        metadata_path = cache / "metadata.json"
        if not metadata_path.exists():
            metadata_path = cache / "meta.json"
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        directory = cache / self.meta.get("split_paths", {}).get(
            split, "splits/valid" if (cache / "splits").exists() else "valid"
        )
        directory = directory.resolve()
        directory.relative_to(cache.resolve())
        ids_path = directory / "ids.json"
        self.ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if len(set(self.ids)) != len(self.ids) or any(group_of(sid) == "" for sid in self.ids):
            raise ValueError("Invalid or duplicate feature-only IDs")
        if expected_count is not None and len(self.ids) != expected_count:
            raise ValueError("Feature-only VALID count differs from the registered cache count")
        recorded = self.meta.get("artifact_sha256", {})
        file_records = self.meta.get("files", {})
        self.hashes = {
            str(metadata_path): digest(metadata_path),
            str(ids_path): digest(ids_path),
        }
        self.features = {}
        for modality in modalities:
            fp, lp = directory / (modality + ".npy"), directory / (modality + "_lengths.npy")
            for path in (fp, lp):
                actual = digest(path)
                key = path.relative_to(cache.resolve()).as_posix()
                known = recorded.get(key, file_records.get(key, {}).get("sha256"))
                if known is None or known != actual:
                    raise ValueError("Missing or mismatched cached feature hash: " + key)
                self.hashes[key] = actual
            x, lengths = np.load(fp, mmap_mode="r"), np.load(lp, mmap_mode="r")
            if x.ndim != 3 or x.shape[0] != len(self.ids) or lengths.shape != (len(self.ids),):
                raise ValueError("Invalid feature-only sequence shapes")
            if (lengths.dtype.kind not in "iu" or np.any(lengths < 0)
                    or np.any(lengths > x.shape[1]) or not np.isfinite(x).all()):
                raise ValueError("Invalid feature-only lengths or nonfinite features")
            if modality == "text":
                pooled = np.asarray(x[:, 0]).copy()
                pooled[lengths == 0] = 0
            else:
                pooled = np.stack([
                    np.asarray(x[i, :int(n)]).mean(0) if n else np.zeros(x.shape[-1], np.float32)
                    for i, n in enumerate(lengths)
                ])
            self.features[modality] = pooled.astype(np.float32)

    def __len__(self):
        return len(self.ids)


def full_selection_manifest(ids, seed=20261013, inner_fraction=.1):
    """One fixed, source-disjoint fit/selection split for the full teachers."""
    ids = list(ids)
    groups = {}
    for sid in ids:
        groups.setdefault(group_of(sid), []).append(sid)
    if len(set(ids)) != len(ids) or len(groups) < 3 or not 0 < inner_fraction < 1:
        raise ValueError("Invalid full-teacher selection cohort")
    rng = np.random.default_rng(seed)
    ordered = rng.permutation(sorted(groups)).tolist()
    selection_count = max(1, int(np.ceil(len(ordered) * inner_fraction)))
    if selection_count >= len(ordered):
        raise ValueError("Empty full-teacher optimization cohort")
    select_groups = set(ordered[:selection_count])
    fit_groups = set(ordered[selection_count:])
    result = {
        "fold": "full",
        "selection_seed": seed,
        "inner_fraction": inner_fraction,
        "fit_sources": sorted(fit_groups),
        "select_sources": sorted(select_groups),
        "fit_ids": [sid for sid in ids if group_of(sid) in fit_groups],
        "select_ids": [sid for sid in ids if group_of(sid) in select_groups],
    }
    if set(result["fit_ids"]) | set(result["select_ids"]) != set(ids):
        raise ValueError("Full-teacher split does not cover TRAIN")
    if set(result["fit_ids"]) & set(result["select_ids"]):
        raise ValueError("Full-teacher ID overlap")
    if fit_groups & select_groups:
        raise ValueError("Full-teacher source overlap")
    return result


def prepare(config, *, allow_fixture=False, smoke_groups=None):
    """Load official TRAIN labels and one pooled train/feature-only valid cache."""
    if config.get("dataset") != "simsv2":
        raise ValueError("Anchor cross-fit is registered for CH-SIMS v2 only")
    csv_path = Path(config["author_csv"])
    csv_hash = digest(csv_path)
    configured_pin = config.get("source_csv_sha256", SOURCE_CSV_SHA256)
    if configured_pin != SOURCE_CSV_SHA256:
        raise ValueError("Configuration cannot override the registered CH-SIMS CSV pin")
    if not allow_fixture and csv_hash != SOURCE_CSV_SHA256:
        raise ValueError("Author CSV differs from the pinned CH-SIMS source file")
    rows = training_rows_only(csv_path)
    if not allow_fixture:
        if len(rows) != 2722 or len({group_of(sid) for sid in rows}) != 129:
            raise ValueError("Pinned CH-SIMS TRAIN must contain 2,722 clips from 129 sources")
    train = NativeSplit(config["cache_dir"], "train", rows, VIEWS["fusion"], pooled=True)
    # PooledSplit copies the arrays into compact in-memory vectors.  Release
    # its per-modality length memmaps now so the cache can be cleaned up on
    # Windows and the four view wrappers share only the pooled arrays.
    train.lengths.clear()
    if not allow_fixture and set(train.ids) != set(rows):
        raise ValueError("Native TRAIN cache differs from pinned author TRAIN IDs")
    # Use author TRAIN row order as the canonical ordering for manifests and
    # merged prediction tables; NativeSplit joins by ID when indexing features.
    ids = list(rows)
    if set(ids) != set(train.ids):
        raise ValueError("Native TRAIN cache differs from the TRAIN-only author rows")
    if smoke_groups is not None:
        if smoke_groups < 8:
            raise ValueError("Smoke mode must use at least eight complete source groups")
        selected_sources = list(dict.fromkeys(group_of(sid) for sid in ids))[:smoke_groups]
        keep = set(selected_sources)
        ids = [sid for sid in ids if group_of(sid) in keep]
        if len(selected_sources) < smoke_groups:
            raise ValueError("Requested smoke source groups exceed the TRAIN cohort")
    manifest = make_folds(ids, config.get("fold_count", 3), config.get("fold_seed", 20261012),
                          config.get("inner_fraction", .1))
    validate_folds(ids, manifest["folds"])
    full = full_selection_manifest(ids, config.get("full_selection_seed", 20261013),
                                   config.get("full_selection_fraction", .1))
    valid = None
    if config.get("predict_valid", True):
        expected_valid = None if allow_fixture else 647
        if not allow_fixture and config.get("expected_valid_count", 647) != 647:
            raise ValueError("Production CH-SIMS VALID count is pinned at 647")
        valid = FeatureOnlyPooledSplit(config["cache_dir"], expected_count=expected_valid)
    return {
        "rows": rows,
        "train": train,
        "ids": ids,
        "fold_manifest": manifest,
        "full_manifest": full,
        "valid_features": valid,
        "author_csv_sha256": csv_hash,
        "fixture_smoke": bool(allow_fixture),
        "smoke_groups": smoke_groups,
    }


def _loader(ds, config, *, shuffle=False, generator=None):
    return DataLoader(ds, batch_size=config["batch_size"], shuffle=shuffle,
                      generator=generator if shuffle else None, num_workers=0)


def _predict_targets(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for features, target, ids in loader:
            prediction = model(move(features, device)).flatten().cpu().numpy()
            target = target.numpy()
            if prediction.shape != target.shape or not np.isfinite(prediction).all():
                raise ValueError("Nonfinite or wrongly shaped native point output")
            records.extend({"id": sid, "target": float(y), "prediction": float(p)}
                           for sid, y, p in zip(ids, target, prediction))
    if not records:
        raise ValueError("Cannot predict an empty cohort")
    metrics = native_metrics(np.asarray([r["prediction"] for r in records], np.float32),
                             np.asarray([r["target"] for r in records], np.float32), "simsv2")
    return metrics, records


def _predict_features(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for features, ids in loader:
            prediction = model(move(features, device)).flatten().cpu().numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("Nonfinite feature-only prediction")
            records.extend({"id": sid, "prediction": float(value)} for sid, value in zip(ids, prediction))
    return records


def _write_predictions(path, records, *, with_target):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "target", "prediction"] if with_target else ["id", "prediction"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fields})
    tmp.replace(path)


def _read_predictions(path, *, with_target):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = ["id", "target", "prediction"] if with_target else ["id", "prediction"]
        if reader.fieldnames != expected:
            raise ValueError("Unexpected prediction CSV columns")
        result = []
        for row in reader:
            record = {"id": row["id"], "prediction": float(row["prediction"])}
            if with_target:
                record["target"] = float(row["target"])
            result.append(record)
        return result


def _artifact_identity(config, data, view, seed, fold, cohort, modalities, device):
    own = Path(__file__)
    dependencies = [own, own.with_name("native_training.py"), own.with_name("source_crossfit.py"),
                    own.with_name("native_metrics.py")]
    return {
        "config": config,
        "dataset": "simsv2",
        "view": view,
        "modalities": list(modalities),
        "seed": seed,
        "rng_seed": seed + (10000 * fold if isinstance(fold, int) else 30000),
        "fold": fold,
        "cohort": cohort,
        "manifest_sha256": manifest_hash(cohort),
        "fold_manifest_sha256": manifest_hash(data["fold_manifest"]),
        "full_selection_manifest_sha256": manifest_hash(data["full_manifest"]),
        "train_ids": list(data["ids"]),
        "feature_hashes": data["train"].hashes,
        "valid_feature_hashes": data["valid_features"].hashes if data["valid_features"] is not None else {},
        "author_csv_sha256": data["author_csv_sha256"],
        "implementation_sha256": {str(path.name): digest(path) for path in dependencies},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "selection": "minimum inner-selection native MAE; first checkpoint wins ties",
        "loss": "unclipped L1 on original CH-SIMS M scores",
        "official_validation_labels_parsed": False,
        "test_evaluated": False,
        "fixture_smoke": data["fixture_smoke"],
        "smoke_groups": data["smoke_groups"],
    }


def train_teacher(config, data, *, view, seed, fold, resume=False, stop_after=None, output_root=None):
    """Train one fold/full view teacher; fold is 0/1/2 or the string 'full'."""
    if view not in VIEWS or seed not in FORMAL_SEEDS:
        raise ValueError("Unknown native view or seed")
    if fold not in (0, 1, 2, "full"):
        raise ValueError("Fold must be 0, 1, 2, or 'full'")
    if config.get("dataset") != "simsv2" or config.get("device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("Invalid AnchorCDF training runtime")
    for key in ("batch_size", "hidden", "max_epochs", "patience"):
        if int(config[key]) < 1:
            raise ValueError("Training counts must be positive")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")

    rng_seed = seed + (10000 * fold if isinstance(fold, int) else 30000)
    torch.set_num_threads(int(config.get("torch_threads", 4)))
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    if config.get("device", "cpu") == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.cuda.manual_seed_all(rng_seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config.get("device", "cpu"))

    modalities = VIEWS[view]
    view_ds = PooledView(data["train"], modalities)
    if fold == "full":
        cohort = data["full_manifest"]
        roles = ("fit", "select")
    else:
        cohort = data["fold_manifest"]["folds"][fold]
        roles = ("fit", "select", "oof")
    dims = {m: view_ds.features[m].shape[-1] for m in modalities}
    model = PooledRegressor(dims, hidden=config["hidden"], dropout=config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=config.get("scheduler_patience", 3),
        factor=config.get("scheduler_factor", .5),
    )
    generator = torch.Generator().manual_seed(rng_seed)
    loaders = {
        role: _loader(subset(view_ds, cohort[role + "_ids"]), config,
                      shuffle=(role == "fit"), generator=generator)
        for role in roles
    }
    if output_root is None:
        output_root = Path(config["run_root"])
    mode = "smoke" if data["smoke_groups"] is not None or data["fixture_smoke"] else "formal"
    folder = "full" if fold == "full" else f"fold_{fold}"
    root = Path(output_root) / mode / view / f"seed_{seed}" / folder
    root.mkdir(parents=True, exist_ok=True)
    identity = _artifact_identity(config, data, view, seed, fold, cohort, modalities, device)
    last_path = root / "last.pt"
    epoch0, best, stale, history = 0, float("inf"), 0, []
    if last_path.exists():
        if not resume:
            raise ValueError("Teacher exists; pass --resume")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Anchor cross-fit checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        epoch0, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_json(root / "identity.json", identity)
    cap = min(config["max_epochs"], epoch0 + stop_after) if stop_after is not None else config["max_epochs"]
    for epoch in range(epoch0, cap):
        if stale >= config["patience"]:
            break
        started = time.perf_counter()
        model.train()
        loss_total, seen, updates = 0.0, 0, 0
        for features, target, _ in loaders["fit"]:
            optimizer.zero_grad(set_to_none=True)
            target = target.to(device)
            prediction = model(move(features, device)).flatten()
            loss = nn.functional.l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite source-crossfit L1 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_norm", 1.0))
            optimizer.step()
            count = len(target)
            loss_total += float(loss.item()) * count
            seen += count
            updates += 1
        if seen == 0:
            raise ValueError("Empty fit cohort")
        metrics, _ = _predict_targets(model, loaders["select"], device)
        scheduler.step(metrics["MAE"])
        row = {
            "epoch": epoch + 1,
            "train_L1": loss_total / seen,
            "inner_select": metrics,
            "optimizer_updates": updates,
            "seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if metrics["MAE"] < best:
            best, stale = metrics["MAE"], 0
            save_checkpoint(root / "best.pt", {
                "identity": identity, "epoch": epoch + 1, "model": model.state_dict(),
                "metrics": metrics,
            })
        else:
            stale += 1
        save_checkpoint(last_path, {
            "identity": identity, "epoch": epoch + 1, "best": best, "stale": stale,
            "history": history, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        })
        save_json(root / "history.json", history)
        print(json.dumps({"view": view, "seed": seed, "fold": fold, **row}, allow_nan=False), flush=True)

    selected = torch.load(root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    complete = stale >= config["patience"] or len(history) >= config["max_epochs"]
    params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if fold == "full":
        train_loader = _loader(subset(view_ds, data["ids"]), config)
        _, train_predictions = _predict_targets(model, train_loader, device)
        train_path = root / "full_train_predictions.csv"
        _write_predictions(train_path, train_predictions, with_target=True)
        valid_path = None
        valid_predictions = []
        if data["valid_features"] is not None:
            valid_view = PooledView(data["valid_features"], modalities)
            valid_loader = _loader(valid_view, config)
            valid_predictions = _predict_features(model, valid_loader, device)
            valid_path = root / "full_valid_predictions.csv"
            _write_predictions(valid_path, valid_predictions, with_target=False)
        result = {
            "root": str(root), "view": view, "seed": seed, "fold": "full",
            "complete": complete, "smoke_only": mode == "smoke", "test_evaluated": False,
            "official_validation_labels_parsed": False,
            "fit_count": len(cohort["fit_ids"]), "select_count": len(cohort["select_ids"]),
            "train_prediction_count": len(train_predictions), "valid_prediction_count": len(valid_predictions),
            "best_epoch": selected["epoch"], "epochs_executed": len(history),
            "optimizer_updates": sum(r["optimizer_updates"] for r in history),
            "loop_seconds": sum(r["seconds"] for r in history), "trainable_parameters": params,
            "checkpoint_sha256": digest(root / "best.pt"),
            "full_train_predictions_file": str(train_path),
            "train_predictions_sha256": digest(train_path),
            "valid_predictions_file": str(valid_path) if valid_path is not None else None,
            "valid_predictions_sha256": digest(valid_path) if valid_path is not None else None,
        }
    else:
        metrics, predictions = _predict_targets(model, loaders["oof"], device)
        oof_path = root / "oof_predictions.csv"
        _write_predictions(oof_path, predictions, with_target=True)
        result = {
            "root": str(root), "view": view, "seed": seed, "fold": fold,
            "complete": complete, "smoke_only": mode == "smoke", "test_evaluated": False,
            "official_validation_labels_parsed": False,
            "manifest_sha256": manifest_hash(data["fold_manifest"]),
            "fit_count": len(cohort["fit_ids"]), "select_count": len(cohort["select_ids"]),
            "oof_count": len(cohort["oof_ids"]), "oof_native": metrics,
            "best_epoch": selected["epoch"], "epochs_executed": len(history),
            "optimizer_updates": sum(r["optimizer_updates"] for r in history),
            "loop_seconds": sum(r["seconds"] for r in history), "trainable_parameters": params,
            "checkpoint_sha256": digest(root / "best.pt"),
            "predictions_file": str(oof_path),
            "predictions_sha256": digest(oof_path),
        }
    save_json(root / "result.json", result)
    return result


def collect_oof(output_root, mode, view, seed, expected_ids):
    """Merge available held-source outputs and enforce exact once-only coverage."""
    root = Path(output_root) / mode / view / f"seed_{seed}"
    parts = []
    for fold in range(3):
        path = root / f"fold_{fold}" / "oof_predictions.csv"
        if not path.exists():
            return None
        parts.extend(_read_predictions(path, with_target=True))
    ids = [r["id"] for r in parts]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError("Source-crossfit OOF predictions do not cover TRAIN exactly once")
    position = {sid: index for index, sid in enumerate(expected_ids)}
    parts.sort(key=lambda row: position[row["id"]])
    merged_path = root / "oof_predictions.csv"
    _write_predictions(merged_path, parts, with_target=True)
    return {
        "path": str(merged_path),
        "count": len(parts),
        "sha256": digest(merged_path),
        "coverage_exactly_once": True,
        "complete": all(json.loads((root / f"fold_{fold}" / "result.json").read_text())["complete"]
                        for fold in range(3)),
    }


def run_suite(config, *, output, seed=None, view=None, fold=None, smoke_groups=None,
              stop_after=None, resume=False, allow_fixture=False):
    """Run selected stage members, updating an explicit suite and manifest."""
    if tuple(config.get("seeds", ())) != FORMAL_SEEDS:
        raise ValueError("The registered source-CV seeds are exactly 13, 42, 2026")
    data = prepare(config, allow_fixture=allow_fixture, smoke_groups=smoke_groups)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_root = Path(config["run_root"])
    seeds = [seed] if seed is not None else list(FORMAL_SEEDS)
    views = [view] if view is not None else list(VIEWS)
    folds = [fold] if fold is not None else [0, 1, 2, "full"]
    if any(item not in FORMAL_SEEDS for item in seeds) or any(item not in VIEWS for item in views):
        raise ValueError("Invalid suite selector")
    results = []
    for current_seed in seeds:
        for current_view in views:
            for current_fold in folds:
                results.append(train_teacher(
                    config, data, view=current_view, seed=current_seed, fold=current_fold,
                    resume=resume, stop_after=stop_after, output_root=output_root,
                ))

    mode = "smoke" if smoke_groups is not None or allow_fixture else "formal"
    oof = {}
    for current_seed in seeds:
        for current_view in views:
            value = collect_oof(output_root, mode, current_view, current_seed, data["ids"])
            if value is not None:
                oof[f"{current_view}/seed_{current_seed}"] = value

    full_matrix_requested = (
        seed is None and view is None and fold is None and smoke_groups is None
        and not allow_fixture and stop_after is None
    )
    expected_keys = {(m, s, f) for m in VIEWS for s in FORMAL_SEEDS for f in (0, 1, 2, "full")}
    completed = {(r["view"], r["seed"], r["fold"]) for r in results if r["complete"]}
    suite_complete = full_matrix_requested and completed == expected_keys and all(
        oof.get(f"{m}/seed_{s}", {}).get("coverage_exactly_once")
        and oof[f"{m}/seed_{s}"].get("complete")
        for m in VIEWS for s in FORMAL_SEEDS
    )
    full_manifest = {
        "fold": data["full_manifest"],
        "fold_manifest": data["fold_manifest"],
    }
    transfer_limit = (
        "The calibration inner holdout reuses this TRAIN OOF table, but point teachers in other folds may have trained on those heldout sources. "
        "This does not establish honest full-pipeline source transfer; nested outer source evaluation must rebuild point/view OOF and calibration before promotion."
    )
    suite_manifest = {
        "stage": "CH-SIMS AnchorCDF native point/view source-CV teachers",
        "dataset": "simsv2",
        "root": str(output_root),
        "author_csv": str(Path(config["author_csv"])),
        "author_csv_sha256": data["author_csv_sha256"],
        "pinned_source_csv_sha256": SOURCE_CSV_SHA256,
        "ids": data["ids"],
        "train_ids": data["ids"],
        "train_feature_hashes": data["train"].hashes,
        "valid_feature_hashes": data["valid_features"].hashes if data["valid_features"] is not None else {},
        "views": {name: list(modalities) for name, modalities in VIEWS.items()},
        "seeds": seeds,
        "selected_views": views,
        "selected_folds": folds,
        "smoke_groups": smoke_groups,
        "fixture_smoke": bool(allow_fixture),
        "fold_splits": full_manifest,
        "has_full": any(r["fold"] == "full" for r in results),
        "full_teacher_scope": "one fixed TRAIN source selection split per seed/view; full TRAIN predictions and optional feature-only VALID predictions",
        "full_teacher_role": "Auxiliary view forecasts for disagreement features; the separately frozen strongest official fusion predictions remain the point anchor.",
        "official_validation_labels_parsed": False,
        "test_evaluated": False,
        "source_transfer_limit": transfer_limit,
    }
    save_json(output.with_suffix(output.suffix + ".manifest.json"), suite_manifest)
    save_json(output.with_suffix(output.suffix + ".config.json"), config)
    report = {
        "root": str(output_root),
        "results": results,
        "oof": oof,
        "has_full": suite_manifest["has_full"],
        "complete": suite_complete,
        "suite_complete": suite_complete,
        "smoke_only": mode == "smoke",
        "test_evaluated": False,
        "official_validation_labels_parsed": False,
        "selected": {"seed": seed, "view": view, "fold": fold, "smoke_groups": smoke_groups},
        "source_transfer_limit": transfer_limit,
    }
    save_json(output, report)
    return report
