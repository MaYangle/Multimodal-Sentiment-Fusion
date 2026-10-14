"""Independent CPU audit for the registered native sequence-gate matrix.

This module deliberately reads TRAIN T/A/V/M and VALID M only. It never parses
TEST scores or opens a TEST feature split. It is an audit utility, not a trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODALITIES = ("text", "audio", "vision")
CHANNELS = ("label_T", "label_A", "label_V")
CASES = (
    "static", "pre_norm", "post_norm", "distilled_static",
    "distilled_pre_norm", "distilled_post_norm", "learned_post_norm",
)
SEEDS = (13, 42, 2026)
TOL = 1e-5
PROMOTION_BOOTSTRAP_DRAWS = 5000
_HASH_MEMO: dict[tuple[str, int, int], str] = {}


def sha256_file(path: str | Path, *, use_memo: bool = True) -> str:
    path = Path(path)
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if use_memo and key in _HASH_MEMO:
        return _HASH_MEMO[key]
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if use_memo:
        _HASH_MEMO[key] = digest
    return digest


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def first_minimum_epoch(rows: list[dict], value_at) -> int | None:
    """Return the first one-based epoch with the minimum finite scalar."""
    candidates = [(index + 1, float(value_at(row))) for index, row in enumerate(rows)]
    candidates = [(epoch, value) for epoch, value in candidates if math.isfinite(value)]
    if not candidates:
        return None
    best = min(value for _, value in candidates)
    return next(epoch for epoch, value in candidates if value == best)


def source_holdout(ids: list[str], seed: int = 20261014, fraction: float = 0.1) -> dict:
    sources = sorted({sid.split("/", 1)[0] for sid in ids})
    if len(sources) < 2 or not 0 < fraction < 1:
        raise ValueError("Source holdout needs multiple sources and a fraction in (0,1)")
    order = list(sources)
    np.random.default_rng(seed).shuffle(order)
    n_held = max(1, math.ceil(len(order) * fraction))
    held = set(order[-n_held:])
    opt = set(order[:-n_held])
    return {
        "all_sources": sources,
        "shuffled_sources": order,
        "opt_sources": sorted(opt),
        "held_sources": sorted(held),
        "opt_ids": [sid for sid in ids if sid.split("/", 1)[0] in opt],
        "held_ids": [sid for sid in ids if sid.split("/", 1)[0] in held],
        "held_count": n_held,
    }


def independent_native_metrics(prediction, target) -> dict:
    """Recompute native CH-SIMS metrics without importing the project metric."""
    # Preserve native float32 boundaries for the author interval convention.
    p = np.asarray(prediction).reshape(-1).astype(np.float32)
    y = np.asarray(target).reshape(-1).astype(np.float32)
    if not len(y) or p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Metric inputs must be finite, nonempty and aligned")
    if np.any(np.abs(y) > 1.0 + 1e-6):
        raise ValueError("Target outside native CH-SIMS range")
    p, y = np.clip(p, -1.0, 1.0), np.clip(y, -1.0, 1.0)
    corr = None if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0 else float(np.corrcoef(p, y)[0, 1])
    out = {"count": int(len(y)), "MAE": float(np.mean(np.abs(p - y))), "Corr": corr}

    def classes(values, boundaries):
        return np.searchsorted(np.asarray(boundaries, dtype=values.dtype), values, side="left")

    for count, boundaries in ((2, [0.0]), (3, [-0.1, 0.1]), (5, [-0.7, -0.1, 0.1, 0.7])):
        py, yy = classes(p, boundaries), classes(y, boundaries)
        out[f"Mult_acc_{count}"] = float(np.mean(py == yy))
        if count == 2:
            total = 0.0
            for label in np.unique(yy):
                support = int(np.count_nonzero(yy == label))
                tp = int(np.count_nonzero((yy == label) & (py == label)))
                denominator = support + int(np.count_nonzero(py == label))
                total += support * (2.0 * tp / denominator if denominator else 0.0)
            out["F1_score"] = float(total / len(yy))
            weak = (y >= -0.4) & (y <= 0.4)
            out["weak_count"] = int(weak.sum())
            out["Mult_acc_2_weak"] = float(np.mean(py[weak] == yy[weak])) if weak.any() else None
    denominator = float(np.sum((y - y.mean()) ** 2))
    numerator = float(np.sum((y - p) ** 2))
    out["R_squre"] = 1.0 - numerator / denominator if denominator else (1.0 if numerator == 0 else 0.0)
    return out


def manual_teacher_kl(scores, targets, present, policy, temperature: float = 0.3) -> tuple[np.ndarray, float]:
    """Compute R* and mean KL(R*||policy) independently in NumPy float64."""
    scores = np.asarray(scores, np.float64)
    targets = np.asarray(targets, np.float64).reshape(-1)
    present = np.asarray(present, bool)
    policy = np.asarray(policy, np.float64)
    if scores.ndim != 2 or scores.shape != present.shape or scores.shape != policy.shape or len(targets) != len(scores):
        raise ValueError("Teacher/policy arrays must have aligned [N,3] shapes")
    logits = -float(temperature) * np.square(scores - targets[:, None])
    teacher = np.zeros_like(logits)
    row_loss = np.zeros(len(scores), np.float64)
    observed = present.any(axis=1)
    for row in np.flatnonzero(observed):
        valid = present[row]
        z = logits[row, valid]
        z = z - np.max(z)
        probs = np.exp(z)
        probs /= probs.sum()
        teacher[row, valid] = probs
        if np.any(policy[row, valid] <= 0):
            raise ValueError("Current policy must assign positive mass to present modalities")
        row_loss[row] = np.sum(probs * (np.log(probs) - np.log(policy[row, valid])))
    return teacher, float(row_loss[observed].mean()) if observed.any() else 0.0


def paired_source_bootstrap(candidate: dict[str, float], reference: dict[str, float],
                            targets: dict[str, float], draws: int = 5000,
                            seed: int = 20261014) -> dict:
    """Whole-source paired bootstrap; candidate/reference are ID keyed."""
    ids = sorted(targets)
    if set(candidate) != set(ids) or set(reference) != set(ids):
        raise ValueError("Bootstrap prediction IDs differ from the unique VALID cohort")
    sources: dict[str, list[str]] = {}
    for sid in ids:
        sources.setdefault(sid.split("/", 1)[0], []).append(sid)
    source_names = sorted(sources)
    rng = np.random.default_rng(seed)
    delta = np.empty(draws, np.float64)
    for draw in range(draws):
        sampled = rng.choice(source_names, size=len(source_names), replace=True)
        rows = [sid for source in sampled for sid in sources[str(source)]]
        delta[draw] = np.mean([
            np.mean([abs(candidate[sid] - targets[sid]) - abs(reference[sid] - targets[sid]) for sid in rows])
        ])
    return {
        "draws": int(draws), "seed": int(seed), "source_count": len(source_names),
        "clip_count": len(ids), "mean_candidate_minus_reference": float(delta.mean()),
        "ci95_percentile": [float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975))],
        "upper_below_zero": bool(np.quantile(delta, 0.975) < 0),
    }


def _read_annotations(path: Path, *, channels: bool, valid_m: bool) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Read TRAIN T/A/V/M and optionally VALID M; never parse any TEST score."""
    records: dict[str, dict] = {}
    ids = {split: [] for split in ("train", "valid", "test")}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "clip_id", "mode", "label", *CHANNELS}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Pinned author CSV is missing required columns")
        for row in reader:
            split = row["mode"]
            if split not in ids:
                raise ValueError("Unexpected author split")
            sid = row["video_id"] + "/" + row["clip_id"]
            if sid in records:
                raise ValueError("Duplicate author sample ID")
            item: dict[str, Any] = {"split": split, "video_id": row["video_id"]}
            if split == "train":
                item["M"] = float(row["label"])
                if channels:
                    item["TAV"] = np.asarray([float(row[name]) for name in CHANNELS], np.float32)
            elif split == "valid" and valid_m:
                item["M"] = float(row["label"])
            records[sid] = item
            ids[split].append(sid)
    return records, ids


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class CacheSplit:
    def __init__(self, root: Path, split: str):
        meta_path = root / "metadata.json"
        meta = _load_json(meta_path)
        default = f"splits/{split}" if (root / "splits").exists() else split
        rel = meta.get("split_paths", {}).get(split, default)
        self.directory = (root / rel).resolve()
        self.directory.relative_to(root.resolve())
        self.ids = _load_json(self.directory / "ids.json")
        self.features = {}
        self.lengths = {}
        for modality in MODALITIES:
            self.features[modality] = np.load(self.directory / f"{modality}.npy", mmap_mode="r")
            self.lengths[modality] = np.load(self.directory / f"{modality}_lengths.npy", mmap_mode="r")
            if self.features[modality].shape[0] != len(self.ids) or self.lengths[modality].shape != (len(self.ids),):
                raise ValueError(f"Invalid cache alignment for {split}/{modality}")

    def batch(self, indices: list[int] | np.ndarray, stats: dict | None = None) -> dict:
        indices = np.asarray(indices, dtype=np.int64)
        result = {}
        length_map = {}
        for modality in MODALITIES:
            x = np.array(self.features[modality][indices], dtype=np.float32, copy=True)
            lengths = np.asarray(self.lengths[modality][indices], dtype=np.int64)
            if stats is not None:
                mean = np.asarray(stats[modality]["mean"], np.float32)
                scale = np.asarray(stats[modality]["scale"], np.float32)
                for row, length in enumerate(lengths.tolist()):
                    if length:
                        x[row, :length] = (x[row, :length] - mean) / scale
                    x[row, length:] = 0
            result[modality] = torch.from_numpy(x)
            length_map[modality] = torch.from_numpy(lengths)
        result["lengths"] = length_map
        return result


def _cache_paths(root: Path, config: dict) -> dict[str, Path]:
    return {key: (root / config[key]).resolve() for key in ("raw_cache", "standardized_cache", "pooled_cache")}


def _verify_cache_identity(root: Path, config: dict, identity: dict, *, hash_arrays: bool,
                           report_check) -> dict:
    paths = _cache_paths(root, config)
    expected_bindings = identity.get("cache_bindings", {})
    binding_names = {
        "raw_metadata_sha256": (paths["raw_cache"] / "metadata.json"),
        "standardized_metadata_sha256": (paths["standardized_cache"] / "metadata.json"),
        "pooled_metadata_sha256": (paths["pooled_cache"] / "metadata.json"),
        "normalization_sha256": (paths["pooled_cache"] / "normalization.json"),
    }
    actual_bindings = {}
    for name, path in binding_names.items():
        actual = sha256_file(path)
        actual_bindings[name] = actual
        report_check(f"cache_binding:{name}", actual == expected_bindings.get(name),
                     {"expected": expected_bindings.get(name), "actual": actual})
    raw_meta = _load_json(paths["raw_cache"] / "metadata.json")
    std_meta = _load_json(paths["standardized_cache"] / "metadata.json")
    pool_meta = _load_json(paths["pooled_cache"] / "metadata.json")
    report_check("cache_raw_has_no_test", "test" not in raw_meta.get("split_counts", {}) and raw_meta.get("split_counts") == {"train": 2722, "valid": 647}, raw_meta.get("split_counts"))
    report_check("cache_standardized_test_not_exported", std_meta.get("test_exported") is False, std_meta.get("test_exported"))
    report_check("cache_standardized_cohort", std_meta.get("counts") == {"train": 2722, "valid": 647}, std_meta.get("counts"))
    report_check("cache_std_bound_to_raw_metadata", std_meta.get("source_metadata_sha256") == actual_bindings["raw_metadata_sha256"], std_meta.get("source_metadata_sha256"))
    report_check("cache_std_bound_to_pooled_metadata", std_meta.get("pooled_standardized_metadata_sha256") == actual_bindings["pooled_metadata_sha256"], std_meta.get("pooled_standardized_metadata_sha256"))
    report_check("cache_std_reuses_pooled_normalization", std_meta.get("normalization_statistics_sha256") == actual_bindings["normalization_sha256"], std_meta.get("normalization_statistics_sha256"))
    norm_fit = pool_meta.get("normalization", {})
    report_check("pooled_scaler_train_only", norm_fit.get("fit_split") == "train" and norm_fit.get("fit_count") == 2722, norm_fit)
    report_check("author_hash_cache_bindings", all(meta.get("author_csv_sha256", "").lower() == config["author_csv_sha256"].lower() for meta in (std_meta, pool_meta)), {
        "std": std_meta.get("author_csv_sha256"), "pooled": pool_meta.get("author_csv_sha256")})
    used = identity.get("used_array_sha256", {})
    actual_arrays = {}
    for relative, expected in used.items():
        if relative == "metadata.json":
            path = (paths["raw_cache"] if identity.get("stage") == "select" else paths["standardized_cache"]) / relative
        else:
            cache = paths["raw_cache"] if identity.get("stage") == "select" else paths["standardized_cache"]
            path = cache / relative
        if not path.is_file():
            report_check(f"array_hash:{relative}", False, "missing")
            continue
        if not hash_arrays and path.suffix == ".npy":
            actual_arrays[relative] = "deferred_live_io"
            continue
        actual = sha256_file(path)
        actual_arrays[relative] = actual
        report_check(f"array_hash:{relative}", actual == expected, {"expected": expected, "actual": actual})
    return {"paths": {key: str(value) for key, value in paths.items()}, "bindings": actual_bindings,
            "used_array_sha256": actual_arrays, "array_hashes_fully_recomputed": hash_arrays}


def _verify_identity(root: Path, identity: dict, config: dict, author_hash: str, report_check) -> None:
    stored = identity.get("identity_sha256")
    actual = sha256_json({key: value for key, value in identity.items() if key != "identity_sha256"})
    report_check("identity_sha256", actual == stored, {"stored": stored, "recomputed": actual})
    report_check("identity_config_matches", identity.get("configuration") == config, identity.get("configuration", {}).get("schema"))
    report_check("identity_author_hash", identity.get("author_csv_sha256") == author_hash, identity.get("author_csv_sha256"))
    version_bound = any(key in identity for key in ("python", "python_version", "torch", "torch_version"))
    report_check("training_runtime_version_binding_reviewed", True,
                 {"training_versions_recorded": version_bound,
                  "detail": "training identity records runtime flags but no Python/PyTorch version" if not version_bound else "versions are recorded"})
    for relative, expected in identity.get("code_and_plan_sha256", {}).items():
        path = root / relative.replace("\\", "/")
        if not path.is_file():
            report_check(f"source_hash:{relative}", expected is None, "missing source file")
            continue
        actual_hash = sha256_file(path)
        report_check(f"source_hash:{relative}", actual_hash == expected, {"expected": expected, "actual": actual_hash})


def _verify_optimizer_state(checkpoint: dict, model: torch.nn.Module, expected_steps: dict[str, int], report_check, prefix: str) -> dict:
    opt = checkpoint.get("optimizer")
    if not isinstance(opt, dict) or not opt.get("param_groups"):
        report_check(f"{prefix}:optimizer_state_present", False, type(opt).__name__)
        return {"present": False}
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    ids = opt["param_groups"][0].get("params", [])
    states = opt.get("state", {})
    report_check(f"{prefix}:optimizer_parameter_count", len(ids) == len(params), {"optimizer": len(ids), "model": len(params)})
    step_counts = {}
    mismatches = []
    for (name, _), pid in zip(params, ids):
        state = states.get(pid, {})
        step_tensor = state.get("step")
        step = int(step_tensor.item()) if isinstance(step_tensor, torch.Tensor) else (int(step_tensor) if step_tensor is not None else 0)
        step_counts[name] = step
        expected = expected_steps.get(name, expected_steps.get("*"))
        if expected is not None and step != expected:
            mismatches.append({"parameter": name, "expected": expected, "actual": step})
    report_check(f"{prefix}:optimizer_update_counts", not mismatches, mismatches[:12])
    rng = checkpoint.get("rng", {})
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "loader_generator"}
    report_check(f"{prefix}:rng_state_present", required_rng.issubset(rng), sorted(set(rng) ^ required_rng))
    if prefix.startswith("select") or prefix.startswith("fusion"):
        report_check(f"{prefix}:validation_rng_state_present", rng.get("validation_loader_generator") is not None,
                     rng.get("validation_loader_generator") is not None)
    return {"present": True, "parameter_steps": step_counts, "state_count": len(states)}


def _close_metrics(actual: dict, expected: dict, tol: float = TOL) -> tuple[bool, dict]:
    diffs = {}
    okay = True
    for key, value in expected.items():
        found = actual.get(key)
        if value is None or found is None:
            same = value is None and found is None
        elif isinstance(value, (int, np.integer)):
            same = int(found) == int(value)
        else:
            same = abs(float(found) - float(value)) <= tol
            diffs[key] = float(found) - float(value)
        okay &= same
    return okay, diffs


def _state_subhash(state: dict[str, torch.Tensor], predicate) -> str:
    return state_fingerprint({name: value for name, value in state.items() if predicate(name)})


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _source_metric_audit(root: Path, config: dict, identity: dict, checkpoint: dict,
                          annotations: dict, report_check) -> dict:
    cache = (root / config["raw_cache"]).resolve()
    split = CacheSplit(cache, "train")
    derived = source_holdout(split.ids, 20261014, 0.1)
    recorded = identity.get("source_split", {})
    for key in ("all_sources", "shuffled_sources", "opt_sources", "held_sources", "held_count"):
        report_check(f"source_split:{key}", recorded.get(key) == derived[key], {"recorded": recorded.get(key), "derived": derived[key]})
    opt_ids, held_ids = derived["opt_ids"], derived["held_ids"]
    report_check("source_split_opt_held_disjoint", not (set(derived["opt_sources"]) & set(derived["held_sources"])), None)
    report_check("source_split_fit_hold_ids", identity.get("opt_ids") == opt_ids and identity.get("held_ids") == held_ids,
                 {"opt_count": len(opt_ids), "held_count": len(held_ids)})
    indices = np.asarray([split.ids.index(sid) for sid in opt_ids], np.int64)
    held_indices = np.asarray([split.ids.index(sid) for sid in held_ids], np.int64)
    stats = {}
    stat_diffs = {}
    for modality in MODALITIES:
        x = split.features[modality]
        length = np.asarray(split.lengths[modality])
        if modality == "text":
            pooled = np.array(x[:, 0], np.float32, copy=True)
            pooled[length == 0] = 0
        else:
            pooled = np.stack([
                np.asarray(x[i, :int(n)], np.float32).mean(0) if int(n) else np.zeros(x.shape[-1], np.float32)
                for i, n in enumerate(length)
            ])
        fit = pooled[indices].astype(np.float64)
        mean = fit.mean(0)
        scale = fit.std(0)
        scale[scale < 1e-6] = 1.0
        stats[modality] = {"mean": mean.astype(np.float32).tolist(), "scale": scale.astype(np.float32).tolist(), "fit_indices": indices.tolist()}
        claimed = identity.get("normalization", {}).get(modality, {})
        mean_delta = float(np.max(np.abs(np.asarray(claimed.get("mean", []), np.float64) - mean))) if claimed.get("mean") else math.inf
        scale_delta = float(np.max(np.abs(np.asarray(claimed.get("scale", []), np.float64) - scale))) if claimed.get("scale") else math.inf
        same_idx = claimed.get("fit_indices") == indices.tolist()
        report_check(f"source_scaler:{modality}", max(mean_delta, scale_delta) <= 1e-6 and same_idx,
                     {"max_mean_delta": mean_delta, "max_scale_delta": scale_delta, "fit_index_equal": same_idx,
                      "fit_rows": len(indices), "held_rows": len(held_indices)})
        stat_diffs[modality] = {"mean": mean_delta, "scale": scale_delta}

    history = checkpoint.get("history", [])
    dims = {m: int(split.features[m].shape[-1]) for m in MODALITIES}
    from jevaffect.sequence_gates import UnimodalSequenceHeads
    budget_model = UnimodalSequenceHeads(dims, dim=int(config["model"]["dim"]))
    opt_batches = math.ceil(len(opt_ids) / int(config["stage1"]["batch_size"]))
    expected_steps = {name: len(history) * opt_batches for name, _ in budget_model.named_parameters()}
    optimizer_audit = _verify_optimizer_state(checkpoint, budget_model, expected_steps, report_check, f"select:{checkpoint.get('seed')}")
    selected = {}
    for index, modality in enumerate(MODALITIES):
        epoch = first_minimum_epoch(history, lambda row, m=modality: row["held_source"][m]["MAE"])
        selected[modality] = epoch
        recorded_epoch = checkpoint.get("selected_epochs", {}).get(modality)
        report_check(f"stage1_first_minimum:{modality}", epoch == recorded_epoch,
                     {"derived_first_minimum_epoch": epoch, "checkpoint_epoch": recorded_epoch})
    report_check("stage1_fixed_50_epochs", len(history) <= 50 and (not checkpoint.get("complete") or len(history) == 50), len(history))

    if not all(checkpoint.get("best_branches", {}).get(m, {}).get("state") for m in MODALITIES):
        return {"status": "incomplete_channel_states", "selected_epochs": selected, "scaler_max_deltas": stat_diffs}
    model = UnimodalSequenceHeads(dims, dim=int(config["model"]["dim"]))
    state = dict(checkpoint["initial_state"])
    for modality in MODALITIES:
        for name, tensor in checkpoint["best_branches"][modality]["state"].items():
            key = name if name.startswith(f"branches.{modality}.") else f"branches.{modality}.{name}"
            state[key] = tensor
    model.load_state_dict(state, strict=True)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for offset in range(0, len(held_indices), 16):
            idx = held_indices[offset:offset + 16]
            features = split.batch(idx, stats)
            outputs.append(model(features).cpu().numpy())
    predictions = np.concatenate(outputs)
    held_truth = np.stack([annotations["records"][sid]["TAV"] for sid in held_ids])
    present = np.stack([[int(split.lengths[m][i]) > 0 for m in MODALITIES] for i in held_indices])
    metrics = {}
    for j, modality in enumerate(MODALITIES):
        keep = present[:, j]
        got = independent_native_metrics(predictions[keep, j], held_truth[keep, j])
        claimed = checkpoint["best_branches"][modality]["metrics"]
        okay, diff = _close_metrics(got, claimed)
        report_check(f"stage1_cpu_selected_metrics:{modality}", okay, {"diffs": diff, "cpu_metrics": got, "saved_metrics": claimed})
        metrics[modality] = got
    return {"status": "audited", "selected_epochs": selected, "scaler_max_deltas": stat_diffs,
            "source_counts": {"all": len(derived["all_sources"]), "opt": len(derived["opt_sources"]), "held": len(derived["held_sources"])},
            "row_counts": {"opt": len(opt_ids), "held": len(held_ids)}, "held_metrics_cpu": metrics,
            "optimizer": optimizer_audit}


def _audit_refit(root: Path, config: dict, seed: int, selection: dict, refit: dict,
                 cache_split: CacheSplit, report_check) -> dict:
    selected = {m: int(selection.get("selected_epochs", {}).get(m, -1)) for m in MODALITIES}
    max_epochs = max(selected.values())
    history = refit.get("history", [])
    expected_active = [sorted(m for m in MODALITIES if epoch <= selected[m]) for epoch in range(1, len(history) + 1)]
    actual_active = [row.get("active_modalities") for row in history]
    report_check(f"refit_selected_counts:{seed}", refit.get("selected_epochs") == selected, selected)
    report_check(f"refit_active_schedule:{seed}", actual_active == expected_active, {"actual": actual_active, "expected": expected_active})
    report_check(f"refit_fit_ids:{seed}", refit.get("fit_ids") == cache_split.ids and len(cache_split.ids) == 2722,
                 {"count": len(refit.get("fit_ids", [])), "cache_count": len(cache_split.ids)})
    report_check(f"refit_selected_checkpoint_binding:{seed}", refit.get("selection_checkpoint_sha256") == selection.get("_file_sha256"),
                 {"checkpoint": refit.get("selection_checkpoint_sha256"), "selection": selection.get("_file_sha256")})
    report_check(f"refit_expected_epoch_budget:{seed}", len(history) <= max_epochs and (not refit.get("complete") or len(history) == max_epochs),
                 {"history_epochs": len(history), "expected": max_epochs})
    batch_count = math.ceil(len(cache_split.ids) / int(config["refit"]["batch_size"]))
    model_module = __import__("jevaffect.sequence_gates", fromlist=["UnimodalSequenceHeads"])
    dims = {m: int(cache_split.features[m].shape[-1]) for m in MODALITIES}
    model = model_module.UnimodalSequenceHeads(dims, dim=int(config["model"]["dim"]))
    model.load_state_dict(refit["last_state"], strict=True)
    expected_steps = {}
    for name, _ in model.named_parameters():
        modality = name.split(".", 2)[1]
        expected_steps[name] = min(len(history), selected[modality]) * batch_count
    optimizer_audit = _verify_optimizer_state(refit, model, expected_steps, report_check, f"refit:{seed}")
    return {"complete": bool(refit.get("complete")), "selected_epochs": selected,
            "history_epochs": len(history), "updates_per_epoch": batch_count,
            "optimizer": optimizer_audit}


def _predict_fusion(model, split: CacheSplit, batch_size: int = 16) -> tuple[list[float], list[dict]]:
    model.eval()
    predictions, route_audit = [], []
    with torch.inference_mode():
        for offset in range(0, len(split.ids), batch_size):
            idx = np.arange(offset, min(offset + batch_size, len(split.ids)), dtype=np.int64)
            features = split.batch(idx)
            out = model(features)
            p = out["prediction"].detach().cpu().numpy().astype(np.float64)
            r = out["ratios"].detach().cpu().numpy()
            pr = out["policy_ratios"].detach().cpu().numpy()
            present = np.stack([split.lengths[m][idx] > 0 for m in MODALITIES], axis=1)
            if not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite(pr).all():
                raise ValueError("CPU model replay returned nonfinite scores or ratios")
            if np.any(np.abs(r[~present]) > 1e-7) or np.any(np.abs(pr[~present]) > 1e-7):
                raise ValueError("Missing modalities have nonzero route ratios")
            if model.case in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm") and not np.allclose(r, pr, atol=1e-7, rtol=0):
                raise ValueError("Learned-route model did not use its target-free policy at evaluation")
            if model.case not in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm") and not np.allclose(r, present.astype(r.dtype), atol=1e-7, rtol=0):
                raise ValueError("Static/teacher reference must use R=1 for each present modality at evaluation")
            route_audit.append({"ratios": r, "policy": pr, "present": present})
            predictions.extend(p.tolist())
    return predictions, route_audit


def _manual_train_kl(model, split: CacheSplit, annotations: dict, case: str, report_check, label: str) -> dict:
    indices = np.arange(min(8, len(split.ids)), dtype=np.int64)
    features = split.batch(indices)
    targets = torch.tensor([annotations["records"][split.ids[i]]["M"] for i in indices], dtype=torch.float32)
    present = np.stack([split.lengths[m][indices] > 0 for m in MODALITIES], axis=1)
    model.train()
    # A fixed seed makes this read-only training-mode diagnostic reproducible.
    torch.manual_seed(89137)
    with torch.no_grad():
        out = model(features, target=targets)
    scores = out["unimodal_scores"].detach().cpu().numpy()
    policy = out["policy_ratios"].detach().cpu().numpy()
    teacher, manual = manual_teacher_kl(scores, targets.numpy(), present, policy)
    actual = float(out["distillation"].detach().cpu())
    distilled = case.startswith("distilled_")
    report_check(f"{label}:manual_training_KL", abs(manual - actual) <= TOL if distilled else actual == 0.0,
                 {"manual": manual, "model": actual, "policy_routed": case in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm"),
                  "teacher_mass_max_error": float(np.max(np.abs(teacher.sum(1) - present.any(1))))})
    if case in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm"):
        route_ok = torch.allclose(out["ratios"], out["policy_ratios"], atol=1e-7, rtol=0)
    elif case in ("static", "distilled_static"):
        denominator = torch.stack([features["lengths"][m] > 0 for m in MODALITIES], 1).sum(1).clamp_min(1)
        expected = torch.stack([features["lengths"][m] > 0 for m in MODALITIES], 1).float() / denominator[:, None]
        route_ok = torch.allclose(out["ratios"], expected, atol=1e-7, rtol=0)
    else:
        expected = torch.from_numpy(teacher).to(out["ratios"].dtype)
        route_ok = torch.allclose(out["ratios"], expected, atol=1e-7, rtol=0)
    report_check(f"{label}:training_route", route_ok, case)
    model.eval()
    return {"manual_kl": manual, "model_kl": actual, "teacher_mass_error": float(np.max(np.abs(teacher.sum(1) - present.any(1))))}


def _audit_fusion(root: Path, config: dict, seed: int, case: str, identity_path: Path,
                  checkpoint_path: Path, result_path: Path, valid_cache: CacheSplit,
                  train_cache: CacheSplit, annotations: dict, report_check, *, replay: bool,
                  diagnostics: bool) -> dict:
    if not identity_path.is_file() or not checkpoint_path.is_file():
        return {"status": "missing"}
    identity = _load_json(identity_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    result = _load_json(result_path) if result_path.is_file() else {}
    _verify_identity(root, identity, config, sha256_file(root / config["author_csv"]), report_check)
    report_check(f"fusion_checkpoint_sha:{seed}:{case}", result.get("checkpoint_sha256") == sha256_file(checkpoint_path) if result else True,
                 {"result_sha": result.get("checkpoint_sha256"), "actual_sha": sha256_file(checkpoint_path)})
    report_check(f"fusion_selected_identity:{seed}:{case}", checkpoint.get("identity", {}).get("identity_sha256") == identity.get("identity_sha256"), None)
    formal = not identity.get("smoke_only", True)
    epochs = len(checkpoint.get("history", []))
    complete = bool(checkpoint.get("complete")) and bool(result.get("complete")) and formal
    report_check(f"fusion_full_budget:{seed}:{case}", epochs <= 50 and (not complete or epochs == 50), epochs)
    history = checkpoint.get("history", [])
    if history:
        best_epoch = first_minimum_epoch(history, lambda row: row["valid"]["MAE"])
        report_check(f"fusion_first_valid_minimum:{seed}:{case}", best_epoch == checkpoint.get("best_epoch"),
                     {"first_min_epoch": best_epoch, "checkpoint_epoch": checkpoint.get("best_epoch")})
        per_epoch = [row.get("batches", 0) for row in history]
        report_check(f"fusion_epoch_batches:{seed}:{case}", all(v == 86 for v in per_epoch), per_epoch[:3] + per_epoch[-3:])
        report_check(f"fusion_total_updates:{seed}:{case}", sum(per_epoch) == epochs * 86, {"epochs": epochs, "updates": sum(per_epoch)})
    model_module = __import__("jevaffect.sequence_gates", fromlist=["SequenceGateRegressor"])
    std_metadata = _load_json((root / config["standardized_cache"]) / "metadata.json")
    dims = {m: int(std_metadata["dimensions"][m]) for m in MODALITIES}
    model = model_module.SequenceGateRegressor(dims, case, **config["model"])
    refit_path = root / config["run_root"] / "refit" / f"seed_{seed}" / "checkpoint.pt"
    refit = _load_checkpoint(refit_path) if refit_path.is_file() else {}
    selection_path = root / config["run_root"] / "select" / f"seed_{seed}" / "checkpoint.pt"
    refit_sha = sha256_file(refit_path) if refit_path.is_file() else None
    selection_sha = sha256_file(selection_path) if selection_path.is_file() else None
    report_check(f"fusion_refit_checkpoint_binding:{seed}:{case}", identity.get("upstream_checkpoint_sha256") == refit_sha,
                 {"identity_upstream": identity.get("upstream_checkpoint_sha256"), "refit_checkpoint": refit_sha})
    report_check(f"fusion_selection_checkpoint_binding:{seed}:{case}", identity.get("selection_checkpoint_sha256") == selection_sha,
                 {"identity_selection": identity.get("selection_checkpoint_sha256"), "selection_checkpoint": selection_sha})
    expected_policy_status = {
        "static": "present_but_unused", "pre_norm": "present_but_unused", "post_norm": "present_but_unused",
        "distilled_static": "trained_by_distillation_but_not_used_for_routing",
        "distilled_pre_norm": "trained_by_distillation_and_used_for_label_free_routing",
        "distilled_post_norm": "trained_by_distillation_and_used_for_label_free_routing",
        "learned_post_norm": "trained_by_prediction_loss_and_used_for_label_free_routing_without_distillation",
    }[case]
    report_check(f"fusion_policy_usage_contract:{seed}:{case}", identity.get("policy_head", {}).get("status") == expected_policy_status,
                 identity.get("policy_head"))
    if refit.get("last_state"):
        model.load_unimodal_state(refit["last_state"])
    model.freeze_knowledge()
    expected_total = int(config["expected_parameters"]["fusion_total"])
    expected_trainable = int(config["expected_parameters"]["fusion_trainable"])
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report_check(f"fusion_parameter_contract:{seed}:{case}", total == expected_total and trainable == expected_trainable,
                 {"total": total, "trainable": trainable, "expected_total": expected_total, "expected_trainable": expected_trainable})
    policy_params = [(name, p) for name, p in model.named_parameters() if "policy" in name.lower()]
    policy_count = sum(p.numel() for _, p in policy_params)
    report_check(f"fusion_policy_count:{seed}:{case}", policy_count == 25027 and all(p.requires_grad for _, p in policy_params), policy_count)
    # The trainer constructs this module on CPU before moving it to CUDA. Rebuild
    # the same seeded CPU initialization to check the recorded shared state hash.
    torch.manual_seed(seed)
    initial_model = model_module.SequenceGateRegressor(dims, case, **config["model"])
    if refit.get("last_state"):
        initial_model.load_unimodal_state(refit["last_state"])
    independent_initial_hash = state_fingerprint(initial_model.state_dict())
    independent_policy_hash = state_fingerprint({name: tensor for name, tensor in initial_model.state_dict().items() if "policy" in name.lower()})
    report_check(f"fusion_independent_initialization_hash:{seed}:{case}",
                 independent_initial_hash == identity.get("initial_fusion_state_sha256"),
                 {"independent": independent_initial_hash, "recorded": identity.get("initial_fusion_state_sha256")})
    if "best_state" in checkpoint:
        model.load_state_dict(checkpoint["best_state"], strict=True)
    if refit.get("last_state") and "best_state" in checkpoint:
        best = checkpoint["best_state"]
        refit_subset = {name: tensor for name, tensor in refit["last_state"].items()
                        if ".adapter." in name or ".decoder." in name}
        fusion_subset = {name: tensor for name, tensor in best.items()
                         if ".adapter." in name or ".decoder." in name}
        frozen_ok = bool(refit_subset) and refit_subset.keys() == fusion_subset.keys() and all(torch.equal(refit_subset[k], fusion_subset[k]) for k in refit_subset)
        report_check(f"fusion_frozen_adapter_decoder_matches_refit:{seed}:{case}", frozen_ok, len(refit_subset))
    policy_state = {name: tensor for name, tensor in checkpoint.get("best_state", {}).items() if "policy" in name.lower()}
    best_policy_hash = state_fingerprint(policy_state) if policy_state else None
    init_policy_hash = identity.get("policy_head", {}).get("initial_state_sha256")
    if not init_policy_hash:
        init_policy_hash = checkpoint.get("initial_policy_state_sha256")
    report_check(f"fusion_independent_initial_policy_hash:{seed}:{case}",
                 independent_policy_hash == init_policy_hash,
                 {"independent": independent_policy_hash, "recorded": init_policy_hash})
    active_expected = case in ("distilled_static", "distilled_pre_norm", "distilled_post_norm", "learned_post_norm")
    unchanged = best_policy_hash == init_policy_hash if init_policy_hash else None
    if case in ("static", "pre_norm", "post_norm"):
        report_check(f"fusion_dormant_policy_unchanged:{seed}:{case}", unchanged is True,
                     {"initial": init_policy_hash, "best": best_policy_hash})
    if case == "distilled_static":
        report_check(f"fusion_distilled_static_policy_active_but_not_routed:{seed}", active_expected and not unchanged,
                     {"active": active_expected, "unchanged": unchanged, "status": identity.get("policy_head")})
    actual_policy_count = len(policy_params)
    del actual_policy_count
    expected_steps = {
        name: (0 if "policy" in name.lower() and case in ("static", "pre_norm", "post_norm") else epochs * 86)
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    # In the three nondistilled reference arms the policy is in the optimizer
    # parameter list but receives no gradient, so AdamW must skip it entirely.
    optimizer_audit = _verify_optimizer_state(checkpoint, model, expected_steps, report_check, f"fusion:{seed}:{case}")
    if case in ("static", "pre_norm", "post_norm"):
        policy_steps = [step for name, step in optimizer_audit.get("parameter_steps", {}).items() if "policy" in name.lower()]
        report_check(f"fusion_dormant_policy_no_adamw_step:{seed}:{case}", not policy_steps or all(step == 0 for step in policy_steps), policy_steps)

    cpu_replay = None
    if replay and complete:
        if valid_cache is None or train_cache is None:
            raise ValueError("Complete CPU replay needs the official TRAIN and VALID cache splits")
        if "best_state" not in checkpoint:
            report_check(f"fusion_best_state_present:{seed}:{case}", False, None)
        else:
            model.eval()
            # Explicit math-only SDPA and disabled MHA fastpath; the model stays
            # on CPU and does not create a CUDA context.
            torch.backends.mha.set_fastpath_enabled(False)
            contract_features = valid_cache.batch(np.arange(min(1, len(valid_cache.ids))))
            try:
                model(contract_features, target=torch.zeros(len(contract_features["text"])))
                target_rejected = False
            except ValueError as exc:
                target_rejected = "must not receive a target" in str(exc)
            report_check(f"fusion_eval_rejects_target:{seed}:{case}", target_rejected,
                         "evaluation mode rejects a supplied target before model computation")
            pred, routes = _predict_fusion(model, valid_cache)
            saved = _load_json(root / config["run_root"] / "fusion" / f"seed_{seed}" / case / "valid_predictions.json")
            cp_saved = checkpoint.get("best_valid_predictions") or []
            ids_expected = valid_cache.ids
            saved_by_id = {row["id"]: row for row in saved}
            checkpoint_by_id = {row["id"]: row for row in cp_saved}
            pred_by_id = dict(zip(ids_expected, pred))
            targets = {sid: annotations["records"][sid]["M"] for sid in ids_expected}
            report_check(f"fusion_valid_cohort_exact:{seed}:{case}", len(saved_by_id) == 647 and list(saved_by_id) == ids_expected and set(checkpoint_by_id) == set(ids_expected),
                         {"saved": len(saved_by_id), "checkpoint": len(checkpoint_by_id), "cache": len(ids_expected)})
            max_delta_json = max((abs(float(saved_by_id[sid]["prediction"]) - pred_by_id[sid]) for sid in ids_expected), default=0.0)
            max_delta_cp = max((abs(float(checkpoint_by_id[sid]["prediction"]) - pred_by_id[sid]) for sid in ids_expected), default=0.0)
            replay_ok = max(max_delta_json, max_delta_cp) <= TOL
            report_check(f"fusion_cpu_prediction_replay:{seed}:{case}", replay_ok,
                         {"max_abs_delta_json": max_delta_json, "max_abs_delta_checkpoint": max_delta_cp,
                          "tolerance": TOL})
            current_metrics = independent_native_metrics(pred, [targets[sid] for sid in ids_expected])
            saved_metrics = checkpoint.get("best_valid_metrics", {})
            metric_ok, metric_diffs = _close_metrics(current_metrics, saved_metrics)
            report_check(f"fusion_cpu_metrics_recomputed:{seed}:{case}", metric_ok,
                         {"metrics": current_metrics, "saved": saved_metrics, "diffs": metric_diffs})
            route_summary = {}
            for row in routes:
                for label, key in (("ratios", "ratios"), ("policy", "policy")):
                    values = row[key]
                    route_summary.setdefault(label, []).append(values)
            route_ok = case not in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm") or all(np.allclose(r, p, atol=1e-7, rtol=0) for r, p in zip(route_summary.get("ratios", []), route_summary.get("policy", [])))
            report_check(f"fusion_target_free_policy_route:{seed}:{case}", route_ok, {"policy_routed_case": case in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm")})
            cpu_replay = {"prediction_count": len(pred), "max_abs_training_device_to_cpu_prediction_delta": max_delta_json,
                          "independent_native_metrics": current_metrics, "policy_routed": case in ("distilled_pre_norm", "distilled_post_norm", "learned_post_norm")}
            if diagnostics and case == "distilled_post_norm":
                # A deliberately target-free branch intervention and source_none
                # sensitivity probe; these diagnostics never select a checkpoint.
                model.eval()
                first = valid_cache.batch(np.arange(min(32, len(valid_cache.ids))))
                with torch.inference_mode():
                    base = model(first)
                    present = torch.stack([first["lengths"][m] > 0 for m in MODALITIES], 1)
                    intervention = []
                    for column in range(3):
                        ratios = torch.zeros_like(base["ratios"])
                        ratios[:, column] = present[:, column].float()
                        altered = model.probe(first, ratios)
                        intervention.append({"modality": MODALITIES[column], "mean_abs_change": float((altered["prediction"] - base["prediction"]).abs().mean())})
                    source_none = model.probe_source_none(first)
                diag_metrics = independent_native_metrics(source_none["prediction"].cpu().numpy(),
                                                          [targets[sid] for sid in valid_cache.ids[:len(source_none["prediction"])]] )
                cpu_replay["target_free_diagnostics"] = {"forced_route_changes": intervention, "source_none_first_batch_metrics": diag_metrics}
                report_check(f"fusion_target_free_diagnostics:{seed}", all(math.isfinite(item["mean_abs_change"]) for item in intervention), cpu_replay["target_free_diagnostics"])
    kl_audit = None
    if complete:
        model.load_state_dict(checkpoint["best_state"], strict=True)
        kl_audit = _manual_train_kl(model, train_cache, annotations, case, report_check, f"fusion:{seed}:{case}")
    if complete and case == "learned_post_norm":
        model.load_state_dict(checkpoint["best_state"], strict=True)
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        attention_dropouts = [(module, module.dropout) for module in model.modules() if isinstance(module, torch.nn.MultiheadAttention)]
        for module, _old_dropout in attention_dropouts:
            module.dropout = 0.0
        features = train_cache.batch(np.arange(min(8, len(train_cache.ids))))
        targets = torch.tensor([annotations["records"][train_cache.ids[i]]["M"] for i in range(min(8, len(train_cache.ids)))], dtype=torch.float32)
        with torch.no_grad():
            without = model(features)
            with_m = model(features, target=targets)
        target_independent = torch.equal(without["ratios"], with_m["ratios"]) and torch.equal(without["prediction"], with_m["prediction"])
        for module, old_dropout in attention_dropouts:
            module.dropout = old_dropout
        report_check(f"fusion_learned_route_target_independent_in_training:{seed}", target_independent, None)
    return {"status": "complete" if complete else ("in_progress" if checkpoint else "missing"),
            "complete": complete, "epochs": epochs, "best_epoch": checkpoint.get("best_epoch"),
            "identity_sha256": identity.get("identity_sha256"), "initial_fusion_state_sha256": identity.get("initial_fusion_state_sha256"),
            "initial_policy_state_sha256": init_policy_hash, "best_policy_state_sha256": best_policy_hash,
            "policy_state_unchanged": unchanged, "optimizer": optimizer_audit,
            "cpu_replay": cpu_replay, "manual_kl": kl_audit}


def _load_prediction_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows = _load_json(path)
    out = {row["id"]: row for row in rows}
    if len(out) != len(rows):
        raise ValueError(f"Duplicate IDs in prediction file {path}")
    return out


def _audit_comparisons(root: Path, config: dict, report_check) -> dict:
    valid_records, valid_ids_by_split = _read_annotations(root / config["author_csv"], channels=False, valid_m=True)
    valid_ids = sorted(valid_ids_by_split["valid"])
    truth = {sid: float(valid_records[sid]["M"]) for sid in valid_ids}
    report_check("promotion_valid_truth_source", len(truth) == 647, {"count": len(truth), "source": "author VALID label M only"})
    candidate_rows = {}
    references: dict[str, dict[int, dict]] = {
        "distilled_static": {}, "distilled_pre_norm": {}, "learned_post_norm": {},
        "strongest_teacher_pre_post": {}, "conditioned_early_fusion": {}, "conditioned_ALMT": {},
    }
    teacher_choices: dict[int, str | None] = {}
    for seed in SEEDS:
        base = root / config["run_root"] / "fusion" / f"seed_{seed}"
        paths = {case: base / case / "valid_predictions.json" for case in CASES}
        loaded = {case: _load_prediction_rows(path) for case, path in paths.items()}
        candidate_rows[seed] = loaded.get("distilled_post_norm", {})
        for case in ("distilled_static", "distilled_pre_norm", "learned_post_norm"):
            references[case][seed] = loaded.get(case, {})
        teacher_options = [(case, loaded.get(case, {})) for case in ("pre_norm", "post_norm")]
        complete_options = [(case, rows) for case, rows in teacher_options if len(rows) == 647]
        if complete_options:
            # Choose the per-seed strongest teacher arm using the same clipped
            # native CH-SIMS MAE that selected its checkpoint, not raw score MAE.
            def native_valid_mae(rows: dict[str, dict]) -> float:
                predictions = [float(rows[sid]["prediction"]) for sid in valid_ids]
                targets = [truth[sid] for sid in valid_ids]
                return float(independent_native_metrics(predictions, targets)["MAE"])

            selected_case, selected_rows = min(complete_options, key=lambda pair: native_valid_mae(pair[1]))
            references["strongest_teacher_pre_post"][seed] = selected_rows
        else:
            selected_case = None
        teacher_choices[seed] = selected_case
        references["strongest_teacher_pre_post"].setdefault(seed, {})
        for label, path in (
            ("conditioned_early_fusion", root / "runs/native_simsv2_standardized_v1/early_fusion" / f"seed_{seed}" / "valid_predictions.json"),
            ("conditioned_ALMT", root / "runs/native_simsv2_sequence_standardized_v1/almt" / f"seed_{seed}" / "valid_predictions.json"),
        ):
            references[label][seed] = _load_prediction_rows(path)
        report_check(f"comparison_predictions_present:{seed}", len(candidate_rows[seed]) in (0, 647),
                     {name: len(by_seed.get(seed, {})) for name, by_seed in references.items()} | {"candidate": len(candidate_rows[seed]), "strongest_teacher_choice": selected_case})
    outcome = {"status": "partial", "candidate": "distilled_post_norm", "comparisons": {}}
    full_candidate = all(len(candidate_rows[s]) == 647 for s in SEEDS)
    full_refs = all(len(rows[s]) == 647 for rows in references.values() for s in SEEDS)
    if not (full_candidate and full_refs):
        outcome["available"] = {"candidate": {s: len(candidate_rows[s]) for s in SEEDS},
                                 "references": {name: {s: len(rows[s]) for s in SEEDS} for name, rows in references.items()}}
        return outcome
    ids = valid_ids
    for seed in SEEDS:
        report_check(f"comparison_same_valid_ids:{seed}", set(candidate_rows[seed]) == set(ids), len(candidate_rows[seed]))
        all_reference_targets_aligned = all(set(rows[seed]) == set(ids) for rows in references.values())
        report_check(f"comparison_same_reference_valid_ids:{seed}", all_reference_targets_aligned,
                     {name: len(rows[seed]) for name, rows in references.items()})
    outcome["status"] = "complete"
    outcome["references"] = {}
    outcome["strongest_teacher_choice_by_seed"] = teacher_choices
    outcome["strongest_teacher_selection_conditioning"] = "chooses the lower-MAE pre_norm/post_norm checkpoint on the same VALID cohort per seed; inferential interval is conditional on that comparison choice"
    for name, by_seed in references.items():
        per_seed = {}
        stacked_c = {}
        stacked_r = {}
        for seed in SEEDS:
            crows, rrows = candidate_rows[seed], by_seed[seed]
            if set(crows) != set(ids) or set(rrows) != set(ids):
                raise ValueError(f"Mismatched VALID IDs for {name}, seed {seed}")
            c = {sid: float(crows[sid]["prediction"]) for sid in ids}
            r = {sid: float(rrows[sid]["prediction"]) for sid in ids}
            cm = independent_native_metrics(list(c.values()), [truth[sid] for sid in ids])
            rm = independent_native_metrics(list(r.values()), [truth[sid] for sid in ids])
            per_seed[seed] = {"candidate_metrics": cm, "reference_metrics": rm,
                              "candidate_lower": cm["MAE"] < rm["MAE"]}
            stacked_c[seed] = c
            stacked_r[seed] = r
        # Every paired bootstrap draw resamples whole sources on the same 647 IDs
        # and averages the three seed-level MAE differences.
        rng = np.random.default_rng(20261014)
        source_ids: dict[str, list[str]] = {}
        for sid in ids:
            source_ids.setdefault(sid.split("/", 1)[0], []).append(sid)
        sources = sorted(source_ids)
        draws = np.empty(PROMOTION_BOOTSTRAP_DRAWS, np.float64)
        for draw in range(PROMOTION_BOOTSTRAP_DRAWS):
            sampled = rng.choice(sources, size=len(sources), replace=True)
            rows = [sid for source in sampled for sid in source_ids[str(source)]]
            per_seed_delta = []
            for seed in SEEDS:
                per_seed_delta.append(np.mean([abs(stacked_c[seed][sid] - truth[sid]) - abs(stacked_r[seed][sid] - truth[sid]) for sid in rows]))
            draws[draw] = np.mean(per_seed_delta)
        ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        outcome["references"][name] = {
            "per_seed": per_seed, "all_seeds_candidate_lower": all(row["candidate_lower"] for row in per_seed.values()),
            "bootstrap": {"draws": PROMOTION_BOOTSTRAP_DRAWS, "seed": 20261014, "source_count": len(sources), "clip_count": len(ids),
                          "candidate_minus_reference_mean": float(draws.mean()), "ci95_percentile": ci, "upper_below_zero": ci[1] < 0},
        }
        report_check(f"candidate_gate:{name}", all(row["candidate_lower"] for row in per_seed.values()) and ci[1] < 0,
                     outcome["references"][name])
    outcome["all_primary_gates_pass"] = all(
        comparison["all_seeds_candidate_lower"] and comparison["bootstrap"]["upper_below_zero"]
        for comparison in outcome["references"].values()
    )
    return outcome


def _partial_candidate_snapshot(fusions: dict) -> dict:
    """Record completed per-seed VALID comparisons without bootstrap inference."""
    references = ("distilled_static", "distilled_pre_norm", "learned_post_norm", "pre_norm", "post_norm")
    by_seed = {}
    observed_failures = []
    for seed in SEEDS:
        rows = fusions.get(str(seed), {})
        candidate = rows.get("distilled_post_norm", {})
        candidate_metrics = (candidate.get("cpu_replay") or {}).get("independent_native_metrics")
        comparisons = {}
        if candidate.get("complete") and candidate_metrics:
            c_mae = float(candidate_metrics["MAE"])
            for reference in references:
                detail = rows.get(reference, {})
                metrics = (detail.get("cpu_replay") or {}).get("independent_native_metrics")
                if detail.get("complete") and metrics:
                    r_mae = float(metrics["MAE"])
                    lower = c_mae < r_mae
                    comparisons[reference] = {"candidate_MAE": c_mae, "reference_MAE": r_mae,
                                              "candidate_minus_reference_MAE": c_mae - r_mae,
                                              "candidate_lower": lower}
                    if not lower and reference in ("distilled_static", "distilled_pre_norm", "learned_post_norm"):
                        observed_failures.append({"seed": seed, "reference": reference,
                                                  "candidate_minus_reference_MAE": c_mae - r_mae})
            teacher_rows = [comparisons[c] for c in ("pre_norm", "post_norm") if c in comparisons]
            if teacher_rows:
                teacher = min(teacher_rows, key=lambda item: item["reference_MAE"])
                comparisons["strongest_teacher_pre_post_on_same_VALID"] = {
                    "candidate_MAE": c_mae, "reference_MAE": teacher["reference_MAE"],
                    "candidate_minus_reference_MAE": c_mae - teacher["reference_MAE"],
                    "candidate_lower": c_mae < teacher["reference_MAE"],
                    "selection_conditioned_on_this_VALID_MAE": True,
                }
            by_seed[str(seed)] = {"candidate_MAE": c_mae, "references": comparisons}
        else:
            by_seed[str(seed)] = {"status": "candidate_or_references_incomplete"}
    if observed_failures:
        gate_status = "failed_on_completed_seed_reference_cell"
    else:
        all_seed_cells = all(
            by_seed[str(seed)].get("candidate_MAE") is not None
            and all(ref in by_seed[str(seed)].get("references", {}) for ref in ("distilled_static", "distilled_pre_norm", "learned_post_norm", "strongest_teacher_pre_post_on_same_VALID"))
            for seed in SEEDS
        )
        gate_status = "all_seed_VALID_clauses_pass_pending_bootstrap" if all_seed_cells else "partial_pending_more_completed_cells"
    return {
        "status": "descriptive_partial_no_bootstrap",
        "registered_candidate": "distilled_post_norm",
        "registered_valid_gate_status": gate_status,
        "per_seed": by_seed,
        "observed_counterexamples": observed_failures,
        "no_bootstrap_or_final_test_inference": True,
        "note": "These per-seed VALID MAEs are descriptive diagnostics. The registered gate is evaluated separately with all required seeds and the paired source bootstrap; this snapshot does not authorize TEST use or promotion.",
    }


def _runtime_report() -> dict:
    pythonpath = os.environ.get("PYTHONPATH")
    executable = str(Path(sys.executable).resolve())
    mixed_fallback = bool(pythonpath and "fresh-env" in pythonpath.lower() and "fresh-env" not in executable.lower())
    return {
        "python_version": sys.version.split()[0], "python_executable": sys.executable,
        "python_prefix": sys.prefix, "python_base_prefix": sys.base_prefix,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "torch_version": str(torch.__version__), "torch_import_path": str(Path(torch.__file__).resolve()),
        "pythonpath": pythonpath,
        "runtime_classification": "mixed bundled interpreter plus project fresh-env site-packages fallback" if mixed_fallback else "single interpreter environment",
        "device_for_audit": "cpu", "mha_fastpath_enabled": bool(torch.backends.mha.get_fastpath_enabled()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sdp_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdp_mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdp_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        "gpu_context_created_by_auditor": False,
        "training_identity_runtime_versions": "not recorded in the formal checkpoints; CPU replay deltas are measured against saved training-device predictions",
        "note": "A mixed bundled interpreter and fresh-env site-packages path is not the isolated fresh environment. The audit performs no CUDA device query or CUDA model execution.",
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def audit(root: Path, *, report_path: Path, hash_arrays: bool = True,
          replay: bool = True, diagnostics: bool = True) -> dict:
    started = time.time()
    root = root.resolve()
    config_path = root / "configs/native_sequence_gates_v1.json"
    config = _load_json(config_path)
    report: dict[str, Any] = {
        "schema": "native_sequence_gate_independent_cpu_audit_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(root), "status": "partial", "runtime": _runtime_report(),
        "scope": {"formal_select_fits": 3, "formal_refit_fits": 3, "formal_fusion_fits": 21,
                  "valid_unique_clips": 647, "seeds": list(SEEDS), "cases": list(CASES),
                  "test_labels_parsed": False, "test_feature_split_opened": False,
                  "gpu_training_or_encoder_use": False, "native_performance_promotion_claim": False},
        "hash_mode": {"full_array_file_hashes_recomputed": hash_arrays,
                      "array_hash_note": "file bytes are hashed from cache artifacts when enabled; progress checkpoints preserve partial findings"},
        "checks": [], "selections": {}, "refits": {}, "fusions": {}, "matrix": {},
    }

    def check(name: str, passed: bool, detail: Any = None) -> None:
        report["checks"].append({"name": name, "status": "pass" if passed else "fail", "detail": detail})
        if not passed:
            report.setdefault("failures", []).append(name)

    author_path = root / config["author_csv"]
    author_hash = sha256_file(author_path)
    check("author_csv_pinned_hash", author_hash == config["author_csv_sha256"], {"actual": author_hash, "expected": config["author_csv_sha256"]})
    report["config_sha256"] = sha256_file(config_path)
    records_tav, ids_train_only = _read_annotations(author_path, channels=True, valid_m=False)
    records_with_valid, all_ids = _read_annotations(author_path, channels=False, valid_m=True)
    check("official_train_valid_counts", len(all_ids["train"]) == 2722 and len(all_ids["valid"]) == 647,
          {"train": len(all_ids["train"]), "valid": len(all_ids["valid"]), "test_ids_only": len(all_ids["test"])})
    check("channel_scores_train_only", all("TAV" in record for sid, record in records_tav.items() if record["split"] == "train")
          and not any("TAV" in record for record in records_tav.values() if record["split"] != "train"), None)
    annotation_records = records_with_valid
    annotation_records.update({sid: item for sid, item in records_tav.items() if item["split"] == "train"})
    annotations = {"records": annotation_records, "ids": all_ids}
    check("train_M_TAV_native_bounds", all(abs(r["M"]) <= 1 and np.all(np.abs(r["TAV"]) <= 1) for r in annotations["records"].values() if r["split"] == "train"), None)
    raw_cache = (root / config["raw_cache"]).resolve()
    raw_train = CacheSplit(raw_cache, "train")
    check("official_train_source_count", len({sid.split("/", 1)[0] for sid in raw_train.ids}) == 129, len({sid.split("/", 1)[0] for sid in raw_train.ids}))
    check("raw_cache_ids_match_train", set(raw_train.ids) == set(all_ids["train"]), {"raw": len(raw_train.ids), "author": len(all_ids["train"])})
    std_root = (root / config["standardized_cache"]).resolve()
    std_train_all = CacheSplit(std_root, "train")
    std_valid_all = CacheSplit(std_root, "valid")
    check("standardized_cache_ids_match_official", std_train_all.ids == all_ids["train"] and std_valid_all.ids == all_ids["valid"],
          {"train_count": len(std_train_all.ids), "valid_count": len(std_valid_all.ids),
           "train_valid_overlap": bool(set(std_train_all.ids) & set(std_valid_all.ids))})

    formal_cells = {}
    for seed in SEEDS:
        select_path = root / config["run_root"] / "select" / f"seed_{seed}"
        select_cp_path = select_path / "checkpoint.pt"
        select = _load_checkpoint(select_cp_path) if select_cp_path.is_file() else {}
        if select:
            select["_file_sha256"] = sha256_file(select_cp_path)
            identity = _load_json(select_path / "identity.json")
            _verify_identity(root, identity, config, author_hash, check)
            cache_report = _verify_cache_identity(root, config, identity, hash_arrays=hash_arrays, report_check=check)
            sel_result = _load_json(select_path / "result.json") if (select_path / "result.json").is_file() else {}
            if sel_result:
                check(f"selection_result_sha:{seed}", sel_result.get("checkpoint_sha256") == select["_file_sha256"], None)
            details = _source_metric_audit(root, config, identity, select, annotations, check)
            select_formal_complete = bool(select.get("complete")) and not bool(identity.get("smoke_only", True)) and bool(sel_result.get("complete"))
            report["selections"][str(seed)] = {"status": "complete" if select_formal_complete else "in_progress",
                                               "result": sel_result, "audit": details, "cache": cache_report,
                                               "checkpoint_sha256": select["_file_sha256"], "complete": select_formal_complete}
        else:
            report["selections"][str(seed)] = {"status": "missing"}
        refit_path = root / config["run_root"] / "refit" / f"seed_{seed}"
        refit_cp_path = refit_path / "checkpoint.pt"
        refit = _load_checkpoint(refit_cp_path) if refit_cp_path.is_file() else {}
        if refit:
            refit_identity = _load_json(refit_path / "identity.json")
            _verify_identity(root, refit_identity, config, author_hash, check)
            cache_report = _verify_cache_identity(root, config, refit_identity, hash_arrays=hash_arrays, report_check=check)
            std_train = CacheSplit((root / config["standardized_cache"]).resolve(), "train")
            detail = _audit_refit(root, config, seed, select, refit, std_train, check) if select else {"status": "selection_missing"}
            refit_result = _load_json(refit_path / "result.json") if (refit_path / "result.json").is_file() else {}
            refit_identity_formal = not bool(refit_identity.get("smoke_only", True))
            detail["complete"] = bool(refit.get("complete")) and refit_identity_formal and bool(refit_result.get("complete"))
            report["refits"][str(seed)] = {**detail, "cache": cache_report, "checkpoint_sha256": sha256_file(refit_cp_path)}
        else:
            report["refits"][str(seed)] = {"status": "missing"}
        for case in CASES:
            fdir = root / config["run_root"] / "fusion" / f"seed_{seed}" / case
            fcp = fdir / "checkpoint.pt"
            if not fcp.is_file():
                report["fusions"].setdefault(str(seed), {})[case] = {"status": "missing"}
                continue
            identity_path = fdir / "identity.json"
            if not identity_path.is_file():
                report["fusions"].setdefault(str(seed), {})[case] = {"status": "identity_missing"}
                continue
            # Defer only the selected-score CPU replay until a formal checkpoint
            # is complete. A partial checkpoint is inspected read-only.
            cp_preview = _load_checkpoint(fcp)
            complete_preview = bool(cp_preview.get("complete")) and (fdir / "result.json").is_file()
            need_data = replay and complete_preview
            if need_data:
                # The hash checker reads arrays already used by the trainer. If
                # this matrix is live, invoke with --defer-array-hashes to avoid
                # large extra disk reads during fitting.
                f_identity = _load_json(identity_path)
                _verify_cache_identity(root, config, f_identity, hash_arrays=hash_arrays, report_check=check)
            std_train = std_train_all if complete_preview else None
            std_valid = std_valid_all if need_data else None
            details = _audit_fusion(root, config, seed, case, identity_path, fcp, fdir / "result.json",
                                    std_valid,
                                    std_train,
                                    annotations, check, replay=need_data, diagnostics=diagnostics)
            report["fusions"].setdefault(str(seed), {})[case] = details
            formal_cells[f"fusion/seed_{seed}/{case}"] = details.get("complete", False)
            _write_report(report_path, report)

    for seed in SEEDS:
        rows = [report["fusions"].get(str(seed), {}).get(case, {}) for case in CASES]
        hashes = [row.get("initial_fusion_state_sha256") for row in rows if row.get("initial_fusion_state_sha256")]
        if hashes:
            checked = len(hashes) == len(CASES)
            check(f"fusion_all_seven_same_initial_hash:{seed}", not checked or len(set(hashes)) == 1,
                  {"available": len(hashes), "expected": len(CASES), "hashes": hashes,
                   "status": "verified" if checked else "pending_more_cases"})

    all_cells = [report["selections"].get(str(s), {}).get("complete", False) for s in SEEDS]
    all_cells += [report["refits"].get(str(s), {}).get("complete", False) for s in SEEDS]
    all_cells += [report["fusions"].get(str(s), {}).get(case, {}).get("complete", False) for s in SEEDS for case in CASES]
    report["matrix"] = {"complete": all(all_cells), "complete_cells": int(sum(all_cells)), "expected_cells": 27,
                        "unique_valid_clip_count": 647, "prediction_cells_are_not_independent_samples": True}
    report["partial_candidate_snapshot"] = _partial_candidate_snapshot(report["fusions"])
    report["promotion_comparison"] = _audit_comparisons(root, config, check) if report["matrix"]["complete"] else {"status": "pending_until_all_27_formal_cells_complete"}
    report["status"] = "complete" if report["matrix"]["complete"] else "partial"
    report["elapsed_seconds"] = round(time.time() - started, 3)
    _write_report(report_path, report)
    return report


def _load_config_hash(path: Path) -> str:
    return sha256_file(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/native_sequence_gate_audit_20261014.json")
    parser.add_argument("--defer-array-hashes", action="store_true", help="Use run-bound SHA manifests and defer large .npy reads during live fitting")
    parser.add_argument("--no-replay", action="store_true", help="Verify stage receipts only; do not recompute selected VALID predictions")
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    # CPU-only numerical policy: no CUDA availability/device query is made.
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    report_path = args.report if args.report.is_absolute() else args.root / args.report
    report = audit(args.root, report_path=report_path,
                   hash_arrays=not args.defer_array_hashes, replay=not args.no_replay,
                   diagnostics=not args.no_diagnostics)
    print(json.dumps({"status": report["status"], "matrix": report["matrix"],
                      "checks": len(report["checks"]), "failures": report.get("failures", []),
                      "report": str(report_path.resolve()), "elapsed_seconds": report["elapsed_seconds"]}, allow_nan=False))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
