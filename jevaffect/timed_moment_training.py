"""Native scalar training for the pre-registered timed A/V moment controls.

This runner consumes only official TRAIN and VALID rows from a frozen timed-grid
cache.  It reuses the already fitted pooled-coordinate statistics and never
opens TEST labels or fits a new normalizer.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jevaffect.anchor_calibration import normalize
from jevaffect.native_metrics import native_metrics
from jevaffect.native_training import author_rows, digest, save_checkpoint, save_json
from jevaffect.timed_moments import TimedMomentRegressor


CASES = ("static_product", "marginal_variance", "paired_cov", "unpaired_cov")
FORMAL_SEEDS = (13, 42, 2026)
EXPECTED_COUNTS = {"train": 2722, "valid": 647}
EXPECTED_DIMS = {"text": 768, "audio": 768, "vision": 384}
TIMED_CACHE_FORMAT = "native-timed-grid-v1"
SOURCE_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
PARAMETER_COUNT = 494101

_FILES = {
    "text": "text.npy",
    "audio_mean": "audio_mean.npy",
    "vision_mean": "vision_mean.npy",
    "audio_grid": "audio_grid.npy",
    "vision_grid": "vision_grid.npy",
    "pair_valid": "pair_valid.npy",
    "presence": "presence.npy",
    "grid_times": "grid_times.npy",
}


def _sha256_json(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_cache_path(cache: Path, metadata: dict, split: str) -> Path:
    configured = metadata.get("split_paths", {}).get(split)
    if configured is None:
        configured = "splits/" + split if (cache / "splits").exists() else split
    directory = (cache / configured).resolve()
    try:
        directory.relative_to(cache.resolve())
    except ValueError as exc:
        raise ValueError("Cache split path escapes cache root") from exc
    if not directory.is_dir():
        raise ValueError("Missing timed-grid split directory: " + str(directory))
    return directory


def _artifact_hash(metadata: dict, relative_path: str) -> str | None:
    record = metadata.get("artifact_sha256", {}).get(relative_path)
    if record is None:
        record = metadata.get("files", {}).get(relative_path)
    if isinstance(record, dict):
        record = record.get("sha256")
    return record if isinstance(record, str) else None


def _verified_file(path: Path, cache: Path, metadata: dict, collected: dict) -> str:
    relative = path.resolve().relative_to(cache.resolve()).as_posix()
    actual = digest(path)
    known = _artifact_hash(metadata, relative)
    if known is None or known != actual:
        raise ValueError("Missing or mismatched timed-grid cache artifact hash: " + relative)
    collected[relative] = actual
    return actual


def _load_normalization(directory: Path, author_csv_sha256: str, allow_fixture: bool):
    directory = Path(directory).resolve()
    metadata_path = directory / "metadata.json"
    stats_path = directory / "normalization.json"
    if not metadata_path.is_file() or not stats_path.is_file():
        raise ValueError("Existing pooled-standardization metadata and normalization.json are required")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stats_sha = digest(stats_path)
    expected_stats_sha = _artifact_hash(metadata, "normalization.json")
    if expected_stats_sha is None or expected_stats_sha != stats_sha:
        raise ValueError("Pooled normalization.json hash differs from its metadata")
    source_hash = metadata.get("author_csv_sha256", metadata.get("official_csv_sha256"))
    if source_hash and source_hash != author_csv_sha256:
        raise ValueError("Pooled statistics were prepared from a different author CSV")
    if not allow_fixture:
        repo_root = Path(__file__).resolve().parents[1]
        if source_hash != SOURCE_CSV_SHA256:
            raise ValueError("Pinned official author CSV provenance is required for pooled statistics")
        if metadata.get("counts") != EXPECTED_COUNTS:
            raise ValueError("Pooled statistics cache must contain all official TRAIN and VALID rows")
        norm_meta = metadata.get("normalization", {})
        if norm_meta.get("fit_split") != "train" or norm_meta.get("fit_count") != EXPECTED_COUNTS["train"]:
            raise ValueError("Pooled statistics must be fitted on all official TRAIN clip means")
        producer = repo_root / "scripts" / "prepare_native_pooled_standardization.py"
        implementation = repo_root / "jevaffect" / "anchor_calibration.py"
        if metadata.get("preparation_sha256") != digest(producer):
            raise ValueError("Pooled statistics producer hash differs from the current implementation")
        if metadata.get("normalization_implementation_sha256") != digest(implementation):
            raise ValueError("Pooled normalization implementation hash differs from current code")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not set(EXPECTED_DIMS) <= set(stats):
        raise ValueError("Pooled normalization statistics are missing a modality")
    for modality, dim in EXPECTED_DIMS.items():
        entry = stats[modality]
        mean = np.asarray(entry.get("mean"), dtype=np.float32)
        scale = np.asarray(entry.get("scale"), dtype=np.float32)
        if mean.shape != (dim,) or scale.shape != (dim,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid pooled normalization coordinates for " + modality)
        fit_indices = entry.get("fit_indices")
        if not isinstance(fit_indices, list) or not fit_indices or len(set(fit_indices)) != len(fit_indices):
            raise ValueError("Invalid pooled normalization fit indices for " + modality)
        if not allow_fixture and fit_indices != list(range(EXPECTED_COUNTS["train"])):
            raise ValueError("Pooled statistics must use every official TRAIN index")
    return stats, {"path": str(stats_path), "sha256": stats_sha,
                   "metadata_path": str(metadata_path), "metadata_sha256": digest(metadata_path),
                   "metadata": metadata}


class TimedMomentSplit(Dataset):
    """Hash-checked native split with frozen pooled statistics applied in memory."""

    def __init__(self, cache_dir, split, rows, stats, limit=None, allow_fixture=False):
        if split not in ("train", "valid"):
            raise ValueError("Timed moment training can only open TRAIN or VALID")
        self.cache = Path(cache_dir).resolve()
        metadata_path = self.cache / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Timed-grid cache metadata.json is required")
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache_format = self.meta.get("format", self.meta.get("cache_format"))
        if cache_format != TIMED_CACHE_FORMAT:
            raise ValueError("Unsupported timed-grid cache format")
        if self.meta.get("test_exported", False):
            raise ValueError("Timed-grid cache must not export TEST data")
        self.metadata_sha256 = digest(metadata_path)
        self.directory = _resolve_cache_path(self.cache, self.meta, split)
        self.ids = json.loads((self.directory / "ids.json").read_text(encoding="utf-8"))
        if not isinstance(self.ids, list) or not self.ids or any(not isinstance(sid, str) for sid in self.ids):
            raise ValueError("Timed-grid IDs must be a nonempty JSON string list")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("Duplicate timed-grid IDs")
        expected = {sid for sid, row in rows.items() if row["split"] == split}
        if set(self.ids) != expected:
            raise ValueError("Timed-grid IDs differ from the complete native author split")

        self.targets = np.asarray([rows[sid]["score"] for sid in self.ids], dtype=np.float32)
        self.hashes = {}
        for required in (self.directory / "ids.json", self.directory / "raw_labels.json"):
            _verified_file(required, self.cache, self.meta, self.hashes)
        raw_labels = np.asarray(json.loads((self.directory / "raw_labels.json").read_text(encoding="utf-8")), dtype=np.float32)
        if raw_labels.shape != self.targets.shape or not np.isfinite(raw_labels).all() or not np.allclose(raw_labels, self.targets, atol=1e-6, rtol=0):
            raise ValueError("Timed-grid raw labels differ from original native M labels")

        arrays = {}
        for name, filename in _FILES.items():
            path = self.directory / filename
            _verified_file(path, self.cache, self.meta, self.hashes)
            arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        n = len(self.ids)
        required_shapes = {
            "text": (n, EXPECTED_DIMS["text"]),
            "audio_mean": (n, EXPECTED_DIMS["audio"]),
            "vision_mean": (n, EXPECTED_DIMS["vision"]),
            "audio_grid": (n, 4, EXPECTED_DIMS["audio"]),
            "vision_grid": (n, 4, EXPECTED_DIMS["vision"]),
            "pair_valid": (n,), "presence": (n, 3), "grid_times": (n, 4),
        }
        for name, shape in required_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"Invalid {name} shape: {arrays[name].shape}, expected {shape}")
        for name in ("text", "audio_mean", "vision_mean", "audio_grid", "vision_grid"):
            if arrays[name].dtype.kind != "f" or not np.isfinite(arrays[name]).all():
                raise ValueError("Timed-grid feature array must be finite floating point: " + name)
        if arrays["pair_valid"].dtype.kind != "b" or arrays["presence"].dtype.kind != "b":
            raise ValueError("pair_valid and presence arrays must be boolean")
        if arrays["grid_times"].dtype != np.dtype("float64") or not np.isfinite(arrays["grid_times"]).all():
            raise ValueError("grid_times must be finite float64 values")
        pair_valid = np.asarray(arrays["pair_valid"], dtype=bool)
        presence = np.asarray(arrays["presence"], dtype=bool)
        if np.any(pair_valid & ~(presence[:, 1] & presence[:, 2])):
            raise ValueError("A timed pair cannot be valid when audio or vision is absent")
        times = np.asarray(arrays["grid_times"])
        if np.any(np.diff(times[pair_valid], axis=1) <= 0):
            raise ValueError("Valid timed-grid timestamps must be strictly increasing")

        # Use the existing TRAIN-fitted clip-pooled coordinates for global means
        # and every valid grid step. Missing globals and invalid pairs are reset
        # to zero only after this affine transformation.
        present_text = presence[:, 0]
        present_audio = presence[:, 1]
        present_vision = presence[:, 2]
        valid_pair = pair_valid & present_audio & present_vision
        transformed = {
            "text": normalize(np.asarray(arrays["text"]), stats["text"]),
            "audio_mean": normalize(np.asarray(arrays["audio_mean"]), stats["audio"]),
            "vision_mean": normalize(np.asarray(arrays["vision_mean"]), stats["vision"]),
            "audio_grid": normalize(np.asarray(arrays["audio_grid"]).reshape(-1, EXPECTED_DIMS["audio"]), stats["audio"]).reshape(n, 4, EXPECTED_DIMS["audio"]),
            "vision_grid": normalize(np.asarray(arrays["vision_grid"]).reshape(-1, EXPECTED_DIMS["vision"]), stats["vision"]).reshape(n, 4, EXPECTED_DIMS["vision"]),
        }
        transformed["text"][~present_text] = 0
        transformed["audio_mean"][~present_audio] = 0
        transformed["vision_mean"][~present_vision] = 0
        transformed["audio_grid"][~valid_pair] = 0
        transformed["vision_grid"][~valid_pair] = 0
        self.features = transformed
        self.pair_valid = valid_pair
        self.presence = presence
        self.grid_times = times
        if limit is not None:
            if limit < 1:
                raise ValueError("Smoke limit must be positive")
            self.ids = self.ids[:limit]
            self.targets = self.targets[:limit]
            self.features = {key: value[:limit] for key, value in self.features.items()}
            self.pair_valid = self.pair_valid[:limit]
            self.presence = self.presence[:limit]
            self.grid_times = self.grid_times[:limit]
        self.smoke_limit = limit

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        features = {key: torch.from_numpy(np.array(values[index], dtype=np.float32, copy=True))
                    for key, values in self.features.items()}
        features["pair_valid"] = torch.tensor(bool(self.pair_valid[index]), dtype=torch.bool)
        return features, torch.tensor(float(self.targets[index]), dtype=torch.float32), self.ids[index]


def _move_features(features, device):
    return {key: value.to(device) for key, value in features.items()}


def evaluate(model, loader, device):
    model.eval()
    predictions, targets, ids = [], [], []
    with torch.no_grad():
        for features, target, sample_ids in loader:
            prediction = model(_move_features(features, device)).flatten().cpu().numpy()
            if prediction.shape != target.numpy().shape or not np.isfinite(prediction).all():
                raise ValueError("Timed moment model returned nonfinite or wrongly shaped scores")
            predictions.extend(prediction.tolist())
            targets.extend(target.tolist())
            ids.extend(sample_ids)
    p = np.asarray(predictions, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    metrics = native_metrics(p, y, "simsv2")
    records = [{"id": sid, "target": float(target), "prediction": float(prediction)}
               for sid, target, prediction in zip(ids, y, p)]
    return metrics, records


def _implementation_hashes(repo_root: Path):
    paths = [
        Path(__file__),
        Path(__file__).with_name("timed_moments.py"),
        Path(__file__).with_name("native_metrics.py"),
        Path(__file__).with_name("native_training.py"),
        Path(__file__).with_name("anchor_calibration.py"),
        repo_root / "scripts" / "train_timed_moments.py",
        repo_root / "scripts" / "prepare_native_pooled_standardization.py",
        repo_root / "docs" / "NATIVE_TIMED_MOMENT_PLAN_V1.md",
    ]
    # Bind the cache producer when the repository contains it.
    for candidate in (repo_root / "scripts" / "prepare_native_timed_grid.py",
                      repo_root / "scripts" / "prepare_timed_moment_cache.py"):
        if candidate.is_file():
            paths.append(candidate)
    result = {}
    for path in paths:
        if path.is_file():
            result[path.resolve().relative_to(repo_root.resolve()).as_posix()] = digest(path)
    return result


def _load_and_validate_data(config, smoke_limit=None, allow_fixture=False):
    csv_path = Path(config["author_csv"]).resolve()
    csv_sha = digest(csv_path)
    if not allow_fixture and csv_sha != SOURCE_CSV_SHA256:
        raise ValueError("Pinned native author CSV required")
    rows = author_rows(csv_path, "simsv2")
    counts = {split: sum(row["split"] == split for row in rows.values()) for split in ("train", "valid")}
    if not allow_fixture and counts != EXPECTED_COUNTS:
        raise ValueError("Pinned native CSV must contain exactly 2,722 TRAIN and 647 VALID labels")
    if not all(counts.values()):
        raise ValueError("Native TRAIN and VALID must both be nonempty")

    normalization_dir = Path(config["normalization_cache_dir"])
    stats, stats_identity = _load_normalization(normalization_dir, csv_sha, allow_fixture)
    cache_dir = Path(config["cache_dir"]).resolve()
    meta_path = cache_dir / "metadata.json"
    if not meta_path.is_file():
        raise ValueError("Timed-grid cache metadata.json is required")
    cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cache_csv_sha = cache_meta.get("author_csv_sha256", cache_meta.get("source_csv_sha256", cache_meta.get("official_csv_sha256")))
    if cache_csv_sha is not None and cache_csv_sha != csv_sha:
        raise ValueError("Timed-grid cache was prepared from a different author CSV")
    cache_counts = cache_meta.get("counts")
    if cache_counts is not None and any(cache_counts.get(split) != counts[split] for split in ("train", "valid")):
        raise ValueError("Timed-grid metadata counts differ from the author CSV")
    if not allow_fixture:
        if cache_meta.get("format", cache_meta.get("cache_format")) != TIMED_CACHE_FORMAT:
            raise ValueError("Pinned native timed-grid cache format required")
        if cache_csv_sha != SOURCE_CSV_SHA256:
            raise ValueError("Timed-grid cache must declare the pinned author CSV hash")
        if cache_counts != EXPECTED_COUNTS:
            raise ValueError("Timed-grid cache must contain all official TRAIN and VALID rows")
        repo_root = Path(__file__).resolve().parents[1]
        producer = repo_root / "scripts" / "prepare_native_timed_grid.py"
        reader = repo_root / "jevaffect" / "native_training.py"
        if cache_meta.get("preparation_sha256") != digest(producer):
            raise ValueError("Timed-grid producer hash differs from the current implementation")
        if cache_meta.get("native_reader_sha256") != digest(reader):
            raise ValueError("Timed-grid native-reader hash differs from current code")

    datasets = {split: TimedMomentSplit(cache_dir, split, rows, stats, limit=smoke_limit, allow_fixture=allow_fixture)
                for split in ("train", "valid")}
    if set(datasets["train"].ids) & set(datasets["valid"].ids):
        raise ValueError("Timed-grid TRAIN and VALID IDs overlap")

    # Statistics and timed inputs must use the same frozen official cohorts.
    pooled_meta = stats_identity["metadata"]
    pooled_author = pooled_meta.get("author_csv_sha256", pooled_meta.get("official_csv_sha256"))
    if pooled_author and pooled_author != csv_sha:
        raise ValueError("Timed-grid author CSV differs from the normalization cache source")
    pooled_counts = pooled_meta.get("counts")
    if pooled_counts is not None and any(pooled_counts.get(split) != counts[split] for split in ("train", "valid")):
        raise ValueError("Normalization cache cohorts differ from the author CSV")
    for split, dataset in datasets.items():
        pooled_split = _resolve_cache_path(normalization_dir.resolve(), pooled_meta, split)
        pooled_ids_path = pooled_split / "ids.json"
        if not pooled_ids_path.is_file():
            raise ValueError("Pooled normalization cache is missing split IDs")
        pooled_ids = json.loads(pooled_ids_path.read_text(encoding="utf-8"))
        if set(pooled_ids) != {sid for sid, row in rows.items() if row["split"] == split}:
            raise ValueError("Pooled normalization IDs differ from the complete author split")
        if split == "train" and pooled_meta.get("normalization", {}).get("fit_ids_sha256"):
            if digest(pooled_ids_path) != pooled_meta["normalization"]["fit_ids_sha256"]:
                raise ValueError("Pooled normalization fit-ID hash mismatch")
    return rows, datasets, stats_identity, csv_sha, counts


def train(config, case, seed, output_dir=None, resume=False, smoke32=False,
          stop_after=None, allow_fixture=False, config_sha256=None):
    """Train one case/seed.  Fixture bypass is available only through this API."""
    if case not in CASES:
        raise ValueError("Unknown timed moment case")
    if seed not in config.get("seeds", FORMAL_SEEDS) or seed not in FORMAL_SEEDS:
        raise ValueError("Formal timed moment seeds are 13, 42 and 2026")
    if tuple(config.get("seeds", FORMAL_SEEDS)) != FORMAL_SEEDS:
        raise ValueError("Formal seed contract is 13, 42, 2026")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if smoke32 and allow_fixture:
        # Fixture row counts can be smaller than 32; the smoke prefix is still
        # exactly the public flow-check limit when data allows it.
        smoke_limit = 32
    else:
        smoke_limit = 32 if smoke32 else None
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    rows, datasets, stats_identity, csv_sha, full_counts = _load_and_validate_data(
        config, smoke_limit=smoke_limit, allow_fixture=allow_fixture)
    if smoke32 and min(len(datasets["train"]), len(datasets["valid"])) < 1:
        raise ValueError("Smoke prefix is empty")
    dims = {"text": EXPECTED_DIMS["text"], "audio": EXPECTED_DIMS["audio"], "vision": EXPECTED_DIMS["vision"]}
    model = TimedMomentRegressor(dims, case, rank=32, hidden=234, dropout=.1)
    param_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if param_count != PARAMETER_COUNT:
        raise ValueError(f"TimedMomentRegressor parameter count is {param_count}, expected {PARAMETER_COUNT}")
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Configured CUDA device is unavailable")
    model.to(device)

    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 1e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    max_epochs = int(config.get("max_epochs", 80))
    patience = int(config.get("patience", 12))
    if allow_fixture:
        if batch_size < 1 or learning_rate <= 0 or weight_decay < 0 or max_epochs < 1 or patience < 1:
            raise ValueError("Invalid fixture training budget")
    elif (batch_size, learning_rate, weight_decay, max_epochs, patience) != (32, 1e-4, 1e-4, 80, 12):
        raise ValueError("Timed-moment training budget is fixed at batch 32, AdamW 1e-4/1e-4, 80 epochs, patience 12")
    if (config.get("rank", 32), config.get("hidden", 234), config.get("dropout", .1)) != (32, 234, .1):
        raise ValueError("Timed-moment model contract is rank 32, hidden 234, dropout .1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
                          generator=generator if split == "train" else None, num_workers=0)
        for split, dataset in datasets.items()
    }

    repo_root = Path(__file__).resolve().parents[1]
    run_root = Path(output_dir if output_dir is not None else config["run_root"]).resolve()
    run_dir = run_root / ("smoke32" if smoke32 else "formal") / case / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    implementation = _implementation_hashes(repo_root)
    effective_config = dict(config)
    effective_config.update({"case": case, "seed": seed, "smoke32": bool(smoke32), "output_dir": str(run_root)})
    identity = {
        "cache_format": TIMED_CACHE_FORMAT,
        "case": case,
        "seed": seed,
        "smoke_limit": 32 if smoke32 else None,
        "config": effective_config,
        "config_sha256": config_sha256 or _sha256_json(effective_config),
        "author_csv": str(Path(config["author_csv"]).resolve()),
        "author_csv_sha256": csv_sha,
        "full_author_counts": full_counts,
        "used_counts": {split: len(data) for split, data in datasets.items()},
        "timed_cache": str(Path(config["cache_dir"]).resolve()),
        "timed_cache_metadata_sha256": datasets["train"].metadata_sha256,
        "features": {split: data.hashes for split, data in datasets.items()},
        "timed_cache_provenance": {key: datasets["train"].meta.get(key) for key in (
            "preparation_sha256", "native_reader_sha256", "source_cache", "source_descriptor",
            "source_truncation", "source_feature_sha256", "time_contract", "coverage")},
        "normalization": {key: value for key, value in stats_identity.items() if key != "metadata"},
        "normalization_provenance": {key: stats_identity["metadata"].get(key) for key in (
            "author_csv_sha256", "source_cache", "source_feature_sha256", "normalization", "counts", "dimensions")},
        "implementation": implementation,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameter_count": param_count,
        "selection": "minimum author VALID native clipped MAE; first checkpoint wins ties",
        "loss": "unclipped L1 on original author scores",
        "budget": {"batch_size": batch_size, "learning_rate": learning_rate, "weight_decay": weight_decay,
                   "max_epochs": max_epochs, "early_stopping_patience": patience,
                   "scheduler": {"name": "ReduceLROnPlateau", "metric": "valid native MAE", "patience": 3, "factor": .5},
                   "gradient_clip_norm": 1.0},
        "test_evaluated": False,
    }

    last_path = run_dir / "last.pt"
    epoch0, best, stale, history = 0, float("inf"), 0, []
    if last_path.exists():
        if not resume:
            raise ValueError("Run exists; pass --resume")
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("identity") != identity:
            raise ValueError("Timed-moment checkpoint identity mismatch")
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
    elif resume and any(run_dir.iterdir()):
        raise ValueError("Run directory has artifacts but no resumable last.pt")

    save_json(run_dir / "identity.json", identity)
    cap = min(max_epochs, epoch0 + stop_after) if stop_after is not None else max_epochs
    start_all = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epoch0, cap):
        if stale >= patience:
            break
        epoch_start = time.perf_counter()
        model.train()
        loss_total, seen = 0.0, 0
        for features, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = target.to(device)
            prediction = model(_move_features(features, device)).flatten()
            loss = nn.functional.l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite timed-moment training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.item()) * len(target)
            seen += len(target)
        metrics, predictions = evaluate(model, loaders["valid"], device)
        score = metrics["MAE"]
        scheduler.step(score)
        row = {"epoch": epoch + 1, "train_L1": loss_total / seen, "valid": metrics,
               "seconds": time.perf_counter() - epoch_start, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        if score < best:
            best, stale = score, 0
            save_checkpoint(run_dir / "best.pt", {"identity": identity, "epoch": epoch + 1,
                            "model": model.state_dict(), "metrics": metrics, "parameter_count": param_count})
            save_json(run_dir / "valid_predictions.json", predictions)
        else:
            stale += 1
        state = {
            "identity": identity, "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(last_path, state)
        save_json(run_dir / "history.json", history)
        print(json.dumps({"case": case, "seed": seed, **row}, allow_nan=False), flush=True)

    if not (run_dir / "best.pt").is_file():
        raise ValueError("No best checkpoint was produced")
    best_checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    if best_checkpoint.get("identity") != identity:
        raise ValueError("Best checkpoint identity mismatch")
    model.load_state_dict(best_checkpoint["model"])
    selected_metrics, selected_predictions = evaluate(model, loaders["valid"], device)
    saved_predictions = json.loads((run_dir / "valid_predictions.json").read_text(encoding="utf-8"))
    if selected_predictions != saved_predictions:
        raise ValueError("Reloaded selected checkpoint differs from saved VALID predictions")
    stopped_by_patience = stale >= patience
    finished_budget = epoch0 >= max_epochs or cap >= max_epochs
    complete = not smoke32 and (stopped_by_patience or (finished_budget and len(history) >= max_epochs))
    loop_seconds = float(sum(item["seconds"] for item in history))
    peak_gpu_bytes = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    result = {
        "case": case, "seed": seed, "dataset": "simsv2", "smoke_only": bool(smoke32),
        "complete": bool(complete), "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]),
        "full_train_count": full_counts["train"], "full_valid_count": full_counts["valid"],
        "best_epoch": best_checkpoint["epoch"], "valid_metrics": selected_metrics,
        "loop_seconds": loop_seconds, "invocation_seconds": time.perf_counter() - start_all,
        "peak_gpu_memory_bytes": peak_gpu_bytes, "trainable_parameters": param_count,
        "max_epochs": max_epochs, "patience": patience, "checkpoint_sha256": digest(run_dir / "best.pt"),
        "last_checkpoint_sha256": digest(last_path) if last_path.exists() else None,
        "predictions_sha256": digest(run_dir / "valid_predictions.json"), "identity_sha256": digest(run_dir / "identity.json"),
        "test_evaluated": False,
    }
    save_json(run_dir / "result.json", result)
    return result


def run_matrix(config, cases=None, seeds=None, output_dir=None, resume=False, smoke32=False,
               stop_after=None, config_sha256=None):
    """Run requested cases/seeds serially and record whether the full matrix finished."""
    chosen_cases = tuple(cases or CASES)
    chosen_seeds = tuple(seeds or FORMAL_SEEDS)
    if not chosen_cases or any(case not in CASES for case in chosen_cases):
        raise ValueError("Invalid timed-moment case selection")
    if not chosen_seeds or any(seed not in FORMAL_SEEDS for seed in chosen_seeds):
        raise ValueError("Invalid timed-moment seed selection")
    results = []
    for case in chosen_cases:
        for seed in chosen_seeds:
            results.append(train(config, case, seed, output_dir=output_dir, resume=resume,
                                 smoke32=smoke32, stop_after=stop_after, config_sha256=config_sha256))
    expected = {(case, seed) for case in CASES for seed in FORMAL_SEEDS}
    completed = {(row["case"], row["seed"]) for row in results if row["complete"]}
    matrix_complete = not smoke32 and completed == expected and len(results) == len(expected)
    matrix = {
        "dataset": "simsv2", "cases": list(chosen_cases), "seeds": list(chosen_seeds),
        "expected_full_matrix": [{"case": case, "seed": seed} for case in CASES for seed in FORMAL_SEEDS],
        "run_count": len(results), "complete_run_count": sum(row["complete"] for row in results),
        "smoke_only": bool(smoke32), "matrix_complete": bool(matrix_complete), "test_evaluated": False,
        "results": results,
    }
    root = Path(output_dir if output_dir is not None else config["run_root"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / ("smoke32_matrix.json" if smoke32 else "matrix.json"), matrix)
    return matrix
