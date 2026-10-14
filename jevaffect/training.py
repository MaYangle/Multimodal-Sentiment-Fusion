"""Deterministic cached-feature training for JevAffect baselines.

This trainer reads only the ``train`` and ``valid`` cache splits. Checkpoints
resume at completed epoch boundaries and are bound to the exact normalized
config, cache metadata, implementation files, seed, and runtime environment.
No test loader or test-evaluation path is provided here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

# Set before importing torch so deterministic CUDA GEMM can be requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from jevaffect.models import build_model


SEEDS = (13, 42, 2026)
_MODALITIES = ("text", "audio", "vision")
_CLASS_ORDER = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
_CONFIG_KEYS = {"dataset", "language", "cache_dir", "run_root", "seeds", "model", "training"}
_TRAINING_DEFAULTS = {
    "max_epochs": 80,
    "batch_size": 32,
    "patience": 12,
    "min_delta": 0.0,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip_norm": 1.0,
    "scheduler_patience": 3,
    "scheduler_factor": 0.5,
    "num_workers": 0,
    "device": "auto",
    "amp": False,
    "amp_dtype": "float16",
}
_TRAINING_KEYS = set(_TRAINING_DEFAULTS)


class TrainingError(RuntimeError):
    """Raised when the cache, config, or run state is invalid."""


class RunIdentityError(TrainingError):
    """Raised when resume would mix a run with a different identity."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingError("{} must be a positive integer".format(name))
    return value


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingError("{} must be a finite number".format(name))
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TrainingError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(number):
        raise TrainingError("{} must be a finite number".format(name))
    if minimum is not None and number < minimum:
        raise TrainingError("{} must be >= {}".format(name, minimum))
    return number


def _absolute_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError("{} must be a non-empty path string".format(name))
    return str(Path(value).expanduser().resolve())


def normalize_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the canonical config used for hashing and runs."""
    if not isinstance(config, Mapping):
        raise TrainingError("config must be a JSON object")
    unknown = set(config) - _CONFIG_KEYS
    missing = _CONFIG_KEYS - set(config)
    if unknown:
        raise TrainingError("unknown top-level config keys: {}".format(sorted(unknown)))
    if missing:
        raise TrainingError("missing top-level config keys: {}".format(sorted(missing)))

    dataset = config["dataset"]
    language = config["language"]
    if not isinstance(dataset, str) or not dataset.strip():
        raise TrainingError("dataset must be a non-empty string")
    if not isinstance(language, str) or not language.strip():
        raise TrainingError("language must be a non-empty string")

    seeds_raw = config["seeds"]
    if not isinstance(seeds_raw, list) or tuple(seeds_raw) != SEEDS:
        raise TrainingError("seeds must be exactly [13, 42, 2026] in that order")

    model = config["model"]
    if not isinstance(model, Mapping):
        raise TrainingError("model must be a JSON object")
    model_unknown = set(model) - {"name", "params"}
    if model_unknown or "name" not in model:
        raise TrainingError("model must contain name and optional params only")
    if model["name"] not in ("text_mlp", "almt"):
        raise TrainingError("model.name must be text_mlp or almt")
    params = model.get("params", {})
    if not isinstance(params, Mapping):
        raise TrainingError("model.params must be a JSON object")

    training_raw = config["training"]
    if not isinstance(training_raw, Mapping):
        raise TrainingError("training must be a JSON object")
    training_unknown = set(training_raw) - _TRAINING_KEYS
    if training_unknown:
        raise TrainingError("unknown training keys: {}".format(sorted(training_unknown)))
    training = dict(_TRAINING_DEFAULTS)
    training.update(training_raw)
    for name in ("max_epochs", "batch_size", "patience"):
        training[name] = _positive_int(training[name], name)
    if (
        isinstance(training["scheduler_patience"], bool)
        or not isinstance(training["scheduler_patience"], int)
        or training["scheduler_patience"] < 0
    ):
        raise TrainingError("scheduler_patience must be a non-negative integer")
    if (
        isinstance(training["num_workers"], bool)
        or not isinstance(training["num_workers"], int)
        or training["num_workers"] != 0
    ):
        raise TrainingError("num_workers must be 0 for exact epoch-boundary resume")
    training["min_delta"] = _finite_float(training["min_delta"], "min_delta", minimum=0.0)
    training["learning_rate"] = _finite_float(training["learning_rate"], "learning_rate", minimum=0.0)
    if training["learning_rate"] == 0.0:
        raise TrainingError("learning_rate must be greater than zero")
    training["weight_decay"] = _finite_float(training["weight_decay"], "weight_decay", minimum=0.0)
    training["grad_clip_norm"] = _finite_float(training["grad_clip_norm"], "grad_clip_norm", minimum=0.0)
    if training["grad_clip_norm"] == 0.0:
        raise TrainingError("grad_clip_norm must be greater than zero")
    training["scheduler_factor"] = _finite_float(training["scheduler_factor"], "scheduler_factor")
    if not 0.0 < training["scheduler_factor"] < 1.0:
        raise TrainingError("scheduler_factor must be in (0, 1)")
    if not isinstance(training["amp"], bool):
        raise TrainingError("amp must be boolean")
    if training["amp_dtype"] not in ("float16", "bfloat16"):
        raise TrainingError("amp_dtype must be float16 or bfloat16")
    if training["device"] not in ("auto", "cpu", "cuda") and not (
        isinstance(training["device"], str) and training["device"].startswith("cuda:")
    ):
        raise TrainingError("device must be auto, cpu, cuda, or cuda:N")

    return {
        "dataset": dataset.strip(),
        "language": language.strip(),
        "cache_dir": _absolute_path(config["cache_dir"], "cache_dir"),
        "run_root": _absolute_path(config["run_root"], "run_root"),
        "seeds": list(SEEDS),
        "model": {"name": model["name"], "params": dict(params)},
        "training": training,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and normalize one UTF-8 JSON config."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot read config {}: {}".format(config_path, exc)) from exc
    return normalize_config(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_metadata(
    cache_dir: Path,
    required_modalities: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        raise TrainingError("cache meta.json is missing: {}".format(meta_path))
    try:
        raw_bytes = meta_path.read_bytes()
        meta = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot read cache meta.json: {}".format(exc)) from exc
    if not isinstance(meta, Mapping):
        raise TrainingError("cache meta.json must be an object")
    if not required_modalities or any(modality not in _MODALITIES for modality in required_modalities):
        raise TrainingError("required_modalities must select known non-empty cache streams")
    feature_dims = meta.get("feature_dims")
    sequence_lengths = meta.get("sequence_lengths")
    if meta.get("text_features_ready") is not True:
        raise TrainingError(
            "cache is not marked text_features_ready=true; token IDs are not contextual embeddings"
        )
    if not isinstance(feature_dims, Mapping) or not isinstance(sequence_lengths, Mapping):
        raise TrainingError("meta.json must contain feature_dims and sequence_lengths")
    dims: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for modality in required_modalities:
        if modality not in feature_dims or modality not in sequence_lengths:
            raise TrainingError("cache metadata lacks {} dimensions/length".format(modality))
        dims[modality] = _positive_int(feature_dims[modality], "feature_dims." + modality)
        lengths[modality] = _positive_int(
            sequence_lengths[modality], "sequence_lengths." + modality
        )

    source_hash = meta.get("source_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-fA-F]{64}", source_hash) is None:
        raise TrainingError("meta.json.source_sha256 must be a 64-character SHA-256 hex string")

    split_paths = meta.get("split_paths", {})
    if not isinstance(split_paths, Mapping):
        raise TrainingError("meta.json.split_paths must be a mapping when provided")
    cache_root = cache_dir.resolve()
    resolved_split_paths: dict[str, str] = {}
    for split in ("train", "valid"):
        relative = split_paths.get(split, split)
        if not isinstance(relative, str) or not relative.strip():
            raise TrainingError("split_paths.{} must be a relative path".format(split))
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise TrainingError("split_paths.{} must be relative to cache_dir".format(split))
        resolved = (cache_root / relative_path).resolve()
        try:
            resolved.relative_to(cache_root)
        except ValueError as exc:
            raise TrainingError("split_paths.{} escapes cache_dir".format(split)) from exc
        resolved_split_paths[split] = str(resolved)

    normalized_meta = dict(meta)
    normalized_meta["feature_dims"] = dims
    normalized_meta["sequence_lengths"] = lengths
    normalized_meta["_resolved_split_paths"] = resolved_split_paths
    fingerprint = {
        "meta_sha256": _sha256_bytes(raw_bytes),
        "source_hash": source_hash,
    }
    return normalized_meta, fingerprint


class FeatureSplit(Dataset):
    """Memory-mapped train or validation cache split; never loads test data."""

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        meta: Mapping[str, Any],
        modalities: tuple[str, ...] = _MODALITIES,
    ) -> None:
        if split not in ("train", "valid"):
            raise TrainingError("the trainer can load only train and valid splits")
        if not modalities or any(value not in _MODALITIES for value in modalities):
            raise TrainingError("modalities must select known non-empty cache streams")
        self.modalities = tuple(modalities)
        self.split = split
        self.directory = Path(meta["_resolved_split_paths"][split])
        if not self.directory.is_dir():
            raise TrainingError("cache split directory is missing: {}".format(self.directory))
        self.feature_dims = dict(meta["feature_dims"])
        self.sequence_lengths = dict(meta["sequence_lengths"])

        ids_path = self.directory / "ids.json"
        labels_path = self.directory / "labels.npy"
        try:
            with ids_path.open("r", encoding="utf-8") as handle:
                ids = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError("cannot read {}: {}".format(ids_path, exc)) from exc
        if not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids):
            raise TrainingError("ids.json must be a list of non-empty strings")
        if len(set(ids)) != len(ids):
            raise TrainingError("ids.json contains duplicate IDs")
        if not ids:
            raise TrainingError("{} split must not be empty".format(split))
        self.ids = tuple(ids)

        try:
            labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise TrainingError("cannot load {}: {}".format(labels_path, exc)) from exc
        if labels.ndim != 1 or labels.shape[0] != len(self.ids) or labels.dtype != np.dtype(np.int64):
            raise TrainingError("labels.npy must be int64 [N] and match ids.json")
        if np.any(labels < 0) or np.any(labels > 4):
            raise TrainingError("train/valid labels must be in [0, 4]; unknown -1 is not allowed")
        valid_path = self.directory / "label_valid.npy"
        if valid_path.is_file():
            label_valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
            if (
                label_valid.ndim != 1
                or label_valid.shape[0] != len(self.ids)
                or label_valid.dtype.kind != "b"
                or not np.all(label_valid)
            ):
                raise TrainingError("all train/valid label_valid.npy entries must be true")
        self.labels = labels

        self.features: dict[str, np.ndarray] = {}
        self.lengths: dict[str, np.ndarray] = {}
        for modality in self.modalities:
            feature_path = self.directory / (modality + ".npy")
            length_path = self.directory / (modality + "_lengths.npy")
            try:
                feature = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                modality_lengths = np.load(length_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise TrainingError("cannot load cache arrays for {}: {}".format(modality, exc)) from exc
            expected_tail = (self.sequence_lengths[modality], self.feature_dims[modality])
            if feature.ndim != 3 or tuple(feature.shape[1:]) != expected_tail:
                raise TrainingError(
                    "{} must have shape [N, {}, {}]".format(modality, *expected_tail)
                )
            if feature.shape[0] != len(self.ids) or feature.dtype.kind != "f":
                raise TrainingError("{} must be floating [N,T,D] and match ids".format(modality))
            if not np.isfinite(feature).all():
                raise TrainingError("{} cache contains NaN or Inf".format(modality))
            if (
                modality_lengths.ndim != 1
                or modality_lengths.shape[0] != len(self.ids)
                or modality_lengths.dtype.kind not in "iu"
            ):
                raise TrainingError("{}_lengths.npy must be integer [N]".format(modality))
            if np.any(modality_lengths < 1) or np.any(
                modality_lengths > self.sequence_lengths[modality]
            ):
                raise TrainingError("{} lengths must be in [1, T]".format(modality))
            self.features[modality] = feature
            self.lengths[modality] = modality_lengths

        self.file_fingerprint = {
            "ids_sha256": _sha256_file(ids_path),
            "labels_sha256": _sha256_file(labels_path),
        }

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        feature_values = {
            modality: torch.from_numpy(
                np.array(self.features[modality][index], dtype=np.float32, copy=True)
            )
            for modality in self.modalities
        }
        feature_values["lengths"] = {
            modality: torch.tensor(int(self.lengths[modality][index]), dtype=torch.long)
            for modality in self.modalities
        }
        return {
            "features": feature_values,
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "id": self.ids[index],
        }


def _resolve_device(config_device: str) -> torch.device:
    if config_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TrainingError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise TrainingError("requested CUDA device index is unavailable")
    return device


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "deterministic_algorithms": True,
    }
    if device.type == "cuda":
        environment["device_name"] = torch.cuda.get_device_name(device)
        environment["device_count"] = torch.cuda.device_count()
    else:
        environment["device_name"] = platform.processor() or "CPU"
        environment["device_count"] = 0
    return environment


def _parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak_memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "peak_cuda_allocated_bytes": None,
            "peak_cuda_reserved_bytes": None,
        }
    return {
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _implementation_hash() -> str:
    module_dir = Path(__file__).resolve().parent
    candidates = [module_dir / "training.py", module_dir / "models.py"]
    # Include the existing ALMT adapter whose behavior is part of those runs.
    candidates.append(module_dir / "almt.py")
    vendor_layer = module_dir.parent / 'vendor' / 'almt' / 'almt_layer.py'
    if vendor_layer.is_file():
        candidates.append(vendor_layer)
    digest = hashlib.sha256()
    for path in candidates:
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    _atomic_write(path, payload.encode("utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TrainingError("{} line {} must be a JSON object".format(path, line_number))
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError("cannot read {}: {}".format(path, exc)) from exc
    return rows


def _prune_resource_rows(path: Path, completed_epoch: int) -> list[dict[str, Any]]:
    """Drop timing rows written for an epoch not committed by last.pt."""
    rows = _read_jsonl(path)
    committed = [row for row in rows if int(row.get("epoch", 0)) <= completed_epoch]
    committed.sort(key=lambda row: int(row["epoch"]))
    epochs = [int(row.get("epoch", 0)) for row in committed]
    if epochs != list(range(1, completed_epoch + 1)):
        raise RunIdentityError("resources.jsonl does not cover all checkpointed epochs")
    if committed != rows:
        _atomic_jsonl(path, committed)
    return committed


def _upsert_resource_row(path: Path, row: Mapping[str, Any]) -> None:
    """Write one timing record per epoch without putting time in last.pt."""
    rows = _read_jsonl(path)
    epoch = int(row["epoch"])
    kept = [current for current in rows if int(current.get("epoch", 0)) != epoch]
    kept.append(dict(row))
    kept.sort(key=lambda current: int(current["epoch"]))
    _atomic_jsonl(path, kept)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    os.close(handle_fd)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _rng_state(loader_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": loader_generator.get_state(),
    }


def _restore_rng_state(state: Mapping[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise RunIdentityError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_states)
    loader_generator.set_state(state["loader_generator"])


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older supported torch releases.
        checkpoint = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrainingError("cannot load checkpoint {}: {}".format(path, exc)) from exc
    if not isinstance(checkpoint, dict):
        raise TrainingError("checkpoint must contain a state dictionary")
    return checkpoint


def _move_features(features: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in features.items():
        if key == "lengths":
            if not isinstance(value, Mapping):
                raise TrainingError("batch lengths must be a modality mapping")
            moved[key] = {name: tensor.to(device, non_blocking=True) for name, tensor in value.items()}
        else:
            if not isinstance(value, Tensor):
                raise TrainingError("batch modality features must be torch tensors")
            moved[key] = value.to(device, non_blocking=True)
    return moved


def _autocast_context(device: torch.device, enabled: bool, amp_dtype: str):
    if not enabled or device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_ids: list[str] = []
    all_labels: list[int] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features = _move_features(batch["features"], device)
            labels = batch["label"].to(device, non_blocking=True)
            with _autocast_context(device, amp_enabled, amp_dtype):
                logits = model(features)
            logits = logits.float()
            if tuple(logits.shape) != (labels.shape[0], 5):
                raise TrainingError("model must return logits with shape [B, 5]")
            if not torch.isfinite(logits).all():
                raise TrainingError("model returned non-finite validation logits")
            loss = F.cross_entropy(logits, labels, reduction="sum")
            predictions = torch.argmax(logits, dim=1)
            total_loss += float(loss.item())
            total_correct += int((predictions == labels).sum().item())
            total_count += int(labels.numel())
            all_ids.extend(str(value) for value in batch["id"])
            all_labels.extend(int(value) for value in labels.cpu().tolist())
            all_logits.append(logits.cpu().numpy().astype(np.float64, copy=False))
    if total_count == 0:
        raise TrainingError("validation loader produced no examples")
    logits_matrix = np.concatenate(all_logits, axis=0)
    probabilities = torch.softmax(torch.from_numpy(logits_matrix), dim=1).numpy()
    return {
        "loss": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "ids": all_ids,
        "labels": all_labels,
        "logits": logits_matrix,
        "probabilities": probabilities,
    }


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device,
    *,
    amp_enabled: bool,
    amp_dtype: str,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        features = _move_features(batch["features"], device)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled, amp_dtype):
            logits = model(features)
            if tuple(logits.shape) != (labels.shape[0], 5):
                raise TrainingError("model must return logits with shape [B, 5]")
            loss = F.cross_entropy(logits.float(), labels)
        if not torch.isfinite(loss):
            raise TrainingError("training loss is not finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        count = int(labels.numel())
        total_loss += float(loss.detach().item()) * count
        total_count += count
    if total_count == 0:
        raise TrainingError("training loader produced no examples")
    return total_loss / total_count


def _checkpoint_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    loader_generator: torch.Generator,
    identity_sha256: str,
    seed: int,
    epoch_completed: int,
    next_epoch: int,
    best_accuracy: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "identity_sha256": identity_sha256,
        "seed": seed,
        "epoch_completed": epoch_completed,
        "next_epoch": next_epoch,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "completed": completed,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": _rng_state(loader_generator),
    }


def _validation_rows(evaluation: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    probabilities = evaluation["probabilities"]
    for sample_id, label, row in zip(evaluation["ids"], evaluation["labels"], probabilities):
        rows.append(
            {
                "ID": sample_id,
                "split": "valid",
                "label": int(label),
                "class_order": list(_CLASS_ORDER),
                "probabilities": [float(value) for value in row.tolist()],
                "argmax": _CLASS_ORDER[int(np.argmax(row))],
                "temperature": 1.0,
            }
        )
    return rows


def train_run(
    config: Mapping[str, Any],
    seed: int,
    run_dir: str | Path,
    *,
    resume: bool = False,
    max_epochs_this_call: int | None = None,
) -> dict[str, Any]:
    """Train or resume one seed in an immutable run directory.

    ``max_epochs_this_call`` is an interruption hook for smoke/resume testing;
    it does not alter the hashed config or mark a partial run complete.
    """
    normalized = normalize_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise TrainingError("seed must be one of 13, 42, 2026")
    if max_epochs_this_call is not None:
        _positive_int(max_epochs_this_call, "max_epochs_this_call")

    device = _resolve_device(normalized["training"]["device"])
    environment = _runtime_environment(device)
    cache_dir = Path(normalized["cache_dir"])
    active_modalities = ("text",) if normalized["model"]["name"] == "text_mlp" else _MODALITIES
    meta, meta_fingerprint = _cache_metadata(cache_dir, active_modalities)
    train = FeatureSplit(cache_dir, "train", meta, modalities=active_modalities)
    valid = FeatureSplit(cache_dir, "valid", meta, modalities=active_modalities)
    for modality in active_modalities:
        if train.features[modality].shape[1:] != valid.features[modality].shape[1:]:
            raise TrainingError("train/valid {} shapes do not match".format(modality))
    cache_fingerprint = {
        **meta_fingerprint,
        "train": train.file_fingerprint,
        "valid": valid.file_fingerprint,
    }
    config_sha256 = _sha256_bytes(_canonical_json(normalized))
    code_sha256 = _implementation_hash()
    identity_payload = {
        "config_sha256": config_sha256,
        "cache_fingerprint": cache_fingerprint,
        "implementation_sha256": code_sha256,
        "environment": environment,
        "seed": seed,
    }
    identity_sha256 = _sha256_bytes(_canonical_json(identity_payload))
    manifest = {
        "format_version": 1,
        "identity_sha256": identity_sha256,
        "config_sha256": config_sha256,
        "cache_fingerprint": cache_fingerprint,
        "implementation_sha256": code_sha256,
        "environment": environment,
        "seed": seed,
        "model": normalized["model"]["name"],
        "dataset": normalized["dataset"],
        "language": normalized["language"],
        "class_order": list(_CLASS_ORDER),
        "padding_behavior": (
            "ALMT adapter consumes every fixed sequence position; lengths are not masked."
            if normalized["model"]["name"] == "almt"
            else "Text classifier uses configured CLS or length-masked mean pooling."
        ),
    }

    output = Path(run_dir).expanduser().resolve()
    manifest_path = output / "run_manifest.json"
    config_path = output / "config.json"
    environment_path = output / "environment.json"
    last_path = output / "last.pt"
    best_path = output / "best.pt"
    metrics_path = output / "metrics.jsonl"
    validation_path = output / "validation_predictions.jsonl"
    resources_path = output / "resources.jsonl"
    model_summary_path = output / "model_summary.json"

    if resume:
        if not output.is_dir() or not manifest_path.is_file() or not last_path.is_file():
            raise RunIdentityError("resume requires an existing run_manifest.json and last.pt")
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
            existing_environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunIdentityError("cannot read existing run manifest") from exc
        if (
            existing_manifest != manifest
            or _sha256_bytes(_canonical_json(existing_config)) != config_sha256
            or existing_environment != environment
        ):
            raise RunIdentityError(
                "resume identity differs (config, cache/source hash, code, environment, or seed); "
                "the existing run was left unchanged"
            )
        checkpoint = _load_checkpoint(last_path)
        if checkpoint.get("identity_sha256") != identity_sha256:
            raise RunIdentityError("last.pt identity does not match the run manifest")
        if int(checkpoint.get("epoch_completed", 0)) > 0 and (
            not best_path.is_file() or not validation_path.is_file()
        ):
            raise RunIdentityError("run is missing its selected checkpoint or validation outputs")
    else:
        if output.exists():
            if not output.is_dir():
                raise RunIdentityError("run path exists and is not a directory")
            if any(output.iterdir()):
                raise RunIdentityError("run directory is non-empty; use exact-config --resume")
        checkpoint = None

    # Every run begins from a clean deterministic seed. On resume, saved RNG
    # state is restored after constructing objects, immediately before epoch
    # iteration, so initialization does not perturb the resumed stream.
    _seed_everything(seed)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed + 1009)
    train_loader = DataLoader(
        train,
        batch_size=normalized["training"]["batch_size"],
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )
    valid_loader = DataLoader(
        valid,
        batch_size=normalized["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )

    model = build_model(
        normalized["model"]["name"],
        meta["feature_dims"],
        meta["sequence_lengths"],
        normalized["model"]["params"],
    ).to(device)
    training = normalized["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=training["scheduler_factor"],
        patience=training["scheduler_patience"],
    )
    amp_enabled = bool(training["amp"] and device.type == "cuda")
    scaler_enabled = amp_enabled and training["amp_dtype"] == "float16"
    scaler = torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    parameter_counts = _parameter_summary(model)
    model_summary = {
        "identity_sha256": identity_sha256,
        "model": normalized["model"]["name"],
        "class_order": list(_CLASS_ORDER),
        "feature_dims": dict(meta["feature_dims"]),
        "sequence_lengths": dict(meta["sequence_lengths"]),
        **parameter_counts,
    }

    if not resume:
        # Prepare all cache/model/runtime dependencies before claiming a run
        # directory, so an import or config error does not strand a folder
        # that appears resumable but has no checkpoint.
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, normalized)
        _atomic_json(environment_path, environment)
        _atomic_json(manifest_path, manifest)
        _atomic_json(model_summary_path, model_summary)
    else:
        try:
            existing_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunIdentityError("cannot read existing model_summary.json") from exc
        if existing_summary != model_summary:
            raise RunIdentityError("model parameter summary differs; existing run was left unchanged")

    max_epochs = training["max_epochs"]
    best_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    next_epoch = 0
    epoch_completed = 0
    completed = False

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        next_epoch = int(checkpoint["next_epoch"])
        epoch_completed = int(checkpoint["epoch_completed"])
        best_accuracy = float(checkpoint["best_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        completed = bool(checkpoint["completed"])
        _restore_rng_state(checkpoint["rng_state"], loader_generator)
        _prune_resource_rows(resources_path, epoch_completed)
        _atomic_jsonl(metrics_path, history)
        if completed:
            return {
                "run_dir": str(output),
                "seed": seed,
                "best_epoch": best_epoch,
                "best_accuracy": best_accuracy,
                "completed": True,
                "resumed": True,
            }
    else:
        # Initial checkpoint permits restart after an interruption before the
        # first epoch; it resumes at epoch zero with the same initialized state.
        initial = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=loader_generator,
            identity_sha256=identity_sha256,
            seed=seed,
            epoch_completed=0,
            next_epoch=0,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
            completed=False,
        )
        _atomic_torch_save(last_path, initial)

    epochs_this_call = 0
    while next_epoch < max_epochs:
        if max_epochs_this_call is not None and epochs_this_call >= max_epochs_this_call:
            break

        epoch_number = next_epoch + 1
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loss = _train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=training["amp_dtype"],
            grad_clip_norm=training["grad_clip_norm"],
        )
        _synchronize(device)
        train_seconds = time.perf_counter() - train_start
        valid_start = time.perf_counter()
        evaluation = _evaluate(
            model,
            valid_loader,
            device,
            amp_enabled=amp_enabled,
            amp_dtype=training["amp_dtype"],
        )
        _synchronize(device)
        valid_seconds = time.perf_counter() - valid_start
        train_valid_seconds = time.perf_counter() - epoch_start
        memory = _cuda_peak_memory(device)
        scheduler.step(evaluation["loss"])
        improved = evaluation["accuracy"] > best_accuracy + training["min_delta"]
        if improved:
            best_accuracy = float(evaluation["accuracy"])
            best_epoch = epoch_number
            stale_epochs = 0
        else:
            stale_epochs += 1

        history.append(
            {
                "epoch": epoch_number,
                "train_loss": train_loss,
                "valid_nll": float(evaluation["loss"]),
                "valid_accuracy": float(evaluation["accuracy"]),
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        next_epoch = epoch_number
        epoch_completed = epoch_number
        stop_early = stale_epochs >= training["patience"]
        completed = stop_early or next_epoch >= max_epochs

        # Save the selected checkpoint before last.pt. If interrupted in
        # between, the epoch is replayed from the previous exact boundary.
        if improved:
            best_state = {
                "format_version": 1,
                "identity_sha256": identity_sha256,
                "seed": seed,
                "epoch": epoch_number,
                "valid_accuracy": float(evaluation["accuracy"]),
                "valid_nll": float(evaluation["loss"]),
                "model_state": model.state_dict(),
            }
            _atomic_torch_save(best_path, best_state)
            _atomic_jsonl(validation_path, _validation_rows(evaluation))

        # Resource measurements are deliberately kept in their own JSONL and
        # out of history/checkpoints so wall-clock noise cannot affect resume.
        _upsert_resource_row(
            resources_path,
            {
                "epoch": epoch_number,
                "device": str(device),
                "train_seconds": train_seconds,
                "valid_seconds": valid_seconds,
                "train_valid_wall_seconds": train_valid_seconds,
                "train_samples_processed": len(train),
                "valid_samples_processed": len(valid),
                "train_samples_per_second": len(train) / train_seconds if train_seconds > 0 else None,
                "valid_samples_per_second": len(valid) / valid_seconds if valid_seconds > 0 else None,
                **memory,
            },
        )

        last_state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            loader_generator=loader_generator,
            identity_sha256=identity_sha256,
            seed=seed,
            epoch_completed=epoch_completed,
            next_epoch=next_epoch,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
            completed=completed,
        )
        _atomic_torch_save(last_path, last_state)
        _atomic_jsonl(metrics_path, history)
        epochs_this_call += 1
        if completed:
            break

    return {
        "run_dir": str(output),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "last_epoch": epoch_completed,
        "completed": completed,
        "resumed": resume,
    }


def run_from_config(
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run one requested seed or the fixed three-seed comparison set."""
    normalized = normalize_config(config)
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS
    ):
        raise TrainingError("seed must be one of 13, 42, 2026")
    if resume and seed is None:
        raise TrainingError("--resume requires selecting exactly one --seed")
    selected = (seed,) if seed is not None else SEEDS
    root = Path(normalized["run_root"])
    results = []
    for current_seed in selected:
        run_dir = (
            root
            / normalized["dataset"]
            / normalized["model"]["name"]
            / ("seed_{}".format(current_seed))
        )
        results.append(
            train_run(normalized, current_seed, run_dir, resume=resume)
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON feature-training config")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Run one of the registered seeds")
    parser.add_argument("--resume", action="store_true", help="Resume the exact selected run")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results = run_from_config(config, seed=args.seed, resume=args.resume)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0

