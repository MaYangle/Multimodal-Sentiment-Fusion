"""Stage-one BC-MLF BI_ENC pretraining on the canonical JevAffect cache.

This driver ports only the native BI_ENC raw-label L1 stage. It does not load
test, download a language model, or invent/fill a missing AV initialization.
The saved best file is a flat ``Model.*`` state dict accepted by the vendored
``AV_Enc.from_pretrained`` strict loader.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from jevaffect.cache import load_cached_split
from jevaffect.training import (
    RunIdentityError,
    TrainingError,
    _atomic_json,
    _atomic_jsonl,
    _atomic_torch_save,
    _cuda_peak_memory,
    _prune_resource_rows,
    _synchronize,
    _upsert_resource_row,
)


SEEDS = (13, 42, 2026)
_MODALITIES = ("text", "audio", "vision")


class BCMLFTrainingError(TrainingError):
    """Raised when the BC-MLF Stage 1 config, cache, or run is invalid."""


class _AttrDict(dict):
    """Minimal EasyDict-compatible mapping used by the unchanged BC-MLF model."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _attrify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _AttrDict({key: _attrify(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_attrify(item) for item in value]
    return value


class _RegressionSplit(Dataset):
    """Raw-score Stage 1 view of one already verified train/valid cache split."""

    def __init__(self, data: Mapping[str, Any], split: str) -> None:
        if split not in ("train", "valid"):
            raise BCMLFTrainingError("BI_ENC may load only train and valid splits")
        self.split = split
        ids = data.get("ids")
        if not isinstance(ids, list) or not ids or any(
            not isinstance(value, str) or not value for value in ids
        ):
            raise BCMLFTrainingError("cached canonical IDs must be a non-empty list[str]")
        if len(set(ids)) != len(ids):
            raise BCMLFTrainingError("cache contains duplicate canonical IDs")
        self.ids = tuple(ids)

        labels = data.get("labels")
        label_valid = data.get("label_valid")
        raw_y = data.get("raw_labels")
        if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64):
            raise BCMLFTrainingError("labels.npy must be int64 manifest class indices")
        if not isinstance(label_valid, np.ndarray) or label_valid.dtype != np.dtype(np.bool_):
            raise BCMLFTrainingError("label_valid.npy must be a boolean vector")
        if labels.shape != (len(ids),) or label_valid.shape != (len(ids),):
            raise BCMLFTrainingError("labels, label_valid, and IDs must have matching lengths")
        if not np.all(label_valid) or np.any(labels < 0) or np.any(labels > 4):
            raise BCMLFTrainingError("all train/valid labels must be mapped to classes 0..4")
        if not isinstance(raw_y, np.ndarray) or raw_y.ndim == 0 or raw_y.shape[0] != len(ids):
            raise BCMLFTrainingError("raw_labels.npy must align with canonical IDs")
        if raw_y.dtype.kind not in "fiu":
            raise BCMLFTrainingError("raw_labels.npy must contain numeric source scores")
        raw_y = np.asarray(raw_y, dtype=np.float32).reshape(len(ids), -1)
        if raw_y.shape[1] != 1 or not np.isfinite(raw_y).all():
            raise BCMLFTrainingError("train/valid raw labels must be one finite score per sample")
        self.raw_y = raw_y[:, 0]

        self.features: dict[str, np.ndarray] = {}
        for modality in ("audio", "vision"):
            feature = data.get(modality)
            if not isinstance(feature, np.ndarray) or feature.ndim != 3:
                raise BCMLFTrainingError("{} features must have shape [N,T,D]".format(modality))
            if feature.shape[0] != len(ids) or feature.dtype.kind != "f":
                raise BCMLFTrainingError("{} features must be floating and align with IDs".format(modality))
            if not np.isfinite(feature).all():
                raise BCMLFTrainingError("{} features contain non-finite values".format(modality))
            self.features[modality] = feature

        self.cache_meta = data.get("meta")
        if not isinstance(self.cache_meta, Mapping):
            raise BCMLFTrainingError("cache split must return verified meta.json as 'meta'")
        for key in ("source_sha256", "feature_dims", "sequence_lengths"):
            if key not in self.cache_meta:
                raise BCMLFTrainingError("canonical meta.json is missing {}".format(key))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "audio": torch.from_numpy(
                np.array(self.features["audio"][index], dtype=np.float32, copy=True)
            ),
            "vision": torch.from_numpy(
                np.array(self.features["vision"][index], dtype=np.float32, copy=True)
            ),
            "raw_y": torch.tensor(float(self.raw_y[index]), dtype=torch.float32),
            "label": torch.tensor(int(index), dtype=torch.long),
            "id": self.ids[index],
        }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BCMLFTrainingError("{} must be a non-empty path".format(name))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def normalize_stage1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {"repo_root", "cache_dir", "run_root", "bienc_config", "seeds", "training"}
    if not isinstance(config, Mapping) or set(config) != required:
        raise BCMLFTrainingError("Stage 1 config keys must be exactly {}".format(sorted(required)))
    repo_root = Path(config["repo_root"]).expanduser().resolve()
    if not (repo_root / "vendor" / "bcmlf").is_dir():
        raise BCMLFTrainingError("repo_root does not contain vendor/bcmlf")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or tuple(seeds) != SEEDS:
        raise BCMLFTrainingError("seeds must be exactly [13, 42, 2026]")
    training = config["training"]
    if not isinstance(training, Mapping):
        raise BCMLFTrainingError("training must be a JSON object")
    required_training = {
        "max_epochs", "batch_size", "accumulation_steps", "early_stop",
        "scheduler_patience", "learning_rate", "grad_clip_value",
        "weight_decay", "device",
    }
    if set(training) != required_training:
        raise BCMLFTrainingError(
            "Stage 1 training keys must be exactly {}".format(sorted(required_training))
        )
    for key in ("max_epochs", "batch_size", "accumulation_steps", "early_stop"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BCMLFTrainingError("training.{} must be a positive integer".format(key))
    if (
        isinstance(training["scheduler_patience"], bool)
        or not isinstance(training["scheduler_patience"], int)
        or training["scheduler_patience"] < 0
    ):
        raise BCMLFTrainingError("scheduler_patience must be a non-negative integer")
    for key, minimum in (("learning_rate", 0.0), ("grad_clip_value", -1.0), ("weight_decay", 0.0)):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise BCMLFTrainingError("training.{} must be finite numeric".format(key))
        if float(value) < minimum or (key == "learning_rate" and float(value) == 0.0):
            raise BCMLFTrainingError("training.{} is out of range".format(key))
    device = training["device"]
    if device not in ("cpu", "auto", "cuda") and not (
        isinstance(device, str) and device.startswith("cuda:")
    ):
        raise BCMLFTrainingError("training.device must be cpu, auto, cuda, or cuda:N")
    return {
        "repo_root": str(repo_root),
        "cache_dir": str(_resolve_path(repo_root, config["cache_dir"], "cache_dir")),
        "run_root": str(_resolve_path(repo_root, config["run_root"], "run_root")),
        "bienc_config": str(_resolve_path(repo_root, config["bienc_config"], "bienc_config")),
        "seeds": list(SEEDS),
        "training": {key: training[key] for key in sorted(required_training)},
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BCMLFTrainingError("cannot read {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise BCMLFTrainingError("{} must contain a JSON object".format(path))
    return value


def _load_cache_splits(cache_dir: Path) -> tuple[_RegressionSplit, _RegressionSplit, dict[str, Any]]:
    # The training adapter intentionally calls the project cache API and never
    # asks it to load the test split. BI_ENC needs audio/vision and raw labels;
    # it does not require contextual text features or raw transcripts.
    train_data = load_cached_split(
        cache_dir, "train", require_text_features=False, require_raw_text=False
    )
    valid_data = load_cached_split(
        cache_dir, "valid", require_text_features=False, require_raw_text=False
    )
    train = _RegressionSplit(train_data, "train")
    valid = _RegressionSplit(valid_data, "valid")
    if train.cache_meta["source_sha256"] != valid.cache_meta["source_sha256"]:
        raise BCMLFTrainingError("train and valid cache source hashes differ")
    if train.cache_meta["feature_dims"] != valid.cache_meta["feature_dims"]:
        raise BCMLFTrainingError("train and valid cache feature dimensions differ")
    if train.cache_meta["sequence_lengths"] != valid.cache_meta["sequence_lengths"]:
        raise BCMLFTrainingError("train and valid cache sequence lengths differ")
    cache_identity = {
        "source_sha256": train.cache_meta["source_sha256"],
        "meta_sha256": _sha256_bytes(_canonical_json(train.cache_meta)),
        "train_ids_sha256": _sha256_bytes(_canonical_json(list(train.ids))),
        "valid_ids_sha256": _sha256_bytes(_canonical_json(list(valid.ids))),
    }
    return train, valid, cache_identity


def _load_bienc_args(
    config_path: Path,
    repo_root: Path,
    train: _RegressionSplit,
    device: torch.device,
    training: Mapping[str, Any],
) -> _AttrDict:
    from vendor.bcmlf.adapters.config_paths import resolve_bcmlf_paths

    source = _read_json(config_path)
    resolved = resolve_bcmlf_paths(source, repo_root=repo_root)
    try:
        dataset = dict(resolved["datasetCommonParams"]["sims"]["unaligned"])
        model_common = dict(resolved["bienc"]["commonParams"])
        model_dataset = dict(resolved["bienc"]["datasetParams"]["sims"])
    except (KeyError, TypeError) as exc:
        raise BCMLFTrainingError("bienc config lacks the expected sims/unaligned sections") from exc

    if model_common.get("need_model_aligned", False):
        raise BCMLFTrainingError("Stage 1 cache inputs are unaligned; native BI_ENC config must keep need_model_aligned=false")
    meta = train.cache_meta
    modalities = ("text", "audio", "vision")
    actual_dims = [int(meta["feature_dims"][name]) for name in modalities]
    actual_lengths = [int(meta["sequence_lengths"][name]) for name in modalities]
    for key, actual in (("feature_dims", actual_dims), ("seq_lens", actual_lengths)):
        expected = dataset.get(key)
        if expected is not None and list(expected) != actual:
            raise BCMLFTrainingError(
                "cache {} {} does not match native BI_ENC config {}".format(key, actual, expected)
            )

    merged: dict[str, Any] = {}
    merged.update(dataset)
    merged.update(model_common)
    merged.update(model_dataset)
    merged.update(
        {
            "model_name": "bienc",
            "dataset_name": "sims",
            "train_mode": "regression",
            "feature_dims": actual_dims,
            "seq_lens": actual_lengths,
            "train_samples": len(train),
            "device": device,
            "max_epochs": training["max_epochs"],
            "update_epochs": training["accumulation_steps"],
            "learning_rate": float(training["learning_rate"]),
            "grad_clip": float(training["grad_clip_value"]),
            "early_stop": training["early_stop"],
            # Source BI_ENC uses this as ReduceLROnPlateau patience.
            "patience": training["scheduler_patience"],
            "KeyEval": "MAE",
            "scheduler": False,
        }
    )
    av_config = merged.get("av_enc")
    if not isinstance(av_config, Mapping):
        raise BCMLFTrainingError("BI_ENC config must include av_enc")
    if int(av_config.get("maxlen", -1)) != actual_lengths[0]:
        raise BCMLFTrainingError("BI_ENC av_enc.maxlen must equal cached text sequence length")
    return _attrify(merged)


class _BIENCPrefixModel(nn.Module):
    """Matches the source AMIO checkpoint prefix expected by AV_Enc loader."""

    def __init__(self, args: _AttrDict) -> None:
        super().__init__()
        from vendor.bcmlf.models.singleTask.BI_ENC import BI_ENC

        self.Model = BI_ENC(args)

    def forward(self, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        return self.Model(None, audio, vision)


def _implementation_hash(repo_root: Path, bienc_config: Path) -> str:
    candidates = (
        Path(__file__).resolve(),
        repo_root / "vendor" / "bcmlf" / "models" / "singleTask" / "BI_ENC.py",
        repo_root / "vendor" / "bcmlf" / "models" / "singleTask" / "MMGPT.py",
        repo_root / "vendor" / "bcmlf" / "models" / "subNets" / "transformers_encoder" / "transformer.py",
        repo_root / "vendor" / "bcmlf" / "transformations" / "pool" / "mmaug.py",
        repo_root / "jevaffect" / "cache.py",
        repo_root / "jevaffect" / "training.py",
        bienc_config,
    )
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            raise BCMLFTrainingError("implementation identity file is missing: {}".format(path))
        digest.update(path.relative_to(repo_root).as_posix().encode("utf-8") if path.is_relative_to(repo_root) else path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _numpy_rng_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _validate_stage1_arrays(train: _RegressionSplit, valid: _RegressionSplit) -> None:
    for split in (train, valid):
        for modality in ("audio", "vision"):
            if split.features[modality].ndim != 3:
                raise BCMLFTrainingError("{} {} must be [N,T,D]".format(split.split, modality))
    if train.features["audio"].shape[1:] != valid.features["audio"].shape[1:]:
        raise BCMLFTrainingError("train and valid audio sequence shapes differ")
    if train.features["vision"].shape[1:] != valid.features["vision"].shape[1:]:
        raise BCMLFTrainingError("train and valid vision sequence shapes differ")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BCMLFTrainingError("CUDA requested but not available")
    return device


def _seed_stage1(seed: int, device: torch.device) -> None:
    """Seed CPU RNGs without touching CUDA when the run is CPU-only."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _stage1_rng_state(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
        "torch_cuda": [],
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_stage1_rng_state(
    state: Mapping[str, Any],
    generator: torch.Generator,
    device: torch.device,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["loader_generator"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if device.type != "cuda":
            raise RunIdentityError("CUDA RNG checkpoint cannot resume on CPU")
        torch.cuda.set_rng_state_all(cuda_states)


def _stage1_environment(device: torch.device) -> dict[str, Any]:
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
    }
    if device.type == "cuda":
        environment["device_name"] = torch.cuda.get_device_name(device)
        environment["device_count"] = torch.cuda.device_count()
        environment["cudnn_version"] = torch.backends.cudnn.version()
    else:
        environment["device_name"] = platform.processor() or "CPU"
        environment["device_count"] = 0
        environment["cudnn_version"] = None
    return environment


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise BCMLFTrainingError("checkpoint must be a dict")
    return value


def _copy_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().to(device="cpu").clone() for key, value in model.state_dict().items()}


def _stabilize_sinusoidal_dtype_buffers(model: nn.Module) -> None:
    """Initialize source dtype-reference buffers that otherwise contain allocator bytes.

    The vendored sinusoidal embedding uses `_float_tensor` only for dtype/device
    conversion. Zeroing this non-learned reference leaves outputs unchanged,
    while making exact-resume state comparisons reproducible.
    """
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            if name.endswith("._float_tensor"):
                buffer.zero_()


def _group_sample_count(total: int, batch_size: int, num_batches: int, group_start: int, group_size: int) -> int:
    group_end = min(group_start + group_size, num_batches)
    return sum(min(batch_size, total - index * batch_size) for index in range(group_start, group_end))


def _train_epoch(
    model: _BIENCPrefixModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_size: int,
    accumulation_steps: int,
    grad_clip_value: float,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    optimizer_updates = 0
    n_batches = len(loader)
    n_examples = len(loader.dataset)
    for batch_index, batch in enumerate(loader):
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_samples = _group_sample_count(
            n_examples, batch_size, n_batches, group_start, accumulation_steps
        )
        if batch_index == group_start:
            optimizer.zero_grad(set_to_none=True)
        audio = batch["audio"].to(device, non_blocking=True)
        vision = batch["vision"].to(device, non_blocking=True)
        raw_y = batch["raw_y"].to(device, non_blocking=True)
        outputs = model(audio, vision)
        predictions = outputs.reshape(-1)
        loss = F.l1_loss(predictions, raw_y, reduction="mean")
        sample_count = int(raw_y.shape[0])
        # Source BI_ENC accumulates mean losses without normalization. Scale
        # each microbatch by its true share of the effective batch, including
        # a short final group, so the gradient is an example-weighted L1 mean.
        (loss * (sample_count / group_samples)).backward()
        total_loss += float(loss.detach().item()) * sample_count
        total_samples += sample_count

        group_end = min(group_start + accumulation_steps, n_batches)
        if batch_index + 1 == group_end:
            if grad_clip_value != -1.0:
                nn.utils.clip_grad_value_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    grad_clip_value,
                )
            optimizer.step()
            optimizer_updates += 1
    if total_samples == 0:
        raise BCMLFTrainingError("empty BI_ENC training split")
    return total_loss / total_samples, optimizer_updates


@torch.no_grad()
def _evaluate(model: _BIENCPrefixModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    sum_abs_error = 0.0
    n_samples = 0
    predictions: list[float] = []
    raw_labels: list[float] = []
    ids: list[str] = []
    for batch in loader:
        audio = batch["audio"].to(device, non_blocking=True)
        vision = batch["vision"].to(device, non_blocking=True)
        raw_y = batch["raw_y"].to(device, non_blocking=True)
        output = model(audio, vision).reshape(-1)
        if output.shape != raw_y.shape or not torch.isfinite(output).all():
            raise BCMLFTrainingError("BI_ENC returned invalid validation predictions")
        absolute = (output - raw_y).abs()
        sum_abs_error += float(absolute.sum().item())
        n_samples += int(raw_y.numel())
        predictions.extend(float(value) for value in output.detach().cpu().tolist())
        raw_labels.extend(float(value) for value in raw_y.detach().cpu().tolist())
        ids.extend(str(value) for value in batch["id"])
    if n_samples == 0:
        raise BCMLFTrainingError("empty BI_ENC validation split")
    return {
        "mae": sum_abs_error / n_samples,
        "predictions": predictions,
        "raw_labels": raw_labels,
        "ids": ids,
        "n_samples": n_samples,
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise BCMLFTrainingError("BI_ENC checkpoint must be a dict")
    return value


def _checkpoint_state(
    model: _BIENCPrefixModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    generator: torch.Generator,
    device: torch.device,
    identity: str,
    seed: int,
    epoch: int,
    next_epoch: int,
    best_mae: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    completed: bool,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "identity_sha256": identity,
        "seed": seed,
        "epoch_completed": epoch,
        "next_epoch": next_epoch,
        "best_mae": best_mae,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "completed": completed,
        "model_state": _copy_state_to_cpu(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "rng_state": _stage1_rng_state(generator, device),
    }


def _valid_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "ID": sample_id,
            "split": "valid",
            "raw_label": float(raw_y),
            "prediction": float(prediction),
            "absolute_error": abs(float(prediction) - float(raw_y)),
        }
        for sample_id, raw_y, prediction in zip(
            result["ids"], result["raw_labels"], result["predictions"]
        )
    ]


def train_bienc_stage(
    config: Mapping[str, Any],
    seed: int,
    run_dir: str | Path,
    *,
    resume: bool = False,
    max_epochs_this_call: int | None = None,
    data_loader=None,
) -> dict[str, Any]:
    """Run one raw-label BI_ENC seed using train/valid only.

    `data_loader` injection exists for deterministic CPU fixtures. Production
    calls use `jevaffect.cache.load_cached_split` with the canonical cache API.
    """
    normalized = normalize_stage1_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise BCMLFTrainingError("seed must be one of 13, 42, 2026")
    if max_epochs_this_call is not None and (
        isinstance(max_epochs_this_call, bool)
        or not isinstance(max_epochs_this_call, int)
        or max_epochs_this_call <= 0
    ):
        raise BCMLFTrainingError("max_epochs_this_call must be a positive integer")
    if data_loader is None:
        data_loader = load_cached_split

    device = _resolve_device(normalized["training"]["device"])
    train_data = data_loader(
        Path(normalized["cache_dir"]),
        "train",
        require_text_features=False,
        require_raw_text=False,
    )
    valid_data = data_loader(
        Path(normalized["cache_dir"]),
        "valid",
        require_text_features=False,
        require_raw_text=False,
    )
    train = _RegressionSplit(train_data, "train")
    valid = _RegressionSplit(valid_data, "valid")
    _validate_stage1_arrays(train, valid)
    if train.cache_meta["source_sha256"] != valid.cache_meta["source_sha256"]:
        raise BCMLFTrainingError("train/valid source hashes differ")
    meta = train.cache_meta
    base_config_path = Path(normalized["bienc_config"])

    # The base config and cache must agree on tensor layout before model
    # construction; otherwise the strict Stage 2 AV_Enc transfer would fail.
    _seed_stage1(seed, device)
    args = _load_bienc_args(
        base_config_path,
        Path(normalized["repo_root"]),
        train,
        device,
        normalized["training"],
    )

    identity_config = {
        "config": normalized,
        "source_model_config_sha256": _sha256_file(base_config_path),
        "cache_source_sha256": meta["source_sha256"],
        "cache_meta_sha256": _sha256_bytes(_canonical_json(meta)),
        "train_ids_sha256": _sha256_bytes(_canonical_json(list(train.ids))),
        "valid_ids_sha256": _sha256_bytes(_canonical_json(list(valid.ids))),
        "train_raw_labels_sha256": _numpy_rng_hash(train.raw_y),
        "valid_raw_labels_sha256": _numpy_rng_hash(valid.raw_y),
        "implementation_sha256": _implementation_hash(
            Path(normalized["repo_root"]), base_config_path
        ),
        "environment": _stage1_environment(device),
        "seed": seed,
    }
    identity = _sha256_bytes(_canonical_json(identity_config))
    output = Path(run_dir).expanduser().resolve()
    manifest = {
        "format_version": 1,
        "identity_sha256": identity,
        "seed": seed,
        "stage": "BI_ENC_raw_label_L1_pretrain",
        "dataset": "simsv2",
        "language": "zh",
        "class_order": ["SNEG", "WNEG", "NEUT", "WPOS", "SPOS"],
        "cache_source_sha256": meta["source_sha256"],
        "cache_meta_sha256": identity_config["cache_meta_sha256"],
        "implementation_sha256": identity_config["implementation_sha256"],
        "environment": identity_config["environment"],
        "feature_dims": dict(meta["feature_dims"]),
        "sequence_lengths": dict(meta["sequence_lengths"]),
        "parameter_counts": None,
        "adaptations": [
            "correctly normalize microbatch gradients by examples in the effective batch",
            "perform optimizer step for a short final accumulation group",
            "zero unused sinusoidal dtype-reference buffers for deterministic transfer",
            "use valid raw-score MAE for checkpoint selection; test is never loaded",
        ],
    }
    run_manifest_path = output / "run_manifest.json"
    config_path = output / "config.json"
    environment_path = output / "environment.json"
    model_summary_path = output / "model_summary.json"
    best_path = output / "bienc_best_encoder.pth"
    last_path = output / "last.pt"
    metrics_path = output / "metrics.jsonl"
    validation_path = output / "validation_predictions.jsonl"
    resources_path = output / "resources.jsonl"

    if resume:
        if not output.is_dir() or not run_manifest_path.is_file() or not last_path.is_file():
            raise RunIdentityError("resume requires an existing run_manifest.json and last.pt")
        existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        expected_manifest = dict(manifest)
        expected_manifest["parameter_counts"] = existing_manifest.get("parameter_counts")
        if existing_manifest != expected_manifest:
            raise RunIdentityError("Stage 1 resume identity differs; existing run was not modified")
        checkpoint = _load_checkpoint(last_path)
        if checkpoint.get("identity_sha256") != identity:
            raise RunIdentityError("BI_ENC last.pt identity mismatch")
        if int(checkpoint.get("epoch_completed", 0)) > 0 and (
            not best_path.is_file() or not validation_path.is_file()
        ):
            raise RunIdentityError("BI_ENC run is missing best transfer weights or valid outputs")
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise RunIdentityError("run directory exists and is non-empty; resume exact config instead")
        checkpoint = None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 2711)
    train_loader = DataLoader(
        train,
        batch_size=normalized["training"]["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid,
        batch_size=normalized["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    model = _BIENCPrefixModel(args).to(device)
    _stabilize_sinusoidal_dtype_buffers(model)
    training = normalized["training"]
    # Source BI_ENC calls AdamW(model.parameters(), lr=...), inheriting the
    # PyTorch default weight_decay=0.01; the source config's weight_decay field
    # is not passed by that trainer. Keep the effective source behavior explicit.
    source_weight_decay = float(training["weight_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=source_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=training["scheduler_patience"],
    )
    parameter_counts = {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }
    summary = {
        "identity_sha256": identity,
        "parameter_counts": parameter_counts,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": source_weight_decay,
            "weight_decay_note": "explicit project value; 0.01 reproduces the effective upstream AdamW default",
        },
        "micro_batch_size": training["batch_size"],
        "accumulation_steps": training["accumulation_steps"],
        "nominal_effective_batch_size": (
            training["batch_size"] * training["accumulation_steps"]
        ),
        "clip": {"kind": "value", "value": float(training["grad_clip_value"])},
    }
    manifest["parameter_counts"] = parameter_counts

    if resume:
        if not model_summary_path.is_file():
            raise RunIdentityError("existing run has no model_summary.json")
        existing_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
        if existing_summary != summary:
            raise RunIdentityError("Stage 1 parameter summary differs; existing run was not modified")
        if checkpoint.get("identity_sha256") != identity:
            raise RunIdentityError("Stage 1 checkpoint identity mismatch")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, normalized)
        _atomic_json(environment_path, identity_config["environment"])
        _atomic_json(run_manifest_path, manifest)
        _atomic_json(model_summary_path, summary)

    max_epochs = training["max_epochs"]
    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    epoch_completed = 0
    next_epoch = 0
    completed = False
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        next_epoch = int(checkpoint["next_epoch"])
        epoch_completed = int(checkpoint["epoch_completed"])
        best_mae = float(checkpoint["best_mae"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        completed = bool(checkpoint["completed"])
        _restore_stage1_rng_state(checkpoint["rng_state"], generator, device)
        _prune_resource_rows(resources_path, epoch_completed)
        _atomic_jsonl(metrics_path, history)
        if completed:
            return {
                "run_dir": str(output),
                "seed": seed,
                "best_epoch": best_epoch,
                "best_valid_mae": best_mae,
                "completed": True,
                "resumed": True,
            }
    else:
        initial = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            device=device,
            identity=identity,
            seed=seed,
            epoch=0,
            next_epoch=0,
            best_mae=best_mae,
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
        train_l1, optimizer_updates = _train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            batch_size=training["batch_size"],
            accumulation_steps=training["accumulation_steps"],
            grad_clip_value=float(training["grad_clip_value"]),
        )
        _synchronize(device)
        train_seconds = time.perf_counter() - train_start
        valid_start = time.perf_counter()
        evaluation = _evaluate(model, valid_loader, device)
        _synchronize(device)
        valid_seconds = time.perf_counter() - valid_start
        train_valid_seconds = time.perf_counter() - epoch_start
        memory = _cuda_peak_memory(device)
        scheduler.step(evaluation["mae"])

        improved = evaluation["mae"] < best_mae - 1e-6
        if improved:
            best_mae = float(evaluation["mae"])
            best_epoch = epoch_number
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch_number,
                "train_l1": train_l1,
                "valid_mae": float(evaluation["mae"]),
                "best_valid_mae": best_mae,
                "best_epoch": best_epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "optimizer_updates": optimizer_updates,
            }
        )
        next_epoch = epoch_number
        epoch_completed = epoch_number
        completed = stale_epochs >= training["early_stop"] or next_epoch >= max_epochs

        if improved:
            _atomic_torch_save(best_path, _copy_state_to_cpu(model))
            _atomic_jsonl(validation_path, _valid_records(evaluation))
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
                "optimizer_updates": optimizer_updates,
                "train_samples_per_second": len(train) / train_seconds if train_seconds > 0 else None,
                "valid_samples_per_second": len(valid) / valid_seconds if valid_seconds > 0 else None,
                **memory,
            },
        )
        last = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            device=device,
            identity=identity,
            seed=seed,
            epoch=epoch_completed,
            next_epoch=next_epoch,
            best_mae=best_mae,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            history=history,
            completed=completed,
        )
        _atomic_torch_save(last_path, last)
        _atomic_jsonl(metrics_path, history)
        epochs_this_call += 1
        if completed:
            break

    return {
        "run_dir": str(output),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_valid_mae": best_mae,
        "last_epoch": epoch_completed,
        "completed": completed,
        "resumed": resume,
    }


def run_stage1(config: Mapping[str, Any], *, seed: int | None = None, resume: bool = False, max_epochs_this_call: int | None = None) -> list[dict[str, Any]]:
    normalized = normalize_stage1_config(config)
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS):
        raise BCMLFTrainingError("seed must be one of 13, 42, 2026")
    if resume and seed is None:
        raise BCMLFTrainingError("resume requires one explicit --seed")
    selected = (seed,) if seed is not None else SEEDS
    results = []
    for current_seed in selected:
        run_dir = Path(normalized["run_root"]) / "bienc_stage1" / ("seed_{}".format(current_seed))
        results.append(
            train_bienc_stage(
                normalized,
                current_seed,
                run_dir,
                resume=resume,
                max_epochs_this_call=max_epochs_this_call,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-call", type=int)
    args = parser.parse_args(argv)
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    results = run_stage1(
        config,
        seed=args.seed,
        resume=args.resume,
        max_epochs_this_call=args.max_epochs_this_call,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0

