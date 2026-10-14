"""BC-MLF MSALM five-class Stage 2 on JevAffect's canonical cache.

The model architecture, local pretrained GPT loading, AV transfer, and
five-class loss come from ``vendor/bcmlf``. This driver ports the source
trainer's selective GPT/AV fine-tuning and four AdamW groups, while using only
train/valid cache splits. Epoch-boundary checkpoints are bound to their
configuration, cache, Stage 1 weights, code, and runtime identity.

Two source-loop corrections are deliberate: accumulated mean losses are
weighted by the examples in the effective batch, and a short last accumulation
group receives exactly one optimizer/scheduler step. This corrects the source
trainer's over-scaled accumulation and dropped-tail behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

# Set before importing torch so deterministic CUDA GEMM is available in GPU
# runs as well as CPU-only fixtures.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
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
CLASS_ORDER = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
_GPT_NO_DECAY = ("bias", "LayerNorm.bias", "LayerNorm.weight", "_LlamaRMSNorm.weight")
_CONFIG_KEYS = {
    "repo_root",
    "cache_dir",
    "run_root",
    "msalm_config",
    "stage1_run_template",
    "seeds",
    "training",
}
_TRAINING_KEYS = {
    "max_epochs",
    "batch_size",
    "update_epochs",
    "early_stop",
    "warmup_epochs",
    "grad_clip_value",
    "num_workers",
    "device",
}


class BCMLFStage2Error(TrainingError):
    """Raised when a Stage 2 cache, config, model, or run is invalid."""


class _AttrDict(dict):
    """Small EasyDict-compatible mapping used by the unmodified BC-MLF model."""

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


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_canonical_json(list(value.shape)))
    # Update from a contiguous view without materializing a second full copy
    # when the cache array is already C-contiguous.
    contiguous = value if value.flags.c_contiguous else np.ascontiguousarray(value)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BCMLFStage2Error("cannot read {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise BCMLFStage2Error("{} must contain a JSON object".format(path))
    return value


def _resolve_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BCMLFStage2Error("{} must be a non-empty path".format(name))
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_stage2_config(
    path: str | Path, *, repo_root_override: str | Path | None = None
) -> dict[str, Any]:
    """Read the Stage 2 JSON and resolve paths relative to its repo root."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BCMLFStage2Error("cannot read config {}: {}".format(config_path, exc)) from exc
    if not isinstance(raw, Mapping):
        raise BCMLFStage2Error("Stage 2 config must be a JSON object")
    if repo_root_override is not None:
        raw = dict(raw)
        raw["repo_root"] = str(Path(repo_root_override).expanduser().resolve())
    return normalize_stage2_config(raw, base_dir=config_path.parent)


def normalize_stage2_config(
    config: Mapping[str, Any], *, base_dir: str | Path | None = None
) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != _CONFIG_KEYS:
        raise BCMLFStage2Error("Stage 2 config keys must be exactly {}".format(sorted(_CONFIG_KEYS)))
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd().resolve()
    root_value = config["repo_root"]
    if not isinstance(root_value, str) or not root_value.strip():
        raise BCMLFStage2Error("repo_root must be a non-empty path")
    repo_root = Path(root_value).expanduser()
    if not repo_root.is_absolute():
        repo_root = (base / repo_root)
    repo_root = repo_root.resolve()
    if not (repo_root / "vendor" / "bcmlf" / "models" / "singleTask" / "MSALM.py").is_file():
        raise BCMLFStage2Error("repo_root must contain vendored BC-MLF MSALM.py")

    seeds = config["seeds"]
    if not isinstance(seeds, list) or tuple(seeds) != SEEDS:
        raise BCMLFStage2Error("seeds must be exactly [13, 42, 2026]")
    training = config["training"]
    if not isinstance(training, Mapping) or set(training) != _TRAINING_KEYS:
        raise BCMLFStage2Error(
            "training keys must be exactly {}".format(sorted(_TRAINING_KEYS))
        )
    for key in ("max_epochs", "batch_size", "update_epochs", "early_stop"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BCMLFStage2Error("training.{} must be a positive integer".format(key))
    for key in ("warmup_epochs", "num_workers"):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BCMLFStage2Error("training.{} must be a non-negative integer".format(key))
    if training["num_workers"] != 0:
        raise BCMLFStage2Error("num_workers must be 0 for exact epoch-boundary resume")
    clip = training["grad_clip_value"]
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(float(clip)):
        raise BCMLFStage2Error("training.grad_clip_value must be finite numeric")
    if float(clip) < -1.0:
        raise BCMLFStage2Error("grad_clip_value must be -1 (disabled) or non-negative")
    device = training["device"]
    if device not in ("auto", "cpu", "cuda") and not (
        isinstance(device, str) and device.startswith("cuda:")
    ):
        raise BCMLFStage2Error("training.device must be auto, cpu, cuda, or cuda:N")

    stage1_template = config["stage1_run_template"]
    if not isinstance(stage1_template, str) or "{seed}" not in stage1_template:
        raise BCMLFStage2Error("stage1_run_template must contain the literal {seed}")
    if "{" in stage1_template.replace("{seed}", "") or "}" in stage1_template.replace("{seed}", ""):
        raise BCMLFStage2Error("stage1_run_template supports only the {seed} field")

    normalized = {
        "repo_root": str(repo_root),
        "cache_dir": str(_resolve_path(repo_root, config["cache_dir"], "cache_dir")),
        "run_root": str(_resolve_path(repo_root, config["run_root"], "run_root")),
        "msalm_config": str(_resolve_path(repo_root, config["msalm_config"], "msalm_config")),
        "stage1_run_template": stage1_template,
        "seeds": list(SEEDS),
        "training": {key: training[key] for key in sorted(_TRAINING_KEYS)},
    }
    if not Path(normalized["msalm_config"]).is_file():
        raise BCMLFStage2Error("MSALM config does not exist: {}".format(normalized["msalm_config"]))
    return normalized


class _FiveClassSplit(Dataset):
    """Validated train or valid view of a canonical raw-text BC-MLF cache."""

    def __init__(self, data: Mapping[str, Any], split: str) -> None:
        if split not in ("train", "valid"):
            raise BCMLFStage2Error("Stage 2 can read only train and valid splits")
        self.split = split
        ids = data.get("ids")
        raw_text = data.get("raw_text")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise BCMLFStage2Error("cache IDs must be a non-empty list[str]")
        if len(set(ids)) != len(ids):
            raise BCMLFStage2Error("cache has duplicate canonical IDs")
        if not isinstance(raw_text, list) or len(raw_text) != len(ids) or any(
            not isinstance(x, str) for x in raw_text
        ):
            raise BCMLFStage2Error("raw_text must be list[str] aligned with IDs")
        self.ids = tuple(ids)
        self.raw_text = tuple(raw_text)

        labels = data.get("labels")
        valid = data.get("label_valid")
        raw_y = data.get("raw_labels")
        if not isinstance(labels, np.ndarray) or labels.dtype != np.dtype(np.int64):
            raise BCMLFStage2Error("labels.npy must be int64 class indices")
        if not isinstance(valid, np.ndarray) or valid.dtype != np.dtype(np.bool_):
            raise BCMLFStage2Error("label_valid.npy must be bool")
        if labels.shape != (len(ids),) or valid.shape != (len(ids),):
            raise BCMLFStage2Error("labels, label_valid, and IDs must align")
        if not bool(np.all(valid)) or np.any(labels < 0) or np.any(labels > 4):
            raise BCMLFStage2Error("all train/valid classes must be known indices 0..4")
        if not isinstance(raw_y, np.ndarray) or raw_y.shape != (len(ids),):
            raise BCMLFStage2Error("raw_labels.npy must be a numeric [N] vector")
        if raw_y.dtype.kind not in "fiu" or not np.isfinite(raw_y).all():
            raise BCMLFStage2Error("train/valid raw_labels must be finite numeric values")
        self.labels = labels
        self.raw_y = raw_y

        self.features: dict[str, np.ndarray] = {}
        for modality in ("audio", "vision"):
            value = data.get(modality)
            if not isinstance(value, np.ndarray) or value.ndim != 3:
                raise BCMLFStage2Error("{} must be a [N,T,D] array".format(modality))
            if value.shape[0] != len(ids) or value.dtype.kind != "f":
                raise BCMLFStage2Error("{} must be floating and aligned with IDs".format(modality))
            if not np.isfinite(value).all():
                raise BCMLFStage2Error("{} contains NaN or Inf".format(modality))
            self.features[modality] = value

        meta = data.get("meta")
        if not isinstance(meta, Mapping):
            raise BCMLFStage2Error("cache reader must return verified meta.json as 'meta'")
        for key in ("source_sha256", "feature_dims", "sequence_lengths", "text_features_ready"):
            if key not in meta:
                raise BCMLFStage2Error("canonical meta.json is missing {}".format(key))
        self.meta = dict(meta)
        self._array_hashes = {
            "labels": _sha256_array(labels),
            "raw_labels": _sha256_array(raw_y),
            "audio": _sha256_array(self.features["audio"]),
            "vision": _sha256_array(self.features["vision"]),
            "raw_text": _sha256_bytes(_canonical_json(list(self.raw_text))),
        }

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
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "raw_y": torch.tensor(float(self.raw_y[index]), dtype=torch.float32),
            "raw_text": self.raw_text[index],
            "id": self.ids[index],
        }


def _load_cache(
    cache_dir: Path,
    data_loader=load_cached_split,
) -> tuple[_FiveClassSplit, _FiveClassSplit, dict[str, Any]]:
    # Raw text is tokenized by the local Chinese GPT tokenizer. Text feature
    # readiness is intentionally not required; no text.npy is read or faked.
    train_data = data_loader(
        cache_dir, "train", require_text_features=False, require_raw_text=True
    )
    valid_data = data_loader(
        cache_dir, "valid", require_text_features=False, require_raw_text=True
    )
    train = _FiveClassSplit(train_data, "train")
    valid = _FiveClassSplit(valid_data, "valid")
    for key in ("source_sha256", "feature_dims", "sequence_lengths"):
        if train.meta.get(key) != valid.meta.get(key):
            raise BCMLFStage2Error("train and valid cache {} metadata differ".format(key))
    if train.meta.get("text_features_ready") is not valid.meta.get("text_features_ready"):
        raise BCMLFStage2Error("train and valid text readiness metadata differ")
    for modality in ("audio", "vision"):
        if train.features[modality].shape[1:] != valid.features[modality].shape[1:]:
            raise BCMLFStage2Error("train and valid {} feature shapes differ".format(modality))
    identity = {
        "source_sha256": train.meta["source_sha256"],
        "meta_sha256": _sha256_bytes(_canonical_json(train.meta)),
        "train_ids_sha256": _sha256_bytes(_canonical_json(list(train.ids))),
        "valid_ids_sha256": _sha256_bytes(_canonical_json(list(valid.ids))),
        "train_array_sha256": train._array_hashes,
        "valid_array_sha256": valid._array_hashes,
    }
    return train, valid, identity


def _stage1_run_path(normalized: Mapping[str, Any], seed: int) -> Path:
    root = Path(normalized["repo_root"])
    template = normalized["stage1_run_template"].format(seed=seed)
    return _resolve_path(root, template, "stage1_run_template")


def _load_stage1_last_checkpoint(path: Path) -> dict[str, Any]:
    """Load the internal Stage 1 checkpoint with a weights-only unpickler.

    The saved NumPy RNG tuple contains an ndarray/dtype, so allow only NumPy's
    reconstruction primitives in addition to PyTorch's built-in safe types.
    Never fall back to unrestricted pickle loading if this safe path fails.
    """
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if not callable(safe_globals):
        raise BCMLFStage2Error(
            "this Torch version lacks safe_globals; refusing unsafe Stage 1 checkpoint loading"
        )
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    allowlisted_numpy = [
        numpy_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]
    try:
        with safe_globals(allowlisted_numpy):
            value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise BCMLFStage2Error(
            "this Torch version does not support weights_only checkpoint loading"
        ) from exc
    except Exception as exc:
        raise BCMLFStage2Error(
            "cannot safely read Stage 1 last.pt: {}".format(exc)
        ) from exc
    if not isinstance(value, dict):
        raise BCMLFStage2Error("Stage 1 last.pt must contain a checkpoint dict")
    return value


def _validate_stage1_run(
    path: Path, source_sha256: str, expected_seed: int
) -> tuple[Path, str]:
    manifest_path = path / "run_manifest.json"
    weight_path = path / "bienc_best_encoder.pth"
    last_path = path / "last.pt"
    if not manifest_path.is_file() or not weight_path.is_file() or not last_path.is_file():
        raise BCMLFStage2Error(
            "Stage 2 requires a completed Stage 1 manifest, last.pt, and best flat encoder weights at {}".format(path)
        )
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != "BI_ENC_raw_label_L1_pretrain":
        raise BCMLFStage2Error("Stage 1 run manifest has an unexpected stage")
    if manifest.get("cache_source_sha256") != source_sha256:
        raise BCMLFStage2Error("Stage 1 and Stage 2 cache source hashes differ")
    if manifest.get("seed") != expected_seed:
        raise BCMLFStage2Error("Stage 1 manifest seed does not match Stage 2 seed")
    manifest_identity = manifest.get("identity_sha256")
    if not isinstance(manifest_identity, str) or re.fullmatch(
        r"[0-9a-fA-F]{64}", manifest_identity
    ) is None:
        raise BCMLFStage2Error("Stage 1 manifest identity_sha256 must be a SHA-256 hex digest")

    checkpoint = _load_stage1_last_checkpoint(last_path)
    if checkpoint.get("identity_sha256") != manifest_identity:
        raise BCMLFStage2Error("Stage 1 last.pt identity does not match its manifest")
    if checkpoint.get("seed") != expected_seed:
        raise BCMLFStage2Error("Stage 1 checkpoint seed does not match Stage 2 seed")
    if checkpoint.get("completed") is not True:
        raise BCMLFStage2Error("Stage 1 training is not marked completed")
    epoch_completed = checkpoint.get("epoch_completed")
    next_epoch = checkpoint.get("next_epoch")
    best_epoch = checkpoint.get("best_epoch")
    best_mae = checkpoint.get("best_mae")
    if (
        isinstance(epoch_completed, bool)
        or not isinstance(epoch_completed, int)
        or epoch_completed <= 0
        or next_epoch != epoch_completed
        or isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= epoch_completed
        or isinstance(best_mae, bool)
        or not isinstance(best_mae, (int, float))
        or not math.isfinite(float(best_mae))
    ):
        raise BCMLFStage2Error("Stage 1 last.pt completion metadata is invalid")
    try:
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        raise BCMLFStage2Error(
            "this Torch version does not support safe weights_only Stage 1 encoder loading"
        )
    except Exception as exc:
        raise BCMLFStage2Error("cannot read Stage 1 flat AV weights: {}".format(exc)) from exc
    if not isinstance(state, Mapping) or not state:
        raise BCMLFStage2Error("Stage 1 AV checkpoint must be a non-empty state mapping")
    if any(not isinstance(key, str) or not key.startswith("Model.") for key in state):
        raise BCMLFStage2Error("Stage 1 checkpoint must contain flat Model.* tensors")
    if any(not torch.is_tensor(value) for value in state.values()):
        raise BCMLFStage2Error("Stage 1 checkpoint must contain only tensor values")
    return weight_path, _sha256_file(weight_path)


def _load_model_config(
    config_path: Path,
    repo_root: Path,
    train: _FiveClassSplit,
    stage1_weights: Path,
    training: Mapping[str, Any],
    device: torch.device,
) -> tuple[_AttrDict, dict[str, Any]]:
    from vendor.bcmlf.adapters.config_paths import resolve_bcmlf_paths

    raw = _read_json(config_path)
    try:
        resolved = resolve_bcmlf_paths(raw, repo_root=repo_root)
        common = dict(resolved["datasetCommonParams"]["sims"]["unaligned"])
        model_common = dict(resolved["msalm"]["commonParams"])
        model_dataset = dict(resolved["msalm"]["datasetParams"]["sims"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BCMLFStage2Error("MSALM config lacks expected sims/unaligned sections: {}".format(exc)) from exc

    if common.get("need_data_aligned", False) or model_common.get("need_model_aligned", False):
        raise BCMLFStage2Error("canonical cached inputs are unaligned; config must keep alignment disabled")
    # The Stage 2 cache deliberately has raw transcripts and A/V features but
    # no contextual text tensor. MSALM's AV_Enc consumes only the A/V feature
    # dimensions; its first, historical text dimension remains a model-config
    # constant and is never read from cache meta.json.
    configured_dims = common.get("feature_dims")
    if not isinstance(configured_dims, Sequence) or len(configured_dims) != 3:
        raise BCMLFStage2Error("native config feature_dims must provide its historical [text,A,V] widths")
    audio_dim = int(train.features["audio"].shape[2])
    vision_dim = int(train.features["vision"].shape[2])
    configured_audio = int(configured_dims[1])
    configured_vision = int(configured_dims[2])
    if (audio_dim, vision_dim) != (configured_audio, configured_vision):
        raise BCMLFStage2Error(
            "cached A/V feature dims ({}, {}) do not match native MSALM config ({}, {})".format(
                audio_dim, vision_dim, configured_audio, configured_vision
            )
        )
    actual_audio_len = int(train.features["audio"].shape[1])
    actual_vision_len = int(train.features["vision"].shape[1])
    configured_lengths = common.get("seq_lens")
    if not isinstance(configured_lengths, Sequence) or len(configured_lengths) != 3:
        raise BCMLFStage2Error("native config seq_lens must contain [text,A,V] lengths")
    if (actual_audio_len, actual_vision_len) != (
        int(configured_lengths[1]), int(configured_lengths[2])
    ):
        raise BCMLFStage2Error("cached A/V sequence lengths do not match native MSALM config")

    merged: dict[str, Any] = {}
    merged.update(common)
    merged.update(model_common)
    merged.update(model_dataset)
    av_config = merged.get("av_enc")
    if not isinstance(av_config, Mapping):
        raise BCMLFStage2Error("MSALM config must include av_enc")
    av_config = dict(av_config)
    if av_config.get("from_pretrained") is not True:
        raise BCMLFStage2Error("Stage 2 requires a pretrained Stage 1 AV encoder; random initialization is forbidden")
    av_config["path_to_pretrained"] = str(stage1_weights)
    av_config["feature_dims"] = [int(configured_dims[0]), audio_dim, vision_dim]
    # Preserve the source AV timeline length (55), independently of text tokens (39).
    if isinstance(av_config.get("maxlen"), bool) or not isinstance(av_config.get("maxlen"), int) or av_config["maxlen"] <= 0:
        raise BCMLFStage2Error("native av_enc.maxlen must be a positive integer")
    merged["av_enc"] = av_config
    max_token_len = int(merged.get("max_token_len", -1))
    if max_token_len <= 0:
        raise BCMLFStage2Error("max_token_len must be positive")
    merged.update(
        {
            "model_name": "msalm",
            "dataset_name": "sims",
            "train_mode": "classification",
            "feature_dims": [int(configured_dims[0]), audio_dim, vision_dim],
            "seq_lens": [max_token_len, actual_audio_len, actual_vision_len],
            "train_samples": len(train),
            "device": device,
            "max_epochs": training["max_epochs"],
            "update_epochs": training["update_epochs"],
            "early_stop": training["early_stop"],
            "batch_size": training["batch_size"],
            "grad_clip": float(training["grad_clip_value"]),
            "KeyEval": "Accuracy",
        }
    )
    return _attrify(merged), {"native_config": resolved, "source_config": raw}


class _MSALMPrefix(nn.Module):
    """Expose source-trainer's model.Model.* namespace and tokenizer path."""

    def __init__(self, args: _AttrDict) -> None:
        super().__init__()
        from vendor.bcmlf.models.singleTask.MSALM import MSALM

        self.Model = MSALM(args)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
    ) -> Mapping[str, Any]:
        return self.Model(
            input_ids,
            audio,
            vision,
            attention_mask=attention_mask,
        )


def _configure_source_trainability(model: nn.Module, args: _AttrDict) -> None:
    """Port MSALM trainer's exact selective GPT and AV unfreezing policy."""
    inner = getattr(model, "Model", None)
    if inner is None:
        raise BCMLFStage2Error("MSALM wrapper must expose model.Model")
    model.requires_grad_(False)
    flavor = str(args["mmgpt"].get("type", "llama")).lower()
    if "gpt" in flavor:
        trainable_names = ["alpha_1", "alpha_2", "ln_1", "ln_2", "attn", "bn_embedding"]
        if bool(args["mmgpt"].get("use_lora", False)):
            trainable_names.extend(("lora_c_fc", "lora_c_proj"))
        else:
            trainable_names.extend(("c_fc", "c_proj"))
        try:
            decoder_layers = inner.lang_encoder.transformer.h
            token_embedding = inner.lang_encoder.transformer.wte
        except AttributeError as exc:
            raise BCMLFStage2Error("GPT model lacks the source trainer's transformer.h/wte interface") from exc
    else:
        trainable_names = [
            "alpha_1", "alpha_2", "ln_1", "ln_2", "attn",
            "lora_gate_proj", "lora_up_proj", "lora_down_proj", "bn_embedding",
        ]
        try:
            decoder_layers = inner.lang_encoder.model.layers
            token_embedding = inner.lang_encoder.model.embed_tokens
        except AttributeError as exc:
            raise BCMLFStage2Error("non-GPT model lacks the source trainer's layer/embedding interface") from exc
    for name, parameter in decoder_layers.named_parameters():
        if "ca_layer" in name and any(token in name for token in trainable_names):
            parameter.requires_grad_(True)
    for name, parameter in token_embedding.named_parameters():
        if any(token in name for token in trainable_names):
            parameter.requires_grad_(True)

    if not hasattr(inner, "av_encoder"):
        raise BCMLFStage2Error("MSALM model lacks av_encoder")
    inner.av_encoder.requires_grad_(True)
    if bool(args.get("use_ulgm", False)):
        for name in ("W_task_0", "W_task_1"):
            if not hasattr(inner, name):
                raise BCMLFStage2Error("use_ulgm requires {}".format(name))
            getattr(inner, name).requires_grad_(True)
    else:
        if not hasattr(inner, "W_task"):
            raise BCMLFStage2Error("MSALM model lacks W_task")
        inner.W_task.requires_grad_(True)
    for name in ("W_bn", "W_text", "W_av"):
        if not hasattr(inner, name):
            raise BCMLFStage2Error("modded loss requires {}".format(name))
        getattr(inner, name).requires_grad_(True)
    for name in ("contribution_gate", "global_branch_logits"):
        value = getattr(inner, name, None)
        if value is not None:
            if isinstance(value, nn.Module):
                value.requires_grad_(True)
            elif isinstance(value, nn.Parameter):
                value.requires_grad_(True)
    if bool(args.get("use_lnorm", False)):
        if not hasattr(inner, "LN"):
            raise BCMLFStage2Error("use_lnorm requires model.Model.LN")
        inner.LN.requires_grad_(True)
    if not any(parameter.requires_grad for parameter in model.parameters()):
        raise BCMLFStage2Error("source freeze policy left no trainable parameters")


def _optimizer_groups(
    model: nn.Module, args: _AttrDict
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the four native MMGPT/AV decay groups in source order."""
    av_prefix = "Model.av_encoder."
    non_av_decay: list[nn.Parameter] = []
    non_av_no_decay: list[nn.Parameter] = []
    av_decay: list[nn.Parameter] = []
    av_no_decay: list[nn.Parameter] = []
    counts = {"non_av_decay": 0, "non_av_no_decay": 0, "av_decay": 0, "av_no_decay": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = any(token in name for token in _GPT_NO_DECAY)
        if name.startswith(av_prefix):
            group = av_no_decay if no_decay else av_decay
            group_key = "av_no_decay" if no_decay else "av_decay"
        else:
            group = non_av_no_decay if no_decay else non_av_decay
            group_key = "non_av_no_decay" if no_decay else "non_av_decay"
        group.append(parameter)
        counts[group_key] += parameter.numel()

    beta_1 = float(args.get("beta_1", 0.9))
    beta_2 = float(args.get("beta_2", 0.999))
    lr_mmgpt = float(args.learning_rate_mmgpt)
    lr_av = float(args.learning_rate_av)
    wd_mmgpt = float(args.weight_decay_mmgpt)
    wd_av = float(args.weight_decay_av)
    groups = [
        {"params": non_av_decay, "weight_decay": wd_mmgpt, "lr": lr_mmgpt, "betas": (beta_1, beta_2)},
        {"params": non_av_no_decay, "weight_decay": 0.0, "lr": lr_mmgpt, "betas": (beta_1, beta_2)},
        {"params": av_decay, "weight_decay": wd_av, "lr": lr_av, "betas": (beta_1, beta_2)},
        {"params": av_no_decay, "weight_decay": 0.0, "lr": lr_av, "betas": (beta_1, beta_2)},
    ]
    if sum(counts.values()) == 0:
        raise BCMLFStage2Error("optimizer has no trainable parameters")
    details = {
        "name": "AdamW",
        "betas": [beta_1, beta_2],
        "groups": [
            {"name": name, "parameters": sum(p.numel() for p in group["params"]),
             "learning_rate": group["lr"], "weight_decay": group["weight_decay"]}
            for name, group in zip(
                ("non_av_decay", "non_av_no_decay", "av_decay", "av_no_decay"), groups
            )
        ],
    }
    return groups, details


class _LinearWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Source BC-MLF's per-optimizer-update linear warmup scheduler."""

    def __init__(self, optimizer, warmup_steps: int, last_epoch: int = -1):
        self.target_lr = [group["lr"] for group in optimizer.param_groups]
        self.warmup_steps = int(warmup_steps)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            fraction = self.last_epoch / max(1, self.warmup_steps)
            return [target * fraction for target in self.target_lr]
        return list(self.target_lr)


def _build_source_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    updates_per_epoch: int,
    warmup_epochs: int,
):
    """Equivalent to BC-MLF get_scheduler, with ceil updates for tail steps."""
    warmup_steps = updates_per_epoch * warmup_epochs
    cosine_steps = max(1, (max_epochs + 1) * updates_per_epoch - warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, eta_min=1e-7, T_0=cosine_steps
    )
    if warmup_steps == 0:
        return cosine
    warmup = _LinearWarmupScheduler(optimizer, warmup_steps)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )


def _resolve_device(value: str) -> torch.device:
    device = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else (
        "cpu" if value == "auto" else value
    ))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BCMLFStage2Error("CUDA requested but not available")
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _rng_state(generator: torch.Generator, device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "loader_generator": generator.get_state(),
        "torch_cuda": [],
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(
    state: Mapping[str, Any], generator: torch.Generator, device: torch.device
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


def _environment(device: torch.device) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_used": device.type == "cuda",
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
    }
    if device.type == "cuda":
        env.update(
            device_name=torch.cuda.get_device_name(device),
            device_count=torch.cuda.device_count(),
            cudnn_version=torch.backends.cudnn.version(),
        )
    else:
        env.update(device_name=platform.processor() or "CPU", device_count=0, cudnn_version=None)
    return env


def _implementation_hash(repo_root: Path, config_path: Path) -> str:
    # Hash the driver, cache/train primitives it relies on, and all vendored
    # BC-MLF Python sources. No ignored work/BC-MLF source path is required.
    files = [
        ("stage2_driver.py", Path(__file__).resolve()),
        ("stage2_config.json", config_path),
        ("jevaffect/cache.py", repo_root / "jevaffect/cache.py"),
        ("jevaffect/training.py", repo_root / "jevaffect/training.py"),
    ]
    vendor_root = repo_root / "vendor" / "bcmlf"
    if not vendor_root.is_dir():
        raise BCMLFStage2Error("vendored BC-MLF source directory is missing")
    files.extend(
        (path.relative_to(repo_root).as_posix(), path)
        for path in sorted(vendor_root.rglob("*.py"))
        if "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    for label, path in files:
        if not path.is_file():
            raise BCMLFStage2Error("implementation fingerprint file is missing: {}".format(path))
        digest.update(label.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _local_model_directory_fingerprint(path: Path) -> str:
    if not path.is_dir():
        raise BCMLFStage2Error(
            "configured local GPT/tokenizer directory is missing: {}".format(path)
        )
    digest = hashlib.sha256()
    paths = sorted((item for item in path.rglob("*") if item.is_file()), key=lambda p: p.relative_to(path).as_posix())
    if not paths:
        raise BCMLFStage2Error("configured local GPT/tokenizer directory is empty: {}".format(path))
    for file_path in paths:
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def _tokenize(model: nn.Module, raw_text: Sequence[str], device: torch.device):
    try:
        encoded = model.Model.tokenizer(
            list(raw_text), return_tensors="pt", padding="max_length", truncation=True
        )
    except Exception as exc:
        raise BCMLFStage2Error("local GPT tokenizer failed: {}".format(exc)) from exc
    if hasattr(encoded, "to"):
        encoded = encoded.to(device)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise BCMLFStage2Error("tokenizer must return matching input_ids/attention_mask [B,L]")
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise BCMLFStage2Error("tokenizer input_ids must be integer tensors")
    return input_ids.long(), attention_mask


def _amp_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _make_scaler(enabled: bool):
    # Source Stage 2 uses GradScaler under its bf16 branch. Keep a scaler slot
    # in all checkpoints; scaling remains disabled for CPU and the default
    # fp32 config, and CUDA bf16 does not require numerical loss scaling.
    try:
        return torch.amp.GradScaler("cuda", enabled=bool(enabled))
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=bool(enabled))


def _count_parameters(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _copy_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise BCMLFStage2Error("checkpoint must contain a dict")
    return value


def _loader(
    split: _FiveClassSplit,
    *,
    batch_size: int,
    generator: torch.Generator,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        split,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def _group_samples(n_examples: int, batch_size: int, n_batches: int, start: int, group_size: int) -> int:
    end = min(start + group_size, n_batches)
    return sum(min(batch_size, n_examples - batch * batch_size) for batch in range(start, end))


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    args: _AttrDict,
    device: torch.device,
    batch_size: int,
    update_epochs: int,
    grad_clip_value: float,
    use_bf16: bool,
    loss_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    from vendor.bcmlf.adapters.five_class import bcmlf_five_class_loss

    model.train()
    n_batches = len(loader)
    n_examples = len(loader.dataset)
    total_samples = 0
    total_correct = 0
    loss_sums = {name: 0.0 for name in ("loss", "task_loss", "bn_loss", "av_loss", "text_loss", "lm_loss", "contrast_loss")}
    updates = 0
    non_av_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("Model.av_encoder.")
    ]
    av_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("Model.av_encoder.")
    ]
    for batch_index, batch in enumerate(loader):
        group_start = (batch_index // update_epochs) * update_epochs
        group_samples = _group_samples(n_examples, batch_size, n_batches, group_start, update_epochs)
        if group_samples <= 0:
            raise BCMLFStage2Error("empty effective accumulation group")
        if batch_index == group_start:
            optimizer.zero_grad(set_to_none=True)

        audio = batch["audio"].to(device, non_blocking=True)
        vision = batch["vision"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        raw_y = batch["raw_y"].to(device, non_blocking=True).float()
        input_ids, attention_mask = _tokenize(model, batch["raw_text"], device)
        with _amp_context(use_bf16):
            outputs = model(input_ids, audio, vision, attention_mask=attention_mask)
            terms = bcmlf_five_class_loss(
                outputs,
                labels,
                raw_y,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **loss_kwargs,
            )
        sample_count = int(labels.numel())
        scaled_loss = terms["loss"] * (sample_count / group_samples)
        if use_bf16:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        for name, value in terms.items():
            loss_sums[name] += float(value.detach().float().item()) * sample_count
        predictions = outputs["task_logits"].detach().argmax(dim=-1)
        total_correct += int((predictions == labels).sum().item())
        total_samples += sample_count

        group_end = min(group_start + update_epochs, n_batches)
        if batch_index + 1 == group_end:
            if use_bf16:
                scaler.unscale_(optimizer)
            if grad_clip_value != -1.0:
                # Source uses value clipping. Apply it to both native parameter
                # partitions, including AV (the source accidentally omitted AV).
                if non_av_parameters:
                    nn.utils.clip_grad_value_(non_av_parameters, grad_clip_value)
                if av_parameters:
                    nn.utils.clip_grad_value_(av_parameters, grad_clip_value)
            if use_bf16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            updates += 1
    if total_samples == 0:
        raise BCMLFStage2Error("empty Stage 2 training split")
    return {
        "losses": {name: value / total_samples for name, value in loss_sums.items()},
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
        "optimizer_updates": updates,
    }


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    n_samples = 0
    correct = 0
    nll_sum = 0.0
    ids: list[str] = []
    labels_all: list[int] = []
    logits_all: list[list[float]] = []
    probabilities_all: list[list[float]] = []
    weights_all: list[list[float] | None] = []
    for batch in loader:
        audio = batch["audio"].to(device, non_blocking=True)
        vision = batch["vision"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).long()
        input_ids, attention_mask = _tokenize(model, batch["raw_text"], device)
        outputs = model(input_ids, audio, vision, attention_mask=attention_mask)
        logits = outputs.get("task_logits")
        if not torch.is_tensor(logits) or logits.ndim != 2 or logits.shape != (len(labels), 5):
            raise BCMLFStage2Error("MSALM task_logits must be [B,5]")
        if not torch.isfinite(logits).all():
            raise BCMLFStage2Error("MSALM task_logits contain NaN or Inf")
        nll_sum += float(F.cross_entropy(logits.float(), labels, reduction="sum").item())
        probabilities = torch.softmax(logits.float(), dim=-1)
        predictions = logits.argmax(dim=-1)
        correct += int((predictions == labels).sum().item())
        n_samples += int(labels.numel())
        ids.extend(str(value) for value in batch["id"])
        labels_all.extend(int(value) for value in labels.detach().cpu().tolist())
        logits_all.extend(logits.float().detach().cpu().tolist())
        probabilities_all.extend(probabilities.detach().cpu().tolist())
        weights = outputs.get("Contribution_weights")
        if torch.is_tensor(weights):
            weights_all.extend(weights.float().detach().cpu().tolist())
        else:
            weights_all.extend([None] * len(labels))
    if not n_samples:
        raise BCMLFStage2Error("empty Stage 2 validation split")
    return {
        "accuracy": correct / n_samples,
        "nll": nll_sum / n_samples,
        "n_samples": n_samples,
        "ids": ids,
        "labels": labels_all,
        "logits": logits_all,
        "probabilities": probabilities_all,
        "contribution_weights": weights_all,
    }


def _validation_records(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for index, sample_id in enumerate(result["ids"]):
        probs = [float(value) for value in result["probabilities"][index]]
        records.append(
            {
                "ID": sample_id,
                "split": "valid",
                "label": int(result["labels"][index]),
                "prediction": int(np.argmax(probs)),
                "logits": [float(value) for value in result["logits"][index]],
                "probabilities_uncalibrated": probs,
                "contribution_weights": result["contribution_weights"][index],
            }
        )
    return records


def _checkpoint_state(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    generator: torch.Generator,
    device: torch.device,
    identity: str,
    seed: int,
    epoch: int,
    next_epoch: int,
    best_accuracy: float,
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
        "best_valid_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "completed": completed,
        "model_state": _copy_state_to_cpu(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": _rng_state(generator, device),
    }


def _model_factory(args: _AttrDict) -> nn.Module:
    return _MSALMPrefix(args)


def train_stage2_seed(
    config: Mapping[str, Any],
    seed: int,
    run_dir: str | Path,
    *,
    resume: bool = False,
    max_epochs_this_call: int | None = None,
    data_loader=None,
    model_factory=None,
) -> dict[str, Any]:
    """Train one Stage 2 seed using only canonical train and valid splits.

    The injected loader/model factory are for CPU fixtures. A normal run uses
    ``jevaffect.cache.load_cached_split`` and the vendored MSALM constructor;
    that constructor loads the local GPT and strictly transfers Stage 1 AV
    tensors before this driver starts training.
    """
    normalized = normalize_stage2_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise BCMLFStage2Error("seed must be one of 13, 42, 2026")
    if max_epochs_this_call is not None and (
        isinstance(max_epochs_this_call, bool)
        or not isinstance(max_epochs_this_call, int)
        or max_epochs_this_call <= 0
    ):
        raise BCMLFStage2Error("max_epochs_this_call must be a positive integer")
    data_loader = load_cached_split if data_loader is None else data_loader
    model_factory = _model_factory if model_factory is None else model_factory
    device = _resolve_device(normalized["training"]["device"])
    _seed_everything(seed, device)

    cache_dir = Path(normalized["cache_dir"])
    train, valid, cache_identity = _load_cache(cache_dir, data_loader)

    repo_root = Path(normalized["repo_root"])
    model_config_path = Path(normalized["msalm_config"])
    stage1_run = _stage1_run_path(normalized, seed)
    stage1_weights, stage1_sha256 = _validate_stage1_run(
        stage1_run, str(train.meta["source_sha256"]), seed
    )
    args, model_config_record = _load_model_config(
        model_config_path,
        repo_root,
        train,
        stage1_weights,
        normalized["training"],
        device,
    )

    # An injected CPU model avoids requiring local Hugging Face files in
    # fixtures. Production always fingerprints every local GPT/tokenizer file.
    lm_directory = Path(os.path.abspath(os.path.expanduser(os.fspath(args.lm))))
    if model_factory is _model_factory:
        lm_fingerprint = _local_model_directory_fingerprint(lm_directory)
    else:
        lm_fingerprint = "fixture-model-factory"
    implementation_sha256 = _implementation_hash(repo_root, model_config_path)
    environment = _environment(device)
    identity_material = {
        "normalized_config": normalized,
        "seed": seed,
        "cache": cache_identity,
        "stage1_run_manifest_sha256": _sha256_file(stage1_run / "run_manifest.json"),
        "stage1_encoder_sha256": stage1_sha256,
        "native_msalm_config_sha256": _sha256_file(model_config_path),
        "implementation_sha256": implementation_sha256,
        "local_gpt_assets_sha256": lm_fingerprint,
        "environment": environment,
        "test_factory": model_factory is not _model_factory,
    }
    identity = _sha256_bytes(_canonical_json(identity_material))
    output = Path(run_dir).expanduser().resolve()
    manifest: dict[str, Any] = {
        "format_version": 1,
        "identity_sha256": identity,
        "seed": seed,
        "stage": "MSALM_five_class_train_valid",
        "dataset": "simsv2",
        "language": "zh",
        "class_order": list(CLASS_ORDER),
        "cache_source_sha256": train.meta["source_sha256"],
        "cache_meta_sha256": cache_identity["meta_sha256"],
        "stage1_encoder_sha256": stage1_sha256,
        "implementation_sha256": implementation_sha256,
        "environment": environment,
        "feature_dims": {
            "audio": int(train.features["audio"].shape[2]),
            "vision": int(train.features["vision"].shape[2]),
            "text_cache_features": train.meta["feature_dims"].get("text"),
        },
        "sequence_lengths": {
            "audio": int(train.features["audio"].shape[1]),
            "vision": int(train.features["vision"].shape[1]),
            "text_tokens": int(args.max_token_len),
        },
        "parameter_counts": None,
        "optimizer": None,
        "adaptations": [
            "sample-weighted effective-batch gradient accumulation",
            "perform exactly one optimizer/scheduler step for a short final accumulation group",
            "ceil-based updates-per-epoch for warmup schedule so the corrected tail step is represented",
            "clip value gradients in both GPT/task and AV partitions; native source omitted AV gradients",
            "select best checkpoint on valid accuracy and report NLL as a secondary diagnostic",
            "never load or evaluate the test split",
        ],
    }
    run_manifest_path = output / "run_manifest.json"
    config_path = output / "config.json"
    environment_path = output / "environment.json"
    model_summary_path = output / "model_summary.json"
    best_path = output / "best_model.pt"
    last_path = output / "last.pt"
    metrics_path = output / "metrics.jsonl"
    validation_path = output / "validation_predictions.jsonl"
    resources_path = output / "resources.jsonl"

    # Refuse accidental overwrite before loading the heavyweight GPT model.
    if resume:
        if not output.is_dir() or not run_manifest_path.is_file() or not last_path.is_file():
            raise RunIdentityError("resume requires an existing run_manifest.json and last.pt")
        existing_manifest = _read_json(run_manifest_path)
        if existing_manifest.get("identity_sha256") != identity:
            raise RunIdentityError("Stage 2 run identity differs; existing run was not modified")
        checkpoint = _torch_load(last_path)
        if checkpoint.get("identity_sha256") != identity:
            raise RunIdentityError("Stage 2 last.pt identity mismatch")
    else:
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise RunIdentityError("run directory exists and is non-empty; resume exact config instead")
        checkpoint = None

    model = model_factory(args).to(device)
    _configure_source_trainability(model, args)
    parameter_counts = _count_parameters(model)
    groups, optimizer_summary = _optimizer_groups(model, args)
    optimizer = torch.optim.AdamW(groups)
    updates_per_epoch = math.ceil(len(train) / normalized["training"]["batch_size"] / normalized["training"]["update_epochs"])
    scheduler = _build_source_scheduler(
        optimizer,
        max_epochs=normalized["training"]["max_epochs"],
        updates_per_epoch=updates_per_epoch,
        warmup_epochs=normalized["training"]["warmup_epochs"],
    )
    use_bf16 = bool(args.get("use_bf16", False))
    if use_bf16 and device.type != "cuda":
        raise BCMLFStage2Error("use_bf16=true requires CUDA")
    scaler = _make_scaler(enabled=use_bf16 and device.type == "cuda")
    summary = {
        "identity_sha256": identity,
        "parameter_counts": parameter_counts,
        "optimizer": optimizer_summary,
        "micro_batch_size": normalized["training"]["batch_size"],
        "accumulation_steps": normalized["training"]["update_epochs"],
        "nominal_effective_batch_size": normalized["training"]["batch_size"] * normalized["training"]["update_epochs"],
        "updates_per_epoch": updates_per_epoch,
        "clip": {"kind": "value", "value": float(normalized["training"]["grad_clip_value"])},
        "loss": {
            "main": "five-class cross entropy",
            "bn": float(args.get("l_bn", 1.0)),
            "av": float(args.get("l_av", 1.0)),
            "text": float(args.get("l_t", 1.0)),
            "clm_gamma": float(args.get("gamma", 1.0)),
            "contrast_weight": float(args.get("lambda_fusion_contrast", 0.0)) if args.get("use_fusion_contrast", False) else 0.0,
            "contrast_temperature": float(args.get("contrast_temperature", 0.2)),
            "contrast_label_sigma": float(args.get("contrast_label_sigma", 0.8)),
            "use_clm": bool(args.get("use_clm", False)),
        },
        "tokenizer": {
            "local_directory": str(lm_directory),
            "max_token_len": int(args.max_token_len),
            "padding": "max_length",
            "truncation": True,
        },
        "model_config": str(model_config_path),
    }
    manifest["parameter_counts"] = parameter_counts
    manifest["optimizer"] = optimizer_summary
    if resume:
        existing_manifest = _read_json(run_manifest_path)
        if existing_manifest != manifest:
            raise RunIdentityError("Stage 2 model/optimizer summary differs; run was not modified")
        if not model_summary_path.is_file():
            raise RunIdentityError("existing Stage 2 run has no model_summary.json")
        if _read_json(model_summary_path) != summary:
            raise RunIdentityError("Stage 2 model summary differs; run was not modified")
        if checkpoint.get("format_version") != 1:
            raise RunIdentityError("unsupported Stage 2 checkpoint version")
        if int(checkpoint.get("seed", -1)) != seed:
            raise RunIdentityError("Stage 2 checkpoint seed mismatch")
        if int(checkpoint.get("epoch_completed", -1)) > 0 and (
            not best_path.is_file() or not validation_path.is_file()
        ):
            raise RunIdentityError("Stage 2 run is missing best model or valid predictions")
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(config_path, normalized)
        _atomic_json(environment_path, environment)
        _atomic_json(run_manifest_path, manifest)
        _atomic_json(model_summary_path, summary)

    training = normalized["training"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 2711)
    train_loader = _loader(
        train, batch_size=training["batch_size"], generator=generator,
        shuffle=True, device=device,
    )
    valid_loader = _loader(
        valid, batch_size=training["batch_size"], generator=generator,
        shuffle=False, device=device,
    )
    loss_kwargs = {
        "l_bn": float(args.get("l_bn", 1.0)),
        "l_av": float(args.get("l_av", 1.0)),
        "l_text": float(args.get("l_t", 1.0)),
        "contrast_weight": (
            float(args.get("lambda_fusion_contrast", 0.0))
            if args.get("use_fusion_contrast", False) else 0.0
        ),
        "contrast_temperature": float(args.get("contrast_temperature", 0.2)),
        "contrast_label_sigma": float(args.get("contrast_label_sigma", 0.8)),
        "use_clm": bool(args.get("use_clm", False)),
        "gamma": float(args.get("gamma", 1.0)),
    }

    best_accuracy = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    next_epoch = 0
    epoch_completed = 0
    completed = False
    if checkpoint is not None:
        try:
            model.load_state_dict(checkpoint["model_state"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            scaler.load_state_dict(checkpoint["scaler_state"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RunIdentityError("strict Stage 2 checkpoint restore failed: {}".format(exc)) from exc
        next_epoch = int(checkpoint["next_epoch"])
        epoch_completed = int(checkpoint["epoch_completed"])
        best_accuracy = float(checkpoint["best_valid_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        stale_epochs = int(checkpoint["stale_epochs"])
        history = list(checkpoint["history"])
        completed = bool(checkpoint["completed"])
        _restore_rng_state(checkpoint["rng_state"], generator, device)
        _prune_resource_rows(resources_path, epoch_completed)
        _atomic_jsonl(metrics_path, history)
        if completed:
            return {
                "run_dir": str(output),
                "seed": seed,
                "best_epoch": best_epoch,
                "best_valid_accuracy": best_accuracy,
                "completed": True,
                "resumed": True,
            }
    else:
        initial = _checkpoint_state(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            generator=generator, device=device, identity=identity, seed=seed,
            epoch=0, next_epoch=0, best_accuracy=best_accuracy, best_epoch=best_epoch,
            stale_epochs=stale_epochs, history=history, completed=False,
        )
        _atomic_torch_save(last_path, initial)

    epochs_this_call = 0
    while next_epoch < training["max_epochs"]:
        if max_epochs_this_call is not None and epochs_this_call >= max_epochs_this_call:
            break
        epoch = next_epoch + 1
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_result = _train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            args=args,
            device=device,
            batch_size=training["batch_size"],
            update_epochs=training["update_epochs"],
            grad_clip_value=float(training["grad_clip_value"]),
            use_bf16=use_bf16,
            loss_kwargs=loss_kwargs,
        )
        _synchronize(device)
        train_seconds = time.perf_counter() - train_start
        valid_start = time.perf_counter()
        valid_result = _evaluate(model, valid_loader, device)
        _synchronize(device)
        valid_seconds = time.perf_counter() - valid_start
        wall_seconds = time.perf_counter() - epoch_start

        improved = valid_result["accuracy"] > best_accuracy + 1e-12
        if improved:
            best_accuracy = float(valid_result["accuracy"])
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["losses"]["loss"],
                "train_task_loss": train_result["losses"]["task_loss"],
                "train_bn_loss": train_result["losses"]["bn_loss"],
                "train_av_loss": train_result["losses"]["av_loss"],
                "train_text_loss": train_result["losses"]["text_loss"],
                "train_lm_loss": train_result["losses"]["lm_loss"],
                "train_contrast_loss": train_result["losses"]["contrast_loss"],
                "train_accuracy": train_result["accuracy"],
                "valid_accuracy": float(valid_result["accuracy"]),
                "valid_nll": float(valid_result["nll"]),
                "best_valid_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "optimizer_updates": train_result["optimizer_updates"],
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            }
        )
        next_epoch = epoch
        epoch_completed = epoch
        completed = stale_epochs >= training["early_stop"] or next_epoch >= training["max_epochs"]
        if improved:
            _atomic_torch_save(best_path, _copy_state_to_cpu(model))
            _atomic_jsonl(validation_path, _validation_records(valid_result))
        _upsert_resource_row(
            resources_path,
            {
                "epoch": epoch,
                "device": str(device),
                "train_seconds": train_seconds,
                "valid_seconds": valid_seconds,
                "train_valid_wall_seconds": wall_seconds,
                "train_samples_processed": train_result["samples"],
                "valid_samples_processed": valid_result["n_samples"],
                "optimizer_updates": train_result["optimizer_updates"],
                "train_samples_per_second": train_result["samples"] / train_seconds if train_seconds > 0 else None,
                "valid_samples_per_second": valid_result["n_samples"] / valid_seconds if valid_seconds > 0 else None,
                **_cuda_peak_memory(device),
            },
        )
        last = _checkpoint_state(
            model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            generator=generator, device=device, identity=identity, seed=seed,
            epoch=epoch_completed, next_epoch=next_epoch, best_accuracy=best_accuracy,
            best_epoch=best_epoch, stale_epochs=stale_epochs, history=history,
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
        "best_valid_accuracy": best_accuracy,
        "last_epoch": epoch_completed,
        "completed": completed,
        "resumed": resume,
    }


def run_stage2(
    config: Mapping[str, Any],
    *,
    seed: int | None = None,
    resume: bool = False,
    max_epochs_this_call: int | None = None,
) -> list[dict[str, Any]]:
    normalized = normalize_stage2_config(config)
    if resume and seed is None:
        raise BCMLFStage2Error("resume requires an explicit --seed")
    selected = [seed] if seed is not None else list(SEEDS)
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in SEEDS for value in selected):
        raise BCMLFStage2Error("seed must be one of 13, 42, 2026")
    results = []
    for current_seed in selected:
        run_dir = Path(normalized["run_root"]) / "bcmlf-msalm-simsv2" / "seed-{}".format(current_seed)
        results.append(
            train_stage2_seed(
                normalized,
                current_seed,
                run_dir,
                resume=resume,
                max_epochs_this_call=max_epochs_this_call,
            )
        )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train BC-MLF five-class MSALM on JevAffect cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", help="JevAffect checkout; overrides config repo_root")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-call", type=int)
    args = parser.parse_args(argv)
    config = load_stage2_config(args.config, repo_root_override=args.repo_root)
    results = run_stage2(
        config,
        seed=args.seed,
        resume=args.resume,
        max_epochs_this_call=args.max_epochs_this_call,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
