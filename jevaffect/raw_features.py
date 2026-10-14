"""Pinned local encoders for raw-video feature extraction.

This module produces per-record sequences, not sentiment predictions.  It
preserves mBERT token states, WavLM frame states, DINOv2 CLS vectors, masks,
audio feature timestamps, and decoded video-frame timestamps.  The CLI writes
variable-length per-record NPZ files; a later, explicit cache-preparation step
must join canonical IDs/labels and pad sequences under the training contract.

The model loader has no network fallback.  It verifies local provenance and
weight hashes before calling Transformers with ``local_files_only=True``.
Torch and Transformers are imported lazily so boundary tests can use CPU mocks.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np


SAMPLE_RATE = 16_000
WHISPER_CHUNK_SECONDS = 30
RAW_FEATURE_SCHEMA_VERSION = 1

# These are the locally verified encoder identities already pinned by the
# project. DINOv2 and Whisper's detailed file lists are also checked against
# raw_encoder_definitions.json, whose SHA is recorded in the output manifest.
ENCODER_PINS: dict[str, dict[str, Any]] = {
    "mbert": {
        "repo": "google-bert/bert-base-multilingual-cased",
        "revision": "3f076fdb1ab68d5b2880cb87a0886f315b8146f8",
        "weight": "model.safetensors",
        "sha256": "876f584f15ebf14887dec17539c114bb99a032e96b9e72507a51c41e205337fc",
        "files": [
            "README.md", "config.json", "tokenizer_config.json", "tokenizer.json",
            "vocab.txt", "model.safetensors",
        ],
    },
    "wavlm": {
        "repo": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "weight": "pytorch_model.bin",
        "sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
        "files": ["README.md", "config.json", "preprocessor_config.json", "pytorch_model.bin"],
    },
    "dinov2-small": {
        "repo": "facebook/dinov2-small",
        "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        "weight": "model.safetensors",
        "sha256": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
        "files": ["README.md", "config.json", "preprocessor_config.json", "model.safetensors"],
        "model_card_license": "apache-2.0",
    },
    "whisper-base": {
        "repo": "openai/whisper-base",
        "revision": "e37978b90ca9030d5170a5c07aadb050351a65bb",
        "weight": "model.safetensors",
        "sha256": "07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b",
        "files": [
            "README.md", "added_tokens.json", "config.json", "generation_config.json",
            "merges.txt", "normalizer.json", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "model.safetensors",
        ],
        "model_card_license": "apache-2.0",
    },
}


class RawFeatureError(ValueError):
    """Invalid input, encoder provenance, or feature output."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RawFeatureError("{} must be a lowercase SHA-256".format(name))
    return value


def _json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RawFeatureError("could not read JSON metadata {}: {}".format(path, exc)) from exc
    if not isinstance(value, dict):
        raise RawFeatureError("metadata must be a JSON object: {}".format(path))
    return value


def _expected_encoder_definitions(path: str | Path) -> tuple[dict[str, dict[str, Any]], str]:
    path = Path(path)
    definitions = _json_load(path)
    expected = {name: ENCODER_PINS[name] for name in ("dinov2-small", "whisper-base")}
    for alias, pin in expected.items():
        actual = definitions.get(alias)
        if not isinstance(actual, dict):
            raise RawFeatureError("encoder definitions lack {!r}".format(alias))
        for key in ("repo", "revision", "weight", "sha256", "files"):
            if actual.get(key) != pin.get(key):
                raise RawFeatureError(
                    "{} {} disagrees with the source-pinned definition".format(alias, key)
                )
        if actual.get("model_card_license") != pin.get("model_card_license"):
            raise RawFeatureError("{} model-card license provenance changed".format(alias))
    return expected, sha256_file(path)


def verify_encoder_installation(
    alias: str,
    model_dir: str | Path,
    pin: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one complete local model directory against a pinned identity.

    ``provenance.json`` must say that the download completed and was verified;
    the expected listed files must exist, and the weight SHA is recomputed.
    If a provenance file includes hashes for ancillary files, those are checked
    too. No file is downloaded or repaired here.
    """
    model_dir = Path(model_dir).resolve()
    provenance_path = model_dir / "provenance.json"
    if not provenance_path.is_file():
        raise RawFeatureError("{} provenance.json is missing in {}".format(alias, model_dir))
    provenance = _json_load(provenance_path)
    if provenance.get("verified") is not True:
        raise RawFeatureError("{} local model provenance is not marked verified".format(alias))
    for key in ("repo", "revision", "weight", "sha256"):
        if provenance.get(key) != pin.get(key):
            raise RawFeatureError("{} provenance {} does not match the pinned value".format(alias, key))
    pinned_files = pin.get("files")
    declared_files = provenance.get("files")
    if not isinstance(pinned_files, list) or not isinstance(declared_files, list):
        raise RawFeatureError("{} provenance has no complete file list".format(alias))
    if declared_files != pinned_files:
        raise RawFeatureError("{} provenance file list differs from the pinned definition".format(alias))
    for relative in pinned_files:
        if not isinstance(relative, str) or Path(relative).name != relative or relative in ("", ".", ".."):
            raise RawFeatureError("{} contains an unsafe declared filename".format(alias))
        if not (model_dir / relative).is_file():
            raise RawFeatureError("{} model file is missing: {}".format(alias, relative))

    weight_name = pin.get("weight")
    if not isinstance(weight_name, str) or Path(weight_name).name != weight_name:
        raise RawFeatureError("{} pinned weight filename is unsafe".format(alias))
    weight_path = model_dir / weight_name
    expected_weight_hash = _sha256_string(pin.get("sha256"), name=alias + " pinned weight hash")
    actual_weight_hash = sha256_file(weight_path)
    if actual_weight_hash != expected_weight_hash:
        raise RawFeatureError("{} weight SHA-256 mismatch".format(alias))

    ancillary_hashes = provenance.get("files_sha256", {})
    if ancillary_hashes is not None:
        if not isinstance(ancillary_hashes, dict):
            raise RawFeatureError("{} files_sha256 must be an object".format(alias))
        for name, expected_hash in ancillary_hashes.items():
            if name not in pinned_files:
                raise RawFeatureError("{} provenance hashes an undeclared file {!r}".format(alias, name))
            expected_hash = _sha256_string(expected_hash, name=alias + " file hash")
            if sha256_file(model_dir / name) != expected_hash:
                raise RawFeatureError("{} ancillary file SHA-256 mismatch: {}".format(alias, name))

    sidecar = model_dir / (weight_name + ".sha256")
    if sidecar.is_file():
        sidecar_parts = sidecar.read_text(encoding="ascii").split()
        sidecar_hash = sidecar_parts[0] if sidecar_parts else ""
        if sidecar_hash != expected_weight_hash:
            raise RawFeatureError("{} weight SHA-256 sidecar disagrees with its pin".format(alias))

    return {
        "alias": alias,
        "repo": pin["repo"],
        "revision": pin["revision"],
        "weight": weight_name,
        "weight_sha256": actual_weight_hash,
        "provenance_sha256": sha256_file(provenance_path),
        "model_card_license": pin.get("model_card_license", provenance.get("model_card_license")),
        "files": list(pinned_files),
        "model_dir_name": model_dir.name,
        "verified": True,
    }


@dataclass
class EncoderBundle:
    """One-time loaded models/processors reused across an entire batch."""

    text_tokenizer: Any
    text_model: Any
    audio_processor: Any
    audio_model: Any
    vision_processor: Any
    vision_model: Any
    whisper_processor: Any = None
    whisper_model: Any = None
    torch_module: Any = None
    device: str = "cpu"
    provenance: Mapping[str, Any] = field(default_factory=dict)


def load_pretrained_models(
    model_root: str | Path,
    definitions_path: str | Path,
    *,
    device: str = "cpu",
    enable_asr: bool = True,
) -> EncoderBundle:
    """Load the four pinned models once using local files only.

    The model directory aliases must be ``mbert``, ``wavlm``,
    ``dinov2-small``, and (when ASR is enabled) ``whisper-base``. The required
    Transformers version is pinned by the parent project's ``encoders`` extra.
    """
    model_root = Path(model_root).resolve()
    definition_pins, definitions_sha256 = _expected_encoder_definitions(definitions_path)
    aliases = ["mbert", "wavlm", "dinov2-small"] + (["whisper-base"] if enable_asr else [])
    verified: dict[str, Any] = {}
    for alias in aliases:
        pin = ENCODER_PINS[alias]
        if alias in definition_pins:
            pin = definition_pins[alias]
        verified[alias] = verify_encoder_installation(alias, model_root / alias, pin)

    try:
        import torch
        from transformers import (
            AutoFeatureExtractor,
            AutoImageProcessor,
            AutoModel,
            AutoTokenizer,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install the project's pinned torch and transformers encoder dependencies before inference."
        ) from exc

    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Every load is local_files_only: there is no implicit network fetch.
    text_tokenizer = AutoTokenizer.from_pretrained(
        str(model_root / "mbert"), local_files_only=True, trust_remote_code=False
    )
    text_model = AutoModel.from_pretrained(
        str(model_root / "mbert"), local_files_only=True, trust_remote_code=False
    )
    audio_processor = AutoFeatureExtractor.from_pretrained(
        str(model_root / "wavlm"), local_files_only=True, trust_remote_code=False
    )
    audio_model = AutoModel.from_pretrained(
        str(model_root / "wavlm"), local_files_only=True, trust_remote_code=False
    )
    vision_processor = _load_dinov2_processor(AutoImageProcessor, model_root / "dinov2-small")
    vision_model = AutoModel.from_pretrained(
        str(model_root / "dinov2-small"), local_files_only=True, trust_remote_code=False
    )
    whisper_processor = None
    whisper_model = None
    if enable_asr:
        whisper_processor = WhisperProcessor.from_pretrained(
            str(model_root / "whisper-base"), local_files_only=True, trust_remote_code=False
        )
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            str(model_root / "whisper-base"), local_files_only=True, trust_remote_code=False
        )

    models = (text_model, audio_model, vision_model) + ((whisper_model,) if whisper_model is not None else ())
    for model in models:
        model.eval()
        model.to(device)
    dimensions = {
        "text": int(text_model.config.hidden_size),
        "audio": int(audio_model.config.hidden_size),
        "vision": int(vision_model.config.hidden_size),
    }
    required_dims = {"text": 768, "audio": 768, "vision": 384}
    if dimensions != required_dims:
        raise RawFeatureError(
            "pinned encoder dimensions are {}, expected {}".format(dimensions, required_dims)
        )
    return EncoderBundle(
        text_tokenizer=text_tokenizer,
        text_model=text_model,
        audio_processor=audio_processor,
        audio_model=audio_model,
        vision_processor=vision_processor,
        vision_model=vision_model,
        whisper_processor=whisper_processor,
        whisper_model=whisper_model,
        torch_module=torch,
        device=device,
        provenance={
            "encoders": verified,
            "definitions_path_name": Path(definitions_path).name,
            "definitions_sha256": definitions_sha256,
            "local_files_only": True,
            "dinov2_image_processor_use_fast": False,
            "dimensions": dimensions,
        },
    )


def _load_dinov2_processor(auto_image_processor: Any, model_dir: str | Path) -> Any:
    """Load the slow image processor explicitly for repeatable preprocessing."""
    return auto_image_processor.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=False,
    )


def _hidden_size(model: Any, name: str, expected: int) -> int:
    config = getattr(model, "config", None)
    value = getattr(config, "hidden_size", None)
    if value is None:
        value = getattr(config, "d_model", None)
    if value is None or int(value) != expected:
        raise RawFeatureError("{} encoder must expose hidden dimension {}".format(name, expected))
    return int(value)


def conv_feature_geometry(kernels: Sequence[int], strides: Sequence[int]) -> tuple[int, int]:
    """Return (receptive_field_samples, output_stride_samples) for valid 1-D convs."""
    if len(kernels) == 0 or len(kernels) != len(strides):
        raise RawFeatureError("WavLM conv_kernel and conv_stride must be equal non-empty lists")
    receptive_field = 1
    jump = 1
    for kernel, stride in zip(kernels, strides):
        kernel = int(kernel)
        stride = int(stride)
        if kernel <= 0 or stride <= 0:
            raise RawFeatureError("WavLM convolution kernels and strides must be positive")
        receptive_field += (kernel - 1) * jump
        jump *= stride
    return receptive_field, jump


def conv_feature_timestamps(
    sample_count: int,
    kernels: Sequence[int],
    strides: Sequence[int],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Centers of WavLM convolutional receptive fields in waveform seconds."""
    receptive_field, jump = conv_feature_geometry(kernels, strides)
    if sample_count < receptive_field:
        return np.empty(0, dtype=np.float64)
    output_count = (int(sample_count) - receptive_field) // jump + 1
    centers = (np.arange(output_count, dtype=np.float64) * jump) + (receptive_field - 1) / 2
    return centers / float(sample_rate)


def whisper_chunk_ranges(
    sample_count: int,
    *,
    sample_rate: int = SAMPLE_RATE,
    chunk_seconds: int = WHISPER_CHUNK_SECONDS,
) -> list[tuple[int, int]]:
    """Partition an audio signal into complete, non-overlapping ASR windows."""
    if sample_count < 0 or sample_rate <= 0 or chunk_seconds <= 0:
        raise RawFeatureError("invalid audio size or Whisper chunk parameters")
    chunk_size = int(sample_rate) * int(chunk_seconds)
    return [(start, min(start + chunk_size, int(sample_count)))
            for start in range(0, int(sample_count), chunk_size)]


def _move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _output_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _last_hidden(output: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, Mapping) and "last_hidden_state" in output:
        return output["last_hidden_state"]
    raise RawFeatureError("encoder output has no last_hidden_state")


def _inference_context(models: EncoderBundle) -> Any:
    torch_module = models.torch_module
    return torch_module.inference_mode() if torch_module is not None else contextlib.nullcontext()


def _validate_float_features(name: str, features: Any, dimension: int) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise RawFeatureError(
            "{} features must be [T,{}], got {}".format(name, dimension, array.shape)
        )
    if not np.isfinite(array).all():
        raise RawFeatureError("{} encoder returned nonfinite features".format(name))
    return np.ascontiguousarray(array)


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("id")
    if not isinstance(value, str) or not value or value.isspace():
        raise RawFeatureError("each raw-video record needs a non-empty string id")
    if "\x00" in value:
        raise RawFeatureError("record id contains a NUL character")
    return value


class RawVideoFeatureExtractor:
    """Extract all available modality sequences with one reusable model bundle."""

    def __init__(
        self,
        models: EncoderBundle,
        *,
        device: Optional[str] = None,
        decoder: Optional[Callable[..., Mapping[str, Any]]] = None,
        fps: float = 2.0,
        frame_size: int = 224,
        max_seconds: float = 120.0,
        frame_batch_size: int = 16,
        max_text_tokens: int = 256,
    ) -> None:
        self.models = models
        self.device = device or models.device
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        if fps <= 0 or frame_size <= 0 or max_seconds <= 0:
            raise ValueError("fps, frame_size, and max_seconds must be positive")
        if frame_batch_size <= 0 or max_text_tokens < 2:
            raise ValueError("frame_batch_size must be positive and text limit at least 2")
        self.fps = float(fps)
        self.frame_size = int(frame_size)
        self.max_seconds = float(max_seconds)
        self.frame_batch_size = int(frame_batch_size)
        self.max_text_tokens = int(max_text_tokens)
        self._decoder = decoder
        torch_module = self.models.torch_module
        self._cuda_measurement_enabled = bool(
            self.device.startswith("cuda")
            and torch_module is not None
            and getattr(torch_module, "cuda", None) is not None
            and torch_module.cuda.is_available()
        )
        if self.device.startswith("cuda") and not self._cuda_measurement_enabled:
            raise RuntimeError("CUDA device was selected without an available torch CUDA runtime")
        self.text_dim = _hidden_size(models.text_model, "mBERT", 768)
        self.audio_dim = _hidden_size(models.audio_model, "WavLM", 768)
        self.vision_dim = _hidden_size(models.vision_model, "DINOv2-small", 384)

    def _synchronize_cuda(self) -> None:
        if self._cuda_measurement_enabled:
            self.models.torch_module.cuda.synchronize(self.device)

    def _begin_gpu_measurement(self) -> dict[str, Optional[int]]:
        if not self._cuda_measurement_enabled:
            return {"allocated_before_bytes": None, "reserved_before_bytes": None}
        cuda = self.models.torch_module.cuda
        allocated = int(cuda.memory_allocated(self.device))
        reserved = int(cuda.memory_reserved(self.device))
        cuda.reset_peak_memory_stats(self.device)
        return {"allocated_before_bytes": allocated, "reserved_before_bytes": reserved}

    def _finish_gpu_measurement(self, start: Mapping[str, Optional[int]]) -> dict[str, Any]:
        if not self._cuda_measurement_enabled:
            return {
                "measured": False,
                "device": None,
                "allocated_before_bytes": None,
                "peak_allocated_bytes": None,
                "peak_allocated_delta_bytes": None,
                "reserved_before_bytes": None,
                "peak_reserved_bytes": None,
            }
        cuda = self.models.torch_module.cuda
        peak_allocated = int(cuda.max_memory_allocated(self.device))
        peak_reserved = int(cuda.max_memory_reserved(self.device))
        before = int(start["allocated_before_bytes"] or 0)
        return {
            "measured": True,
            "device": self.device,
            "allocated_before_bytes": before,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_delta_bytes": max(0, peak_allocated - before),
            "reserved_before_bytes": start["reserved_before_bytes"],
            "peak_reserved_bytes": peak_reserved,
        }

    def _decode(self, video_path: Path, audio_path: Path | None = None, *, allow_missing_video: bool = False) -> Mapping[str, Any]:
        decoder = self._decoder
        if decoder is None:
            try:
                from jevaffect.media import decode_video
            except ImportError as exc:
                raise RuntimeError(
                    "jevaffect.media.decode_video is required; use the parent project's media module"
                ) from exc
            decoder = decode_video
        return decoder(
            video_path,
            fps=self.fps,
            size=self.frame_size,
            max_seconds=self.max_seconds,
            **({'audio_path':audio_path,'allow_missing_video':allow_missing_video} if audio_path is not None else {}),
        )

    def _encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        if not isinstance(text, str):
            raise RawFeatureError("transcript must be text")
        if not text.strip():
            return (
                np.empty((0, self.text_dim), dtype=np.float32),
                np.empty(0, dtype=np.bool_),
                np.empty(0, dtype=np.int64),
                False,
            )
        tokenizer = self.models.text_tokenizer
        full_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        if isinstance(full_ids, np.ndarray):
            full_length = int(full_ids.shape[-1])
        elif full_ids and isinstance(full_ids[0], (list, tuple)):
            full_length = len(full_ids[0])
        else:
            full_length = len(full_ids)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_tokens,
            padding=False,
        )
        inputs = _move_batch_to_device(encoded, self.device)
        if "input_ids" not in inputs or "attention_mask" not in inputs:
            raise RawFeatureError("mBERT tokenizer must return input_ids and attention_mask")
        with _inference_context(self.models):
            output = self.models.text_model(**inputs)
        states = _output_array(_last_hidden(output))
        if states.ndim != 3 or states.shape[0] != 1:
            raise RawFeatureError("mBERT last_hidden_state must have shape [1,T,D]")
        features = _validate_float_features("text", states[0], self.text_dim)
        mask = _output_array(inputs["attention_mask"])[0].astype(np.bool_, copy=False)
        token_ids = _output_array(inputs["input_ids"])[0].astype(np.int64, copy=False)
        if features.shape[0] != len(mask) or len(token_ids) != len(mask):
            raise RawFeatureError("mBERT states, token IDs, and attention mask lengths differ")
        if not bool(mask.any()):
            return features, mask, token_ids, False
        return features, mask, token_ids, full_length > len(mask)

    def _encode_audio(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1:
            raise RawFeatureError("decoded audio must be a one-dimensional mono waveform")
        if not np.isfinite(waveform).all():
            raise RawFeatureError("decoded audio contains nonfinite samples")
        if waveform.size == 0:
            return (
                np.empty((0, self.audio_dim), dtype=np.float32),
                np.empty(0, dtype=np.bool_),
                np.empty(0, dtype=np.float64),
                "missing_or_empty_stream",
            )
        config = self.models.audio_model.config
        kernels = getattr(config, "conv_kernel", None)
        strides = getattr(config, "conv_stride", None)
        if kernels is None or strides is None:
            raise RawFeatureError("WavLM config lacks conv_kernel/conv_stride for timestamps")
        expected_times = conv_feature_timestamps(waveform.size, kernels, strides)
        if expected_times.size == 0:
            return (
                np.empty((0, self.audio_dim), dtype=np.float32),
                np.empty(0, dtype=np.bool_),
                expected_times,
                "too_short_for_wavlm",
            )
        encoded = self.models.audio_processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = _move_batch_to_device(encoded, self.device)
        if "input_values" not in inputs:
            raise RawFeatureError("WavLM feature extractor must return input_values")
        # One clip per forward pass means no batch padding is introduced.
        with _inference_context(self.models):
            output = self.models.audio_model(input_values=inputs["input_values"])
        states = _output_array(_last_hidden(output))
        if states.ndim != 3 or states.shape[0] != 1:
            raise RawFeatureError("WavLM last_hidden_state must have shape [1,T,D]")
        features = _validate_float_features("audio", states[0], self.audio_dim)
        if len(features) != len(expected_times):
            raise RawFeatureError(
                "WavLM output length {} disagrees with convolution timing length {}".format(
                    len(features), len(expected_times)
                )
            )
        return features, np.ones(len(features), dtype=np.bool_), expected_times, "present"

    def _encode_vision(
        self, frames: np.ndarray, frame_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        frames = np.asarray(frames)
        frame_times = np.asarray(frame_times, dtype=np.float64)
        if frames.ndim != 4 or (frames.shape[0] and frames.shape[-1] != 3):
            raise RawFeatureError("decoded frames must have shape [T,H,W,3]")
        if frame_times.shape != (frames.shape[0],) or not np.isfinite(frame_times).all():
            raise RawFeatureError("frame timestamp count/nonfinite values do not match decoded frames")
        if frames.shape[0] == 0:
            return (
                np.empty((0, self.vision_dim), dtype=np.float32),
                np.empty(0, dtype=np.bool_),
                frame_times,
                "missing_or_empty_stream",
            )
        if frames.dtype != np.uint8:
            if frames.dtype.kind not in "uif" or not np.isfinite(frames).all():
                raise RawFeatureError("decoded video frames must be finite RGB pixel arrays")
            if np.any(frames < 0) or np.any(frames > 255):
                raise RawFeatureError("decoded RGB values fall outside 0..255")
            frames = frames.astype(np.uint8)
        output_rows = []
        for start in range(0, len(frames), self.frame_batch_size):
            frame_chunk = frames[start:start + self.frame_batch_size]
            processed = self.models.vision_processor(images=list(frame_chunk), return_tensors="pt")
            inputs = _move_batch_to_device(processed, self.device)
            if "pixel_values" not in inputs:
                raise RawFeatureError("DINOv2 image processor must return pixel_values")
            with _inference_context(self.models):
                output = self.models.vision_model(pixel_values=inputs["pixel_values"])
            states = _output_array(_last_hidden(output))
            if states.ndim != 3 or states.shape[0] != len(frame_chunk) or states.shape[1] < 1:
                raise RawFeatureError("DINOv2 last_hidden_state must have shape [T,patches,D]")
            # The published small encoder has a 384-wide CLS representation.
            output_rows.append(_validate_float_features("vision", states[:, 0, :], self.vision_dim))
        features = np.concatenate(output_rows, axis=0)
        return features, np.ones(len(features), dtype=np.bool_), frame_times, "present"

    def _transcribe(self, waveform: np.ndarray, language: Optional[str]) -> dict[str, Any]:
        processor = self.models.whisper_processor
        model = self.models.whisper_model
        if processor is None or model is None:
            return {"text": "", "status": "asr_encoder_not_loaded", "chunks": []}
        ranges = whisper_chunk_ranges(len(waveform))
        if not ranges:
            return {"text": "", "status": "no_audio_for_asr", "chunks": []}
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise RawFeatureError("asr_language must be a non-empty language code or null")
        chunk_rows: list[dict[str, Any]] = []
        for start, end in ranges:
            chunk = np.asarray(waveform[start:end], dtype=np.float32)
            prepared = processor(
                chunk,
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = _move_batch_to_device(prepared, self.device)
            if "input_features" not in inputs:
                raise RawFeatureError("Whisper processor must return input_features")
            if "attention_mask" not in inputs:
                raise RawFeatureError("Whisper processor must return attention_mask for generation")
            generation_args: dict[str, Any] = {
                "task": "transcribe",
                "do_sample": False,
                # Explicit task/language values supersede the model's stored
                # forced_decoder_ids and avoid Transformers' conflict warning.
                "forced_decoder_ids": None,
            }
            if language is not None:
                generation_args["language"] = language
            with _inference_context(self.models):
                generated = model.generate(
                    inputs["input_features"],
                    attention_mask=inputs["attention_mask"],
                    **generation_args,
                )
            decoded = processor.batch_decode(generated, skip_special_tokens=True)
            text = decoded[0].strip() if decoded else ""
            chunk_rows.append({
                "start_seconds": start / SAMPLE_RATE,
                "end_seconds": end / SAMPLE_RATE,
                "text": text,
            })
        # Join separately decoded windows with a separator. Every sample is
        # covered exactly once; no >30-second tail is silently truncated.
        transcript = " ".join(row["text"] for row in chunk_rows if row["text"])
        return {
            "text": transcript,
            "status": "present" if transcript else "empty_asr_output",
            "chunks": chunk_rows,
            "task": "transcribe",
            "chunk_seconds": WHISPER_CHUNK_SECONDS,
            "overlap_seconds": 0,
            "language": language,
        }

    def _model_results_verified(self) -> bool:
        encoders = self.models.provenance.get("encoders")
        return bool(
            not self.models.provenance.get("fixture", False)
            and isinstance(encoders, Mapping)
            and encoders
            and all(
                isinstance(item, Mapping) and item.get("verified") is True
                for item in encoders.values()
            )
        )

    def extract_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Return arrays and JSON-safe metadata for one ``id``/``video_path``."""
        if not isinstance(record, Mapping):
            raise RawFeatureError("record must be a mapping")
        sample_id = _record_id(record)
        video_path_value = record.get("video_path")
        if not isinstance(video_path_value, str) or not video_path_value:
            raise RawFeatureError("record {!r} needs a video_path".format(sample_id))
        video_path = Path(video_path_value).expanduser().resolve()
        raw_audio_path = record.get('audio_path')
        if raw_audio_path is not None and (not isinstance(raw_audio_path,str) or not raw_audio_path):
            raise RawFeatureError('audio_path must be a nonempty path when provided')
        audio_path = Path(raw_audio_path).expanduser().resolve() if raw_audio_path is not None else None
        allow_missing_video = record.get('allow_missing_video',False)
        if not isinstance(allow_missing_video,bool):
            raise RawFeatureError('allow_missing_video must be a boolean')
        provided_transcript = record.get("transcript")
        if provided_transcript is not None and not isinstance(provided_transcript, str):
            raise RawFeatureError("provided transcript for {!r} must be a string".format(sample_id))

        # Synchronize before the record clock so queued work from a preceding
        # record cannot inflate this record's latency.
        self._synchronize_cuda()
        record_started = time.perf_counter()
        decode_started = time.perf_counter()
        decoded = self._decode(video_path,audio_path,allow_missing_video=allow_missing_video)
        decode_wall_seconds = time.perf_counter() - decode_started
        if not isinstance(decoded, Mapping) or "audio" not in decoded or "frames" not in decoded:
            raise RawFeatureError("media decoder must return audio and frames")
        decode_meta = decoded.get("metadata", {})
        if not isinstance(decode_meta, Mapping):
            raise RawFeatureError("media decoder metadata must be an object")
        waveform = np.asarray(decoded["audio"], dtype=np.float32)
        frames = np.asarray(decoded["frames"])
        frame_times = np.asarray(decoded.get("frame_times", []), dtype=np.float64)
        sample_rate = decode_meta.get("audio_sample_rate", SAMPLE_RATE)
        if int(sample_rate) != SAMPLE_RATE:
            raise RawFeatureError(
                "media decoder returned {} Hz audio; expected {} Hz".format(sample_rate, SAMPLE_RATE)
            )
        if frame_times.ndim != 1 or not np.isfinite(frame_times).all():
            raise RawFeatureError("decoded frame timestamps must be a finite one-dimensional sequence")
        if len(frame_times) > 1 and np.any(np.diff(frame_times) < 0):
            raise RawFeatureError("decoded frame timestamps are not monotonic")
        if frames.ndim == 4 and frames.shape[0] and frames.shape[1:3] != (self.frame_size, self.frame_size):
            raise RawFeatureError(
                "decoded frames have spatial size {}; expected {}x{}".format(
                    frames.shape[1:3], self.frame_size, self.frame_size
                )
            )
        source_video_sha = decode_meta.get("source_sha256")
        if source_video_sha is None and decode_meta.get('video_status')=='missing_file' and allow_missing_video:
            source_video_sha = None
        else:
            if source_video_sha is None:
                source_video_sha = sha256_file(video_path)
            source_video_sha = _sha256_string(source_video_sha, name="source video SHA-256")

        self._synchronize_cuda()
        gpu_memory_start = self._begin_gpu_measurement()
        feature_started = time.perf_counter()
        audio_features, audio_mask, audio_times, audio_status = self._encode_audio(waveform)
        vision_features, vision_mask, vision_times, vision_status = self._encode_vision(frames, frame_times)

        asr_language = record.get("asr_language")
        asr_result: dict[str, Any] = {"text": "", "status": "not_requested", "chunks": []}
        if provided_transcript is not None and provided_transcript.strip():
            transcript = provided_transcript
            transcript_source = "provided"
        elif self.models.whisper_model is not None and self.models.whisper_processor is not None and waveform.size:
            asr_result = self._transcribe(waveform, asr_language)
            transcript = asr_result["text"]
            transcript_source = "whisper" if transcript else "whisper_empty"
        else:
            transcript = ""
            transcript_source = "missing"
            if provided_transcript is not None:
                asr_result["status"] = "empty_provided_transcript"
            elif waveform.size == 0:
                asr_result["status"] = "no_audio_for_asr"
            else:
                asr_result["status"] = "asr_encoder_not_loaded"

        text_features, text_mask, text_token_ids, text_truncated = self._encode_text(transcript)
        # The timing includes host preprocessing and completes only after all
        # asynchronous model work for this sample has finished.
        self._synchronize_cuda()
        feature_wall_seconds = time.perf_counter() - feature_started
        gpu_memory = self._finish_gpu_measurement(gpu_memory_start)
        record_wall_seconds = time.perf_counter() - record_started
        arrays = {
            "text": text_features,
            "text_mask": text_mask,
            "text_token_ids": text_token_ids,
            "audio": audio_features,
            "audio_mask": audio_mask,
            "audio_times": audio_times,
            "vision": vision_features,
            "vision_mask": vision_mask,
            "vision_times": vision_times,
        }
        metadata = {
            "id": sample_id,
            "source_video_path": str(video_path),
            "source_video_sha256": source_video_sha,
            "decode": dict(decode_meta),
            "preprocessing": {
                "audio_sample_rate": SAMPLE_RATE,
                "video_fps": self.fps,
                "video_frame_size": self.frame_size,
                "max_seconds": self.max_seconds,
                "mbert_max_tokens": self.max_text_tokens,
                "text_truncated": text_truncated,
                "video_frame_timestamps_kind": decode_meta.get(
                    "frame_timestamps_kind", "decoder_timeline"
                ),
                "audio_feature_times": "centers of WavLM convolutional receptive fields",
            },
            "transcript": transcript,
            "transcript_source": transcript_source,
            "asr": asr_result,
            "modalities": {
                "text": {"valid": bool(text_mask.any()), "status": "present" if text_mask.any() else "missing",
                         "length": int(len(text_features)), "dim": self.text_dim,
                         "mask_semantics": "true for mBERT positions incl. special tokens"},
                "audio": {"valid": bool(audio_mask.any()), "status": audio_status,
                          "length": int(len(audio_features)), "dim": self.audio_dim,
                          "mask_semantics": "true for each WavLM output frame"},
                "vision": {"valid": bool(vision_mask.any()), "status": vision_status,
                           "length": int(len(vision_features)), "dim": self.vision_dim,
                           "mask_semantics": "true for each decoded video frame"},
            },
            "encoder_provenance": dict(self.models.provenance),
            "performance": {
                "record_wall_seconds": record_wall_seconds,
                "decode_wall_seconds": decode_wall_seconds,
                "feature_pipeline_wall_seconds": feature_wall_seconds,
                "feature_pipeline_gpu_synchronized": self._cuda_measurement_enabled,
                "gpu_memory": gpu_memory,
                "timing_scope": (
                    "record includes media decode and all feature work; feature pipeline includes host preprocessing "
                    "and is CUDA-synchronized when a CUDA runtime is active"
                ),
            },
            "prediction": None,
            "fixture": bool(self.models.provenance.get("fixture", False)),
            "model_results": self._model_results_verified(),
            "feature_stage": (
                "CPU mock fixture; not model results"
                if self.models.provenance.get("fixture", False)
                else "pretrained raw-video features; no trained sentiment head"
            ),
        }
        return {"arrays": arrays, "metadata": metadata}


def read_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Read strict JSONL records; relative video paths are relative to the list."""
    path = Path(path).resolve()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RawFeatureError("invalid JSONL line {}: {}".format(line_number, exc)) from exc
                if not isinstance(row, dict):
                    raise RawFeatureError("JSONL line {} must be an object".format(line_number))
                sample_id = _record_id(row)
                if sample_id in seen:
                    raise RawFeatureError("duplicate input record ID {!r}".format(sample_id))
                seen.add(sample_id)
                video_path = row.get("video_path")
                if not isinstance(video_path, str) or not video_path:
                    raise RawFeatureError("JSONL record {!r} needs video_path".format(sample_id))
                candidate = Path(video_path).expanduser()
                if not candidate.is_absolute():
                    candidate = path.parent / candidate
                row["video_path"] = str(candidate.resolve())
                records.append(row)
    except OSError as exc:
        raise RawFeatureError("could not read input JSONL {}: {}".format(path, exc)) from exc
    if not records:
        raise RawFeatureError("input JSONL contains no records")
    return records


def _record_filename(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest() + ".npz"


def write_raw_feature_cache(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    extractor: RawVideoFeatureExtractor,
    *,
    input_manifest_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Write a new, immutable per-record feature bundle directory.

    Each NPZ contains numeric arrays only and can be opened with
    ``allow_pickle=False``. This output deliberately has no labels or canonical
    train/valid/test contract; it is a raw modality feature stage.
    """
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError("refusing to overwrite existing feature output: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name + ".staging-", dir=str(target.parent)))
    try:
        records_dir = stage / "records"
        records_dir.mkdir()
        index_rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for record in records:
            result = extractor.extract_record(record)
            meta = result["metadata"]
            sample_id = _record_id(record)
            if sample_id in seen:
                raise RawFeatureError("duplicate record ID {!r}".format(sample_id))
            seen.add(sample_id)
            rel = Path("records") / _record_filename(sample_id)
            artifact_path = stage / rel
            with artifact_path.open("wb") as handle:
                np.savez_compressed(handle, **result["arrays"])
            index_rows.append({
                "id": sample_id,
                "artifact": rel.as_posix(),
                "artifact_sha256": sha256_file(artifact_path),
                "metadata": meta,
            })
        if not index_rows:
            raise RawFeatureError("cannot write an empty raw feature cache")
        index_path = stage / "index.jsonl"
        with index_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in index_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        input_sha = sha256_file(input_manifest_path) if input_manifest_path is not None else None
        fixture_flags = [bool(row["metadata"].get("fixture")) for row in index_rows]
        model_flags = [bool(row["metadata"].get("model_results")) for row in index_rows]
        contract = {
            "schema_version": RAW_FEATURE_SCHEMA_VERSION,
            "cache_kind": "raw_video_feature_records",
            "status": "ready_for_canonical_cache_join",
            "sample_count": len(index_rows),
            "input_manifest_sha256": input_sha,
            "encoder_provenance": dict(extractor.models.provenance),
            "performance_metrics": {
                "per_record": [
                    "record_wall_seconds", "decode_wall_seconds", "feature_pipeline_wall_seconds",
                    "CUDA-synchronized feature timing when using a real CUDA runtime",
                    "peak allocated/reserved CUDA memory scoped to each feature pipeline",
                ],
            },
            "feature_arrays": {
                "text": "per-record [T,768] mBERT last_hidden_state; includes special tokens",
                "text_mask": "per-record bool attention mask aligned to text",
                "text_token_ids": "per-record int64 mBERT IDs aligned to text",
                "audio": "per-record [T,768] WavLM last_hidden_state",
                "audio_mask": "per-record bool mask aligned to audio",
                "audio_times": "per-record float64 seconds at WavLM receptive-field centers",
                "vision": "per-record [T,384] DINOv2-small CLS state",
                "vision_mask": "per-record bool mask aligned to vision",
                "vision_times": "per-record float64 FFmpeg frame timestamps",
            },
            "files": {
                "index.jsonl": sha256_file(index_path),
                "records": [row["artifact"] for row in index_rows],
            },
            "labels_included": False,
            "canonical_ids_and_splits_joined": False,
            "trained_sentiment_head_included": False,
            "fixture_results": any(fixture_flags),
            "all_records_are_fixtures": all(fixture_flags),
            "model_results": all(model_flags),
        }
        if contract["all_records_are_fixtures"]:
            contract["status"] = "fixture_only_not_model_results"
        elif contract["fixture_results"]:
            contract["status"] = "mixed_fixture_and_model_results"
        contract_path = stage / "meta.json"
        contract_path.write_text(
            json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        contract["meta_sha256"] = sha256_file(contract_path)
        # A compact sidecar avoids a self-referential hash in meta.json.
        (stage / "meta.sha256").write_text(
            "{}  meta.json\n".format(contract["meta_sha256"]), encoding="ascii", newline="\n"
        )
        if target.exists():
            raise FileExistsError("feature output appeared during extraction: {}".format(target))
        os.replace(stage, target)
        return contract
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
