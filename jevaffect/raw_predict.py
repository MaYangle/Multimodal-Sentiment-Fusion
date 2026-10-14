"""Safe checkpoint validation and whole-raw-video pilot inference.

This wrapper is intentionally narrow: it accepts JSONL records, runs the
parent project's pinned raw-feature extractor, pads/truncates to the exact
training cache schema, loads the selected ``best.pt`` with
``weights_only=True``, and emits pilot probabilities plus input/timing
metadata. It does not export attention weights, claim interpretability, or
implement a competition submission bridge.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np


MODALITIES = ("text", "audio", "vision")
ALMT_PADDING_BEHAVIOR = "ALMT adapter consumes every fixed sequence position; lengths are not masked."
TEXT_PADDING_BEHAVIOR = "Text classifier uses configured CLS or length-masked mean pooling."


class RawPredictionError(RuntimeError):
    """A run artifact or raw record does not meet the trained model contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RawPredictionError("cannot read {} {}: {}".format(label, path, exc)) from exc
    if not isinstance(value, dict):
        raise RawPredictionError("{} must contain a JSON object: {}".format(label, path))
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RawPredictionError("{} must be a positive integer".format(label))
    return value


def _sha_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RawPredictionError("{} must be a lowercase 64-character SHA-256".format(label))
    return value


def _canonical_class_order(classes: Sequence[str]) -> tuple[str, ...]:
    if isinstance(classes, (str, bytes)):
        raise RawPredictionError("class order must be a sequence of five class names")
    result = tuple(classes)
    if len(result) != 5 or any(not isinstance(value, str) for value in result):
        raise RawPredictionError("class order must contain five unique strings")
    if len(set(result)) != 5:
        raise RawPredictionError("class order must contain five unique strings")
    return result


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    manifest_path: Path
    cache_meta_path: Path
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    cache_meta: Mapping[str, Any]
    model_name: str
    model_params: Mapping[str, Any]
    feature_dims: Mapping[str, int]
    sequence_lengths: Mapping[str, int]
    required_modalities: tuple[str, ...]
    class_order: tuple[str, ...]
    identity_sha256: str
    config_sha256: str
    cache_meta_sha256: str
    checkpoint_sha256: str


def load_run_spec(
    run_dir: str | Path,
    cache_meta_path: str | Path,
    *,
    class_order: Sequence[str],
) -> RunSpec:
    """Verify run/config/cache identity and shape metadata before loading weights."""
    classes = _canonical_class_order(class_order)
    root = Path(run_dir).expanduser().resolve()
    cache_path = Path(cache_meta_path).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    config_path = root / "config.json"
    checkpoint_path = root / "best.pt"
    for label, path in (
        ("run_manifest.json", manifest_path),
        ("config.json", config_path),
        ("best.pt", checkpoint_path),
        ("cache meta.json", cache_path),
    ):
        if not path.is_file():
            raise RawPredictionError("required {} is missing: {}".format(label, path))

    manifest = _read_json_object(manifest_path, "run manifest")
    config = _read_json_object(config_path, "run config")
    cache_meta = _read_json_object(cache_path, "cache metadata")
    if manifest.get("format_version") != 1:
        raise RawPredictionError("unsupported run manifest format_version")

    expected_config_sha = _sha_string(manifest.get("config_sha256"), "run manifest config_sha256")
    actual_config_sha = hashlib.sha256(_canonical_json(config)).hexdigest()
    if actual_config_sha != expected_config_sha:
        raise RawPredictionError("config.json does not match run_manifest.json config_sha256")

    identity = _sha_string(manifest.get("identity_sha256"), "run manifest identity_sha256")
    cache_fingerprint = manifest.get("cache_fingerprint")
    environment = manifest.get("environment")
    seed = manifest.get("seed")
    implementation_sha = _sha_string(
        manifest.get("implementation_sha256"), "run manifest implementation_sha256"
    )
    if not isinstance(cache_fingerprint, Mapping) or not isinstance(environment, Mapping):
        raise RawPredictionError("run manifest lacks cache_fingerprint/environment objects")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RawPredictionError("run manifest seed must be a non-negative integer")
    identity_payload = {
        "config_sha256": expected_config_sha,
        "cache_fingerprint": dict(cache_fingerprint),
        "implementation_sha256": implementation_sha,
        "environment": dict(environment),
        "seed": seed,
    }
    if hashlib.sha256(_canonical_json(identity_payload)).hexdigest() != identity:
        raise RawPredictionError("run_manifest identity_sha256 is internally inconsistent")

    config_model = config.get("model")
    if not isinstance(config_model, Mapping):
        raise RawPredictionError("config.json model must be an object")
    model_name = config_model.get("name")
    if model_name not in ("text_mlp", "almt") or manifest.get("model") != model_name:
        raise RawPredictionError("run manifest and config model names do not match a supported model")
    model_params = config_model.get("params", {})
    if not isinstance(model_params, Mapping):
        raise RawPredictionError("config.json model.params must be an object")
    for key in ("dataset", "language"):
        if not isinstance(config.get(key), str) or manifest.get(key) != config[key]:
            raise RawPredictionError("run manifest and config {} do not match".format(key))
    seeds = config.get("seeds")
    if not isinstance(seeds, list) or seed not in seeds:
        raise RawPredictionError("checkpoint run seed is absent from config.json seeds")
    if manifest.get("class_order") != list(classes):
        raise RawPredictionError("run manifest class_order differs from the project protocol")

    cache_meta_sha = sha256_file(cache_path)
    if cache_fingerprint.get("meta_sha256") != cache_meta_sha:
        raise RawPredictionError("cache meta.json SHA-256 differs from the trained run fingerprint")
    source_sha = _sha_string(cache_meta.get("source_sha256"), "cache source_sha256")
    if cache_fingerprint.get("source_hash") != source_sha:
        raise RawPredictionError("cache source hash differs from the trained run fingerprint")
    if cache_meta.get("schema_version") != 1 or cache_meta.get("text_features_ready") is not True:
        raise RawPredictionError("cache metadata is not a supported contextual-feature cache")
    cache_classes = cache_meta.get("class_order")
    label_contract = cache_meta.get("label_contract")
    if label_contract is not None:
        if not isinstance(label_contract, Mapping):
            raise RawPredictionError("cache label_contract must be an object when present")
        contract_classes = label_contract.get("class_order")
        if contract_classes is not None:
            if cache_classes is not None and cache_classes != contract_classes:
                raise RawPredictionError("cache top-level and label_contract class orders disagree")
            cache_classes = contract_classes
    if cache_classes != list(classes):
        raise RawPredictionError("cache class_order differs from the project protocol")

    required = MODALITIES if model_name == "almt" else ("text",)
    feature_meta = cache_meta.get("feature_dims")
    length_meta = cache_meta.get("sequence_lengths")
    if not isinstance(feature_meta, Mapping) or not isinstance(length_meta, Mapping):
        raise RawPredictionError("cache meta.json lacks feature_dims/sequence_lengths")
    dims: dict[str, int] = {}
    lengths: dict[str, int] = {}
    for modality in MODALITIES:
        raw_dim = feature_meta.get(modality)
        raw_length = length_meta.get(modality)
        if modality not in required and (raw_dim is None or raw_length is None):
            continue
        dims[modality] = _positive_int(raw_dim, "feature_dims." + modality)
        lengths[modality] = _positive_int(raw_length, "sequence_lengths." + modality)
    if any(modality not in dims or modality not in lengths for modality in required):
        raise RawPredictionError("cache schema lacks dimensions/caps required by {}".format(model_name))

    expected_padding = ALMT_PADDING_BEHAVIOR if model_name == "almt" else TEXT_PADDING_BEHAVIOR
    if manifest.get("padding_behavior") != expected_padding:
        raise RawPredictionError("run padding_behavior differs from the selected model contract")

    return RunSpec(
        run_dir=root,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        manifest_path=manifest_path,
        cache_meta_path=cache_path,
        config=config,
        manifest=manifest,
        cache_meta=cache_meta,
        model_name=model_name,
        model_params=dict(model_params),
        feature_dims=dims,
        sequence_lengths=lengths,
        required_modalities=tuple(required),
        class_order=classes,
        identity_sha256=identity,
        config_sha256=expected_config_sha,
        cache_meta_sha256=cache_meta_sha,
        checkpoint_sha256=sha256_file(checkpoint_path),
    )


def load_classifier_model(
    spec: RunSpec,
    *,
    device: str = "cpu",
    torch_module: Any = None,
    model_factory: Optional[Callable[..., Any]] = None,
) -> tuple[Any, Any, dict[str, float]]:
    """Load only the selected best checkpoint; unsafe pickle fallback is forbidden."""
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for checkpoint inference") from exc
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be 'cpu' or a CUDA device such as 'cuda:0'")
    cuda_enabled = device.startswith("cuda")
    if cuda_enabled and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if cuda_enabled:
        torch_module.cuda.synchronize(device)

    total_started = time.perf_counter()
    checkpoint_started = time.perf_counter()
    try:
        checkpoint = torch_module.load(
            spec.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:
        raise RawPredictionError(
            "this PyTorch runtime cannot load checkpoints with weights_only=True; refusing unsafe fallback"
        ) from exc
    except (OSError, RuntimeError, ValueError, pickle.UnpicklingError, EOFError) as exc:
        raise RawPredictionError("cannot safely load best.pt: {}".format(exc)) from exc
    checkpoint_load_wall = time.perf_counter() - checkpoint_started
    if sha256_file(spec.checkpoint_path) != spec.checkpoint_sha256:
        raise RawPredictionError("best.pt changed while it was being loaded")
    if not isinstance(checkpoint, Mapping):
        raise RawPredictionError("best.pt must contain a state mapping")
    if checkpoint.get("format_version") != 1:
        raise RawPredictionError("unsupported best.pt format_version")
    if checkpoint.get("identity_sha256") != spec.identity_sha256:
        raise RawPredictionError("best.pt identity differs from run_manifest.json")
    if checkpoint.get("seed") != spec.manifest.get("seed"):
        raise RawPredictionError("best.pt seed differs from run_manifest.json")
    if "class_order" in checkpoint and checkpoint["class_order"] != list(spec.class_order):
        raise RawPredictionError("best.pt class_order differs from the project protocol")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or not state:
        raise RawPredictionError("best.pt lacks a non-empty model_state mapping")

    if model_factory is None:
        try:
            from jevaffect.models import build_model
        except ImportError as exc:
            raise RuntimeError("jevaffect.models.build_model is required") from exc
        model_factory = build_model
    build_started = time.perf_counter()
    model = model_factory(
        spec.model_name,
        spec.feature_dims,
        spec.sequence_lengths,
        spec.model_params,
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    if cuda_enabled:
        torch_module.cuda.synchronize(device)
    model_build_load_wall = time.perf_counter() - build_started
    return model, checkpoint, {
        "checkpoint_load_wall_seconds": checkpoint_load_wall,
        "model_build_load_device_wall_seconds": model_build_load_wall,
        "total_classifier_initialization_wall_seconds": time.perf_counter() - total_started,
    }


def _valid_sequence(
    arrays: Mapping[str, Any],
    modality: str,
    *,
    expected_dim: Optional[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    value = arrays.get(modality)
    if value is None:
        return np.empty((0, expected_dim or 0), dtype=np.float32), np.empty(0, dtype=np.bool_), 0
    sequence = np.asarray(value, dtype=np.float32)
    if sequence.ndim != 2:
        raise RawPredictionError("raw {} features must have shape [T,D]".format(modality))
    if expected_dim is not None and sequence.shape[1] != expected_dim:
        raise RawPredictionError(
            "raw {} feature width {} differs from trained width {}".format(
                modality, sequence.shape[1], expected_dim
            )
        )
    if not np.isfinite(sequence).all():
        raise RawPredictionError("raw {} features contain NaN or Inf".format(modality))
    mask = np.asarray(arrays.get(modality + "_mask", np.ones(len(sequence), dtype=np.bool_)), dtype=np.bool_)
    if mask.shape != (len(sequence),):
        raise RawPredictionError("raw {} mask does not align with features".format(modality))
    valid = np.ascontiguousarray(sequence[mask], dtype=np.float32)
    return sequence, mask, len(valid)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _inference_context(torch_module: Any) -> Any:
    if torch_module is None or not hasattr(torch_module, "inference_mode"):
        return contextlib.nullcontext()
    return torch_module.inference_mode()


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (5,) or not np.isfinite(logits).all():
        raise RawPredictionError("model must return five finite logits for one record")
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    total = float(values.sum())
    if not math.isfinite(total) or total <= 0:
        raise RawPredictionError("model logits did not produce finite probabilities")
    probabilities = values / total
    if not np.isfinite(probabilities).all() or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-12):
        raise RawPredictionError("softmax output is invalid")
    return probabilities


def _default_prediction_factory() -> tuple[tuple[str, ...], Callable[..., dict[str, Any]]]:
    try:
        from jevaffect.protocol import CLASSES, make_prediction_record
    except ImportError as exc:
        raise RuntimeError("jevaffect.protocol is required for class labels") from exc
    return tuple(CLASSES), make_prediction_record


class RawVideoPredictor:
    """Predict one clip using a validated trained run and fresh raw features."""

    def __init__(
        self,
        spec: RunSpec,
        model: Any,
        feature_extractor: Any,
        *,
        class_order: Sequence[str],
        low_confidence_threshold: float,
        torch_module: Any,
        device: str = "cpu",
        prediction_factory: Optional[Callable[..., dict[str, Any]]] = None,
    ) -> None:
        classes = _canonical_class_order(class_order)
        if classes != spec.class_order:
            raise RawPredictionError("prediction class order differs from validated run order")
        if isinstance(low_confidence_threshold, bool) or not isinstance(low_confidence_threshold, (int, float)):
            raise ValueError("low confidence threshold must be a number in [0,1]")
        threshold = float(low_confidence_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("low confidence threshold must be a number in [0,1]")
        if device != "cpu" and not device.startswith("cuda"):
            raise ValueError("device must be 'cpu' or a CUDA device")
        self._cuda_enabled = bool(
            device.startswith("cuda")
            and torch_module is not None
            and getattr(torch_module, "cuda", None) is not None
            and torch_module.cuda.is_available()
        )
        if device.startswith("cuda") and not self._cuda_enabled:
            raise RuntimeError("CUDA prediction requested without an available torch CUDA runtime")
        self.spec = spec
        self.model = model
        self.feature_extractor = feature_extractor
        self.class_order = classes
        self.low_confidence_threshold = threshold
        self.torch = torch_module
        self.device = device
        extractor_text_cap = getattr(feature_extractor, "max_text_tokens", None)
        model_text_cap = spec.sequence_lengths.get("text")
        if (
            extractor_text_cap is not None
            and model_text_cap is not None
            and int(extractor_text_cap) != int(model_text_cap)
        ):
            raise RawPredictionError(
                "raw mBERT token cap {} differs from trained cache cap {}; construct the extractor from cache meta".format(
                    extractor_text_cap, model_text_cap
                )
            )
        if prediction_factory is None:
            runtime_classes, prediction_factory = _default_prediction_factory()
            if runtime_classes != classes:
                raise RawPredictionError("jevaffect.protocol class order differs from run manifest")
        self.prediction_factory = prediction_factory

    def _synchronize(self) -> None:
        if self._cuda_enabled:
            self.torch.cuda.synchronize(self.device)

    def _gpu_memory_start(self) -> dict[str, Optional[int]]:
        if not self._cuda_enabled:
            return {"allocated_before_bytes": None, "reserved_before_bytes": None}
        cuda = self.torch.cuda
        allocated = int(cuda.memory_allocated(self.device))
        reserved = int(cuda.memory_reserved(self.device))
        cuda.reset_peak_memory_stats(self.device)
        return {"allocated_before_bytes": allocated, "reserved_before_bytes": reserved}

    def _gpu_memory_end(self, start: Mapping[str, Optional[int]]) -> dict[str, Any]:
        if not self._cuda_enabled:
            return {
                "measured": False, "device": None,
                "allocated_before_bytes": None, "peak_allocated_bytes": None,
                "peak_allocated_delta_bytes": None, "reserved_before_bytes": None,
                "peak_reserved_bytes": None,
            }
        cuda = self.torch.cuda
        peak_allocated = int(cuda.max_memory_allocated(self.device))
        peak_reserved = int(cuda.max_memory_reserved(self.device))
        allocated_before = int(start["allocated_before_bytes"] or 0)
        return {
            "measured": True,
            "device": self.device,
            "allocated_before_bytes": allocated_before,
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_delta_bytes": max(0, peak_allocated - allocated_before),
            "reserved_before_bytes": start["reserved_before_bytes"],
            "peak_reserved_bytes": peak_reserved,
        }

    def _tensor(self, array: np.ndarray, *, integer: bool = False) -> Any:
        dtype = self.torch.long if integer else self.torch.float32
        tensor = self.torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=self.device)
        return tensor.to(self.device) if hasattr(tensor, "to") else tensor

    def predict_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        sample_id = record.get("id") if isinstance(record, Mapping) else None
        if not isinstance(sample_id, str) or not sample_id or sample_id.isspace():
            raise RawPredictionError("raw record requires a non-empty string id")
        self._synchronize()
        whole_started = time.perf_counter()
        raw_result = self.feature_extractor.extract_record(record)
        arrays = raw_result.get("arrays")
        raw_meta = raw_result.get("metadata")
        if not isinstance(arrays, Mapping) or not isinstance(raw_meta, Mapping):
            raise RawPredictionError("raw feature extractor returned an invalid record bundle")

        prepared: dict[str, np.ndarray] = {}
        modality_info: dict[str, dict[str, Any]] = {}
        time_references: dict[str, Any] = {}
        for modality in MODALITIES:
            expected_dim = self.spec.feature_dims.get(modality)
            raw_seq, mask, valid_count = _valid_sequence(
                arrays, modality, expected_dim=expected_dim
            )
            required = modality in self.spec.required_modalities
            valid_seq = raw_seq[mask]
            cap = self.spec.sequence_lengths.get(modality) if required else None
            if required:
                reports = raw_meta.get("modalities", {})
                report = reports.get(modality) if isinstance(reports, Mapping) else None
                if isinstance(report, Mapping) and "valid" in report:
                    if bool(report["valid"]) != bool(valid_count > 0):
                        raise RawPredictionError(
                            "record {!r} has inconsistent {} validity metadata and feature mask".format(
                                sample_id, modality
                            )
                        )
                if valid_count < 1:
                    details = report if isinstance(report, Mapping) else {}
                    status = details.get("status", "missing") if isinstance(details, Mapping) else "missing"
                    raise RawPredictionError(
                        "record {!r} cannot enter {}: required {} modality is invalid ({})".format(
                            sample_id, self.spec.model_name, modality, status
                        )
                    )
                if cap is None or expected_dim is None:
                    raise RawPredictionError("run schema lacks {} model dimensions/cap".format(modality))
                used = min(valid_count, cap)
                padded = np.zeros((cap, expected_dim), dtype=np.float32)
                padded[:used] = valid_seq[:used]
                prepared[modality] = padded
            else:
                used = 0

            modality_info[modality] = {
                "valid": bool(valid_count > 0),
                "used_for_model": required,
                "raw_sequence_length": int(len(raw_seq)),
                "valid_sequence_length": int(valid_count),
                "model_cap": cap,
                "used_sequence_length": int(used),
                "right_padding_positions": int(cap - used) if required and cap is not None else 0,
                "truncated_valid_positions": int(max(0, valid_count - cap)) if required and cap is not None else 0,
                "feature_dim": int(raw_seq.shape[1]) if raw_seq.ndim == 2 else None,
            }
            if modality == "text":
                time_references[modality] = {
                    "kind": "token_positions",
                    "time_aligned": False,
                    "note": "mBERT contextual tokens do not carry word timestamps",
                }
            else:
                times_key = "audio_times" if modality == "audio" else "vision_times"
                times = np.asarray(arrays.get(times_key, np.empty(0)), dtype=np.float64)
                if times.shape != (len(raw_seq),) or not np.isfinite(times).all():
                    if required:
                        raise RawPredictionError(
                            "{} timestamp array does not align with required model features".format(modality)
                        )
                    time_references[modality] = {
                        "array_key": times_key,
                        "unit": "seconds",
                        "seconds": [],
                        "used_by_model": False,
                        "valid": False,
                        "warning": "timestamp array does not align; modality is unused by this model",
                    }
                    continue
                selected_times = times[mask]
                if required:
                    selected_times = selected_times[:used]
                reference = {
                    "array_key": times_key,
                    "unit": "seconds",
                    "seconds": [float(value) for value in selected_times],
                    "used_by_model": required,
                    "valid": True,
                }
                if modality == "audio":
                    reference["meaning"] = "centers of WavLM convolutional receptive fields from clip start"
                else:
                    reference["meaning"] = raw_meta.get("preprocessing", {}).get(
                        "video_frame_timestamps_kind", "FFmpeg frame timestamps"
                    )
                time_references[modality] = reference

        model_inputs: dict[str, Any] = {
            modality: self._tensor(prepared[modality][None, ...])
            for modality in self.spec.required_modalities
        }
        if self.spec.model_name == "text_mlp":
            text_length = modality_info["text"]["used_sequence_length"]
            model_inputs["lengths"] = {
                "text": self._tensor(np.asarray([text_length], dtype=np.int64), integer=True)
            }

        # Time only classifier inference after inputs are ready. CUDA timing is
        # synchronized around the forward and output transfer, with memory
        # peaks reset per record.
        self._synchronize()
        gpu_start = self._gpu_memory_start()
        inference_started = time.perf_counter()
        with _inference_context(self.torch):
            logits = self.model(model_inputs)
        logits_array = _to_numpy(logits)
        self._synchronize()
        inference_wall = time.perf_counter() - inference_started
        gpu_memory = self._gpu_memory_end(gpu_start)
        if logits_array.shape != (1, len(self.class_order)):
            raise RawPredictionError(
                "trained model returned logits shape {}; expected [1,5]".format(logits_array.shape)
            )
        probabilities = _softmax(logits_array[0])
        prediction = self.prediction_factory(sample_id, [float(value) for value in probabilities])
        confidence = float(prediction["choice_confidence"])
        prediction.update({
            "model": self.spec.model_name,
            "run_identity_sha256": self.spec.identity_sha256,
            "class_order": list(self.class_order),
            "modality_lengths": modality_info,
            "timestamp_references": time_references,
            "transcript_source": raw_meta.get("transcript_source"),
            "asr_status": (
                raw_meta.get("asr", {}).get("status")
                if isinstance(raw_meta.get("asr"), Mapping)
                else None
            ),
            "raw_feature_performance": raw_meta.get("performance"),
            "performance": {
                "whole_raw_to_prediction_wall_seconds": time.perf_counter() - whole_started,
                "whole_raw_to_prediction_gpu_synchronized": self._cuda_enabled,
                "classifier_inference_wall_seconds": inference_wall,
                "classifier_gpu_synchronized": self._cuda_enabled,
                "classifier_gpu_memory": gpu_memory,
            },
            "low_confidence": confidence < self.low_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "confidence_note": (
                "choice_confidence is the protocol's linear rescaling of max softmax probability; "
                "it is not calibrated correctness probability"
            ),
            "fixture": bool(raw_meta.get("fixture", False)),
            "model_results": bool(
                not raw_meta.get("fixture", False) and raw_meta.get("model_results", False)
            ),
            "prediction_stage": (
                "CPU mock fixture; not model results"
                if raw_meta.get("fixture", False)
                else "trained checkpoint pilot inference"
            ),
            "attention_weights_exported": False,
            "attention_interpretability_claim": False,
        })
        return prediction


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_prediction_bundle(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    predictor: RawVideoPredictor,
    *,
    input_manifest_path: str | Path,
    classifier_initialization: Mapping[str, Any],
    encoder_initialization: Mapping[str, Any],
) -> dict[str, Any]:
    """Write immutable pilot JSONL outputs; every prediction remains in the full file."""
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError("refusing to overwrite prediction output: {}".format(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=target.name + ".staging-", dir=str(target.parent)))
    try:
        predictions: list[dict[str, Any]] = []
        low_confidence: list[dict[str, Any]] = []
        timings: list[dict[str, Any]] = []
        ids: list[str] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise RawPredictionError("each input record must be an object")
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id or sample_id.isspace():
                raise RawPredictionError("each input record needs a non-empty string id")
            ids.append(sample_id)
        if len(set(ids)) != len(ids):
            raise RawPredictionError("input records contain duplicate IDs")

        seen: set[str] = set()
        for record in records:
            result = predictor.predict_record(record)
            sample_id = result["ID"]
            if sample_id in seen:
                raise RawPredictionError("duplicate prediction ID {!r}".format(sample_id))
            seen.add(sample_id)
            predictions.append(result)
            if result["low_confidence"]:
                low_confidence.append(result)
            timings.append({"ID": sample_id, **result["performance"]})
        if not predictions:
            raise RawPredictionError("input contains no records")

        _write_jsonl(stage / "predictions.jsonl", predictions)
        _write_jsonl(stage / "low_confidence.jsonl", low_confidence)
        _write_jsonl(stage / "timings.jsonl", timings)
        info = {
            "scope": "pilot_inference_only",
            "formal_benchmark": False,
            "competition_submission_ready": False,
            "attention_weights_exported": False,
            "attention_interpretability_claim": False,
            "model": predictor.spec.model_name,
            "prediction_device": predictor.device,
            "run_identity_sha256": predictor.spec.identity_sha256,
            "run_config_sha256": predictor.spec.config_sha256,
            "cache_meta_sha256": predictor.spec.cache_meta_sha256,
            "cache_source_sha256": predictor.spec.cache_meta["source_sha256"],
            "checkpoint_sha256": predictor.spec.checkpoint_sha256,
            "seed": predictor.spec.manifest["seed"],
            "class_order": list(predictor.class_order),
            "feature_dims": dict(predictor.spec.feature_dims),
            "sequence_lengths": dict(predictor.spec.sequence_lengths),
            "required_modalities": list(predictor.spec.required_modalities),
            "padding_behavior": predictor.spec.manifest["padding_behavior"],
            "low_confidence_threshold": predictor.low_confidence_threshold,
            "confidence_note": (
                "choice_confidence is a deterministic max-softmax rescaling, not calibrated probability"
            ),
            "input_manifest_sha256": sha256_file(input_manifest_path),
            "record_count": len(predictions),
            "low_confidence_count": len(low_confidence),
            "fixture_results": any(bool(row.get("fixture")) for row in predictions),
            "all_records_are_fixtures": all(bool(row.get("fixture")) for row in predictions),
            "model_results": all(bool(row.get("model_results")) for row in predictions),
            "classifier_initialization": dict(classifier_initialization),
            "raw_encoder_initialization": dict(encoder_initialization),
            "encoder_provenance": dict(
                getattr(getattr(predictor.feature_extractor, "models", None), "provenance", {})
            ),
            "output_files": ["predictions.jsonl", "low_confidence.jsonl", "timings.jsonl"],
        }
        (stage / "run_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if target.exists():
            raise FileExistsError("prediction output appeared during inference: {}".format(target))
        os.replace(stage, target)
        return info
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise
