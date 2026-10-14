"""Staged training for native CH-SIMSv2 multi-key scalar sequence gates.

Stage one chooses modality-head epoch counts with a source-held split inside
official TRAIN. Refit uses the original initialization and the complete
official TRAIN set. Stage two fits the matched scalar-gate cases against only
the native aggregate label and selects checkpoints on native VALID MAE.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from jevaffect.anchor_calibration import normalize, normalizer
from jevaffect.native_metrics import native_metrics
from jevaffect.native_training import NativeSplit, save_checkpoint, save_json


ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("text", "audio", "vision")
CHANNEL_COLUMNS = ("label_T", "label_A", "label_V")
CASES = ("static", "pre_norm", "post_norm", "distilled_static", "distilled_pre_norm", "distilled_post_norm", "learned_post_norm")
DISTILLED_CASES = frozenset(("distilled_static", "distilled_pre_norm", "distilled_post_norm"))
POLICY_ROUTED_CASES = frozenset(("distilled_pre_norm", "distilled_post_norm", "learned_post_norm"))
POLICY_ACTIVE_CASES = frozenset((*DISTILLED_CASES, "learned_post_norm"))
SEEDS = (13, 42, 2026)
AUTHOR_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
FORMAL_COUNTS = {"train": 2722, "valid": 647}
FORMAL_DIMS = {"text": 768, "audio": 768, "vision": 384}
FORMAL_SOURCE_COUNT = 129
SOURCE_SPLIT_SEED = 20261014
IDENTITY_FILES = (
    "jevaffect/sequence_gates.py",
    "jevaffect/sequence_gate_training.py",
    "scripts/train_sequence_gates.py",
    "jevaffect/native_metrics.py",
    "jevaffect/native_training.py",
    "jevaffect/anchor_calibration.py",
    "scripts/prepare_native_pooled_standardization.py",
    "scripts/prepare_native_sequence_standardization.py",
    "docs/NATIVE_SEQUENCE_GATE_PLAN_V1.md",
)


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _load_config(config: dict | str | Path, config_path: str | Path | None = None) -> tuple[dict, Path | None]:
    if isinstance(config, dict):
        return dict(config), Path(config_path).resolve() if config_path else None
    path = Path(config).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    return value, path


def read_author_annotations(path: str | Path, *, channels: bool, allow_fixture: bool = False) -> dict:
    """Read IDs/splits and only the targets permitted for this stage.

    With ``channels=True``, TRAIN M/T/A/V are parsed; VALID and TEST score
    columns are never converted or validated. With ``channels=False``, M is
    parsed for TRAIN and VALID and all T/A/V and TEST scores are ignored.
    """
    path = Path(path)
    if not allow_fixture and file_sha256(path) != AUTHOR_CSV_SHA256:
        raise ValueError("Author CSV SHA256 differs from the pinned CH-SIMSv2 file")
    records: dict[str, dict] = {}
    split_ids: dict[str, list[str]] = {name: [] for name in ("train", "valid", "test")}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "clip_id", "mode", "label", *CHANNEL_COLUMNS}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Author CSV is missing required native annotation columns")
        for row in reader:
            split = row["mode"]
            if split not in split_ids:
                raise ValueError("Author CSV contains an unknown mode")
            sid = f"{row['video_id']}/{row['clip_id']}"
            if sid in records:
                raise ValueError("Author CSV contains duplicate clip IDs")
            record = {"split": split, "video_id": row["video_id"]}
            should_read_m = split == "train" or (not channels and split == "valid")
            if should_read_m:
                record["score"] = _parse_native_score(row["label"], "label")
            if channels and split == "train":
                record["channels"] = np.asarray([_parse_native_score(row[name], name) for name in CHANNEL_COLUMNS], dtype=np.float32)
            records[sid] = record
            split_ids[split].append(sid)
    return {"records": records, "ids": split_ids, "channels": channels}


def _parse_native_score(value: str, column: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid native author score in {column}") from exc
    if not math.isfinite(score) or abs(score) > 1.0:
        raise ValueError(f"Native author score outside [-1,1] in {column}")
    return score


class SequenceGateSplit(Dataset):
    """NativeSplit adapter that transforms only valid sequence steps."""

    def __init__(self, split: NativeSplit, *, stats: dict | None = None, labels: np.ndarray | None = None):
        self.split = split
        self.stats = stats
        self.ids = list(split.ids)
        self.native_targets = np.asarray(split.targets, dtype=np.float32)
        if labels is None:
            self.labels = self.native_targets
        else:
            values = np.asarray(labels, dtype=np.float32)
            if values.shape[0] != len(self.ids) or not np.isfinite(values).all():
                raise ValueError("Sequence-gate labels must align with cache IDs and be finite")
            self.labels = values

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        features, _, sid = self.split[index]
        if self.stats is not None:
            for modality in MODALITIES:
                length = int(features["lengths"][modality])
                values = features[modality]
                if length:
                    values[:length] = torch.from_numpy(normalize(values[:length].numpy(), self.stats[modality]))
                values[length:] = 0
        return features, torch.as_tensor(self.labels[index], dtype=torch.float32), sid


def _source_id(sample_id: str) -> str:
    if not isinstance(sample_id, str) or "/" not in sample_id:
        raise ValueError("SIMSv2 IDs must be video_id/clip_id")
    return sample_id.split("/", 1)[0]


def source_holdout(ids: list[str], seed: int = SOURCE_SPLIT_SEED, fraction: float = .10) -> dict:
    sources = sorted({_source_id(sid) for sid in ids})
    if len(sources) < 2 or not 0 < fraction < 1:
        raise ValueError("Source-held selection requires at least two sources and a valid fraction")
    order = list(sources)
    np.random.default_rng(seed).shuffle(order)
    held_count = max(1, math.ceil(len(order) * fraction))
    held = set(order[-held_count:])
    opt = set(order[:-held_count])
    opt_ids = [sid for sid in ids if _source_id(sid) in opt]
    held_ids = [sid for sid in ids if _source_id(sid) in held]
    if not opt_ids or not held_ids or opt & held:
        raise ValueError("Invalid source-held TRAIN split")
    return {
        "seed": int(seed),
        "fraction": float(fraction),
        "all_sources": sources,
        "shuffled_sources": order,
        "opt_sources": sorted(opt),
        "held_sources": sorted(held),
        "opt_ids": opt_ids,
        "held_ids": held_ids,
        "held_count": held_count,
    }


def _pool_raw(split: NativeSplit) -> dict[str, np.ndarray]:
    pooled = {}
    for modality in MODALITIES:
        values = split.features[modality]
        lengths = split.lengths[modality]
        if modality == "text":
            output = np.asarray(values[:, 0], dtype=np.float32).copy()
            output[np.asarray(lengths) == 0] = 0
        else:
            output = np.stack([
                np.asarray(values[index, :int(length)], dtype=np.float32).mean(0)
                if int(length) else np.zeros(values.shape[-1], dtype=np.float32)
                for index, length in enumerate(lengths)
            ])
        pooled[modality] = output.astype(np.float32, copy=False)
    return pooled


def _load_model_module():
    try:
        from jevaffect import sequence_gates
    except ImportError as exc:
        raise RuntimeError("jevaffect.sequence_gates must be implemented before training") from exc
    return sequence_gates


def _branch_parameters(model, modality: str) -> list[torch.nn.Parameter]:
    branches = getattr(model, "branches", None)
    if branches is None or modality not in branches:
        raise ValueError("Sequence-gate model must expose branches ModuleDict for all modalities")
    params = list(branches[modality].parameters())
    if not params:
        raise ValueError("Each modality head must have trainable parameters")
    return params


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _restore_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    module.load_state_dict(state, strict=True)


def _capture_rng(generator: torch.Generator, device: torch.device,
                 validation_generator: torch.Generator | None = None) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "loader_generator": generator.get_state(),
        "validation_loader_generator": validation_generator.get_state() if validation_generator is not None else None,
    }


def _runtime_flags() -> dict:
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "mha_fastpath_enabled": bool(torch.backends.mha.get_fastpath_enabled()),
        "sdp_flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        "sdp_mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "sdp_math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
    }


def _configure_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def _restore_rng(state: dict, generator: torch.Generator, device: torch.device,
                 validation_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda" and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader_generator"])
    validation_state = state.get("validation_loader_generator")
    if validation_state is not None:
        if validation_generator is None:
            raise ValueError("Checkpoint includes a validation loader RNG but no validation loader is configured")
        validation_generator.set_state(validation_state)


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    _configure_runtime()
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _atomic_state(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(path, checkpoint)


def _cache_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Native sequence cache is missing metadata.json: " + str(path))
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _validate_cache_chain(config: dict, author_sha: str, *, allow_fixture: bool = False) -> dict:
    raw_cache = _resolve_path(config["raw_cache"])
    standardized_cache = _resolve_path(config["standardized_cache"])
    pooled_cache = _resolve_path(config["pooled_cache"])
    raw_meta, std_meta, pool_meta = (_cache_metadata(path) for path in (raw_cache, standardized_cache, pooled_cache))
    if std_meta.get("test_exported") is not False or pool_meta.get("test_exported") is not False:
        raise ValueError("Standardized training caches must explicitly record that TEST features were not exported")
    raw_split_counts = raw_meta.get("split_counts", {})
    if "test" in raw_split_counts or (not allow_fixture and "not included" not in str(raw_meta.get("test_data", "")).lower()):
        raise ValueError("Raw feature cache must exclude TEST data from its join, arrays, and metadata counts")
    if not allow_fixture:
        raw_is_simsv2 = raw_meta.get("dataset") in (None, "simsv2") and raw_meta.get("formal_benchmark") is True
        if not raw_is_simsv2 or std_meta.get("dataset") != "simsv2" or pool_meta.get("dataset") != "simsv2":
            raise ValueError("Formal native sequence-gate training requires CH-SIMSv2 cache contracts")
        if raw_meta.get("official_csv_sha256", "").lower() != author_sha.lower():
            raise ValueError("Raw feature cache is bound to a different author CSV")
        if raw_split_counts != FORMAL_COUNTS:
            raise ValueError("Raw feature cache must contain only the complete official TRAIN/VALID cohorts")
        if std_meta.get("source_metadata_sha256") != file_sha256(raw_cache / "metadata.json"):
            raise ValueError("Standardized sequence cache is detached from the pinned raw metadata")
        if std_meta.get("pooled_standardized_metadata_sha256") != file_sha256(pooled_cache / "metadata.json"):
            raise ValueError("Standardized sequence cache is detached from the pooled normalization cache")
        if std_meta.get("normalization_statistics_sha256") != file_sha256(pooled_cache / "normalization.json"):
            raise ValueError("Standardized sequence cache does not reuse the pinned pooled scaler")
        if Path(std_meta.get("source_cache", "")).resolve() != raw_cache:
            raise ValueError("Standardized sequence cache raw source path changed")
        if Path(std_meta.get("pooled_standardized_cache", "")).resolve() != pooled_cache:
            raise ValueError("Standardized sequence cache pooled scaler path changed")
        for meta in (std_meta, pool_meta):
            if meta.get("author_csv_sha256", "").lower() != author_sha.lower():
                raise ValueError("Author CSV hash differs from the standardized cache binding")
        if pool_meta.get("normalization", {}).get("fit_split") != "train" or pool_meta.get("normalization", {}).get("fit_count") != FORMAL_COUNTS["train"]:
            raise ValueError("Pooled normalization must be fit on all official TRAIN rows")
        if std_meta.get("counts") != FORMAL_COUNTS:
            raise ValueError("Standardized cache does not contain the complete pinned TRAIN/VALID cohorts")
        if std_meta.get("dimensions") != FORMAL_DIMS:
            raise ValueError("Standardized cache dimensions differ from the pinned sequence schema")
    if std_meta.get("author_csv_sha256", "").lower() != author_sha.lower():
        raise ValueError("Standardized cache is bound to a different author CSV")
    return {
        "raw_cache": str(raw_cache),
        "standardized_cache": str(standardized_cache),
        "pooled_cache": str(pooled_cache),
        "raw_metadata_sha256": file_sha256(raw_cache / "metadata.json"),
        "standardized_metadata_sha256": file_sha256(standardized_cache / "metadata.json"),
        "pooled_metadata_sha256": file_sha256(pooled_cache / "metadata.json"),
        "normalization_sha256": file_sha256(pooled_cache / "normalization.json"),
        "raw_metadata": raw_meta,
        "standardized_metadata": std_meta,
        "pooled_metadata": pool_meta,
    }


def _identity(config: dict, config_path: Path | None, cache_chain: dict, author_path: Path, *, stage: str,
              data_hashes: dict, upstream_hash: str | None = None, allow_fixture: bool = False) -> dict:
    _configure_runtime()
    files = {}
    for relative in IDENTITY_FILES:
        path = ROOT / relative
        if path.is_file():
            files[relative] = file_sha256(path)
        elif not allow_fixture:
            raise ValueError("Formal training identity is missing a required implementation or plan file: " + relative)
        else:
            files[relative] = None
    if config_path is not None:
        files[config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path)] = file_sha256(config_path)
    base = {
        "schema": config.get("schema"),
        "stage": stage,
        "configuration": config,
        "runtime_flags": _runtime_flags(),
        "author_csv_sha256": file_sha256(author_path),
        "code_and_plan_sha256": files,
        "cache_bindings": {key: cache_chain[key] for key in (
            "raw_metadata_sha256", "standardized_metadata_sha256", "pooled_metadata_sha256", "normalization_sha256")},
        "used_array_sha256": data_hashes,
        "upstream_checkpoint_sha256": upstream_hash,
    }
    base["identity_sha256"] = _sha_json(base)
    return base


def _dataset_hashes(cache: Path, splits: list[NativeSplit]) -> dict:
    hashes = {"metadata.json": file_sha256(cache / "metadata.json")}
    for split in splits:
        # NativeSplit.hashes is populated with verified paths; make keys relative to
        # the known cache root without re-reading the large arrays.
        for key, value in split.hashes.items():
            path = Path(key)
            try:
                relative = path.resolve().relative_to(cache.resolve()).as_posix()
            except (ValueError, OSError):
                relative = path.as_posix()
            hashes[relative] = value
    return dict(sorted(hashes.items()))


def _target_rows(annotations: dict, split: str, ids: list[str], *, channels: bool) -> np.ndarray:
    rows = []
    for sid in ids:
        record = annotations["records"].get(sid)
        if record is None or record["split"] != split:
            raise ValueError("Author annotations and cached IDs differ")
        field = "channels" if channels else "score"
        if field not in record:
            raise ValueError("Requested label was not read for this split")
        rows.append(record[field])
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Native targets must be finite")
    return result


def _native_split(cache: Path, split: str, annotations: dict, *, pooled: bool = False, limit: int | None = None) -> NativeSplit:
    rows = annotations["records"]
    return NativeSplit(cache, split, rows, MODALITIES, pooled=pooled, limit=limit)


def _formal_data_check(annotations: dict, raw_train: NativeSplit | None, std_train: NativeSplit | None,
                       std_valid: NativeSplit | None, cache_chain: dict, allow_fixture: bool) -> None:
    if allow_fixture:
        return
    if raw_train is not None and len(raw_train) != FORMAL_COUNTS["train"]:
        raise ValueError("Formal source selection requires all 2,722 native TRAIN rows")
    if std_train is not None and len(std_train) != FORMAL_COUNTS["train"]:
        raise ValueError("Formal refit/fusion requires all 2,722 native TRAIN rows")
    if std_valid is not None and len(std_valid) != FORMAL_COUNTS["valid"]:
        raise ValueError("Formal fusion requires all 647 native VALID rows")
    if len(annotations["ids"]["train"]) != FORMAL_COUNTS["train"] or len(annotations["ids"]["valid"]) != FORMAL_COUNTS["valid"]:
        raise ValueError("Official author split counts changed")
    sources = {_source_id(sid) for sid in annotations["ids"]["train"]}
    if len(sources) != FORMAL_SOURCE_COUNT:
        raise ValueError("Official TRAIN source count differs from the pinned 129 videos")


def _validate_raw_native_targets(split: NativeSplit, annotations: dict) -> None:
    author_targets = _target_rows(annotations, "train", split.ids, channels=False)
    if not np.allclose(split.targets, author_targets, atol=1e-6, rtol=0):
        raise ValueError("Raw cache M labels disagree with native TRAIN author M scores")


def _create_output(root: Path, run_identity: dict, *, resume: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "identity.json"
    if marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("identity_sha256") != run_identity.get("identity_sha256"):
            raise ValueError("Refusing to resume a sequence-gate output with changed run identity")
        if not resume:
            raise FileExistsError("Output already exists; pass --resume to continue its exact checkpoint")
    else:
        if resume:
            raise FileNotFoundError("--resume requires a prior identity.json")
        save_json(marker, run_identity)
    return root


def _read_stage_checkpoint(path: Path, identity: dict, *, resume: bool) -> dict | None:
    if not path.exists():
        if resume:
            raise FileNotFoundError("--resume checkpoint is missing: " + str(path))
        return None
    if not resume:
        raise FileExistsError("Checkpoint exists; pass --resume to continue: " + str(path))
    checkpoint = _load_checkpoint(path)
    if checkpoint.get("identity", {}).get("identity_sha256") != identity.get("identity_sha256"):
        raise ValueError("Checkpoint identity differs from the current configuration or data")
    return checkpoint


def _loader(dataset: Dataset, batch_size: int, *, shuffle: bool, generator: torch.Generator) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False, generator=generator)


def _move_features(features: dict, device: torch.device) -> dict:
    moved = {
        **{modality: features[modality].to(device, non_blocking=False) for modality in MODALITIES},
        "lengths": {modality: features["lengths"][modality].to(device, non_blocking=False) for modality in MODALITIES},
    }
    if "masks" in features:
        moved["masks"] = {modality: features["masks"][modality].to(device, non_blocking=False) for modality in MODALITIES}
    return moved


def _channel_state(model: torch.nn.Module, modality: str) -> dict[str, torch.Tensor]:
    prefix = f"branches.{modality}."
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items() if name.startswith(prefix)}
    if not state:
        raise ValueError("Could not isolate a modality head state for source-held selection")
    return state


def _mean_vector(output: dict, name: str, *, width: int = 3) -> list[float]:
    value = output.get(name)
    if not isinstance(value, torch.Tensor) or value.shape != (len(output["prediction"]), width):
        raise ValueError(f"Sequence-gate forward output must include {name}[N,{width}]")
    if not torch.isfinite(value).all():
        raise ValueError(f"Sequence-gate {name} contains nonfinite values")
    return value.detach().mean(0).cpu().tolist()


@torch.no_grad()
def _evaluate_heads(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    predictions, targets, masks = [], [], []
    for features, y, _ in loader:
        masks.append(torch.stack([features["lengths"][m] > 0 for m in MODALITIES], dim=1).numpy())
        features = _move_features(features, device)
        predictions.append(model(features).detach().cpu().numpy())
        targets.append(y.numpy())
    pred, truth, present = np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    if pred.shape != truth.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("Unimodal head prediction schema must be [N,3] in T/A/V order")
    result = {}
    for index, modality in enumerate(MODALITIES):
        keep = present[:, index]
        result[modality] = native_metrics(pred[keep, index], truth[keep, index], "simsv2") if keep.any() else {"count": 0, "MAE": None}
    return result


def _stage_path(output: Path, stage: str, seed: int, case: str | None = None) -> Path:
    base = output / stage / f"seed_{seed}"
    return base / case if case else base


def _torch_save_hash(path: Path) -> str:
    return file_sha256(path)


def _stateless_branch_update(optimizer: torch.optim.Optimizer, model: torch.nn.Module, active: set[str], clip: float) -> None:
    for modality in MODALITIES:
        params = _branch_parameters(model, modality)
        if modality in active:
            torch.nn.utils.clip_grad_norm_(params, clip)
        else:
            for parameter in params:
                parameter.grad = None


def _profile_peak(device: torch.device) -> int | None:
    return int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None


def _stage1_data(config: dict, annotations: dict, *, smoke: int | None, allow_fixture: bool):
    raw_cache = _resolve_path(config["raw_cache"])
    raw = _native_split(raw_cache, "train", annotations, pooled=False, limit=smoke)
    _validate_raw_native_targets(raw, annotations)
    sources = source_holdout(raw.ids)
    opt_ids = set(sources["opt_ids"])
    opt_indices = [i for i, sid in enumerate(raw.ids) if sid in opt_ids]
    pooled = _pool_raw(raw)
    stats = {m: normalizer(pooled[m], opt_indices) for m in MODALITIES}
    labels = _target_rows(annotations, "train", raw.ids, channels=True)
    dataset = SequenceGateSplit(raw, stats=stats, labels=labels)
    opt = Subset(dataset, opt_indices)
    held_indices = [i for i, sid in enumerate(raw.ids) if sid not in opt_ids]
    held = Subset(dataset, held_indices)
    _formal_data_check(annotations, raw, None, None, {}, allow_fixture or smoke is not None)
    return raw, dataset, opt, held, stats, sources, opt_indices, held_indices


def _stage2_data(config: dict, annotations: dict, *, smoke: int | None, allow_fixture: bool):
    cache = _resolve_path(config["standardized_cache"])
    train_split = _native_split(cache, "train", annotations, pooled=False, limit=smoke)
    valid_split = _native_split(cache, "valid", annotations, pooled=False, limit=smoke)
    train = SequenceGateSplit(train_split)
    valid = SequenceGateSplit(valid_split)
    _formal_data_check(annotations, None, train_split, valid_split, {}, allow_fixture or smoke is not None)
    return train_split, valid_split, train, valid


def _stage_identity_files(config: dict, config_path: Path | None, cache_chain: dict, author_path: Path,
                          stage: str, splits: list[NativeSplit], upstream_hash: str | None,
                          allow_fixture: bool) -> dict:
    cache = _resolve_path(config["raw_cache"] if stage == "select" else config["standardized_cache"])
    data_hashes = _dataset_hashes(cache, splits)
    return _identity(config, config_path, cache_chain, author_path, stage=stage,
                     data_hashes=data_hashes, upstream_hash=upstream_hash, allow_fixture=allow_fixture)


def _validate_formal_contract(config: dict, annotations: dict, cache_chain: dict, *, allow_fixture: bool) -> None:
    if allow_fixture:
        return
    if config.get("schema") != "native_sequence_scalar_gates_v1":
        raise ValueError("Unsupported native sequence-gate configuration schema")
    if config.get("dataset") != "simsv2" or config.get("author_csv_sha256", "").lower() != AUTHOR_CSV_SHA256:
        raise ValueError("Formal sequence-gate config must pin the official CH-SIMSv2 author CSV")
    if tuple(config.get("seeds", ())) != SEEDS:
        raise ValueError("Formal matrix must retain public seeds 13, 42, and 2026")
    if tuple(config.get("cases", ())) != CASES:
        raise ValueError("Formal matrix must contain the three controls and three distilled-policy cases")
    expected_model = {"dim": 128, "queries": 8, "depth": 3, "heads": 4, "ff_dim": 256, "dropout": .3}
    if config.get("model") != expected_model:
        raise ValueError("Formal sequence-gate architecture parameters changed")
    if config.get("expected_parameters") != {
        "unimodal_total": 353091, "fusion_total": 2084871, "fusion_trainable": 1978692, "policy_head": 25027,
    }:
        raise ValueError("Formal stage parameter-count contract changed")
    if config.get("source_holdout") != {
        "seed": SOURCE_SPLIT_SEED,
        "fraction": .1,
        "unit": "official TRAIN video_id",
        "held_count_rule": "ceil(fraction * sorted unique TRAIN sources) after deterministic seeded shuffle",
    }:
        raise ValueError("Formal source-held split contract changed")
    if config.get("stage1") != {
        "epochs": 50, "batch_size": 32, "learning_rate": .0001, "weight_decay": .0001, "gradient_clip": 1.0,
        "loss": "sum of three per-modality mean absolute errors over present modality rows",
        "selection": "minimum clipped native channel MAE on source-held official TRAIN; first minimum wins ties",
    }:
        raise ValueError("Formal source-held selection budget or optimizer changed")
    if config.get("refit") != {
        "batch_size": 32, "learning_rate": .0001, "weight_decay": .0001, "gradient_clip": 1.0,
        "loss": "sum of channel L1 terms; each branch stops after its source-selected epoch count",
        "normalization": "reuse existing pooled scaler fit on all official TRAIN; no additional statistics are fit",
    }:
        raise ValueError("Formal all-TRAIN head refit contract changed")
    if config.get("fusion") != {
        "epochs": 50, "batch_size": 32, "learning_rate": .00003, "weight_decay": .00001,
        "gradient_clip": 1.0, "contrastive_weight": .01, "distillation_weight": .1,
        "loss": "native aggregate M L1 plus 0.01 contrastive plus 0.1 teacher-to-policy KL distillation",
        "distillation": "KL(stopgrad(softmax(-0.3*(unimodal_score-TRAIN_M)^2)),policy_ratios); TRAIN targets only",
        "selection": "minimum native VALID M MAE; first minimum wins ties",
        "updates_per_epoch": 86, "total_updates_per_fit": 4300,
    }:
        raise ValueError("Formal fusion optimizer or update budget changed")
    if config.get("formal_matrix") != {
        "train_rows": 2722, "valid_rows": 647, "train_sources": 129, "source_held_sources": 13,
        "selection_fits": 3, "refit_fits": 3, "fusion_fits": 21,
        "test_loader": False, "test_annotations_read": False,
    }:
        raise ValueError("Formal matrix count or TEST boundary contract changed")
    if len(annotations["ids"]["train"]) != FORMAL_COUNTS["train"] or len(annotations["ids"]["valid"]) != FORMAL_COUNTS["valid"]:
        raise ValueError("Formal author split counts changed")


def _model_dims(split: NativeSplit) -> dict[str, int]:
    dims = {m: int(split.features[m].shape[-1]) for m in MODALITIES}
    if any(dims[m] <= 0 for m in MODALITIES):
        raise ValueError("Sequence-gate input dimensions must be positive")
    return dims


def _check_parameter_contract(model: torch.nn.Module, config: dict, *, frozen: bool = False) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    key = "fusion_total" if frozen else "unimodal_total"
    expected = config.get("expected_parameters", {}).get(key)
    if expected is not None and total != int(expected):
        raise ValueError(f"Model parameter count changed: expected {expected}, found {total}")
    if frozen:
        expected_trainable = config.get("expected_parameters", {}).get("fusion_trainable")
        if expected_trainable is not None and trainable != int(expected_trainable):
            raise ValueError(f"Frozen fusion trainable parameter count changed: expected {expected_trainable}, found {trainable}")
    return total, trainable


def select_heads(config: dict | str | Path, seed: int, *, config_path: Path | None = None,
                 output: str | Path | None = None, smoke: int | None = None, stop_after: int | None = None,
                 resume: bool = False, allow_fixture: bool = False) -> dict:
    """Source-held stage-one selection for one initialization seed."""
    cfg, inferred_path = _load_config(config, config_path)
    config_path = config_path or inferred_path
    if seed not in cfg.get("seeds", SEEDS):
        raise ValueError("Seed is outside the configured formal matrix")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    author_path = _resolve_path(cfg["author_csv"])
    annotations = read_author_annotations(author_path, channels=True, allow_fixture=allow_fixture)
    cache_chain = _validate_cache_chain(cfg, file_sha256(author_path), allow_fixture=allow_fixture)
    _validate_formal_contract(cfg, annotations, cache_chain, allow_fixture=allow_fixture)
    raw, dataset, opt, held, stats, source_split, opt_indices, held_indices = _stage1_data(
        cfg, annotations, smoke=smoke, allow_fixture=allow_fixture)
    output_root = _resolve_path(output if output is not None else cfg["run_root"])
    stage_dir = _stage_path(output_root, "select", seed)
    identity = _stage_identity_files(cfg, config_path, cache_chain, author_path, "select", [raw], None, allow_fixture)
    identity["source_split"] = {key: source_split[key] for key in (
        "seed", "fraction", "all_sources", "shuffled_sources", "opt_sources", "held_sources", "held_count")}
    identity["source_split_sha256"] = _sha_json(identity["source_split"])
    identity["normalization"] = stats
    identity["normalization_sha256"] = _sha_json(stats)
    identity["opt_ids"] = [raw.ids[i] for i in opt_indices]
    identity["held_ids"] = [raw.ids[i] for i in held_indices]
    identity["smoke_only"] = smoke is not None or allow_fixture
    identity["runtime_device"] = str(torch.device(cfg.get("device", "cuda")))
    identity["smoke_epoch_budget"] = cfg.get("smoke", {}).get("stage1_epochs", 2) if smoke is not None else None
    identity["identity_sha256"] = _sha_json({k: v for k, v in identity.items() if k != "identity_sha256"})
    _create_output(stage_dir, identity, resume=resume)

    model_module = _load_model_module()
    device_name = cfg.get("device", "cuda")
    device = torch.device(device_name)
    _seed_everything(seed, device)
    dims = _model_dims(raw)
    model = model_module.UnimodalSequenceHeads(dims, dim=int(cfg["model"]["dim"])).to(device)
    _check_parameter_contract(model, cfg)
    initial_state = _state_cpu(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["stage1"]["learning_rate"],
                                  weight_decay=cfg["stage1"]["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = _loader(opt, cfg["stage1"]["batch_size"], shuffle=True, generator=generator)
    valid_generator = torch.Generator().manual_seed(seed + 1)
    valid_loader = _loader(held, cfg["stage1"]["batch_size"], shuffle=False, generator=valid_generator)
    checkpoint_path = stage_dir / "checkpoint.pt"
    checkpoint = _read_stage_checkpoint(checkpoint_path, identity, resume=resume)
    history: list[dict] = []
    best = {m: {"mae": math.inf, "epoch": None, "state": None, "metrics": None} for m in MODALITIES}
    start_epoch = 0
    prior_elapsed = 0.0
    prior_peak = None
    if checkpoint is not None:
        _restore_state(model, checkpoint["last_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        best = checkpoint["best_branches"]
        start_epoch = int(checkpoint["epoch"])
        initial_state = checkpoint["initial_state"]
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        prior_peak = checkpoint.get("peak_gpu_bytes")
        _restore_rng(checkpoint["rng"], generator, device, valid_generator)

    target_epochs = int(cfg.get("smoke", {}).get("stage1_epochs", 2) if smoke is not None else cfg["stage1"]["epochs"])
    run_until = min(target_epochs, start_epoch + stop_after) if stop_after is not None else target_epochs
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, run_until):
        epoch_started = time.perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        for features, target, _ in train_loader:
            features = _move_features(features, device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            if prediction.shape != target.shape or prediction.shape[1] != 3:
                raise ValueError("UnimodalSequenceHeads must return [N,3] predictions in T/A/V order")
            present = torch.stack([features["lengths"][m] > 0 for m in MODALITIES], dim=1)
            losses = torch.stack([
                torch.abs(prediction[present[:, index], index] - target[present[:, index], index]).mean()
                if torch.any(present[:, index]) else prediction[:, index].sum() * 0.
                for index in range(3)
            ])
            loss = losses.sum()
            loss.backward()
            active = {m for index, m in enumerate(MODALITIES) if torch.any(present[:, index])}
            _stateless_branch_update(optimizer, model, active, float(cfg["stage1"]["gradient_clip"]))
            optimizer.step()
            batch_size = len(target)
            total_loss += float(loss.detach()) * batch_size
            seen += batch_size
        metrics = _evaluate_heads(model, valid_loader, device)
        for index, modality in enumerate(MODALITIES):
            mae = float(metrics[modality]["MAE"])
            if mae < best[modality]["mae"]:
                best[modality] = {
                    "mae": mae,
                    "epoch": epoch + 1,
                    "state": _channel_state(model, modality),
                    "metrics": metrics[modality],
                }
        history.append({"epoch": epoch + 1, "train_L1_sum": total_loss / max(seen, 1), "held_source": metrics,
                        "seconds": time.perf_counter() - epoch_started})
        peak = _profile_peak(device)
        if prior_peak is not None and peak is not None:
            peak = max(int(prior_peak), peak)
        checkpoint = {
            "kind": "sequence_gate_select_v1", "identity": identity, "seed": seed, "epoch": epoch + 1,
            "complete": epoch + 1 == target_epochs, "initial_state": initial_state,
            "last_state": _state_cpu(model), "best_branches": best, "optimizer": optimizer.state_dict(),
            "rng": _capture_rng(generator, device, valid_generator), "history": history,
            "selected_epochs": {m: best[m]["epoch"] for m in MODALITIES},
            "source_split": identity["source_split"], "normalization": stats,
            "fit_ids": identity["opt_ids"], "fit_sources": identity["source_split"]["opt_sources"],
            "held_ids": identity["held_ids"], "held_sources": identity["source_split"]["held_sources"],
            "input_hashes": identity["used_array_sha256"], "peak_gpu_bytes": peak,
            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        }
        _atomic_state(checkpoint_path, checkpoint)
        save_json(stage_dir / "history.json", history)

    complete = run_until == target_epochs
    final = _load_checkpoint(checkpoint_path)
    if complete:
        if any(final["selected_epochs"][m] is None for m in MODALITIES):
            raise ValueError("Source-held selection did not select an epoch for every channel")
        save_json(stage_dir / "result.json", {
            "stage": "select", "seed": seed, "complete": True, "selected_epochs": final["selected_epochs"],
            "best_channel_metrics": {m: final["best_branches"][m]["metrics"] for m in MODALITIES},
            "source_split_sha256": identity["source_split_sha256"], "normalization_sha256": identity["normalization_sha256"],
            "checkpoint_sha256": _torch_save_hash(checkpoint_path), "peak_gpu_bytes": final["peak_gpu_bytes"],
            "elapsed_seconds": final["elapsed_seconds"], "smoke_only": identity["smoke_only"],
        })
    return {"stage": "select", "seed": seed, "complete": complete,
            "selected_epochs": final.get("selected_epochs"), "checkpoint": str(checkpoint_path)}


def refit_heads(config: dict | str | Path, seed: int, *, config_path: Path | None = None,
                output: str | Path | None = None, smoke: int | None = None, stop_after: int | None = None,
                resume: bool = False, allow_fixture: bool = False) -> dict:
    """Refit selected T/A/V heads on all official TRAIN standardized sequences."""
    cfg, inferred_path = _load_config(config, config_path)
    config_path = config_path or inferred_path
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    output_root = _resolve_path(output if output is not None else cfg["run_root"])
    selection_path = _stage_path(output_root, "select", seed) / "checkpoint.pt"
    if not selection_path.is_file():
        raise FileNotFoundError("Stage-one source-held checkpoint is required before refit")
    selection = _load_checkpoint(selection_path)
    if not selection.get("complete"):
        raise ValueError("Stage-one selection must be complete before refitting")
    author_path = _resolve_path(cfg["author_csv"])
    annotations = read_author_annotations(author_path, channels=True, allow_fixture=allow_fixture)
    cache_chain = _validate_cache_chain(cfg, file_sha256(author_path), allow_fixture=allow_fixture)
    _validate_formal_contract(cfg, annotations, cache_chain, allow_fixture=allow_fixture)
    train_split = _native_split(_resolve_path(cfg["standardized_cache"]), "train", annotations, pooled=False, limit=smoke)
    train_channels = _target_rows(annotations, "train", train_split.ids, channels=True)
    train = SequenceGateSplit(train_split, labels=train_channels)
    _formal_data_check(annotations, None, train_split, None, {}, allow_fixture or smoke is not None)
    upstream_hash = _torch_save_hash(selection_path)
    identity = _stage_identity_files(cfg, config_path, cache_chain, author_path, "refit", [train_split], upstream_hash, allow_fixture)
    identity["selection_identity_sha256"] = selection["identity"]["identity_sha256"]
    identity["selected_epochs"] = selection["selected_epochs"]
    identity["smoke_only"] = smoke is not None or allow_fixture
    identity["runtime_device"] = str(torch.device(cfg.get("device", "cuda")))
    identity["smoke_epoch_budget"] = "source-selected<=stage1_epoch_budget" if smoke is not None else None
    identity["identity_sha256"] = _sha_json({k: v for k, v in identity.items() if k != "identity_sha256"})
    stage_dir = _stage_path(output_root, "refit", seed)
    _create_output(stage_dir, identity, resume=resume)

    model_module = _load_model_module()
    device = torch.device(cfg.get("device", "cuda"))
    _seed_everything(seed, device)
    model = model_module.UnimodalSequenceHeads(_model_dims(train_split), dim=int(cfg["model"]["dim"])).to(device)
    _check_parameter_contract(model, cfg)
    _restore_state(model, selection["initial_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["refit"]["learning_rate"],
                                  weight_decay=cfg["refit"]["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = _loader(train, cfg["refit"]["batch_size"], shuffle=True, generator=generator)
    checkpoint_path = stage_dir / "checkpoint.pt"
    checkpoint = _read_stage_checkpoint(checkpoint_path, identity, resume=resume)
    history = []
    start_epoch = 0
    prior_elapsed = 0.0
    prior_peak = None
    if checkpoint is not None:
        _restore_state(model, checkpoint["last_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        prior_peak = checkpoint.get("peak_gpu_bytes")
        _restore_rng(checkpoint["rng"], generator, device)
    selected_epochs = {m: int(selection["selected_epochs"][m]) for m in MODALITIES}
    total_epochs = max(selected_epochs.values())
    run_until = min(total_epochs, start_epoch + stop_after) if stop_after is not None else total_epochs
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, run_until):
        epoch_started = time.perf_counter()
        active = {m for m in MODALITIES if epoch < selected_epochs[m]}
        model.train()
        total_loss, seen = 0.0, 0
        for features, target, _ in loader:
            features = _move_features(features, device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            present = torch.stack([features["lengths"][m] > 0 for m in MODALITIES], dim=1)
            losses = torch.stack([
                torch.abs(prediction[present[:, index], index] - target[present[:, index], index]).mean()
                if torch.any(present[:, index]) else prediction[:, index].sum() * 0.
                for index in range(3)
            ])
            loss = sum(losses[index] for index, modality in enumerate(MODALITIES) if modality in active)
            loss.backward()
            present_active = {m for index, m in enumerate(MODALITIES) if m in active and torch.any(present[:, index])}
            _stateless_branch_update(optimizer, model, present_active, float(cfg["refit"]["gradient_clip"]))
            optimizer.step()
            batch_size = len(target)
            total_loss += float(loss.detach()) * batch_size
            seen += batch_size
        history.append({"epoch": epoch + 1, "active_modalities": sorted(active), "train_L1_sum": total_loss / max(seen, 1),
                        "seconds": time.perf_counter() - epoch_started})
        peak = _profile_peak(device)
        if prior_peak is not None and peak is not None:
            peak = max(int(prior_peak), peak)
        checkpoint = {
            "kind": "sequence_gate_refit_v1", "identity": identity, "seed": seed, "epoch": epoch + 1,
            "complete": epoch + 1 == total_epochs, "selected_epochs": selected_epochs,
            "selection_checkpoint_sha256": upstream_hash, "last_state": _state_cpu(model),
            "optimizer": optimizer.state_dict(), "rng": _capture_rng(generator, device), "history": history,
            "fit_ids": train_split.ids, "input_hashes": identity["used_array_sha256"],
            "peak_gpu_bytes": peak, "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        }
        _atomic_state(checkpoint_path, checkpoint)
        save_json(stage_dir / "history.json", history)
    complete = run_until == total_epochs
    final = _load_checkpoint(checkpoint_path)
    if complete:
        save_json(stage_dir / "result.json", {
            "stage": "refit", "seed": seed, "complete": True, "selected_epochs": selected_epochs,
            "checkpoint_sha256": _torch_save_hash(checkpoint_path), "selection_checkpoint_sha256": upstream_hash,
            "fit_count": len(train_split), "peak_gpu_bytes": final["peak_gpu_bytes"],
            "elapsed_seconds": final["elapsed_seconds"], "smoke_only": identity["smoke_only"],
        })
    return {"stage": "refit", "seed": seed, "complete": complete, "checkpoint": str(checkpoint_path)}


def _frozen_knowledge_hash(model: torch.nn.Module) -> str:
    selected = {name: tensor for name, tensor in model.state_dict().items()
                if ".adapter." in name or ".decoder." in name or name.endswith(".adapter") or name.endswith(".decoder")}
    if not selected:
        raise ValueError("Could not identify frozen adapter/decoder weights for the knowledge hash")
    return _state_fingerprint(selected)


def _check_frozen_knowledge(model: torch.nn.Module) -> str:
    model.freeze_knowledge()
    trainable = [name for name, parameter in model.named_parameters()
                 if (".adapter." in name or ".decoder." in name) and parameter.requires_grad]
    if trainable:
        raise ValueError("freeze_knowledge() left adapter or decoder parameters trainable")
    return _frozen_knowledge_hash(model)


def _policy_head_report(model: torch.nn.Module, case: str, expected_count: int | None = None) -> dict:
    params = [(name, parameter) for name, parameter in model.named_parameters() if "policy" in name.lower()]
    if not params:
        raise ValueError("The six-case scalar-gate model must expose its policy head in every case")
    if any(not parameter.requires_grad for _, parameter in params):
        raise ValueError("The policy head must remain trainable")
    routes = case in POLICY_ROUTED_CASES
    distilled = case in DISTILLED_CASES
    active = case in POLICY_ACTIVE_CASES
    if not active:
        status = "present_but_unused"
    elif distilled and routes:
        status = "trained_by_distillation_and_used_for_label_free_routing"
    elif distilled:
        status = "trained_by_distillation_but_not_used_for_routing"
    else:
        status = "trained_by_prediction_loss_and_used_for_label_free_routing_without_distillation"
    count = sum(parameter.numel() for _, parameter in params)
    if expected_count is not None and count != int(expected_count):
        raise ValueError(f"Policy-head parameter count changed: expected {expected_count}, found {count}")
    return {"status": status, "parameter_count": count,
            "distilled": distilled, "active": active, "used_for_routing": routes}


def _policy_state_fingerprint(model: torch.nn.Module) -> str:
    state = {name: tensor for name, tensor in model.state_dict().items() if "policy" in name.lower()}
    if not state:
        raise ValueError("Policy head state is missing")
    return _state_fingerprint(state)


def _fusion_forward(model: torch.nn.Module, features: dict, target: torch.Tensor | None):
    output = model(features, target=target) if model.training else model(features)
    if not isinstance(output, dict) or not {"prediction", "contrastive", "distillation", "ratios", "policy_ratios"}.issubset(output):
        raise ValueError("SequenceGateRegressor output must contain scores, ratios, and both auxiliary scalars")
    prediction = output["prediction"].reshape(-1)
    if not torch.isfinite(prediction).all() or len(prediction) != len(features["text"]):
        raise ValueError("SequenceGateRegressor prediction shape or values are invalid")
    for key in ("contrastive", "distillation"):
        value = output[key]
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value):
            raise ValueError(f"SequenceGateRegressor {key} term must be one finite scalar")
    for key in ("ratios", "policy_ratios"):
        value = output[key]
        if value.shape != (len(prediction), 3) or not torch.isfinite(value).all() or torch.any(value < 0):
            raise ValueError(f"SequenceGateRegressor {key} must be finite nonnegative [N,3]")
    if model.case in POLICY_ROUTED_CASES and not torch.allclose(output["ratios"], output["policy_ratios"], atol=1e-7, rtol=0):
        raise ValueError("Distilled pre/post-normalization cases must route with the learned policy")
    distillation = float(output["distillation"].detach())
    if (not model.training or model.case not in DISTILLED_CASES) and distillation != 0.0:
        raise ValueError("Distillation must be zero for inference and nondistilled controls")
    if model.training and model.case in DISTILLED_CASES and distillation < -1e-7:
        raise ValueError("KL distillation loss must be nonnegative")
    return output


@torch.no_grad()
def _evaluate_fusion(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, list[dict]]:
    model.eval()
    pred_rows, true_rows, ids = [], [], []
    for features, target, sample_ids in loader:
        features = _move_features(features, device)
        output = _fusion_forward(model, features, None)
        pred_rows.append(output["prediction"].reshape(-1).detach().cpu().numpy())
        true_rows.append(target.numpy())
        ids.extend(sample_ids)
    predictions, targets = np.concatenate(pred_rows), np.concatenate(true_rows)
    metrics = native_metrics(predictions, targets, "simsv2")
    table = [{"id": sid, "prediction": float(pred)} for sid, pred in zip(ids, predictions)]
    return metrics, table


@torch.no_grad()
def _gate_intervention_stats(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict | None:
    """Measure the selected model before and after target-free one-branch probes."""
    probe = getattr(model, "probe", None)
    if not callable(probe):
        return None
    model.eval()
    baseline, source_none, targets = [], [], []
    interventions = {m: [] for m in MODALITIES}
    for features, target, _ in loader:
        features = _move_features(features, device)
        present = torch.stack([features["lengths"][m] > 0 for m in MODALITIES], dim=1)
        base_ratios = present.to(features["text"].dtype)
        base_output = model(features)
        baseline.append(base_output["prediction"].reshape(-1).cpu().numpy())
        source_none_probe = getattr(model, "probe_source_none", None)
        if callable(source_none_probe):
            source_none.append(source_none_probe(features)["prediction"].reshape(-1).cpu().numpy())
        targets.append(target.numpy())
        for index, modality in enumerate(MODALITIES):
            ratios = torch.zeros_like(base_ratios)
            ratios[:, index] = present[:, index].to(ratios.dtype)
            output = probe(features, ratios)
            interventions[modality].append(output["prediction"].reshape(-1).cpu().numpy())
    base = np.concatenate(baseline)
    truth = np.concatenate(targets)
    result = {"baseline": native_metrics(base, truth, "simsv2"), "interventions": {}}
    if source_none:
        alternate = np.concatenate(source_none)
        result["source_none"] = {
            "after": native_metrics(alternate, truth, "simsv2"),
            "mean_signed_prediction_change": float(np.mean(alternate - base)),
            "mean_absolute_prediction_change": float(np.mean(np.abs(alternate - base))),
        }
    for modality in MODALITIES:
        after = np.concatenate(interventions[modality])
        result["interventions"][modality] = {
            "gate_ratios": {"selected_modality": modality, "other_modalities": 0.0},
            "after": native_metrics(after, truth, "simsv2"),
            "mean_signed_prediction_change": float(np.mean(after - base)),
            "mean_absolute_prediction_change": float(np.mean(np.abs(after - base))),
        }
    return result


def fit_fusion_case(config: dict | str | Path, seed: int, case: str, *, config_path: Path | None = None,
                    output: str | Path | None = None, smoke: int | None = None, stop_after: int | None = None,
                    resume: bool = False, allow_fixture: bool = False,
                    datasets: tuple[NativeSplit, NativeSplit, SequenceGateSplit, SequenceGateSplit] | None = None) -> dict:
    """Fit one matched scalar-gate fusion case on native M."""
    cfg, inferred_path = _load_config(config, config_path)
    config_path = config_path or inferred_path
    if case not in cfg.get("cases", CASES):
        raise ValueError("Unknown scalar-gate case")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    output_root = _resolve_path(output if output is not None else cfg["run_root"])
    refit_path = _stage_path(output_root, "refit", seed) / "checkpoint.pt"
    if not refit_path.is_file():
        raise FileNotFoundError("Complete stage-one refit checkpoint is required before fusion")
    refit = _load_checkpoint(refit_path)
    if not refit.get("complete"):
        raise ValueError("Stage-one refit must be complete before fusion")
    selection_path = _stage_path(output_root, "select", seed) / "checkpoint.pt"
    selection = _load_checkpoint(selection_path)
    if not selection.get("complete"):
        raise ValueError("Complete source-held selection is required before fusion")

    author_path = _resolve_path(cfg["author_csv"])
    annotations = read_author_annotations(author_path, channels=False, allow_fixture=allow_fixture)
    cache_chain = _validate_cache_chain(cfg, file_sha256(author_path), allow_fixture=allow_fixture)
    _validate_formal_contract(cfg, annotations, cache_chain, allow_fixture=allow_fixture)
    if datasets is None:
        train_split, valid_split, train, valid = _stage2_data(cfg, annotations, smoke=smoke, allow_fixture=allow_fixture)
    else:
        train_split, valid_split, train, valid = datasets
    upstream_hash = _torch_save_hash(refit_path)
    stage_dir = _stage_path(output_root, "fusion", seed, case)

    model_module = _load_model_module()
    device = torch.device(cfg.get("device", "cuda"))
    _seed_everything(seed, device)
    model = model_module.SequenceGateRegressor(_model_dims(train_split), case, **cfg["model"]).to(device)
    model.load_unimodal_state(refit["last_state"])
    frozen_hash = _check_frozen_knowledge(model)
    parameter_count, _ = _check_parameter_contract(model, cfg, frozen=True)
    initial_state_hash = _state_fingerprint(_state_cpu(model))
    policy_report = _policy_head_report(model, case, cfg.get("expected_parameters", {}).get("policy_head"))
    initial_policy_hash = _policy_state_fingerprint(model)
    identity = _stage_identity_files(cfg, config_path, cache_chain, author_path, "fusion", [train_split, valid_split], upstream_hash, allow_fixture)
    identity["selection_checkpoint_sha256"] = _torch_save_hash(selection_path)
    identity["refit_identity_sha256"] = refit["identity"]["identity_sha256"]
    identity["case"] = case
    identity["seed"] = seed
    identity["initial_fusion_state_sha256"] = initial_state_hash
    identity["policy_head"] = policy_report
    identity["smoke_only"] = smoke is not None or allow_fixture
    identity["runtime_device"] = str(device)
    identity["smoke_epoch_budget"] = cfg.get("smoke", {}).get("fusion_epochs", 2) if smoke is not None else None
    identity["identity_sha256"] = _sha_json({k: v for k, v in identity.items() if k != "identity_sha256"})
    _create_output(stage_dir, identity, resume=resume)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg["fusion"]["learning_rate"], weight_decay=cfg["fusion"]["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = _loader(train, cfg["fusion"]["batch_size"], shuffle=True, generator=generator)
    valid_generator = torch.Generator().manual_seed(seed + 1)
    valid_loader = _loader(valid, cfg["fusion"]["batch_size"], shuffle=False, generator=valid_generator)
    checkpoint_path = stage_dir / "checkpoint.pt"
    checkpoint = _read_stage_checkpoint(checkpoint_path, identity, resume=resume)
    history = []
    best_epoch = None
    best_mae = math.inf
    best_state = None
    best_metrics = None
    best_predictions = None
    start_epoch = 0
    prior_elapsed = 0.0
    prior_peak = None
    if checkpoint is not None:
        _restore_state(model, checkpoint["last_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        best_epoch = checkpoint["best_epoch"]
        best_mae = checkpoint["best_valid_mae"]
        best_state = checkpoint["best_state"]
        best_metrics = checkpoint["best_valid_metrics"]
        best_predictions = checkpoint["best_valid_predictions"]
        start_epoch = int(checkpoint["epoch"])
        prior_elapsed = float(checkpoint.get("elapsed_seconds", 0.0))
        prior_peak = checkpoint.get("peak_gpu_bytes")
        _restore_rng(checkpoint["rng"], generator, device, valid_generator)
    target_epochs = int(cfg.get("smoke", {}).get("fusion_epochs", 2) if smoke is not None else cfg["fusion"]["epochs"])
    run_until = min(target_epochs, start_epoch + stop_after) if stop_after is not None else target_epochs
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, run_until):
        epoch_started = time.perf_counter()
        model.train()
        loss_total, l1_total, contrastive_total, distillation_total, seen = 0., 0., 0., 0., 0
        ratio_total = np.zeros(3, np.float64)
        policy_ratio_total = np.zeros(3, np.float64)
        variance_total = np.zeros(3, np.float64)
        batches = 0
        for features, target, _ in train_loader:
            features = _move_features(features, device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = _fusion_forward(model, features, target)
            l1 = torch.abs(output["prediction"].reshape(-1) - target).mean()
            contrastive = output["contrastive"]
            distillation = output["distillation"]
            loss = (l1 + float(cfg["fusion"]["contrastive_weight"]) * contrastive
                    + float(cfg["fusion"]["distillation_weight"]) * distillation)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["fusion"]["gradient_clip"]))
            optimizer.step()
            batch_size = len(target)
            loss_total += float(loss.detach()) * batch_size
            l1_total += float(l1.detach()) * batch_size
            contrastive_total += float(contrastive.detach()) * batch_size
            distillation_total += float(distillation.detach()) * batch_size
            ratio_total += np.asarray(_mean_vector(output, "ratios")) * batch_size
            policy_ratio_total += np.asarray(_mean_vector(output, "policy_ratios")) * batch_size
            variance_total += np.asarray(_mean_vector(output, "branch_variances")) * batch_size
            seen += batch_size
            batches += 1
        valid_metrics, valid_predictions = _evaluate_fusion(model, valid_loader, device)
        if float(valid_metrics["MAE"]) < best_mae:
            best_mae = float(valid_metrics["MAE"])
            best_epoch = epoch + 1
            best_state = _state_cpu(model)
            best_metrics = valid_metrics
            best_predictions = valid_predictions
        history.append({
            "epoch": epoch + 1,
            "train_loss": loss_total / max(seen, 1),
            "train_L1": l1_total / max(seen, 1),
            "contrastive": contrastive_total / max(seen, 1),
            "distillation": distillation_total / max(seen, 1),
            "valid": valid_metrics,
            "mean_ratios": (ratio_total / max(seen, 1)).tolist(),
            "mean_policy_ratios": (policy_ratio_total / max(seen, 1)).tolist(),
            "mean_branch_variances": (variance_total / max(seen, 1)).tolist(),
            "seconds": time.perf_counter() - epoch_started,
            "batches": batches,
        })
        peak = _profile_peak(device)
        if prior_peak is not None and peak is not None:
            peak = max(int(prior_peak), peak)
        checkpoint = {
            "kind": "sequence_gate_fusion_v1", "identity": identity, "seed": seed, "case": case,
            "epoch": epoch + 1, "complete": epoch + 1 == target_epochs, "last_state": _state_cpu(model),
            "best_state": best_state, "best_epoch": best_epoch, "best_valid_mae": best_mae,
            "best_valid_metrics": best_metrics, "best_valid_predictions": best_predictions,
            "optimizer": optimizer.state_dict(), "rng": _capture_rng(generator, device, valid_generator), "history": history,
            "frozen_knowledge_sha256": frozen_hash, "parameter_count": parameter_count,
            "initial_fusion_state_sha256": initial_state_hash,
            "policy_head": policy_report, "initial_policy_state_sha256": initial_policy_hash,
            "fit_ids": train_split.ids, "valid_ids": valid_split.ids,
            "input_hashes": identity["used_array_sha256"], "peak_gpu_bytes": peak,
            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
        }
        _atomic_state(checkpoint_path, checkpoint)
        save_json(stage_dir / "history.json", history)

    complete = run_until == target_epochs
    final = _load_checkpoint(checkpoint_path)
    if complete:
        if len(final["best_valid_predictions"] or []) != len(valid_split):
            if not allow_fixture and smoke is None:
                raise ValueError("Formal native VALID prediction receipt must contain 647 rows")
        _restore_state(model, final["best_state"])
        if _frozen_knowledge_hash(model) != frozen_hash:
            raise ValueError("Frozen adapter/decoder weights changed during fusion training")
        best_policy_hash = _policy_state_fingerprint(model)
        policy_state_unchanged = best_policy_hash == initial_policy_hash
        if not policy_report["active"] and not policy_state_unchanged:
            raise ValueError("Non-distilled control unexpectedly updated the unused policy head")
        intervention_stats = _gate_intervention_stats(model, valid_loader, device)
        save_json(stage_dir / "valid_predictions.json", final["best_valid_predictions"])
        if intervention_stats is not None:
            save_json(stage_dir / "gate_intervention_stats.json", intervention_stats)
        save_json(stage_dir / "result.json", {
            "stage": "fusion", "seed": seed, "case": case, "complete": True,
            "best_epoch": final["best_epoch"], "valid_metrics": final["best_valid_metrics"],
            "valid_prediction_count": len(final["best_valid_predictions"]),
            "checkpoint_sha256": _torch_save_hash(checkpoint_path), "frozen_knowledge_sha256": frozen_hash,
            "parameter_count": parameter_count, "initial_fusion_state_sha256": initial_state_hash,
            "policy_head": policy_report, "initial_policy_state_sha256": initial_policy_hash,
            "best_policy_state_sha256": best_policy_hash,
            "policy_state_unchanged": policy_state_unchanged,
            "unused_policy_unchanged": policy_state_unchanged if not policy_report["active"] else None,
            "peak_gpu_bytes": final["peak_gpu_bytes"],
            "elapsed_seconds": final["elapsed_seconds"], "smoke_only": identity["smoke_only"],
            "gate_intervention_stats": intervention_stats,
        })
    return {"stage": "fusion", "seed": seed, "case": case, "complete": complete,
            "best_epoch": final.get("best_epoch"), "checkpoint": str(checkpoint_path)}


def _validate_matrix(config: dict, output: Path) -> dict:
    expected = {
        "select": [(seed, None) for seed in SEEDS],
        "refit": [(seed, None) for seed in SEEDS],
        "fusion": [(seed, case) for seed in SEEDS for case in CASES],
    }
    records = {}
    fusion_receipts = {}
    for stage, keys in expected.items():
        for seed, case in keys:
            path = _stage_path(output, stage, seed, case) / "result.json"
            key = f"{stage}/seed_{seed}" + (f"/{case}" if case else "")
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                records[key] = bool(result.get("complete")) and not bool(result.get("smoke_only"))
                if stage == "fusion" and records[key]:
                    fusion_receipts.setdefault(seed, []).append(result)
            else:
                records[key] = False
    for seed, receipts in fusion_receipts.items():
        for row in receipts:
            case = row.get("case")
            expected_status = (
                "trained_by_prediction_loss_and_used_for_label_free_routing_without_distillation"
                if case == "learned_post_norm" else
                "present_but_unused" if case not in POLICY_ACTIVE_CASES else
                "trained_by_distillation_and_used_for_label_free_routing" if case in POLICY_ROUTED_CASES else
                "trained_by_distillation_but_not_used_for_routing"
            )
            if row.get("policy_head", {}).get("status") != expected_status:
                raise ValueError(f"Fusion policy usage receipt is incorrect for seed {seed}, case {case}")
            if case not in POLICY_ACTIVE_CASES and row.get("unused_policy_unchanged") is not True:
                raise ValueError(f"Nondistilled policy head changed for seed {seed}, case {case}")
        if len(receipts) == len(CASES):
            initial_hashes = {row.get("initial_fusion_state_sha256") for row in receipts}
            parameter_counts = {row.get("parameter_count") for row in receipts}
            if len(initial_hashes) != 1 or len(parameter_counts) != 1:
                raise ValueError(f"Fusion cases for seed {seed} did not share identical initial weights and parameter count")
    return {"complete": all(records.values()), "tasks": records}


def train(config: dict | str | Path, *, stage: str = "all", seed: int | None = None, case: str | None = None,
          config_path: Path | None = None, output: str | Path | None = None, smoke: int | None = None,
          stop_after: int | None = None, resume: bool = False) -> list[dict]:
    """Public runner API; fixture bypasses are intentionally Python-only."""
    cfg, inferred_path = _load_config(config, config_path)
    config_path = config_path or inferred_path
    if stage not in ("select", "refit", "fusion", "all"):
        raise ValueError("stage must be select, refit, fusion, or all")
    seeds = [seed] if seed is not None else list(cfg["seeds"])
    cases = [case] if case is not None else list(cfg["cases"])
    if stage not in ("fusion", "all") and case is not None:
        raise ValueError("--case is only meaningful for fusion training")
    if stage in ("fusion", "all") and any(value not in cfg["cases"] for value in cases):
        raise ValueError("Requested case is outside the configured case matrix")
    output_base = _resolve_path(output if output is not None else cfg["run_root"])
    output_root = output_base / f"smoke_{smoke}" if smoke is not None else output_base
    results = []
    for current_seed in seeds:
        if stage in ("select", "all"):
            results.append(select_heads(cfg, current_seed, config_path=config_path, output=output_root, smoke=smoke,
                                        stop_after=stop_after, resume=resume))
            if stage == "all" and not results[-1]["complete"]:
                break
        if stage in ("refit", "all"):
            results.append(refit_heads(cfg, current_seed, config_path=config_path, output=output_root, smoke=smoke,
                                       stop_after=stop_after, resume=resume))
            if stage == "all" and not results[-1]["complete"]:
                break
        if stage in ("fusion", "all"):
            # All three arms use identical initial weights and loader seeds. Reuse
            # the same verified mmap datasets across cases when running a matrix.
            dataset_bundle = None
            if case is None and len(cases) > 1:
                author_path = _resolve_path(cfg["author_csv"])
                annotation = read_author_annotations(author_path, channels=False)
                _, _, train_data, valid_data = _stage2_data(cfg, annotation, smoke=smoke, allow_fixture=False)
                dataset_bundle = (train_data.split, valid_data.split, train_data, valid_data)
            for current_case in cases:
                results.append(fit_fusion_case(cfg, current_seed, current_case, config_path=config_path, output=output_root,
                                                smoke=smoke, stop_after=stop_after, resume=resume, datasets=dataset_bundle))
                if stage == "all" and not results[-1]["complete"]:
                    break
            if stage == "all" and results and results[-1].get("stage") == "fusion" and not results[-1]["complete"]:
                break
    if smoke is None:
        matrix = _validate_matrix(cfg, output_root)
        matrix["smoke_only"] = False
        save_json(output_root / "matrix_status.json", matrix)
    return results

