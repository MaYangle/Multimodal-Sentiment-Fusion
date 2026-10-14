"""Pinned CH-SIMS v2 native-score MacBERT text tuning controls.

The registered comparison is a top-four-layer MacBERT fine-tune against the
same CLS/dropout/scalar-head model with a frozen encoder. It uses only the
author train and validation text/labels and selects the first minimum native
validation MAE checkpoint.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import csv
import hashlib
import inspect
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from jevaffect.native_metrics import native_metrics
from jevaffect.native_training import digest, save_checkpoint, save_json

ROOT = Path(__file__).resolve().parents[1]
CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
MODEL_SHA256 = "db0506d985574b80c33eec1cf13bd4c130585568753175871095ca13dbad9e23"
MODEL_ID = "hfl/chinese-macbert-base"
MODEL_REVISION = "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
EXPECTED_ROWS = {"train": 2722, "valid": 647, "test": 1034}
SEEDS = (13, 42, 2026)
MODES = ("top4", "frozen")
PINNED_MANIFEST_FILES = {
    "added_tokens.json": {"sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a", "bytes": 2},
    "config.json": {"sha256": "648c8a2b51c12551f94bae19a66c21a59c9bd8aa0ebb99f2a0af7236cdb99daa", "bytes": 659},
    "pytorch_model.bin": {"sha256": MODEL_SHA256, "bytes": 411582858},
    "README.md": {"sha256": "1516e1e24fff28dfeaed65f714bc676e4739d4c70e6c9f356fbf41d2712251e5", "bytes": 3842},
    "special_tokens_map.json": {"sha256": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3", "bytes": 112},
    "tokenizer_config.json": {"sha256": "61785aeaba176fba6d6489f27dccfd2ddee6aee2af0e590451cab7d8b57e0874", "bytes": 19},
    "vocab.txt": {"sha256": "45bbac6b341c319adc98a532532882e91a9cefc0329aa57bac9ae761c27b291c", "bytes": 109540},
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def parse_author_csv(path: str | Path):
    """Validate the pinned cohort and return only train/valid text and scores.

    Test rows contribute IDs to duplicate detection and the cohort digest. The
    test label and text fields are never accessed or passed to a tokenizer.
    """
    path = Path(path)
    actual_csv_sha = digest(path)
    if actual_csv_sha != CSV_SHA256:
        raise ValueError("Pinned CH-SIMS v2 author CSV SHA-256 mismatch")

    records = {"train": [], "valid": []}
    all_ids: set[str] = set()
    ids_by_split = {"train": [], "valid": [], "test": []}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "clip_id", "mode", "label", "text"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Author CSV is missing required columns")
        for row in reader:
            split = row.get("mode")
            if split not in ids_by_split:
                raise ValueError("Unknown author CSV split")
            video_id, clip_id = row.get("video_id", ""), row.get("clip_id", "")
            if not video_id or not clip_id:
                raise ValueError("Empty author sample ID")
            sample_id = video_id + "/" + clip_id
            if sample_id in all_ids:
                raise ValueError("Duplicate author sample ID: " + sample_id)
            all_ids.add(sample_id)
            ids_by_split[split].append(sample_id)

            if split == "test":
                # Deliberately do not read test label or text values.
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Empty official " + split + " transcript")
            try:
                score = float(row["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid continuous author score") from exc
            if not math.isfinite(score) or abs(score) > 1.0:
                raise ValueError("CH-SIMS v2 score outside [-1,1]")
            records[split].append({"id": sample_id, "text": text, "target": score})

    counts = {split: len(values) for split, values in ids_by_split.items()}
    if counts != EXPECTED_ROWS:
        raise ValueError("Unexpected pinned author cohort counts: " + repr(counts))
    if set(ids_by_split["train"]) & set(ids_by_split["valid"]):
        raise ValueError("Train/validation ID overlap")
    cohort = {
        "csv_sha256": actual_csv_sha,
        "counts": counts,
        "ids_sha256": {
            split: _sha_bytes(_canonical(ids)) for split, ids in ids_by_split.items()
        },
    }
    return records, cohort


def verify_pinned_pretrained(model_dir: str | Path) -> dict:
    """Verify every file named by the download manifest before loading it."""
    model_dir = Path(model_dir).resolve()
    manifest_path = model_dir / "jevaffect_download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Missing pinned MacBERT download manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Pinned MacBERT model/revision mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Pinned MacBERT manifest has no file records")
    if files != PINNED_MANIFEST_FILES:
        raise ValueError("Pinned MacBERT manifest file records changed")
    verified = {}
    for name, record in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
            raise ValueError("Unsafe path in MacBERT manifest: " + name)
        path = model_dir / relative
        if not path.is_file():
            raise FileNotFoundError("Missing MacBERT manifest file: " + name)
        actual = digest(path)
        expected = record.get("sha256")
        size = record.get("bytes")
        if actual != expected or path.stat().st_size != size:
            raise ValueError("MacBERT file hash/size mismatch: " + name)
        verified[name] = {"sha256": actual, "bytes": path.stat().st_size}
    weights = verified.get("pytorch_model.bin")
    if weights is None or weights["sha256"] != MODEL_SHA256:
        raise ValueError("Pinned MacBERT weights SHA-256 mismatch")
    for required in ("config.json", "vocab.txt", "tokenizer_config.json"):
        if required not in verified:
            raise ValueError("Missing required pinned tokenizer/model file: " + required)
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "manifest_sha256": digest(manifest_path),
        "files": verified,
    }


class MacBERTScalarRegressor(nn.Module):
    """MacBERT CLS with dropout 0.1 and an original-score scalar head."""

    def __init__(self, encoder: nn.Module, mode: str, dropout: float = 0.1):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be 'top4' or 'frozen'")
        layers = getattr(getattr(encoder, "encoder", None), "layer", None)
        if layers is None or len(layers) != 12:
            raise ValueError("Pinned MacBERT-base must expose twelve encoder layers")
        hidden_size = int(encoder.config.hidden_size)
        self.encoder = encoder
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

        self.encoder.requires_grad_(False)
        if mode == "top4":
            for layer in self.encoder.encoder.layer[-4:]:
                layer.requires_grad_(True)
        # The pooler is unused by CLS extraction and remains frozen in both arms.
        if getattr(self.encoder, "pooler", None) is not None:
            self.encoder.pooler.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :]
        return self.head(self.dropout(cls)).squeeze(-1)


def _load_pinned_tokenizer(model_dir: Path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        str(model_dir), revision=MODEL_REVISION, local_files_only=True,
    )


def _load_pinned_model(model_dir: Path, mode: str):
    from transformers import AutoModel
    encoder = AutoModel.from_pretrained(
        str(model_dir), revision=MODEL_REVISION, local_files_only=True,
        attn_implementation="eager",
    )
    if int(encoder.config.hidden_size) != 768 or len(encoder.encoder.layer) != 12:
        raise ValueError("Pinned MacBERT-base width/depth changed")
    if (float(encoder.config.hidden_dropout_prob) != 0.1
            or float(encoder.config.attention_probs_dropout_prob) != 0.1):
        raise ValueError("Pinned MacBERT underlying dropout must remain 0.1")
    return MacBERTScalarRegressor(encoder, mode=mode, dropout=0.1)


class TokenRows(Dataset):
    def __init__(self, records, tokenized):
        self.records = records
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {
            "id": row["id"],
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


def _tokenize(tokenizer, records, max_tokens: int):
    encoded = tokenizer(
        [row["text"] for row in records], padding="max_length", truncation=True,
        max_length=max_tokens, return_tensors="pt",
    )
    input_ids, attention_mask = encoded.get("input_ids"), encoded.get("attention_mask")
    if not torch.is_tensor(input_ids) or not torch.is_tensor(attention_mask):
        raise ValueError("Tokenizer must return tensor input_ids and attention_mask")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("Invalid token tensor shapes")
    if input_ids.shape != (len(records), max_tokens):
        raise ValueError("Tokenizer returned an unexpected sequence length")
    return {"input_ids": input_ids.long(), "attention_mask": attention_mask.long()}


def _rng_state(generator: torch.Generator, device: torch.device):
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "loader": generator.get_state(),
    }


def _restore_rng(state, generator: torch.Generator, device: torch.device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader"])


def _seed_everything(seed: int, device: torch.device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _runtime_identity(device: torch.device, model_factory, tokenizer_factory, num_threads: int):
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": transformers_version,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda if device.type == "cuda" else None,
        "torch_num_threads": num_threads,
        "loader_factory": _callable_identity(model_factory, "pinned_model_loader"),
        "tokenizer_factory": _callable_identity(tokenizer_factory, "pinned_tokenizer_loader"),
    }


def _callable_identity(factory, default_name: str):
    if factory is None:
        return {"name": default_name, "revision": MODEL_REVISION,
                "local_files_only": True,
                "attn_implementation": "eager" if default_name == "pinned_model_loader" else None}
    module = getattr(factory, "__module__", type(factory).__module__)
    qualname = getattr(factory, "__qualname__", type(factory).__qualname__)
    try:
        source_sha = _sha_bytes(inspect.getsource(factory).encode("utf-8"))
    except (OSError, TypeError):
        source_sha = None
    return {"module": module, "qualname": qualname, "source_sha256": source_sha}


def _implementation_identity():
    paths = (
        Path(__file__),
        ROOT / "scripts" / "train_native_text_tuning.py",
        ROOT / "jevaffect" / "native_metrics.py",
        ROOT / "jevaffect" / "native_training.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): digest(path)
        for path in paths if path.is_file()
    }


def _trainable_counts(model: MacBERTScalarRegressor):
    encoder_count = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_count = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    return {"encoder": encoder_count, "head": head_count,
            "total": encoder_count + head_count}


def _total_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    predictions, targets, ids = [], [], []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        if output.shape != batch["target"].shape or not torch.isfinite(output).all():
            raise ValueError("Nonfinite or wrongly shaped scalar regression output")
        predictions.extend(output.float().cpu().tolist())
        targets.extend(batch["target"].tolist())
        ids.extend(batch["id"])
    # Match native_training.evaluate: preserve float32 at CH-SIMS boundaries.
    p = np.asarray(predictions, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    metrics = native_metrics(p, y, "simsv2")
    rows = [
        {"id": sample_id, "target": float(target), "prediction": float(prediction)}
        for sample_id, target, prediction in zip(ids, y, p)
    ]
    return metrics, rows


def _save_predictions(path: Path, predictions):
    save_json(path, predictions)


def train(config: dict, seed: int, *, mode: str, resume: bool = False,
          smoke_limit: int | None = None, stop_after: int | None = None,
          model_factory: Callable | None = None,
          tokenizer_factory: Callable | None = None):
    """Train one registered CH-SIMS mode/seed; injectable factories serve tests."""
    if config.get("dataset") != "simsv2":
        raise ValueError("This native text tuning entry point only supports CH-SIMS v2")
    if (config.get("model_id") != MODEL_ID
            or config.get("model_revision") != MODEL_REVISION
            or config.get("model_sha256") != MODEL_SHA256
            or config.get("author_csv_sha256") != CSV_SHA256
            or config.get("expected_rows") != EXPECTED_ROWS):
        raise ValueError("Config does not match the registered CH-SIMS/MacBERT pins")
    if tuple(config.get("seeds", ())) != SEEDS or seed not in SEEDS:
        raise ValueError("Formal seed contract is 13, 42, 2026")
    if mode not in MODES:
        raise ValueError("mode must be 'top4' or 'frozen'")
    if smoke_limit is not None and smoke_limit < 1:
        raise ValueError("smoke_limit must be positive")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if config.get("batch_size") != 32 or config.get("max_epochs") != 12 or config.get("patience") != 3:
        raise ValueError("The registered tuning budget is fixed at batch 32, 12 epochs, patience 3")
    if config.get("max_tokens") != 64 or config.get("dropout") != 0.1:
        raise ValueError("The registered text architecture requires max_tokens=64 and dropout=0.1")
    if config.get("encoder_lr") != 1e-5 or config.get("head_lr") != 2e-4:
        raise ValueError("The registered encoder/head learning rates are fixed")
    if config.get("weight_decay") != 0.01 or config.get("max_grad_norm") != 1.0:
        raise ValueError("The registered optimizer settings are fixed")
    if config.get("warmup_fraction") != 0.1:
        raise ValueError("The registered warmup fraction is fixed at 0.1")

    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable")
    num_threads = int(config.get("torch_num_threads", 4))
    torch.set_num_threads(num_threads)
    _seed_everything(seed, device)

    csv_path = _resolve(config["author_csv"])
    model_dir = _resolve(config["pretrained_dir"])
    records, cohort = parse_author_csv(csv_path)
    model_identity = verify_pinned_pretrained(model_dir)
    if smoke_limit is not None:
        records = {split: rows[:smoke_limit] for split, rows in records.items()}
    if not records["train"] or not records["valid"]:
        raise ValueError("Training and validation cohorts must be nonempty")

    if tokenizer_factory is None:
        tokenizer = _load_pinned_tokenizer(model_dir)
    else:
        tokenizer = tokenizer_factory(model_dir)
    tokenized = {
        split: _tokenize(tokenizer, records[split], config["max_tokens"])
        for split in ("train", "valid")
    }
    datasets = {split: TokenRows(records[split], tokenized[split]) for split in ("train", "valid")}
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True,
                             generator=generator, num_workers=0),
        "valid": DataLoader(datasets["valid"], batch_size=config["batch_size"], shuffle=False,
                             num_workers=0),
    }

    if model_factory is None:
        model = _load_pinned_model(model_dir, mode)
    else:
        model = model_factory(model_dir, mode)
    if not isinstance(model, MacBERTScalarRegressor) or model.mode != mode:
        raise TypeError("Model factory must return a MacBERTScalarRegressor for the requested mode")
    model.to(device)
    counts = _trainable_counts(model)
    total_parameters = _total_parameter_count(model)
    if counts["head"] == 0 or (mode == "frozen" and counts["encoder"] != 0):
        raise ValueError("Model trainable-parameter configuration is invalid")
    if mode == "top4" and counts["encoder"] == 0:
        raise ValueError("Top-four mode has no trainable encoder parameters")

    parameter_groups = []
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_params:
        parameter_groups.append({"params": encoder_params, "lr": config["encoder_lr"]})
    parameter_groups.append({"params": model.head.parameters(), "lr": config["head_lr"]})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config["weight_decay"])
    updates_per_epoch = math.ceil(len(datasets["train"]) / config["batch_size"])
    total_updates = config["max_epochs"] * updates_per_epoch
    warmup_steps = round(total_updates * config["warmup_fraction"])

    def lr_factor(step):
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return max(0.0, float(total_updates - step) / max(1, total_updates - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    runtime = _runtime_identity(device, model_factory, tokenizer_factory, num_threads)
    source_identity = {
        "csv_sha256": cohort["csv_sha256"],
        "cohort": cohort,
        "active_ids_sha256": {
            split: _sha_bytes(_canonical([r["id"] for r in records[split]]))
            for split in ("train", "valid")
        },
        "active_counts": {split: len(rows) for split, rows in records.items()},
        "model_tokenizer": model_identity,
    }
    effective_config = dict(config)
    effective_config.update({
        "mode": mode,
        "seed": seed,
        "smoke_limit": smoke_limit,
        "objective": "unclipped L1 on original CH-SIMS scores in [-1,1]",
        "selection": "minimum native validation MAE; first checkpoint wins ties",
        "attention_implementation": "eager",
        "test_scored": False,
    })
    identity = {
        "config": effective_config,
        "seed": seed,
        "mode": mode,
        "implementation": _implementation_identity(),
        "runtime": runtime,
        "source": source_identity,
        "architecture": {
            "encoder": MODEL_ID,
            "revision": MODEL_REVISION,
            "mode": mode,
            "head": "CLS -> dropout(0.1) -> Linear(hidden,1)",
            "freeze": "embeddings+bottom8+pooler frozen; top4 trainable" if mode == "top4" else "all encoder parameters frozen",
            "trainable_parameters": counts,
            "total_parameters": total_parameters,
        },
        "optimizer": {
            "kind": "AdamW",
            "encoder_lr": config["encoder_lr"] if mode == "top4" else None,
            "head_lr": config["head_lr"],
            "weight_decay": config["weight_decay"],
            "max_grad_norm": config["max_grad_norm"],
            "total_steps": total_updates,
            "warmup_steps": warmup_steps,
        },
    }
    identity_sha = _sha_bytes(_canonical(identity))

    run_root = _resolve(config["run_root"])
    if smoke_limit is not None:
        run_root = run_root / ("smoke_" + str(smoke_limit))
    root = run_root / mode / f"seed_{seed}"
    last_path, best_path = root / "last.pt", root / "best.pt"
    if root.exists() and not resume:
        raise FileExistsError("Run exists; pass --resume for the identical run")
    if resume and not last_path.is_file():
        raise FileNotFoundError("No last checkpoint to resume")
    root.mkdir(parents=True, exist_ok=True)

    start_epoch, best_mae, best_epoch, stale, history = 1, float("inf"), 0, 0, []
    completed = False
    optimizer_updates = 0
    peak_cuda_allocated_bytes = None
    if resume:
        # This pickle contains RNG tuples as well as tensors and is only loaded
        # from the checkpoint written by this run directory.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("identity") != identity or state.get("identity_sha256") != identity_sha:
            raise ValueError("Config/code/model/tokenizer/CSV/runtime/seed/cohort resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _restore_rng(state["rng_state"], generator, device)
        start_epoch = int(state["epoch"]) + 1
        best_mae, best_epoch = state["best_mae"], state["best_epoch"]
        stale, history, completed = state["stale"], state["history"], state["completed"]
        optimizer_updates = int(state["optimizer_updates"])
        peak_cuda_allocated_bytes = state["peak_cuda_allocated_bytes"]
        if completed and not best_path.is_file():
            raise ValueError("Completed run lacks selected checkpoint")
    else:
        save_json(root / "config.json", effective_config)
        save_json(root / "identity.json", identity)
        save_json(root / "environment.json", runtime)

    this_call_epochs = 0
    for epoch in range(start_epoch, config["max_epochs"] + 1):
        if completed:
            break
        if stop_after is not None and this_call_epochs >= stop_after:
            break
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        total_loss, seen = 0.0, 0
        for batch in loaders["train"]:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(ids, mask)
            loss = nn.functional.l1_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite original-scale L1 loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer_updates += 1
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)

        valid_metrics, valid_predictions = _evaluate(model, loaders["valid"], device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            epoch_peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
            peak_cuda_allocated_bytes = max(
                peak_cuda_allocated_bytes or 0, epoch_peak_cuda_allocated_bytes)
        else:
            epoch_peak_cuda_allocated_bytes = None
        score = valid_metrics["MAE"]
        improved = score < best_mae  # strict comparison preserves the first tie
        if improved:
            best_mae, best_epoch, stale = score, epoch, 0
            save_checkpoint(best_path, {
                "identity": identity,
                "identity_sha256": identity_sha,
                "epoch": epoch,
                "model": model.state_dict(),
                "valid_metrics": valid_metrics,
            })
            _save_predictions(root / "valid_predictions.json", valid_predictions)
        else:
            stale += 1

        row = {
            "epoch": epoch,
            "train_L1": total_loss / seen,
            "valid": valid_metrics,
            "best_valid_MAE": best_mae,
            "best_epoch": best_epoch,
            "encoder_lr": optimizer.param_groups[0]["lr"] if mode == "top4" else None,
            "head_lr": optimizer.param_groups[-1]["lr"],
            "seconds": time.perf_counter() - started,
            "optimizer_updates": optimizer_updates,
            "peak_cuda_allocated_bytes": epoch_peak_cuda_allocated_bytes,
        }
        history.append(row)
        completed = stale >= config["patience"] or epoch == config["max_epochs"]
        save_json(root / "history.json", history)
        save_checkpoint(last_path, {
            "identity": identity,
            "identity_sha256": identity_sha,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": _rng_state(generator, device),
            "history": history,
            "best_mae": best_mae,
            "best_epoch": best_epoch,
            "stale": stale,
            "completed": completed,
            "optimizer_updates": optimizer_updates,
            "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        })
        print(json.dumps({"seed": seed, "mode": mode, **row}, allow_nan=False), flush=True)
        this_call_epochs += 1
        if completed:
            break

    if not best_path.is_file():
        raise RuntimeError("No validation checkpoint was selected")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    if best_state.get("identity_sha256") != identity_sha:
        raise ValueError("Selected checkpoint identity mismatch")
    model.load_state_dict(best_state["model"])
    reloaded_metrics, reloaded_predictions = _evaluate(model, loaders["valid"], device)
    saved_predictions = json.loads((root / "valid_predictions.json").read_text(encoding="utf-8"))
    if reloaded_predictions != saved_predictions or reloaded_metrics != best_state["valid_metrics"]:
        raise ValueError("Reloaded selected checkpoint differs from saved validation outputs")

    last_state = torch.load(last_path, map_location="cpu", weights_only=False)
    result = {
        "dataset": "simsv2",
        "model": "native_macbert_text",
        "root": str(root),
        "mode": mode,
        "seed": seed,
        "train_count": len(datasets["train"]),
        "valid_count": len(datasets["valid"]),
        "smoke_only": smoke_limit is not None,
        "complete": bool(last_state["completed"]),
        "suite_complete": bool(last_state["completed"] and smoke_limit is None),
        "last_epoch": int(last_state["epoch"]),
        "best_epoch": int(best_state["epoch"]),
        "valid_metrics": reloaded_metrics,
        "total_parameters": total_parameters,
        "trainable_parameter_count": counts["total"],
        "trainable_parameters": counts,
        "optimizer_updates": int(last_state["optimizer_updates"]),
        "peak_cuda_allocated_bytes": last_state["peak_cuda_allocated_bytes"],
        "loop_seconds": sum(row["seconds"] for row in history),
        "train_valid_seconds": sum(row["seconds"] for row in history),
        "identity_sha256": identity_sha,
        "checkpoint_sha256": digest(best_path),
        "predictions_sha256": digest(root / "valid_predictions.json"),
        "test_evaluated": False,
    }
    save_json(root / "result.json", result)
    return result
