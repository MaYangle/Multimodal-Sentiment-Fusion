"""Author-label regression baselines, separate from historical five-class runs."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from jevaffect.native_metrics import native_metrics


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save_checkpoint(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def author_rows(path, dataset):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    separator = "/" if dataset == "simsv2" else "_"
    result = {}
    for r in rows:
        sid = r["video_id"] + separator + r["clip_id"]
        if sid in result or r["mode"] not in ("train", "valid", "test"):
            raise ValueError("Duplicate ID or invalid author split")
        # Test labels are not parsed by the training pipeline.
        if r["mode"] != "test":
            score = float(r["label"])
            if not np.isfinite(score) or abs(score) > (1 if dataset == "simsv2" else 3):
                raise ValueError("Invalid continuous author score")
            result[sid] = {"split": r["mode"], "score": score}
        else:
            result[sid] = {"split": "test"}
    return result


class NativeSplit(Dataset):
    def __init__(self, cache, split, rows, modalities, pooled=True, limit=None):
        if split not in ("train", "valid"):
            raise ValueError("Training can only open train and valid")
        cache = Path(cache)
        metadata_path = cache / "metadata.json"
        if not metadata_path.exists():
            metadata_path = cache / "meta.json"
        self.meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        directory = cache / self.meta.get("split_paths", {}).get(split, "splits/" + split if (cache / "splits").exists() else split)
        directory = directory.resolve()
        directory.relative_to(cache.resolve())
        self.ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        expected = {sid for sid, r in rows.items() if r["split"] == split}
        if len(set(self.ids)) != len(self.ids) or set(self.ids) != expected:
            raise ValueError("Cache IDs differ from complete author split; no implicit cohort joining")
        self.targets = np.asarray([rows[sid]["score"] for sid in self.ids], dtype=np.float32)
        self.hashes = {str(metadata_path): digest(metadata_path), str(directory / "ids.json"): digest(directory / "ids.json")}
        raw_path = directory / "raw_labels.json"
        if raw_path.exists():
            raw = np.asarray(json.loads(raw_path.read_text()), dtype=np.float32)
            if not np.allclose(raw.reshape(-1), self.targets, atol=1e-6, rtol=0):
                raise ValueError("Cache raw scores disagree with author CSV")
        self.features, self.lengths = {}, {}
        recorded = self.meta.get("artifact_sha256", {})
        file_records = self.meta.get("files", {})
        for m in modalities:
            fp, lp = directory / (m + ".npy"), directory / (m + "_lengths.npy")
            for path in (fp, lp):
                actual = digest(path)
                key = path.relative_to(cache.resolve()).as_posix()
                known = recorded.get(key, file_records.get(key, {}).get("sha256"))
                if known is None or known != actual:
                    raise ValueError("Missing or mismatched cached feature hash: " + key)
                self.hashes[key] = actual
            x, lengths = np.load(fp, mmap_mode="r"), np.load(lp, mmap_mode="r")
            if x.ndim != 3 or x.shape[0] != len(self.ids) or lengths.shape != (len(self.ids),):
                raise ValueError("Invalid sequence shapes")
            if lengths.dtype.kind not in "iu" or np.any(lengths < 0) or np.any(lengths > x.shape[1]) or not np.isfinite(x).all():
                raise ValueError("Invalid lengths or nonfinite features")
            self.lengths[m] = lengths
            if pooled:
                # Text CLS; masked means for acoustic and visual sequences.
                if m == "text":
                    values = np.asarray(x[:, 0]).copy()
                    values[lengths == 0] = 0
                else:
                    values = np.stack([np.asarray(x[i, :int(n)]).mean(0) if n else np.zeros(x.shape[-1], np.float32) for i, n in enumerate(lengths)])
                self.features[m] = values.astype(np.float32)
            else:
                self.features[m] = x
        if limit is not None:
            if limit < 1:
                raise ValueError("Smoke limit must be positive")
            self.ids = self.ids[:limit]
            self.targets = self.targets[:limit]
        self.pooled = pooled

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        features = {m: torch.from_numpy(np.array(x[i], copy=True)) for m, x in self.features.items()}
        if not self.pooled:
            features["lengths"] = {m: torch.tensor(int(x[i])) for m, x in self.lengths.items()}
        return features, torch.tensor(self.targets[i]), self.ids[i]


class PooledRegressor(nn.Module):
    def __init__(self, dims, hidden=256, dropout=.1):
        super().__init__()
        self.modalities = tuple(dims)
        self.head = nn.Sequential(nn.Linear(sum(dims.values()), hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, features):
        return self.head(torch.cat([features[m] for m in self.modalities], -1)).flatten()


def move(features, device):
    return {key: move(value, device) if isinstance(value, dict) else value.to(device) for key, value in features.items()}


def evaluate(model, loader, device, dataset):
    model.eval()
    predictions, targets, ids = [], [], []
    with torch.no_grad():
        for x, y, sid in loader:
            p = model(move(x, device)).flatten().cpu().numpy()
            if p.shape != y.numpy().shape or not np.isfinite(p).all():
                raise ValueError("Nonfinite or wrongly shaped regression output")
            predictions.extend(p.tolist()); targets.extend(y.tolist()); ids.extend(sid)
    p, y = np.asarray(predictions, np.float32), np.asarray(targets, np.float32)
    return native_metrics(p, y, dataset), [{"id": sid, "target": float(t), "prediction": float(v)} for sid, t, v in zip(ids, y, p)]


def train(config, seed, resume=False, smoke=None, stop_after=None):
    if config["dataset"] not in ("simsv2", "mosei") or config["model"] not in ("text", "early_fusion", "almt"):
        raise ValueError("Unknown dataset/model")
    if tuple(config["seeds"]) != (13, 42, 2026) or seed not in config["seeds"]:
        raise ValueError("Formal seed contract is 13, 42, 2026")
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    rows = author_rows(config["author_csv"], config["dataset"])
    mods = ("text",) if config["model"] == "text" else ("text", "audio", "vision")
    pooled = config["model"] != "almt"
    ds = {s: NativeSplit(config["cache_dir"], s, rows, mods, pooled, smoke) for s in ("train", "valid")}
    if set(ds["train"].ids) & set(ds["valid"].ids):
        raise ValueError("Train/validation ID overlap")
    dims = {m: ds["train"].features[m].shape[-1] for m in mods}
    if pooled:
        model = PooledRegressor(dims, config.get("hidden", 256), config.get("dropout", .1))
    else:
        from jevaffect.almt import ALMTClassifier
        lengths = {m: ds["train"].features[m].shape[1] for m in mods}
        # Restore the author's original scalar regression head without modifying
        # the historical classifier API or its checkpoint identity.
        model = ALMTClassifier(feature_dims=dims, lengths=lengths, dim=128, token_len=8)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds[s], batch_size=config["batch_size"], shuffle=s == "train", generator=generator if s == "train" else None, num_workers=0) for s in ds}
    root = Path(config["run_root"]) / ("smoke_" if smoke else "") / config["model"] / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    implementation = {str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in [Path(__file__), Path(__file__).with_name("native_metrics.py"), Path(__file__).with_name("almt.py")]}
    if not pooled:
        for p in (Path(__file__).parents[1] / "vendor/almt").glob("*.py"):
            implementation[str(p.relative_to(Path(__file__).parents[1]))] = digest(p)
    identity = {"config": config, "seed": seed, "smoke_limit": smoke,
                "author_csv_sha256": digest(config["author_csv"]), "features": {s: ds[s].hashes for s in ds},
                "implementation": implementation, "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "selection": "minimum validation native MAE; first checkpoint wins ties", "loss": "unclipped L1 on original scores"}
    path = root / "last.pt"
    epoch0, best, stale, history = 0, float("inf"), 0, []
    if path.exists():
        if not resume:
            raise ValueError("Run exists; pass --resume")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Checkpoint identity mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        epoch0, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_json(root / "identity.json", identity)
    cap = min(config["max_epochs"], epoch0 + stop_after) if stop_after else config["max_epochs"]
    for epoch in range(epoch0, cap):
        if stale >= config["patience"]:
            break
        start = time.perf_counter()
        model.train(); loss_total = 0.; seen = 0
        for x, y, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            y = y.to(device); p = model(move(x, device)).flatten()
            loss = nn.functional.l1_loss(p, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            loss_total += loss.item() * len(y); seen += len(y)
        metrics, predictions = evaluate(model, loaders["valid"], device, config["dataset"])
        score = metrics["MAE"]
        scheduler.step(score)
        row = {"epoch": epoch + 1, "train_L1": loss_total / seen, "valid": metrics, "seconds": time.perf_counter() - start}
        history.append(row)
        if score < best:
            best, stale = score, 0
            save_checkpoint(root / "best.pt", {"identity": identity, "epoch": epoch + 1, "model": model.state_dict(), "metrics": metrics})
            save_json(root / "valid_predictions.json", predictions)
        else:
            stale += 1
        state = {"identity": identity, "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                 "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(path, state); save_json(root / "history.json", history)
        print(json.dumps({"seed": seed, **row}, allow_nan=False), flush=True)
    checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics, predictions = evaluate(model, loaders["valid"], device, config["dataset"])
    saved = json.loads((root / "valid_predictions.json").read_text())
    if predictions != saved:
        raise ValueError("Loaded best checkpoint predictions differ from saved selection predictions")
    complete = len(history) >= config["max_epochs"] or stale >= config["patience"]
    result = {"dataset": config["dataset"], "model": config["model"], "seed": seed,
              "train_count": len(ds["train"]), "valid_count": len(ds["valid"]), "smoke_only": smoke is not None,
              "complete": complete, "best_epoch": checkpoint["epoch"], "valid_metrics": metrics,
              "loop_seconds": sum(r["seconds"] for r in history), "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "checkpoint_sha256": digest(root / "best.pt"), "predictions_sha256": digest(root / "valid_predictions.json"),
              "test_evaluated": False}
    save_json(root / "result.json", result)
    return result
