"""Fixed native reliability factor experiment; validation only, exact resume."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from jevaffect.native_training import NativeSplit, author_rows, digest, move, save_json, save_checkpoint
from jevaffect.native_metrics import native_metrics
from jevaffect.regression_uncertainty import uncertainty_metrics

MODALITIES = ("text", "audio", "vision")
CONDITIONS = {"S0G0": (False, "uniform"), "S0G1": (False, "dynamic"),
              "S1G0": (True, "uniform"), "S1G1": (True, "dynamic")}


class ReliabilitySplit(NativeSplit):
    def __init__(self, cache, split, rows, author_csv, limit=None):
        super().__init__(cache, split, rows, MODALITIES, pooled=True, limit=limit)
        self.auxiliary = np.zeros((len(self.ids), 3), np.float32)
        self.auxiliary_valid = np.zeros((len(self.ids), 3), bool)
        # Validation auxiliary labels are not parsed or passed to the model.
        if split == "train":
            with Path(author_csv).open(encoding="utf-8-sig", newline="") as f:
                records = {r["video_id"] + "/" + r["clip_id"]: r for r in csv.DictReader(f) if r["mode"] == "train"}
            for i, sid in enumerate(self.ids):
                for j, suffix in enumerate(("T", "A", "V")):
                    raw = records[sid].get("label_" + suffix)
                    if raw not in (None, ""):
                        value = float(raw)
                        if not np.isfinite(value) or abs(value) > 1:
                            raise ValueError("Invalid native auxiliary annotation")
                        self.auxiliary[i, j] = value
                        self.auxiliary_valid[i, j] = True

    def __getitem__(self, index):
        features, target, sid = super().__getitem__(index)
        features["availability"] = torch.tensor([self.lengths[m][index] > 0 for m in MODALITIES], dtype=torch.bool)
        return {"features": features, "target": target, "id": sid,
                "auxiliary": torch.from_numpy(self.auxiliary[index].copy()),
                "auxiliary_valid": torch.from_numpy(self.auxiliary_valid[index].copy())}


def objective(outputs, target, auxiliary, auxiliary_valid, use_auxiliary, regularization, auxiliary_weight):
    from jevaffect.nig import nig_loss
    availability = outputs["availability"]
    fused = nig_loss(outputs["fused"], target, regularization=regularization, reduction="mean")
    branch_losses = nig_loss(outputs["branches"], target[:, None].expand_as(availability), regularization=regularization, reduction="none")
    branches = branch_losses[availability].mean()
    valid = auxiliary_valid & availability
    if use_auxiliary:
        if not valid.any():
            raise ValueError("Auxiliary condition needs real training annotations")
        auxiliary_loss = (outputs["unimodal_prediction"] - auxiliary).abs()[valid].mean()
    else:
        auxiliary_loss = target.new_zeros(())
    total = fused + branches + auxiliary_weight * auxiliary_loss
    return total, {"fused_NIG": float(fused.detach()), "branch_NIG": float(branches.detach()), "auxiliary_L1": float(auxiliary_loss.detach())}


def evaluate(model, loader, device, dataset, transform=None):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            features = move(batch["features"], device)
            if transform is not None:
                features = transform(features, batch["id"])
            output = model(features)
            fused = output["fused"].detach().cpu().numpy()
            weights = output["weights"].detach().cpu().numpy()
            if not np.isfinite(fused).all() or not np.isfinite(weights).all():
                raise ValueError("Nonfinite distribution prediction")
            if not torch.allclose(output["prediction"], output["fused"][:, 0]):
                raise ValueError("Prediction does not match distribution mean")
            availability = output["availability"].cpu().numpy()
            branch = output["branches"].detach().cpu().numpy()
            for i, sid in enumerate(batch["id"]):
                predictions.append({"id": sid, "target": float(batch["target"][i]), "prediction": float(fused[i, 0]),
                                    "nig": fused[i].tolist(), "weights": weights[i].tolist(),
                                    "availability": availability[i].tolist(), "branch_nig": branch[i].tolist()})
    target = np.asarray([r["target"] for r in predictions], np.float32)
    parameters = np.asarray([r["nig"] for r in predictions], np.float32)
    native = native_metrics(parameters[:, 0], target, dataset)
    distribution = uncertainty_metrics(parameters, target, [-1, 1] if dataset == "simsv2" else [-3, 3])
    return {"native": native, "uncertainty": distribution}, predictions


def train_case(config, condition, seed, resume=False, smoke=None, stop_after=None):
    from jevaffect.reliability import ReliabilityRegressor
    if condition not in CONDITIONS or config["dataset"] != "simsv2" or tuple(config["seeds"]) != (13, 42, 2026) or seed not in config["seeds"]:
        raise ValueError("Unregistered reliability condition/dataset/seed")
    use_auxiliary, fusion = CONDITIONS[condition]
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    rows = author_rows(config["author_csv"], config["dataset"])
    datasets = {s: ReliabilitySplit(config["cache_dir"], s, rows, config["author_csv"], smoke) for s in ("train", "valid")}
    if set(datasets["train"].ids) & set(datasets["valid"].ids):
        raise ValueError("Overlapping train/validation IDs")
    if use_auxiliary and not datasets["train"].auxiliary_valid.all():
        raise ValueError("S1 requires complete real auxiliary targets")
    dims = {m: datasets["train"].features[m].shape[-1] for m in MODALITIES}
    model = ReliabilityRegressor(dims, hidden=config["hidden"], dropout=config["dropout"], fusion=fusion).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(datasets[s], batch_size=config["batch_size"], shuffle=s == "train", num_workers=0,
                             generator=generator if s == "train" else None) for s in datasets}
    root = Path(config["run_root"]) / condition / ("smoke" if smoke else "formal") / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__).with_name(name) for name in ("reliability_training.py", "reliability.py", "nig.py", "native_training.py", "native_metrics.py", "regression_uncertainty.py")]
    identity = {"config": config, "condition": condition, "seed": seed, "smoke": smoke,
                "author_csv_sha256": digest(config["author_csv"]), "feature_hashes": {s: datasets[s].hashes for s in datasets},
                "implementation": {p.name: digest(p) for p in sources}, "torch": torch.__version__, "numpy": np.__version__,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "selection": "minimum native validation MAE, first tie", "test_loader_present": False,
                "auxiliary_validation_labels_used": False}
    from scipy import __version__ as scipy_version
    identity["scipy"] = scipy_version
    last = root / "last.pt"
    epoch0, best, stale, history = 0, float("inf"), 0, []
    if last.exists():
        if not resume:
            raise ValueError("Run already exists; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Reliability resume identity mismatch")
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
        started = time.perf_counter(); model.train()
        sums = {"total": 0., "fused_NIG": 0., "branch_NIG": 0., "auxiliary_L1": 0.}
        seen, updates = 0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device)
            output = model(move(batch["features"], device))
            loss, components = objective(output, target, batch["auxiliary"].to(device), batch["auxiliary_valid"].to(device),
                                         use_auxiliary, config["regularization"], config["auxiliary_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reliability loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            n = len(target); seen += n; updates += 1
            sums["total"] += loss.item() * n
            for key, value in components.items():
                sums[key] += value * n
        metrics, predictions = evaluate(model, loaders["valid"], device, config["dataset"])
        score = metrics["native"]["MAE"]; scheduler.step(score)
        row = {"epoch": epoch + 1, "train_loss": {key: value / seen for key, value in sums.items()},
               "valid": metrics, "updates": updates, "seconds": time.perf_counter() - started}
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
        save_checkpoint(last, state); save_json(root / "history.json", history)
        print(json.dumps({"condition": condition, "seed": seed, "epoch": epoch + 1, "train_loss": row["train_loss"],
                          "valid_MAE": score, "seconds": row["seconds"]}, allow_nan=False), flush=True)
    selected = torch.load(root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    metrics, predictions = evaluate(model, loaders["valid"], device, config["dataset"])
    if predictions != json.loads((root / "valid_predictions.json").read_text()):
        raise ValueError("Best checkpoint reload changed recorded predictions")
    result = {"condition": condition, "seed": seed, "smoke_only": smoke is not None,
              "complete": len(history) >= config["max_epochs"] or stale >= config["patience"],
              "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]), "best_epoch": selected["epoch"],
              "epochs_executed": len(history), "optimizer_updates": sum(r["updates"] for r in history),
              "loop_seconds": sum(r["seconds"] for r in history), "trainable_parameters": sum(p.numel() for p in model.parameters()),
              "valid_metrics": metrics, "test_evaluated": False, "root": str(root),
              "checkpoint_sha256": digest(root / "best.pt"), "predictions_sha256": digest(root / "valid_predictions.json")}
    save_json(root / "result.json", result)
    return result
