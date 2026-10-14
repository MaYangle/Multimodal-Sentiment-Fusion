"""Registered joint-context NIG comparison without auxiliary supervision."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from jevaffect.native_training import NativeSplit, author_rows, digest, move, save_json, save_checkpoint
from jevaffect.context_nig import CASES, build_context_model
from jevaffect.nig import nig_loss
from jevaffect.reliability_training import evaluate

MODALITIES = ("text", "audio", "vision")


class DistributionSplit(NativeSplit):
    def __init__(self, cache, split, rows, limit=None):
        super().__init__(cache, split, rows, MODALITIES, pooled=True, limit=limit)

    def __getitem__(self, index):
        features, target, sid = super().__getitem__(index)
        features["availability"] = torch.tensor([self.lengths[m][index] > 0 for m in MODALITIES], dtype=torch.bool)
        return {"features": features, "target": target, "id": sid}


def summed_objective(output, target, regularization):
    available = output["availability"]
    individual = nig_loss(output["branches"], target[:, None].expand_as(available), regularization, "none")
    # Sum the mean loss of every eligible expert, exactly the source weighting
    # convention. A fully unavailable expert makes no contribution.
    total_branch = target.new_zeros(())
    for i in range(available.shape[1]):
        if available[:, i].any():
            total_branch = total_branch + individual[available[:, i], i].mean()
    fused = nig_loss(output["fused"], target, regularization, "mean")
    return total_branch + fused, {"branch_sum_NIG": float(total_branch.detach()), "fused_NIG": float(fused.detach())}


def train_case(config, case, seed, resume=False, smoke=None, stop_after=None):
    if case not in CASES or config["dataset"] != "simsv2" or tuple(config["seeds"]) != (13, 42, 2026) or seed not in config["seeds"]:
        raise ValueError("Unregistered context experiment")
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    rows = author_rows(config["author_csv"], "simsv2")
    datasets = {s: DistributionSplit(config["cache_dir"], s, rows, smoke) for s in ("train", "valid")}
    if set(datasets["train"].ids) & set(datasets["valid"].ids):
        raise ValueError("Overlapping train/validation IDs")
    dims = {m: datasets["train"].features[m].shape[-1] for m in MODALITIES}
    model = build_context_model(case, dims, config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(datasets[s], batch_size=config["batch_size"], shuffle=s == "train", num_workers=0,
                             generator=generator if s == "train" else None) for s in datasets}
    root = Path(config["run_root"]) / case / ("smoke" if smoke else "formal") / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    names = ("context_training.py", "context_nig.py", "reliability.py", "reliability_training.py", "nig.py", "native_training.py", "native_metrics.py", "regression_uncertainty.py")
    from scipy import __version__ as scipy_version
    identity = {"config": config, "case": case, "seed": seed, "smoke": smoke, "author_csv_sha256": digest(config["author_csv"]),
                "feature_hashes": {s: datasets[s].hashes for s in datasets}, "implementation": {n: digest(Path(__file__).with_name(n)) for n in names},
                "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy_version, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "auxiliary_annotations_used": False, "selection": "minimum native valid MAE; first tie", "loss": "sum eligible branch mean NIG plus fused NIG"}
    epoch0, best, stale, history = 0, float("inf"), 0, []
    last = root / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Run exists; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Context run identity mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        epoch0, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"]); random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_json(root / "identity.json", identity)
    cap = min(config["max_epochs"], epoch0 + stop_after) if stop_after else config["max_epochs"]
    for epoch in range(epoch0, cap):
        if stale >= config["patience"]:
            break
        start = time.perf_counter(); model.train(); seen, updates = 0, 0
        sums = {"total": 0., "branch_sum_NIG": 0., "fused_NIG": 0.}
        nonzero_gradient_parameters = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = batch["target"].to(device)
            output = model(move(batch["features"], device))
            loss, parts = summed_objective(output, target, config["regularization"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite summed NIG loss")
            loss.backward()
            if updates == 0:
                nonzero_gradient_parameters = sum(p.numel() for p in model.parameters() if p.grad is not None and bool(torch.any(p.grad != 0)))
            nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            n = len(target); seen += n; updates += 1
            sums["total"] += loss.item() * n
            for key in parts:
                sums[key] += parts[key] * n
        metrics, predictions = evaluate(model, loaders["valid"], device, "simsv2")
        score = metrics["native"]["MAE"]; scheduler.step(score)
        row = {"epoch": epoch + 1, "train_loss": {k: v / seen for k, v in sums.items()}, "valid": metrics,
               "updates": updates, "first_batch_nonzero_gradient_parameters": nonzero_gradient_parameters,
               "seconds": time.perf_counter() - start}
        history.append(row)
        if score < best:
            best, stale = score, 0
            save_checkpoint(root / "best.pt", {"identity": identity, "epoch": epoch + 1, "model": model.state_dict(), "metrics": metrics})
            save_json(root / "valid_predictions.json", predictions)
        else:
            stale += 1
        save_checkpoint(last, {"identity": identity, "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        save_json(root / "history.json", history)
        print(json.dumps({"case": case, "seed": seed, "epoch": epoch + 1, "valid_MAE": score, "loss": row["train_loss"], "seconds": row["seconds"]}), flush=True)
    selected = torch.load(root / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(selected["model"])
    metrics, predictions = evaluate(model, loaders["valid"], device, "simsv2")
    if predictions != json.loads((root / "valid_predictions.json").read_text()):
        raise ValueError("Context checkpoint reload changed predictions")
    result = {"case": case, "seed": seed, "root": str(root), "smoke_only": smoke is not None,
              "complete": len(history) >= config["max_epochs"] or stale >= config["patience"], "test_evaluated": False,
              "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]), "best_epoch": selected["epoch"],
              "epochs_executed": len(history), "optimizer_updates": sum(r["updates"] for r in history),
              "loop_seconds": sum(r["seconds"] for r in history), "parameters_total": sum(p.numel() for p in model.parameters()),
              "parameters_requires_grad": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "first_batch_nonzero_gradient_parameters": history[0]["first_batch_nonzero_gradient_parameters"],
              "valid_metrics": metrics, "checkpoint_sha256": digest(root / "best.pt"), "predictions_sha256": digest(root / "valid_predictions.json")}
    save_json(root / "result.json", result)
    return result
