"""Four-case native CH-SIMS annotation-reference study, exact epoch resume."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import csv
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from jevaffect.native_training import NativeSplit, author_rows, digest, save_json, save_checkpoint, move
from jevaffect.native_metrics import native_metrics
from jevaffect.reference_residual import REFERENCE_CASES, build_reference_model


class ReferenceSplit(NativeSplit):
    def __init__(self, cache, split, rows, csv_path, limit=None):
        super().__init__(cache, split, rows, ("text", "audio", "vision"), pooled=True, limit=limit)
        self.reference = np.zeros(len(self.ids), np.float32); self.reference_valid = np.zeros(len(self.ids), bool)
        if split == "train":
            with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
                records = {r["video_id"] + "/" + r["clip_id"]: r for r in csv.DictReader(stream) if r["mode"] == "train"}
            for i, sid in enumerate(self.ids):
                raw = records[sid].get("label_T")
                if raw not in (None, ""):
                    value = float(raw)
                    if not np.isfinite(value) or abs(value) > 1:
                        raise ValueError("Invalid native reference annotation")
                    self.reference[i], self.reference_valid[i] = value, True

    def __getitem__(self, index):
        features, target, sid = super().__getitem__(index)
        return {"features": features, "target": target, "id": sid,
                "reference": torch.tensor(self.reference[index]), "reference_valid": torch.tensor(self.reference_valid[index])}


def reference_objective(output, target, reference, valid, supervised, weight):
    main = (output["prediction"] - target).abs().mean()
    t_loss, d_loss = target.new_zeros(()), target.new_zeros(())
    if supervised:
        if not valid.any():
            raise ValueError("Supervised case requires real TRAIN text annotations")
        t_loss = (output["text_reference"][valid] - reference[valid]).abs().mean()
        d_loss = (output["annotation_correction"][valid] - (target[valid] - reference[valid])).abs().mean()
    return main + weight * (t_loss + d_loss), {"main_L1": float(main.detach()), "reference_L1": float(t_loss.detach()), "annotation_difference_L1": float(d_loss.detach())}


def evaluate_reference(model, loader, device):
    model.eval(); records = []
    with torch.no_grad():
        for batch in loader:
            output = model(move(batch["features"], device))
            values = {k: v.cpu().tolist() for k, v in output.items()}
            for i, sid in enumerate(batch["id"]):
                record = {"id": sid, "target": float(batch["target"][i]), **{k: v[i] for k, v in values.items()}}
                records.append(record)
    prediction = np.array([r["prediction"] for r in records], np.float32); target = np.array([r["target"] for r in records], np.float32)
    return native_metrics(prediction, target, "simsv2"), records


def train_reference(config, case, seed, resume=False, smoke=None, stop_after=None):
    if config["dataset"] != "simsv2" or case not in REFERENCE_CASES or tuple(config["seeds"]) != (13, 42, 2026) or seed not in config["seeds"]:
        raise ValueError("Unregistered annotation-reference experiment")
    torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); device = torch.device(config["device"])
    rows = author_rows(config["author_csv"], "simsv2")
    datasets = {s: ReferenceSplit(config["cache_dir"], s, rows, config["author_csv"], smoke) for s in ("train", "valid")}
    dims = {m: datasets["train"].features[m].shape[-1] for m in ("text", "audio", "vision")}
    model = build_reference_model(case, dims, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds, batch_size=config["batch_size"], shuffle=s == "train", num_workers=0,
                            generator=generator if s == "train" else None) for s, ds in datasets.items()}
    root = Path(config["run_root"]) / case / ("smoke" if smoke is not None else "formal") / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    names = ("reference_training.py", "reference_residual.py", "native_training.py", "native_metrics.py")
    identity = {"config": config, "case": case, "seed": seed, "smoke_limit": smoke,
                "source_csv_sha256": digest(config["author_csv"]), "features": {s: ds.hashes for s, ds in datasets.items()},
                "implementation": {n: digest(Path(__file__).with_name(n)) for n in names}, "torch": torch.__version__, "numpy": np.__version__,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "selection": "minimum native VALID MAE, first tie", "validation_reference_labels_parsed": False,
                "reference_supervised": case.endswith("1"), "target_semantics": "real TRAIN text label and M-T annotation difference; not causal AV utility"}
    epoch0, best, stale, history = 0, float("inf"), 0, []
    last = root / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Run exists; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Reference run identity mismatch")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        epoch0, best, stale, history = state["epoch"], state["best"], state["stale"], state["history"]
        torch.set_rng_state(state["torch_rng"]); random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"]); generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_json(root / "identity.json", identity)
    cap = min(config["max_epochs"], epoch0 + stop_after) if stop_after else config["max_epochs"]
    for epoch in range(epoch0, cap):
        if stale >= config["patience"]:
            break
        start = time.perf_counter(); model.train(); total, seen, updates = 0., 0, 0
        nonzero_parameters = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True); target = batch["target"].to(device)
            output = model(move(batch["features"], device))
            loss, _ = reference_objective(output, target, batch["reference"].to(device), batch["reference_valid"].to(device), case.endswith("1"), config["auxiliary_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite reference loss")
            loss.backward()
            if updates == 0:
                nonzero_parameters = sum(p.numel() for p in model.parameters() if p.grad is not None and bool(torch.any(p.grad != 0)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            total += loss.item() * len(target); seen += len(target); updates += 1
        metrics, predictions = evaluate_reference(model, loaders["valid"], device); score = metrics["MAE"]; scheduler.step(score)
        history.append({"epoch": epoch + 1, "train_loss": total / seen, "valid": metrics, "updates": updates,
                        "first_batch_nonzero_gradient_parameters": nonzero_parameters, "seconds": time.perf_counter() - start})
        if score < best:
            best, stale = score, 0
            save_checkpoint(root / "best.pt", {"identity": identity, "epoch": epoch + 1, "model": model.state_dict(), "metrics": metrics})
            save_json(root / "valid_predictions.json", predictions)
        else:
            stale += 1
        save_checkpoint(last, {"identity": identity, "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        save_json(root / "history.json", history)
        print(json.dumps({"case": case, "seed": seed, "epoch": epoch + 1, "valid_MAE": score, "seconds": history[-1]["seconds"]}), flush=True)
    selected = torch.load(root / "best.pt", map_location=device, weights_only=False); model.load_state_dict(selected["model"])
    metrics, predictions = evaluate_reference(model, loaders["valid"], device)
    if predictions != json.loads((root / "valid_predictions.json").read_text()):
        raise ValueError("Reference checkpoint reload changed outputs")
    result = {"case": case, "seed": seed, "root": str(root), "smoke_only": smoke is not None,
              "complete": stale >= config["patience"] or len(history) >= config["max_epochs"], "test_evaluated": False,
              "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]), "parameters": sum(p.numel() for p in model.parameters()),
              "first_batch_nonzero_gradient_parameters": history[0]["first_batch_nonzero_gradient_parameters"],
              "best_epoch": selected["epoch"], "epochs_executed": len(history), "optimizer_updates": sum(r["updates"] for r in history),
              "loop_seconds": sum(r["seconds"] for r in history), "valid_metrics": metrics,
              "checkpoint_sha256": digest(root / "best.pt"), "predictions_sha256": digest(root / "valid_predictions.json")}
    save_json(root / "result.json", result); return result
