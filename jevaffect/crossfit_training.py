"""Train source-disjoint joint-NIG teachers using official TRAIN labels only."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from jevaffect.context_nig import build_context_model
from jevaffect.context_training import DistributionSplit, summed_objective, MODALITIES
from jevaffect.native_training import digest, move, save_json, save_checkpoint
from jevaffect.native_metrics import native_metrics
from jevaffect.source_crossfit import training_rows_only, make_folds, subset, manifest_hash, validate_folds


def predict(model, loader, device):
    records = []; model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model(move(batch["features"], device))
            tensors = {key: out[key].cpu().tolist() for key in ("fused", "branches", "availability")}
            for i, sid in enumerate(batch["id"]):
                records.append({"id": sid, "target": float(batch["target"][i]), "prediction": tensors["fused"][i][0],
                                "fused_nig": tensors["fused"][i], "branch_nig": tensors["branches"][i],
                                "availability": tensors["availability"][i]})
    metric = native_metrics(np.array([r["prediction"] for r in records], np.float32),
                            np.array([r["target"] for r in records], np.float32), "simsv2")
    return metric, records


def prepare(config, smoke=None):
    rows = training_rows_only(config["author_csv"])
    dataset = DistributionSplit(config["cache_dir"], "train", rows)
    # Smoke takes complete source groups, avoiding partial-group leakage.
    if smoke is not None:
        sources = list(dict.fromkeys(s.split("/")[0] for s in dataset.ids))[:smoke]
        wanted = [s for s in dataset.ids if s.split("/")[0] in sources]
    else:
        wanted = dataset.ids
    manifest = make_folds(wanted, config["fold_count"], config["fold_seed"], config["inner_fraction"])
    return dataset, manifest


def train_fold(config, dataset, manifest, seed, fold, resume=False, smoke=None, stop_after=None):
    validate_folds([s for f in manifest["folds"] for s in f["oof_ids"]], manifest["folds"])
    if seed not in (13, 42, 2026) or config["dataset"] != "simsv2":
        raise ValueError("Unregistered dataset/seed")
    torch.set_num_threads(4); rng_seed = seed + 10000 * fold
    random.seed(rng_seed); np.random.seed(rng_seed); torch.manual_seed(rng_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rng_seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config["device"])
    dims = {m: dataset.features[m].shape[-1] for m in MODALITIES}
    model = build_context_model("joint_dynamic", dims, config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(rng_seed)
    cohort = manifest["folds"][fold]
    loaders = {role: DataLoader(subset(dataset, cohort[role + "_ids"]), batch_size=config["batch_size"],
                   shuffle=role == "fit", generator=generator if role == "fit" else None, num_workers=0) for role in ("fit", "select", "oof")}
    root = Path(config["run_root"]) / ("smoke" if smoke is not None else "formal") / f"seed_{seed}" / f"fold_{fold}"
    root.mkdir(parents=True, exist_ok=True)
    names = ("crossfit_training.py", "source_crossfit.py", "context_training.py", "context_nig.py", "nig.py", "reliability.py", "native_training.py", "native_metrics.py")
    identity = {"config": config, "seed": seed, "rng_seed": rng_seed, "fold": fold, "smoke_groups": smoke,
                "manifest_sha256": manifest_hash(manifest), "cohort": cohort, "feature_hashes": dataset.hashes,
                "author_csv_sha256": digest(config["author_csv"]), "torch": torch.__version__, "numpy": np.__version__,
                "implementation": {n: digest(Path(__file__).with_name(n)) for n in names}, "device": str(device),
                "official_validation_labels_parsed": False, "test_evaluated": False}
    epoch0, best, stale, history = 0, float("inf"), 0, []
    last = root / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Existing fold; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Cross-fit identity mismatch")
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
        for batch in loaders["fit"]:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = summed_objective(model(move(batch["features"], device)), batch["target"].to(device), config["regularization"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite OOF loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            n = len(batch["target"]); total += loss.item() * n; seen += n; updates += 1
        metrics, _ = predict(model, loaders["select"], device)
        score = metrics["MAE"]; scheduler.step(score)
        history.append({"epoch": epoch + 1, "train_loss": total / seen, "inner_select": metrics, "updates": updates, "seconds": time.perf_counter() - start})
        if score < best:
            best, stale = score, 0
            save_checkpoint(root / "best.pt", {"identity": identity, "epoch": epoch + 1, "model": model.state_dict(), "metrics": metrics})
        else:
            stale += 1
        save_checkpoint(last, {"identity": identity, "epoch": epoch + 1, "best": best, "stale": stale, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
        save_json(root / "history.json", history)
        print(json.dumps({"seed": seed, "fold": fold, "epoch": epoch + 1, "inner_MAE": score, "seconds": history[-1]["seconds"]}), flush=True)
    selected = torch.load(root / "best.pt", map_location=device, weights_only=False); model.load_state_dict(selected["model"])
    metrics, records = predict(model, loaders["oof"], device)
    prediction_path = root / "oof_predictions.json"
    if prediction_path.exists() and epoch0 == len(history) and json.loads(prediction_path.read_text()) != records:
        raise ValueError("OOF checkpoint reload changed predictions")
    save_json(prediction_path, records)
    result = {"root": str(root), "seed": seed, "fold": fold, "smoke_only": smoke is not None,
              "complete": stale >= config["patience"] or len(history) >= config["max_epochs"], "test_evaluated": False,
              "manifest_sha256": manifest_hash(manifest), "counts": {k: len(v.dataset) for k, v in loaders.items()},
              "best_epoch": selected["epoch"], "epochs_executed": len(history), "optimizer_updates": sum(r["updates"] for r in history),
              "loop_seconds": sum(r["seconds"] for r in history), "oof_native": metrics,
              "checkpoint_sha256": digest(root / "best.pt"), "predictions_sha256": digest(prediction_path)}
    save_json(root / "result.json", result)
    return result
