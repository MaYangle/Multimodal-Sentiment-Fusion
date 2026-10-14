"""Frozen-budget MOSEI ALMT padding baseline correction."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from jevaffect.author_features import AuthorFeatureSplit, COHORT, MODALITIES
from jevaffect.native_training import PooledRegressor, evaluate, move, digest, save_json, save_checkpoint


def train_masked_author(config, seed, resume=False, smoke=None, stop_after=None):
    if config["dataset"] != "mosei" or config["cohort"] != COHORT or config["model"] != "almt_masked" or tuple(config["seeds"]) != (13, 42, 2026) or seed not in config["seeds"]:
        raise ValueError("Unregistered author-feature experiment")
    datasets = {}
    try:
        mods = ("text",) if config["model"] == "text" else MODALITIES
        for split in ("train", "valid"):
            datasets[split] = AuthorFeatureSplit(config["cache_dir"], split, mods, config["model"] != "almt_masked", smoke)
        return _train(config, seed, datasets, resume, smoke, stop_after)
    finally:
        for dataset in datasets.values():
            dataset.close()


def _train(config, seed, datasets, resume, smoke, stop_after):
    torch.set_num_threads(4); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True); device = torch.device(config["device"])
    if any(ds.meta["source_sha256"] != config["expected_source_sha256"] for ds in datasets.values()):
        raise ValueError("Prepared source version differs from experiment")
    if {sid.split("|")[0] for sid in datasets["train"].ids} & {sid.split("|")[0] for sid in datasets["valid"].ids}:
        raise ValueError("Overlapping source videos")
    dims = {m: datasets["train"].features[m].shape[-1] for m in datasets["train"].features}
    if config["model"] != "almt_masked":
        model = PooledRegressor(dims, config["hidden"], config["dropout"])
    else:
        from jevaffect.masked_almt import MaskedALMTRegressor
        model = MaskedALMTRegressor(feature_dims=dims, lengths={m: datasets["train"].features[m].shape[1] for m in dims}, dim=128, token_len=8)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=.5)
    generator = torch.Generator().manual_seed(seed)
    loaders = {s: DataLoader(ds, batch_size=config["batch_size"], shuffle=s == "train", num_workers=0,
                            generator=generator if s == "train" else None) for s, ds in datasets.items()}
    root = Path(config["run_root"]) / ("smoke" if smoke is not None else "formal") / config["model"] / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).parents[1]
    files = [Path(__file__), Path(__file__).with_name("author_features.py"), Path(__file__).with_name("native_training.py"), Path(__file__).with_name("native_metrics.py")]
    if config["model"] == "almt_masked":
        files += [Path(__file__).with_name("masked_almt.py"), Path(__file__).with_name("almt.py"), *sorted((project / "vendor/almt").glob("*.py"))]
    identity = {"config": config, "seed": seed, "smoke_limit": smoke, "features": {s: ds.hashes for s, ds in datasets.items()},
                "source_sha256": datasets["train"].meta["source_sha256"], "implementation": {p.relative_to(project).as_posix(): digest(p) for p in files},
                "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "selection": "minimum native VALID MAE; first tie", "loss": "unclipped L1 on original [-3,3] scores",
                "input_form": datasets["train"].meta["input_form"], "text_pooling": "valid_mean",
                "padding_contract": "valid-key mask and cumulative valid-rank positions in initial projections"}
    epoch0, best, stale, history = 0, float("inf"), 0, []
    last = root / "last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Run exists; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Author experiment identity mismatch")
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
        for x, y, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True); y = y.to(device); p = model(move(x, device)).flatten()
            loss = torch.nn.functional.l1_loss(p, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite author-feature training loss")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            total += loss.item() * len(y); seen += len(y); updates += 1
        metrics, predictions = evaluate(model, loaders["valid"], device, "mosei"); score = metrics["MAE"]; scheduler.step(score)
        history.append({"epoch": epoch + 1, "train_L1": total / seen, "valid": metrics, "updates": updates, "seconds": time.perf_counter() - start})
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
        print(json.dumps({"model": config["model"], "seed": seed, "epoch": epoch + 1, "valid_MAE": score, "seconds": history[-1]["seconds"]}), flush=True)
    selected = torch.load(root / "best.pt", map_location=device, weights_only=False); model.load_state_dict(selected["model"])
    metrics, predictions = evaluate(model, loaders["valid"], device, "mosei")
    if predictions != json.loads((root / "valid_predictions.json").read_text()):
        raise ValueError("Native author checkpoint reload changed predictions")
    result = {"dataset": "mosei", "cohort": COHORT, "model": config["model"], "seed": seed, "root": str(root),
              "smoke_only": smoke is not None, "complete": stale >= config["patience"] or len(history) >= config["max_epochs"],
              "test_evaluated": False, "train_count": len(datasets["train"]), "valid_count": len(datasets["valid"]),
              "best_epoch": selected["epoch"], "epochs_executed": len(history), "optimizer_updates": sum(r["updates"] for r in history),
              "loop_seconds": sum(r["seconds"] for r in history), "parameters": sum(p.numel() for p in model.parameters()),
              "valid_metrics": metrics, "checkpoint_sha256": digest(root / "best.pt"), "predictions_sha256": digest(root / "valid_predictions.json")}
    save_json(root / "result.json", result); return result
