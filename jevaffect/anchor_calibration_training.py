"""TRAIN-OOF calibration fitting; official VALID evaluated only after selection."""
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from jevaffect.anchor_calibration import CASES, BOUNDARIES, CalibrationHead, objective, disagreement, normalizer, normalize, diagnostics, describe
from jevaffect.anchor_cdf import parameters, bin_probabilities, error_band_probability
from jevaffect.native_training import digest, save_json, save_checkpoint


def load_table(directory, seed, split):
    root = Path(directory); metadata = json.loads((root / "metadata.json").read_text())
    key = f"seed_{seed}/{split}.npz"; path = root / key
    if split not in ("train", "valid") or metadata["dataset"] != "simsv2" or seed not in metadata["seeds"] or digest(path) != metadata["artifact_sha256"][key]:
        raise ValueError("Changed native calibration table or split")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in ("features", "anchor", "target", "views")}
    n = len(arrays["anchor"])
    if arrays["features"].ndim != 2 or arrays["features"].shape[0] != n or arrays["views"].shape != (n, 3) or any(arrays[k].shape != (n,) for k in ("anchor", "target")) or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in arrays.values()) or np.any(np.abs(arrays["anchor"]) > 1) or np.any(np.abs(arrays["target"]) > 1):
        raise ValueError("Invalid native calibration arrays")
    ids_path = root / f"seed_{seed}/{split}_ids.json"
    if digest(ids_path) != metadata["artifact_sha256"][ids_path.relative_to(root).as_posix()]:
        raise ValueError("Calibration ID table changed")
    arrays["ids"] = json.loads(ids_path.read_text())
    if len(arrays["ids"]) != n or len(set(arrays["ids"])) != n:
        raise ValueError("Calibration ID count/uniqueness differs")
    return arrays, metadata


def prepare_training(data, metadata):
    fit = np.array(metadata["calibration_split"]["fit_indices"], np.int64)
    select = np.array(metadata["calibration_split"]["select_indices"], np.int64)
    if not len(fit) or not len(select) or set(fit) | set(select) != set(range(len(data["ids"]))) or set(fit) & set(select) or len(fit)+len(select) != len(data["ids"]):
        raise ValueError("Invalid inner calibration partition")
    sources = [{data["ids"][i].split("/")[0] for i in indices} for indices in (fit, select)]
    if sources[0] & sources[1]:
        raise ValueError("Inner calibration head source overlap")
    x = np.concatenate((data["features"], data["anchor"][:, None]), 1)
    d = disagreement(data["views"])
    v = np.concatenate((data["views"], np.abs(data["views"])), 1)
    stats = {name: normalizer(values, fit) for name, values in (("context", x), ("differences", d), ("views", v))}
    return (fit, select), stats


def _limit(data, count):
    return {key: value[:count] for key, value in data.items()}


def tensors(data, stats, case):
    x = np.concatenate((data["features"], data["anchor"][:, None]), 1)
    if case == "views":
        aux = np.concatenate((data["views"], np.abs(data["views"])), 1); key = "views"
    else:
        aux = disagreement(data["views"]); key = "differences"
    return TensorDataset(torch.from_numpy(normalize(x, stats["context"])), torch.from_numpy(normalize(aux, stats[key])),
                         torch.from_numpy(data["anchor"]), torch.from_numpy(data["target"]))


def _selection(model, loader, case, device):
    total, n = 0., 0; model.eval()
    with torch.no_grad():
        for x, d, m, y in loader:
            x, d, m, y = (v.to(device) for v in (x, d, m, y))
            loss = objective(model(x, d, m, case), y, m, case)
            total += float(loss) * len(y); n += len(y)
    return total / n


def evaluate(model, data, stats, case, batch_size=64, device="cpu"):
    loader = DataLoader(tensors(data, stats, case), batch_size=batch_size, shuffle=False)
    described = []; model.eval()
    with torch.no_grad():
        for x, d, m, _ in loader:
            output = model(x.to(device), d.to(device), m.to(device), case)
            described.append(describe(output))
    values = {key: np.concatenate([r[key] for r in described]) for key in described[0] if key != "intervals"}
    if "intervals" in described[0]:
        values["intervals"] = {level: tuple(np.concatenate([r["intervals"][level][j] for r in described]) for j in (0, 1)) for level in described[0]["intervals"]}
    if case != "mdn" and "median" in values and np.max(np.abs(values["median"] - data["anchor"])) > 1e-6:
        raise ValueError("AnchorCDF forecast median moved")
    return diagnostics(data["anchor"], data["target"], values), _records(data, values)


def _records(data, values):
    records = []
    for i, sid in enumerate(data["ids"]):
        row = {"id": sid, "target": float(data["target"][i]), "prediction": float(data["anchor"][i]), "p_hit": float(values["p_hit"][i])}
        for key in ("median", "scale", "mixture_weight"):
            if key in values:
                row[key] = float(values[key][i])
        if "bins" in values:
            row["bin_probabilities"] = values["bins"][i].tolist()
            row["intervals"] = {str(level): [float(pair[0][i]), float(pair[1][i])] for level, pair in values["intervals"].items()}
        records.append(row)
    return records


def fit(config, case, seed, resume=False, stop_after=None, smoke=False):
    if config["dataset"] != "simsv2" or case not in (*CASES, "global") or tuple(config["seeds"]) != (13,42,2026) or seed not in config["seeds"]:
        raise ValueError("Unregistered native calibration case/seed")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data, metadata = load_table(config["table_dir"], seed, "train")
    if not smoke and len(data["ids"]) != 2722:
        raise ValueError("Formal native calibration requires all2722 TRAIN rows")
    cohorts, stats = prepare_training(data, metadata)
    if smoke:
        cohorts = tuple(indices[:32] for indices in cohorts)
        x = np.concatenate((data["features"], data["anchor"][:,None]), 1)
        stats = {name: normalizer(values, cohorts[0]) for name, values in (("context", x),
            ("differences", disagreement(data["views"])), ("views", np.concatenate((data["views"], np.abs(data["views"])),1)))}
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    root = Path(config["run_root"]) / ("smoke" if smoke else "formal") / case / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    names = ("anchor_calibration_training.py", "anchor_calibration.py", "anchor_cdf.py", "native_metrics.py", "native_training.py")
    identity = {"config": config, "case": case, "seed": seed, "smoke_only": smoke,
        "metadata_sha256": digest(Path(config["table_dir"]) / "metadata.json"), "normalization": stats,
        "implementation": {name: digest(Path(__file__).with_name(name)) for name in names},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "device": str(device)},
        "selection": "TRAIN OOF source-inner Bernoulli score" if case == "direct_hit" else "TRAIN OOF source-inner native-bin score",
        "scope": "head-inner selection is not nested full-pipeline source transfer", "test_evaluated": False}
    if case == "global":
        return _global(config, data, metadata, cohorts, stats, root, identity, smoke, resume)
    model = CalibrationHead(data["features"].shape[1] + 1, config["hidden"], config["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    ds = tensors(data, stats, case)
    loaders = [DataLoader(torch.utils.data.Subset(ds, indices.tolist()), batch_size=config["batch_size"], shuffle=i==0,
                generator=generator if i==0 else None, num_workers=0) for i, indices in enumerate(cohorts)]
    last = root / "last.pt"; epoch0, best, stale, history, gradients = 0, float("inf"), 0, [], None
    if last.exists():
        if not resume:
            raise ValueError("Calibration exists; pass --resume")
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            raise ValueError("Calibration identity changed")
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"])
        epoch0, best, stale, history, gradients = state["epoch"], state["best"], state["stale"], state["history"], state["active_gradient_parameters"]
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"]); generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_json(root / "identity.json", identity)
    cap = min(config["max_epochs"], epoch0 + stop_after) if stop_after else config["max_epochs"]
    for epoch in range(epoch0, cap):
        if stale >= config["patience"]:
            break
        started = time.perf_counter(); model.train(); total, count, updates = 0., 0, 0
        for x, d, m, y in loaders[0]:
            x, d, m, y = (v.to(device) for v in (x, d, m, y))
            opt.zero_grad(set_to_none=True); loss = objective(model(x, d, m, case), y, m, case)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite calibration objective")
            loss.backward()
            if gradients is None:
                gradients = sum(int(torch.count_nonzero(p.grad)) for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_norm"]); opt.step()
            total += float(loss.detach()) * len(y); count += len(y); updates += 1
        score = _selection(model, loaders[1], case, device)
        history.append({"epoch": epoch+1, "fit_objective": total/count, "selection_objective": score,
                        "updates": updates, "seconds": time.perf_counter()-started})
        if score < best:
            best, stale = score, 0
            save_checkpoint(root / "best.pt", {"identity": identity, "epoch": epoch+1, "model": model.state_dict(), "selection_objective": score})
        else:
            stale += 1
        save_checkpoint(last, {"identity": identity, "epoch": epoch+1, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "best": best, "stale": stale, "history": history, "active_gradient_parameters": gradients,
            "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "loader_rng": generator.get_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type=="cuda" else []})
        save_json(root / "history.json", history)
        print(json.dumps({"case": case, "seed": seed, **history[-1]}), flush=True)
    selected = torch.load(root / "best.pt", map_location="cpu", weights_only=False); model.load_state_dict(selected["model"])
    if _selection(model, loaders[1], case, device) != selected["selection_objective"]:
        raise ValueError("Selected calibration checkpoint changed inner score")
    # Official validation values enter only after all optimization/selection.
    valid, _ = load_table(config["table_dir"], seed, "valid")
    if smoke:
        valid = _limit(valid, 32)
    elif len(valid["ids"]) != 647:
        raise ValueError("Formal native calibration requires all647 VALID rows")
    metrics, records = evaluate(model, valid, stats, case, config["batch_size"], device)
    pred_path = root / "valid_predictions.json"; save_json(pred_path, records)
    result = {"root": str(root), "case": case, "seed": seed, "complete": stale>=config["patience"] or len(history)>=config["max_epochs"],
        "smoke_only": smoke, "best_epoch": selected["epoch"], "epochs_executed": len(history), "optimizer_updates": sum(r["updates"] for r in history),
        "parameters": sum(p.numel() for p in model.parameters()), "first_batch_active_gradient_parameters": gradients,
        "loop_seconds": sum(r["seconds"] for r in history), "valid": metrics, "checkpoint_sha256": digest(root / "best.pt"),
        "predictions_sha256": digest(pred_path), "test_evaluated": False}
    save_json(root / "result.json", result); return result


def _global(config, data, metadata, cohorts, stats, root, identity, smoke, resume):
    from scipy.optimize import minimize_scalar
    if (root / "result.json").exists():
        if not resume:
            raise ValueError("Global calibration exists; pass --resume")
        previous = json.loads((root / "identity.json").read_text())
        if previous != identity:
            raise ValueError("Global calibration identity changed")
        result = json.loads((root / "result.json").read_text())
        if digest(root / "valid_predictions.json") != result["predictions_sha256"]:
            raise ValueError("Saved global calibration predictions changed")
        return result
    indices = cohorts[0]; m = torch.from_numpy(data["anchor"][indices]); y = torch.from_numpy(data["target"][indices])
    def objective_scale(log_scale):
        scale = math.exp(float(log_scale))
        raw = math.log(math.expm1(max(scale - 1e-4, 1e-12)))
        params = parameters(m, torch.full_like(m, raw), torch.zeros_like(m), mixture=False)
        out = {"bins": bin_probabilities(params, BOUNDARIES.astype(np.float64).tolist())}
        return float(objective(out, y, m, "scale"))
    started = time.perf_counter()
    optimized = minimize_scalar(objective_scale, bounds=(math.log(config["global_scale_min"]),math.log(config["global_scale_max"])), method="bounded", options={"xatol": 1e-8})
    if not optimized.success:
        raise ValueError("Global scale fit failed")
    scale = math.exp(float(optimized.x)); valid, _ = load_table(config["table_dir"], identity["seed"], "valid")
    if smoke:
        valid = _limit(valid, 32)
    elif len(valid["ids"]) != 647:
        raise ValueError("Formal native calibration requires647 VALID rows")
    raw = math.log(math.expm1(max(scale - 1e-4, 1e-12)))
    m = torch.from_numpy(valid["anchor"])
    params = parameters(m, torch.full_like(m, raw), torch.zeros_like(m), mixture=False)
    out = {"parameters": params, "bins": bin_probabilities(params, BOUNDARIES.astype(np.float64).tolist()), "p_hit": error_band_probability(params, float(np.float32(.1)))}
    values = describe(out); metrics = diagnostics(valid["anchor"], valid["target"], values); records = _records(valid, values)
    save_json(root / "identity.json", identity); save_json(root / "valid_predictions.json", records)
    save_json(root / "global_scale.json", {"scale": scale, "fit_objective": float(optimized.fun), "fit_rows": indices.tolist(), "calls": int(optimized.nfev)})
    result = {"root": str(root), "case": "global", "seed": identity["seed"], "complete": True, "smoke_only": smoke,
              "parameters": 1, "optimizer_calls": int(optimized.nfev), "loop_seconds": time.perf_counter()-started,
              "valid": metrics, "scale": scale, "predictions_sha256": digest(root / "valid_predictions.json"), "test_evaluated": False}
    save_json(root / "result.json", result); return result
