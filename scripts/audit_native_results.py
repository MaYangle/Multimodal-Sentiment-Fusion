"""Recompute every native baseline score from exact IDs and author targets."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.native_training import author_rows, digest, save_json, NativeSplit, PooledRegressor, evaluate
from jevaffect.native_metrics import native_metrics


def audit(suite, output, run_root=None):
    source = json.loads(Path(suite).read_text())
    if not source["suite_complete"]:
        raise ValueError("Incomplete suite")
    models = {}
    evidence = []
    for result in source["results"]:
        dataset, model, seed = result["dataset"], result["model"], result["seed"]
        root = (Path(run_root) if run_root is not None else Path("runs/native_" + dataset + "_v1")) / model / f"seed_{seed}"
        identity = json.loads((root / "identity.json").read_text())
        config = identity["config"]
        if config["model"] != model or config["dataset"] != dataset or identity["seed"] != seed or Path(config["run_root"]).resolve() / model / f"seed_{seed}" != root.resolve():
            raise ValueError("Native run configuration/seed/path mismatch")
        for name, expected in identity["implementation"].items():
            if digest(Path(__file__).resolve().parents[1] / name) != expected:
                raise ValueError("Trained native implementation changed")
        author = identity["config"]["author_csv"]
        if digest(author) != identity["author_csv_sha256"]:
            raise ValueError("Author annotations changed")
        rows = author_rows(author, dataset)
        prediction_file = root / "valid_predictions.json"
        if digest(prediction_file) != result["predictions_sha256"] or digest(root / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Run artifact hash mismatch")
        records = json.loads(prediction_file.read_text())
        ids = [r["id"] for r in records]
        if len(set(ids)) != len(ids) or set(ids) != {sid for sid, r in rows.items() if r["split"] == "valid"}:
            raise ValueError("Validation predictions differ from complete official cohort")
        y = np.array([rows[sid]["score"] for sid in ids], np.float32)
        if not np.array_equal(y, np.array([r["target"] for r in records], np.float32)):
            raise ValueError("Prediction targets disagree with raw author targets")
        p = np.array([r["prediction"] for r in records], np.float32)
        metrics = native_metrics(p, y, dataset)
        for name, value in metrics.items():
            saved = result["valid_metrics"][name]
            if value != saved:
                raise ValueError("Metric mismatch: " + name)
        history = json.loads((root / "history.json").read_text())
        selected = min(history, key=lambda r: r["valid"]["MAE"])
        if selected["epoch"] != result["best_epoch"] or not result["complete"] or result["test_evaluated"] or result["smoke_only"]:
            raise ValueError("Incorrect selection epoch or run scope")
        checkpoint = torch.load(root / "best.pt", weights_only=False, map_location="cpu", mmap=True)
        if checkpoint["identity"] != identity or checkpoint["epoch"] != result["best_epoch"] or checkpoint["metrics"] != metrics:
            raise ValueError("Native best checkpoint identity/selection/metric mismatch")
        reload_audit = {"reloaded": False}
        if model in ("text", "early_fusion"):
            torch.set_num_threads(4)
            modalities = ("text",) if model == "text" else ("text", "audio", "vision")
            splits = {s: NativeSplit(config["cache_dir"], s, rows, modalities, pooled=True) for s in ("train", "valid")}
            if {s: d.hashes for s, d in splits.items()} != identity["features"]:
                raise ValueError("Native checkpoint feature inputs changed")
            regressor = PooledRegressor({m: x.shape[-1] for m, x in splits["train"].features.items()}, config["hidden"], config["dropout"])
            regressor.load_state_dict(checkpoint["model"], strict=True)
            _, actual = evaluate(regressor, torch.utils.data.DataLoader(splits["valid"], batch_size=config["batch_size"], shuffle=False), "cpu", dataset)
            if [r["id"] for r in actual] != ids or any(a["target"] != b["target"] for a,b in zip(actual,records)):
                raise ValueError("Native checkpoint prediction IDs/targets differ")
            delta = max(abs(a["prediction"]-b["prediction"]) for a,b in zip(actual,records))
            if delta > 1e-6:
                raise ValueError("CPU-reloaded native point predictions differ")
            last = torch.load(root / "last.pt", weights_only=False, map_location="cpu", mmap=True)
            if last["identity"] != identity or last["history"] != history or not (last["stale"] >= config["patience"] or len(history) >= config["max_epochs"]):
                raise ValueError("Native last checkpoint/resume/completion state mismatch")
            reload_audit = {"reloaded": True, "cpu_gpu_prediction_max_delta": delta, "epochs":len(history),
                "actual_optimizer_updates":len(history)*int(np.ceil(len(splits["train"])/config["batch_size"]))}
            for d in splits.values(): d.lengths.clear()
        models.setdefault(model, []).append(result)
        evidence.append({"model": model, "seed": seed, "validation_ids": len(ids), "author_target_match": True,
                         "metric_recomputation_match": True, "checkpoint_hash_verified": True, "minimum_MAE_epoch_verified": True,
                         "checkpoint_forward_audit": reload_audit})
    summary = {}
    for model, results in models.items():
        if {r["seed"] for r in results} != {13, 42, 2026} or len(results) != 3:
            raise ValueError("Missing or repeated formal seed")
        summary[model] = {name: {"mean": float(np.mean([r["valid_metrics"][name] for r in results])),
                                 "sample_sd": float(np.std([r["valid_metrics"][name] for r in results], ddof=1))}
                          for name in results[0]["valid_metrics"] if name not in ("count", "weak_count", "nonzero_count")}
        summary[model]["total_loop_seconds"] = sum(r["loop_seconds"] for r in results)
        summary[model]["trainable_parameters"] = results[0]["trainable_parameters"]
    report = {"scope": "official validation selection results; not final-test results", "test_evaluated": False,
              "selection_rule": "minimum validation native MAE, fixed seeds 13/42/2026",
              "suite_sha256": digest(suite), "summary": summary, "evidence": evidence}
    save_json(output, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-root", type=Path, help="Alternate run root; default retains the original native dataset run path")
    args = parser.parse_args()
    audit(args.suite, args.output, args.run_root)
