"""Recompute native publisher-cohort scores and audit actual run identities."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest, save_json
from jevaffect.native_metrics import native_metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=("text", "early_fusion", "almt", "almt_masked"), default=["text", "early_fusion", "almt"])
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if len(set(a.models)) != len(a.models) or not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"]) != 3 * len(a.models):
        raise ValueError("Exact complete registered model/three-seed suite required")
    grouped = {}; evidence = []; shared_meta = None
    expected_pairs = {(m, s) for m in a.models for s in (13, 42, 2026)}
    if {(r["model"], r["seed"]) for r in suite["results"]} != expected_pairs:
        raise ValueError("Missing or repeated case/seed")
    for result in suite["results"]:
        root = Path(result["root"]); identity = json.loads((root / "identity.json").read_text()); config = identity["config"]
        cache = Path(config["cache_dir"]); meta_path = cache / "metadata.json"; meta = json.loads(meta_path.read_text())
        if shared_meta is None:
            shared_meta = digest(meta_path)
        if digest(meta_path) != shared_meta or meta["source_sha256"] != config["expected_source_sha256"] or identity["source_sha256"] != meta["source_sha256"]:
            raise ValueError("Models use different input source/cache versions")
        for split, recorded in identity["features"].items():
            for name, expected in recorded.items():
                file = meta_path if Path(name).name == "metadata.json" else cache / name
                if digest(file) != expected:
                    raise ValueError("Run feature hash changed: " + name)
        for name, expected in identity["implementation"].items():
            if digest(Path(__file__).resolve().parents[1] / name) != expected:
                raise ValueError("Run implementation changed: " + name)
        ids = json.loads((cache / "valid/ids.json").read_text()); targets = np.load(cache / "valid/targets.npy")
        records_path = root / "valid_predictions.json"
        if digest(records_path) != result["predictions_sha256"] or digest(root / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Prediction/checkpoint artifact changed")
        records = json.loads(records_path.read_text())
        if [r["id"] for r in records] != ids or not np.array_equal(np.array([r["target"] for r in records], np.float32), targets):
            raise ValueError("Published validation IDs/scores differ")
        metrics = native_metrics(np.array([r["prediction"] for r in records], np.float32), targets, "mosei")
        if metrics != result["valid_metrics"]:
            raise ValueError("Native metric recomputation differs")
        history = json.loads((root / "history.json").read_text()); selected = min(history, key=lambda r: r["valid"]["MAE"])
        if selected["epoch"] != result["best_epoch"] or not result["complete"] or result["test_evaluated"] or result["smoke_only"]:
            raise ValueError("Selection/completion/scope mismatch")
        if result["optimizer_updates"] != sum(r["updates"] for r in history) or result["epochs_executed"] != len(history):
            raise ValueError("Executed budget differs from history")
        grouped.setdefault(result["model"], []).append(result)
        evidence.append({"model": result["model"], "seed": result["seed"], "count": len(ids), "id_target_metric_hash_epoch_audit_passed": True})
    summary = {}
    for model, rows in grouped.items():
        summary[model] = {name: {"mean": float(np.mean([r["valid_metrics"][name] for r in rows])),
                               "sample_sd": float(np.std([r["valid_metrics"][name] for r in rows], ddof=1))}
                          for name in rows[0]["valid_metrics"] if name not in ("count", "nonzero_count")}
        summary[model].update(parameters=rows[0]["parameters"], epochs=[r["epochs_executed"] for r in rows],
                              optimizer_updates=[r["optimizer_updates"] for r in rows], total_loop_seconds=sum(r["loop_seconds"] for r in rows))
    report = {"scope": "published MultiBench cohort VALID selection scores, not final test or exact-current-SDK cohort",
              "test_evaluated": False, "suite_sha256": digest(a.suite), "cache_metadata_sha256": shared_meta,
              "summary": summary, "evidence": evidence, "new_contribution_claimed": False}
    save_json(a.output, report); print(json.dumps(summary, indent=2))
