"""Audit all saved context stress predictions and summarize paired seed effects."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest, save_json
from jevaffect.native_metrics import native_metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stress", type=Path, required=True); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    stress = json.loads(a.stress.read_text()); suite = json.loads(a.suite.read_text())
    if not stress["complete"] or stress["test_evaluated"] or stress["suite_sha256"] != digest(a.suite):
        raise ValueError("Incomplete or changed stress suite")
    indexed = {(r["case"], r["seed"]): r for r in stress["results"]}
    if len(indexed) != 18 or len(stress["results"]) != 18:
        raise ValueError("Expected 18 distinct cases/seeds")
    names = list(stress["results"][0]["conditions"]); checked = []
    if len(names) != 11:
        raise ValueError("Expected eleven conditions")
    for case in suite["results"]:
        root = Path(case["root"]); clean_path = root / "valid_predictions.json"
        if digest(clean_path) != case["predictions_sha256"]:
            raise ValueError("Clean predictions changed")
        clean = json.loads(clean_path.read_text()); clean_ids = [r["id"] for r in clean]
        targets = np.array([r["target"] for r in clean], np.float32)
        for name in names:
            path = root / ("stress_" + name.replace(":", "_").replace("+", "_") + ".json")
            records = json.loads(path.read_text())
            if [r["id"] for r in records] != clean_ids or not np.array_equal(np.array([r["target"] for r in records], np.float32), targets):
                raise ValueError("Stress IDs or scores changed")
            metrics = native_metrics(np.array([r["prediction"] for r in records], np.float32), targets, "simsv2")
            known = indexed[case["case"], case["seed"]]["conditions"][name]["native"]
            if any(abs(metrics[k] - known[k]) > 1e-8 for k in metrics):
                raise ValueError("Stress native metric mismatch")
            checked.append({"case": case["case"], "seed": case["seed"], "condition": name, "count": len(records), "sha256": digest(path)})
    summary = {}
    for case in sorted(set(c for c, _ in indexed)):
        summary[case] = {}
        for name in names:
            rows = [indexed[case, seed]["conditions"][name] for seed in (13, 42, 2026)]
            maes = [r["native"]["MAE"] for r in rows]
            summary[case][name] = {"MAE_mean": float(np.mean(maes)), "MAE_sample_sd": float(np.std(maes, ddof=1)),
                                  "coverage90_mean": float(np.mean([r["uncertainty"]["intervals"]["0.9"]["coverage"] for r in rows]))}
    save_json(a.output, {"scope": "validation pooled-feature stress; actual saved-prediction audit", "test_evaluated": False,
                         "checked_evaluations": checked, "summary": summary, "suite_sha256": digest(a.suite), "stress_sha256": digest(a.stress)})
    print(json.dumps(summary, indent=2))
