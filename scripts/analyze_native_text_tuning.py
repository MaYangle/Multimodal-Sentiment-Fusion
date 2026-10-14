"""Post-training native text adaptation errors; no test-score parsing."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--baseline-suite", type=Path, required=True)
    p.add_argument("--baseline-run-root", type=Path, default=Path("runs/native_simsv2_v1"))
    p.add_argument("--author-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); rows = []
    with a.author_csv.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mode"] != "valid":
                continue
            rows.append({"id": row["video_id"] + "/" + row["clip_id"], "y": float(row["label"]),
                "conflict": min(float(row["label_" + m]) for m in ("T", "A", "V")) < 0 < max(float(row["label_" + m]) for m in ("T", "A", "V"))})
    ids = [r["id"] for r in rows]; labels = np.array([r["y"] for r in rows], np.float32)
    groups = {"all": np.ones(len(rows), bool), "exact_neutral": labels == 0,
              "weak": np.abs(labels) <= .4, "strong": np.abs(labels) >= .8,
              "opposite_annotation_sign": np.array([r["conflict"] for r in rows])}
    models = {}
    for path, subset in ((a.suite, None), (a.baseline_suite, ("text", "early_fusion"))):
        suite = json.loads(path.read_text())
        if not suite["suite_complete"] or suite.get("smoke_only", False) or any(not r["complete"] or r["smoke_only"] or r["test_evaluated"] for r in suite["results"]):
            raise ValueError("Complete formal suites required")
        for result in suite["results"]:
            if subset is not None and result["model"] not in subset:
                continue
            root = Path(result["root"]) if "root" in result else a.baseline_run_root / result["model"] / f"seed_{result['seed']}"
            records_path = root / "valid_predictions.json"
            if digest(records_path) != result["predictions_sha256"] or digest(root / "best.pt") != result["checkpoint_sha256"]:
                raise ValueError("Frozen experiment artifact changed")
            known = {r["id"]: r for r in json.loads(records_path.read_text())}
            if set(known) != set(ids) or not np.array_equal(np.array([known[sid]["target"] for sid in ids], np.float32), labels):
                raise ValueError("VALID cohorts/targets differ")
            predictions = np.clip(np.array([known[sid]["prediction"] for sid in ids], np.float32), -1, 1)
            name = result.get("mode", result["model"])
            models.setdefault(name, []).append((result["seed"], predictions))
    report = {"scope": "post-training official VALID development errors; annotation strata are not inference inputs",
        "test_evaluated": False, "author_csv_sha256": digest(a.author_csv),
        "groups": {name: int(mask.sum()) for name, mask in groups.items()}, "models": {}, "paired": []}
    for name, values in models.items():
        if {seed for seed, _ in values} != {13, 42, 2026} or len(values) != 3:
            raise ValueError("Each model requires all three seeds")
        report["models"][name] = {group: {"MAE_mean": float(np.mean([np.abs(pred[mask] - labels[mask]).mean() for _, pred in values])),
                    "MAE_sample_sd": float(np.std([np.abs(pred[mask] - labels[mask]).mean() for _, pred in values], ddof=1))} for group, mask in groups.items()}
    for seed, top in models["top4"]:
        frozen = dict(models["frozen"])[seed]; fusion = dict(models["early_fusion"])[seed]
        report["paired"].append({"seed": seed, "top4_minus_frozen_MAE": float(np.abs(top - labels).mean() - np.abs(frozen - labels).mean()),
                                 "top4_minus_fusion_MAE": float(np.abs(top - labels).mean() - np.abs(fusion - labels).mean())})
    a.output.parent.mkdir(parents=True, exist_ok=True); save_json(a.output, report); print(json.dumps(report, indent=2))
