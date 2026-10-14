"""Paired native VALID errors and source-cluster intervals for publisher cohort."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if not suite["suite_complete"] or suite["smoke_only"]:
        raise ValueError("Complete formal suite required")
    cache = Path(json.loads((Path(suite["results"][0]["root"]) / "identity.json").read_text())["config"]["cache_dir"])
    ids = json.loads((cache / "valid/ids.json").read_text()); y = np.load(cache / "valid/targets.npy")
    lengths = np.load(cache / "valid/text_lengths.npy")
    x = np.load(cache / "valid/vision.npy", mmap_mode="r"); zero_vision = np.all(x == 0, axis=(1, 2)); x._mmap.close()
    groups = {"all": np.ones(len(y), bool), "exact_neutral": y == 0, "weak_nonzero": (np.abs(y) <= 1) & (y != 0),
              "strong_sentiment": np.abs(y) >= 2, "long50_steps": lengths == 50, "short10_or_less": lengths <= 10, "all_zero_vision": zero_vision}
    report = {"scope": "VALID development errors; original labels; no final test", "test_evaluated": False,
              "suite_sha256": digest(a.suite), "groups": {k: int(v.sum()) for k, v in groups.items()}, "models": {}, "paired_comparisons": []}
    errors = {}; sources = np.array([s.split("|")[0] for s in ids]); unique_sources = sorted(set(sources))
    source_indices = [np.flatnonzero(sources == s) for s in unique_sources]; source_sizes = np.array([len(i) for i in source_indices])
    rng = np.random.default_rng(105); draws = rng.integers(len(unique_sources), size=(2000, len(unique_sources))); denominator = source_sizes[draws].sum(1)
    for model in ("text", "early_fusion", "almt"):
        predictions = []
        for seed in (13, 42, 2026):
            result = next(r for r in suite["results"] if r["model"] == model and r["seed"] == seed)
            path = Path(result["root"]) / "valid_predictions.json"
            if digest(path) != result["predictions_sha256"]:
                raise ValueError("Predictions changed")
            records = json.loads(path.read_text())
            if [r["id"] for r in records] != ids:
                raise ValueError("Unpaired validation IDs")
            predictions.append([r["prediction"] for r in records])
        predictions = np.asarray(predictions); errors[model] = np.abs(predictions - y)
        worst = np.argsort(-errors[model].mean(0))[:10]
        report["models"][model] = {"groups": {k: {"MAE_mean": float(errors[model][:, mask].mean()), "MAE_seed_sd": float(errors[model][:, mask].mean(1).std(ddof=1))} for k, mask in groups.items() if mask.any()},
                                  "worst_ids": [{"id": ids[i], "target": float(y[i]), "mean_prediction": float(predictions[:, i].mean()), "MAE": float(errors[model][:, i].mean())} for i in worst]}
    for candidate, baseline in (("early_fusion", "text"), ("almt", "text"), ("almt", "early_fusion")):
        per_seed = []
        for i, seed in enumerate((13, 42, 2026)):
            difference = errors[candidate][i] - errors[baseline][i]
            totals = np.array([difference[ix].sum() for ix in source_indices]); boot = totals[draws].sum(1) / denominator
            per_seed.append({"seed": seed, "MAE_difference": float(difference.mean()), "source_percentile95": np.quantile(boot, [.025, .975]).tolist()})
        report["paired_comparisons"].append({"candidate": candidate, "baseline": baseline, "negative_difference_favors_candidate": True, "seeds": per_seed})
    report["bootstrap"] = {"seed": 105, "replicates": 2000, "source_videos": len(unique_sources), "limitation": "Conditional on three validation-selected runs; does not account for training/selection uncertainty or multiple comparisons."}
    save_json(a.output, report); print(json.dumps(report["paired_comparisons"], indent=2))
