"""Frozen validation diagnostics: redundant experts and proper opinion pool."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.nig import fuse_nig, predictive_variance
from jevaffect.regression_uncertainty import uncertainty_metrics
from jevaffect.student_pool import pool_metrics
from jevaffect.native_training import digest, save_json

if __name__ == "__main__":
    suite = json.loads(Path("docs/native_context_nig_formal_v1.json").read_text())
    if not suite["suite_complete"]:
        raise ValueError("Context suite incomplete")
    report = {"scope": "frozen validation diagnostic; no new training or calibration", "test_evaluated": False,
              "new_algorithm_claimed": False, "results": []}
    for result in suite["results"]:
        if result["case"] != "joint_dynamic":
            continue
        path = Path(result["root"]) / "valid_predictions.json"
        if digest(path) != result["predictions_sha256"]:
            raise ValueError("Predictions changed")
        records = json.loads(path.read_text())
        params = np.array([r["branch_nig"] for r in records], np.float64)
        available = np.array([r["availability"] for r in records], bool)
        targets = np.array([r["target"] for r in records], np.float64)
        base, _ = fuse_nig(torch.from_numpy(params), torch.from_numpy(available))
        duplicated = np.tile(params, (1, 2, 1)); dup_mask = np.tile(available, (1, 2))
        dupe, _ = fuse_nig(torch.from_numpy(duplicated), torch.from_numpy(dup_mask))
        mean_difference = float((base[:, 0] - dupe[:, 0]).abs().max())
        if mean_difference > 1e-10:
            raise ValueError("Duplication changed point means")
        pool = pool_metrics(params, available, targets, [-1, 1])
        pool_mean = np.array(pool.pop("point_mean"))
        if np.max(np.abs(pool_mean - base[:, 0].numpy())) > 1e-10:
            raise ValueError("Proper pool changed gamma-weighted point means")
        report["results"].append({"seed": result["seed"], "unchanged_mean_max_difference": mean_difference,
            "duplicate_predictive_variance_ratio_mean": float((predictive_variance(dupe) / predictive_variance(base)).mean()),
            "base_NIG": uncertainty_metrics(base.numpy(), targets, [-1, 1]),
            "duplicated_NIG": uncertainty_metrics(dupe.numpy(), targets, [-1, 1]), "proper_linear_pool": pool})
        save_json("docs/expert_redundancy_diagnostic.json", report)
        print(json.dumps({"seed": result["seed"], "variance_ratio": report["results"][-1]["duplicate_predictive_variance_ratio_mean"],
                          "coverage90_base": report["results"][-1]["base_NIG"]["intervals"]["0.9"]["coverage"],
                          "coverage90_dupe": report["results"][-1]["duplicated_NIG"]["intervals"]["0.9"]["coverage"],
                          "coverage90_linear_pool": pool["intervals"]["0.9"]["coverage"]}), flush=True)
