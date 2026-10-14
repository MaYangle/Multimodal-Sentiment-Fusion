"""Summarize actual three-seed results and predeclared dependence gates."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    report = json.loads(a.results.read_text())
    if not report["complete"] or report["test_evaluated"] or [r["seed"] for r in report["results"]] != [13, 42, 2026]:
        raise ValueError("Complete registered three-seed validation required")
    summary = {}
    for name in report["results"][0]["methods"]:
        rows = [r["methods"][name] for r in report["results"]]
        values = {"native_MAE": [r["native"]["MAE"] for r in rows],
                  "density_NLL": [r["uncertainty"]["density_NLL"] for r in rows],
                  "coverage90": [r["uncertainty"]["intervals"]["0.9"]["coverage"] for r in rows],
                  "width90": [r["uncertainty"]["intervals"]["0.9"]["mean_width"] for r in rows]}
        summary[name] = {k: {"mean": float(np.mean(v)), "sample_sd": float(np.std(v, ddof=1))} for k, v in values.items()}
    paired = []
    for row in report["results"]:
        methods = row["methods"]
        corr = methods["correlated"]["uncertainty"]["density_NLL"]
        paired.append({"seed": row["seed"], "correlated_minus_global_NIG_NLL": corr - methods["global_NIG"]["uncertainty"]["density_NLL"],
                       "correlated_minus_independent_NLL": corr - methods["independent"]["uncertainty"]["density_NLL"],
                       "correlated_minus_scaled_pool_NLL": corr - methods["scaled_proper_pool"]["uncertainty"]["density_NLL"],
                       "point_mean_max_difference": max(r["point_mean_max_difference"] for r in methods.values())})
    matrices = np.array([r["raw_correlation"] for r in report["results"]]); off = ~np.eye(4, dtype=bool)
    output = {"scope": "validation; all correction fits use official TRAIN OOF", "results_sha256": digest(a.results),
              "test_evaluated": False, "summary": summary, "paired_seeds": paired,
              "raw_correlation_off_diagonal_mean": float(matrices[:, off].mean()),
              "max_seed_pair_correlation_difference": float(np.max(np.ptp(matrices, axis=0))),
              "max_within_seed_fold_correlation_difference": float(max(np.max(np.ptp(r["fold_raw_correlations"], axis=0)) for r in report["results"])),
              "oof_loop_seconds": sum(r["oof_loop_seconds"] for r in report["results"]),
              "all_fitted_scales_interior": not any(f["at_boundary"] for r in report["results"] for f in r["fitted_scales"].values()),
              "correlation_consistently_beats_scalar_control": all(r["correlated_minus_global_NIG_NLL"] < 0 for r in paired),
              "new_contribution_verified": False}
    save_json(a.output, output); print(json.dumps(output, indent=2))
