"""Fit known uncertainty controls on TRAIN OOF; evaluate frozen valid outputs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.native_training import digest, save_json, author_rows
from jevaffect.native_metrics import native_metrics
from jevaffect.source_crossfit import training_rows_only, validate_folds, manifest_hash
from jevaffect.dependence import fit_correlation, covariance_variance, nig_variance, replace_variance, fit_variance_scale, fit_pool_scale
from jevaffect.regression_uncertainty import uncertainty_metrics
from jevaffect.student_pool import pool_metrics


def arrays(records):
    return (np.array([r["branch_nig"] for r in records], np.float64),
            np.array([r["availability"] for r in records], bool),
            np.array([r["fused_nig"] if "fused_nig" in r else r["nig"] for r in records], np.float64),
            np.array([r["target"] for r in records], np.float64))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--oof-suite", type=Path, required=True)
    p.add_argument("--final-suite", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); config = json.loads(a.config.read_text())
    suites = [json.loads(path.read_text()) for path in (a.oof_suite, a.final_suite)]
    oof, final = suites
    if any(not s["suite_complete"] or s["smoke_only"] for s in suites):
        raise ValueError("Complete formal suites required")
    train_rows = training_rows_only(config["author_csv"])
    validate_folds(list(train_rows), oof["manifest"]["folds"])
    if len(oof["results"]) != 9 or {(r["seed"], r["fold"]) for r in oof["results"]} != {(s, f) for s in config["seeds"] for f in range(3)}:
        raise ValueError("Expected exact nine-fold seed contract")
    report = {"scope": "TRAIN OOF fit, frozen official validation evaluation", "new_formula_claimed": False,
              "test_evaluated": False, "config": config, "oof_suite_sha256": digest(a.oof_suite),
              "final_suite_sha256": digest(a.final_suite), "implementation_sha256": {n: digest(Path("jevaffect") / n) for n in ("dependence.py", "student_pool.py", "regression_uncertainty.py")},
              "results": []}
    for seed in config["seeds"]:
        folds = sorted([r for r in oof["results"] if r["seed"] == seed], key=lambda x: x["fold"])
        training_records = []; fold_corr = []
        for r in folds:
            root = Path(r["root"]); path = root / "oof_predictions.json"
            if digest(path) != r["predictions_sha256"] or digest(root / "best.pt") != r["checkpoint_sha256"]:
                raise ValueError("OOF artifact changed")
            identity = json.loads((root / "identity.json").read_text())
            if identity["config"] != config or identity["manifest_sha256"] != manifest_hash(oof["manifest"]) or identity["cohort"] != oof["manifest"]["folds"][r["fold"]]:
                raise ValueError("OOF config or source cohort changed")
            if identity["official_validation_labels_parsed"] or identity["author_csv_sha256"] != digest(config["author_csv"]):
                raise ValueError("Invalid OOF provenance")
            history = json.loads((root / "history.json").read_text())
            best = min(history, key=lambda row: row["inner_select"]["MAE"])
            if best["epoch"] != r["best_epoch"]:
                raise ValueError("OOF model selection mismatch")
            records = json.loads(path.read_text())
            if [x["id"] for x in records] != identity["cohort"]["oof_ids"]:
                raise ValueError("OOF prediction IDs differ")
            for x in records:
                if x["id"] not in train_rows or abs(x["target"] - train_rows[x["id"]]["score"]) > 1e-6:
                    raise ValueError("OOF target differs from native TRAIN score")
            training_records.extend(records)
            ep, av, _, y = arrays(records)
            fold_corr.append(fit_correlation(ep, av, y, config["correlation_shrinkage"])[1].tolist())
        ep, av, fused, y = arrays(training_records)
        corr, raw_corr = fit_correlation(ep, av, y, config["correlation_shrinkage"])
        bounds = config["log_variance_scale_bounds"]
        variances = {"global_NIG": nig_variance(fused), "independent": covariance_variance(ep, av, np.eye(ep.shape[1])),
                     "correlated": covariance_variance(ep, av, corr)}
        fitted = {name: fit_variance_scale(fused, var, y, bounds) for name, var in variances.items()}
        fitted["proper_pool"] = fit_pool_scale(ep, av, y, bounds)
        selected = [r for r in final["results"] if r["case"] == "joint_dynamic" and r["seed"] == seed]
        if len(selected) != 1:
            raise ValueError("Expected frozen joint_dynamic checkpoint per seed")
        selected = selected[0]; root = Path(selected["root"]); path = root / "valid_predictions.json"
        if digest(path) != selected["predictions_sha256"] or digest(root / "best.pt") != selected["checkpoint_sha256"]:
            raise ValueError("Frozen final artifact changed")
        final_identity = json.loads((root / "identity.json").read_text())
        for key, value in final_identity["config"].items():
            if key not in ("run_root",) and config.get(key) != value:
                raise ValueError("OOF/final training policy mismatch: " + key)
        if any(json.loads((Path(r["root"]) / "identity.json").read_text())["feature_hashes"] != final_identity["feature_hashes"]["train"] for r in folds):
            raise ValueError("OOF/final cache differs")
        records = json.loads(path.read_text()); valid_rows = {sid: r for sid, r in author_rows(config["author_csv"], "simsv2").items() if r["split"] == "valid"}
        if len(records) != len(valid_rows) or {r["id"] for r in records} != set(valid_rows):
            raise ValueError("Wrong frozen validation cohort")
        for r in records:
            if abs(r["target"] - valid_rows[r["id"]]["score"]) > 1e-6:
                raise ValueError("Invalid validation target")
        vp, va, vf, vy = arrays(records)
        validation_variances = {"global_NIG": nig_variance(vf), "independent": covariance_variance(vp, va, np.eye(vp.shape[1])),
                               "correlated": covariance_variance(vp, va, corr)}
        distributions = {"original_NIG": vf}
        for name, var in validation_variances.items():
            distributions[name] = replace_variance(vf, var, fitted[name]["variance_scale"])
        row = {"seed": seed, "fit_count": len(training_records), "valid_count": len(records), "raw_correlation": raw_corr.tolist(),
               "shrunk_correlation": corr.tolist(), "fold_raw_correlations": fold_corr, "fitted_scales": fitted,
               "oof_loop_seconds": sum(r["loop_seconds"] for r in folds), "oof_epochs": [r["epochs_executed"] for r in folds],
               "final_checkpoint_sha256": selected["checkpoint_sha256"], "valid_predictions_sha256": selected["predictions_sha256"], "methods": {}}
        artifact_root = Path(config["run_root"]) / "readouts" / f"seed_{seed}"; artifact_root.mkdir(parents=True, exist_ok=True)
        for name, params in distributions.items():
            if not np.array_equal(params[:, :3], vf[:, :3]):
                raise ValueError("Correction changed point mean or degrees of freedom")
            row["methods"][name] = {"native": native_metrics(params[:, 0].astype(np.float32), vy.astype(np.float32), "simsv2"),
                                    "uncertainty": uncertainty_metrics(params, vy, [-1, 1]), "point_mean_max_difference": 0.}
            save_json(artifact_root / (name + ".json"), [{"id": r["id"], "target": r["target"], "nig": params[i].tolist()} for i, r in enumerate(records)])
        for name, q in (("proper_pool", 1.), ("scaled_proper_pool", fitted["proper_pool"]["variance_scale"])):
            params = vp.copy(); params[:, :, 3] *= q
            metrics = pool_metrics(params, va, vy, [-1, 1]); mean = np.array(metrics.pop("point_mean"))
            difference = float(np.max(np.abs(mean - vf[:, 0])))
            if difference > 1e-5:
                raise ValueError("Proper pool mean differs beyond float32 fusion tolerance")
            row["methods"][name] = {"native": native_metrics(mean.astype(np.float32), vy.astype(np.float32), "simsv2"),
                                    "uncertainty": metrics, "point_mean_max_difference": difference}
            save_json(artifact_root / (name + ".json"), [{"id": r["id"], "target": r["target"], "expert_nig": params[i].tolist(), "availability": va[i].tolist()} for i, r in enumerate(records)])
        save_json(artifact_root / "fit.json", row)
        report["results"].append(row); save_json(a.output, report)
        print(json.dumps({"seed": seed, "q": {k: v["variance_scale"] for k, v in fitted.items()},
                          "valid_NLL": {k: v["uncertainty"]["density_NLL"] for k, v in row["methods"].items()}}), flush=True)
    report["complete"] = True; save_json(a.output, report)


if __name__ == "__main__":
    main()
