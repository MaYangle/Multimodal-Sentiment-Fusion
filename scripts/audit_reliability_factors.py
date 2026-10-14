"""Independent per-ID recomputation and paired-seed reliability factor effects."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import author_rows, digest, save_json
from jevaffect.native_metrics import native_metrics
from jevaffect.regression_uncertainty import uncertainty_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"]) != 12:
        raise ValueError("Expected completed 4×3 formal suite")
    lookup = {}; counts = set(); inputs = []
    for result in suite["results"]:
        root = Path(result["root"])
        identity = json.loads((root / "identity.json").read_text())
        if result["smoke_only"] or result["test_evaluated"] or not result["complete"] or identity["auxiliary_validation_labels_used"]:
            raise ValueError("Incorrect research scope")
        source = identity["config"]["author_csv"]
        if identity["author_csv_sha256"] != digest(source):
            raise ValueError("Author source changed")
        rows = author_rows(source, identity["config"]["dataset"])
        predictions_path = root / "valid_predictions.json"
        if digest(predictions_path) != result["predictions_sha256"] or digest(root / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Changed run artifacts")
        records = json.loads(predictions_path.read_text())
        ids = [r["id"] for r in records]
        if len(set(ids)) != len(ids) or set(ids) != {sid for sid, row in rows.items() if row["split"] == "valid"}:
            raise ValueError("Incomplete or wrong validation cohort")
        target = np.asarray([rows[sid]["score"] for sid in ids], np.float32)
        if not np.array_equal(target, np.asarray([r["target"] for r in records], np.float32)):
            raise ValueError("Original score mismatch")
        parameters = np.asarray([r["nig"] for r in records], np.float32)
        weights = np.asarray([r["weights"] for r in records])
        available = np.asarray([r["availability"] for r in records], bool)
        if np.any(weights < 0) or not np.allclose(weights.sum(1), 1, atol=1e-6) or np.any(weights[~available] != 0):
            raise ValueError("Invalid modality decision weights")
        if not np.array_equal(parameters[:, 0], np.asarray([r["prediction"] for r in records], np.float32)):
            raise ValueError("Distribution mean differs from point prediction")
        native = native_metrics(parameters[:, 0], target, "simsv2")
        uncertainty = uncertainty_metrics(parameters, target, [-1, 1])
        if native != result["valid_metrics"]["native"] or uncertainty != result["valid_metrics"]["uncertainty"]:
            raise ValueError("Recomputed metrics differ")
        history = json.loads((root / "history.json").read_text())
        if min(history, key=lambda r: r["valid"]["native"]["MAE"])["epoch"] != result["best_epoch"]:
            raise ValueError("Wrong selected epoch")
        key = (result["condition"], result["seed"])
        if key in lookup:
            raise ValueError("Duplicate factor seed")
        lookup[key] = result
        counts.add(result["trainable_parameters"])
        inputs.append(identity["feature_hashes"])
    if len(counts) != 1 or any(h != inputs[0] for h in inputs):
        raise ValueError("Factor conditions do not match parameter count and input bytes")
    summary = {}; effects = []
    for condition in ("S0G0", "S0G1", "S1G0", "S1G1"):
        selected = [lookup[condition, seed] for seed in (13, 42, 2026)]
        stats = {}
        for metric in ("MAE", "Corr", "Mult_acc_2"):
            values = [r["valid_metrics"]["native"][metric] for r in selected]
            stats[metric] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
        stats["uncertainty"] = {"density_NLL_mean": float(np.mean([r["valid_metrics"]["uncertainty"]["density_NLL"] for r in selected])),
                                 "coverage90_mean": float(np.mean([r["valid_metrics"]["uncertainty"]["intervals"]["0.9"]["coverage"] for r in selected])),
                                 "width90_mean": float(np.mean([r["valid_metrics"]["uncertainty"]["intervals"]["0.9"]["mean_width"] for r in selected])),
                                 "error_vs_variance_spearman_mean": float(np.mean([r["valid_metrics"]["uncertainty"]["error_vs_variance_spearman"] for r in selected]))}
        stats["epochs"] = [r["epochs_executed"] for r in selected]
        stats["total_optimizer_updates"] = sum(r["optimizer_updates"] for r in selected)
        stats["total_loop_seconds"] = sum(r["loop_seconds"] for r in selected)
        summary[condition] = stats
    for seed in (13, 42, 2026):
        mae = {c: lookup[c, seed]["valid_metrics"]["native"]["MAE"] for c in summary}
        g0, g1 = mae["S0G1"] - mae["S0G0"], mae["S1G1"] - mae["S1G0"]
        effects.append({"seed": seed, "dynamic_minus_uniform_without_aux": g0,
                        "dynamic_minus_uniform_with_aux": g1, "interaction": g1 - g0,
                        "aux_minus_noaux_uniform": mae["S1G0"] - mae["S0G0"],
                        "aux_minus_noaux_dynamic": mae["S1G1"] - mae["S0G1"]})
    baseline = json.loads(Path("docs/native_simsv2_baseline_audit.json").read_text())
    comparison = {model: r["MAE"] for model, r in baseline["summary"].items()}
    report = {"scope": "validation selection factor experiment; not final test", "test_evaluated": False,
              "all_12_prediction_audits_passed": True, "same_feature_bytes": True, "same_parameter_count": list(counts)[0],
              "suite_sha256": digest(a.suite), "summary": summary, "paired_seed_MAE_effects": effects,
              "effect_direction": "negative difference means lower MAE", "existing_point_baselines": comparison,
              "promoted": False, "interpretation": "A factor effect alone does not establish novelty or independent-test gains; compare the simple-fusion baseline and stress/calibration controls before promotion."}
    save_json(a.output, report)
    print(json.dumps({"summary": summary, "effects": effects}, indent=2))


if __name__ == "__main__":
    main()
