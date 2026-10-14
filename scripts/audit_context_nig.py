"""Per-ID audit of all registered context-NIG cases and shared input bytes."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.context_nig import CASES
from jevaffect.native_training import author_rows, digest, save_json
from jevaffect.native_metrics import native_metrics
from jevaffect.regression_uncertainty import uncertainty_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"]) != 18:
        raise ValueError("Expected all six cases and three formal seeds")
    grouped = {}; input_hashes = None; evidence = []
    for result in suite["results"]:
        root = Path(result["root"])
        identity = json.loads((root / "identity.json").read_text())
        if identity["auxiliary_annotations_used"] or result["test_evaluated"] or result["smoke_only"] or not result["complete"]:
            raise ValueError("Unregistered supervision/scope")
        csv = identity["config"]["author_csv"]
        if digest(csv) != identity["author_csv_sha256"]:
            raise ValueError("Author metadata changed")
        rows = author_rows(csv, "simsv2")
        file = root / "valid_predictions.json"
        if digest(file) != result["predictions_sha256"] or digest(root / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Changed prediction/checkpoint bytes")
        records = json.loads(file.read_text())
        ids = [r["id"] for r in records]
        if len(set(ids)) != len(ids) or set(ids) != {sid for sid, r in rows.items() if r["split"] == "valid"}:
            raise ValueError("Incorrect validation ID coverage")
        target = np.array([rows[sid]["score"] for sid in ids], np.float32)
        params = np.array([r["nig"] for r in records], np.float32)
        if not np.array_equal(target, np.array([r["target"] for r in records], np.float32)):
            raise ValueError("Native targets differ")
        if native_metrics(params[:, 0], target, "simsv2") != result["valid_metrics"]["native"] or uncertainty_metrics(params, target, [-1, 1]) != result["valid_metrics"]["uncertainty"]:
            raise ValueError("Recomputed metrics differ")
        weights = np.array([r["weights"] for r in records])
        available = np.array([r["availability"] for r in records], bool)
        if weights.shape != available.shape or not np.allclose(weights.sum(1), 1, atol=1e-6) or np.any(weights < 0) or np.any(weights[~available] != 0):
            raise ValueError("Invalid expert eligibility/weights")
        if result["case"].startswith("joint_off_") and available[:, 3].any():
            raise ValueError("Removed joint expert remains eligible")
        history = json.loads((root / "history.json").read_text())
        if min(history, key=lambda r: r["valid"]["native"]["MAE"])["epoch"] != result["best_epoch"]:
            raise ValueError("Wrong selected checkpoint")
        if input_hashes is None:
            input_hashes = identity["feature_hashes"]
        if input_hashes != identity["feature_hashes"]:
            raise ValueError("Cases use different feature bytes")
        grouped.setdefault(result["case"], []).append(result)
        evidence.append({"case": result["case"], "seed": result["seed"], "count": len(ids), "target_metric_hash_epoch_audit_passed": True})
    summary = {}
    for case in CASES:
        results = grouped[case]
        if len(results) != 3 or {r["seed"] for r in results} != {13, 42, 2026}:
            raise ValueError("Missing or duplicated seed")
        stats = {}
        for metric in ("MAE", "Corr", "Mult_acc_2"):
            values = [r["valid_metrics"]["native"][metric] for r in results]
            stats[metric] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
        stats["coverage90_mean"] = float(np.mean([r["valid_metrics"]["uncertainty"]["intervals"]["0.9"]["coverage"] for r in results]))
        stats["width90_mean"] = float(np.mean([r["valid_metrics"]["uncertainty"]["intervals"]["0.9"]["mean_width"] for r in results]))
        stats["error_variance_spearman_mean"] = float(np.mean([r["valid_metrics"]["uncertainty"]["error_vs_variance_spearman"] for r in results]))
        for name in ("parameters_total", "parameters_requires_grad", "first_batch_nonzero_gradient_parameters"):
            stats[name] = [r[name] for r in results]
        stats["epochs"] = [r["epochs_executed"] for r in results]
        stats["updates"] = [r["optimizer_updates"] for r in results]
        stats["total_loop_seconds"] = sum(r["loop_seconds"] for r in results)
        summary[case] = stats
    paired = []
    for seed in (13, 42, 2026):
        values = {c: next(r["valid_metrics"]["native"]["MAE"] for r in grouped[c] if r["seed"] == seed) for c in CASES}
        paired.append({"seed": seed, **{f"joint_minus_three_{f}": values["joint_" + f] - values["three_" + f] for f in ("uniform", "dynamic")},
                       **{f"joint_minus_off_{f}": values["joint_" + f] - values["joint_off_" + f] for f in ("uniform", "dynamic")}})
    report = {"scope": "native validation selection results; not frozen test", "test_evaluated": False,
              "suite_sha256": digest(a.suite), "shared_input_bytes_verified": True, "summary": summary, "paired_MAE": paired,
              "evidence": evidence, "promoted_as_original_method": False,
              "capacity_limit": "Joint-off loses effective capacity; the wider three-expert reference controls overall capacity to within0.2%."}
    save_json(a.output, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
