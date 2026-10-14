"""Audit native M prediction and registered structure/auxiliary factor effects."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import author_rows, digest, save_json
from jevaffect.native_metrics import native_metrics
from jevaffect.reference_residual import REFERENCE_CASES

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"]) != 12 or {(r["case"], r["seed"]) for r in suite["results"]} != {(c, s) for c in REFERENCE_CASES for s in (13, 42, 2026)}:
        raise ValueError("Complete exact12-case formal study required")
    grouped = {}; checked = []; predictions = {}; shared_hashes = None
    for result in suite["results"]:
        root = Path(result["root"]); identity = json.loads((root / "identity.json").read_text()); cfg = identity["config"]
        if digest(cfg["author_csv"]) != identity["source_csv_sha256"] or identity["validation_reference_labels_parsed"]:
            raise ValueError("Changed annotations or invalid auxiliary-label scope")
        if shared_hashes is None:
            shared_hashes = identity["features"]
        if shared_hashes != identity["features"]:
            raise ValueError("Factor cases use different feature bytes")
        for hashes in identity["features"].values():
            for name, expected in hashes.items():
                path = Path(name) if Path(name).name in ("meta.json", "metadata.json", "ids.json") else Path(cfg["cache_dir"]) / name
                if digest(path) != expected:
                    raise ValueError("Changed source feature: " + name)
        for name, expected in identity["implementation"].items():
            if digest(Path("jevaffect") / name) != expected:
                raise ValueError("Changed training/model implementation")
        if digest(root / "best.pt") != result["checkpoint_sha256"] or digest(root / "valid_predictions.json") != result["predictions_sha256"]:
            raise ValueError("Changed saved run artifact")
        records = json.loads((root / "valid_predictions.json").read_text()); rows = author_rows(cfg["author_csv"], "simsv2")
        ids = [r["id"] for r in records]
        if len(set(ids)) != len(ids) or set(ids) != {sid for sid, r in rows.items() if r["split"] == "valid"}:
            raise ValueError("Wrong complete validation IDs")
        y = np.array([rows[sid]["score"] for sid in ids], np.float32)
        if not np.array_equal(y, np.array([r["target"] for r in records], np.float32)):
            raise ValueError("M scores changed")
        pred = np.array([r["prediction"] for r in records], np.float32)
        if native_metrics(pred, y, "simsv2") != result["valid_metrics"]:
            raise ValueError("Native metric recomputation differs")
        for r in records:
            if abs(r["additive_joint"] - r["text_reference"] - r["annotation_correction"]) > 1e-6:
                raise ValueError("Additive score identity broken")
            primary = r["additive_joint"] if result["case"].startswith("sum") else r["direct_joint"]
            if r["prediction"] != primary:
                raise ValueError("Primary decoder differs")
        history = json.loads((root / "history.json").read_text()); best = min(history, key=lambda h: h["valid"]["MAE"])
        if best["epoch"] != result["best_epoch"] or not result["complete"] or result["test_evaluated"]:
            raise ValueError("Selection/completion mismatch")
        grouped.setdefault(result["case"], []).append(result); predictions[result["case"], result["seed"]] = records
        checked.append({"case": result["case"], "seed": result["seed"], "count": len(ids), "M_target_metric_decoder_hash_epoch_audit_passed": True})
    summary = {}
    for case, results in grouped.items():
        summary[case] = {name: {"mean": float(np.mean([r["valid_metrics"][name] for r in results])), "sample_sd": float(np.std([r["valid_metrics"][name] for r in results], ddof=1))} for name in ("MAE", "Corr", "Mult_acc_2")}
        summary[case].update(parameters=[r["parameters"] for r in results], actual_gradient_parameters=[r["first_batch_nonzero_gradient_parameters"] for r in results],
                             epochs=[r["epochs_executed"] for r in results], updates=[r["optimizer_updates"] for r in results], loop_seconds=sum(r["loop_seconds"] for r in results))
    paired = []
    for seed in (13, 42, 2026):
        maes = {c: next(r["valid_metrics"]["MAE"] for r in grouped[c] if r["seed"] == seed) for c in REFERENCE_CASES}
        paired.append({"seed": seed, "sum1_minus_flat1": maes["sum1"] - maes["flat1"], "sum0_minus_flat0": maes["sum0"] - maes["flat0"],
                       "aux_effect_flat": maes["flat1"] - maes["flat0"], "aux_effect_sum": maes["sum1"] - maes["sum0"],
                       "interaction": maes["sum1"] - maes["sum0"] - maes["flat1"] + maes["flat0"]})
    # Post-training diagnostic only; these VALID reference labels were not parsed by trainer.
    source_csv = cfg["author_csv"]
    with Path(source_csv).open(encoding="utf-8-sig", newline="") as stream:
        ref = {r["video_id"] + "/" + r["clip_id"]: float(r["label_T"]) for r in csv.DictReader(stream) if r["mode"] == "valid"}
    ordered = [r["id"] for r in predictions["flat0", 13]]; y = np.array([rows[sid]["score"] for sid in ordered]); t = np.array([ref[sid] for sid in ordered]); difference = y - t
    groups = {"all": np.ones(len(y), bool), "same_annotation": np.abs(difference) < 1e-6, "large_annotation_difference": np.abs(difference) >= .4,
              "opposite_M_T_polarity": (y > 0) != (t > 0), "weak_M": np.abs(y) <= .4, "strong_M": np.abs(y) >= .8}
    strata = {}
    for case in REFERENCE_CASES:
        arrays = [predictions[case, seed] for seed in (13, 42, 2026)]
        if any([r["id"] for r in records] != ordered for records in arrays):
            raise ValueError("Unpaired diagnostic order")
        errors = np.abs(np.clip(np.array([[r["prediction"] for r in records] for records in arrays]), -1, 1) - y)
        strata[case] = {g: {"count": int(mask.sum()), "MAE_mean": float(errors[:, mask].mean()), "MAE_seed_sd": float(errors[:, mask].mean(1).std(ddof=1))} for g, mask in groups.items() if mask.any()}
    report = {"scope": "native VALID selection and post-training annotation strata, not final test", "test_evaluated": False,
              "suite_sha256": digest(a.suite), "summary": summary, "paired_seeds": paired, "evidence": checked, "annotation_strata": strata,
              "gate_consistent_vs_flat1": all(p["sum1_minus_flat1"] < 0 for p in paired), "gate_better_than_early_fusion_mean": summary["sum1"]["MAE"]["mean"] < .422070,
              "new_contribution_verified": False}
    save_json(a.output, report); print(json.dumps({"summary": summary, "paired_seeds": paired}, indent=2))
