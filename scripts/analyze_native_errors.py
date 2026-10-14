"""Validation-only residual analysis against author unimodal disagreement."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import save_json, digest


def main():
    source = Path("data/author/simsv2/meta.csv")
    with source.open(encoding="utf-8-sig", newline="") as f:
        rows = {r["video_id"] + "/" + r["clip_id"]: r for r in csv.DictReader(f) if r["mode"] in ("train", "valid")}
    report = {"scope": "official validation error analysis; not independent test", "test_evaluated": False,
              "csv_sha256": digest(source), "groups": {}, "models": {}}
    ids = sorted(sid for sid, r in rows.items() if r["mode"] == "valid")
    scores = np.array([[float(rows[sid]["label_" + m]) for m in ("T", "A", "V")] for sid in ids])
    target = np.array([float(rows[sid]["label"]) for sid in ids])
    spread = scores.max(1) - scores.min(1)
    conflict = (scores.min(1) < 0) & (scores.max(1) > 0)
    groups = {"all": np.ones(len(ids), bool), "opposite_sign_modalities": conflict,
              "no_opposite_sign": ~conflict, "weak_sentiment": np.abs(target) <= .4,
              "strong_sentiment": np.abs(target) >= .8, "exact_neutral": target == 0}
    for name, mask in groups.items():
        report["groups"][name] = {"count": int(mask.sum()), "definition": name}
    for model in ("text", "early_fusion", "almt"):
        predictions = []
        for seed in (13, 42, 2026):
            records = json.loads((Path("runs/native_simsv2_v1") / model / f"seed_{seed}/valid_predictions.json").read_text())
            lookup = {r["id"]: r["prediction"] for r in records}
            if set(lookup) != set(ids):
                raise ValueError("Wrong selection IDs")
            predictions.append([lookup[sid] for sid in ids])
        predictions = np.clip(np.array(predictions), -1, 1)
        errors = np.abs(predictions - target)
        averaged_error = errors.mean(0)
        stats = {name: {"MAE_mean": float(errors[:, mask].mean()),
                        "MAE_seed_sd": float(errors[:, mask].mean(1).std(ddof=1))}
                 for name, mask in groups.items() if mask.any()}
        worst = np.argsort(-averaged_error)[:10]
        report["models"][model] = {"group_errors": stats, "error_vs_label_spread_pearson": float(np.corrcoef(averaged_error, spread)[0, 1]),
                                     "worst_validation_ids": [{"id": ids[i], "target": float(target[i]),
                                                              "mean_prediction": float(predictions[:, i].mean()),
                                                              "MAE": float(averaged_error[i]), "annotation_spread": float(spread[i])} for i in worst]}
    report["interpretation_limit"] = "Unimodal labels are analysis targets, never model inputs. Correlation with disagreement does not establish modality reliability, causal conflict or novelty. Validation has already been used for checkpoint selection."
    save_json("docs/native_simsv2_error_analysis.json", report)
    print(json.dumps({m: r["group_errors"] for m, r in report["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
