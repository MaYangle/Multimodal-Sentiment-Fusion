"""Independently audit the tiny real MOSEI training-flow exercise."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/mosei_raw/pilot/cross_time_learned/seed_13"
CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    label_path = ROOT / "data/mosei/manifests/mosei_valid_manifest.jsonl"
    labels = {row["ID"]: CLASSES.index(row["class"]) for line in label_path.read_text(encoding="utf-8").splitlines()
              if line for row in [json.loads(line)]}
    predictions_path = RUN / "validation_predictions.jsonl"
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line]
    if len(predictions) != 22 or len({row["ID"] for row in predictions}) != 22:
        raise ValueError("Pilot validation IDs/count invalid")
    matrix = [[0] * 5 for _ in range(5)]
    for row in predictions:
        probs = row["probabilities"]
        if (row["ID"] not in labels or row["label"] != labels[row["ID"]] or
                row["split"] != "valid" or tuple(row["class_order"]) != CLASSES or
                len(probs) != 5 or abs(sum(probs) - 1) > 1e-5 or
                any(not math.isfinite(x) or x < 0 for x in probs)):
            raise ValueError("Pilot prediction violates competition-ID protocol")
        predicted = max(range(5), key=lambda index: probs[index])
        if row["argmax"] != CLASSES[predicted]:
            raise ValueError("Stored argmax differs from probabilities")
        matrix[row["label"]][predicted] += 1
    accuracy = sum(matrix[i][i] for i in range(5)) / 22
    summary = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    if not summary["completed"] or abs(accuracy - summary["best_accuracy"]) > 1e-12 or manifest["valid_count"] != 22:
        raise ValueError("Pilot checkpoint/independent score mismatch")
    report = {"scope": "real 54-train/22-valid MOSEI missing-media flow smoke; NOT a generalization benchmark",
              "validation_count": 22, "selected_accuracy": accuracy,
              "selected_epoch": summary["best_epoch"], "finished_epoch": summary["last_epoch"],
              "train_final_accuracy": json.loads((RUN / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[-1])["train_accuracy"],
              "confusion_matrix": matrix, "checkpoint_sha256": sha(RUN / "best.pt"),
              "prediction_sha256": sha(predictions_path), "label_manifest_sha256": sha(label_path),
              "source_sha256": manifest["source_sha256"], "missing_video_train": 4,
              "missing_video_valid": 2, "exact_resume_used": True,
              "test_labels_used": False, "not_for_method_ranking": True}
    output = ROOT / "docs/mosei_pilot_training_flow.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
