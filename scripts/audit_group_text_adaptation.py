"""Audit frozen versus top-four-layer MacBERT text on the same group holdout."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
SOURCE = ROOT / "data/feature_cache/simsv2_macbert_train_group_holdout_v1/splits/valid/manifest_labels.json"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def score(rows: list[dict], labels: dict[str, int]) -> dict:
    if len(rows) != 540 or {row["ID"] for row in rows} != set(labels):
        raise ValueError("Group holdout prediction IDs differ")
    confusion = [[0] * 5 for _ in range(5)]
    for row in rows:
        probs = row["probabilities"]
        if (row["label"] != labels[row["ID"]] or tuple(row["class_order"]) != CLASSES or
                row["split"] != "valid" or len(probs) != 5 or
                any(not math.isfinite(x) or x < 0 for x in probs) or abs(sum(probs) - 1) > 1e-5):
            raise ValueError("Text prediction row violates frozen protocol")
        best = max(range(5), key=lambda i: probs[i])
        if row["argmax"] != CLASSES[best]:
            raise ValueError("Argmax differs from probabilities")
        confusion[row["label"]][best] += 1
    f1 = []
    for cls in range(5):
        tp = confusion[cls][cls]
        precision = tp / max(1, sum(line[cls] for line in confusion))
        recall = tp / max(1, sum(confusion[cls]))
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": sum(confusion[i][i] for i in range(5)) / 540,
            "macro_f1": statistics.mean(f1), "confusion_matrix": confusion}


def main() -> None:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    labels = {row["ID"]: row["class_index"] for row in manifest}
    if len(labels) != 540:
        raise ValueError("Frozen group holdout manifest changed")
    report = {"scope": "TRAIN-only source-video-disjoint 2,182/540 selection; no official valid/test",
              "manifest_labels_sha256": sha(SOURCE), "runs": {}}
    for seed in (13, 42, 2026):
        path = ROOT / f"runs/text_backbone_group_controls/simsv2_macbert_train_group_holdout_v1_text/text_mlp/seed_{seed}/validation_predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        report["runs"][f"frozen_{seed}"] = {**score(rows, labels), "prediction_sha256": sha(path)}
    path = ROOT / "runs/simsv2_macbert_top4_group/seed_13/validation_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    report["runs"]["top4_finetuned_13"] = {**score(rows, labels), "prediction_sha256": sha(path)}
    report["frozen_three_seed"] = {"accuracy_mean": statistics.mean(report["runs"][f"frozen_{s}"]["accuracy"] for s in (13, 42, 2026)),
                                   "accuracy_sample_sd": statistics.stdev(report["runs"][f"frozen_{s}"]["accuracy"] for s in (13, 42, 2026)),
                                   "macro_f1_mean": statistics.mean(report["runs"][f"frozen_{s}"]["macro_f1"] for s in (13, 42, 2026))}
    report["limitation"] = "Top-four-layer fine-tune was one exploratory seed; later fusion models still use frozen MacBERT features."
    output = ROOT / "docs/macbert_group_text_adaptation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frozen": report["frozen_three_seed"], "top4": report["runs"]["top4_finetuned_13"]["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
