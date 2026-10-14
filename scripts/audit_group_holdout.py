"""Audit the source-video-disjoint holdout derived from CH-SIMS TRAIN only."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/feature_cache/simsv2_macbert_train_group_holdout_v1/splits/valid"
CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")
PREDICTIONS = {
    "almt_13": ROOT / "runs/group_holdout_baselines/simsv2_macbert_train_group_holdout_v1/almt/seed_13/validation_predictions.jsonl",
    "almt_42": ROOT / "runs/group_holdout_baselines/simsv2_macbert_train_group_holdout_v1/almt/seed_42/validation_predictions.jsonl",
    "almt_2026": ROOT / "runs/group_holdout_baselines/simsv2_macbert_train_group_holdout_v1/almt/seed_2026/validation_predictions.jsonl",
    "jev_time_13": ROOT / "runs/jev_temporal_group_learned_time/seed_13/validation_predictions.jsonl",
    "jev_time_42": ROOT / "runs/jev_temporal_group_learned_time/seed_42/validation_predictions.jsonl",
    "jev_time_2026": ROOT / "runs/jev_temporal_group_learned_time/seed_2026/validation_predictions.jsonl",
    "cross_token_13": ROOT / "runs/cross_token/group/cross_time_learned/seed_13/validation_predictions.jsonl",
    "cross_token_42": ROOT / "runs/cross_token/group/cross_time_learned/seed_42/validation_predictions.jsonl",
    "cross_token_2026": ROOT / "runs/cross_token/group/cross_time_learned/seed_2026/validation_predictions.jsonl",
    "no_cross_13": ROOT / "runs/cross_token/group/no_cross_time_learned/seed_13/validation_predictions.jsonl",
}


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def metrics(gold: list[int], predicted: list[int]) -> dict:
    matrix = [[0] * 5 for _ in range(5)]
    for expected, actual in zip(gold, predicted, strict=True):
        matrix[expected][actual] += 1
    per_class = []
    for index, name in enumerate(CLASSES):
        tp = matrix[index][index]
        support = sum(matrix[index])
        total_pred = sum(row[index] for row in matrix)
        precision = tp / total_pred if total_pred else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"class": name, "support": support, "precision": precision,
                          "recall": recall, "f1": f1})
    return {"accuracy": sum(matrix[i][i] for i in range(5)) / len(gold),
            "macro_f1": sum(item["f1"] for item in per_class) / 5,
            "confusion_matrix": matrix, "per_class": per_class}


def main() -> None:
    manifest = json.loads((CACHE / "manifest_labels.json").read_text(encoding="utf-8"))
    ids = [row["ID"] for row in manifest]
    expected = {row["ID"]: int(row["class_index"]) for row in manifest}
    train_ids = json.loads((CACHE.parent / "train" / "ids.json").read_text(encoding="utf-8"))
    train_groups = {sample_id.split("/")[0] for sample_id in train_ids}
    overlap_groups = {sample_id.split("/")[0] for sample_id in ids} & train_groups
    overlaps = [sample_id.split("/")[0] in train_groups for sample_id in ids]
    if len(ids) != 540 or len(expected) != 540 or any(row["label"] != CLASSES[row["class_index"]] for row in manifest):
        raise ValueError("Frozen validation manifest is invalid")
    result = {"scope": "selection-set audit on train-only source-video-disjoint holdout; no official validation/test scores",
              "valid_count": len(ids), "classes": list(CLASSES),
              "manifest_sha256": sha(CACHE / "manifest_labels.json"),
              "train_source_groups_overlapping_valid": len(overlap_groups),
              "valid_records_with_source_group_in_train": sum(overlaps), "runs": {}}
    predicted_by_run = {}
    for name, path in PREDICTIONS.items():
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        by_id = {row["ID"]: row for row in rows}
        if len(rows) != 540 or len(by_id) != 540 or set(by_id) != set(expected):
            raise ValueError(f"{name}: IDs/count differ from frozen manifest")
        y_hat = []
        nll = 0.0
        for sample_id in ids:
            row = by_id[sample_id]
            if row["split"] != "valid" or row["label"] != expected[sample_id] or tuple(row["class_order"]) != CLASSES:
                raise ValueError(f"{name}: metadata mismatch for {sample_id}")
            probs = row["probabilities"]
            if len(probs) != 5 or any(not math.isfinite(x) or x < 0 for x in probs) or abs(sum(probs) - 1) > 1e-5:
                raise ValueError(f"{name}: invalid distribution for {sample_id}")
            pred = max(range(5), key=lambda i: probs[i])
            if row["argmax"] != CLASSES[pred]:
                raise ValueError(f"{name}: argmax mismatch for {sample_id}")
            y_hat.append(pred)
            nll += -math.log(max(probs[expected[sample_id]], 1e-15))
        predicted_by_run[name] = y_hat
        item = metrics([expected[x] for x in ids], y_hat)
        item.update({"prediction_sha256": sha(path), "nll": nll / len(ids)})
        item["accuracy_source_group_overlap"] = (sum(pred == expected[sid] for pred, sid, flag in zip(y_hat, ids, overlaps, strict=True) if flag) / sum(overlaps)) if any(overlaps) else None
        item["accuracy_source_group_disjoint"] = sum(pred == expected[sid] for pred, sid, flag in zip(y_hat, ids, overlaps, strict=True) if not flag) / max(1, len(ids) - sum(overlaps))
        result["runs"][name] = item
    result["three_seed"] = {}
    for family in ("almt", "jev_time", "cross_token"):
        accuracies = [result["runs"][family + "_" + str(seed)]["accuracy"] for seed in (13, 42, 2026)]
        f1s = [result["runs"][family + "_" + str(seed)]["macro_f1"] for seed in (13, 42, 2026)]
        result["three_seed"][family] = {"accuracy_mean": statistics.mean(accuracies),
                                         "accuracy_sample_sd": statistics.stdev(accuracies),
                                         "macro_f1_mean": statistics.mean(f1s),
                                         "macro_f1_sample_sd": statistics.stdev(f1s)}
    result["three_seed_accuracy_mean_difference_jev_minus_almt"] = (result["three_seed"]["jev_time"]["accuracy_mean"] - result["three_seed"]["almt"]["accuracy_mean"])
    result["three_seed_accuracy_mean_difference_cross_minus_almt"] = (result["three_seed"]["cross_token"]["accuracy_mean"] - result["three_seed"]["almt"]["accuracy_mean"])
    result["three_seed_accuracy_mean_difference_cross_minus_jev"] = (result["three_seed"]["cross_token"]["accuracy_mean"] - result["three_seed"]["jev_time"]["accuracy_mean"])
    result["paired_vs_almt_13"] = {}
    for name in ("jev_time_13", "cross_token_13", "no_cross_13"):
        a, b = predicted_by_run["almt_13"], predicted_by_run[name]
        y = [expected[x] for x in ids]
        result["paired_vs_almt_13"][name] = {
            "both_correct": sum(x == gold and z == gold for x, z, gold in zip(a, b, y, strict=True)),
            "almt_only_correct": sum(x == gold and z != gold for x, z, gold in zip(a, b, y, strict=True)),
            "candidate_only_correct": sum(x != gold and z == gold for x, z, gold in zip(a, b, y, strict=True)),
            "both_wrong": sum(x != gold and z != gold for x, z, gold in zip(a, b, y, strict=True)),
        }
    result["paired_same_seed_main_vs_almt"] = {}
    y = [expected[x] for x in ids]
    for family in ("jev_time", "cross_token"):
        for seed in (13, 42, 2026):
            a = predicted_by_run[f"almt_{seed}"]
            b = predicted_by_run[f"{family}_{seed}"]
            result["paired_same_seed_main_vs_almt"][family + "_" + str(seed)] = {
                "almt_only_correct": sum(x == gold and z != gold for x, z, gold in zip(a, b, y, strict=True)),
                "candidate_only_correct": sum(x != gold and z == gold for x, z, gold in zip(a, b, y, strict=True)),
            }
    out = ROOT / "docs/group_holdout_model_paired_audit.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": {k: {"accuracy": v["accuracy"], "macro_f1": v["macro_f1"]}
                                                    for k, v in result["runs"].items()},
                      "paired": result["paired_vs_almt_13"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
