"""Group-disjoint five-fold validation temperature fitting with no test access.

The checkpoint was already selected on this official validation set, so these
calibration diagnostics remain selection-set evidence. Each calibration
prediction uses a temperature fitted without that source-video group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scaled(probs: list[float], temperature: float) -> list[float]:
    logits = [math.log(max(x, 1e-15)) / temperature for x in probs]
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    denominator = sum(weights)
    return [x / denominator for x in weights]


def nll(rows: list[dict], temperature: float) -> float:
    return sum(-math.log(max(scaled(row["probabilities"], temperature)[row["label"]], 1e-15))
               for row in rows) / len(rows)


def fit_temperature(rows: list[dict]) -> float:
    if not rows:
        raise ValueError("Cannot fit on an empty fold")
    low, high = math.log(0.25), math.log(8.0)
    ratio = (math.sqrt(5) - 1) / 2
    x1, x2 = high - ratio * (high - low), low + ratio * (high - low)
    f1, f2 = nll(rows, math.exp(x1)), nll(rows, math.exp(x2))
    for _ in range(90):
        if f1 <= f2:
            high, x2, f2 = x2, x1, f1
            x1 = high - ratio * (high - low)
            f1 = nll(rows, math.exp(x1))
        else:
            low, x1, f1 = x1, x2, f2
            x2 = low + ratio * (high - low)
            f2 = nll(rows, math.exp(x2))
    return math.exp((low + high) / 2)


def ece15(rows: list[dict]) -> float:
    total = len(rows)
    error = 0.0
    for bin_index in range(15):
        subset = []
        for row in rows:
            confidence = max(row["probabilities"])
            bucket = min(14, int(confidence * 15))
            if bucket == bin_index:
                subset.append(row)
        if subset:
            accuracy = sum(row["probabilities"].index(max(row["probabilities"])) == row["label"]
                           for row in subset) / len(subset)
            mean_confidence = sum(max(row["probabilities"]) for row in subset) / len(subset)
            error += len(subset) / total * abs(accuracy - mean_confidence)
    return error


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--crossfit-predictions", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.crossfit_predictions.exists():
        raise FileExistsError("Calibration output already exists: " + str(args.output))
    labels = json.loads(args.manifest_labels.read_text(encoding="utf-8"))
    expected = {row["ID"]: int(row["class_index"]) for row in labels}
    records = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != len(expected) or {row["ID"] for row in records} != set(expected):
        raise ValueError("Prediction IDs differ from frozen validation manifest")
    if len(expected) != 647:
        raise ValueError("Expected the official CH-SIMS validation count of 647")
    for row in records:
        if (row["split"] != "valid" or row["label"] != expected[row["ID"]] or
                tuple(row["class_order"]) != CLASSES):
            raise ValueError("A prediction row disagrees with the frozen protocol")
        probabilities = row["probabilities"]
        if len(probabilities) != 5 or any(not math.isfinite(x) or x < 0 for x in probabilities) or abs(sum(probabilities) - 1) > 1e-5:
            raise ValueError("Invalid probability row")
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(row["ID"].split("/")[0], []).append(row)
    order = sorted(groups, key=lambda key: (-len(groups[key]), hashlib.sha256(key.encode()).hexdigest()))
    folds: list[list[dict]] = [[] for _ in range(5)]
    for group in order:
        folds[min(range(5), key=lambda index: len(folds[index]))].extend(groups[group])
    calibrated = {}
    fold_reports = []
    for held_index, held in enumerate(folds):
        fitting = [row for index, fold in enumerate(folds) if index != held_index for row in fold]
        fitting_groups = {row["ID"].split("/")[0] for row in fitting}
        held_groups = {row["ID"].split("/")[0] for row in held}
        if fitting_groups & held_groups:
            raise ValueError("Calibration fold source-group leakage")
        temperature = fit_temperature(fitting)
        for row in held:
            calibrated[row["ID"]] = {"ID": row["ID"], "label": row["label"],
                                     "probabilities": scaled(row["probabilities"], temperature),
                                     "temperature": temperature, "held_fold": held_index}
        fold_reports.append({"fold": held_index, "fit_count": len(fitting), "held_count": len(held),
                             "fit_groups": len(fitting_groups), "held_groups": len(held_groups),
                             "temperature": temperature})
    held_rows = [calibrated[row["ID"]] for row in records]
    original_accuracy = sum(row["probabilities"].index(max(row["probabilities"])) == row["label"]
                            for row in records) / len(records)
    held_accuracy = sum(row["probabilities"].index(max(row["probabilities"])) == row["label"]
                        for row in held_rows) / len(held_rows)
    if held_accuracy != original_accuracy:
        raise ValueError("Temperature scaling unexpectedly changed an argmax")
    final_temperature = fit_temperature(records)
    output = {"scope": "official validation selection set; no test scoring",
              "limitation": "The model checkpoint was selected on this validation set before calibration cross-fitting",
              "prediction_sha256": sha(args.predictions), "manifest_labels_sha256": sha(args.manifest_labels),
              "source_group_disjoint_calibration_folds": True, "folds": fold_reports,
              "original_accuracy": original_accuracy,
              "original_nll": nll(records, 1.0), "original_ece15": ece15(records),
              "crossfit_accuracy": held_accuracy, "crossfit_nll": nll(held_rows, 1.0),
              "crossfit_ece15": ece15(held_rows),
              "final_inference_temperature_fit_on_all_validation": final_temperature,
              "test_labels_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    args.crossfit_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.crossfit_predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in held_rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "original_nll": output["original_nll"],
                      "crossfit_nll": output["crossfit_nll"], "original_ece15": output["original_ece15"],
                      "crossfit_ece15": output["crossfit_ece15"], "final_temperature": final_temperature}))


if __name__ == "__main__":
    main()
