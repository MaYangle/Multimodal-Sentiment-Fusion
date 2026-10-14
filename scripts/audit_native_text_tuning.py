"""Audit six full native MacBERT adaptation controls without scoring TEST."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.native_text_tuning import parse_author_csv, verify_pinned_pretrained, _canonical
from jevaffect.native_training import digest, save_json
from jevaffect.native_metrics import native_metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); suite = json.loads(a.suite.read_text())
    expected = {(mode, seed) for mode in ("top4", "frozen") for seed in (13, 42, 2026)}
    if not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"]) != 6 or {(r["mode"], r["seed"]) for r in suite["results"]} != expected:
        raise ValueError("Complete six-run native MacBERT suite required")
    summaries = {}; evidence = []
    for result in suite["results"]:
        root = Path(result["root"]); identity = json.loads((root / "identity.json").read_text()); config = identity["config"]
        identity_sha = hashlib.sha256(_canonical(identity)).hexdigest()
        if identity_sha != result["identity_sha256"] or identity["mode"] != result["mode"] or identity["seed"] != result["seed"]:
            raise ValueError("Text experiment identity changed")
        project = Path(__file__).resolve().parents[1]
        for name, expected_sha in identity["implementation"].items():
            if digest(project / name) != expected_sha:
                raise ValueError("Text implementation changed: " + name)
        csv_path = Path(config["author_csv"])
        csv_path = csv_path if csv_path.is_absolute() else project / csv_path
        records, cohort = parse_author_csv(csv_path)
        model_dir = Path(config["pretrained_dir"])
        model_dir = model_dir if model_dir.is_absolute() else project / model_dir
        if cohort != identity["source"]["cohort"] or verify_pinned_pretrained(model_dir) != identity["source"]["model_tokenizer"]:
            raise ValueError("Author/model/tokenizer source changed")
        predictions = json.loads((root / "valid_predictions.json").read_text())
        if [r["id"] for r in predictions] != [r["id"] for r in records["valid"]] or not np.array_equal(np.array([r["target"] for r in predictions], np.float32), np.array([r["target"] for r in records["valid"]], np.float32)):
            raise ValueError("Text native VALID IDs/targets differ")
        metrics = native_metrics(np.array([r["prediction"] for r in predictions], np.float32), np.array([r["target"] for r in predictions], np.float32), "simsv2")
        if metrics != result["valid_metrics"] or digest(root / "valid_predictions.json") != result["predictions_sha256"] or digest(root / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Text scores/prediction/checkpoint hash differ")
        history = json.loads((root / "history.json").read_text()); selected = min(history, key=lambda row: row["valid"]["MAE"])
        if selected["epoch"] != result["best_epoch"] or len(history) != result["last_epoch"] or not result["complete"] or result["smoke_only"] or result["test_evaluated"]:
            raise ValueError("Text selection/budget/scope differs")
        steps = math.ceil(result["train_count"] / config["batch_size"])
        if result["optimizer_updates"] != steps * len(history) or any(row["optimizer_updates"] != steps * row["epoch"] for row in history):
            raise ValueError("Text optimizer update counts differ")
        state = torch.load(root / "best.pt", map_location="cpu", weights_only=False, mmap=True)
        if state["identity"] != identity or state["identity_sha256"] != identity_sha or state["epoch"] != result["best_epoch"] or state["valid_metrics"] != metrics:
            raise ValueError("Selected native text checkpoint differs")
        actual_count = sum(value.numel() for value in state["model"].values())
        # State dict includes parameter tensors only for pinned BERT/head.
        if actual_count != result["total_parameters"] or result["trainable_parameters"]["head"] != 769 or (result["mode"] == "frozen" and result["trainable_parameters"]["encoder"] != 0):
            raise ValueError("Text architecture/gradient eligibility count differs")
        del state
        summaries.setdefault(result["mode"], []).append(result)
        evidence.append({"mode": result["mode"], "seed": result["seed"], "id_target_metric_hash_epoch_checkpoint_audit_passed": True})
    summary = {}
    for mode, rows in summaries.items():
        summary[mode] = {key: {"mean": float(np.mean([r["valid_metrics"][key] for r in rows])),
                              "sample_sd": float(np.std([r["valid_metrics"][key] for r in rows], ddof=1))}
                         for key in rows[0]["valid_metrics"] if key not in ("count", "weak_count")}
        summary[mode].update(total_parameters=rows[0]["total_parameters"], trainable_parameters=rows[0]["trainable_parameters"],
            epochs=[r["last_epoch"] for r in rows], updates=[r["optimizer_updates"] for r in rows],
            peak_cuda_allocated_bytes=max(r["peak_cuda_allocated_bytes"] for r in rows),
            total_loop_seconds=sum(r["loop_seconds"] for r in rows))
    report = {"scope": "native official CH-SIMS VALID development/selection results; known text adaptation baseline",
              "suite_sha256": digest(a.suite), "summary": summary, "evidence": evidence,
              "test_evaluated": False, "original_contribution_claimed": False}
    a.output.parent.mkdir(parents=True, exist_ok=True); save_json(a.output, report)
    print(json.dumps(summary, indent=2))
