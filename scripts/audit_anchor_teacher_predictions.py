"""Recompute all 48 selected teacher outputs on CPU without TEST access."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import (prepare, PooledView, VIEWS, _loader,
    _predict_targets, _predict_features, _read_predictions)
from jevaffect.native_training import PooledRegressor, digest, save_json
from jevaffect.source_crossfit import subset


def compare_records(actual, expected, tolerance=1e-6):
    if [r["id"] for r in actual] != [r["id"] for r in expected]:
        raise ValueError("Teacher reloaded prediction ID/order mismatch")
    if any(a.get("target") != b.get("target") for a, b in zip(actual, expected)):
        raise ValueError("Teacher reloaded original M target mismatch")
    delta = max(abs(a["prediction"]-b["prediction"]) for a, b in zip(actual, expected))
    if delta > tolerance:
        raise ValueError(f"Teacher CPU/GPU prediction mismatch: {delta}")
    return delta


def audit(suite_path, config_path):
    torch.set_num_threads(4)
    cfg = json.loads(Path(config_path).read_text())
    suite = json.loads(Path(suite_path).read_text())
    expected = {(v, s, f) for v in VIEWS for s in (13,42,2026) for f in (0,1,2,"full")}
    if not suite["suite_complete"] or len(suite["results"]) != 48 or {(r["view"],r["seed"],r["fold"]) for r in suite["results"]} != expected:
        raise ValueError("Complete 48-teacher suite required")
    data = prepare(cfg)
    evidence = []
    for result in suite["results"]:
        root = Path(result["root"])
        if digest(root/"best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Selected teacher checkpoint changed")
        state = torch.load(root/"best.pt", map_location="cpu", weights_only=False, mmap=True)
        identity = json.loads((root/"identity.json").read_text())
        if state["identity"] != identity or identity["config"] != cfg:
            raise ValueError("Teacher reloaded identity differs")
        mods = VIEWS[result["view"]]
        model = PooledRegressor({m:data["train"].features[m].shape[-1] for m in mods}, cfg["hidden"],cfg["dropout"])
        model.load_state_dict(state["model"])
        cohort = identity["cohort"]
        selection, _ = _predict_targets(model, _loader(subset(PooledView(data["train"],mods),cohort["select_ids"]),cfg), "cpu")
        selection_delta = abs(selection["MAE"]-state["metrics"]["MAE"])
        if selection_delta > 1e-6:
            raise ValueError("Selected teacher inner-MAE mismatch")
        if result["fold"] == "full":
            _, actual = _predict_targets(model,_loader(PooledView(data["train"],mods),cfg),"cpu")
            path = Path(result["full_train_predictions_file"])
            if digest(path) != result["train_predictions_sha256"]:
                raise ValueError("Full teacher TRAIN predictions changed")
            train_delta = compare_records(actual,_read_predictions(path,with_target=True))
            actual_valid = _predict_features(model,_loader(PooledView(data["valid_features"],mods),cfg),"cpu")
            path = Path(result["valid_predictions_file"])
            if digest(path) != result["valid_predictions_sha256"]:
                raise ValueError("Teacher label-free VALID predictions changed")
            valid_delta = compare_records(actual_valid,_read_predictions(path,with_target=False))
            prediction_delta = max(train_delta,valid_delta)
        else:
            _, actual = _predict_targets(model,_loader(subset(PooledView(data["train"],mods),cohort["oof_ids"]),cfg),"cpu")
            path = Path(result["predictions_file"])
            if digest(path) != result["predictions_sha256"]:
                raise ValueError("OOF predictions changed")
            prediction_delta = compare_records(actual,_read_predictions(path,with_target=True))
        evidence.append({"view":result["view"],"seed":result["seed"],"fold":result["fold"],
            "cpu_gpu_prediction_max_absolute_difference":prediction_delta,"inner_selection_MAE_difference":selection_delta})
    return {"suite_sha256":digest(suite_path), "config_sha256":digest(config_path),
        "evidence":evidence,"passed":True,"cpu_gpu_tolerance":1e-6,
        "official_VALID_labels_parsed":False,"test_evaluated":False,
        "scope":"CPU checkpoint reload and point predictions; source/config/history/hash audit is separately bound in calibration table metadata"}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",type=Path,required=True);p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();report=audit(a.suite,a.config);save_json(a.output,report)
    print(json.dumps({"teachers":len(report["evidence"]),"passed":True,
        "max_prediction_difference":max(r["cpu_gpu_prediction_max_absolute_difference"] for r in report["evidence"])}))
