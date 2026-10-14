"""Frozen native feature sensitivity, with explicit masks for corrected ALMT."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from jevaffect.author_features import AuthorFeatureSplit, MODALITIES
from jevaffect.native_predict import load_model
from jevaffect.native_training import evaluate, digest, save_json

CONDITIONS = ("identity", "joint_permute", "av_permute", "reverse_all", "front_padding")


class SequenceView(Dataset):
    def __init__(self, source, condition, modalities, pooled):
        self.source, self.condition, self.modalities, self.pooled = source, condition, modalities, pooled

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        features, target, sid = self.source[i]
        seed = int.from_bytes(hashlib.sha256((sid + "|105").encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed); n = int(features["lengths"]["text"])
        common = rng.permutation(n); output = {}
        for m in self.modalities:
            x = features[m].numpy(); selected = x[:n].copy()
            if self.condition == "joint_permute":
                selected = selected[common]
            elif self.condition == "av_permute" and m in ("audio", "vision"):
                selected = selected[rng.permutation(n)]
            elif self.condition == "reverse_all":
                selected = selected[::-1].copy()
            if self.pooled:
                # True valid-span pooling, including front-padding control.
                output[m] = torch.from_numpy(selected.mean(0).astype(np.float32))
            else:
                value = np.zeros_like(x)
                start = len(x) - n if self.condition == "front_padding" else 0
                value[start:start + n] = selected
                output[m] = torch.from_numpy(value)
        if not self.pooled:
            output["lengths"] = {m: features["lengths"][m] for m in self.modalities}
            output["valid_masks"] = {}
            for m in self.modalities:
                mask = torch.zeros(len(features[m]), dtype=torch.bool)
                start = len(mask) - n if self.condition == "front_padding" else 0
                mask[start:start + n] = True
                output["valid_masks"][m] = mask
        return output, target, sid


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"))
    p.add_argument("--models", nargs="+", choices=("text", "early_fusion", "almt", "almt_masked"), default=["text", "early_fusion", "almt"])
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    pairs = {(m, seed) for m in a.models for seed in (13, 42, 2026)}
    if len(set(a.models)) != len(a.models) or not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"]) != len(pairs) or {(r["model"], r["seed"]) for r in suite["results"]} != pairs:
        raise ValueError("Complete registered model/three-seed suite required")
    config = json.loads((Path(suite["results"][0]["root"]) / "identity.json").read_text())["config"]
    torch.set_num_threads(4); device = torch.device(a.device or config["device"])
    source = AuthorFeatureSplit(config["cache_dir"], "valid", pooled=False)
    report = {"scope": "frozen VALID feature sensitivity, not raw robustness or causal intervention", "test_evaluated": False,
              "suite_sha256": digest(a.suite), "implementation_sha256": digest(__file__), "device": str(device), "results": []}
    try:
        for result in suite["results"]:
            root = Path(result["root"])
            if digest(root / "best.pt") != result["checkpoint_sha256"] or digest(root / "valid_predictions.json") != result["predictions_sha256"]:
                raise ValueError("Frozen artifact changed")
            spec = load_model(root / "best.pt", config["cache_dir"], device)
            model = spec["model"]; reference = json.loads((root / "valid_predictions.json").read_text())
            row = {"model": result["model"], "seed": result["seed"], "conditions": {}}
            for condition in CONDITIONS:
                view = SequenceView(source, condition, spec["modalities"], spec["pooled"])
                metrics, predictions = evaluate(model, DataLoader(view, batch_size=config["batch_size"], shuffle=False), device, "mosei")
                if [r["id"] for r in predictions] != source.ids:
                    raise ValueError("Sensitivity IDs changed")
                difference = max(abs(r["prediction"] - ref["prediction"]) for r, ref in zip(predictions, reference))
                if condition == "identity" and difference > 1e-5:
                    raise ValueError("Identity differs from frozen checkpoint predictions")
                if spec["pooled"] and difference > 1e-5:
                    raise ValueError("Pooled model changed under multiset-preserving transform")
                if result["model"] == "almt_masked" and condition == "front_padding" and difference > 1e-5:
                    raise ValueError("Corrected ALMT changed under equivalent padding layout")
                path = root / ("sequence_" + condition + ".json"); save_json(path, predictions)
                row["conditions"][condition] = {"native": metrics, "max_prediction_change": difference, "predictions_sha256": digest(path)}
            report["results"].append(row); save_json(a.output, report)
            print(json.dumps({"model": row["model"], "seed": row["seed"], "MAE": {k: v["native"]["MAE"] for k, v in row["conditions"].items()}}), flush=True)
        report["complete"] = True; save_json(a.output, report)
    finally:
        source.close()
