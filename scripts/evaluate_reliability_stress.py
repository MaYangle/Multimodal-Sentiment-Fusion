"""Shared validation-only modality stress and native uncertainty evaluation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from jevaffect.native_training import author_rows, digest, save_json
from jevaffect.reliability_training import ReliabilitySplit, evaluate, MODALITIES


def transformation(name, std):
    keep = None
    if name.startswith("keep_"):
        keep = set(name.removeprefix("keep_").split("+"))
    else:
        _, modality, strength = name.split(":")
        strength = float(strength)

    def apply(features, ids):
        result = {m: features[m].clone() for m in MODALITIES}
        available = features["availability"].clone()
        if keep is not None:
            for i, m in enumerate(MODALITIES):
                if m not in keep:
                    available[:, i] = False
                    result[m].zero_()
        else:
            perturbations = []
            for sid in ids:
                seed = int.from_bytes(hashlib.sha256(("105:" + name + ":" + sid).encode()).digest()[:8], "little")
                noise = np.random.default_rng(seed).normal(size=std[modality].shape)
                perturbations.append((noise * std[modality] * strength).astype(np.float32))
            result[modality] += torch.from_numpy(np.stack(perturbations)).to(result[modality].device)
        result["availability"] = available
        return result
    return apply


def main():
    from jevaffect.reliability import ReliabilityRegressor
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if not suite["suite_complete"] or suite["smoke_only"]:
        raise ValueError("Only completed formal suite accepted")
    first = suite["results"][0]
    config = json.loads((Path(first["root"]) / "identity.json").read_text())["config"]
    rows = author_rows(config["author_csv"], config["dataset"])
    train = ReliabilitySplit(config["cache_dir"], "train", rows, config["author_csv"])
    valid = ReliabilitySplit(config["cache_dir"], "valid", rows, config["author_csv"])
    std = {m: train.features[m].std(0).astype(np.float32) for m in MODALITIES}
    dims = {m: train.features[m].shape[-1] for m in MODALITIES}
    conditions = ["keep_" + "+".join(m for i, m in enumerate(MODALITIES) if mask & (1 << i)) for mask in range(1, 8)]
    conditions += [f"noise:{m}:{strength}" for m in ("audio", "vision") for strength in (.5, 1.)]
    device = torch.device(config["device"])
    torch.set_num_threads(4)
    loader = DataLoader(valid, batch_size=config["batch_size"], shuffle=False)
    report = {"scope": "validation stress diagnostic, not frozen test", "test_evaluated": False,
              "suite_sha256": digest(a.suite), "shared_corruption_seed": 105, "noise_scale": "per-feature training pooled STD",
              "corruption_domain": "pooled features; not raw waveform or video", "results": []}
    for case in suite["results"]:
        root = Path(case["root"])
        if digest(root / "best.pt") != case["checkpoint_sha256"]:
            raise ValueError("Changed checkpoint")
        checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=False)
        identity = checkpoint["identity"]
        if identity["config"] != config or identity["feature_hashes"] != {"train": train.hashes, "valid": valid.hashes}:
            raise ValueError("Stress dataset/config differs from training")
        fusion = "dynamic" if case["condition"].endswith("G1") else "uniform"
        model = ReliabilityRegressor(dims, hidden=config["hidden"], dropout=config["dropout"], fusion=fusion).to(device)
        model.load_state_dict(checkpoint["model"])
        row = {"condition": case["condition"], "seed": case["seed"], "conditions": {}}
        for name in conditions:
            metrics, predictions = evaluate(model, loader, device, config["dataset"], transformation(name, std))
            row["conditions"][name] = metrics
            # Clean and corrupt per-ID distributions remain recoverable outside Git.
            save_json(root / ("stress_" + name.replace(":", "_").replace("+", "_") + ".json"), predictions)
        report["results"].append(row)
        save_json(a.output, report)
        print(json.dumps({"condition": case["condition"], "seed": case["seed"],
                          "MAE": {n: m["native"]["MAE"] for n, m in row["conditions"].items()}}), flush=True)
    report["complete"] = True
    save_json(a.output, report)


if __name__ == "__main__":
    main()
