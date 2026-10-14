"""Common validation stress with the same frozen context-NIG checkpoints."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from jevaffect.context_nig import build_context_model
from jevaffect.context_training import DistributionSplit
from jevaffect.reliability_training import evaluate
from jevaffect.native_training import author_rows, digest, save_json
from evaluate_reliability_stress import transformation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    suite = json.loads(a.suite.read_text())
    if not suite["suite_complete"] or suite["smoke_only"]:
        raise ValueError("Incomplete formal suite")
    config = json.loads((Path(suite["results"][0]["root"]) / "identity.json").read_text())["config"]
    rows = author_rows(config["author_csv"], "simsv2")
    datasets = {s: DistributionSplit(config["cache_dir"], s, rows) for s in ("train", "valid")}
    modalities = ("text", "audio", "vision")
    std = {m: datasets["train"].features[m].std(0) for m in modalities}
    dims = {m: datasets["train"].features[m].shape[-1] for m in modalities}
    names = ["keep_" + "+".join(m for i, m in enumerate(modalities) if mask & (1 << i)) for mask in range(1, 8)]
    names += [f"noise:{m}:{s}" for m in ("audio", "vision") for s in (.5, 1.)]
    device = torch.device(config["device"]); torch.set_num_threads(4)
    loader = DataLoader(datasets["valid"], batch_size=config["batch_size"], shuffle=False)
    report = {"scope": "validation pooled-feature stress; no final test", "suite_sha256": digest(a.suite), "test_evaluated": False,
              "noise_seed": 105, "noise_scale": "training pooled per-feature STD", "results": []}
    for result in suite["results"]:
        root = Path(result["root"])
        if digest(root / "best.pt") != result["checkpoint_sha256"]:
            raise ValueError("Changed checkpoint")
        checkpoint = torch.load(root / "best.pt", map_location=device, weights_only=False)
        if checkpoint["identity"]["feature_hashes"] != {s: datasets[s].hashes for s in datasets}:
            raise ValueError("Stress input hashes differ")
        model = build_context_model(result["case"], dims, config).to(device)
        model.load_state_dict(checkpoint["model"])
        row = {"case": result["case"], "seed": result["seed"], "conditions": {}}
        for name in names:
            metrics, records = evaluate(model, loader, device, "simsv2", transformation(name, std))
            row["conditions"][name] = metrics
            save_json(root / ("stress_" + name.replace(":", "_").replace("+", "_") + ".json"), records)
        report["results"].append(row); save_json(a.output, report)
        print(json.dumps({"case": row["case"], "seed": row["seed"], "MAE": {n: r["native"]["MAE"] for n, r in row["conditions"].items()}}), flush=True)
    report["complete"] = True; save_json(a.output, report)


if __name__ == "__main__":
    main()
