"""Summarize paired validation stress effects and actual gate responses."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stress", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    stress, suite = json.loads(a.stress.read_text()), json.loads(a.suite.read_text())
    if not stress.get("complete") or stress["test_evaluated"] or digest(a.suite) != stress["suite_sha256"]:
        raise ValueError("Incomplete or mismatched stress experiment")
    conditions = {}; effects = []; gates = []
    indexed = {(r["condition"], r["seed"]): r for r in stress["results"]}
    names = list(stress["results"][0]["conditions"])
    for condition in ("S0G0", "S0G1", "S1G0", "S1G1"):
        conditions[condition] = {}
        for name in names:
            values = [indexed[condition, seed]["conditions"][name]["native"]["MAE"] for seed in (13, 42, 2026)]
            coverage = [indexed[condition, seed]["conditions"][name]["uncertainty"]["intervals"]["0.9"]["coverage"] for seed in (13, 42, 2026)]
            conditions[condition][name] = {"MAE_mean": float(np.mean(values)), "MAE_sample_sd": float(np.std(values, ddof=1)), "coverage90_mean": float(np.mean(coverage))}
    for name in names:
        by_seed = []
        for seed in (13, 42, 2026):
            mae = {c: indexed[c, seed]["conditions"][name]["native"]["MAE"] for c in conditions}
            by_seed.append({"seed": seed, "without_aux_G1_minus_G0": mae["S0G1"] - mae["S0G0"],
                            "with_aux_G1_minus_G0": mae["S1G1"] - mae["S1G0"],
                            "interaction": mae["S1G1"] - mae["S1G0"] - mae["S0G1"] + mae["S0G0"]})
        effects.append({"stress_condition": name, "paired_seeds": by_seed})
    for case in suite["results"]:
        root = Path(case["root"])
        clean = json.loads((root / "valid_predictions.json").read_text())
        ids = [r["id"] for r in clean]
        clean_weights = np.array([r["weights"] for r in clean])
        for modality, index in (("audio", 1), ("vision", 2)):
            name = f"noise:{modality}:1.0"
            records = json.loads((root / ("stress_" + name.replace(":", "_") + ".json")).read_text())
            if [r["id"] for r in records] != ids:
                raise ValueError("Stress/clean record order differs")
            weights = np.array([r["weights"] for r in records])
            change = weights[:, index] - clean_weights[:, index]
            gates.append({"condition": case["condition"], "seed": case["seed"], "corrupted_modality": modality,
                          "clean_mean_weight": float(clean_weights[:, index].mean()), "noisy_mean_weight": float(weights[:, index].mean()),
                          "weight_change_mean": float(change.mean()), "fraction_weight_decreased": float((change < -1e-7).mean())})
    save_json(a.output, {"scope": "validation-only pooled-feature stress diagnostic", "test_evaluated": False,
                        "stress_sha256": digest(a.stress), "summary": conditions, "paired_seed_effects": effects,
                        "gate_responses": gates, "novel_contribution_verified": False,
                        "limitation": "Weights can respond to corruption without being calibrated reliability estimates; raw-media robustness and final-test gains remain unverified."})
    print(json.dumps(conditions, indent=2))


if __name__ == "__main__":
    main()
