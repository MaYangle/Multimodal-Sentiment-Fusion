"""Compare a fresh full author-feature suite against recorded native baselines."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reproduced", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-root", type=Path)
    p.add_argument("--reproduced-root", type=Path)
    a = p.parse_args()
    suites = [json.loads(path.read_text()) for path in (a.reference, a.reproduced)]
    expected = {(model, seed) for model in ("text", "early_fusion", "almt") for seed in (13, 42, 2026)}
    mappings = []
    roots = [a.reference_root or a.reference.resolve().parent.parent,
             a.reproduced_root or a.reproduced.resolve().parent.parent]
    for suite in suites:
        rows = suite["results"]
        if not suite["suite_complete"] or suite["smoke_only"] or len(rows) != 9 or {(r["model"], r["seed"]) for r in rows} != expected:
            raise ValueError("Two complete nine-run formal native suites required")
        mappings.append({(r["model"], r["seed"]): r for r in rows})
    evidence = []
    for pair in sorted(expected):
        left, right = (mapping[pair] for mapping in mappings)
        identities, predictions = [], []
        for row, base in zip((left, right), roots):
            root = Path(row["root"])
            if not root.is_absolute():
                root = base / root
            if digest(root / "best.pt") != row["checkpoint_sha256"] or digest(root / "valid_predictions.json") != row["predictions_sha256"] or not row["complete"] or row["test_evaluated"]:
                raise ValueError("Changed or incomplete reproduced artifact")
            identities.append(json.loads((root / "identity.json").read_text()))
            predictions.append(json.loads((root / "valid_predictions.json").read_text()))
        configs = [{k: v for k, v in identity["config"].items() if k not in ("cache_dir", "run_root")} for identity in identities]
        run_roots = []
        for identity, base in zip(identities, roots):
            path = Path(identity["config"]["run_root"])
            run_roots.append((path if path.is_absolute() else base / path).resolve())
        if run_roots[0] == run_roots[1]:
            raise ValueError("Reference and reproduced suites must be independently trained run roots")
        if configs[0] != configs[1] or identities[0]["implementation"] != identities[1]["implementation"]:
            raise ValueError("Numerical configuration or implementation differs")
        versions_equal = all(identities[0][k] == identities[1][k] for k in ("torch", "numpy", "device", "gpu", "source_sha256"))
        if [(r["id"], r["target"]) for r in predictions[0]] != [(r["id"], r["target"]) for r in predictions[1]]:
            raise ValueError("Reproduction VALID cohort or targets differ")
        difference = max(abs(x["prediction"] - y["prediction"]) for x, y in zip(*predictions))
        mae_difference = right["valid_metrics"]["MAE"] - left["valid_metrics"]["MAE"]
        selection_budget_equal = all(left[k] == right[k] for k in ("best_epoch", "epochs_executed", "optimizer_updates", "parameters", "train_count", "valid_count"))
        passed = versions_equal and selection_budget_equal and difference <= 1e-5 and abs(mae_difference) <= 1e-6
        evidence.append({"model": pair[0], "seed": pair[1], "max_prediction_difference": difference,
            "MAE_difference": mae_difference, "versions_equal": versions_equal,
            "selection_budget_equal": selection_budget_equal, "passed": passed})
    report = {"scope": "nine full native MOSEI feature baseline reproductions; not original-method/raw-media/final-test reproduction",
        "reference_suite_sha256": digest(a.reference), "reproduced_suite_sha256": digest(a.reproduced),
        "prediction_tolerance": 1e-5, "MAE_tolerance": 1e-6, "results": evidence,
        "all_passed": all(row["passed"] for row in evidence), "test_evaluated": False}
    save_json(a.output, report); print(json.dumps(report, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)
