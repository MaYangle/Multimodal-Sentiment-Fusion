"""Compare label-free feature inference with saved selected-model predictions."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import digest


def audit(cases):
    report = {"time_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "CPU feature inference vs saved GPU VALID predictions",
              "test_evaluated": False, "results": []}
    for case in cases:
        run, output = Path(case["run"]), Path(case["predictions"])
        known = {r["id"]: r for r in json.loads((run / "valid_predictions.json").read_text())}
        records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
        if not records or len({r["id"] for r in records}) != len(records):
            raise ValueError("Empty or duplicate inference records")
        checkpoint_hash = digest(run / "best.pt")
        errors = []
        for row in records:
            if row["id"] not in known or row["checkpoint_sha256"] != checkpoint_hash:
                raise ValueError("Unknown VALID ID or mismatched checkpoint")
            errors.append(abs(row["score_raw"] - known[row["id"]]["prediction"]))
            components = row.get("components", {})
            if components.get("decoder") == "additive":
                if abs(row["score_raw"] - components["text_reference"] - components["annotation_correction"]) > 1e-5:
                    raise ValueError("Additive inference components disagree")
        if max(errors) > 1e-5:
            raise ValueError("Inference differs from selected model outputs")
        report["results"].append({"name": case["name"], "count": len(records),
            "max_absolute_prediction_difference": max(errors),
            "checkpoint_sha256": checkpoint_hash, "inference_predictions_sha256": digest(output),
            "reference_predictions_sha256": digest(run / "valid_predictions.json")})
    report["all_passed"] = True
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = audit(json.loads(a.manifest.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
