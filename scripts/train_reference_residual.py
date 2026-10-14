"""Run four annotation-reference cases serially with three native seeds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.reference_training import train_reference
from jevaffect.reference_residual import REFERENCE_CASES
from jevaffect.native_training import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--case", choices=REFERENCE_CASES)
    p.add_argument("--seed", type=int, choices=(13, 42, 2026)); p.add_argument("--smoke", type=int)
    p.add_argument("--stop-after", type=int); p.add_argument("--resume", action="store_true"); a = p.parse_args()
    config = json.loads(a.config.read_text()); results = []
    for case in ([a.case] if a.case else REFERENCE_CASES):
        for seed in ([a.seed] if a.seed else config["seeds"]):
            results.append(train_reference(config, case, seed, a.resume, a.smoke, a.stop_after))
            save_json(a.output, {"results": results, "suite_complete": False, "smoke_only": a.smoke is not None})
    save_json(a.output, {"results": results, "suite_complete": all(r["complete"] for r in results), "smoke_only": a.smoke is not None})
