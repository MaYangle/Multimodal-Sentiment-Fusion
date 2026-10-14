"""Run registered source-attributed context-NIG cases serially."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.context_nig import CASES
from jevaffect.context_training import train_case
from jevaffect.native_training import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--case", choices=CASES)
    p.add_argument("--seed", type=int, choices=(13, 42, 2026))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", type=int)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if (a.smoke is not None and a.smoke < 1) or (a.stop_after is not None and a.stop_after < 1):
        p.error("Smoke limit and stop-after must be positive")
    config = json.loads(a.config.read_text()); results = []
    for case in ([a.case] if a.case else CASES):
        for seed in ([a.seed] if a.seed else config["seeds"]):
            results.append(train_case(config, case, seed, a.resume, a.smoke, a.stop_after))
            if a.output:
                save_json(a.output, {"results": results, "suite_complete": False, "smoke_only": a.smoke is not None})
    if a.output:
        save_json(a.output, {"results": results, "suite_complete": all(r["complete"] for r in results), "smoke_only": a.smoke is not None})
    print(json.dumps(results, indent=2))
