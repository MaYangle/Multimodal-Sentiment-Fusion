"""Serial three-seed native MOSEI author-feature baselines."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.author_training import train_author
from jevaffect.native_training import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--config", type=Path, required=True)
    cases = p.add_mutually_exclusive_group()
    cases.add_argument("--model", choices=("text", "early_fusion", "almt"))
    cases.add_argument("--models", nargs="+", choices=("text", "early_fusion", "almt"))
    p.add_argument("--seed", type=int, choices=(13, 42, 2026))
    p.add_argument("--smoke", type=int); p.add_argument("--stop-after", type=int); p.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if (a.smoke is not None and a.smoke < 1) or (a.stop_after is not None and a.stop_after < 1):
        p.error("Positive limits required")
    base = json.loads(a.config.read_text()); results = []
    selected_models = a.models or ([a.model] if a.model else ("text", "early_fusion", "almt"))
    if len(set(selected_models)) != len(selected_models):
        p.error("Repeated model case")
    for model in selected_models:
        config = dict(base, model=model)
        for seed in ([a.seed] if a.seed else config["seeds"]):
            results.append(train_author(config, seed, a.resume, a.smoke, a.stop_after))
            save_json(a.output, {"results": results, "suite_complete": False, "smoke_only": a.smoke is not None})
    save_json(a.output, {"results": results, "suite_complete": all(r["complete"] for r in results), "smoke_only": a.smoke is not None})
