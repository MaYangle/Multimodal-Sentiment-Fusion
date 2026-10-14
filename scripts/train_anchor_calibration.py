"""Fit registered native probability-shape controls on TRAIN OOF records."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_calibration_training import fit
from jevaffect.native_training import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case", choices=("global", "scale", "mixture", "views", "disagreement", "direct_hit", "mdn"))
    p.add_argument("--seed", type=int, choices=(13,42,2026)); p.add_argument("--stop-after", type=int)
    p.add_argument("--resume", action="store_true"); p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.stop_after is not None and a.stop_after < 1:
        p.error("Positive epoch limit required")
    cfg = json.loads(a.config.read_text()); results = []; a.output.parent.mkdir(parents=True, exist_ok=True)
    for case in ([a.case] if a.case else cfg["cases"]):
        for seed in ([a.seed] if a.seed else cfg["seeds"]):
            results.append(fit(cfg, case, seed, a.resume, a.stop_after, a.smoke))
            save_json(a.output, {"results": results, "suite_complete": False, "smoke_only": a.smoke})
    save_json(a.output, {"results": results, "suite_complete": all(r["complete"] for r in results), "smoke_only": a.smoke})
