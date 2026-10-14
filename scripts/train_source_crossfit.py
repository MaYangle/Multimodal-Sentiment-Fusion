"""Serial native TRAIN source-group cross-fit; no official validation scoring."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.crossfit_training import prepare, train_fold
from jevaffect.native_training import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true"); p.add_argument("--smoke-groups", type=int)
    p.add_argument("--stop-after", type=int); p.add_argument("--seed", type=int, choices=(13, 42, 2026))
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    dataset, manifest = prepare(config, a.smoke_groups)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(a.output.with_suffix(".manifest.json"), manifest)
    report = {"manifest": manifest, "results": [], "suite_complete": False, "smoke_only": a.smoke_groups is not None,
              "official_validation_labels_parsed": False, "test_evaluated": False}
    for seed in ([a.seed] if a.seed else config["seeds"]):
        for fold in range(config["fold_count"]):
            report["results"].append(train_fold(config, dataset, manifest, seed, fold, a.resume, a.smoke_groups, a.stop_after))
            save_json(a.output, report)
    report["suite_complete"] = all(r["complete"] for r in report["results"]); save_json(a.output, report)
