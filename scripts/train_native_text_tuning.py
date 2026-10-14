"""Run the pinned CH-SIMS v2 native MacBERT text tuning controls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevaffect.native_text_tuning import MODES, SEEDS, train
from jevaffect.native_training import save_json


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/native_simsv2_text_tuning_v1.json")
    parser.add_argument("--output", type=Path, required=True, help="Suite JSON result file")
    parser.add_argument("--run-root", type=Path, help="Optional override of checkpoint/run directory")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-limit", type=int,
                        help="Train/evaluate only the first N official train and valid rows")
    parser.add_argument("--stop-after", type=int,
                        help="Stop after N epochs in this call; resume with the same identity")
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.run_root is not None:
        config["run_root"] = str(args.run_root.resolve())
    modes = (args.mode,) if args.mode else tuple(config["modes"])
    seeds = (args.seed,) if args.seed else SEEDS
    if len(set(modes)) != len(modes):
        parser.error("Duplicate registered mode")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for mode in modes:
        for seed in seeds:
            results.append(train(
                config, seed, mode=mode, resume=args.resume,
                smoke_limit=args.smoke_limit, stop_after=args.stop_after,
            ))
            save_json(args.output, {"results": results, "suite_complete": False, "smoke_only": args.smoke_limit is not None})
    report = {"results": results, "suite_complete": all(r["complete"] for r in results), "smoke_only": args.smoke_limit is not None}
    save_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
