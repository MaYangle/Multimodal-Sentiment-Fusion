"""Train the CH-SIMS AnchorCDF point/view teachers with source-disjoint CV."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import FORMAL_SEEDS, VIEWS, run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Suite JSON path")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--view", choices=tuple(VIEWS))
    parser.add_argument("--fold", choices=("0", "1", "2", "full"))
    parser.add_argument("--smoke-groups", type=int,
                        help="Train on the first N complete TRAIN source groups; N must be at least 8")
    parser.add_argument("--stop-after", type=int,
                        help="Stop each selected teacher after this many additional epochs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke_groups is not None and args.smoke_groups < 8:
        parser.error("--smoke-groups must be at least 8")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    fold = args.fold if args.fold == "full" or args.fold is None else int(args.fold)
    result = run_suite(
        config,
        output=args.output,
        seed=args.seed,
        view=args.view,
        fold=fold,
        smoke_groups=args.smoke_groups,
        stop_after=args.stop_after,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
