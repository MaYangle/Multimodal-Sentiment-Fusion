"""Train source-nested point anchors and export held-source shape tables."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.native_training import digest
from jevaffect.nested_shape_points import FORMAL_SEEDS, run_nested_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Suite JSON path; artifacts are written beside it")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS,
                        help="Select one point seed; omitted runs all three")
    parser.add_argument("--fold", choices=("0", "1", "2"),
                        help="Select one outer fold; omitted runs all three")
    parser.add_argument("--smoke-groups", type=int,
                        help="Use the first N complete TRAIN source groups; N must be at least eight")
    parser.add_argument("--stop-after", type=int,
                        help="Stop each selected point model after this many additional epochs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.smoke_groups is not None and args.smoke_groups < 8:
        parser.error("--smoke-groups must be at least eight")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_hash = digest(args.config)
    result = run_nested_suite(
        config,
        output=args.output,
        seed=args.seed,
        fold=None if args.fold is None else int(args.fold),
        smoke_groups=args.smoke_groups,
        stop_after=args.stop_after,
        resume=args.resume,
        config_sha256=config_hash,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
