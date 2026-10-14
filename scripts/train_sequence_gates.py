"""Train the staged native CH-SIMSv2 scalar sequence-gate study."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.sequence_gate_training import SEEDS, CASES, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Run root; smoke runs are placed in a smoke_N subdirectory")
    parser.add_argument("--stage", choices=("select", "refit", "fusion", "all"), default="all")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--smoke", type=int, help="Sample count per split on the configured device; never marks the formal matrix complete")
    parser.add_argument("--smoke32", action="store_true", help="32-row flow check on the configured device")
    parser.add_argument("--stop-after", type=int, help="Epochs this invocation before writing a resumable checkpoint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    if args.smoke is not None and args.smoke < 1:
        parser.error("--smoke must be positive")
    if args.smoke32 and args.smoke is not None:
        parser.error("Use either --smoke32 or --smoke N")
    smoke = 32 if args.smoke32 else args.smoke
    config = json.loads(args.config.read_text(encoding="utf-8"))
    results = train(
        config,
        stage=args.stage,
        seed=args.seed,
        case=args.case,
        config_path=args.config,
        output=args.output,
        smoke=smoke,
        stop_after=args.stop_after,
        resume=args.resume,
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
