"""Train the four matched-budget native timed-moment controls serially."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.timed_moment_training import CASES, FORMAL_SEEDS, digest, run_matrix


def _load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    for key in ("author_csv", "cache_dir", "normalization_cache_dir", "run_root"):
        value = Path(config[key])
        if not value.is_absolute():
            config[key] = str((root / value).resolve())
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Override the config run_root")
    parser.add_argument("--case", choices=CASES, help="Run one case; defaults to all four")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, help="Run one seed; defaults to all three")
    parser.add_argument("--smoke32", action="store_true", help="Flow check using at most 32 rows per split")
    parser.add_argument("--stop-after", type=int, help="Epochs per case/seed in this invocation, for resume checks")
    parser.add_argument("--resume", action="store_true", help="Resume existing case/seed checkpoints")
    args = parser.parse_args(argv)
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    config = _load_config(args.config)
    output = args.output.resolve() if args.output else None
    matrix = run_matrix(config,
                        cases=[args.case] if args.case else None,
                        seeds=[args.seed] if args.seed else None,
                        output_dir=output,
                        resume=args.resume,
                        smoke32=args.smoke32,
                        stop_after=args.stop_after,
                        config_sha256=digest(args.config))
    print(json.dumps(matrix, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
