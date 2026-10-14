"""Train continuous sentiment baselines from author annotations."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(13, 42, 2026))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", type=int, help="Prefix samples per split; flow check only")
    parser.add_argument("--stop-after", type=int, help="Epochs this invocation, for resume checks")
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after must be positive")
    config = json.loads(args.config.read_text())
    results = [train(config, s, args.resume, args.smoke, args.stop_after) for s in ([args.seed] if args.seed else config["seeds"])]
    print(json.dumps(results, indent=2), flush=True)
