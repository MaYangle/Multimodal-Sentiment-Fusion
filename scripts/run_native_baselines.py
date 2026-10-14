"""Run the fixed baseline suite serially to avoid competing for GPU memory."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import train, save_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for file in args.configs:
        config = json.loads(file.read_text())
        for seed in config["seeds"]:
            result = train(config, seed, resume=args.resume)
            results.append(result)
            save_json(args.output, {"results": results, "suite_complete": False})
    save_json(args.output, {"results": results, "suite_complete": all(r["complete"] for r in results)})
