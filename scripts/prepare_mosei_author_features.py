"""Resolve caller paths and run the frozen V1 author-feature preparation recipe."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trust-author-pickle", action="store_true")
    a = p.parse_args()
    source, receipt, output = a.source.resolve(), a.receipt.resolve(), a.output_dir.resolve()
    if (output / "metadata.json").exists():
        existing = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        if existing.get("normalization") != "none" or existing.get("text_pooling") != "valid_mean":
            raise ValueError("The raw recipe cannot reuse a cache with different preprocessing")
    project = Path(__file__).resolve().parents[1]
    recipe = project / "scripts/prepare_mosei_author_features_v1.py"
    sys.argv = [str(recipe), "--source", str(source), "--receipt", str(receipt), "--output-dir", str(output)]
    if a.trust_author_pickle:
        sys.argv.append("--trust-author-pickle")
    os.chdir(project)
    runpy.run_path(str(recipe), run_name="__main__")
