"""Predict native continuous scores from label-free NPZ sequence features."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.native_predict import predict_features

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True); p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True); p.add_argument("--device", default="cpu")
    a = p.parse_args(); records = [json.loads(line) for line in a.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in records:
        if "target" in r or "label" in r:
            raise ValueError("Inference input must be label free")
        path = Path(r["features_npz"])
        if not path.is_absolute():
            r["features_npz"] = str((a.input_jsonl.parent / path).resolve())
    outputs = predict_features(records, a.checkpoint, a.cache_dir, a.device)
    a.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a.output_jsonl.write_text("".join(json.dumps(r) + "\n" for r in outputs), encoding="utf-8")
    print(json.dumps({"count": len(outputs), "output": str(a.output_jsonl), "labels_read": False}))
