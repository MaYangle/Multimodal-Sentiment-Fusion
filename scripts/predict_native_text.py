"""Label-free scalar inference with a native frozen/top4 MacBERT checkpoint."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.native_text_tuning import verify_pinned_pretrained, _load_pinned_model, _load_pinned_tokenizer, _tokenize
from jevaffect.native_training import digest

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pretrained-dir", type=Path, required=True)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args(); torch.set_num_threads(4)
    records = [json.loads(line) for line in a.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({r.get("id") for r in records}) != len(records) or any(not isinstance(r.get("id"), str) or not r["id"] for r in records):
        raise ValueError("Nonempty unique string IDs required")
    if any(not isinstance(r.get("text"), str) or not r["text"].strip() or any(k in r for k in ("target", "label", "label_M", "label_T", "label_A", "label_V")) for r in records):
        raise ValueError("Provide nonempty text and no sentiment labels")
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    identity = state["identity"]
    if identity["config"]["dataset"] != "simsv2" or identity["config"]["model"] != "native_macbert_text":
        raise ValueError("This entry supports native CH-SIMS text checkpoints")
    for name, expected in identity["implementation"].items():
        if digest(Path(__file__).resolve().parents[1] / name) != expected:
            raise ValueError("Trained text implementation changed: " + name)
    if verify_pinned_pretrained(a.pretrained_dir) != identity["source"]["model_tokenizer"]:
        raise ValueError("Pretrained/tokenizer contract differs from training")
    model = _load_pinned_model(a.pretrained_dir, identity["mode"])
    model.load_state_dict(state["model"], strict=True); model.to(a.device).eval()
    tokenizer = _load_pinned_tokenizer(a.pretrained_dir)
    checkpoint_sha = digest(a.checkpoint); output = []
    with torch.no_grad():
        for start in range(0, len(records), 32):
            batch_records = records[start:start + 32]
            tokens = _tokenize(tokenizer, batch_records, identity["config"]["max_tokens"])
            predictions = model(tokens["input_ids"].to(a.device), tokens["attention_mask"].to(a.device)).cpu().tolist()
            for record, score in zip(batch_records, predictions):
                if not np.isfinite(score):
                    raise ValueError("Nonfinite native text output")
                output.append({"id": record["id"], "dataset": "simsv2", "input_form": "provided text",
                    "mode": identity["mode"], "score_raw": score, "score_range_clipped": float(np.clip(score, -1, 1)),
                    "checkpoint_sha256": checkpoint_sha})
    a.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a.output_jsonl.write_text("".join(json.dumps(r) + "\n" for r in output), encoding="utf-8")
    print(json.dumps({"count": len(output), "labels_read": False, "output": str(a.output_jsonl)}))
