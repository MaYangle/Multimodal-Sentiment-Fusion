"""CPU architecture demonstration with synthetic inputs, not an accuracy demo."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from jevaffect.sequence_gates import SequenceGateRegressor


def main():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    dims = {"text": 768, "audio": 768, "vision": 384}
    model = SequenceGateRegressor(dims, "distilled_post_norm").eval()
    features = {m: torch.randn(2, 8, d) for m, d in dims.items()}
    features["lengths"] = {
        "text": torch.tensor([8, 8]),
        "audio": torch.tensor([8, 0]),
        "vision": torch.tensor([8, 0]),
    }
    with torch.no_grad():
        out = model(features)
    print(json.dumps({
        "input": "synthetic; random untrained weights; no accuracy claim",
        "parameters": sum(p.numel() for p in model.parameters()),
        "prediction": out["prediction"].tolist(),
        "route_text_audio_vision": out["ratios"].tolist(),
        "labels_read": False,
    }, indent=2))


if __name__ == "__main__":
    main()
