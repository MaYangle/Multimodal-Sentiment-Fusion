"""Group-holdout counterfactual audit of JevAffect V2's base gate.

The learned gate is a candidate evidence score, not an explanation by itself.
This checks the temporal base subnetwork's masking dependence. The bounded
cross-token correction in final V2 predictions is excluded from this gate
audit; results must not be described as final-model causal attribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jevaffect.jev_temporal import JevTemporalOrdinal  # noqa: E402
from jevaffect.group_temporal_data import load_group_temporal_train_valid  # noqa: E402
from jevaffect.metrics import classification_metrics  # noqa: E402


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_numpy(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError("Unsupported audit JSON value: " + type(value).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.report.exists() or args.details.exists():
        raise FileExistsError("Audit output already exists")
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if config.get("model") != "JevAffectV2" or config.get("gate_kind") != "learned" or config.get("temporal") is not True:
        raise ValueError("Evidence audit requires the learned temporal model")
    train, valid, source = load_group_temporal_train_valid(args.cache)
    if source["source_sha256"] != config["cache_source_sha256"] or source["meta_sha256"] != manifest["cache_meta_sha256"]:
        raise ValueError("Checkpoint/cache source identity mismatch")
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=True)
    if checkpoint["identity_sha256"] != manifest["identity_sha256"] or checkpoint["class_order"] != ["SNEG", "WNEG", "NEUT", "WPOS", "SPOS"]:
        raise ValueError("Selected checkpoint does not match run/class contract")
    model = JevTemporalOrdinal(hidden=config["hidden"], windows=config["windows"],
                               gate_kind="learned", temporal=True).to(args.device)
    model.load_state_dict({key.removeprefix("base."): value for key, value in checkpoint["model"].items()
                           if key.startswith("base.")}, strict=True)
    model.eval()
    loader = DataLoader(valid, batch_size=16, shuffle=False, num_workers=0)
    all_gates, all_effects, by_modality = [], [], {"audio": [[], []], "vision": [[], []]}
    full_y, full_pred = [], []
    ablated_pred = {"audio": [], "vision": []}
    details = []
    begin = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            values = {k: v.to(args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            labels = values["label"].long()
            output = model(values)
            duration = output["time_windows_seconds"][:, -1, 1]
            full_loss = F.cross_entropy(output["logits"], labels, reduction="none")
            full_y.extend(labels.cpu().tolist())
            full_pred.extend(output["logits"].argmax(-1).cpu().tolist())
            gate = output["gate"].cpu().numpy()
            valid_windows = output["valid_windows"].cpu().numpy()
            effects = np.full(gate.shape, np.nan, dtype=np.float32)
            for channel, modality in enumerate(("audio", "vision")):
                removed_all = dict(values)
                removed_all[modality + "_mask"] = torch.zeros_like(values[modality + "_mask"])
                ablated_pred[modality].extend(model(removed_all, duration_override=duration)["logits"].argmax(-1).cpu().tolist())
                timestamps = values[modality + "_times"]
                original_mask = values[modality + "_mask"]
                for window in range(config["windows"]):
                    lower = output["time_windows_seconds"][:, window, 0:1]
                    upper = output["time_windows_seconds"][:, window, 1:2]
                    is_window = (timestamps >= lower) & ((timestamps <= upper) if window == config["windows"] - 1 else (timestamps < upper))
                    removed = dict(values)
                    removed[modality + "_mask"] = original_mask & ~is_window
                    alternate = model(removed, duration_override=duration)
                    difference = F.cross_entropy(alternate["logits"], labels, reduction="none") - full_loss
                    effects[:, window, channel] = difference.cpu().numpy()
                    for score, effect, enabled in zip(gate[:, window, channel], effects[:, window, channel], valid_windows[:, window, channel], strict=True):
                        if enabled:
                            all_gates.append(float(score)); all_effects.append(float(effect))
                            by_modality[modality][0].append(float(score)); by_modality[modality][1].append(float(effect))
            for index, sample_id in enumerate(batch["id"]):
                eligible = valid_windows[index]
                slots = [(float(gate[index, w, c]), float(effects[index, w, c]), w, c)
                         for w in range(config["windows"]) for c in range(2) if eligible[w, c]]
                if not slots:
                    raise ValueError("Validation record has no audiovisual evidence windows")
                highest = max(slots, key=lambda item: item[0])
                details.append({"ID": sample_id, "label": int(labels[index].item()),
                                "predicted": int(output["logits"][index].argmax().item()),
                                "valid_window_count": len(slots),
                                "highest_gate": highest[0], "highest_gate_mask_loss_change": highest[1],
                                "mean_window_mask_loss_change": float(np.mean([slot[1] for slot in slots])),
                                "highest_gate_window": highest[2],
                                "highest_gate_modality": ("audio", "vision")[highest[3]]})
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    def correlation(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) < 2 or np.std(xs) < 1e-10 or np.std(ys) < 1e-10:
            return None
        return float(np.corrcoef(xs, ys)[0, 1])
    result = {"scope": "train-only source-video-disjoint selection set; no official validation/test scoring",
              "interpretation_limit": "Only the temporal base subnetwork is masked; V2's final cross-token residual is excluded. Prediction dependence is not human-grounded causal evidence.",
              "cache_source_sha256": source["source_sha256"], "checkpoint_sha256": sha(args.run_dir / "best.pt"),
              "valid_count": len(full_y), "window_observations": len(all_gates),
              "gate_vs_mask_loss_change_pearson": correlation(all_gates, all_effects),
              "by_modality": {modality: {"count": len(pair[0]),
                                          "gate_vs_mask_loss_change_pearson": correlation(*pair),
                                          "mean_mask_loss_change": float(np.mean(pair[1]))}
                              for modality, pair in by_modality.items()},
              "mean_highest_gate_window_mask_loss_change": float(np.mean([row["highest_gate_mask_loss_change"] for row in details])),
              "mean_uniform_valid_window_mask_loss_change": float(np.mean([row["mean_window_mask_loss_change"] for row in details])),
              "full_metrics": classification_metrics(np.asarray(full_y), np.asarray(full_pred)),
              "whole_modality_removed_metrics": {modality: classification_metrics(np.asarray(full_y), np.asarray(predictions))
                                                  for modality, predictions in ablated_pred.items()},
              "audit_wall_seconds": time.perf_counter() - begin,
              "test_labels_used": False}
    if result["valid_count"] != 540 or not math.isfinite(result["mean_highest_gate_window_mask_loss_change"]):
        raise ValueError("Evidence audit count/nonfinite mismatch")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=json_numpy) + "\n", encoding="utf-8")
    args.details.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "accuracy": result["full_metrics"]["accuracy"],
                      "gate_utility_pearson": result["gate_vs_mask_loss_change_pearson"],
                      "top_gate_effect": result["mean_highest_gate_window_mask_loss_change"],
                      "average_effect": result["mean_uniform_valid_window_mask_loss_change"],
                      "audio_removed_accuracy": result["whole_modality_removed_metrics"]["audio"]["accuracy"],
                      "vision_removed_accuracy": result["whole_modality_removed_metrics"]["vision"]["accuracy"]}), flush=True)
    train.base.close(); valid.base.close()


if __name__ == "__main__":
    main()
