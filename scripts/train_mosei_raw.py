"""Train projected 128-D cross-token fusion on competition-ID CMU-MOSEI.

The small 54/22 media pilot checks real separate-WAV and missing-MP4 training,
not model performance. The full 16,327/1,871 cohort is used only after every
train/valid media feature is verified. No author-ID crosswalk or test access.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jevaffect.cross_token_jev import CrossTokenJev  # noqa: E402
from jevaffect.mosei_raw_data import load_mosei_raw_train_valid  # noqa: E402
from jevaffect.metrics import classification_metrics, negative_log_likelihood, top_label_ece_15  # noqa: E402
from jevaffect.protocol import CLASSES  # noqa: E402
from jevaffect.training import _atomic_json, _atomic_jsonl, _atomic_torch_save  # noqa: E402


SEEDS = (13, 42, 2026)
COHORTS = {
    "pilot": ("data/mosei/pilot_feature_cache", "data/mosei/pilot_media/input_records.jsonl", (54, 22)),
    "full": ("data/mosei/full_feature_cache", "data/mosei/full_media/input_records.jsonl", (16327, 1871)),
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def rng_state(generator: torch.Generator) -> dict:
    n = np.random.get_state()
    return {"python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
            "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "loader": generator.get_state()}


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader"])


def move(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def objective(output: dict, labels: torch.Tensor) -> tuple[torch.Tensor, float]:
    main = F.cross_entropy(output["logits"], labels)
    branches = []
    for index, modality in enumerate(("text", "audio", "vision")):
        valid = output["valid_modalities"][:, index]
        if bool(valid.any()):
            branches.append(F.cross_entropy(output[modality + "_logits"][valid], labels[valid]))
    loss = main + 0.15 * sum(branches)
    return loss, float(main.detach().item())


@torch.no_grad()
def evaluate(model: CrossTokenJev, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    ids, labels, probabilities = [], [], []
    for batch in loader:
        values = move(batch, device)
        output = model(values)
        ids.extend(batch["id"])
        labels.extend(values["label"].cpu().tolist())
        probabilities.extend(output["logits"].softmax(-1).cpu().tolist())
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    metrics = classification_metrics(y, p.argmax(1))
    return {"ids": ids, "labels": labels, "probabilities": probabilities,
            "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
            "nll": negative_log_likelihood(y, p), "ece15": top_label_ece_15(y, p)}


def train_seed(seed: int, cohort: str, *, resume: bool = False,
               max_epochs_this_call: int | None = None, cross_attention: bool = True,
               use_av_time: bool = True, gate_kind: str = "learned") -> dict:
    if seed not in SEEDS or cohort not in COHORTS or gate_kind not in ("learned", "uniform"):
        raise ValueError("Unregistered seed, cohort or gate")
    if not torch.cuda.is_available():
        raise RuntimeError("This real training protocol requires local CUDA")
    device = torch.device("cuda")
    relative_cache, relative_inputs, expected_counts = COHORTS[cohort]
    cache = ROOT / relative_cache
    train, valid, source = load_mosei_raw_train_valid(
        feature_dir=cache, input_manifest=ROOT / relative_inputs,
        manifest_dir=ROOT / "data/mosei/manifests", expected_counts=expected_counts)
    if (len(train), len(valid)) != expected_counts:
        raise ValueError("Frozen cohort count mismatch")
    source_sha = hashlib.sha256(canonical(source)).hexdigest()
    meta_sha = source["train"]["feature_meta_sha256"]
    settings = {"seed": seed, "cohort": cohort,
                "source_sha256": source_sha, "meta_sha256": meta_sha,
                "source_video_cross_split_overlap": source["source_group_overlap"],
                "missing_video_train": source["train"]["missing_video_count"],
                "missing_video_valid": source["valid"]["missing_video_count"],
                "model": "CrossTokenJev", "width": 128, "heads": 4,
                "cross_attention": cross_attention, "use_av_time": use_av_time,
                "gate_kind": gate_kind, "dropout": 0.15, "branch_ce_weight": 0.15,
                "max_epochs": 80, "patience": 12, "batch_size": 32,
                "learning_rate": 3e-4, "weight_decay": 1e-4,
                "scheduler_patience": 3, "scheduler_factor": 0.5,
                "selection": "validation accuracy, NLL tie-break; no test access",
                "pilot_not_formal_benchmark": cohort == "pilot"}
    relevant = ["scripts/train_mosei_raw.py", "jevaffect/cross_token_jev.py",
                "jevaffect/mosei_raw_data.py"]
    implementation_sha = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in relevant)).hexdigest()
    identity = hashlib.sha256(canonical({"settings": settings,
                                         "implementation_sha256": implementation_sha,
                                         "torch": torch.__version__, "numpy": np.__version__})).hexdigest()
    variant = ("cross" if cross_attention else "no_cross") + "_" + ("time" if use_av_time else "no_time") + "_" + gate_kind
    folder = ROOT / "runs" / "mosei_raw" / cohort / variant / f"seed_{seed}"
    last_path, best_path = folder / "last.pt", folder / "best.pt"
    if folder.exists() and not resume:
        raise FileExistsError("Run exists; only exact --resume is allowed")
    if resume and not last_path.is_file():
        raise FileNotFoundError("No exact checkpoint to resume")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train, batch_size=32, shuffle=True, generator=generator, num_workers=0)
    validation = DataLoader(valid, batch_size=32, shuffle=False, num_workers=0)
    model = CrossTokenJev(width=128, heads=4, dropout=0.15,
                          cross_attention=cross_attention, use_av_time=use_av_time,
                          gate_kind=gate_kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    folder.mkdir(parents=True, exist_ok=True)
    history, resources = [], []
    start, stale, best_accuracy, best_nll, best_epoch = 1, 0, -1.0, float("inf"), 0
    if resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["identity_sha256"] != identity:
            raise ValueError("Run code/config/data identity changed")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng(checkpoint["rng_state"], generator)
        history, resources = checkpoint["history"], checkpoint["resources"]
        start, stale = checkpoint["epoch"] + 1, checkpoint["stale"]
        best_accuracy, best_nll, best_epoch = checkpoint["best_accuracy"], checkpoint["best_nll"], checkpoint["best_epoch"]
        if checkpoint["completed"]:
            report = folder / "summary.json"
            if not best_path.is_file() or not report.is_file():
                raise ValueError("Completed run lacks selected artifacts")
            return json.loads(report.read_text(encoding="utf-8"))
    else:
        _atomic_json(folder / "config.json", settings)
        _atomic_json(folder / "run_manifest.json", {
            "identity_sha256": identity, "implementation_sha256": implementation_sha,
            "source_sha256": source_sha, "cache_meta_sha256": meta_sha,
            "train_label_manifest_sha256": source["train"]["label_manifest_sha256"],
            "valid_label_manifest_sha256": source["valid"]["label_manifest_sha256"],
            "train_count": len(train), "valid_count": len(valid),
            "class_order": list(CLASSES), "test_scored": False})
        _atomic_json(folder / "environment.json", {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda})
    this_call = 0
    for epoch in range(start, 81):
        if stale >= 12 or (max_epochs_this_call is not None and this_call >= max_epochs_this_call):
            break
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        began = time.perf_counter()
        total_loss, total_correct = 0.0, 0
        for batch in loader:
            values = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(values)
            loss, _ = objective(output, values["label"].long())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cross-token training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch["id"])
            total_correct += int((output["logits"].argmax(-1) == values["label"]).sum())
        checked = evaluate(model, validation, device)
        torch.cuda.synchronize()
        if len(checked["ids"]) != len(valid) or not math.isfinite(checked["nll"]):
            raise ValueError("Invalid validation output")
        improved = checked["accuracy"] > best_accuracy or (checked["accuracy"] == best_accuracy and checked["nll"] < best_nll)
        if improved:
            best_accuracy, best_nll, best_epoch, stale = checked["accuracy"], checked["nll"], epoch, 0
            _atomic_torch_save(best_path, {"identity_sha256": identity, "epoch": epoch,
                                           "model": model.state_dict(), "class_order": list(CLASSES),
                                           "source_sha256": source_sha})
            _atomic_jsonl(folder / "validation_predictions.jsonl", [
                {"ID": sample_id, "split": "valid", "label": int(label),
                 "probabilities": probs, "argmax": CLASSES[int(np.argmax(probs))],
                 "class_order": list(CLASSES)}
                for sample_id, label, probs in zip(checked["ids"], checked["labels"], checked["probabilities"], strict=True)])
        else:
            stale += 1
        scheduler.step(checked["accuracy"])
        row = {"epoch": epoch, "train_loss": total_loss / len(train),
               "train_accuracy": total_correct / len(train),
               "valid_accuracy": checked["accuracy"], "valid_macro_f1": checked["macro_f1"],
               "valid_nll": checked["nll"], "valid_ece15": checked["ece15"],
               "best_accuracy": best_accuracy, "best_epoch": best_epoch,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        resources.append({"epoch": epoch, "train_valid_wall_seconds": time.perf_counter() - began,
                          "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)})
        completed = stale >= 12 or epoch >= 80
        _atomic_torch_save(last_path, {"identity_sha256": identity, "epoch": epoch,
                                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                       "scheduler": scheduler.state_dict(), "rng_state": rng_state(generator),
                                       "history": history, "resources": resources,
                                       "best_accuracy": best_accuracy, "best_nll": best_nll,
                                       "best_epoch": best_epoch, "stale": stale,
                                       "completed": completed})
        _atomic_jsonl(folder / "metrics.jsonl", history)
        _atomic_jsonl(folder / "resources.jsonl", resources)
        print(json.dumps(row), flush=True)
        this_call += 1
        if completed:
            break
    summary = {"seed": seed, "cohort": cohort, "variant": variant,
               "best_epoch": best_epoch, "best_accuracy": best_accuracy,
               "last_epoch": history[-1]["epoch"] if history else 0,
               "completed": stale >= 12 or len(history) >= 80,
               "source_sha256": source_sha, "test_scored": False,
               "pilot_not_formal_benchmark": cohort == "pilot"}
    _atomic_json(folder / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=13)
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs-this-call", type=int)
    parser.add_argument("--no-cross-attention", action="store_true")
    parser.add_argument("--no-av-time", action="store_true")
    parser.add_argument("--gate-kind", choices=("learned", "uniform"), default="learned")
    args = parser.parse_args()
    print(json.dumps(train_seed(args.seed, args.cohort, resume=args.resume,
                                max_epochs_this_call=args.max_epochs_this_call,
                                cross_attention=not args.no_cross_attention,
                                use_av_time=not args.no_av_time,
                                gate_kind=args.gate_kind)), flush=True)


if __name__ == "__main__":
    main()
