"""Label-free inference from locally prepared native sequence features."""
import json
from pathlib import Path
import numpy as np
import torch
from jevaffect.native_training import PooledRegressor, digest, move
from jevaffect.context_nig import CASES, build_context_model
from jevaffect.reference_residual import REFERENCE_CASES, build_reference_model


def load_model(checkpoint_path, cache_dir, device="cpu"):
    # Checkpoints must be this project's trusted locally trained artifacts.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = checkpoint["identity"]; config = identity["config"]; cache = Path(cache_dir)
    metadata = cache / "metadata.json"
    if not metadata.exists():
        metadata = cache / "meta.json"
    known = identity.get("features", identity.get("feature_hashes"))["train"]
    recorded = [v for k, v in known.items() if Path(k).name in ("meta.json", "metadata.json")]
    if len(recorded) != 1 or digest(metadata) != recorded[0]:
        raise ValueError("Inference cache descriptor differs from training")
    meta = json.loads(metadata.read_text())
    text_pooling = meta.get("text_pooling", "cls")
    if text_pooling not in ("cls", "valid_mean"):
        raise ValueError("Unsupported trained text pooling contract")
    directory = cache / meta.get("split_paths", {}).get("train", "splits/train" if (cache / "splits").exists() else "train")
    modalities = ("text",) if config.get("model") == "text" else ("text", "audio", "vision")
    shapes = {}
    for m in modalities:
        normalization = meta.get("normalization")
        if (isinstance(normalization,dict)
                and normalization.get("kind") == "TRAIN clip-pooled coordinate mean/population STD; tiny STD<1e-6 uses1"):
            dimension = meta["dimensions"][m]
            if not isinstance(dimension,int) or dimension<1:
                raise ValueError("Invalid normalized pooled input dimension")
            sequence_shapes = meta.get("sequence_shapes")
            if sequence_shapes is not None:
                shape = sequence_shapes.get(m) if isinstance(sequence_shapes,dict) else None
                if (not isinstance(shape,list) or len(shape)!=2
                        or any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in shape)
                        or shape[1]!=dimension):
                    raise ValueError("Invalid normalized sequence input shape")
                shapes[m]=tuple(shape)
            else:
                shapes[m]=(1,dimension)
            continue
        # Header/dimension inspection only; inference does not read train labels.
        x = np.load(directory / (m + ".npy"), mmap_mode="r", allow_pickle=False)
        shapes[m] = tuple(x.shape[1:]); x._mmap.close()
    dims = {m: shape[1] for m, shape in shapes.items()}
    case = identity.get("case")
    if case in REFERENCE_CASES:
        model = build_reference_model(case, dims, config); pooled = True
    elif case in CASES:
        model = build_context_model(case, dims, config); pooled = True
    elif config.get("model") in ("text", "early_fusion"):
        model = PooledRegressor(dims, config.get("hidden", 256), config.get("dropout", .1)); pooled = True
    elif config.get("model") in ("almt", "almt_masked"):
        if config["model"] == "almt_masked":
            from jevaffect.masked_almt import MaskedALMTRegressor
            model = MaskedALMTRegressor(feature_dims=dims, lengths={m: s[0] for m, s in shapes.items()}, dim=128, token_len=8)
        else:
            from jevaffect.almt import ALMTClassifier
            model = ALMTClassifier(feature_dims=dims, lengths={m: s[0] for m, s in shapes.items()}, dim=128, token_len=8)
            model.classifier = torch.nn.Linear(model.classifier.in_features, 1)
        pooled = False
    else:
        raise ValueError("Unsupported native checkpoint family")
    # Verify code for the trained architecture; relocated file paths are allowed.
    for name, expected in identity["implementation"].items():
        normalized = name.replace("\\", "/")
        path = Path(__file__).parent / normalized if "/" not in normalized else Path(__file__).parents[1] / normalized
        if digest(path) != expected:
            raise ValueError("Trained implementation changed: " + name)
    model.load_state_dict(checkpoint["model"], strict=True); model.to(device).eval()
    return {"model": model, "pooled": pooled, "shapes": shapes, "modalities": modalities,
            "metadata_sha256": recorded[0], "dataset": config["dataset"], "device": torch.device(device), "case": case,
            "text_pooling": text_pooling, "reference_supervised": identity.get("reference_supervised", False)}


def read_features(path, spec):
    values, availability, lengths = {}, [], {}
    with np.load(path, allow_pickle=False) as archive:
        for m in spec["modalities"]:
            x = np.asarray(archive[m], dtype=np.float32); length = np.asarray(archive[m + "_length"])
            if x.shape != spec["shapes"][m] or length.shape != () or length.dtype.kind not in "iu":
                raise ValueError("Feature shape or integer scalar length differs from trained schema")
            n = int(length)
            if not 0 <= n <= len(x) or not np.isfinite(x[:n]).all():
                raise ValueError("Invalid length or valid feature values")
            availability.append(n > 0); lengths[m] = torch.tensor([n])
            if spec["pooled"]:
                value = x[0] if m == "text" and n and spec.get("text_pooling", "cls") == "cls" else (x[:n].mean(0) if n else np.zeros(x.shape[1], np.float32))
            else:
                value = np.zeros_like(x); value[:n] = x[:n]
            values[m] = torch.from_numpy(np.array(value, copy=True))[None]
    if not any(availability):
        raise ValueError("All modalities missing")
    if spec["case"] in CASES:
        values["availability"] = torch.tensor([availability], dtype=torch.bool)
    if not spec["pooled"]:
        values["lengths"] = lengths
    return values


def predict_features(records, checkpoint_path, cache_dir, device="cpu"):
    spec = load_model(checkpoint_path, cache_dir, device); outputs = []; seen = set()
    checkpoint_sha256 = digest(checkpoint_path)
    with torch.no_grad():
        for record in records:
            if not record.get("id") or record["id"] in seen or record.get("metadata_sha256") != spec["metadata_sha256"]:
                raise ValueError("Unique ID and exact declared encoder/cache descriptor required")
            seen.add(record["id"])
            features = read_features(record["features_npz"], spec)
            out = spec["model"](move(features, spec["device"]))
            point = out["prediction"] if isinstance(out, dict) else out
            score = float(point.reshape(-1)[0].cpu())
            if not np.isfinite(score):
                raise ValueError("Nonfinite native prediction")
            native_range = [-1., 1.] if spec["dataset"] == "simsv2" else [-3., 3.]
            row = {"id": record["id"], "dataset": spec["dataset"], "score_raw": score,
                            "score_range_clipped": float(np.clip(score, *native_range)),
                            "feature_sha256": digest(record["features_npz"]), "checkpoint_sha256": checkpoint_sha256}
            if spec["case"] in REFERENCE_CASES and (spec["reference_supervised"] or spec["case"].startswith("sum")):
                row["components"] = {key: float(out[key].reshape(-1)[0].cpu()) for key in ("text_reference", "annotation_correction")}
                row["components"].update(decoder="additive" if spec["case"].startswith("sum") else "direct", reference_supervised=spec["reference_supervised"])
            outputs.append(row)
    if not outputs:
        raise ValueError("Empty inference manifest")
    return outputs
