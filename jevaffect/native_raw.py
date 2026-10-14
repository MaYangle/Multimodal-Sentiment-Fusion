"""Pinned CH-SIMS original-media features for native scalar checkpoints.

Matches the existing provided-transcript MacBERT/WavLM/DINOv2 contract.
Author-linked MOSEI GloVe/COVAREP/FACET features have a different contract.
"""
import json
from pathlib import Path
import numpy as np
from jevaffect.native_training import digest
from jevaffect.raw_features import (
    EncoderBundle, ENCODER_PINS, RawVideoFeatureExtractor,
    verify_encoder_installation, _load_dinov2_processor,
)


def read_contract(cache_dir, spec, raw_source_cache=None):
    cache = Path(cache_dir)
    path = cache / "metadata.json"
    if not path.exists():
        path = cache / "meta.json"
    if digest(path) != spec["metadata_sha256"] or spec["dataset"] != "simsv2" or spec["text_pooling"] != "cls":
        raise ValueError("Native raw inference requires the bound CH-SIMS CLS encoder contract")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    statistics = None
    source_metadata = metadata
    if "normalization" in metadata and "source_descriptor" not in metadata:
        if metadata["normalization"].get("kind") != "TRAIN clip-pooled coordinate mean/population STD; tiny STD<1e-6 uses1" or metadata["normalization"].get("fit_split") != "train" or metadata.get("test_exported") is not False:
            raise ValueError("Unsupported native standardized input contract")
        from jevaffect.anchor_calibration import normalize
        if digest(Path(__file__).with_name("anchor_calibration.py")) != metadata["normalization_implementation_sha256"]:
            raise ValueError("Trained normalization implementation changed")
        source_cache = Path(raw_source_cache or metadata["source_cache"])
        source_path = source_cache / "metadata.json"
        if not source_path.exists(): source_path = source_cache / "meta.json"
        known = [value for name,value in metadata["source_feature_sha256"]["train"].items() if Path(name).name in ("metadata.json","meta.json")]
        if len(known) != 1 or digest(source_path) != known[0]:
            raise ValueError("Raw encoder source descriptor differs from standardized training")
        stats_path = cache / "normalization.json"
        if digest(stats_path) != metadata["artifact_sha256"]["normalization.json"]:
            raise ValueError("Trained native normalization statistics changed")
        statistics = json.loads(stats_path.read_text())
        source_metadata = json.loads(source_path.read_text())
    source = source_metadata["source_descriptor"]
    if source["text_condition"] != "provided transcripts, not ASR":
        raise ValueError("Unsupported trained transcript condition")
    base = source["base_raw_cache_descriptor"]
    preprocessing = base["encoder_preprocessing"]
    expected = {"audio_sample_rate": 16000, "max_seconds": 120., "mbert_max_tokens": 64,
                "video_fps": 2., "video_frame_size": 224,
                "video_frame_timestamps_kind": "ffmpeg_resampled_output_timeline",
                "audio_feature_times": "centers of WavLM convolutional receptive fields"}
    if preprocessing != expected or base["truncation_policy"] != "keep first target steps in original order; no resampling/interpolation":
        raise ValueError("Unsupported trained decoding or truncation recipe")
    encoder = source["replacement_text_encoder"]
    if encoder["model"] != "hfl/chinese-macbert-base" or encoder["revision"] != "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1" or encoder["files"]["pytorch_model.bin"]["sha256"] != "db0506d985574b80c33eec1cf13bd4c130585568753175871095ca13dbad9e23":
        raise ValueError("Unsupported trained text backbone")
    extraction_shapes = {m:tuple(base["target_shapes"][m]) for m in spec["shapes"]}
    for modality, shape in spec["shapes"].items():
        expected_shape = (1,extraction_shapes[modality][1]) if statistics is not None else extraction_shapes[modality]
        if tuple(shape) != expected_shape:
            raise ValueError("Trained feature shape differs from preprocessing recipe")
    for alias in ("wavlm", "dinov2-small"):
        recorded = base["encoder_provenance"]["encoders"][alias]
        pin = ENCODER_PINS[alias]
        if recorded["repo"] != pin["repo"] or recorded["revision"] != pin["revision"] or recorded["weight_sha256"] != pin["sha256"]:
            raise ValueError("Unsupported trained audio/vision backbone")
    return {"text_encoder": encoder, "preprocessing": preprocessing,
            "metadata_sha256": spec["metadata_sha256"], "shapes": spec["shapes"],
            "extraction_shapes":extraction_shapes,"normalization":statistics,
            "normalization_sha256":digest(cache/"normalization.json") if statistics is not None else None}


def pooled_standardized_features(arrays,statistics,shapes):
    """Truncate/pad before pooling, then apply the training-only fitted scaler."""
    from jevaffect.anchor_calibration import normalize
    output={}
    for modality,shape in shapes.items():
        x=np.asarray(arrays[modality],np.float32);n=int(arrays[modality+"_length"])
        if tuple(shape)!=(1,x.shape[1]) or not 0<=n<=len(x):
            raise ValueError("Invalid native pooled normalization schema")
        value=x[0] if modality=="text" and n else (x[:n].mean(0) if n else np.zeros(x.shape[1],np.float32))
        normalized=normalize(value[None],statistics[modality]) if n else np.zeros(shape,np.float32)
        output[modality]=normalized;output[modality+"_length"]=np.array(int(n>0),np.int64)
    return output


def load_raw_encoders(contract, macbert_dir, model_root, device="cpu"):
    """Local-only encoder load after all trained artifact hashes are verified."""
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor
    macbert = Path(macbert_dir)
    expected = contract["text_encoder"]
    actual = json.loads((macbert / "jevaffect_download_manifest.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("Local MacBERT descriptor differs from trained encoder")
    for name, record in expected["files"].items():
        if Path(name).name != name or digest(macbert / name) != record["sha256"]:
            raise ValueError("Local MacBERT file failed integrity: " + name)
    model_root = Path(model_root)
    verified = {alias: verify_encoder_installation(alias, model_root / alias, ENCODER_PINS[alias])
                for alias in ("wavlm", "dinov2-small")}
    verified["macbert"] = {"verified": True, "repo": expected["model"], "revision": expected["revision"],
        "weight_sha256": expected["files"]["pytorch_model.bin"]["sha256"], "files_sha256": expected["files"]}
    tokenizer = AutoTokenizer.from_pretrained(macbert, local_files_only=True, trust_remote_code=False)
    text = AutoModel.from_pretrained(macbert, local_files_only=True, trust_remote_code=False)
    audio_processor = AutoFeatureExtractor.from_pretrained(model_root / "wavlm", local_files_only=True, trust_remote_code=False)
    audio = AutoModel.from_pretrained(model_root / "wavlm", local_files_only=True, trust_remote_code=False)
    vision_processor = _load_dinov2_processor(AutoImageProcessor, model_root / "dinov2-small")
    vision = AutoModel.from_pretrained(model_root / "dinov2-small", local_files_only=True, trust_remote_code=False)
    for model in (text, audio, vision):
        model.to(device).eval()
    if {"text": text.config.hidden_size, "audio": audio.config.hidden_size, "vision": vision.config.hidden_size} != {"text": 768, "audio": 768, "vision": 384}:
        raise ValueError("Pinned encoder dimensions differ")
    return EncoderBundle(text_tokenizer=tokenizer, text_model=text, audio_processor=audio_processor,
        audio_model=audio, vision_processor=vision_processor, vision_model=vision, torch_module=torch, device=device,
        provenance={"encoders": verified, "local_files_only": True, "fixture": False,
                    "dimensions": {"text": 768, "audio": 768, "vision": 384}, "transcript_condition": "provided", "dinov2_image_processor_use_fast": False})


class NativeMediaExtractor(RawVideoFeatureExtractor):
    """Keep the MacBERT cache's exact padded-tokenization condition."""
    def _encode_text(self, text):
        import torch
        if not isinstance(text, str) or not text.strip():
            raise ValueError("This checkpoint was trained with provided nonempty transcripts")
        tokenizer = self.models.text_tokenizer
        full = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        batch = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_text_tokens)
        batch = {k: value.to(self.device) for k, value in batch.items()}
        with torch.inference_mode():
            states = self.models.text_model(**batch).last_hidden_state[0].float().cpu().numpy()
        mask = batch["attention_mask"][0].cpu().numpy().astype(np.bool_)
        tokens = batch["input_ids"][0].cpu().numpy().astype(np.int64)
        return states[mask], np.ones(int(mask.sum()), np.bool_), tokens[mask], len(full) > self.max_text_tokens


def pad_native_features(extracted, shapes):
    """Keep first valid steps; expose original lengths and actual truncation."""
    arrays, audit = {}, {}
    for modality, shape in shapes.items():
        x = np.asarray(extracted["arrays"][modality], np.float32)
        mask = np.asarray(extracted["arrays"][modality + "_mask"])
        if x.ndim != 2 or x.shape[1] != shape[1] or mask.dtype != np.bool_ or mask.shape != (len(x),) or not np.isfinite(x).all():
            raise ValueError("Invalid extracted native feature/mask schema")
        count = int(mask.sum())
        if count == 0 or not np.array_equal(mask, np.arange(len(x)) < count):
            raise ValueError("Required modality missing or unexpected non-prefix raw mask")
        n = min(count, shape[0]); padded = np.zeros(shape, np.float32); padded[:n] = x[:n]
        arrays[modality] = padded; arrays[modality + "_length"] = np.array(n, np.int64)
        audit[modality] = {"raw_valid_steps": count, "kept_steps": n, "dropped_steps": count - n}
    return arrays, audit
