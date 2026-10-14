"""CH-SIMS MP4 + provided transcript -> original continuous sentiment score."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.native_predict import load_model, predict_features
from jevaffect.native_raw import read_contract, load_raw_encoders, NativeMediaExtractor, pad_native_features, pooled_standardized_features
from jevaffect.native_training import save_json, digest

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--raw-source-cache",type=Path,help="Relocated raw encoder descriptor for a standardized checkpoint")
    p.add_argument("--macbert-dir", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args(); torch.set_num_threads(4)
    records = [json.loads(line) for line in a.input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Nonempty unique inference IDs required")
    for record in records:
        if any(key in record for key in ("label", "target", "label_M", "label_T", "label_A", "label_V")) or not record.get("transcript", "").strip():
            raise ValueError("Provide label-free media records and nonempty transcripts")
        path = Path(record["video_path"])
        record["video_path"] = str(path if path.is_absolute() else (a.input_jsonl.parent / path).resolve())
    if a.output_dir.exists():
        raise ValueError("Use a new output directory; prior predictions are not overwritten")
    spec = load_model(a.checkpoint, a.cache_dir, a.device)
    contract = read_contract(a.cache_dir, spec, a.raw_source_cache)
    models = load_raw_encoders(contract, a.macbert_dir, a.model_root, a.device)
    extractor = NativeMediaExtractor(models, device=a.device, fps=2., frame_size=224, max_seconds=120., max_text_tokens=64)
    a.output_dir.mkdir(parents=True)
    feature_rows, audits = [], []
    for i, record in enumerate(records):
        extracted = extractor.extract_record(record)
        arrays, truncation = pad_native_features(extracted, contract["extraction_shapes"])
        if contract["normalization"] is not None:
            arrays=pooled_standardized_features(arrays,contract["normalization"],spec["shapes"])
        feature = a.output_dir / f"example_{i}.npz"; np.savez_compressed(feature, **arrays)
        feature_rows.append({"id": record["id"], "features_npz": str(feature.resolve()), "metadata_sha256": spec["metadata_sha256"]})
        meta = extracted["metadata"]
        meta.update(trained_feature_truncation=truncation, text_encoder_name="hfl/chinese-macbert-base", sentiment_labels_read=False)
        if contract["normalization"] is not None:
            meta["native_pooled_standardization"]={"normalization_sha256":contract["normalization_sha256"],"fit_split":"train","input_steps":1}
        meta["modalities"]["text"]["mask_semantics"] = "MacBERT valid tokens including special tokens"
        meta["preprocessing"]["text_max_tokens"] = meta["preprocessing"].pop("mbert_max_tokens")
        audits.append(meta)
        print(json.dumps({"id": record["id"], "features_ready": True, "truncation": truncation}), flush=True)
    predictions = predict_features(feature_rows, a.checkpoint, a.cache_dir, a.device)
    (a.output_dir / "features.jsonl").write_text("".join(json.dumps(r) + "\n" for r in feature_rows), encoding="utf-8")
    (a.output_dir / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in predictions), encoding="utf-8")
    save_json(a.output_dir / "media_audit.json", {"input_form": "original MP4 plus provided transcript", "dataset": "simsv2",
        "device": a.device, "count": len(predictions), "input_jsonl_sha256": digest(a.input_jsonl),
        "implementation": {name: digest(Path(__file__).parents[1] / name) for name in ("scripts/predict_native_media.py", "jevaffect/native_raw.py", "jevaffect/raw_features.py", "jevaffect/media.py", "jevaffect/native_predict.py")},
        "metadata_sha256": spec["metadata_sha256"], "labels_read": False, "records": audits})
    print(json.dumps({"count": len(predictions), "predictions": str(a.output_dir / "predictions.jsonl"), "labels_read": False}))
