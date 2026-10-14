#!/usr/bin/env python
"""Pilot whole-raw-video inference using a verified JevAffect training run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="training run directory with run_manifest/config/best.pt")
    parser.add_argument("--cache-meta", required=True, help="the exact cache meta.json used to train this run")
    parser.add_argument("--input-jsonl", required=True, help="records with id, video_path, optional transcript/asr_language")
    parser.add_argument("--output-dir", required=True, help="new directory for pilot predictions and timings")
    parser.add_argument("--model-root", required=True, help="local verified mbert/wavlm/dinov2-small/whisper-base models")
    parser.add_argument("--definitions", required=True, help="pinned raw_encoder_definitions.json")
    parser.add_argument("--project-root", help="parent repository containing jevaffect.media/models/protocol")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    parser.add_argument("--low-confidence-threshold", type=float, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    implementation_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(implementation_root))
    if args.project_root:
        sys.path.insert(0, str(Path(args.project_root).expanduser().resolve()))
    import jevaffect
    implementation_package = str((implementation_root / "jevaffect").resolve())
    if implementation_package not in jevaffect.__path__:
        jevaffect.__path__.append(implementation_package)

    try:
        import torch
        from jevaffect.protocol import CLASSES, make_prediction_record
        from jevaffect.raw_features import (
            RawVideoFeatureExtractor,
            load_pretrained_models,
            read_jsonl_records,
        )
        from jevaffect.raw_predict import (
            RawVideoPredictor,
            load_classifier_model,
            load_run_spec,
            write_prediction_bundle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "The parent project's jevaffect package, PyTorch, and encoder dependencies are required"
        ) from exc

    spec = load_run_spec(args.run_dir, args.cache_meta, class_order=CLASSES)
    model, _checkpoint, classifier_initialization = load_classifier_model(
        spec,
        device=args.device,
        torch_module=torch,
    )
    encoder_started = time.perf_counter()
    encoder_bundle = load_pretrained_models(
        args.model_root,
        args.definitions,
        device=args.device,
        enable_asr=True,
    )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize(args.device)
    encoder_initialization = {
        "raw_encoder_load_wall_seconds": time.perf_counter() - encoder_started,
        "device": args.device,
        "local_files_only": True,
        "encoder_provenance": dict(encoder_bundle.provenance),
    }
    feature_extractor = RawVideoFeatureExtractor(
        encoder_bundle,
        device=args.device,
        fps=args.fps,
        frame_size=args.frame_size,
        max_seconds=args.max_seconds,
        frame_batch_size=args.frame_batch_size,
        max_text_tokens=spec.sequence_lengths["text"],
    )
    predictor = RawVideoPredictor(
        spec,
        model,
        feature_extractor,
        class_order=CLASSES,
        low_confidence_threshold=args.low_confidence_threshold,
        torch_module=torch,
        device=args.device,
        prediction_factory=make_prediction_record,
    )
    records = read_jsonl_records(args.input_jsonl)
    info = write_prediction_bundle(
        records,
        args.output_dir,
        predictor,
        input_manifest_path=args.input_jsonl,
        classifier_initialization=classifier_initialization,
        encoder_initialization=encoder_initialization,
    )
    print(json.dumps(info, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
