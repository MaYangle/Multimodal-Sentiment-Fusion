#!/usr/bin/env python
"""CLI for the pinned, local-only raw-video feature extractor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="JSONL records with id, video_path, optional transcript")
    parser.add_argument("--output-dir", required=True, help="new output directory; existing outputs are refused")
    parser.add_argument("--model-root", required=True, help="directory containing mbert/wavlm/dinov2-small/whisper-base")
    parser.add_argument("--definitions", required=True, help="pinned raw_encoder_definitions.json")
    parser.add_argument("--project-root", help="repository root that contains jevaffect.media")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda:N")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--frame-size", type=int, default=224)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--asr-language", help="optional Whisper language code, e.g. zh or en; no translation is used")
    parser.add_argument("--disable-asr", action="store_true", help="leave text missing when no transcript is supplied")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    implementation_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(implementation_root))
    if args.project_root:
        sys.path.insert(0, str(Path(args.project_root).resolve()))

    # Import after adding the integrated project root so the extended package
    # path can resolve the existing jevaffect.media implementation.
    from jevaffect.raw_features import (
        RawVideoFeatureExtractor,
        load_pretrained_models,
        read_jsonl_records,
        write_raw_feature_cache,
    )

    records = read_jsonl_records(args.input_jsonl)
    if args.asr_language:
        for row in records:
            row.setdefault("asr_language", args.asr_language)
    bundle = load_pretrained_models(
        args.model_root,
        args.definitions,
        device=args.device,
        enable_asr=not args.disable_asr,
    )
    extractor = RawVideoFeatureExtractor(
        bundle,
        device=args.device,
        fps=args.fps,
        frame_size=args.frame_size,
        max_seconds=args.max_seconds,
        frame_batch_size=args.frame_batch_size,
        max_text_tokens=args.max_text_tokens,
    )
    result = write_raw_feature_cache(
        records,
        args.output_dir,
        extractor,
        input_manifest_path=args.input_jsonl,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
