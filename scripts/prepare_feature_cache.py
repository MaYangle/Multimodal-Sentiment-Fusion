"""Prepare a verified MMSA/SIMS-v2 NumPy feature cache; never imports torch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevaffect.cache import CachePreparationError, convert_source_pickle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pickle", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="Author split/ID manifest (.jsonl or .csv)")
    parser.add_argument("--expected-sha256", required=True,
                        help="Verified source-pickle SHA-256; checked before unpickling")
    parser.add_argument("--source-reference", required=True,
                        help="Dataset/version/source URL or approved storage provenance")
    parser.add_argument("--trust-source-pickle", action="store_true", required=True,
                        help="Acknowledge that pickle executes code and this file is trusted")
    parser.add_argument("--source-id-format", required=True,
                        choices=("verbatim", "video_clip_path", "video_dollar_clip"),
                        help="Explicit ID grammar in source split records")
    parser.add_argument("--manifest-id-format", default="video_clip_path",
                        choices=("verbatim", "video_clip_path", "video_dollar_clip"))
    parser.add_argument("--label-field", default="regression_labels")
    parser.add_argument("--expected-text-dim", type=int, default=768,
                        help="Text context dimension required for a training-ready text.npy")
    parser.add_argument("--raw-label-tolerance", type=float, default=1e-6)
    parser.add_argument("--infer-zero-padding-lengths", action="store_true",
                        help="Opt in to a risky trailing-all-zero-row length heuristic if source lengths are absent")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = convert_source_pickle(
            args.source_pickle,
            args.output_dir,
            expected_sha256=args.expected_sha256,
            trust_source_pickle=args.trust_source_pickle,
            manifest_path=args.manifest,
            source_id_format=args.source_id_format,
            manifest_id_format=args.manifest_id_format,
            label_field=args.label_field,
            expected_text_dim=args.expected_text_dim,
            raw_label_tolerance=args.raw_label_tolerance,
            infer_zero_padding_lengths=args.infer_zero_padding_lengths,
            source_reference=args.source_reference,
        )
    except (CachePreparationError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print("feature cache preparation failed: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
