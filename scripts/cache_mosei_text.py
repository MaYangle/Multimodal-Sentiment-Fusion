"""Cache verified mBERT features from author-provided CMU-MOSEI transcripts.

This text-only artifact reads only the author's train and valid manifests. It
never reads or encodes the test manifest, never uses the provisional
competition crosswalk, and does not create the shared JevAffect multimodal
cache contract (audio and vision features must be merged separately).

The script is intended to live in ``<project>/scripts/`` beside
``download_encoders.py`` and ``prepare_mosei_manifest.py``. The encoder must
already exist locally and match the pinned ``mbert`` definition; no model is
downloaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from jevaffect.protocol import CLASSES  # noqa: E402


SPLITS = ("train", "valid")
AUTHOR_COHORT = "mmsa_author"
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_FIELDS = (
    "ID",
    "author_id",
    "dataset",
    "cohort",
    "split",
    "text",
    "transcript_sha256",
    "raw_label",
    "class",
    "source_csv_sha256",
)


class AuthorTextCacheError(ValueError):
    """Raised when author manifests, encoder provenance, or cache state is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def combined_source_sha256(source_hashes: dict[str, Any]) -> str:
    """Hash the canonical source/encoder provenance object for training identity."""
    payload = json.dumps(
        source_hashes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_text_mlp_cache_metadata(
    source_hashes: dict[str, Any],
    *,
    feature_dim: int,
    max_length: int,
    split_counts: dict[str, int],
    encoder: dict[str, Any],
    complete: bool,
) -> dict[str, Any]:
    """Build the metadata fields consumed by jevaffect.training text_mlp."""
    if set(split_counts) != set(SPLITS) or any(split_counts[split] <= 0 for split in SPLITS):
        raise AuthorTextCacheError("train/valid cache split counts must be positive")
    if feature_dim <= 0 or max_length <= 0:
        raise AuthorTextCacheError("text feature dimension and sequence length must be positive")
    return {
        "schema_version": "mosei-author-text-embedding-v1",
        "dataset": "CMU-MOSEI",
        "cohort": AUTHOR_COHORT,
        "transcript_condition": "author_provided_transcripts",
        "modalities": ["text"],
        "class_order": list(CLASSES),
        "source_hashes": source_hashes,
        "source_sha256": combined_source_sha256(source_hashes),
        "encoder": encoder,
        "feature_dims": {"text": feature_dim},
        "sequence_lengths": {"text": max_length},
        "split_paths": {"train": "train", "valid": "valid"},
        "text_features_ready": bool(complete),
        "complete": bool(complete),
        "test_encoded": False,
        "test_manifest_read": False,
        "crosswalk_used": False,
        "cache_contract_compatible_for_text_mlp": bool(complete),
        "cache_contract_compatible_for_multimodal": False,
        "cache_contract_note": (
            "This cache satisfies jevaffect.training text_mlp input fields only. "
            "It has no audio/vision arrays and is not ready for multimodal training."
        ),
        "split_counts": split_counts,
    }


def write_split_sidecars(split_dir: Path, rows: list[dict[str, Any]], np_module) -> None:
    """Write IDs and labels in the exact layout consumed by FeatureSplit."""
    ids = [row["ID"] for row in rows]
    labels = np_module.asarray([CLASSES.index(row["class"]) for row in rows], dtype=np_module.int64)
    if not ids or len(set(ids)) != len(ids):
        raise AuthorTextCacheError("split IDs must be non-empty and unique")
    write_json(split_dir / "ids.json", ids)
    write_json(split_dir / "raw_labels.json", [row["raw_label"] for row in rows])
    write_json(split_dir / "transcript_hashes.json", [row["transcript_sha256"] for row in rows])
    with (split_dir / "labels.npy").open("wb") as handle:
        np_module.save(handle, labels, allow_pickle=False)


class TextFeatureMemmapWriter:
    """Stream mBERT features into the text_mlp-compatible split arrays."""

    def __init__(
        self,
        split_dir: Path,
        *,
        row_count: int,
        max_length: int,
        hidden_size: int,
        resume_offset: int = 0,
        np_module=None,
    ) -> None:
        if np_module is None:
            import numpy as np_module
        self.np = np_module
        if not 0 <= resume_offset <= row_count:
            raise AuthorTextCacheError("invalid text feature resume offset")
        mode = "r+" if resume_offset else "w+"
        self.features = self.np.lib.format.open_memmap(
            split_dir / "text.npy",
            mode=mode,
            dtype=self.np.float32,
            shape=(row_count, max_length, hidden_size),
        )
        self.lengths = self.np.lib.format.open_memmap(
            split_dir / "text_lengths.npy",
            mode=mode,
            dtype=self.np.int64,
            shape=(row_count,),
        )
        self.row_count = row_count
        self.max_length = max_length
        self.hidden_size = hidden_size

    def write_batch(self, begin: int, values, lengths) -> int:
        values = self.np.asarray(values, dtype=self.np.float32)
        lengths = self.np.asarray(lengths, dtype=self.np.int64)
        if values.ndim != 3 or values.shape[1:] != (self.max_length, self.hidden_size):
            raise AuthorTextCacheError("text batch does not match configured [N,T,D] shape")
        if lengths.shape != (values.shape[0],):
            raise AuthorTextCacheError("text lengths must have one value per batch row")
        if not self.np.isfinite(values).all():
            raise AuthorTextCacheError("text feature batch contains NaN or Inf")
        if self.np.any(lengths < 1) or self.np.any(lengths > self.max_length):
            raise AuthorTextCacheError("text lengths must be in [1, max_length]")
        end = begin + values.shape[0]
        if begin < 0 or end > self.row_count:
            raise AuthorTextCacheError("text feature write is outside split row bounds")
        self.features[begin:end] = values
        self.lengths[begin:end] = lengths
        self.features.flush()
        self.lengths.flush()
        return end

    def close(self) -> None:
        self.features.flush()
        self.lengths.flush()
        del self.features
        del self.lengths


def load_author_train_valid(author_manifest_dir: Path):
    """Load only author train/valid JSONL manifests and validate row provenance."""
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    all_ids: set[str] = set()
    source_csv_hashes: set[str] = set()
    for split in SPLITS:
        path = author_manifest_dir / "mosei_author_{}_manifest.jsonl".format(split)
        if not path.is_file():
            raise AuthorTextCacheError("missing author {} manifest: {}".format(split, path))
        hashes[split] = sha256_file(path)
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for row_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuthorTextCacheError(
                        "invalid JSON in {} row {}".format(split, row_number)
                    ) from exc
                if not isinstance(row, dict):
                    raise AuthorTextCacheError(
                        "{} row {} must be a JSON object".format(split, row_number)
                    )
                missing = [key for key in REQUIRED_FIELDS if key not in row]
                if missing:
                    raise AuthorTextCacheError(
                        "{} row {} missing fields: {}".format(
                            split, row_number, ", ".join(missing)
                        )
                    )
                identifier = row["ID"]
                if not isinstance(identifier, str) or not identifier.strip():
                    raise AuthorTextCacheError("{} row {} has invalid ID".format(split, row_number))
                if identifier != row["author_id"]:
                    raise AuthorTextCacheError("author_id and ID disagree for {}".format(identifier))
                if identifier in all_ids:
                    raise AuthorTextCacheError("duplicate author ID: {}".format(identifier))
                if row["dataset"] != "CMU-MOSEI" or row["cohort"] != AUTHOR_COHORT:
                    raise AuthorTextCacheError(
                        "{} is not from the author MOSEI cohort".format(identifier)
                    )
                if row["split"] != split:
                    raise AuthorTextCacheError(
                        "{} is stored in {} manifest but has split {!r}".format(
                            identifier, split, row["split"]
                        )
                    )
                if not isinstance(row["text"], str):
                    raise AuthorTextCacheError("{} transcript is not a string".format(identifier))
                if row["transcript_sha256"] != sha256_text(row["text"]):
                    raise AuthorTextCacheError("{} transcript hash mismatch".format(identifier))
                if row["class"] not in CLASSES:
                    raise AuthorTextCacheError("{} has invalid mapped class".format(identifier))
                if not isinstance(row["source_csv_sha256"], str) or not HASH_RE.fullmatch(
                    row["source_csv_sha256"]
                ):
                    raise AuthorTextCacheError("{} has invalid source CSV hash".format(identifier))
                all_ids.add(identifier)
                source_csv_hashes.add(row["source_csv_sha256"])
                records.append(row)
        if not records:
            raise AuthorTextCacheError("author {} manifest is empty".format(split))
        records_by_split[split] = records

    if len(source_csv_hashes) != 1:
        raise AuthorTextCacheError("train/valid manifests have inconsistent author CSV hashes")
    return records_by_split, hashes, next(iter(source_csv_hashes))


def load_pinned_mbert_definition() -> dict[str, Any]:
    """Reuse the project's pinned mBERT definition from the adjacent downloader."""
    downloader_path = SCRIPT_DIR / "download_encoders.py"
    if not downloader_path.is_file():
        raise AuthorTextCacheError(
            "download_encoders.py must be beside this script in the project scripts directory"
        )
    spec = importlib.util.spec_from_file_location("_jevaffect_encoder_definitions", downloader_path)
    if spec is None or spec.loader is None:
        raise AuthorTextCacheError("cannot load the adjacent encoder definition script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    models = getattr(module, "MODELS", None)
    if not isinstance(models, dict) or "mbert" not in models:
        raise AuthorTextCacheError("download_encoders.py has no pinned mbert definition")
    return dict(models["mbert"])


def verify_mbert_directory(model_dir: Path, definition: dict[str, Any]) -> dict[str, Any]:
    provenance_path = model_dir / "provenance.json"
    if not provenance_path.is_file():
        raise AuthorTextCacheError("encoder provenance.json is missing")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise AuthorTextCacheError("encoder provenance.json is invalid") from exc
    if (
        not provenance.get("verified")
        or provenance.get("repo") != definition.get("repo")
        or provenance.get("revision") != definition.get("revision")
        or provenance.get("sha256") != definition.get("sha256")
    ):
        raise AuthorTextCacheError("local encoder provenance does not match the pinned mBERT definition")
    weight_path = model_dir / definition["weight"]
    if not weight_path.is_file() or sha256_file(weight_path) != definition["sha256"]:
        raise AuthorTextCacheError("local mBERT weight is missing or has a SHA-256 mismatch")
    expected_file_hashes = provenance.get("files_sha256")
    if isinstance(expected_file_hashes, dict):
        for filename in definition.get("files", []):
            path = model_dir / filename
            if not path.is_file() or sha256_file(path) != expected_file_hashes.get(filename):
                raise AuthorTextCacheError("local encoder file hash mismatch: {}".format(filename))
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-manifest-dir", type=Path, required=True,
                        help="Directory with mosei_author_{train,valid}_manifest.jsonl")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Existing verified local mBERT directory; no download is attempted")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_cache(args: argparse.Namespace) -> dict[str, Any]:
    if not 4 <= args.max_length <= 512 or args.batch_size < 1:
        raise AuthorTextCacheError("invalid max length or batch size")
    records_by_split, manifest_hashes, author_csv_sha = load_author_train_valid(
        args.author_manifest_dir
    )
    definition = load_pinned_mbert_definition()
    provenance = verify_mbert_directory(args.model_dir, definition)

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version

    if args.device == "cuda" and not torch.cuda.is_available():
        raise AuthorTextCacheError("CUDA was requested but is not available")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    weight_sha = definition["sha256"]
    source_hashes = {
        "author_csv_sha256": author_csv_sha,
        "author_manifest_sha256": manifest_hashes,
        "encoder_sha256": weight_sha,
        "encoder_revision": provenance["revision"],
        "encoder_files_sha256": provenance.get("files_sha256", {}),
        "cohort": AUTHOR_COHORT,
        "transcript_condition": "author_provided_transcripts",
    }
    source_sha = combined_source_sha256(source_hashes)
    identity = {
        "dataset": "CMU-MOSEI",
        "cohort": AUTHOR_COHORT,
        "transcript_condition": "author_provided_transcripts",
        "source_sha256": source_sha,
        "source_hashes": source_hashes,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "numpy": np.__version__,
        "dtype": "float32",
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    progress_path = args.output / "progress.json"
    if args.output.exists():
        if not args.resume or not progress_path.is_file():
            raise AuthorTextCacheError("output exists; resume requires progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise AuthorTextCacheError("resume data/encoder/config identity mismatch")
        if progress.get("complete"):
            return {"complete": True, "already_complete": True,
                    "completed_rows": progress.get("completed_rows", {}),
                    "test_encoded": False}
    else:
        args.output.mkdir(parents=True)
        progress = {
            "identity": identity,
            "completed_rows": {},
            "truncated_rows": {},
            "complete": False,
        }
        write_json(progress_path, progress)

    model_dir = args.model_dir.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(args.device).eval()
    hidden = int(model.config.hidden_size)
    started = time.perf_counter()

    split_counts = {split: len(records_by_split[split]) for split in SPLITS}
    meta = build_text_mlp_cache_metadata(
        source_hashes,
        feature_dim=hidden,
        max_length=args.max_length,
        split_counts=split_counts,
        encoder=provenance,
        complete=False,
    )
    write_json(args.output / "meta.json", meta)

    for split in SPLITS:
        rows = records_by_split[split]
        split_dir = args.output / split
        split_dir.mkdir(exist_ok=True)
        write_split_sidecars(split_dir, rows, np)

        offset = int(progress["completed_rows"].get(split, 0))
        if offset < 0 or offset > len(rows):
            raise AuthorTextCacheError("invalid resume offset for {}".format(split))
        writer = TextFeatureMemmapWriter(
            split_dir,
            row_count=len(rows),
            max_length=args.max_length,
            hidden_size=hidden,
            resume_offset=offset,
            np_module=np,
        )
        for begin in range(offset, len(rows), args.batch_size):
            batch_rows = rows[begin:begin + args.batch_size]
            texts = [row["text"] for row in batch_rows]
            full_lengths = [
                len(tokens)
                for tokens in tokenizer(
                    texts, add_special_tokens=True, truncation=False
                )["input_ids"]
            ]
            batch = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            attention_mask = batch["attention_mask"]
            batch_lengths = attention_mask.sum(1).cpu().numpy()
            with torch.inference_mode():
                outputs = model(
                    **{key: value.to(args.device) for key, value in batch.items()}
                )
                hidden_states = outputs.last_hidden_state
                hidden_states = hidden_states * attention_mask.to(args.device).unsqueeze(-1)
                feature_values = hidden_states.float().cpu().numpy()
            if not np.isfinite(feature_values).all():
                raise AuthorTextCacheError("nonfinite mBERT features in {}".format(split))
            end = writer.write_batch(begin, feature_values, batch_lengths)
            progress["completed_rows"][split] = end
            progress["truncated_rows"][split] = progress["truncated_rows"].get(split, 0) + sum(
                length > args.max_length for length in full_lengths
            )
            write_json(progress_path, progress)
        writer.close()

    if args.device == "cuda":
        torch.cuda.synchronize()
    meta = build_text_mlp_cache_metadata(
        source_hashes,
        feature_dim=hidden,
        max_length=args.max_length,
        split_counts=split_counts,
        encoder=provenance,
        complete=True,
    )
    meta.update(
        truncated_rows=progress["truncated_rows"],
        encode_seconds_this_invocation=time.perf_counter() - started,
        peak_cuda_allocated_bytes=(
            torch.cuda.max_memory_allocated() if args.device == "cuda" else None
        ),
    )
    meta["artifact_sha256"] = {
        path.relative_to(args.output).as_posix(): sha256_file(path)
        for split in SPLITS
        for path in (args.output / split).iterdir()
        if path.is_file()
    }
    write_json(args.output / "meta.json", meta)
    progress["complete"] = True
    write_json(progress_path, progress)
    return {
        "complete": True,
        "cohort": AUTHOR_COHORT,
        "counts": progress["completed_rows"],
        "test_encoded": False,
        "cache_contract_compatible_for_text_mlp": True,
        "cache_contract_compatible_for_multimodal": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_cache(args)
    except (AuthorTextCacheError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        print("author MOSEI text cache failed: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
