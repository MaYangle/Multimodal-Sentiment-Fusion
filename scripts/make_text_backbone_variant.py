"""Build an independently hashed CH-SIMS raw-feature cache with MacBERT text.

Only official train/valid records are used. Audio and vision are byte copies of
the existing frozen WavLM/DINOv2 cache; every text sequence comes from the
revision-pinned MacBERT provided-transcript cache. Source caches are immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def build(base: Path, text: Path, output: Path) -> dict:
    base, text, output = base.resolve(), text.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Output exists; source caches are never overwritten: " + str(output))
    if output in (base, text) or base in output.parents or text in output.parents:
        raise ValueError("Output must be a new sibling, not inside either source cache")
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    detail = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    tmeta = json.loads((text / "meta.json").read_text(encoding="utf-8"))
    if meta["cache_kind"] != "jevaffect_raw_npz_train_valid_full" or meta["split_counts"] != {"train": 2722, "valid": 647}:
        raise ValueError("Base is not the frozen full CH-SIMS train/valid cache")
    if not tmeta.get("complete") or not tmeta.get("text_features_ready") or tmeta.get("test_encoded") is not False:
        raise ValueError("Text cache is not complete, verified, train/valid-only output")
    if tmeta["encoder"]["model"] != "hfl/chinese-macbert-base" or tmeta["source_hashes"]["csv"] != meta["source_descriptor"]["official_csv_sha256"]:
        raise ValueError("Text encoder or official CSV does not match base cache")
    if tmeta["feature_dims"]["text"] != 768 or tmeta["sequence_lengths"]["text"] != 64:
        raise ValueError("Text tensor dimensions differ from comparison backbone")
    if tmeta["class_order"] != meta["class_order"]:
        raise ValueError("Class order differs")
    per_split = {}
    for split in ("train", "valid"):
        bdir, tdir = base / "splits" / split, text / split
        ids_b = json.loads((bdir / "ids.json").read_text(encoding="utf-8"))
        ids_t = json.loads((tdir / "ids.json").read_text(encoding="utf-8"))
        if ids_b != ids_t or len(ids_b) != meta["split_counts"][split]:
            raise ValueError(split + " MacBERT and raw-feature ID order/count differ")
        if not np.array_equal(np.load(bdir / "labels.npy", allow_pickle=False),
                              np.load(tdir / "labels.npy", allow_pickle=False)):
            raise ValueError(split + " labels differ")
        values = np.load(tdir / "text.npy", mmap_mode="r", allow_pickle=False)
        lengths = np.load(tdir / "text_lengths.npy", allow_pickle=False)
        if values.shape != (len(ids_b), 64, 768) or lengths.shape != (len(ids_b),) or not np.isfinite(values).all():
            raise ValueError(split + " replacement text tensor invalid")
        if np.any(lengths < 1) or np.any(lengths > 64):
            raise ValueError(split + " replacement text lengths invalid")
        per_split[split] = {"rows": len(ids_b), "max_text_length": int(lengths.max()),
                            "text_tensor_sha256": sha(tdir / "text.npy")}
    output.mkdir(parents=True)
    changed = {}
    for split in ("train", "valid"):
        bdir, tdir, odir = base / "splits" / split, text / split, output / "splits" / split
        odir.mkdir(parents=True)
        for original in bdir.iterdir():
            if original.is_file() and original.name not in ("text.npy", "text_mask.npy", "text_lengths.npy", "record_audit.jsonl"):
                shutil.copy2(original, odir / original.name)
        for name in ("text.npy", "text_lengths.npy"):
            shutil.copy2(tdir / name, odir / name)
        lengths = np.load(odir / "text_lengths.npy", allow_pickle=False)
        mask = np.arange(64)[None, :] < lengths[:, None]
        np.save(odir / "text_mask.npy", mask.astype(np.bool_), allow_pickle=False)
        old_rows = [json.loads(line) for line in (bdir / "record_audit.jsonl").read_text(encoding="utf-8").splitlines() if line]
        ids = json.loads((odir / "ids.json").read_text(encoding="utf-8"))
        if len(old_rows) != len(ids) or any(row["ID"] != sample_id for row, sample_id in zip(old_rows, ids, strict=True)):
            raise ValueError(split + " source record-audit IDs differ")
        with (odir / "record_audit.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row, length in zip(old_rows, lengths, strict=True):
                row["original_raw_artifact_text_length"] = row["raw_lengths"]["text"]
                row["raw_lengths"]["text"] = int(length)
                row["kept_lengths"]["text"] = int(length)
                row["truncated"]["text"] = 0
                row["raw_artifact_role"] = "audio/vision provenance; mBERT text replaced by MacBERT cache"
                row["text_override_source_sha256"] = tmeta["source_sha256"]
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        changed[split] = per_split[split]
    source_descriptor = {
        "derivation": "verified train/valid text-backbone substitution; audio and vision copied byte-for-byte",
        "base_raw_cache_source_sha256": meta["source_sha256"],
        "base_raw_cache_meta_sha256": sha(base / "meta.json"),
        "base_raw_cache_descriptor": meta["source_descriptor"],
        "replacement_text_cache_source_sha256": tmeta["source_sha256"],
        "replacement_text_cache_meta_sha256": sha(text / "meta.json"),
        "replacement_text_encoder": tmeta["encoder"],
        "official_csv_sha256": tmeta["source_hashes"]["csv"],
        "text_condition": "provided transcripts, not ASR",
        "test_data_encoded": False,
    }
    source_sha = canonical_sha({"kind": "macbert-text-raw-av-v1", "sources": source_descriptor})
    for item in (meta, detail):
        item["source_descriptor"] = source_descriptor
        item["source_sha256"] = source_sha
        item["truncation"]["text"] = {"observed_max_raw_steps": max(x["max_text_length"] for x in per_split.values()),
                                      "policy": "MacBERT tokenizer truncates at 64 tokens; no record exceeded cap",
                                      "target_steps": 64, "total_dropped_frames_or_tokens": 0, "truncated_records": 0}
    meta["cache_kind"] = "jevaffect_raw_npz_train_valid_full"
    meta["text_encoder_override"] = "hfl/chinese-macbert-base"
    detail["builder_sha256"] = sha(Path(__file__))
    detail["files"] = {}
    for path in sorted((output / "splits").rglob("*")):
        if path.is_file():
            relative = path.relative_to(output).as_posix()
            detail["files"][relative] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    write_json(output / "meta.json", meta)
    write_json(output / "metadata.json", detail)
    (output / "bundle.sha256").write_text(
        sha(output / "meta.json") + "  meta.json\n" + sha(output / "metadata.json") + "  metadata.json\n",
        encoding="ascii")
    # Reuse the same strict loader that training calls; it rehashes all arrays.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jevaffect.cmc_raw_cache import load_raw_feature_pilot_train_valid
    train, valid, identity = load_raw_feature_pilot_train_valid(output)
    if len(train) != 2722 or len(valid) != 647 or identity["source_sha256"] != source_sha:
        raise ValueError("Constructed cache failed the frozen full-cohort contract")
    train.close()
    valid.close()
    report = {"output": str(output), "source_sha256": source_sha,
              "audio_vision_source_sha256": meta["source_descriptor"]["base_raw_cache_source_sha256"],
              "text_source_sha256": tmeta["source_sha256"], "splits": changed,
              "strict_loader_passed": True, "test_scored": False}
    write_json(output / "variant_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.text_cache, args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
