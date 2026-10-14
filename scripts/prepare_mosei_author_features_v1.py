"""Prepare TRAIN/VALID and label-free frozen TEST IDs from complete publisher PKL."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.author_features import aligned_suffix, temporal_ids, COHORT, MODALITIES
from jevaffect.native_training import digest, save_json

SOURCE_SHA = "31007381c01dd346945ed8a64c7321239de5aaf1ef05bf112e952848a823186d"

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--source", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--trust-author-pickle", action="store_true"); a = p.parse_args()
    if not a.trust_author_pickle:
        p.error("Only prepare the recorded trusted publisher package")
    receipt = json.loads(a.receipt.read_text()); source_sha = digest(a.source)
    if source_sha != SOURCE_SHA or receipt.get("sha256") != source_sha or not receipt.get("transfer_complete") or receipt["bytes"] != a.source.stat().st_size:
        raise ValueError("Complete pinned publisher package required")
    metadata_path = a.output_dir / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta["source_sha256"] != source_sha or meta["preparation_sha256"] != digest(__file__) or meta["adapter_sha256"] != digest(Path("jevaffect/author_features.py")):
            raise ValueError("Existing cache differs; use a new output directory")
        for key, expected in meta["artifact_sha256"].items():
            if digest(a.output_dir / key) != expected:
                raise ValueError("Existing cache integrity mismatch")
        print("Verified prepared cache reused"); raise SystemExit(0)
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Incomplete output directory: inspect instead of overwrite")
    with a.source.open("rb") as stream:
        data = pickle.load(stream)
    expected = {"train": 16265, "valid": 1869, "test": 4643}
    ids = {s: temporal_ids(data[s]["id"]) for s in expected}
    if any(len(ids[s]) != expected[s] for s in expected) or any(set(ids[s]) & set(ids[t]) for s, t in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Published cohort changed or overlaps")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": 1, "cohort": COHORT, "dataset": "mosei", "source_sha256": source_sha,
            "receipt_sha256": digest(a.receipt), "split_counts": expected, "feature_shapes": {}, "split_paths": {s: s for s in ("train", "valid")},
            "text_pooling": "valid_mean", "input_form": "author-linked aligned50 GloVe300/acoustic74/vision35",
            "padding": "aligned suffix shifted forward; zero suffix padding; interior zeros retained",
            "normalization": "none", "test_labels_or_features_exported": False,
            "source_fold_variant": "retain published split; four -NFrJFQijFE train records outside current SDK lists",
            "preparation_sha256": digest(__file__), "adapter_sha256": digest(Path("jevaffect/author_features.py")),
            "public_loader": json.loads(Path("docs/mosei_multibench_cohort_audit.json").read_text())["public_loader"],
            "preparation": {}, "artifact_sha256": {}}
    for split in ("train", "valid"):
        part = data[split]; prepared, lengths, starts, counters = aligned_suffix(part)
        y = np.asarray(part["labels"], np.float32).reshape(-1)
        if y.shape != (len(ids[split]),) or not np.isfinite(y).all() or np.any(np.abs(y) > 3):
            raise ValueError("Invalid original native scores")
        directory = a.output_dir / split; directory.mkdir()
        save_json(directory / "ids.json", ids[split]); np.save(directory / "targets.npy", y)
        np.save(directory / "source_leading_padding.npy", starts)
        for m in MODALITIES:
            np.save(directory / (m + ".npy"), prepared[m]); np.save(directory / (m + "_lengths.npy"), lengths)
            meta["feature_shapes"][m] = list(prepared[m].shape[1:])
        for path in directory.iterdir():
            meta["artifact_sha256"][path.relative_to(a.output_dir).as_posix()] = digest(path)
        meta["preparation"][split] = {"count": len(y), "leading_padding_removed": int(starts.sum()), "quality_counts": counters}
        print(json.dumps({"split": split, "count": len(y), "quality_counts": counters}), flush=True)
    test_path = a.output_dir / "frozen_test_ids.json"; save_json(test_path, ids["test"])
    meta["artifact_sha256"][test_path.name] = digest(test_path)
    save_json(metadata_path, meta); print(json.dumps(meta, indent=2))
