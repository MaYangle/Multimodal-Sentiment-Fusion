"""Freeze a source-video-disjoint CH-SIMS holdout from official TRAIN only.

The official validation/test samples and labels are never read. This extra
diagnostic split addresses the near-total source-video overlap of the author's
train/validation protocol; it is not an official challenge score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import StratifiedGroupKFold


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def build(base: Path, output: Path) -> dict:
    base, output = base.resolve(), output.resolve()
    if output.exists() or output == base or base in output.parents:
        raise ValueError("Output must be a new independent cache directory")
    meta_path, detail_path = base / "meta.json", base / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    sidecar = {parts[1]: parts[0] for line in (base / "bundle.sha256").read_text(encoding="ascii").splitlines()
               if (parts := line.split())}
    if (sidecar != {"meta.json": sha(meta_path), "metadata.json": sha(detail_path)} or
            meta["cache_kind"] != "jevaffect_raw_npz_train_valid_full" or
            meta["source_descriptor"]["replacement_text_encoder"]["model"] != "hfl/chinese-macbert-base" or
            meta["split_counts"]["train"] != 2722):
        raise ValueError("Source is not the verified full MacBERT/raw-AV cohort")
    source_dir = base / "splits/train"
    source_ids = json.loads((source_dir / "ids.json").read_text(encoding="utf-8"))
    labels = np.load(source_dir / "labels.npy", allow_pickle=False)
    if len(source_ids) != 2722 or labels.shape != (2722,) or len(set(source_ids)) != 2722:
        raise ValueError("Official TRAIN IDs/labels are invalid")
    groups = np.asarray([sample_id.split("/")[0] for sample_id in source_ids])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=13)
    candidates = list(splitter.split(np.zeros((len(labels), 1)), labels, groups))
    selected = min(range(5), key=lambda index: (abs(len(candidates[index][1]) - len(labels) / 5), index))
    indices = {"train": np.sort(candidates[selected][0]), "valid": np.sort(candidates[selected][1])}
    if set(indices["train"]) & set(indices["valid"]) or set(indices["train"]) | set(indices["valid"]) != set(range(2722)):
        raise ValueError("Group partition does not cover official TRAIN exactly")
    group_sets = {split: {str(groups[index]) for index in selected_indices}
                  for split, selected_indices in indices.items()}
    if group_sets["train"] & group_sets["valid"]:
        raise ValueError("Source-video leakage in diagnostic holdout")
    for split, selected_indices in indices.items():
        if len(np.unique(labels[selected_indices])) != 5:
            raise ValueError(split + " lacks a sentiment class")
    manifest = {
        "scope": "derived solely from official CH-SIMS v2 TRAIN; official validation/test unused",
        "selection": "StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=13); hold fold closest to 20% of TRAIN by row count, lowest index on tie",
        "sklearn_version": sklearn_version, "selected_fold_index": selected,
        "source_cache_sha256": meta["source_sha256"], "source_meta_sha256": sha(meta_path),
        "source_train_ids_sha256": sha(source_dir / "ids.json"),
        "source_train_labels_sha256": sha(source_dir / "labels.npy"),
        "class_order": meta["class_order"], "source_group_cross_split_overlap": 0,
        "splits": {split: {"indices_into_official_train": selected_indices.tolist(),
                           "ids": [source_ids[int(i)] for i in selected_indices],
                           "count": len(selected_indices),
                           "source_group_count": len(group_sets[split]),
                           "class_counts": np.bincount(labels[selected_indices], minlength=5).tolist()}
                   for split, selected_indices in indices.items()},
        "official_valid_rows_used": 0, "official_test_rows_used": 0,
    }
    output.mkdir(parents=True)
    write_json(output / "group_holdout_manifest.json", manifest)
    manifest_sha = sha(output / "group_holdout_manifest.json")
    source_json = {name: json.loads((source_dir / name).read_text(encoding="utf-8"))
                   for name in ("raw_labels.json", "manifest_labels.json")}
    audit_rows = [json.loads(line) for line in (source_dir / "record_audit.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(audit_rows) != 2722:
        raise ValueError("Source audit row count differs from official TRAIN")
    for split, selected_indices in indices.items():
        destination = output / "splits" / split
        destination.mkdir(parents=True)
        chosen = selected_indices.tolist()
        write_json(destination / "ids.json", [source_ids[i] for i in chosen])
        for name, values in source_json.items():
            write_json(destination / name, [values[i] for i in chosen])
        (destination / "record_audit.jsonl").write_text(
            "".join(json.dumps(audit_rows[i], ensure_ascii=False, sort_keys=True) + "\n" for i in chosen),
            encoding="utf-8")
        write_json(destination / "id_mapping.json", [
            {"ID": source_ids[i], "official_train_index": i} for i in chosen])
        for array_path in sorted(source_dir.glob("*.npy")):
            source_array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if source_array.shape[0] != 2722:
                raise ValueError(array_path.name + " does not align with official TRAIN")
            destination_array = np.lib.format.open_memmap(destination / array_path.name, mode="w+",
                                                        dtype=source_array.dtype,
                                                        shape=(len(chosen),) + source_array.shape[1:])
            for start in range(0, len(chosen), 64):
                subset = chosen[start:start + 64]
                destination_array[start:start + len(subset)] = source_array[subset]
            destination_array.flush()
            del destination_array
    source_descriptor = {"derived_from_official_train_only": True,
                         "parent_full_cache_source_sha256": meta["source_sha256"],
                         "parent_meta_sha256": sha(meta_path),
                         "group_manifest_sha256": manifest_sha,
                         "author_validation_and_test_labels_used": False,
                         "backbones": "MacBERT-base/WavLM-base-plus/DINOv2-small, frozen"}
    new_source = canonical_sha(source_descriptor)
    group_meta = {"schema_version": 1, "cache_kind": "jevaffect_group_disjoint_train_holdout_v1",
                  "status": "train_only_diagnostic_not_official_validation",
                  "dataset": "CH-SIMS-v2.0", "language": "zh",
                  "class_order": meta["class_order"], "feature_dims": meta["feature_dims"],
                  "sequence_lengths": meta["sequence_lengths"], "text_features_ready": True,
                  "split_paths": {"train": "splits/train", "valid": "splits/valid"},
                  "split_counts": {split: len(values) for split, values in indices.items()},
                  "source_descriptor": source_descriptor, "source_sha256": new_source,
                  "test_data": "not included", "official_validation_data": "not included"}
    files = {}
    for path in sorted((output / "splits").rglob("*")):
        if path.is_file():
            files[path.relative_to(output).as_posix()] = {"sha256": sha(path),
                                                            "size_bytes": path.stat().st_size}
    group_detail = {"files": files, "source_sha256": new_source,
                    "source_descriptor": source_descriptor, "class_order": meta["class_order"],
                    "group_manifest_sha256": manifest_sha,
                    "official_valid_or_test_data_read": False}
    write_json(output / "meta.json", group_meta)
    write_json(output / "metadata.json", group_detail)
    (output / "bundle.sha256").write_text(
        sha(output / "meta.json") + "  meta.json\n" +
        sha(output / "metadata.json") + "  metadata.json\n", encoding="ascii")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jevaffect.training import FeatureSplit, _cache_metadata
    checked, _ = _cache_metadata(output, ("text", "audio", "vision"))
    train = FeatureSplit(output, "train", checked)
    valid = FeatureSplit(output, "valid", checked)
    if {sample_id.split("/")[0] for sample_id in train.ids} & {sample_id.split("/")[0] for sample_id in valid.ids}:
        raise ValueError("Training reader revealed source-group overlap")
    report = {"output": str(output), "source_sha256": new_source,
              "group_manifest_sha256": manifest_sha, "selected_fold_index": selected,
              "split_counts": group_meta["split_counts"],
              "class_counts": {split: manifest["splits"][split]["class_counts"] for split in indices},
              "source_group_cross_split_overlap": 0,
              "training_reader_passed": True, "official_valid_or_test_used": False}
    write_json(output / "group_holdout_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.output), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
