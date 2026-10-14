"""Known TRAIN-only pooled-coordinate normalization on verified source cache."""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.author_features import AuthorFeatureSplit, MODALITIES
from jevaffect.feature_standardization import fit_pooled_standardization, standardize_sequence
from jevaffect.native_training import digest, save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); a = p.parse_args()
    source_meta = a.cache_dir / "metadata.json"; meta = json.loads(source_meta.read_text())
    if meta["normalization"] != "none" or a.output_dir.resolve() == a.cache_dir.resolve():
        raise ValueError("Separate raw source and new normalized output required")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise ValueError("Inspect/reuse existing output explicitly; never overwrite")
    dataset = AuthorFeatureSplit(a.cache_dir, "train")
    try:
        stats = {m: fit_pooled_standardization(dataset.features[m]) for m in MODALITIES}
    finally:
        dataset.close()
    a.output_dir.mkdir(parents=True, exist_ok=True); result_meta = copy.deepcopy(meta); result_meta["artifact_sha256"] = {}
    for key, expected in meta["artifact_sha256"].items():
        path = a.cache_dir / key
        if digest(path) != expected:
            raise ValueError("Raw source cache changed: " + key)
        if path.name in (m + ".npy" for m in MODALITIES):
            continue
        target = a.output_dir / key; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        result_meta["artifact_sha256"][key] = digest(target)
    save_json(a.output_dir / "normalization.json", stats)
    result_meta["artifact_sha256"]["normalization.json"] = digest(a.output_dir / "normalization.json")
    for split in ("train", "valid"):
        for m in MODALITIES:
            source = np.load(a.cache_dir / split / (m + ".npy"), mmap_mode="r")
            lengths = np.load(a.cache_dir / split / (m + "_lengths.npy"))
            path = a.output_dir / split / (m + ".npy")
            output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=source.shape)
            quality_zero = np.zeros(len(source), bool)
            for i, length in enumerate(lengths):
                quality_zero[i] = not np.any(source[i, :int(length)] != 0)
                output[i] = standardize_sequence(source[i], int(length), stats[m])
            output.flush(); output._mmap.close(); source._mmap.close()
            result_meta["artifact_sha256"][path.relative_to(a.output_dir).as_posix()] = digest(path)
            quality_path = path.with_name(m + "_raw_all_zero.npy"); np.save(quality_path, quality_zero)
            result_meta["artifact_sha256"][quality_path.relative_to(a.output_dir).as_posix()] = digest(quality_path)
    result_meta["normalization"] = {"kind": "TRAIN clip-pooled coordinate mean/population STD", "fit_split": "train",
        "fit_ids_sha256": meta["artifact_sha256"]["train/ids.json"], "source_cache_metadata_sha256": digest(source_meta),
        "preparation_sha256": digest(__file__), "implementation_sha256": digest(Path(__file__).parents[1] / "jevaffect/feature_standardization.py")}
    result_meta["input_form"] += "; known TRAIN-only pooled-coordinate standardization"
    save_json(a.output_dir / "metadata.json", result_meta); print(json.dumps(result_meta["normalization"], indent=2))
