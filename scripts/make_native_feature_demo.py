"""Create two label-free VALID feature fixtures locally; never opens test."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from jevaffect.native_training import digest

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); a = p.parse_args()
    metadata = a.cache_dir / "metadata.json"
    if not metadata.exists():
        metadata = a.cache_dir / "meta.json"
    meta = json.loads(metadata.read_text()); directory = a.cache_dir / meta.get("split_paths", {}).get("valid", "splits/valid" if (a.cache_dir / "splits").exists() else "valid")
    directory.resolve().relative_to(a.cache_dir.resolve())
    ids = json.loads((directory / "ids.json").read_text()); records = []; a.output_dir.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(ids[:2]):
        arrays = {}
        for m in ("text", "audio", "vision"):
            fp, lp = directory / (m + ".npy"), directory / (m + "_lengths.npy")
            for path in (fp, lp):
                key = path.relative_to(a.cache_dir).as_posix()
                expected = meta.get("artifact_sha256", {}).get(key, meta.get("files", {}).get(key, {}).get("sha256"))
                if digest(path) != expected:
                    raise ValueError("Demo cache integrity mismatch")
            x = np.load(fp, mmap_mode="r"); lengths = np.load(lp, mmap_mode="r")
            arrays[m] = np.array(x[i], copy=True); arrays[m + "_length"] = np.array(lengths[i], copy=True)
            x._mmap.close(); lengths._mmap.close()
        file = f"example_{i}.npz"; np.savez_compressed(a.output_dir / file, **arrays)
        records.append({"id": sid, "features_npz": file, "metadata_sha256": digest(metadata)})
    (a.output_dir / "input.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps({"count": len(records), "split": "valid", "labels_read": False, "directory": str(a.output_dir)}))
