"""Inspect a completed trusted publisher feature package without scoring test.

Run only after the whole-payload receipt and checksum exist. This records the
actual schema; it does not guess ID crosswalks or silently drop empty text.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trust-author-pickle", action="store_true")
    a = p.parse_args()
    if not a.trust_author_pickle:
        p.error("Use --trust-author-pickle only for the recorded publisher package")
    receipt = json.loads(a.receipt.read_text())
    if not receipt.get("transfer_complete") or a.source.stat().st_size != receipt["bytes"]:
        raise ValueError("A complete publisher transfer receipt is required")
    with a.source.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != receipt["sha256"]:
        raise ValueError("Payload checksum differs from completed transfer")
    with a.source.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or not {"train", "valid", "test"} <= set(data):
        raise ValueError("Expected publisher train/valid/test dictionary")
    report = {"source_sha256": actual, "receipt_sha256": hashlib.sha256(a.receipt.read_bytes()).hexdigest(),
              "root_keys": [str(k) for k in data], "splits": {}, "test_scored": False,
              "identity_join_validated": False, "mapped_labels_created": False}
    for split in ("train", "valid", "test"):
        if not isinstance(data[split], dict):
            raise ValueError("Unexpected split object")
        fields = {}; samples = set()
        for name, value in data[split].items():
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            if isinstance(value, (list, tuple, np.ndarray)):
                array = np.asarray(value)
                fields[str(name)] = {"shape": list(array.shape), "dtype": str(array.dtype)}
                if array.ndim:
                    samples.add(array.shape[0])
                if name in ("id", "ids", "video_id"):
                    fields[str(name)]["first_ids_repr"] = [repr(v)[:120] for v in array[:3]]
                if split != "test" and name in ("labels", "regression_labels"):
                    if array.dtype.kind not in "fi":
                        raise ValueError("Author regression score field must be numeric")
                    fields[str(name)]["finite_count"] = int(np.isfinite(array).sum())
                    fields[str(name)]["min"] = float(np.min(array))
                    fields[str(name)]["max"] = float(np.max(array))
            else:
                fields[str(name)] = {"type": type(value).__name__}
        report["splits"][split] = {"fields": fields, "observed_leading_counts": sorted(samples)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
