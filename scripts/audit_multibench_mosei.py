"""Audit complete author-linked temporal-ID package against pinned SDK folds."""
import argparse
import ast
import hashlib
import json
import pickle
from decimal import Decimal
from pathlib import Path
import numpy as np
import requests

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--source", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trust-author-pickle", action="store_true"); a = p.parse_args()
    if not a.trust_author_pickle:
        p.error("Only inspect the recorded trusted publisher package")
    receipt = json.loads(a.receipt.read_text())
    with a.source.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if not receipt.get("transfer_complete") or receipt["sha256"] != sha or receipt["bytes"] != a.source.stat().st_size:
        raise ValueError("Whole publisher transfer and checksum required")
    pinned = json.loads(Path("docs/author_channel_probe.json").read_text())["mosei_folds"]
    response = requests.get(pinned["source_url"], timeout=30); response.raise_for_status()
    if hashlib.sha256(response.content).hexdigest() != pinned["source_sha256"]:
        raise ValueError("Pinned SDK fold source changed")
    code_root = Path("work/mosei_author_source_audit"); code_root.mkdir(parents=True, exist_ok=True)
    (code_root / "cmu_mosei_std_folds.py").write_bytes(response.content)
    folds = {}
    for node in ast.parse(response.text).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("standard_train_fold", "standard_valid_fold", "standard_test_fold"):
                    folds[target.id] = set(ast.literal_eval(node.value))
    with a.source.open("rb") as f:
        data = pickle.load(f)
    report = {"source_sha256": sha, "receipt_sha256": hashlib.sha256(a.receipt.read_bytes()).hexdigest(),
              "sdk_revision": pinned["sdk_revision"], "sdk_fold_sha256": pinned["source_sha256"],
              "scope": "actual temporal IDs and source folds; no ordinal joins or test scoring", "test_scored": False,
              "splits": {}, "identity_cohort": "MultiBench author-linked processed temporal-ID cohort; separate from MMSA ordinal CSV"}
    mapping = {"train": "standard_train_fold", "valid": "standard_valid_fold", "test": "standard_test_fold"}
    all_ids = set(); source_sets = {}
    for split in ("train", "valid", "test"):
        part = data[split]; ids = np.asarray(part["id"])
        if ids.ndim != 2 or ids.shape[1] != 3:
            raise ValueError("Expected video/start/end source IDs")
        canonical = []; sources = set(); errors = []; invalid_intervals = []
        negative_start_count, nonpositive_duration_count = 0, 0
        for video, start, end in ids:
            if not video or "|" in video or not Decimal(start).is_finite() or not Decimal(end).is_finite():
                raise ValueError("Nonfinite or malformed temporal source identity")
            if not Decimal(0) <= Decimal(start) < Decimal(end):
                invalid_intervals.append([str(video), str(start), str(end)])
            negative_start_count += int(Decimal(start) < 0)
            nonpositive_duration_count += int(Decimal(end) <= Decimal(start))
            # Preserve original exact timestamp strings; no ordinal guesswork.
            canonical.append("|".join((video, start, end))); sources.add(video)
            if video not in folds[mapping[split]]:
                errors.append(video)
        duplicate_ids = len(canonical) - len(set(canonical))
        cross_split_duplicates = len(all_ids & set(canonical))
        all_ids.update(canonical); source_sets[split] = sources
        fields = {}
        for m in ("text", "audio", "vision"):
            x = np.asarray(part[m])
            if x.ndim != 3 or x.shape[0] != len(ids):
                raise ValueError("Feature shape differs from IDs")
            finite = np.isfinite(x)
            nonzero = np.any(x != 0, axis=-1)
            fields[m] = {"shape": list(x.shape), "dtype": str(x.dtype), "nan_count": int(np.isnan(x).sum()),
                         "positive_inf_count": int(np.isposinf(x).sum()), "negative_inf_count": int(np.isneginf(x).sum()),
                         "all_zero_records": int((~nonzero.any(1)).sum()), "all_zero_steps": int((~nonzero).sum()),
                         "nonfinite_records": int((~finite).any(axis=(1, 2)).sum())}
            if m == "text":
                fields[m]["author_sum_zero_drop_count"] = int((x.sum(axis=(1, 2)) == 0).sum())
                fields[m]["leading_zero_record_count"] = int((~nonzero[:, 0]).sum())
                fields[m]["interior_zero_record_count"] = int(sum(bool(n.any()) and bool((~n[np.flatnonzero(n)[0]:]).any()) for n in nonzero))
        row = {"count": len(ids), "source_videos": len(sources), "SDK_fold_membership_errors": len(errors),
               "invalid_interval_count": len(invalid_intervals), "first_invalid_intervals": invalid_intervals[:10],
               "negative_start_count": negative_start_count, "nonpositive_duration_count": nonpositive_duration_count,
               "duplicate_temporal_ids": duplicate_ids, "cross_split_duplicate_ids": cross_split_duplicates,
               "first_wrong_sources": sorted(set(errors))[:10], "temporal_ids_sha256": hashlib.sha256(json.dumps(canonical).encode()).hexdigest(), "features": fields}
        if split != "test":
            y = np.asarray(part["labels"], np.float64).reshape(-1)
            if y.shape != (len(ids),) or not np.isfinite(y).all() or np.any(np.abs(y) > 3):
                raise ValueError("Invalid native continuous sentiment scores")
            row["labels"] = {"shape": list(np.asarray(part["labels"]).shape), "min": float(y.min()), "max": float(y.max()), "unique_scores": len(np.unique(y))}
        report["splits"][split] = row
    overlaps = {f"{s}+{t}": sorted(source_sets[s] & source_sets[t]) for s, t in (("train", "valid"), ("train", "test"), ("valid", "test"))}
    report["source_split_overlaps"] = overlaps
    report["SDK_source_splits_passed"] = not any(overlaps.values()) and not any(r["SDK_fold_membership_errors"] for r in report["splits"].values())
    report["temporal_identity_quality_passed"] = not any(r["invalid_interval_count"] or r["duplicate_temporal_ids"] or r["cross_split_duplicate_ids"] for r in report["splits"].values())
    # Pin public loader as evidence only; never import its dependency/execution path.
    try:
        latest = requests.get("https://api.github.com/repos/pliang279/MultiBench/commits/main", timeout=30); latest.raise_for_status()
        revision = latest.json()["sha"]; url = f"https://raw.githubusercontent.com/pliang279/MultiBench/{revision}/datasets/affect/get_data.py"
        loader = requests.get(url, timeout=30); loader.raise_for_status(); (code_root / "multibench_get_data.py").write_bytes(loader.content)
        report["public_loader"] = {"revision": revision, "url": url, "sha256": hashlib.sha256(loader.content).hexdigest(), "imported_or_executed": False}
    except requests.RequestException as error:
        report["public_loader_error"] = type(error).__name__
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))
