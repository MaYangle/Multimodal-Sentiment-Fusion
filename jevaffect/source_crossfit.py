"""Explicit source-disjoint cohorts WITHIN official training; never relabel splits."""
import csv
import hashlib
import json
import numpy as np
from torch.utils.data import Subset


def training_rows_only(path):
    result = {}
    with open(path, encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != "train":
                continue  # Do not parse official validation or test labels.
            sid = row["video_id"] + "/" + row["clip_id"]
            score = float(row["label"])
            if sid in result or not np.isfinite(score) or abs(score) > 1:
                raise ValueError("Invalid or duplicate native training score")
            result[sid] = {"split": "train", "score": score}
    return result


def group_of(sid):
    if sid.count("/") != 1 or not all(sid.split("/")):
        raise ValueError("Expected Chinese source/clip ID")
    return sid.split("/")[0]


def make_folds(ids, fold_count=3, seed=20261012, inner_fraction=.1):
    ids = list(ids)
    if len(set(ids)) != len(ids) or not 0 < inner_fraction < 1 or fold_count < 2:
        raise ValueError("Invalid cross-fit cohort")
    groups = {}
    for sid in ids:
        groups.setdefault(group_of(sid), []).append(sid)
    if len(groups) < fold_count + 2:
        raise ValueError("Too few sources for disjoint optimization/selection/OOF")
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(fold_count)]; sizes = np.zeros(fold_count, int)
    for g in rng.permutation(sorted(groups)):
        k = int(np.argmin(sizes)); buckets[k].append(str(g)); sizes[k] += len(groups[g])
    folds = []
    for k in range(fold_count):
        held = set(buckets[k]); remaining = sorted(set(groups) - held)
        inner_rng = np.random.default_rng(seed + k + 1)
        n = max(1, int(np.ceil(len(remaining) * inner_fraction)))
        if n >= len(remaining):
            raise ValueError("Empty inner optimization cohort")
        select = set(inner_rng.permutation(remaining)[:n].tolist())
        fit = set(remaining) - select
        cohort = {"fold": k, "fit_ids": [s for s in ids if group_of(s) in fit],
                  "select_ids": [s for s in ids if group_of(s) in select],
                  "oof_ids": [s for s in ids if group_of(s) in held]}
        folds.append(cohort)
    validate_folds(ids, folds)
    return {"scope": "official TRAIN only; inner selection is not official validation",
            "seed": seed, "inner_fraction": inner_fraction, "source_count": len(groups), "folds": folds}


def validate_folds(ids, folds):
    universe = set(ids); seen = []
    for fold in folds:
        splits = [fold[key] for key in ("fit_ids", "select_ids", "oof_ids")]
        if any(not x or len(set(x)) != len(x) for x in splits):
            raise ValueError("Empty or duplicate cohort")
        sets = [set(x) for x in splits]; sources = [{group_of(s) for s in x} for x in splits]
        if set.union(*sets) != universe or any(sources[i] & sources[j] for i in range(3) for j in range(i)):
            raise ValueError("Missing IDs or source leakage")
        seen.extend(splits[2])
    if len(seen) != len(ids) or set(seen) != universe:
        raise ValueError("OOF must cover every training ID exactly once")


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def subset(dataset, ids):
    index = {sid: i for i, sid in enumerate(dataset.ids)}
    if len(set(ids)) != len(ids) or not set(ids) <= set(index):
        raise ValueError("Invalid subset")
    return Subset(dataset, [index[s] for s in ids])
