"""TRAIN-only author unimodal annotation coverage and source-held error slices."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import SOURCE_CSV_SHA256
from jevaffect.native_training import digest, save_json

CHANNELS = ("M", "T", "A", "V")


def load_train_annotations(path):
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] != "train":
                continue  # Do not parse any VALID/TEST annotation score.
            sid = row["video_id"] + "/" + row["clip_id"]
            if sid in result:
                raise ValueError("Duplicate TRAIN author ID")
            scores = {}
            for channel in CHANNELS:
                raw = row["label" if channel == "M" else "label_"+channel].strip()
                if raw.lower() in ("", "nan", "none", "null"):
                    scores[channel] = None
                    continue
                value = float(raw)
                if not np.isfinite(value) or abs(value) > 1:
                    raise ValueError("Invalid native annotation: " + sid + "/" + channel)
                scores[channel] = value
            if scores["M"] is None:
                raise ValueError("Missing primary native TRAIN score")
            result[sid] = scores
    if not result:
        raise ValueError("TRAIN annotation cohort is empty")
    return result


def summarize(annotations):
    ids = sorted(annotations)
    arrays = {c: np.array([np.nan if annotations[s][c] is None else annotations[s][c] for s in ids], np.float32) for c in CHANNELS}
    complete = np.isfinite(np.stack(list(arrays.values()))).all(0)
    five = {c: np.searchsorted(np.array([-.7,-.1,.1,.7],np.float32), x, side="left") for c,x in arrays.items()}
    slices = {"all_train": np.ones(len(ids),bool), "all_channels_available": complete,
        "unimodal_sign_agreement": complete & ((arrays["T"]>0)==(arrays["A"]>0)) & ((arrays["T"]>0)==(arrays["V"]>0)),
        "unimodal_sign_conflict": complete & (((arrays["T"]>0)!=(arrays["A"]>0)) | ((arrays["T"]>0)!=(arrays["V"]>0))),
        "text_M_sign_conflict": complete & ((arrays["T"]>0)!=(arrays["M"]>0)),
        "text_M_five_bin_conflict": complete & (five["T"]!=five["M"])}
    coverage = {}
    for c,x in arrays.items():
        good = np.isfinite(x)
        values,counts = np.unique(x[good],return_counts=True)
        coverage[c] = {"available": int(good.sum()), "missing": int((~good).sum()),
            "observed_scores": [{"score":float(v),"count":int(n)} for v,n in zip(values,counts)]}
    differences = {}
    for c in ("T","A","V"):
        good = np.isfinite(arrays[c])
        differences[c] = {"paired_count":int(good.sum()),"mean_abs_difference_from_M":float(np.abs(arrays[c][good]-arrays["M"][good]).mean()) if good.any() else None,
            "sign_match_with_M":float(np.mean((arrays[c][good]>0)==(arrays["M"][good]>0))) if good.any() else None,
            "five_bin_match_with_M":float(np.mean(five[c][good]==five["M"][good])) if good.any() else None}
    return ids,arrays,slices,{"count":len(ids),"channels":coverage,"channel_M_agreement":differences,"slice_counts":{k:int(v.sum()) for k,v in slices.items()}}


def diagnose(author_csv, oof_suite, output):
    if digest(author_csv) != SOURCE_CSV_SHA256:
        raise ValueError("Pinned author CSV required")
    annotations = load_train_annotations(author_csv)
    if len(annotations) != 2722:
        raise ValueError("Complete native TRAIN annotation cohort required")
    ids,arrays,slices,description = summarize(annotations)
    suite = json.loads(Path(oof_suite).read_text(encoding="utf-8"))
    if not suite.get("suite_complete") or suite.get("smoke_only") or suite.get("test_evaluated") or suite.get("official_VALID_used"):
        raise ValueError("Completed TRAIN-only source-held OOF suite required")
    errors = []
    records = []
    for seed in (13,42,2026):
        record = suite["oof"][f"fusion/seed_{seed}"]
        path = Path(record["path"])
        if digest(path) != record["sha256"] or record["count"] != 2722 or not record["coverage_exactly_once"] or not record["complete"]:
            raise ValueError("OOF prediction hash or coverage mismatch")
        with path.open(encoding="utf-8",newline="") as stream:
            rows = list(csv.DictReader(stream))
        by_id = {r["id"]:r for r in rows}
        if len(rows) != len(by_id) or set(by_id) != set(ids):
            raise ValueError("OOF ID cohort differs from TRAIN")
        y = np.array([float(by_id[s]["target"]) for s in ids],np.float32)
        p = np.array([float(by_id[s]["prediction"]) for s in ids],np.float32)
        if not np.array_equal(y,arrays["M"]) or not np.isfinite(p).all():
            raise ValueError("OOF target mismatch/nonfinite output")
        e = np.abs(np.clip(p,-1,1)-y)
        errors.append(e)
        records.append({"seed":seed,"predictions_path":str(path),"sha256":record["sha256"],
            "native_MAE_by_annotation_slice":{k:float(e[mask].mean()) if mask.any() else None for k,mask in slices.items()}})
    stacked = np.stack(errors)
    report = {"author_csv_sha256":digest(author_csv),"source_held_oof_suite_sha256":digest(oof_suite),
        "implementation_sha256":digest(__file__),"annotation_summary":description,"source_held_fusion":records,
        "three_seed_mean_native_MAE_by_slice":{k:float(stacked[:,mask].mean()) if mask.any() else None for k,mask in slices.items()},
        "VALID_scores_parsed":False,"TEST_scores_parsed":False,"new_model_fits":0,
        "scope":"Descriptive TRAIN annotation slices and already-audited source-held OOF predictions. Annotation differences are subjective task-specific ratings, not physical reliability, rater uncertainty or evidence of causal utility. Channel labels are unavailable to the inference router; any future prediction must be learned from optimization TRAIN only. These slices establish no original mechanism or improvement."}
    save_json(output,report)
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--author-csv",type=Path,required=True)
    p.add_argument("--oof-suite",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=diagnose(a.author_csv,a.oof_suite,a.output)
    print(json.dumps({"annotation_summary":r["annotation_summary"],"three_seed_mean_native_MAE_by_slice":r["three_seed_mean_native_MAE_by_slice"]},indent=2))
