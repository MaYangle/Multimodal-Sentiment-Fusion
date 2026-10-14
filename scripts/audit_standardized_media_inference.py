"""Compare label-free MP4 inference with the trained standardized TRAIN vectors."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_predict import load_model,read_features
from jevaffect.native_training import digest,save_json


def audit(directory,checkpoint,cache):
    directory,checkpoint,cache=map(Path,(directory,checkpoint,cache));torch.set_num_threads(4)
    spec=load_model(checkpoint,cache,"cpu")
    media=json.loads((directory/"media_audit.json").read_text())
    rows=[json.loads(s) for s in (directory/"features.jsonl").read_text().splitlines() if s.strip()]
    predictions=[json.loads(s) for s in (directory/"predictions.jsonl").read_text().splitlines() if s.strip()]
    ids=json.loads((cache/"train"/"ids.json").read_text());index={sid:i for i,sid in enumerate(ids)}
    if media["labels_read"] or media["metadata_sha256"]!=spec["metadata_sha256"] or [r["id"] for r in rows]!=[r["id"] for r in predictions] or len(rows)!=len(media["records"]):
        raise ValueError("Label-free standardized media audit schema changed")
    features={m:np.load(cache/"train"/(m+".npy"),mmap_mode="r") for m in spec["modalities"]}
    evidence=[]
    with torch.no_grad():
        for record,prediction,source in zip(rows,predictions,media["records"]):
            sid=record["id"]
            if sid not in index or source["id"]!=sid or digest(source["source_video_path"])!=source["source_video_sha256"] or source["sentiment_labels_read"]:
                raise ValueError("Media integrity or no-target claim differs")
            extracted=read_features(record["features_npz"],spec)
            cached={m:torch.from_numpy(np.array(features[m][index[sid],0],copy=True))[None] for m in spec["modalities"]}
            actual=float(spec["model"](extracted)[0]);expected=float(spec["model"](cached)[0])
            if abs(actual-prediction["score_raw"])>1e-7 or prediction["checkpoint_sha256"]!=digest(checkpoint):
                raise ValueError("Saved media prediction differs from reloaded checkpoint")
            differences={m:float(np.max(np.abs(extracted[m].numpy()-cached[m].numpy()))) for m in spec["modalities"]}
            if any(v>2e-4 for v in differences.values()) or abs(actual-expected)>1e-5:
                raise ValueError("Standardized CPU encoder flow exceeds declared numerical tolerance")
            evidence.append({"id":sid,"source_video_sha256":source["source_video_sha256"],"feature_max_differences":differences,
                "raw_media_score":actual,"cached_vector_score":expected,"absolute_prediction_difference":abs(actual-expected)})
    for array in features.values():array._mmap.close()
    return {"media_audit_sha256":digest(directory/"media_audit.json"),"checkpoint_sha256":digest(checkpoint),
        "metadata_sha256":spec["metadata_sha256"],"records":evidence,"labels_read":False,"test_evaluated":False,
        "scope":"Two genuine competition-acquired TRAIN MP4 flow checks; no held-out performance or author-original byte-equivalence claim"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--cache-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=audit(a.directory,a.checkpoint,a.cache_dir);save_json(a.output,r);print(json.dumps(r,indent=2))
