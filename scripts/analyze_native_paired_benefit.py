"""Development-only paired AV benefit diagnostic; never evaluates TEST."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import digest,save_json


def read_points(path,ids,targets):
    with Path(path).open(encoding="utf-8",newline="") as f:
        if Path(path).suffix==".csv":rows=list(csv.DictReader(f))
        else:rows=json.load(f)
    lookup={r["id"]:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!=set(ids):raise ValueError("Paired forecast cohort differs")
    y=np.array([float(lookup[s]["target"]) for s in ids],np.float32)
    if not np.array_equal(y,targets):raise ValueError("Paired native targets differ")
    return np.clip(np.array([float(lookup[s]["prediction"]) for s in ids],np.float32),-1,1)


def summarize_pair(t,f,y,masks):
    et,ef=np.abs(t-y),np.abs(f-y);gain=et-ef
    return {name:{"count":int(mask.sum()),"text_MAE":float(et[mask].mean()),"fusion_MAE":float(ef[mask].mean()),
        "fusion_benefit_mean":float(gain[mask].mean()),"fusion_better_fraction":float((gain[mask]>0).mean()),
        "oracle_pair_MAE":float(np.minimum(et[mask],ef[mask]).mean()),"gain_quantiles":np.quantile(gain[mask],[0,.25,.5,.75,1]).tolist()}
        for name,mask in masks.items() if mask.any()}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--author-csv",type=Path,required=True)
    p.add_argument("--normalized-suite",type=Path,required=True);p.add_argument("--normalized-run-root",type=Path,required=True)
    p.add_argument("--raw-suite",type=Path,required=True);p.add_argument("--raw-run-root",type=Path,required=True)
    p.add_argument("--source-normalized-suite",type=Path,required=True);p.add_argument("--source-raw-suite",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    with a.author_csv.open(encoding="utf-8-sig",newline="") as f:rows=[r for r in csv.DictReader(f) if r["mode"] in ("train","valid")]
    report={"author_csv_sha256":digest(a.author_csv),"cohorts":{},"test_evaluated":False,
        "scope":"Exploratory paired model benefit; raw text and standardized fusion have explicitly different preprocessing. Oracle/annotation strata cannot be deployed or used as model inputs."}
    normalized=json.loads(a.normalized_suite.read_text());raw=json.loads(a.raw_suite.read_text())
    sn=json.loads(a.source_normalized_suite.read_text());sr=json.loads(a.source_raw_suite.read_text())
    if not all(s["suite_complete"] for s in (normalized,raw,sn,sr)):raise ValueError("Complete frozen suites required")
    for split in ("train","valid"):
        rr=[r for r in rows if r["mode"]==split];ids=[r["video_id"]+"/"+r["clip_id"] for r in rr]
        y=np.array([float(r["label"]) for r in rr],np.float32)
        unimodal=np.array([[float(r["label_"+m]) for m in ("T","A","V")] for r in rr],np.float32)
        conflict=(unimodal.min(1)<0)&(unimodal.max(1)>0)
        fixed={"all":np.ones(len(y),bool),"neutral":y==0,"weak_nonzero":(np.abs(y)>0)&(np.abs(y)<=np.float32(.4)),
               "strong":np.abs(y)>np.float32(.4),"annotation_sign_conflict":conflict,"no_sign_conflict":~conflict}
        per_seed={}
        for seed in (13,42,2026):
            if split=="valid":
                paths=[a.raw_run_root/"text"/f"seed_{seed}"/"valid_predictions.json",a.normalized_run_root/"early_fusion"/f"seed_{seed}"/"valid_predictions.json"]
                expected=[next(r["predictions_sha256"] for r in raw["results"] if r["model"]=="text" and r["seed"]==seed),
                          next(r["predictions_sha256"] for r in normalized["results"] if r["model"]=="early_fusion" and r["seed"]==seed)]
            else:
                entries=[sr["oof"][f"text/seed_{seed}"],sn["oof"][f"fusion/seed_{seed}"]]
                paths=[Path(r["path"]) for r in entries];expected=[r["sha256"] for r in entries]
            if any(digest(path)!=sha for path,sha in zip(paths,expected)):raise ValueError("Paired frozen prediction hashes differ")
            t,f=[read_points(path,ids,y) for path in paths]
            edges=np.array([0,.1,.4,.7,1.0001],np.float32)
            bins=np.searchsorted(edges[1:],np.abs(t),side="left")
            masks={**fixed,**{f"text_abs_bin_{j}":bins==j for j in range(4)}}
            per_seed[str(seed)]=summarize_pair(t,f,y,masks)
        report["cohorts"][split]={"count":len(ids),"source_count":len(set(s.split("/")[0] for s in ids)),"seeds":per_seed}
    save_json(a.output,report)
    print(json.dumps({split:{seed:groups for seed,groups in cohort["seeds"].items()} for split,cohort in report["cohorts"].items()},indent=2))
