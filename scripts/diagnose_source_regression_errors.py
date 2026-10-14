"""Decompose only TRAIN source-OOF errors; MSE decomposition is not MAE."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_calibration_training import load_table
from jevaffect.anchor_crossfit import _read_predictions
from jevaffect.native_training import digest, save_json
from scripts.audit_source_text_capacity import bootstrap


def decompose(point, target, ids):
    errors=np.asarray(point,np.float64)-np.asarray(target,np.float64)
    sources=np.array([sid.split("/")[0] for sid in ids])
    bias=np.empty_like(errors);within=np.empty_like(errors);pair_sums=[];pair_counts=[];rows=[]
    for source in sorted(set(sources)):
        idx=np.flatnonzero(sources==source);e=errors[idx];y=np.asarray(target)[idx]
        mean=float(e.mean());bias[idx]=mean*mean;within[idx]=(e-mean)**2
        pairs=np.abs(e[:,None]-e[None,:])[np.triu_indices(len(e),1)]
        if len(pairs):
            pair_sums.append(float(pairs.sum()));pair_counts.append(len(pairs))
        rows.append({"source":source,"clips":len(e),"target_mean":float(y.mean()),"target_range":float(np.ptp(y)),
            "error_mean":mean,"MAE":float(np.abs(e).mean()),"MSE":float(np.mean(e*e)),
            "within_error_variance":float(np.mean((e-mean)**2)),"source_mean_error_squared":mean*mean,
            "within_source_pair_MAE":float(pairs.mean()) if len(pairs) else None})
    mse=float(np.mean(errors*errors));b=float(bias.mean());w=float(within.mean())
    if abs(mse-b-w)>1e-12:
        raise ValueError("Source MSE decomposition identity failed")
    return {"native_MAE":float(np.abs(errors).mean()),"MSE":mse,"source_bias_MSE":b,"within_source_MSE":w,
        "source_mean_absolute_error":float(np.mean([r["MAE"] for r in rows])),
        "within_source_pair_MAE_equal_source":float(np.mean([r["within_source_pair_MAE"] for r in rows if r["within_source_pair_MAE"] is not None])),
        "within_source_pair_MAE_pair_weighted":float(sum(pair_sums)/sum(pair_counts)),
        "source_rows":rows},bias,within


def diagnose(table_dir, capacity_config, capacity_audit):
    cfg=json.loads(Path(capacity_config).read_text());audit=json.loads(Path(capacity_audit).read_text())
    if not audit["complete_nine_control_teachers"] or audit["config_sha256"]!=digest(capacity_config):
        raise ValueError("Audited wide-text control required")
    components={};summaries={};canonical=None
    for seed in (13,42,2026):
        train,_=load_table(table_dir,seed,"train");ids=train["ids"]
        if canonical is not None and ids!=canonical:
            raise ValueError("TRAIN IDs differ across seeds")
        canonical=ids
        wide=_read_predictions(Path(cfg["run_root"])/"formal"/"text"/f"seed_{seed}"/"oof_predictions.csv",with_target=True)
        if [r["id"] for r in wide]!=ids or not np.array_equal(np.array([r["target"] for r in wide],np.float32),train["target"]):
            raise ValueError("Wide-text OOF IDs/targets differ")
        forecasts={"text256":train["views"][:,0],"fusion256":train["anchor"],
            "text640":np.clip(np.array([r["prediction"] for r in wide],np.float32),-1,1)}
        summaries[str(seed)]={}
        for model,point in forecasts.items():
            summary,bias,within=decompose(point,train["target"],ids)
            summaries[str(seed)][model]=summary;components[(model,seed)]=(bias,within)
    contrasts={}
    for control in ("text256","text640"):
        bdelta=np.stack([components[("fusion256",s)][0]-components[(control,s)][0] for s in (13,42,2026)])
        wdelta=np.stack([components[("fusion256",s)][1]-components[(control,s)][1] for s in (13,42,2026)])
        total=bdelta+wdelta
        contrasts[control]={"fusion_minus_control_source_bias_MSE":bootstrap(bdelta,canonical),
            "fusion_minus_control_within_source_MSE":bootstrap(wdelta,canonical),
            "fusion_minus_control_total_MSE":bootstrap(total,canonical),
            "fraction_of_extra_MSE_from_source_mean_by_seed":[(float(b.mean()/t.mean()) if t.mean()>0 else None) for b,t in zip(bdelta,total)]}
    return {"table_metadata_sha256":digest(Path(table_dir)/"metadata.json"),"capacity_audit_sha256":digest(capacity_audit),
        "models":summaries,"contrasts":contrasts,"train_count":len(canonical),"source_count":len(set(s.split("/")[0] for s in canonical)),
        "scope":"TRAIN OOF development diagnostic; source-video IDs are not verified speaker IDs. MSE decomposes exactly, MAE does not; no causal-nuisance inference.",
        "test_evaluated":False,"official_VALID_used":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table-dir",type=Path,required=True);p.add_argument("--capacity-config",type=Path,required=True)
    p.add_argument("--capacity-audit",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();report=diagnose(a.table_dir,a.capacity_config,a.capacity_audit);save_json(a.output,report)
    print(json.dumps({"contrasts":report["contrasts"],"MSE":{s:{m:{k:r[k] for k in ("MSE","source_bias_MSE","within_source_MSE","within_source_pair_MAE_equal_source")} for m,r in models.items()} for s,models in report["models"].items()}},indent=2))
