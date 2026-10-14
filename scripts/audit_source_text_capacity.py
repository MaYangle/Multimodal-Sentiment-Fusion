"""Audit the nine registered wide-text source-CV controls against frozen views."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import prepare, PooledView, _loader, _predict_targets, _read_predictions
from jevaffect.native_training import PooledRegressor, digest, save_json
from jevaffect.native_metrics import native_metrics
from jevaffect.source_crossfit import subset
from scripts.prepare_anchor_calibration_tables import _validate_teacher, merge_oof_records
from scripts.audit_anchor_teacher_predictions import compare_records


def bootstrap(delta_by_seed, ids, reps=5000):
    groups = np.array([sid.split("/")[0] for sid in ids])
    indices = [np.flatnonzero(groups==g) for g in sorted(set(groups))]
    count = np.array([len(i) for i in indices])
    sums = np.stack([delta_by_seed[:,i].astype(np.float64).sum(1) for i in indices],1)
    draws = np.random.default_rng(20261013).integers(0,len(indices),(reps,len(indices)))
    values = sums[:,draws].sum(2)/count[draws].sum(1)[None]
    return {"delta_by_seed":delta_by_seed.astype(np.float64).mean(1).tolist(),
        "seed_percentile95":np.quantile(values,[.025,.975],axis=1).T.tolist(),
        "three_seed_delta":float(delta_by_seed.astype(np.float64).mean()),
        "three_seed_percentile95":np.quantile(values.mean(0),[.025,.975]).tolist(),
        "replicates":reps,"seed":20261013,"source_count":len(indices),
        "scope":"Conditional paired clip-weighted bootstrap; source folds were used for this development question, no final-test claim"}


def audit(config_path, baseline_suite):
    cfg=json.loads(Path(config_path).read_text());torch.set_num_threads(4)
    original_cfg=json.loads(Path(str(baseline_suite)+".config.json").read_text())
    if {k:v for k,v in cfg.items() if k not in ("hidden","predict_valid","run_root")} != {k:v for k,v in original_cfg.items() if k not in ("hidden","predict_valid","run_root")} or cfg["hidden"]!=640 or cfg["predict_valid"]:
        raise ValueError("Wide-text control changed beyond registered width/output/VALID access")
    original=json.loads(Path(baseline_suite).read_text())
    if not original["suite_complete"]:
        raise ValueError("Frozen source-view baseline suite incomplete")
    data=prepare(cfg);ids=data["ids"];targets={sid:data["rows"][sid]["score"] for sid in ids}
    rows=[];evidence=[];values={};baseline_values={};budget=[]
    formal_root=Path(cfg["run_root"])/"formal"/"text"
    for seed in (13,42,2026):
        fold_records={}
        for fold in (0,1,2):
            root=formal_root/f"seed_{seed}"/f"fold_{fold}"
            result=json.loads((root/"result.json").read_text())
            if result["view"]!="text" or result["seed"]!=seed or result["fold"]!=fold:
                raise ValueError("Capacity-control identity mismatch")
            cohort=data["fold_manifest"]["folds"][fold]
            _,_,identity,history,one_budget=_validate_teacher(root,result,cohort,data["fold_manifest"],data["full_manifest"],
                cfg,{"ids":ids},data["author_csv_sha256"],data["train"].hashes,{},Path(__file__).resolve().parents[1])
            state=torch.load(root/"best.pt",map_location="cpu",weights_only=False,mmap=True)
            model=PooledRegressor({"text":768},640,.1);model.load_state_dict(state["model"])
            metrics,actual=_predict_targets(model,_loader(subset(PooledView(data["train"],("text",)),cohort["oof_ids"]),cfg),"cpu")
            path=Path(result["predictions_file"])
            if digest(path)!=result["predictions_sha256"] or result["trainable_parameters"]!=492801:
                raise ValueError("Capacity-control predictions/parameter count changed")
            saved=_read_predictions(path,with_target=True);delta=compare_records(actual,saved)
            if abs(metrics["MAE"]-result["oof_native"]["MAE"])>1e-6:
                raise ValueError("Capacity-control checkpoint OOF native MAE mismatch")
            rows.append(result);budget.append(one_budget);fold_records[fold]=saved
            evidence.append({"seed":seed,"fold":fold,"checkpoint_predictions_audit_passed":True,"cpu_gpu_max_delta":delta})
        merged=merge_oof_records(fold_records,data["fold_manifest"]["folds"],ids,targets)
        y=np.array([r["target"] for r in merged],np.float32)
        p=np.clip(np.array([r["prediction"] for r in merged],np.float32),-1,1)
        values[seed]=p
        for view in ("fusion","text"):
            key=f"{view}/seed_{seed}";path=Path(original["oof"][key]["path"])
            if digest(path)!=original["oof"][key]["sha256"]:
                raise ValueError("Frozen source OOF baseline changed")
            records=_read_predictions(path,with_target=True)
            if [r["id"] for r in records]!=ids or not np.array_equal(np.array([r["target"] for r in records],np.float32),y):
                raise ValueError("Source-CV baseline IDs/targets differ")
            baseline_values[(view,seed)]=np.clip(np.array([r["prediction"] for r in records],np.float32),-1,1)
    y=np.array([targets[sid] for sid in ids],np.float32)
    contrasts={view:bootstrap(np.stack([np.abs(values[s]-y)-np.abs(baseline_values[(view,s)]-y) for s in (13,42,2026)]),ids) for view in ("fusion","text")}
    scores={str(s):{"wide_text":native_metrics(values[s],y,"simsv2"),
        "text256":native_metrics(baseline_values[("text",s)],y,"simsv2"),
        "fusion256":native_metrics(baseline_values[("fusion",s)],y,"simsv2")} for s in (13,42,2026)}
    gate=all(d<0 for d in contrasts["fusion"]["delta_by_seed"]) and contrasts["fusion"]["three_seed_percentile95"][1]<0
    return {"config_sha256":digest(config_path),"frozen_source_suite_sha256":digest(baseline_suite),
        "complete_nine_control_teachers":True,"evidence":evidence,"native_OOF":scores,"paired_contrasts":contrasts,
        "registered_diagnostic_gap_survives_capacity_control":gate,"teacher_budgets":budget,
        "source_manifest":data["fold_manifest"],"train_count":2722,"sources":129,
        "test_evaluated":False,"official_VALID_used":False,"original_contribution_claimed":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--baseline-suite",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();report=audit(a.config,a.baseline_suite);save_json(a.output,report)
    print(json.dumps({"native_OOF":report["native_OOF"],"contrasts":report["paired_contrasts"],"diagnostic_gate":report["registered_diagnostic_gap_survives_capacity_control"]},indent=2))
