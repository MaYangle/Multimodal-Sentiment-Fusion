"""Reload all nine fit-source standardized teachers and audit their statistics."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import prepare,PooledView,_loader,_predict_targets,_read_predictions
from jevaffect.standardized_source import transform_fold
from jevaffect.source_crossfit import subset
from jevaffect.native_training import PooledRegressor,digest,save_json
from jevaffect.native_metrics import native_metrics
from scripts.prepare_anchor_calibration_tables import _validate_teacher,merge_oof_records
from scripts.audit_anchor_teacher_predictions import compare_records
from scripts.audit_source_text_capacity import bootstrap


def audit(config_path,suite_path,baseline_suite,capacity_audit):
    torch.set_num_threads(4)
    cfg=json.loads(Path(config_path).read_text());suite=json.loads(Path(suite_path).read_text())
    raw=json.loads(Path(baseline_suite).read_text());wide=json.loads(Path(capacity_audit).read_text())
    if not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"])!=9 or {(r["seed"],r["fold"]) for r in suite["results"]}!={(s,f) for s in (13,42,2026) for f in (0,1,2)}:
        raise ValueError("Complete nine standardized source teachers required")
    if not raw["suite_complete"] or not wide["complete_nine_control_teachers"]:
        raise ValueError("Complete frozen source comparators required")
    original_cfg=json.loads(Path(str(baseline_suite)+".config.json").read_text())
    if {k:v for k,v in cfg.items() if k not in ("run_root","predict_valid")}!={k:v for k,v in original_cfg.items() if k not in ("run_root","predict_valid")}:
        raise ValueError("Normalization baseline changed beyond input transform and output access")
    data=prepare(cfg);ids=data["ids"];targets={sid:data["rows"][sid]["score"] for sid in ids}
    prepared={f:transform_fold(cfg,data,f) for f in (0,1,2)};predictions={};budget=[];evidence=[]
    for row in suite["results"]:
        seed,fold=row["seed"],row["fold"]
        actual_cfg,actual_data=prepared[fold];cohort=data["fold_manifest"]["folds"][fold]
        root=Path(row["root"])
        _,_,identity,history,one_budget=_validate_teacher(root,row,cohort,data["fold_manifest"],data["full_manifest"],
            actual_cfg,{"ids":ids},data["author_csv_sha256"],data["train"].hashes,{},Path(__file__).resolve().parents[1])
        stats=identity["config"]["preprocessing"]
        if stats!=actual_cfg["preprocessing"] or not set(stats["fit_ids"]).isdisjoint(cohort["select_ids"]+cohort["oof_ids"]):
            raise ValueError("Scaler fit cohort or saved statistics changed")
        # Independently compute raw optimization statistics, not just file hashes.
        index={s:i for i,s in enumerate(data["train"].ids)};indices=[index[s] for s in cohort["fit_ids"]]
        for modality,x in data["train"].features.items():
            values=x[indices].astype(np.float64);mean=values.mean(0).astype(np.float32)
            std=values.std(0);std[std<1e-6]=1;std=std.astype(np.float32)
            if not np.array_equal(mean,np.asarray(stats["statistics"][modality]["mean"],np.float32)) or not np.array_equal(std,np.asarray(stats["statistics"][modality]["scale"],np.float32)):
                raise ValueError("Scaler does not use optimization-only population statistics")
        state=torch.load(root/"best.pt",weights_only=False,map_location="cpu",mmap=True)
        model=PooledRegressor({m:x.shape[-1] for m,x in actual_data["train"].features.items()},cfg["hidden"],cfg["dropout"]);model.load_state_dict(state["model"])
        view=PooledView(actual_data["train"],("text","audio","vision"))
        inner,_=_predict_targets(model,_loader(subset(view,cohort["select_ids"]),cfg),"cpu")
        if abs(inner["MAE"]-state["metrics"]["MAE"])>1e-6:
            raise ValueError("Standardized checkpoint inner score mismatch")
        metrics,records=_predict_targets(model,_loader(subset(view,cohort["oof_ids"]),cfg),"cpu")
        path=Path(row["predictions_file"])
        if digest(path)!=row["predictions_sha256"] or row["trainable_parameters"]!=492033:
            raise ValueError("Standardized source prediction hash/capacity differs")
        saved=_read_predictions(path,with_target=True);delta=compare_records(records,saved)
        if abs(metrics["MAE"]-row["oof_native"]["MAE"])>1e-6:
            raise ValueError("Standardized OOF metrics do not correspond to checkpoint")
        predictions[(seed,fold)]=saved;budget.append(one_budget)
        evidence.append({"seed":seed,"fold":fold,"fit_only_scaler_and_checkpoint_audit_passed":True,"cpu_gpu_prediction_max_delta":delta})
    y=np.array([targets[s] for s in ids],np.float32);errors={};scores={}
    for seed in (13,42,2026):
        merged=merge_oof_records({f:predictions[(seed,f)] for f in (0,1,2)},data["fold_manifest"]["folds"],ids,targets)
        own=_read_predictions(suite["oof"][f"fusion/seed_{seed}"]["path"],with_target=True)
        if own!=merged or digest(suite["oof"][f"fusion/seed_{seed}"]["path"])!=suite["oof"][f"fusion/seed_{seed}"]["sha256"]:
            raise ValueError("Standardized merged OOF changed")
        point=np.clip(np.array([r["prediction"] for r in merged],np.float32),-1,1)
        errors[("standardized",seed)]=np.abs(point-y);scores[str(seed)]=native_metrics(point,y,"simsv2")
        for modality in ("fusion","text"):
            record=raw["oof"][f"{modality}/seed_{seed}"];path=Path(record["path"])
            if digest(path)!=record["sha256"]:raise ValueError("Frozen OOF comparator changed")
            rr=_read_predictions(path,with_target=True)
            if [r["id"] for r in rr]!=ids or not np.array_equal(np.array([r["target"] for r in rr],np.float32),y):raise ValueError("Frozen comparator IDs/targets differ")
            errors[(modality,seed)]=np.abs(np.clip(np.array([r["prediction"] for r in rr],np.float32),-1,1)-y)
        wide_root=Path(wide["teacher_budgets"][0]["root"]).parents[1]/f"seed_{seed}"/"oof_predictions.csv"
        rr=_read_predictions(wide_root,with_target=True)
        if [r["id"] for r in rr]!=ids or not np.array_equal(np.array([r["target"] for r in rr],np.float32),y):raise ValueError("Wide text comparator IDs/targets differ")
        errors[("wide_text",seed)]=np.abs(np.clip(np.array([r["prediction"] for r in rr],np.float32),-1,1)-y)
    contrasts={c:bootstrap(np.stack([errors[("standardized",s)]-errors[(c,s)] for s in (13,42,2026)]),ids) for c in ("fusion","text","wide_text")}
    return {"suite_sha256":digest(suite_path),"config_sha256":digest(config_path),"complete":True,"evidence":evidence,
        "native_OOF":scores,"paired_contrasts":contrasts,"teacher_budgets":budget,"raw_feature_sha256":data["train"].hashes,
        "scope":"Known fit-source normalization control; TRAIN OOF development results, no original-method or final-test claim",
        "official_VALID_used":False,"test_evaluated":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--suite",type=Path,required=True)
    p.add_argument("--baseline-suite",type=Path,required=True);p.add_argument("--capacity-audit",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();r=audit(a.config,a.suite,a.baseline_suite,a.capacity_audit);save_json(a.output,r)
    print(json.dumps({"native_OOF":r["native_OOF"],"paired_contrasts":r["paired_contrasts"]},indent=2))
