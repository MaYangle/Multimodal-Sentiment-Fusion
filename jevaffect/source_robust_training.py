"""Source-TRAIN-only robustness baselines with independent source-CV selection."""
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,WeightedRandomSampler
from jevaffect.source_robust import CASES,SourceRiskModel
from jevaffect.anchor_crossfit import prepare,PooledView,_loader,_predict_targets,_write_predictions,_read_predictions
from jevaffect.source_crossfit import group_of,subset,manifest_hash
from jevaffect.native_training import digest,move,save_json,save_checkpoint
from jevaffect.native_metrics import native_metrics


def fit_source_groups(ids):
    sources=sorted({group_of(s) for s in ids})
    if not sources or any(not s for s in sources):
        raise ValueError("Nonempty optimization source IDs required")
    mapping={s:i for i,s in enumerate(sources)}
    groups=np.array([mapping[group_of(s)] for s in ids],np.int64)
    counts=np.bincount(groups,minlength=len(sources))
    return sources,groups,counts


def make_fit_loader(dataset,ids,config,case,generator):
    if len(dataset)!=len(ids):
        raise ValueError("Fit sampler IDs do not match dataset")
    if case=="clip_erm":
        return DataLoader(dataset,batch_size=config["batch_size"],shuffle=True,generator=generator,num_workers=0)
    _,groups,counts=fit_source_groups(ids)
    sampler=WeightedRandomSampler(torch.from_numpy(1/counts[groups].astype(np.float64)),len(dataset),replacement=True,generator=generator)
    return DataLoader(dataset,batch_size=config["batch_size"],sampler=sampler,generator=generator,num_workers=0)


def train_fold(config,data,case,seed,fold,resume=False,stop_after=None):
    if case not in CASES or seed not in (13,42,2026) or fold not in (0,1,2) or config["predict_valid"] or config["dataset"]!="simsv2" or (stop_after is not None and stop_after<1):
        raise ValueError("Invalid registered TRAIN-only source control")
    rng_seed=seed+10000*fold
    torch.set_num_threads(config.get("torch_threads",4));torch.use_deterministic_algorithms(True)
    random.seed(rng_seed);np.random.seed(rng_seed);torch.manual_seed(rng_seed)
    device=torch.device(config["device"])
    if device.type=="cuda":
        torch.cuda.manual_seed_all(rng_seed)
    cohort=data["fold_manifest"]["folds"][fold]
    view=PooledView(data["train"],("text","audio","vision"))
    ds={role:subset(view,cohort[role+"_ids"]) for role in ("fit","select","oof")}
    sources,_,counts=fit_source_groups(cohort["fit_ids"])
    mapping={s:i for i,s in enumerate(sources)}
    dims={m:v.shape[-1] for m,v in view.features.items()}
    model=SourceRiskModel(dims,len(sources),config["hidden"],config["dropout"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["learning_rate"],weight_decay=config["weight_decay"])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",patience=config["scheduler_patience"],factor=config["scheduler_factor"])
    generator=torch.Generator().manual_seed(rng_seed)
    fit_loader=make_fit_loader(ds["fit"],cohort["fit_ids"],config,case,generator)
    select_loader=_loader(ds["select"],config);oof_loader=_loader(ds["oof"],config)
    smoke=data["fixture_smoke"] or data["smoke_groups"] is not None
    root=Path(config["run_root"])/("smoke" if smoke else "formal")/case/f"seed_{seed}"/f"fold_{fold}"
    root.mkdir(parents=True,exist_ok=True)
    names=("source_robust.py","source_robust_training.py","anchor_crossfit.py","native_training.py","native_metrics.py","source_crossfit.py")
    identity={"config":config,"case":case,"seed":seed,"rng_seed":rng_seed,"fold":fold,"cohort":cohort,
        "fold_manifest_sha256":manifest_hash(data["fold_manifest"]),"author_csv_sha256":data["author_csv_sha256"],
        "train_ids":data["ids"],"feature_hashes":data["train"].hashes,"domains":sources,"domain_counts":counts.tolist(),
        "feature_dimensions":dims,"smoke_only":smoke,"smoke_groups":data["smoke_groups"],"fixture_smoke":data["fixture_smoke"],
        "implementation":{n:digest(Path(__file__).with_name(n)) for n in names},
        "runtime":{"torch":torch.__version__,"numpy":np.__version__,"device":str(device)},
        "selection":"minimum source-inner native MAE; first checkpoint wins ties",
        "official_VALID_used":False,"test_evaluated":False,"scope":"known source-risk/domain-adversarial regression baselines"}
    epoch0,best,stale,history,active=0,float("inf"),0,[],None
    last=root/"last.pt"
    if last.exists():
        if not resume:
            raise ValueError("Source run exists; pass --resume")
        state=torch.load(last,map_location="cpu",weights_only=False)
        if state["identity"]!=identity:
            raise ValueError("Source control identity changed")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"]);scheduler.load_state_dict(state["scheduler"])
        epoch0,best,stale,history,active=state["epoch"],state["best"],state["stale"],state["history"],state["active_gradients"]
        torch.set_rng_state(state["torch_rng"]);np.random.set_state(state["numpy_rng"]);random.setstate(state["python_rng"]);generator.set_state(state["loader_rng"])
        if device.type=="cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    save_json(root/"identity.json",identity)
    cap=min(config["max_epochs"],epoch0+stop_after) if stop_after is not None else config["max_epochs"]
    steps=math.ceil(len(ds["fit"])/config["batch_size"])
    for epoch in range(epoch0,cap):
        if stale>=config["patience"]:
            break
        model.train();start=time.perf_counter();totals=np.zeros(3);seen=0;updates=0;sample_counts=np.zeros(len(sources),np.int64)
        for batch_index,(features,target,ids) in enumerate(fit_loader):
            groups=torch.tensor([mapping[group_of(s)] for s in ids],dtype=torch.long,device=device)
            sample_counts+=np.bincount(groups.cpu().numpy(),minlength=len(sources))
            optimizer.zero_grad(set_to_none=True)
            fraction=(epoch*steps+batch_index)/max(1,config["max_epochs"]*steps-1)
            loss,info=model.training_objective(move(features,device),target.to(device),groups,case,fraction,config["dro_step_size"],config["dann_gradient_scale"])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite native source objective")
            loss.backward()
            if active is None:
                active={name:sum(int(torch.count_nonzero(p.grad)) for p in module.parameters() if p.grad is not None)
                        for name,module in (("point",model.regressor),("domain",model.domain_classifier))}
            torch.nn.utils.clip_grad_norm_(model.regressor.parameters(),config["gradient_norm"])
            torch.nn.utils.clip_grad_norm_(model.domain_classifier.parameters(),config["gradient_norm"])
            optimizer.step();n=len(target);seen+=n;updates+=1
            totals+=np.array([float(info[k]) for k in ("clip_L1","risk_objective","domain_CE")])*n
        metrics,_=_predict_targets(model,select_loader,device);scheduler.step(metrics["MAE"])
        probability=model.source_probability().detach().cpu().numpy()
        row={"epoch":epoch+1,"train_clip_L1":float(totals[0]/seen),"risk_objective":float(totals[1]/seen),
             "domain_CE":float(totals[2]/seen),"inner_select":metrics,"optimizer_updates":updates,
             "sampled_source_counts":sample_counts.tolist(),"source_probability":probability.tolist(),
             "reversal_scale_end":info["reversal_scale"],"learning_rate":optimizer.param_groups[0]["lr"],"seconds":time.perf_counter()-start}
        history.append(row)
        if metrics["MAE"]<best:
            best,stale=metrics["MAE"],0;save_checkpoint(root/"best.pt",{"identity":identity,"epoch":epoch+1,"metrics":metrics,"model":model.state_dict()})
        else:
            stale+=1
        save_checkpoint(last,{"identity":identity,"epoch":epoch+1,"best":best,"stale":stale,"history":history,"active_gradients":active,
            "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
            "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),"python_rng":random.getstate(),
            "loader_rng":generator.get_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else []})
        save_json(root/"history.json",history)
        print(json.dumps({"case":case,"seed":seed,"fold":fold,"epoch":epoch+1,"train_L1":row["train_clip_L1"],"inner_MAE":metrics["MAE"],"updates":updates}),flush=True)
    selected=torch.load(root/"best.pt",map_location="cpu",weights_only=False);model.load_state_dict(selected["model"])
    recomputed,_=_predict_targets(model,select_loader,device)
    if recomputed!=selected["metrics"]:
        raise ValueError("Selected source checkpoint metrics changed")
    metrics,records=_predict_targets(model,oof_loader,device)
    path=root/"oof_predictions.csv";_write_predictions(path,records,with_target=True)
    result={"root":str(root),"case":case,"seed":seed,"fold":fold,"complete":stale>=config["patience"] or len(history)>=config["max_epochs"],
        "smoke_only":smoke,"official_VALID_used":False,"test_evaluated":False,"best_epoch":selected["epoch"],"epochs_executed":len(history),
        "optimizer_updates":sum(r["optimizer_updates"] for r in history),"loop_seconds":sum(r["seconds"] for r in history),
        "fit_count":len(ds["fit"]),"select_count":len(ds["select"]),"oof_count":len(ds["oof"]),"source_count":len(sources),
        "point_parameters":sum(p.numel() for p in model.regressor.parameters()),"domain_parameters":sum(p.numel() for p in model.domain_classifier.parameters()),
        "first_batch_nonzero_gradients":active,"oof_native":metrics,"checkpoint_sha256":digest(root/"best.pt"),
        "predictions_file":str(path),"predictions_sha256":digest(path)}
    save_json(root/"result.json",result);return result


def collect(config,case,seed,ids,rows,smoke):
    root=Path(config["run_root"])/("smoke" if smoke else "formal")/case/f"seed_{seed}"
    parts=[]
    for fold in (0,1,2):
        path=root/f"fold_{fold}"/"oof_predictions.csv"
        if not path.exists():
            return None
        result=json.loads((path.parent/"result.json").read_text())
        if not result["complete"]:
            return None
        parts.extend(_read_predictions(path,with_target=True))
    if len(parts)!=len(ids) or len({r["id"] for r in parts})!=len(ids) or {r["id"] for r in parts}!=set(ids):
        raise ValueError("Source OOF does not cover TRAIN once")
    by_id={r["id"]:r for r in parts};parts=[by_id[s] for s in ids]
    if any(np.float32(r["target"])!=np.float32(rows[r["id"]]["score"]) for r in parts):
        raise ValueError("Source OOF native target changed")
    path=root/"oof_predictions.csv";_write_predictions(path,parts,with_target=True)
    p=np.array([r["prediction"] for r in parts],np.float32);y=np.array([r["target"] for r in parts],np.float32)
    return {"path":str(path),"sha256":digest(path),"count":len(ids),"native":native_metrics(p,y,"simsv2"),"coverage_exactly_once":True}


def run_suite(config,output,case=None,seed=None,fold=None,smoke_groups=None,resume=False,stop_after=None,allow_fixture=False):
    if tuple(config["cases"])!=CASES or tuple(config["seeds"])!=(13,42,2026) or config["predict_valid"]:
        raise ValueError("Registered source cases/seeds and TRAIN-only input required")
    data=prepare(config,allow_fixture=allow_fixture,smoke_groups=smoke_groups)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    results=[];smoke=data["fixture_smoke"] or smoke_groups is not None
    cohorts=data["fold_manifest"]
    save_json(out.with_suffix(out.suffix+".manifest.json"),{"ids":data["ids"],"source_csv_sha256":data["author_csv_sha256"],
        "feature_hashes":data["train"].hashes,"fold_manifest":cohorts,"config":config,"smoke_only":smoke,
        "official_VALID_used":False,"test_evaluated":False})
    for c in ([case] if case else CASES):
        for s in ([seed] if seed else (13,42,2026)):
            for f in ([fold] if fold is not None else (0,1,2)):
                results.append(train_fold(config,data,c,s,f,resume,stop_after))
                save_json(out,{"results":results,"suite_complete":False,"smoke_only":smoke,"test_evaluated":False})
    oof={}
    for c in ([case] if case else CASES):
        for s in ([seed] if seed else (13,42,2026)):
            value=collect(config,c,s,data["ids"],data["rows"],smoke)
            if value is not None:
                oof[f"{c}/seed_{s}"]=value
    complete=case is None and seed is None and fold is None and not smoke and stop_after is None and len(results)==54 and all(r["complete"] for r in results) and len(oof)==18
    report={"results":results,"oof":oof,"suite_complete":complete,"smoke_only":smoke,"official_VALID_used":False,"test_evaluated":False}
    save_json(out,report);return report
